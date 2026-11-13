"""One step for every kind (``FunctionalModel.step``), and what is hoisted out
of it follows theta.

``derived_values`` of one parameter vector is shared by a rollout's steps (and
windows); of a batch of parameter vectors it is per row.  A rows or batched
rollout derives once per rollout, as a windows or scalar rollout does, instead
of once per step inside the step.
"""
import datetime
import unittest
from unittest import mock

import torch
from torch.utils import _pytree as pytree

import twin4build as tb
from twin4build.simulator._functional import STEP_KINDS, functional_rollout_batched, functional_rollout_rows
from twin4build.tests.simulator.test_compiled_batched_step import NEEDS_CPP
from twin4build.tests.simulator.test_fusion_batched import START, STEP, build
from twin4build.utils.rgetattr import rgetattr

ROWS, STEPS = 3, 4


def _fused(device="cpu", model_id="step_kinds", zones=3):
    """Heated zones with radiators (three unless ``zones`` says otherwise),
    batched and fused (a state-space block that derives its matrices), as a
    functional map with the zones' air capacity and the radiators' UA in
    theta: ``(fm, y0, tape, theta0, fused)``."""
    model = build(n_pairs=zones, model_id=model_id)
    for k in range(zones):
        model.add_connection(model.components[f"Zone{k}"], tb.SensorSystem(id=f"T{k}"), "indoorTemperature", "measuredValue")
    model.load(draw_semantic_model=False, draw_simulation_model=False)
    batched = model.batch_components()
    batched.load(draw_semantic_model=False, draw_simulation_model=False)
    batched.to(device=device, dtype=torch.float64)
    end = START + datetime.timedelta(seconds=(STEPS + 2) * STEP)
    batched.initialize(start_time=[START], end_time=[end], step_size=[STEP])
    zone_meta = model._component_to_meta["Zone0"][0]
    rad_meta = model._component_to_meta["Radiator0"][0]
    fused = batched.simulation_model._fusion_member_to_fused[zone_meta.id]
    paths = [f"{fused._member_keys[zone_meta.id]}.C_air", f"{fused._member_keys[rad_meta.id]}.UA"]
    theta_spec = [(fused, paths[0], slice(0, zones)), (fused, paths[1], slice(zones, 2 * zones))]
    simulator = tb.Simulator(batched, execution_mode="functional", execution_backend="eager", compile_step=False)
    layout, fm = simulator.build_functional_model(
        theta_spec=theta_spec, measurements=[batched.components[f"T{k}"] for k in range(zones)], step_size=STEP
    )
    recording = simulator.record_exogenous_inputs(fm, [START], [end], [STEP], layout=layout)
    fm.prepare_routes(torch.device(device))
    theta0 = torch.cat([rgetattr(fused, path).get().reshape(-1).to(torch.float64) for path in paths]).to(device)
    return fm, recording.Y0[0], recording.exogenous_tape[0], theta0, fused


def _rows(theta0, y0, tape):
    Theta = torch.stack([theta0 * (1.0 + 0.2 * b) for b in range(ROWS)])  # distinct rows
    offset = torch.arange(ROWS, dtype=y0.dtype, device=y0.device)[:, None] * 0.5
    Y = (y0.unsqueeze(0).expand(ROWS, -1) + offset).contiguous()
    U = tape[:STEPS].unsqueeze(1).expand(-1, ROWS, -1).contiguous()
    return Theta, Y, U


class TestOwnCode(unittest.TestCase):
    """Dynamo counts a function's compiled variants per code object and gives
    up beyond a limit (``recompile_limit``): every compiled step has a code
    object of its own, so the steps of several models compile in one process."""

    def _compile_two_models(self):
        """The compiled scalar step of two models of different sizes in one
        process, with room for one variant per code object (Dynamo's own
        tracing and guards, the traced graph run as it is)."""
        real = torch.compile
        models = [_fused(model_id=f"step_own_code_{zones}", zones=zones) for zones in (3, 2)]
        try:
            torch._dynamo.reset()
            with (
                mock.patch("torch.compile", side_effect=lambda fn, **kwargs: real(fn, fullgraph=True, dynamic=False, backend="eager")),
                torch._dynamo.config.patch(recompile_limit=1),
            ):
                for fm, y0, tape, theta0, _ in models:
                    out = fm.step("scalar", compiled=True)(y0, theta0, tape[0])
                    eager = fm.step("scalar")(y0, theta0, tape[0])
                    torch.testing.assert_close(out[0], eager[0], rtol=1e-10, atol=1e-10)
        finally:
            torch._dynamo.reset()

    def test_the_steps_of_several_models_compile_in_one_process(self):
        self._compile_two_models()

    def test_on_one_code_object_the_second_model_runs_into_the_limit(self):
        # the reason for the copy: without it the models' steps share one budget
        from twin4build.simulator import _functional

        with mock.patch.object(_functional, "_own_code", side_effect=lambda fn, name: fn):
            with self.assertRaisesRegex(Exception, "(?i)recompile|fullgraph"):
                self._compile_two_models()

    def test_every_compiled_step_is_handed_over_with_its_own_code_object(self):
        fm, *_ = _fused(model_id="step_own_code")
        handed = []
        with mock.patch("torch.compile", side_effect=lambda fn, **kwargs: handed.append(fn) or fn):
            for kind in STEP_KINDS:
                fm.step(kind, compiled=True)
        self.assertEqual(len(handed), len(STEP_KINDS))
        self.assertEqual(len({id(fn.__code__) for fn in handed}), len(STEP_KINDS))
        self.assertEqual(sorted(fn.__name__ for fn in handed), sorted(f"step_{kind}" for kind in STEP_KINDS))


class TestStepKinds(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fm, cls.y0, cls.tape, cls.theta0, cls.fused = _fused()

    def test_a_batch_of_thetas_derives_per_row_and_shares_the_structure(self):
        fm = self.fm
        Theta, _, _ = _rows(self.theta0, self.y0, self.tape)
        rows, spec_rows = pytree.tree_flatten(fm.derived_values(Theta))
        differing = 0
        for b in range(ROWS):
            own, spec = pytree.tree_flatten(fm.derived_values(Theta[b]))
            self.assertEqual(spec, spec_rows)
            for leaf, one in zip(rows, own):
                if not isinstance(one, torch.Tensor):
                    self.assertEqual(leaf, one)
                elif one.is_floating_point():
                    self.assertEqual(tuple(leaf.shape), (ROWS, *one.shape))
                    torch.testing.assert_close(leaf[b], one, rtol=1e-13, atol=0)
                    differing += int(b > 0 and not torch.equal(leaf[b], leaf[0]))
                else:  # an index tensor or a flag: the structure's, shared
                    self.assertEqual(tuple(leaf.shape), tuple(one.shape))
                    self.assertTrue(torch.equal(leaf, one))
        self.assertGreater(differing, 0)  # the rows' matrices do differ

    def test_every_kind_steps_each_row_as_the_scalar_step_steps_it_alone(self):
        fm = self.fm
        Theta, Y, U = _rows(self.theta0, self.y0, self.tape)
        scalar = fm.step("scalar")
        for kind, (y_rows, theta_rows, u_rows) in STEP_KINDS.items():
            if kind == "scalar":
                continue
            with self.subTest(kind=kind):
                theta = Theta if theta_rows else self.theta0
                u = U[0] if u_rows else self.tape[0]
                constants = fm.derived_values(theta)
                hoisted_y, hoisted_m = fm.step(kind)(Y, theta, u, constants)
                plain_y, plain_m = fm.step(kind)(Y, theta, u)  # derives inside the step
                torch.testing.assert_close(hoisted_y, plain_y, rtol=1e-13, atol=1e-13)
                torch.testing.assert_close(hoisted_m, plain_m, rtol=1e-13, atol=1e-13)
                for b in range(ROWS):
                    y_b, m_b = scalar(Y[b], theta[b] if theta_rows else theta, u[b] if u_rows else u)
                    torch.testing.assert_close(hoisted_y[b], y_b, rtol=1e-13, atol=1e-13)
                    torch.testing.assert_close(hoisted_m[b], m_b, rtol=1e-13, atol=1e-13)
                self.assertFalse(torch.allclose(hoisted_y[0], hoisted_y[-1]))

    def test_a_rows_or_batched_rollout_derives_once_not_once_per_step(self):
        fm, fused = self.fm, self.fused
        Theta, Y, U = _rows(self.theta0, self.y0, self.tape)
        real = type(fused).derive
        for name, roll in (
            ("rows", lambda steps: functional_rollout_rows(fm, Y, Theta, U[:steps])),
            ("batched", lambda steps: functional_rollout_batched(fm, Y, Theta, self.tape[:steps], step=fm.step("batched"))),
        ):
            with self.subTest(rollout=name):
                counts = []
                for steps in (1, STEPS):
                    with mock.patch.object(type(fused), "derive", autospec=True, side_effect=real) as derive:
                        roll(steps)
                    counts.append(derive.call_count)
                self.assertEqual(counts[0], counts[1])  # the same for one step as for four
                self.assertGreater(counts[0], 0)

    def test_the_gradient_through_the_hoisted_rows_is_the_unhoisted_one(self):
        fm = self.fm
        Theta, Y, U = _rows(self.theta0, self.y0, self.tape)
        step = fm.step("rows")

        def loss(theta, hoist):
            constants = fm.derived_values(theta) if hoist else None
            y, total = Y, 0.0
            for t in range(STEPS):
                y, m = step(y, theta, U[t], constants)
                total = total + m.square().sum()
            return total

        grads = []
        for hoist in (True, False):
            theta = Theta.clone().requires_grad_(True)
            grads.append(torch.autograd.grad(loss(theta, hoist), theta)[0])
        torch.testing.assert_close(grads[0], grads[1], rtol=1e-11, atol=1e-13)
        self.assertGreater(float(grads[0].abs().max()), 0.0)

    @NEEDS_CPP
    def test_the_compiled_rows_step_takes_the_hoisted_values(self):
        fm = self.fm
        Theta, Y, U = _rows(self.theta0, self.y0, self.tape)
        constants = fm.derived_values(Theta)
        with torch.no_grad():
            compiled = fm.step("rows", compiled=True)(Y, Theta, U[0], constants)
            eager = fm.step("rows")(Y, Theta, U[0], constants)
        for got, want in zip(compiled, eager):
            torch.testing.assert_close(got, want, rtol=1e-11, atol=1e-11)

    def test_the_old_names_are_the_factory_steps(self):
        fm = self.fm
        self.assertIs(fm.rows_step, fm.step("rows"))
        self.assertIs(fm.batched_step, fm.step("batched"))
        self.assertIs(fm.window_step, fm.step("windows"))
        with self.assertRaises(KeyError):
            fm.step("columns")


@unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
class TestDerivedValuesDevice(unittest.TestCase):
    def test_every_derived_tensor_is_on_thetas_device(self):
        fm, y0, tape, theta0, _ = _fused(device="cuda", model_id="step_kinds_cuda")
        for theta in (theta0, torch.stack([theta0, theta0 * 1.1])):
            for leaf in pytree.tree_leaves(fm.derived_values(theta)):
                if isinstance(leaf, torch.Tensor):
                    self.assertEqual(leaf.device.type, "cuda")


if __name__ == "__main__":
    unittest.main()
