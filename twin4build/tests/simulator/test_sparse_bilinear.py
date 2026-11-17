"""Sparse bilinear terms for fused blocks.

A fused block's ``E`` and ``F`` pass as their entries (``SparseBilinear``) in
place of dense ``(n_c, m, n, n)`` / ``(n_c, m, n, m)`` tensors that are almost
all zeros (a storey-wide block: ~158 states x 588 inputs, where broadcasting
``u`` over the dense terms took ~90 % of a fused step's time).  The effective
matrices, the one-step map, its gradients and the fused models' trajectories
are the same as with the dense form (``fused_sparse=False``).
"""

# Standard library imports
import datetime
import unittest

# Third party imports
import torch
from dateutil import tz

# Local application imports
import twin4build as tb
from twin4build.systems.utils.discrete_statespace_system import (
    SparseBilinear,
    bilinear_inputs,
    bilinear_onestep,
    bilinear_plan,
    effective_matrices,
    with_plan,
)

tb._IS_TESTING = True
torch.manual_seed(0)
START = datetime.datetime(2024, 1, 4, tzinfo=tz.gettz("Europe/Copenhagen"))


def random_sparse(n_c, m, n, cols, nnz, dtype=torch.float64):
    k = torch.randint(0, m, (nnz,))
    i = torch.randint(0, n, (nnz,))
    j = torch.randint(0, cols, (nnz,))
    values = torch.randn(n_c, nnz, dtype=dtype) * 1e-3
    dense = torch.zeros(n_c, m, n, cols, dtype=dtype)
    dense.index_put_((torch.arange(n_c)[:, None], k[None, :], i[None, :], j[None, :]), values, accumulate=True)
    return SparseBilinear(k, i, j, values), dense


class TestSparseEffectiveMatrices(unittest.TestCase):
    def setUp(self):
        self.n_c, self.n, self.m = 3, 7, 11
        self.A = torch.randn(self.n_c, self.n, self.n, dtype=torch.float64) * 1e-3
        self.B = torch.randn(self.n_c, self.n, self.m, dtype=torch.float64) * 1e-3
        self.Es, self.Ed = random_sparse(self.n_c, self.m, self.n, self.n, 25)
        self.Fs, self.Fd = random_sparse(self.n_c, self.m, self.n, self.m, 30)

    def test_the_effective_matrices_equal_the_dense_ones(self):
        for shape in ((self.n_c, self.m), (4, self.n_c, self.m)):  # one window, four windows
            u = torch.randn(*shape, dtype=torch.float64)
            A_s, B_s = effective_matrices(self.A, self.B, self.Es, self.Fs, u)
            A_d, B_d = effective_matrices(self.A, self.B, self.Ed, self.Fd, u)
            torch.testing.assert_close(A_s, A_d, rtol=0, atol=1e-15)
            torch.testing.assert_close(B_s, B_d, rtol=0, atol=1e-15)

    def test_the_one_step_map_and_its_gradients(self):
        C = torch.randn(self.n_c, 2, self.n, dtype=torch.float64)
        D = torch.zeros(self.n_c, 2, self.m, dtype=torch.float64)
        x = torch.randn(self.n_c, self.n, dtype=torch.float64)
        u = torch.randn(self.n_c, self.m, dtype=torch.float64, requires_grad=True)
        vals = self.Es.values.clone().requires_grad_(True)
        Es = SparseBilinear(self.Es.k, self.Es.i, self.Es.j, vals)
        x1, y1 = bilinear_onestep(self.A, self.B, C, D, Es, self.Fs, x, u, 600.0, transform_mode=True)
        g_u, g_v = torch.autograd.grad((x1.square().sum() + y1.sum()), (u, vals))
        u2 = u.detach().clone().requires_grad_(True)
        x2, y2 = bilinear_onestep(self.A, self.B, C, D, self.Ed, self.Fd, x, u2, 600.0, transform_mode=True)
        (g_u2,) = torch.autograd.grad((x2.square().sum() + y2.sum()), (u2,))
        torch.testing.assert_close(x1, x2, rtol=1e-12, atol=1e-14)
        torch.testing.assert_close(y1, y2, rtol=1e-12, atol=1e-14)
        torch.testing.assert_close(g_u, g_u2, rtol=1e-10, atol=1e-14)
        self.assertTrue(bool(torch.isfinite(g_v).all()) and float(g_v.abs().sum()) > 0)

    def test_under_vmap(self):
        us = torch.randn(5, self.n_c, self.m, dtype=torch.float64)
        f_s = torch.func.vmap(lambda u: effective_matrices(self.A, self.B, self.Es, self.Fs, u)[1])
        f_d = torch.func.vmap(lambda u: effective_matrices(self.A, self.B, self.Ed, self.Fd, u)[1])
        torch.testing.assert_close(f_s(us), f_d(us), rtol=0, atol=1e-15)

    def test_the_inputs_the_terms_act_on(self):
        m = self.m
        torch.testing.assert_close(bilinear_inputs(self.Es, m, "cpu"), bilinear_inputs(self.Ed, m, "cpu"))
        torch.testing.assert_close(bilinear_inputs(self.Fs, m, "cpu"), bilinear_inputs(self.Fd, m, "cpu"))
        self.assertFalse(bool(bilinear_inputs(None, m, "cpu").any()))


class TestPlannedSum(unittest.TestCase):
    """The fixed-order sum (``bilinear_plan``): the dense form's values and
    gradients, and on a GPU the same bits on every call and in a CUDA graph
    (``index_add``'s atomic adds failed a capture's parity check in float32)."""

    def setUp(self):
        self.n_c, self.n, self.m = 3, 7, 11
        self.A = torch.randn(self.n_c, self.n, self.n, dtype=torch.float64) * 1e-3
        self.B = torch.randn(self.n_c, self.n, self.m, dtype=torch.float64) * 1e-3
        Es, self.Ed = random_sparse(self.n_c, self.m, self.n, self.n, 60)  # many repeats on 49 positions
        Fs, self.Fd = random_sparse(self.n_c, self.m, self.n, self.m, 80)
        self.Es, self.Fs = with_plan(Es, self.n), with_plan(Fs, self.m)

    def test_the_plan_covers_every_entry_once(self):
        slot, inputs, group, flat = self.Es.plan
        nnz = self.Es.k.numel()
        real = slot[slot < nnz]
        self.assertEqual(sorted(real.tolist()), list(range(nnz)))
        self.assertEqual(len(set(inputs.tolist())), inputs.numel())
        self.assertEqual(len(set(flat.tolist())), flat.numel())
        places = group[group < slot.numel()]
        self.assertEqual(len(set(places.tolist())), nnz)

    def test_the_effective_matrices_equal_the_dense_ones(self):
        for shape in ((self.n_c, self.m), (4, self.n_c, self.m)):
            u = torch.randn(*shape, dtype=torch.float64)
            A_s, B_s = effective_matrices(self.A, self.B, self.Es, self.Fs, u)
            A_d, B_d = effective_matrices(self.A, self.B, self.Ed, self.Fd, u)
            torch.testing.assert_close(A_s, A_d, rtol=0, atol=1e-15)
            torch.testing.assert_close(B_s, B_d, rtol=0, atol=1e-15)

    def test_under_vmap(self):
        us = torch.randn(5, self.n_c, self.m, dtype=torch.float64)
        f_s = torch.func.vmap(lambda u: effective_matrices(self.A, self.B, self.Es, self.Fs, u)[0])
        f_d = torch.func.vmap(lambda u: effective_matrices(self.A, self.B, self.Ed, self.Fd, u)[0])
        torch.testing.assert_close(f_s(us), f_d(us), rtol=0, atol=1e-15)

    def test_the_gradients_equal_the_dense_ones(self):
        u = torch.randn(4, self.n_c, self.m, dtype=torch.float64, requires_grad=True)
        vals = self.Es.values.clone().requires_grad_(True)
        A_s, B_s = effective_matrices(self.A, self.B, self.Es._replace(values=vals), self.Fs, u)
        g_u, g_v = torch.autograd.grad(A_s.square().sum() + B_s.sum(), (u, vals))
        u2 = u.detach().clone().requires_grad_(True)
        vals2 = self.Es.values.clone().requires_grad_(True)
        A_i, B_i = effective_matrices(
            self.A, self.B, self.Es._replace(values=vals2, plan=None), self.Fs._replace(plan=None), u2
        )
        g_u2, g_v2 = torch.autograd.grad(A_i.square().sum() + B_i.sum(), (u2, vals2))
        torch.testing.assert_close(g_u, g_u2, rtol=1e-12, atol=1e-15)
        torch.testing.assert_close(g_v, g_v2, rtol=1e-12, atol=1e-15)

    def test_the_plan_is_built_inside_a_transform(self):
        """A block assembled for the first time inside a functorch transform
        (an optimizer's ``jacrev`` over an object-mode simulation) builds its
        plan from wrapped index tensors; the plan is plain tensors, which
        later calls reuse inside or outside any transform."""
        k, i, j = self.Es.k, self.Es.i, self.Es.j
        reference = bilinear_plan(k, i, j, self.n)
        plans = []

        def f(v):
            plans.append(bilinear_plan(k + 0 * v.sum().long(), i, j, self.n))
            return v.square().sum()

        torch.func.jacrev(f)(torch.ones(3, dtype=torch.float64))
        for built, ref in zip(plans[0], reference):
            self.assertFalse(torch._C._functorch.is_functorch_wrapped_tensor(built))
            self.assertTrue(torch.equal(built, ref))

    @unittest.skipUnless(torch.cuda.is_available(), "needs a GPU: atomic adds are a GPU's")
    def test_float32_on_a_gpu_the_same_bits_every_call_and_in_a_cuda_graph(self):
        from twin4build.simulator._step_graph import StepGraph

        dev, n_c, m, n, nnz = "cuda", 4, 300, 60, 6000  # thousands of entries on few positions
        k = torch.randint(0, m, (nnz,), device=dev)
        i = torch.randint(0, n, (nnz,), device=dev)
        j = torch.randint(0, 8, (nnz,), device=dev)
        vals = torch.randn(n_c, nnz, device=dev, dtype=torch.float32, requires_grad=True)
        E = with_plan(SparseBilinear(k, i, j, vals), n)
        A = torch.randn(n_c, n, n, device=dev, dtype=torch.float32)
        B = torch.zeros(n_c, n, m, device=dev, dtype=torch.float32)
        u = torch.randn(16, n_c, m, device=dev, dtype=torch.float32, requires_grad=True)

        def run():
            a = effective_matrices(A, B, E, None, u)[0]
            g = torch.autograd.grad((a * a).sum(), (u, vals))
            return (a.detach(),) + tuple(x.detach() for x in g)

        first, second = run(), run()
        for x, y in zip(first, second):
            self.assertTrue(torch.equal(x, y))

        def step(uu):
            a = effective_matrices(A, B, E._replace(values=vals.detach()), None, uu)[0]
            return (a,)

        graph = StepGraph(step, 0, name="planned sum")
        graph.set_static()
        graph(u.detach())  # captures; raises when the replay differs from the eager step


def fused_model(model_id, pairs=2):
    """Composite rooms with air flows (bilinear flow x temperature terms) in
    pairs joined by a fused wall."""
    m = tb.Model(id=model_id)

    def sched(v, sid):
        return tb.ScheduleSystem(weekday_ruleset={"ruleset_default_value": float(v)}, id=sid)

    values = {"out": 5.0, "sup_t": 18.0, "zero": 0.0, "co2": 420.0, "flow": 0.3, "heat": 900.0}
    s = {k: sched(v, k) for k, v in values.items()}
    rooms = []
    for k in range(pairs):
        pair = []
        for side in ("A", "B"):
            z = tb.BuildingSpaceSystem(
                thermal={"C_air": 2e5, "C_wall": 5e6, "R_out": 0.02, "R_in": 0.01}, id=f"Z{side}{k}"
            )
            for port, key in (
                ("outdoorTemperature", "out"),
                ("supplyAirTemperature", "sup_t"),
                ("globalIrradiation", "zero"),
                ("numberOfPeople", "zero"),
                ("outdoorCo2Concentration", "co2"),
            ):
                m.add_connection(s[key], z, "scheduleValue", port)
            m.add_connection(s["heat"] if side == "A" else s["zero"], z, "scheduleValue", "heatGain")
            for port in ("supplyAirFlowRate", "exhaustAirFlowRate"):
                m.add_connection(s["flow"], z, "scheduleValue", port, input_port_index=0)
            pair.append(z)
        a, b = pair
        wall = tb.WallSystem(C=1e5, R_a=0.02, R_b=0.02, id=f"W{k}")
        m.add_connection(a, wall, "indoorTemperature", "temperatureA")
        m.add_connection(b, wall, "indoorTemperature", "temperatureB")
        m.add_connection(wall, a, "heatFlowRateA", "wallHeatGain", input_port_index=0)
        m.add_connection(wall, b, "heatFlowRateB", "wallHeatGain", input_port_index=0)
        rooms += pair
    m.load(draw_semantic_model=False, draw_simulation_model=False)
    return m, rooms


def simulate(model, mode):
    kwargs = {} if mode == "object" else {"execution_mode": "functional", "execution_backend": "eager"}
    tb.Simulator(model, **kwargs).simulate(
        start_time=START,
        end_time=START + datetime.timedelta(hours=6),
        step_size=600,
        show_progress_bar=False,
    )


def histories(rooms):
    return [
        torch.cat([r.output[p].history().detach().reshape(-1) for p in ("indoorTemperature", "indoorCO2")])
        for r in rooms
    ]


class TestFusedSparse(unittest.TestCase):
    def _run(self, sparse, mode, tag, batched=False):
        with tb.settings.override(fused_sparse=sparse):
            model, rooms = fused_model(f"sparse_bilinear_{tag}_{sparse}_{mode}_{batched}")
            if batched:
                meta = model.batch_components()
                meta.load(draw_semantic_model=False, draw_simulation_model=False)
                simulate(meta, mode)
                out = []
                for r in rooms:
                    m, i_c = model._component_to_meta[r.id]
                    out.append(
                        torch.cat(
                            [
                                m.output[p].history().detach().reshape(-1, m._n_c_batched)[:, i_c]
                                for p in ("indoorTemperature", "indoorCO2")
                            ]
                        )
                    )
                return out, meta
            simulate(model, mode)
            return histories(rooms), model

    def test_the_assembled_matrices_equal_the_dense_ones(self):
        _, model = self._run(True, "object", "assemble")
        blocks = list(model.simulation_model._fused_components.values())
        self.assertTrue(blocks)
        for blk in blocks:
            params = blk._forward_params()
            with tb.settings.override(fused_sparse=True):
                A, B, C, D, E, F = blk._assemble(params)
            with tb.settings.override(fused_sparse=False):
                A2, B2, C2, D2, E2, F2 = blk._assemble(params)
            self.assertIsInstance(E, SparseBilinear)
            self.assertIsNotNone(E.plan)  # a fused block's terms are summed by their plan
            for sparse, dense in ((E, E2), (F, F2)):
                full = torch.zeros_like(dense)
                n_c = dense.shape[0]
                full.index_put_(
                    (torch.arange(n_c)[:, None], sparse.k[None, :], sparse.i[None, :], sparse.j[None, :]),
                    sparse.values,
                    accumulate=True,
                )
                torch.testing.assert_close(full, dense, rtol=0, atol=0)
                self.assertLess(sparse.values.shape[1], dense[0].numel())
            for a, b in ((A, A2), (B, B2), (C, C2), (D, D2)):
                torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_a_block_assembled_first_inside_a_transform(self):
        """The first assembly inside ``jacrev`` builds the indices and their
        plan from wrapped tensors; the block keeps them unwrapped, and an
        assembly outside the transform reuses them."""
        _, model = self._run(True, "object", "transform")
        blk = next(iter(model.simulation_model._fused_components.values()))
        params = dict(blk._forward_params())
        key = next(name for name in blk.PARAM_NAMES if name.endswith("thermal.C_air"))
        blk._sparse_cache.clear()

        def entries(scale):
            return blk._assemble({**params, key: params[key] * scale})[4].values.sum()

        gradient = torch.func.jacrev(entries)(torch.ones(1, dtype=torch.float64))
        self.assertTrue(bool(torch.isfinite(gradient).all()))
        (structure,) = blk._sparse_cache.values()
        E = blk._assemble(params)[4]
        for t in (*E[:3], *E.plan):
            self.assertFalse(torch._C._functorch.is_functorch_wrapped_tensor(t))
        self.assertIs(E.k, structure[0][0])

    def test_a_compiled_step_finds_the_structure_built_at_initialize(self):
        """A step that rebuilds the matrices inside ``torch.compile`` (the
        batched step) cannot build the indices and their plan (numpy on the
        host): ``initialize`` builds them."""
        _, model = self._run(True, "object", "compiled")
        blk = next(iter(model.simulation_model._fused_components.values()))
        model.initialize(START, START + datetime.timedelta(hours=1), 600)
        self.assertEqual(len(blk._sparse_cache), 1)
        params = dict(blk._forward_params())
        key = next(name for name in blk.PARAM_NAMES if name.endswith("thermal.C_air"))

        def entries(scale):
            return blk._assemble({**params, key: params[key] * scale})[4].values.sum()

        scale = torch.ones(1, dtype=torch.float64)
        torch._dynamo.reset()
        try:
            compiled = torch.compile(entries, backend="eager", fullgraph=True)(scale)
        finally:
            torch._dynamo.reset()
        torch.testing.assert_close(compiled, entries(scale), rtol=0, atol=0)

    def test_object_and_functional_trajectories_equal_the_dense_ones(self):
        for mode in ("object", "functional"):
            sparse, _ = self._run(True, mode, "traj")
            dense, _ = self._run(False, mode, "traj")
            for s, d in zip(sparse, dense):
                torch.testing.assert_close(s, d, rtol=1e-12, atol=1e-10)

    def test_batched_rooms(self):
        sparse, _ = self._run(True, "object", "batched", batched=True)
        dense, _ = self._run(False, "object", "batched", batched=True)
        for s, d in zip(sparse, dense):
            torch.testing.assert_close(s, d, rtol=1e-12, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
