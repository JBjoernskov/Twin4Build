"""Multiple shooting over batched windows: the windows' initial states are
decision variables and the continuity defects between consecutive windows
are residual columns.  Checks: the solver vector and structure grow as
declared; at the recorded initial states with a loose tolerance the loss is
the plain objective's; with the variables set to the previous window's end
states the windows reproduce one continuous rollout; and the CUDA
per-step-graph path agrees with the eager one."""
import datetime
import unittest

import numpy as np
import torch

import twin4build as tb
import twin4build.estimator._single_shooting as ss

tb._IS_TESTING = True

from twin4build.tests.estimator.example_fixture import (
    EXAMPLE_START,
    STEP_SIZE,
    example_measurements,
    example_parameters,
    load_model,
)

HOURS = 12
#: The structural tests below use the explicit opt-out (the first window at
#: its recorded state: one set of variables for two windows); the default
#: frees every window's initial state (``test_first_window_state_is_a_variable_too``).
MS = {
    "initial_states": "all",
    "continuity_sd_rel": 0.0025,
    "continuity_sd_abs": 0.0,
    "initial_state_bound_rel": 0.25,
    "initial_state_bound_abs": 0.0,
    "estimate_first_state": False,
}


def _estimator(device, *, shooting=None, transcription="multiple_shooting", windows=2, sim_kwargs=None, gap_hours=0, schedule=None, maxiter=1):
    """An estimator fitted over ``windows`` periods of ``HOURS``; ``shooting``
    holds the initial-state options of ``transcription``."""
    model = load_model()
    if device == "cuda":
        model.to(device="cuda", dtype=torch.float64)
    est = tb.Estimator(tb.Simulator(model, execution_mode="functional", **(sim_kwargs or dict(execution_backend="eager", compile_step=False))))
    t0 = EXAMPLE_START[0]
    # contiguous windows unless ``gap_hours`` leaves time between them
    starts = [t0 + datetime.timedelta(hours=(HOURS + gap_hours) * p) for p in range(windows)]
    est.estimate(
        parameters=example_parameters(model),
        measurements=example_measurements(model),
        start_time=starts,
        end_time=[s + datetime.timedelta(hours=HOURS) for s in starts],
        step_size=STEP_SIZE,
        n_warmup=2,
        method=("scipy", "SLSQP", "ad", transcription),
        options={"maxiter": maxiter, **(shooting or {})},
        schedule=schedule,
    )
    return est


def _largest_jump(result):
    values = [np.abs(np.asarray(v, dtype=float)) for v in result["continuity_jumps_instances"].values()]
    return max(float(np.nanmax(v)) for v in values if np.isfinite(v).any())


class TestMultipleShooting(unittest.TestCase):
    def test_variables_columns_and_blocks_grow_as_declared(self):
        est = _estimator("cpu", shooting=MS)
        obj = est._functional_objective
        self.assertIsNotNone(obj)
        D = int(obj.layout.width)
        self.assertGreater(obj.n_init, 0)
        self.assertEqual(obj.n_init, obj.init_n_slow)  # two windows: one set of variables
        self.assertLessEqual(obj.init_n_slow, D)
        self.assertEqual(obj.n_theta_ext, obj.n_theta + obj.n_init)
        # the solver's vector and bounds were extended and the result cut back
        self.assertEqual(len(est._last_x_norm), obj.n_theta)
        self.assertEqual(len(obj.init_entry_names()), obj.n_init)
        theta_block, column_block, n_blocks = obj.parameter_structure()
        self.assertEqual(len(theta_block), obj.n_theta_ext)
        self.assertEqual(len(column_block), len(est._measurements) + obj.init_n_slow)
        self.assertTrue((np.asarray(theta_block)[obj.n_theta :] >= 0).all())
        self.assertTrue((np.asarray(column_block)[len(est._measurements) :] >= 0).all())
        x = torch.tensor(np.concatenate([est._last_x_norm, obj.init_x0_norm.cpu().numpy()]), dtype=torch.float64)
        cols = obj.column_loss(x)
        self.assertEqual(int(cols.numel()), len(est._measurements) + obj.init_n_slow)
        torch.testing.assert_close(cols.sum(), obj.loss(x), rtol=1e-10, atol=1e-12)
        self.assertEqual(int(obj.residual_vector(x).numel()), int(obj.residual_vector(x).numel()))
        state = obj.init_values(x.numpy())
        self.assertEqual(sorted(state), [1])
        self.assertEqual(len(state[1]), obj.init_n_slow)

    def test_first_window_state_is_a_variable_too(self):
        """By default every window's initial state is a variable, the first
        window's included (no warm-up needed anywhere); the defect columns
        are unchanged, and at the recorded states the loss equals the
        opt-out configuration's."""
        default = _estimator("cpu", shooting=MS)  # the explicit opt-out
        first = _estimator("cpu", shooting={k: v for k, v in MS.items() if k != "estimate_first_state"})  # the default
        d_obj, f_obj = default._functional_objective, first._functional_objective
        self.assertEqual(f_obj.init_first, 0)
        self.assertEqual(f_obj.n_init, 2 * f_obj.init_n_slow)  # two windows: two sets
        self.assertEqual(f_obj.n_init, 2 * d_obj.n_init)
        self.assertEqual(len(f_obj.init_entry_names()), f_obj.n_init)
        self.assertTrue(f_obj.init_entry_names()[0].startswith("init[0]:"))
        theta_block, column_block, _ = f_obj.parameter_structure()
        self.assertEqual(len(theta_block), f_obj.n_theta_ext)
        self.assertEqual(len(column_block), len(first._measurements) + f_obj.init_n_slow)
        theta = torch.tensor(np.asarray(default._x0_norm, dtype=np.float64), dtype=torch.float64)
        x_first = torch.cat([theta, f_obj.init_x0_norm.cpu()])
        x_default = torch.cat([theta, d_obj.init_x0_norm.cpu()])
        torch.testing.assert_close(f_obj.loss(x_first) * f_obj.loss_scale, d_obj.loss(x_default) * d_obj.loss_scale, rtol=1e-10, atol=1e-12)
        state = f_obj.init_values(x_first.numpy())
        self.assertEqual(sorted(state), [0, 1])
        _cols, grad = f_obj.batched_column_loss_and_grad(x_first.unsqueeze(0))
        n_slow = f_obj.init_n_slow
        self.assertGreater(float(grad[0, f_obj.n_theta : f_obj.n_theta + n_slow].abs().max()), 0.0)  # window 0's variables are live

    def test_result_carries_the_states_and_load_sets_the_model_state(self):
        """The result stores the estimated initial states per executing
        component and per original component; ``load_estimation_result``
        sets them as the model's state, so the next ``initialize`` of the
        estimated periods starts there."""
        import pickle

        est = _estimator("cpu", shooting={k: v for k, v in MS.items() if k != "estimate_first_state"})
        obj = est._functional_objective
        with open(est.result_savedir_pickle, "rb") as handle:
            result = pickle.load(handle)
        state = result["estimated_initial_state"]
        self.assertEqual({c.id for c in obj.layout.components}, set(state))
        for comp, (n_c, ss) in zip(obj.layout.components, obj.layout.shapes):
            self.assertEqual(tuple(state[comp.id].shape), (2, n_c, ss))  # two windows
        instances = result["estimated_initial_state_instances"]
        self.assertTrue(instances)
        self.assertEqual(sum(int(v.shape[1]) for v in instances.values()), int(obj.layout.width))
        self.assertEqual(sorted(result["estimated_initial_state_labels"]), [0, 1])
        model = est.simulator.model
        starts = [EXAMPLE_START[0] + datetime.timedelta(hours=HOURS * p) for p in range(2)]
        ends = [s + datetime.timedelta(hours=HOURS) for s in starts]
        model.load_estimation_result(filename=est.result_savedir_pickle)
        model.initialize(starts, ends, [STEP_SIZE, STEP_SIZE])
        for comp in obj.layout.components:
            torch.testing.assert_close(comp.get_state().detach().cpu(), state[comp.id].to(comp.get_state().dtype), rtol=1e-12, atol=1e-12)
        # the second window alone: its own row
        model.initialize(starts[1:], ends[1:], [STEP_SIZE])
        for comp in obj.layout.components:
            torch.testing.assert_close(comp.get_state().detach().cpu()[0], state[comp.id][1].to(comp.get_state().dtype), rtol=1e-12, atol=1e-12)
        # parameters only: the state is left alone
        model.clear_state()
        model.load_estimation_result(filename=est.result_savedir_pickle, initial_state=False)
        self.assertIsNone(model.simulation_model._initial_state)

    def test_loose_tolerance_at_recorded_states_is_the_plain_objective(self):
        plain = _estimator("cpu", transcription="single_shooting")
        loose = _estimator("cpu", shooting=dict(MS, continuity_sd_abs=1e9, continuity_sd_rel=0.0))
        p_obj, l_obj = plain._functional_objective, loose._functional_objective
        theta = torch.tensor(np.asarray(plain._x0_norm, dtype=np.float64), dtype=torch.float64)
        x_ext = torch.cat([theta, l_obj.init_x0_norm.cpu()])
        # the plain vector stands in for the extended one (variables at their
        # recorded values); both objectives roll the windows out in transform
        # mode (the eager vmapped step), so the values agree to round-off
        # (each objective's loss scale was set on its own construction path, so
        # compare the unscaled sums of squares)
        plain_value = p_obj.loss(theta, transform_mode=True) * p_obj.loss_scale
        torch.testing.assert_close(l_obj.loss(theta) * l_obj.loss_scale, plain_value, rtol=1e-10, atol=1e-12)
        torch.testing.assert_close(l_obj.loss(x_ext) * l_obj.loss_scale, plain_value, rtol=1e-10, atol=1e-12)
        torch.testing.assert_close(l_obj.raw_residuals(x_ext), p_obj.raw_residuals(theta, transform_mode=True), rtol=1e-10, atol=1e-12)

    def test_zero_defect_reproduces_the_continuous_rollout(self):
        est = _estimator("cpu", shooting=MS)
        obj = est._functional_objective
        theta = torch.tensor(np.asarray(est._last_x_norm, dtype=np.float64), dtype=torch.float64)
        theta_phys = obj._denorm(theta)
        Y0, tape = obj._windows()
        # the end state of window 0 at theta, written as window 1's initial state
        out, end = est.simulator.rollout_functional_windows(obj.composer, Y0, theta_phys, tape, return_end=True)
        init_phys = end[:-1][:, obj.init_index]
        init_norm = ((init_phys - obj.init_lb) / (obj.init_ub - obj.init_lb)).reshape(-1)
        x_ext = torch.cat([theta, init_norm])
        # the defects vanish ...
        _Ms, defects = obj._rollout(theta_phys, obj._denorm_init(init_norm))
        self.assertLess(float(defects.abs().max()), 1e-9)
        # ... and window 1 continues window 0: one continuous rollout over both
        # (the whole augmented end state chained, feedback lags included)
        Y0_chain = torch.stack([Y0[0], end[0]])
        chained = est.simulator.rollout_functional_windows(obj.composer, Y0_chain, theta_phys, tape)
        continuous_tape = torch.cat([tape[:, 0], tape[:, 1]], dim=0)
        continuous = est.simulator.rollout_functional(obj.composer, Y0[0], theta_phys, continuous_tape, transform_mode=True)
        n_t = tape.shape[0]
        torch.testing.assert_close(chained[0], continuous[:n_t], rtol=1e-8, atol=1e-10)
        torch.testing.assert_close(chained[1], continuous[n_t:], rtol=1e-8, atol=1e-10)
        # the objective's own chaining writes the selected states only (the
        # recorded feedback lags stay): the windows agree with the continuous
        # run once that one-step transient has decayed.  How far it has
        # decayed half a window on depends on the parameters one SLSQP
        # iteration reached, which differ a little between torch and SciPy
        # versions (0.13 % on Python 3.10's): half a percent.
        Y0_vars = obj._Y0_with(Y0, obj._denorm_init(init_norm))
        chained_vars = est.simulator.rollout_functional_windows(obj.composer, Y0_vars, theta_phys, tape)
        torch.testing.assert_close(chained_vars[1][n_t // 2 :], continuous[n_t + n_t // 2 :], rtol=5e-3, atol=1e-3)
        # the column losses at the chained point carry no defect contribution
        cols = obj.column_loss(x_ext)
        self.assertLess(float(cols[len(est._measurements) :].abs().max()), 1e-16)

    def test_jumps_are_the_scored_defects_in_physical_units(self):
        """``continuity_jumps`` is the start of window p+1 minus the end of
        window p: the negative of the defect the objective scores, times the
        tolerance; it vanishes where window 1 continues window 0."""
        est = _estimator("cpu", shooting=MS)
        obj = est._functional_objective
        theta = torch.tensor(np.asarray(est._last_x_norm, dtype=np.float64), dtype=torch.float64)
        x_ext = torch.cat([theta, obj.init_x0_norm.cpu()])
        jumps = obj.continuity_jumps(x_ext)
        tolerance = obj.continuity_tolerance()
        self.assertEqual(set(jumps), {c.id for c in obj.layout.components})
        _Ms, defects = obj._rollout(*obj._physical(x_ext))
        flat = torch.cat([jumps[c.id].reshape(1, -1) for c in obj.layout.components], dim=1)[0]
        flat_sd = torch.cat([tolerance[c.id].reshape(1, -1) for c in obj.layout.components], dim=1)[0]
        idx = obj.init_index.cpu()
        torch.testing.assert_close(flat[idx], -(defects[0].cpu() * obj.init_sd.cpu()), rtol=1e-10, atol=1e-12)
        torch.testing.assert_close(flat_sd[idx], obj.init_sd.cpu(), rtol=0, atol=0)
        self.assertGreater(float(flat[idx].abs().max()), 0.0)  # the recorded states do not continue the rollout
        # the end state of window 0 as window 1's start: no jump
        theta_phys = obj._denorm(theta)
        Y0, tape = obj._windows()
        _out, end = est.simulator.rollout_functional_windows(obj.composer, Y0, theta_phys, tape, return_end=True)
        init_norm = ((end[:-1][:, obj.init_index] - obj.init_lb) / (obj.init_ub - obj.init_lb)).reshape(-1)
        zero = obj.continuity_jumps(torch.cat([theta, init_norm]))
        flat_zero = torch.cat([zero[c.id].reshape(1, -1) for c in obj.layout.components], dim=1)[0]
        self.assertLess(float(flat_zero[idx].abs().max()), 1e-9)

    def test_result_carries_the_jumps_and_the_summary_ranks_them(self):
        import pickle

        from twin4build.estimator._continuity import COLUMNS, continuity_summary

        est = _estimator("cpu", shooting=MS)
        with open(est.result_savedir_pickle, "rb") as handle:
            result = pickle.load(handle)
        jumps, tolerance = result["continuity_jumps_instances"], result["continuity_tolerance_instances"]
        self.assertEqual(set(jumps), set(result["estimated_initial_state_instances"]))
        for cid, block in jumps.items():
            self.assertEqual(int(block.shape[0]), 1)  # two windows: one boundary
            self.assertEqual(tuple(tolerance[cid].shape[1:]), tuple(block.shape[1:]))
        table = continuity_summary(jumps, tolerance)
        self.assertEqual(list(table.columns), COLUMNS)
        self.assertEqual(len(table), int(sum(np.isfinite(np.asarray(b)).sum() for b in jumps.values())))
        ratio = table["mean |jump| / tolerance"].to_numpy()
        self.assertTrue(np.all(np.diff(ratio[np.isfinite(ratio)]) <= 0))  # the largest first

    def test_the_tolerance_follows_how_much_a_state_varies(self):
        """``continuity_sd_ref="range"`` (the default) scales each state's tolerance by
        how much it moves over the windows' starts and ends; ``"value"`` by
        its magnitude; both floored as documented."""
        ranged = _estimator("cpu", shooting={k: v for k, v in MS.items() if k != "continuity_sd_rel"})
        valued = _estimator("cpu", shooting=dict(MS, continuity_sd_ref="value"))
        r_obj, v_obj = ranged._functional_objective, valued._functional_objective
        torch.testing.assert_close(r_obj.init_sd, torch.clamp(0.01 * r_obj.init_sd_reference, min=1e-9), rtol=1e-12, atol=0)
        torch.testing.assert_close(v_obj.init_sd, torch.clamp(0.0025 * v_obj.init_sd_reference, min=1e-9), rtol=1e-12, atol=0)
        torch.testing.assert_close(v_obj.init_sd_reference, v_obj.init_magnitude, rtol=0, atol=0)
        # the range is floored at a thousandth of the magnitude and cannot exceed twice it
        self.assertTrue(bool((r_obj.init_sd_reference >= 1e-3 * r_obj.init_magnitude - 1e-15).all()))
        self.assertTrue(bool((r_obj.init_sd_reference <= 2.0 * r_obj.init_magnitude + 1e-12).all()))
        # a state that barely moves gets a tighter tolerance than its magnitude would give it
        self.assertLess(float((r_obj.init_sd / v_obj.init_sd).min()), 1.0)
        with self.assertRaises(ValueError):
            _estimator("cpu", shooting=dict(MS, continuity_sd_ref="joules"))

    def test_every_state_says_what_heat_it_holds(self):
        """``state_heat_capacities``: one entry per state of every executing
        component, the parameters' capacities for the thermal states (a
        fused room's air and wall from ``C_air`` and ``C_wall``, a
        radiator's elements from ``thermalMassHeatCapacity / nelements``),
        NaN for the states that hold no heat (CO2, a controller's memory)."""
        est = _estimator("cpu", shooting=MS)
        obj = est._functional_objective
        thermal = holds_nothing = 0
        for comp, (n_c, ss) in zip(obj.layout.components, obj.layout.shapes):
            caps = comp.state_heat_capacities()
            if ss == 0:  # a component without state (the occupancy)
                self.assertIsNone(caps, comp.id)
                continue
            self.assertEqual(tuple(caps.shape), (n_c, ss), comp.id)
            thermal += int(torch.isfinite(caps).sum())
            holds_nothing += int((~torch.isfinite(caps)).sum())
            if type(comp).__name__ == "FusedStateSpaceSystem":
                col = 0
                for member in comp._members:
                    for _prefix, unit in member._ss_units():
                        width = unit.state_size()
                        part = caps[:, col : col + width]
                        name = type(unit).__name__
                        if name == "BuildingSpaceThermalSystem":
                            torch.testing.assert_close(part[:, 0], unit.C_air.get().reshape(-1).expand(n_c).to(part.dtype))
                            torch.testing.assert_close(part[:, 1], unit.C_wall.get().reshape(-1).expand(n_c).to(part.dtype))
                        elif name == "SpaceHeaterSystem":
                            expected = (unit.thermalMassHeatCapacity.get().reshape(-1, 1) / unit.nelements).expand(n_c, width)
                            torch.testing.assert_close(part, expected.to(part.dtype))
                        elif name == "BuildingSpaceMassSystem":
                            self.assertTrue(bool(torch.isnan(part).all()))
                        col += width
                self.assertEqual(col, ss)
        self.assertGreater(thermal, 0)
        self.assertGreater(holds_nothing, 0)

    def test_the_energy_tolerance_holds_a_jumps_heat(self):
        """``continuity_sd_ref="energy"``: a state that stores heat is held
        to at most ``continuity_energy_tol / C`` (tighter than its range
        tolerance when it is massive), a state that holds no heat keeps its
        range tolerance; the energy tolerance needs the energy reference."""
        energy_tol = 5e3  # J: tighter than the range for the massive states of the example
        ranged = _estimator("cpu", shooting=MS)._functional_objective
        by_energy = _estimator("cpu", shooting=dict(MS, continuity_sd_ref="energy", continuity_energy_tol=energy_tol))._functional_objective
        capacity = by_energy.init_capacity
        holds_heat = torch.isfinite(capacity) & (capacity > 0)
        self.assertTrue(bool(holds_heat.any()) and bool((~holds_heat).any()))
        expected = torch.where(holds_heat, torch.minimum(ranged.init_sd, energy_tol / torch.where(holds_heat, capacity, torch.ones_like(capacity))), ranged.init_sd)
        torch.testing.assert_close(by_energy.init_sd, expected.clamp(min=1e-9), rtol=1e-12, atol=0)
        self.assertTrue(bool((by_energy.init_sd[holds_heat] < ranged.init_sd[holds_heat]).any()))  # some are tightened
        torch.testing.assert_close(by_energy.init_sd[~holds_heat], ranged.init_sd[~holds_heat], rtol=0, atol=0)
        with self.assertRaises(ValueError):
            _estimator("cpu", shooting=dict(MS, continuity_energy_tol=energy_tol))  # the range reference

    def test_the_summary_tells_a_one_sided_state_from_noise(self):
        from twin4build.estimator._continuity import continuity_summary

        jumps = {
            "node": np.array([[-0.8], [-0.9], [-0.1], [-0.3], [-0.5], [-0.9], [-0.4]]),  # re-set downward every time
            "wall": np.array([[0.3, np.nan], [-0.2, np.nan], [0.25, np.nan], [-0.3, np.nan], [0.2, np.nan], [-0.25, np.nan], [0.3, np.nan]]),
        }
        tolerance = {"node": np.array([[0.05]]), "wall": np.array([[0.05, np.nan]])}
        table = continuity_summary(jumps, tolerance).set_index(["component", "state"])
        self.assertEqual(len(table), 2)  # the wall's second state was no variable
        node, wall = table.loc[("node", 0)], table.loc[("wall", 0)]
        self.assertAlmostEqual(node["mean jump"], -3.9 / 7)
        self.assertEqual(node["one-sided"], 1.0)
        self.assertAlmostEqual(node["mean |jump| / tolerance"], 3.9 / 7 / 0.05)
        self.assertLess(wall["one-sided"], 0.75)
        self.assertEqual(list(continuity_summary(jumps, tolerance)["component"])[0], "node")

    def test_a_class_selects_the_states_of_fused_members(self):
        """The room is fused with its radiator and wall: selecting its class
        picks its span of the block's state vector, and a class the model
        does not have selects nothing (the states stay fixed)."""
        zone_class = type(load_model().components["office"]).__name__
        est = _estimator("cpu", shooting=dict(MS, initial_states=[(zone_class, "all")]))
        obj = est._functional_objective
        model = est.simulator.model.simulation_model
        expected = []
        for comp, (start, _stop), (n_c, ss) in zip(obj.layout.components, obj.layout.slices, obj.layout.shapes):
            for leaf, owner, offset, width in model._stateful_leaves():
                if leaf is comp and type(owner).__name__ == zone_class:
                    expected += [start + i_c * ss + offset + k for i_c in range(n_c) for k in range(width)]
        self.assertTrue(expected)
        self.assertEqual(sorted(obj.init_index.tolist()), sorted(expected))
        nothing = _estimator("cpu", shooting=dict(MS, initial_states=[("NoSuchSystem", "all")]))
        self.assertEqual(nothing._functional_objective.n_init, 0)

    def test_the_periods_must_be_contiguous(self):
        with self.assertRaises(ValueError) as raised:
            _estimator("cpu", shooting=MS, gap_hours=1)
        self.assertIn("estimate_initial_state", str(raised.exception))

    def test_the_options_belong_to_their_transcription(self):
        """The tie's options need the multiple_shooting transcription,
        ``estimate_initial_state`` is single shooting's, and none of them may
        change between the phases of a schedule; the old keyword arguments
        are gone."""
        from twin4build.utils.method_spec import parse_method

        method = ("scipy", "SLSQP", "ad")
        self.assertEqual(parse_method(method + ("multiple_shooting",), allowed_methods=[method], default_methods=[method], allow_transcription=True), (method, "multiple_shooting"))
        cases = [
            dict(shooting=dict(MS, estimate_initial_state=True)),  # multiple shooting estimates them anyway
            dict(transcription="single_shooting", shooting={"continuity_sd_rel": 0.01}),  # a tie without the transcription
            dict(transcription="single_shooting", shooting={"initial_states": "all"}),  # without estimate_initial_state
            dict(shooting=MS, schedule=[{}, {"options": {"update_multipliers": True}}]),  # per phase
        ]
        for case in cases:
            with self.subTest(case=case), self.assertRaises(ValueError):
                _estimator("cpu", **case)
        for legacy in ({"multiple_shooting": {}}, {"initial_state": True}):
            with self.subTest(legacy=legacy), self.assertRaises(TypeError):
                tb.Estimator(tb.Simulator(load_model())).estimate(**legacy)

    def test_initial_states_without_the_tie(self):
        """Single shooting with ``estimate_initial_state``: every period's
        start is a variable, the periods need not be contiguous, and nothing
        ties them: no defect columns, no jumps."""
        est = _estimator("cpu", transcription="single_shooting", shooting={"estimate_initial_state": True}, gap_hours=1)
        obj = est._functional_objective
        self.assertFalse(obj.init_continuity)
        self.assertEqual(obj.init_first, 0)
        self.assertEqual(obj.n_init, 2 * obj.init_n_slow)
        _theta_block, column_block, _ = obj.parameter_structure()
        self.assertEqual(len(column_block), len(est._measurements))  # measured columns only
        theta = torch.tensor(np.asarray(est._last_x_norm, dtype=np.float64), dtype=torch.float64)
        x_ext = torch.cat([theta, obj.init_x0_norm.cpu()])
        self.assertEqual(obj.continuity_jumps(x_ext), {})
        self.assertIsNone(obj._rollout(*obj._physical(x_ext))[1])
        import pickle

        with open(est.result_savedir_pickle, "rb") as handle:
            result = pickle.load(handle)
        self.assertEqual(sorted(result["estimated_initial_state_labels"]), [0, 1])
        self.assertNotIn("continuity_jumps_instances", result)

    def test_the_shift_enters_the_defect(self):
        """The defect is (end - next start + shift) / sd, and the next shift
        is this one plus the gap left."""
        est = _estimator("cpu", shooting=MS)
        obj = est._functional_objective
        theta = torch.tensor(np.asarray(est._last_x_norm, dtype=np.float64), dtype=torch.float64)
        x_ext = torch.cat([theta, obj.init_x0_norm.cpu()])
        _Ms, plain = obj._rollout(*obj._physical(x_ext))
        shift = torch.linspace(-0.3, 0.3, obj.init_n_slow, dtype=plain.dtype).unsqueeze(0)
        obj.init_shift = shift
        _Ms, shifted = obj._rollout(*obj._physical(x_ext))
        torch.testing.assert_close(shifted, plain + shift / obj.init_sd, rtol=1e-12, atol=1e-12)
        nxt = obj.continuity_shift_next(x_ext)
        flat = torch.cat([nxt[c.id].reshape(1, -1) for c in obj.layout.components], dim=1)[0]
        torch.testing.assert_close(flat[obj.init_index.cpu()], (shifted[0] * obj.init_sd).cpu(), rtol=1e-12, atol=1e-12)
        # a shift given by component id reaches the same columns
        by_id = {}
        model = est.simulator.model.simulation_model
        for cid, block in model._instance_state({k: v for k, v in nxt.items()}).items():
            by_id[cid] = torch.nan_to_num(block, nan=0.0)
        again = obj._selected_from_instances(by_id, 1, plain.device, plain.dtype, obj.init_index)
        torch.testing.assert_close(again[0], flat[obj.init_index.cpu()].to(again.dtype), rtol=1e-12, atol=1e-12)

    def test_multiplier_updates_shrink_the_jumps(self):
        """Three phases with update_multipliers: the tolerance stays, the
        shift grows, and the largest jump left falls from phase to phase."""
        import pickle

        # a loose tie on the room's states (the test's SLSQP barely moves the
        # controllers' internal states in 20 iterations): the first fit leaves a jump
        zone_class = type(load_model().components["office"]).__name__
        config = dict(MS, continuity_sd_ref="value", continuity_sd_rel=0.02, initial_states=[(zone_class, "all")])
        single = _estimator("cpu", shooting=config, maxiter=20)
        with open(single.result_savedir_pickle, "rb") as handle:
            first = pickle.load(handle)
        phased = _estimator("cpu", shooting=dict(config, update_multipliers=True), schedule=[{}, {}, {}], maxiter=20)
        with open(phased.result_savedir_pickle, "rb") as handle:
            last = pickle.load(handle)
        self.assertIn("shift", phased._multiple_shooting)
        self.assertTrue(last["continuity_shift_instances"])
        self.assertLess(_largest_jump(last), 0.5 * _largest_jump(first))

    def test_gradient_reaches_the_initial_state_variables(self):
        est = _estimator("cpu", shooting=MS)
        obj = est._functional_objective
        x = torch.tensor(np.concatenate([est._last_x_norm, obj.init_x0_norm.cpu().numpy()]), dtype=torch.float64)
        cols, grad = obj.batched_column_loss_and_grad(x.unsqueeze(0))
        self.assertEqual(tuple(grad.shape), (1, obj.n_theta_ext))
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(float(grad[0, obj.n_theta :].abs().max()), 0.0)

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_cuda_step_graphs_match_eager(self):
        try:
            ss.WINDOW_BATCHING = True
            eager = _estimator("cuda", shooting=MS, sim_kwargs=dict(execution_backend="eager", compile_step=False))
            step = _estimator("cuda", shooting=MS, sim_kwargs=dict(execution_backend="cuda_graph", cuda_graph_scope="step"))
        finally:
            ss.WINDOW_BATCHING = True
        results = []
        # one shared point for both backends (each estimator's own solve
        # moved its vector elsewhere), off the recorded states so the
        # defects are live
        o = eager._functional_objective
        x_shared = torch.tensor(np.concatenate([eager._last_x_norm, o.init_x0_norm.cpu().numpy()]), dtype=torch.float64, device="cuda") * 0.95 + 0.02
        for est in (eager, step):
            obj = est._functional_objective
            x = x_shared.clone()
            cols, grad = obj.batched_column_loss_and_grad(x.unsqueeze(0))
            selector = torch.eye(cols.shape[1], dtype=torch.float64, device="cuda")[:2]
            cols2, grads2 = obj.batched_column_gradients(x.unsqueeze(0).expand(2, -1), selector)
            results.append((cols.cpu(), grad.cpu(), cols2.cpu(), grads2.cpu()))
        torch.testing.assert_close(results[1][0], results[0][0], rtol=1e-4, atol=1e-6)
        torch.testing.assert_close(results[1][1], results[0][1], rtol=1e-3, atol=1e-6)
        torch.testing.assert_close(results[1][2], results[0][2], rtol=1e-4, atol=1e-6)
        torch.testing.assert_close(results[1][3], results[0][3], rtol=1e-3, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
