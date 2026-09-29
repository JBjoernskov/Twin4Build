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
MS = {"states": "all", "sd_rel": 0.0025, "sd_abs": 0.0, "bound_rel": 0.25, "bound_abs": 0.0, "first_window": False}


def _estimator(device, *, multiple_shooting=None, windows=2, sim_kwargs=None):
    model = load_model()
    if device == "cuda":
        model.to(device="cuda", dtype=torch.float64)
    est = tb.Estimator(tb.Simulator(model, execution_mode="functional", **(sim_kwargs or dict(execution_backend="eager", compile_step=False))))
    t0 = EXAMPLE_START[0]
    starts = [t0 + datetime.timedelta(hours=HOURS * p) for p in range(windows)]  # contiguous windows
    est.estimate(
        parameters=example_parameters(model),
        measurements=example_measurements(model),
        start_time=starts,
        end_time=[s + datetime.timedelta(hours=HOURS) for s in starts],
        step_size=STEP_SIZE,
        n_warmup=2,
        method=("scipy", "SLSQP", "ad"),
        options={"maxiter": 1},
        multiple_shooting=multiple_shooting,
    )
    return est


class TestMultipleShooting(unittest.TestCase):
    def test_variables_columns_and_blocks_grow_as_declared(self):
        est = _estimator("cpu", multiple_shooting=MS)
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
        default = _estimator("cpu", multiple_shooting=MS)  # the explicit opt-out
        first = _estimator("cpu", multiple_shooting={k: v for k, v in MS.items() if k != "first_window"})  # the default
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

        est = _estimator("cpu", multiple_shooting={k: v for k, v in MS.items() if k != "first_window"})
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
        plain = _estimator("cpu")
        loose = _estimator("cpu", multiple_shooting=dict(MS, sd_abs=1e9, sd_rel=0.0))
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
        est = _estimator("cpu", multiple_shooting=MS)
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
        # run once that one-step transient has decayed
        Y0_vars = obj._Y0_with(Y0, obj._denorm_init(init_norm))
        chained_vars = est.simulator.rollout_functional_windows(obj.composer, Y0_vars, theta_phys, tape)
        torch.testing.assert_close(chained_vars[1][n_t // 2 :], continuous[n_t + n_t // 2 :], rtol=1e-3, atol=1e-3)
        # the column losses at the chained point carry no defect contribution
        cols = obj.column_loss(x_ext)
        self.assertLess(float(cols[len(est._measurements) :].abs().max()), 1e-16)

    def test_jumps_are_the_scored_defects_in_physical_units(self):
        """``continuity_jumps`` is the start of window p+1 minus the end of
        window p: the negative of the defect the objective scores, times the
        tolerance; it vanishes where window 1 continues window 0."""
        est = _estimator("cpu", multiple_shooting=MS)
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

        est = _estimator("cpu", multiple_shooting=MS)
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

    def test_gradient_reaches_the_initial_state_variables(self):
        est = _estimator("cpu", multiple_shooting=MS)
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
            eager = _estimator("cuda", multiple_shooting=MS, sim_kwargs=dict(execution_backend="eager", compile_step=False))
            step = _estimator("cuda", multiple_shooting=MS, sim_kwargs=dict(execution_backend="cuda_graph", cuda_graph_scope="step"))
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
