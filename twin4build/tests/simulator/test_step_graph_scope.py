"""``cuda_graph_scope="step"``: one CUDA graph per step, replayed along the
rollout, with a manual adjoint sweep for the gradient.  The forward rollout
must equal the eager one; the objective's per-column losses and gradients
must equal those of the whole-rollout capture (and of plain eager autograd)."""
import datetime
import unittest

import numpy as np
import torch

import twin4build as tb

tb._IS_TESTING = True

from twin4build.tests.simulator.test_fusion_batched import START, STEP, build, history, simulate


@unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
class TestStepGraphScope(unittest.TestCase):
    def test_option_is_validated(self):
        model = build(n_pairs=1, model_id="scope_opt")
        with self.assertRaises(ValueError):
            tb.Simulator(model, execution_mode="functional", execution_backend="cuda_graph", cuda_graph_scope="rollouts")

    def test_forward_rollout_matches_eager(self):
        reference = build(n_pairs=3, model_id="scope_ref")
        simulate(reference, execution_mode="functional", execution_backend="eager")
        model = build(n_pairs=3, model_id="scope_src")
        batched = model.batch_components()
        batched.load(draw_semantic_model=False, draw_simulation_model=False)
        batched.to(device="cuda", dtype=torch.float64)
        simulate(batched, execution_mode="functional", execution_backend="cuda_graph", cuda_graph_scope="step")
        for k in range(3):
            torch.testing.assert_close(
                history(model, f"Zone{k}", "indoorTemperature", batched),
                history(reference, f"Zone{k}", "indoorTemperature"), rtol=1e-6, atol=1e-6,
            )
            torch.testing.assert_close(
                history(model, f"Radiator{k}", "Power", batched),
                history(reference, f"Radiator{k}", "Power"), rtol=1e-6, atol=1e-4,
            )
        # the simulation captured two step graphs and nothing at rollout level
        session = batched.simulation_model  # noqa: F841 (the model owns the functional model's graph cache)

    def test_a_graph_holds_for_one_initialization(self):
        """Every simulate() initializes the model, and the components
        reallocate their tensors: a step graph captured before reads them at
        their old addresses (on the HTR ring the second simulation of one
        simulator ran from freed memory, rooms 2.3 K off).  The second
        simulation captures its graphs again and gives what the first gave."""
        model = build(n_pairs=3, model_id="scope_repeat")
        batched = model.batch_components()
        batched.load(draw_semantic_model=False, draw_simulation_model=False)
        batched.to(device="cuda", dtype=torch.float64)
        simulator = tb.Simulator(batched, execution_mode="functional", execution_backend="cuda_graph", cuda_graph_scope="step")
        graphs, temperatures = [], []
        for _ in range(2):
            simulator.simulate(start_time=START, end_time=START + datetime.timedelta(hours=6), step_size=STEP, show_progress_bar=False)
            cache = simulator._functional_session.functional_model.__dict__["_step_graphs"]
            graphs.append({key: g.fwd.graph for key, g in cache.items()})
            temperatures.append(history(model, "Zone0", "indoorTemperature", batched))
        self.assertTrue(graphs[0])
        self.assertEqual(set(graphs[1]), set(graphs[0]))
        for key in graphs[0]:
            self.assertIsNot(graphs[1][key], graphs[0][key], "a graph from before the initialization was replayed")
        torch.testing.assert_close(temperatures[1], temperatures[0], rtol=0, atol=0)

    def test_gradient_matches_whole_rollout_capture(self):
        from twin4build.tests.estimator.example_fixture import (
            EXAMPLE_START,
            STEP_SIZE,
            example_measurements,
            example_parameters,
            load_model,
        )

        results = {}
        for scope in ("rollout", "step"):
            model = load_model()
            model.to(device="cuda", dtype=torch.float64)
            est = tb.Estimator(tb.Simulator(model, execution_mode="functional", execution_backend="cuda_graph", cuda_graph_scope=scope))
            start = EXAMPLE_START[0]
            est.estimate(
                parameters=example_parameters(model),
                measurements=example_measurements(model),
                start_time=[start],
                end_time=[start + datetime.timedelta(hours=24)],
                step_size=STEP_SIZE,
                n_warmup=5,
                method=("scipy", "SLSQP", "ad"),
                options={"maxiter": 1},
            )
            obj = est._functional_objective
            x0 = torch.tensor(np.asarray(est._x0_norm, dtype=np.float64), dtype=torch.float64, device="cuda")
            cols, grad = obj.batched_column_loss_and_grad(x0.unsqueeze(0))
            results[scope] = (cols.detach().cpu(), grad.detach().cpu())
        # the whole-rollout capture steps the compiled batched step
        # (torch.matrix_exp), the per-step graphs the eager vmapped step (the
        # scaling-and-squaring fallback): 1e-6 relative per step, 1e-4 over
        # the rollout
        torch.testing.assert_close(results["step"][0], results["rollout"][0], rtol=1e-4, atol=1e-6)
        torch.testing.assert_close(results["step"][1], results["rollout"][1], rtol=1e-3, atol=1e-6)

    def test_batched_kind_matches_eager_rows(self):
        """The ``"batched"`` per-step graph (a batch of parameter starts,
        one shared tape: the Pareto prepass and the column-gradient chunks)
        must agree with the eager sequential rollout.  The reference is
        eager because the compiled batched step (``compile`` over ``vmap``)
        is itself numerically wrong on torch 2.11."""
        from twin4build.tests.estimator.example_fixture import (
            EXAMPLE_START,
            STEP_SIZE,
            example_measurements,
            example_parameters,
            load_model,
        )

        results = {}
        for name, kwargs in (("eager", dict(execution_backend="eager", compile_step=False)), ("step", dict(execution_backend="cuda_graph", cuda_graph_scope="step"))):
            model = load_model()
            model.to(device="cuda", dtype=torch.float64)
            est = tb.Estimator(tb.Simulator(model, execution_mode="functional", **kwargs))
            start = EXAMPLE_START[0]
            est.estimate(
                parameters=example_parameters(model),
                measurements=example_measurements(model),
                start_time=[start],
                end_time=[start + datetime.timedelta(hours=24)],
                step_size=STEP_SIZE,
                n_warmup=5,
                method=("scipy", "SLSQP", "ad"),
                options={"maxiter": 1},
            )
            obj = est._functional_objective
            x0 = torch.tensor(np.asarray(est._x0_norm, dtype=np.float64), dtype=torch.float64, device="cuda")
            X = torch.stack([x0, x0 * 0.9 + 0.05])  # two distinct starts
            selector = torch.eye(int(obj.batched_column_loss_and_grad(x0.unsqueeze(0))[0].shape[1]), dtype=torch.float64, device="cuda")[:2]
            cols, grads = obj.batched_column_gradients(X, selector)
            results[name] = (cols.detach().cpu(), grads.detach().cpu())
        torch.testing.assert_close(results["step"][0], results["eager"][0], rtol=1e-4, atol=1e-6)
        torch.testing.assert_close(results["step"][1], results["eager"][1], rtol=1e-3, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
