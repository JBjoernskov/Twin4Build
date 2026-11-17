"""Equal-length periods roll out as one batch of windows: the objective's
per-column losses and gradients must equal the sequential per-period
rollout's, on CPU (eager vmap) and, with CUDA, under per-step graphs."""
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


def _objective(device, batching, **sim_kwargs):
    ss.WINDOW_BATCHING = batching
    model = load_model()
    if device == "cuda":
        model.to(device="cuda", dtype=torch.float64)
    est = tb.Estimator(tb.Simulator(model, execution_mode="functional", **sim_kwargs))
    starts = [EXAMPLE_START[0], EXAMPLE_START[0] + datetime.timedelta(days=2)]
    est.estimate(
        parameters=example_parameters(model),
        measurements=example_measurements(model),
        start_time=starts,
        end_time=[s + datetime.timedelta(hours=24) for s in starts],
        step_size=STEP_SIZE,
        n_warmup=5,
        method=("scipy", "SLSQP", "ad"),
        options={"maxiter": 1},
    )
    obj = est._functional_objective
    x0 = torch.tensor(np.asarray(est._x0_norm, dtype=np.float64), dtype=torch.float64, device=device)
    cols, grad = obj.batched_column_loss_and_grad(x0.unsqueeze(0))
    # the batched path with B > 1 rows (the column-gradient chunks)
    selector = torch.eye(cols.shape[1], dtype=torch.float64, device=device)[:2]
    cols2, grads2 = obj.batched_column_gradients(x0.unsqueeze(0).expand(2, -1), selector)
    return (cols.detach().cpu(), grad.detach().cpu(), cols2.detach().cpu(), grads2.detach().cpu(), obj._windows() is not None)


class TestWindowBatching(unittest.TestCase):
    def test_cpu_windows_match_sequential(self):
        try:
            seq = _objective("cpu", False, execution_backend="eager", compile_step=False)
            win = _objective("cpu", True, execution_backend="eager", compile_step=False)
        finally:
            ss.WINDOW_BATCHING = True
        self.assertFalse(seq[4])
        self.assertTrue(win[4])
        torch.testing.assert_close(win[0], seq[0], rtol=1e-8, atol=1e-10)
        torch.testing.assert_close(win[1], seq[1], rtol=1e-6, atol=1e-9)
        torch.testing.assert_close(win[2], seq[2], rtol=1e-8, atol=1e-10)
        torch.testing.assert_close(win[3], seq[3], rtol=1e-6, atol=1e-9)

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_cuda_step_graphs_windows_match_sequential(self):
        try:
            # the reference is the eager sequential rollout: the compiled batched
            # step (rollout scope, B > 1) is itself miscompiled on torch 2.11
            # (see FunctionalModel.window_step)
            seq = _objective("cuda", False, execution_backend="eager", compile_step=False)
            win = _objective("cuda", True, execution_backend="cuda_graph", cuda_graph_scope="step")
        finally:
            ss.WINDOW_BATCHING = True
        self.assertTrue(win[4])
        # the eager reference uses torch.matrix_exp, the vmapped step the
        # scaling-and-squaring fallback: 1e-6 relative per step, 1e-5 over the rollout
        torch.testing.assert_close(win[0], seq[0], rtol=1e-4, atol=1e-6)
        torch.testing.assert_close(win[1], seq[1], rtol=1e-3, atol=1e-6)
        torch.testing.assert_close(win[2], seq[2], rtol=1e-4, atol=1e-6)
        torch.testing.assert_close(win[3], seq[3], rtol=1e-3, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
