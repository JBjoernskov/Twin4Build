import unittest
from types import SimpleNamespace

import numpy as np
import torch

from twin4build.estimator._batched_shooting import (
    BatchedShootingEvaluator,
    solve_batched_shooting,
)


class _QuadraticResidual:
    def __init__(self):
        self.est = SimpleNamespace(_device=torch.device("cpu"))
        self._sd = torch.ones(1, dtype=torch.float64)
        self.target = torch.tensor([0.25, 0.75], dtype=torch.float64)

    def residual_vector(self, x, transform_mode=False):
        return x - self.target

    def loss(self, x, transform_mode=False):
        return torch.sum(self.residual_vector(x).square())

    def batched_loss(self, x):
        return torch.func.vmap(self.loss)(x)

    def batched_value_and_grad(self, x):
        grad, value = torch.func.vmap(torch.func.grad_and_value(self.loss))(x)
        return value, grad

    def batched_residual_and_jacobian(self, x):
        residual = torch.func.vmap(self.residual_vector)(x)
        jacobian = torch.func.vmap(torch.func.jacfwd(self.residual_vector))(x)
        return residual, jacobian


class TestBatchedShootingSolvers(unittest.TestCase):
    def test_derivative_bundles(self):
        objective = _QuadraticResidual()
        evaluator = BatchedShootingEvaluator(objective)
        x = torch.tensor([[0.8, 0.1], [0.1, 0.9]], dtype=torch.float64)
        value, grad = evaluator.value_grad(x)
        residual, jac = evaluator.residual_jacobian(x)
        value_h, grad_h, hess = evaluator.value_grad_hessian(x)
        torch.testing.assert_close(value, torch.sum(residual.square(), dim=1))
        torch.testing.assert_close(value_h, value)
        torch.testing.assert_close(grad_h, grad)
        torch.testing.assert_close(
            jac,
            torch.eye(2, dtype=torch.float64).expand(2, 2, 2),
        )
        torch.testing.assert_close(
            hess,
            (2 * torch.eye(2, dtype=torch.float64)).expand(2, 2, 2),
        )

    def test_all_methods_converge_from_same_starts(self):
        starts = np.array([[0.9, 0.1], [0.0, 1.0], [0.4, 0.4]])
        objective = _QuadraticResidual()
        for method in ("batched-bfgs", "batched-lm", "batched-newton"):
            with self.subTest(method=method):
                result = solve_batched_shooting(
                    objective,
                    method,
                    starts,
                    np.zeros(2),
                    np.ones(2),
                    {
                        "maxiter": 30,
                        "gtol": 1e-7,
                        "ftol": 1e-12,
                        "batch_size": 2,
                    },
                )
                np.testing.assert_allclose(
                    result.x, objective.target.numpy(), atol=1e-5
                )
                self.assertLess(result.fun, 1e-10)
                self.assertEqual(len(result.multistart_audit), 3)
                for chunk_history in result.iteration_history:
                    values = [
                        row["best_objective"] for row in chunk_history
                    ]
                    self.assertTrue(
                        all(b <= a + 1e-12 for a, b in zip(values, values[1:]))
                    )

    def test_bound_active_solution(self):
        objective = _QuadraticResidual()
        objective.target = torch.tensor([-1.0, 2.0], dtype=torch.float64)
        result = solve_batched_shooting(
            objective,
            "batched-lm",
            np.array([[0.5, 0.5]]),
            np.zeros(2),
            np.ones(2),
            {"maxiter": 20, "gtol": 1e-8},
        )
        np.testing.assert_allclose(result.x, [0.0, 1.0], atol=1e-6)

    def test_deterministic_chunking(self):
        objective = _QuadraticResidual()
        starts = np.array([[0.9, 0.1], [0.2, 0.2], [0.7, 0.8]])
        a = solve_batched_shooting(
            objective,
            "batched-bfgs",
            starts,
            np.zeros(2),
            np.ones(2),
            {"batch_size": 1, "maxiter": 20},
        )
        b = solve_batched_shooting(
            objective,
            "batched-bfgs",
            starts,
            np.zeros(2),
            np.ones(2),
            {"batch_size": 3, "maxiter": 20},
        )
        np.testing.assert_allclose(a.x, b.x, atol=1e-10)
        self.assertAlmostEqual(a.fun, b.fun, places=12)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_cuda_graph_replay_tracks_all_start_slots(self):
        objective = _QuadraticResidual()
        objective.est._device = torch.device("cuda")
        objective._sd = objective._sd.cuda()
        objective.target = objective.target.cuda()
        evaluator = BatchedShootingEvaluator(objective, capture=True)
        x0 = torch.tensor([[0.8, 0.1], [0.1, 0.9]], dtype=torch.float64, device="cuda")
        eager = objective.batched_value_and_grad(x0)
        captured = evaluator.value_grad(x0)
        torch.testing.assert_close(captured, eager)
        captured_snapshot = tuple(value.clone() for value in captured)
        x1 = x0 + torch.tensor(
            [[-0.05, 0.02], [0.03, -0.04]],
            dtype=torch.float64,
            device="cuda",
        )
        torch.testing.assert_close(
            evaluator.value_grad(x1), objective.batched_value_and_grad(x1)
        )
        torch.testing.assert_close(captured, captured_snapshot)


if __name__ == "__main__":
    unittest.main()
