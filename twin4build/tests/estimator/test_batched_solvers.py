import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import torch

from twin4build.estimator._batched_solvers import (
    BatchedObjectiveEvaluator,
    _batched_armijo,
    _safeguard_sqp_direction,
    _solve_box_qp,
    solve_batched_multistart,
)


class _QuadraticResidual:
    def __init__(self):
        self.est = SimpleNamespace(
            _device=torch.device("cpu"),
            simulator=SimpleNamespace(execution_backend="eager"),
        )
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


class _BarrierQuadratic(_QuadraticResidual):
    def __init__(self):
        super().__init__()
        self.target = torch.tensor([0.55], dtype=torch.float64)

    def loss(self, x, transform_mode=False):
        quadratic = torch.sum((x - self.target).square())
        return torch.where(x[0] > 0.6, torch.full_like(quadratic, torch.nan), quadratic)


class _InconsistentFlat(_QuadraticResidual):
    def batched_loss(self, x):
        return torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)

    def batched_value_and_grad(self, x):
        return self.batched_loss(x), torch.ones_like(x)


class TestBatchedShootingSolvers(unittest.TestCase):
    def test_derivative_bundles(self):
        objective = _QuadraticResidual()
        evaluator = BatchedObjectiveEvaluator(objective)
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
        for method in (
            "batched-sqp",
            "batched-bfgs",
            "batched-lm",
            "batched-newton",
        ):
            with self.subTest(method=method):
                result = solve_batched_multistart(
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
                    values = [row["best_objective"] for row in chunk_history]
                    self.assertTrue(
                        all(b <= a + 1e-12 for a, b in zip(values, values[1:]))
                    )

    def test_box_qp_releases_coupled_active_variables(self):
        hess = torch.tensor([[[2.0, 1.0], [1.0, 2.0]]], dtype=torch.float64)
        grad = torch.tensor([[-2.0, 0.0]], dtype=torch.float64)
        x = torch.full((1, 2), 0.5, dtype=torch.float64)
        step = _solve_box_qp(
            hess,
            grad,
            x,
            torch.zeros(2, dtype=torch.float64),
            torch.ones(2, dtype=torch.float64),
            torch.ones(1, dtype=torch.bool),
        )
        torch.testing.assert_close(
            step, torch.tensor([[0.5, -0.25]], dtype=torch.float64)
        )

    def test_sqp_resets_hessian_and_falls_back_for_non_descent_qp(self):
        dtype = torch.float64
        hess = torch.diag_embed(torch.tensor([[1e-7, 1e7]], dtype=dtype))
        grad = torch.tensor([[2.0, -1.0]], dtype=dtype)
        x = torch.full((1, 2), 0.5, dtype=dtype)
        lb = torch.zeros(2, dtype=dtype)
        ub = torch.ones(2, dtype=dtype)
        active = torch.ones(1, dtype=torch.bool)
        eye = torch.eye(2, dtype=dtype).expand(1, 2, 2).clone()
        ascent = torch.tensor([[0.1, 0.0]], dtype=dtype)

        with patch(
            "twin4build.estimator._batched_solvers._solve_box_qp",
            side_effect=(ascent, ascent),
        ):
            direction, guarded_hess, reset, fallback = _safeguard_sqp_direction(
                hess, grad, x, lb, ub, active, eye
            )

        self.assertTrue(bool(reset[0]))
        self.assertTrue(bool(fallback[0]))
        torch.testing.assert_close(guarded_hess, eye)
        torch.testing.assert_close(direction, -grad)
        self.assertLess(float(torch.sum(grad * direction)), 0.0)

    def test_box_qp_singular_solve_uses_feasible_projected_descent(self):
        dtype = torch.float64
        grad = torch.tensor([[3.0, -4.0]], dtype=dtype)
        x = torch.full((1, 2), 0.5, dtype=dtype)
        step = _solve_box_qp(
            torch.zeros((1, 2, 2), dtype=dtype),
            grad,
            x,
            torch.zeros(2, dtype=dtype),
            torch.ones(2, dtype=dtype),
            torch.ones(1, dtype=torch.bool),
        )

        torch.testing.assert_close(step, torch.tensor([[-0.5, 0.5]], dtype=dtype))
        self.assertLess(float(torch.sum(grad * step)), 0.0)

    def test_batched_armijo_reaches_small_descent_steps(self):
        class NarrowDescentEvaluator:
            @staticmethod
            def values(x):
                value = x[:, 0]
                return -value + 1e5 * value.square()

        dtype = torch.float64
        x = torch.zeros((1, 1), dtype=dtype)
        f = torch.zeros(1, dtype=dtype)
        g = -torch.ones((1, 1), dtype=dtype)
        direction = torch.ones((1, 1), dtype=dtype)
        active = torch.ones(1, dtype=torch.bool)
        x_new, f_new, accepted, _ = _batched_armijo(
            NarrowDescentEvaluator(),
            x,
            f,
            g,
            direction,
            active,
            -torch.ones(1, dtype=dtype),
            torch.ones(1, dtype=dtype),
            candidates=25,
        )

        self.assertTrue(bool(accepted[0]))
        self.assertGreater(float(x_new[0, 0]), 0.0)
        self.assertLess(float(f_new[0]), 0.0)

    def test_sqp_retries_rejected_line_search_after_hessian_reset(self):
        objective = _QuadraticResidual()
        call_count = 0

        def reject_then_accept(
            evaluator,
            x,
            f,
            g,
            direction,
            active,
            lb,
            ub,
            candidates,
            **kwargs,
        ):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                zeros = torch.zeros_like(active, dtype=torch.int64)
                return x, f, torch.zeros_like(active), zeros, f, zeros
            trial = torch.clamp(x + 0.1 * direction, lb, ub)
            ones = active.to(torch.int64)
            alpha = torch.full_like(f, 0.1)
            return trial, evaluator.values(trial), active.clone(), ones, alpha, ones

        with patch(
            "twin4build.estimator._batched_solvers._adaptive_armijo",
            side_effect=reject_then_accept,
        ):
            result = solve_batched_multistart(
                objective,
                "batched-sqp",
                np.array([[0.9, 0.1]]),
                np.zeros(2),
                np.ones(2),
                {"maxiter": 1},
            )

        audit = result.multistart_audit[0]
        self.assertEqual(call_count, 2)
        self.assertNotEqual(audit["status"], "failed_line_search")
        self.assertGreaterEqual(audit["hessian_resets"], 1)
        self.assertEqual(audit["restoration_uses"], 1)

    def test_sqp_block_curvature_converges_on_separable_objective(self):
        objective = _QuadraticResidual()
        objective.target = torch.tensor(
            [0.15, 0.25, 0.75, 0.85], dtype=torch.float64
        )
        result = solve_batched_multistart(
            objective,
            "batched-sqp",
            np.array([[0.9, 0.8, 0.2, 0.1]]),
            np.zeros(4),
            np.ones(4),
            {
                "maxiter": 30,
                "gtol": 1e-8,
                "sqp_curvature_blocks": [[0, 2], [1, 3]],
            },
        )
        np.testing.assert_allclose(result.x, objective.target.numpy(), atol=1e-6)
        self.assertEqual(result.curvature_block_sizes, [2, 2])
        self.assertTrue(result.success)

    def test_adaptive_line_search_recovers_after_nonfinite_candidate(self):
        objective = _BarrierQuadratic()
        result = solve_batched_multistart(
            objective,
            "batched-sqp",
            np.array([[0.0]]),
            np.zeros(1),
            np.ones(1),
            {"maxiter": 20, "gtol": 1e-8},
        )
        self.assertTrue(result.success)
        self.assertGreater(result.multistart_audit[0]["finite_candidates"], 0)
        self.assertNotEqual(result.status, "failed_nonfinite")
        np.testing.assert_allclose(result.x, objective.target.numpy(), atol=1e-6)

    def test_line_search_status_message_and_selected_counters(self):
        result = solve_batched_multistart(
            _InconsistentFlat(),
            "batched-sqp",
            np.array([[0.5, 0.5], [0.6, 0.6]]),
            np.zeros(2),
            np.ones(2),
            {"maxiter": 2, "batch_size": 2, "sqp_max_backtracks": 3},
        )
        self.assertEqual(result.status, "failed_line_search")
        self.assertIn("line search", result.message)
        self.assertEqual(result.nfev, result.multistart_audit[0]["nfev"])
        self.assertEqual(result.njev, result.multistart_audit[0]["njev"])
        self.assertEqual(result.aggregate_nfev, 2 * result.nfev)
        self.assertEqual(result.aggregate_njev, 2 * result.njev)

    def test_bound_active_solution(self):
        objective = _QuadraticResidual()
        objective.target = torch.tensor([-1.0, 2.0], dtype=torch.float64)
        result = solve_batched_multistart(
            objective,
            "batched-lm",
            np.array([[0.5, 0.5]]),
            np.zeros(2),
            np.ones(2),
            {"maxiter": 20, "gtol": 1e-8},
        )
        np.testing.assert_allclose(result.x, [0.0, 1.0], atol=1e-6)

    def test_lm_nonfinite_start_does_not_abort_other_slots(self):
        objective = _QuadraticResidual()
        starts = np.array([[np.nan, 0.5], [0.9, 0.1]])
        result = solve_batched_multistart(
            objective,
            "batched-lm",
            starts,
            np.zeros(2),
            np.ones(2),
            {"maxiter": 20, "gtol": 1e-8},
        )
        self.assertEqual(
            result.multistart_audit[0]["status"],
            "failed_nonfinite",
        )
        self.assertTrue(result.multistart_audit[1]["success"])
        np.testing.assert_allclose(result.x, objective.target.numpy(), atol=1e-6)

    def test_deterministic_chunking(self):
        objective = _QuadraticResidual()
        starts = np.array([[0.9, 0.1], [0.2, 0.2], [0.7, 0.8]])
        a = solve_batched_multistart(
            objective,
            "batched-bfgs",
            starts,
            np.zeros(2),
            np.ones(2),
            {"batch_size": 1, "maxiter": 20},
        )
        b = solve_batched_multistart(
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
        objective.est.simulator.execution_backend = "cuda_graph"
        objective._sd = objective._sd.cuda()
        objective.target = objective.target.cuda()
        evaluator = BatchedObjectiveEvaluator(objective)
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
        self.assertEqual(evaluator.stats["value_grad"]["captures"], 1)
        self.assertEqual(evaluator.stats["value_grad"]["replays"], 1)


if __name__ == "__main__":
    unittest.main()
