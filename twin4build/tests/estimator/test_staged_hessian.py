"""Focused tests for staged Gauss-Newton/exact-Hessian plumbing."""

import unittest
import warnings

import numpy as np

from twin4build.estimator._casadi_ipopt import solve_ipopt_constrained
from twin4build.estimator._transcription import (
    _normalize_hessian_hybrid,
    _normalize_hessian_stages,
)


class TestHessianStageConfiguration(unittest.TestCase):
    def test_defaults_and_overrides(self):
        self.assertIsNone(_normalize_hessian_stages(False))
        defaults = _normalize_hessian_stages(True)
        self.assertEqual(defaults["stage1_maxiter"], 40)
        self.assertEqual(defaults["switch_rule"], "feasible_stall")
        custom = _normalize_hessian_stages(
            {"switch_rule": "cost_aware", "probe_interval": 3}
        )
        self.assertEqual(custom["switch_rule"], "cost_aware")
        self.assertEqual(custom["probe_interval"], 3)

    def test_invalid_configuration_fails_loudly(self):
        with self.assertRaisesRegex(ValueError, "stage1_maxiter"):
            _normalize_hessian_stages({"stage1_maxiter": 0})
        with self.assertRaisesRegex(ValueError, "switch_rule"):
            _normalize_hessian_stages({"switch_rule": "unknown"})

    def test_in_place_hybrid_defaults_and_validation(self):
        self.assertIsNone(_normalize_hessian_hybrid(False))
        defaults = _normalize_hessian_hybrid(True)
        self.assertEqual(defaults["min_iterations"], 5)
        self.assertEqual(defaults["hard_max_iterations"], 12)
        self.assertEqual(defaults["contraction_window"], 4)
        with self.assertRaisesRegex(ValueError, "hard_max_iterations"):
            _normalize_hessian_hybrid({"min_iterations": 5, "hard_max_iterations": 4})


class TestIpoptDualWarmStart(unittest.TestCase):
    @staticmethod
    def _solve(**kwargs):
        return solve_ipopt_constrained(
            x0=np.array([0.0]),
            lb=np.array([-2.0]),
            ub=np.array([2.0]),
            fun=lambda x: float((x[0] - 1.0) ** 2),
            grad=lambda x: np.array([2.0 * (x[0] - 1.0)]),
            n_g=1,
            g_fun=lambda x: np.array([x[0] - 0.5]),
            g_jac_vals=lambda x: np.array([1.0]),
            jac_rows=np.array([0]),
            jac_cols=np.array([0]),
            options={"maxiter": 20},
            hess_vals=lambda x, sigma, lam: np.array([2.0 * sigma]),
            hess_rows=np.array([0]),
            hess_cols=np.array([0]),
            early_stopping=None,
            **kwargs,
        )

    def test_matching_duals_are_returned_and_reused(self):
        first = self._solve()
        self.assertAlmostEqual(first.x[0], 0.5, places=7)
        self.assertEqual(first.lam_x.shape, (1,))
        self.assertEqual(first.lam_g.shape, (1,))
        self.assertGreaterEqual(first.elapsed, 0.0)

        second = self._solve(
            lam_x0=first.lam_x,
            lam_g0=first.lam_g,
            mu_init=first.mu_final,
        )
        self.assertTrue(second.warm_start_duals)
        self.assertAlmostEqual(second.x[0], 0.5, places=7)

    def test_invalid_duals_fall_back_to_primal_only(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = self._solve(
                lam_x0=np.array([np.nan]),
                lam_g0=np.array([0.0]),
            )
        self.assertFalse(result.warm_start_duals)
        self.assertTrue(any("dual warm start" in str(item.message) for item in caught))

    def test_guarded_in_place_hessian_transition(self):
        state = {"use_exact": False}
        modes = []

        def hess(x, sigma, lam):
            modes.append("exact" if state["use_exact"] else "structured")
            return np.array(
                [2.0 * sigma + (2.0 * lam[0] if state["use_exact"] else 0.0)]
            )

        result = solve_ipopt_constrained(
            x0=np.array([0.25]),
            lb=np.array([-2.0]),
            ub=np.array([2.0]),
            fun=lambda x: float((x[0] - 1.5) ** 2),
            grad=lambda x: np.array([2.0 * (x[0] - 1.5)]),
            n_g=1,
            g_fun=lambda x: np.array([x[0] ** 2 - 1.0]),
            g_jac_vals=lambda x: np.array([2.0 * x[0]]),
            jac_rows=np.array([0]),
            jac_cols=np.array([0]),
            options={"maxiter": 20, "acceptable_iter": 0},
            hess_vals=hess,
            hess_rows=np.array([0]),
            hess_cols=np.array([0]),
            early_stopping={
                "switch_rule": "guarded",
                "switch_in_place": True,
                "switch_state": state,
                "min_iterations": 1,
                "hard_max_iterations": 2,
                "feas_tol": 1e-8,
            },
        )
        self.assertTrue(state["use_exact"])
        self.assertEqual(result.switch_iteration, 2)
        self.assertIn("structured", modes)
        self.assertIn("exact", modes)
        self.assertIn("objective", result.callback_stats)
        self.assertIn("n_call_nlp_f", result.solver_counters)


if __name__ == "__main__":
    unittest.main()
