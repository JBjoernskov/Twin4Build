"""Regression tests for two collocation warm-start defects found on the
canonical 1-zone benchmark (Sep 2026):

1. The boundary-state box must CONTAIN the warm-start trajectory.  A fixed
   +/-6 (std-normalized) box clipped the warm start's initial transient, so
   IPOPT projected a feasible warm start into an infeasible one and excluded
   the true trajectory from the feasible set.
2. Early stopping must not count infeasible iterates as stagnation.  IPOPT's
   cold-start iterates sit above ``feas_tol`` for long stretches while the
   objective is still falling; counting them stopped the solve at its first
   (poor) feasible incumbent.
"""

import numpy as np
import pytest
import torch

from twin4build.estimator._collocation import _DEFAULT_STATE_MARGIN, _trajectory_box


class TestTrajectoryBox:
    def test_box_contains_every_warm_start_value_with_margin(self):
        torch.manual_seed(0)
        y = torch.randn(50, 4, dtype=torch.float64)
        y[0, 1] = 18.6  # an initial transient far outside a +/-6 box
        y[3, 2] = -12.0
        lo, hi = _trajectory_box(y, _DEFAULT_STATE_MARGIN)
        assert lo.shape == hi.shape == (50 * 4,)
        flat = y.reshape(-1).numpy()
        assert np.all(flat >= lo + _DEFAULT_STATE_MARGIN - 1e-12)
        assert np.all(flat <= hi - _DEFAULT_STATE_MARGIN + 1e-12)
        # Per-dimension, tiled over segments: dimension 1 is widened by the
        # transient, dimension 0 is not.
        hi_dim = hi.reshape(50, 4)[0]
        assert hi_dim[1] == pytest.approx(18.6 + _DEFAULT_STATE_MARGIN)
        assert hi_dim[0] < 18.6
        assert np.array_equal(hi.reshape(50, 4)[7], hi_dim)

    def test_margin_is_applied_symmetrically(self):
        y = torch.zeros(3, 2, dtype=torch.float64)
        lo, hi = _trajectory_box(y, 2.5)
        assert np.allclose(lo, -2.5) and np.allclose(hi, 2.5)


def _es_state(**overrides):
    es = {
        "feas_tol": 1e-2,
        "patience": 3,
        "min_delta_rel": 1e-3,
        "theta_tol": 1e-4,
        "n_theta": 2,
        "f_best": np.inf,
        "z_best": None,
        "stall_f": 0,
        "stall_theta": 0,
        "stop_reason": None,
    }
    es.update(overrides)
    return es


class TestEarlyStoppingCounter:
    """The pure update rule, driven with a synthetic iterate sequence that
    mirrors the cold-start failure: a poor first feasible incumbent, then a
    long INFEASIBLE descent, then a much better feasible landing point."""

    def test_infeasible_descent_does_not_count_as_stagnation(self):
        from twin4build.solvers.ipopt import early_stopping_step

        es = _es_state()
        x = np.zeros(4)
        assert not early_stopping_step(es, x, 96.0, 0.005)  # first feasible incumbent
        for k in range(40):  # infeasible, objective falling 9 -> 1.05
            f = 9.0 - k * 0.2
            assert not early_stopping_step(es, x + 0.01 * k, f, 0.05), es["stop_reason"]
        assert es["stall_f"] == 0 and es["stall_theta"] == 0
        assert es["f_best"] == 96.0  # infeasible iterates never become the incumbent
        assert not early_stopping_step(es, x + 1.0, 1.05, 0.001)
        assert es["f_best"] == 1.05

    def test_feasible_stagnation_still_stops_and_keeps_best(self):
        from twin4build.solvers.ipopt import early_stopping_step

        es = _es_state()
        x = np.zeros(4)
        assert not early_stopping_step(es, x, 2.0, 0.0)
        assert not early_stopping_step(es, x + 1.0, 1.0, 0.0)  # improvement resets
        stops = [early_stopping_step(es, x + 1.0, 1.0 + 1e-6, 0.0) for _ in range(3)]
        assert stops == [False, False, True]
        assert "stagnant" in es["stop_reason"]
        assert es["f_best"] == 1.0

    def test_no_incumbent_until_first_feasible(self):
        from twin4build.solvers.ipopt import early_stopping_step

        es = _es_state()
        for k in range(10):
            assert not early_stopping_step(es, np.full(4, float(k)), 100.0 - k, 1.0)
        assert es["z_best"] is None and es["f_best"] == np.inf


class TestEarlyStoppingIgnoresInfeasibleIterates:
    """Drive ``solve_ipopt_constrained`` on a tiny NLP whose start is far from
    the equality constraint, so IPOPT spends its first iterations infeasible.
    With the old counter that approach phase counted as 'stagnant' and a
    small patience cut the solve at its first feasible incumbent; now the
    solve must reach the constrained optimum."""

    @staticmethod
    def _problem():
        def obj(z):
            return float((z[0] - 3.0) ** 2 + (z[1] + 1.0) ** 2)

        def grad(z):
            return np.array([2 * (z[0] - 3.0), 2 * (z[1] + 1.0)])

        def g(z):
            return np.array([z[0] * z[1] - 1.0])

        def gjac(z):
            return np.array([z[1], z[0]])

        return obj, grad, g, gjac

    def test_infeasible_approach_does_not_trigger_stop(self):
        pytest.importorskip("casadi")
        from scipy.optimize import minimize

        from twin4build.solvers.ipopt import solve_ipopt_constrained

        obj, grad, g, gjac = self._problem()
        ref = minimize(
            obj,
            np.array([2.0, 0.5]),
            jac=grad,
            method="SLSQP",
            constraints=[{"type": "eq", "fun": g, "jac": gjac}],
            options={"ftol": 1e-12, "maxiter": 500},
        )
        assert ref.success

        res = solve_ipopt_constrained(
            # Same branch of x*y = 1 as the reference, but far from it
            # (x*y = 25), so the first iterations are infeasible.
            np.array([5.0, 5.0]),
            np.array([-10.0, -10.0]),
            np.array([10.0, 10.0]),
            obj,
            grad,
            1,
            g,
            gjac,
            np.array([0, 0]),
            np.array([0, 1]),
            options={"maxiter": 200, "tol": 1e-8},
            early_stopping={"n_theta": 2, "patience": 2, "feas_tol": 1e-6},
        )
        assert abs(g(res.x)[0]) < 1e-5, res.message
        assert obj(res.x) == pytest.approx(ref.fun, abs=1e-4), res.message
