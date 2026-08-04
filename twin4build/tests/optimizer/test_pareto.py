"""Tests for the augmented epsilon-constraint Pareto sweep
(:mod:`twin4build.optimizer._pareto`).

Model under test: one thermal zone with its heater schedule as the decision
variable; f1 = heater power ("min"), f2 = indoor temperature ("max") -- a
monotone trade-off with a known-shape front (more heat costs more energy and
raises the temperature), so front feasibility, monotonicity and dominance
have crisp expected outcomes.
"""

# Standard library imports
import datetime
import os
import shutil
import unittest

# Third party imports
import numpy as np

# Local application imports
import twin4build as tb
from twin4build.examples.gpu_benchmark_scaling import (
    N_HOURS,
    OPT_STEP,
    START,
    build_chain_model,
)
from twin4build.optimizer._pareto import _pareto_mask

tb._IS_TESTING = True


class TestParetoMask(unittest.TestCase):
    def test_dominated_point_is_filtered(self):
        # A clean min-min front plus one injected dominated point (worse in
        # BOTH objectives than the second front point).
        f1 = np.array([0.0, 1.0, 2.0, 1.5])
        f2 = np.array([3.0, 2.0, 1.0, 2.5])
        mask = _pareto_mask(f1, f2)
        np.testing.assert_array_equal(mask, [True, True, True, False])

    def test_duplicates_survive(self):
        # Ties (within tolerance) do not dominate each other.
        f1 = np.array([1.0, 1.0])
        f2 = np.array([2.0, 2.0])
        self.assertTrue(_pareto_mask(f1, f2).all())


class TestParetoFront(unittest.TestCase):
    """End-to-end sweeps on the 1-zone chain model."""

    N_POINTS = 5
    MAXITER = 30

    @classmethod
    def setUpClass(cls):
        cls.model, cls.zones, _, _, cls.heaters = build_chain_model(
            1, "test_pareto_model"
        )
        cls.simulator = tb.Simulator(cls.model)
        cls.end = START + datetime.timedelta(hours=N_HOURS)

    @classmethod
    def tearDownClass(cls):
        path = "generated_files/models/test_pareto_model"
        if os.path.exists(path):
            shutil.rmtree(path)

    def _sweep(self, batched_prepass: bool):
        optimizer = tb.Optimizer(self.simulator)
        return optimizer.pareto_front(
            start_time=START,
            end_time=self.end,
            step_size=OPT_STEP,
            variables=[(self.heaters[0], "scheduleValue", 0.0, 3000.0)],
            objective1=(self.heaters[0], "scheduleValue", "min"),
            objective2=(self.zones[0], "indoorTemperature", "max"),
            n_points=self.N_POINTS,
            batched_prepass=batched_prepass,
            options={"maxiter": self.MAXITER},
        )

    def test_front_properties(self):
        res = self._sweep(batched_prepass=True)

        n = len(res.eps)
        self.assertEqual(n, self.N_POINTS)
        for arr in (res.f1, res.f2, res.f1_min, res.f2_min, res.slope,
                    res.success, res.nit, res.pareto_mask):
            self.assertEqual(len(arr), n)
        self.assertEqual(res.theta.shape[0], n)

        # Anchors bracket the front: the first row is the f1 anchor (best
        # f1_min in the set), the last row the f2 anchor (best f2_min).
        self.assertLessEqual(res.f1_min[0], res.f1_min.min() + 1e-9)
        self.assertLessEqual(res.f2_min[-1], res.f2_min.min() + 1e-9)

        # Epsilon feasibility: f2_norm <= eps + tol at every swept point.
        ideal2, nadir2 = res.ideal[1], res.nadir[1]
        f2n = (res.f2_min - ideal2) / (nadir2 - ideal2)
        np.testing.assert_array_less(f2n, res.eps + 1e-3)

        # Monotone trade-off: tightening eps (better f2) costs f1.
        order = np.argsort(res.eps)[::-1]  # loosest (1.0) -> tightest (0.0)
        f1_sorted = res.f1_min[order]
        self.assertTrue(
            np.all(np.diff(f1_sorted) >= -1e-6),
            f"f1_min not monotone along the front: {f1_sorted}",
        )

        # A clean monotone front has no dominated points.
        self.assertTrue(res.pareto_mask.all())

        # The trade-off is real: physical heater power spans a wide range.
        self.assertGreater(res.f1[-1] - res.f1[0], 100.0)
        # Front slope (marginal f1 price of tightening f2) is positive.
        self.assertTrue(np.all(res.slope[1:-1] > 0))

    def test_prepass_matches_sequential(self):
        res_pre = self._sweep(batched_prepass=True)
        res_seq = self._sweep(batched_prepass=False)

        # Anchors are computed identically (the prepass only affects the
        # interior warm starts).
        np.testing.assert_allclose(res_pre.f1_min[0], res_seq.f1_min[0], atol=1e-8)
        np.testing.assert_allclose(res_pre.f2_min[-1], res_seq.f2_min[-1], atol=1e-8)

        # Interior points agree within a loose tolerance (both are polished
        # by the same exact SLSQP subproblems; only the warm start differs).
        np.testing.assert_allclose(
            res_pre.f1_min[1:-1], res_seq.f1_min[1:-1], atol=5e-2
        )
        np.testing.assert_allclose(
            res_pre.f2_min[1:-1], res_seq.f2_min[1:-1], atol=5e-2
        )

    def test_apply_writes_solution(self):
        res = self._sweep(batched_prepass=True)
        res.apply(len(res.eps) - 1)  # the f2 anchor: heater at full power
        applied = (
            self.heaters[0].output["scheduleValue"].history(i_s=0)
            .detach().cpu().numpy()
        )
        self.assertGreater(applied.mean(), 2000.0)


class TestOptimizeReturnsResult(unittest.TestCase):
    """Regression: _scipy_solver used to swallow the scipy result object."""

    def test_optimize_returns_scipy_result(self):
        model, zones, _, _, heaters = build_chain_model(
            1, "test_pareto_optret_model"
        )
        try:
            simulator = tb.Simulator(model)
            optimizer = tb.Optimizer(simulator)
            end = START + datetime.timedelta(hours=N_HOURS)
            result = optimizer.optimize(
                start_time=START,
                end_time=end,
                step_size=OPT_STEP,
                variables=[(heaters[0], "scheduleValue", 0.0, 3000.0)],
                objectives=[(heaters[0], "scheduleValue", "min")],
                method=("scipy", "SLSQP", "ad"),
                options={"maxiter": 2},
            )
            self.assertIsNotNone(result)
            self.assertTrue(hasattr(result, "x"))
            self.assertTrue(hasattr(result, "fun"))
        finally:
            path = "generated_files/models/test_pareto_optret_model"
            if os.path.exists(path):
                shutil.rmtree(path)


if __name__ == "__main__":
    unittest.main()
