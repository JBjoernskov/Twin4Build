"""Local identifiability analysis (``twin4build.estimator._identifiability``).

Unit tests on hand-made Jacobians where the answer is known, plus one
integration run through ``Estimator.estimate`` on the example model that
checks the report is produced, attached to the result and names every
parameter.
"""

# Standard library imports
import datetime
import unittest

# Third party imports
import numpy as np

# Local application imports
import twin4build as tb
from twin4build.estimator._identifiability import (
    DEAD_REL_NORM,
    TRADEOFF_CORR,
    analyze,
)

tb._IS_TESTING = True


def _independent_jacobian(n_res=200, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_res, 3))


class TestAnalyze(unittest.TestCase):
    def test_well_posed_problem_is_clean(self):
        J = _independent_jacobian()
        r = np.random.default_rng(1).standard_normal(J.shape[0]) * 1e-3
        rep = analyze(J, r, ["a", "b", "c"])
        self.assertTrue(rep.clean, rep.lines())
        self.assertEqual(rep.dead, [])
        self.assertEqual(rep.flat_directions, [])
        self.assertEqual(rep.tradeoffs, [])
        self.assertTrue(np.all(rep.std_err < 1.0))
        self.assertLess(rep.condition_number, 5.0)

    def test_dead_column_is_reported(self):
        J = _independent_jacobian()
        J[:, 1] = 0.0
        rep = analyze(J, None, ["a", "dead", "c"])
        self.assertEqual(rep.dead, [1])
        self.assertLess(rep.rel_col_norm[1], DEAD_REL_NORM)
        self.assertIn("dead: no residual reacts to it", rep.lines()[0])
        # the live parameters are still analysed
        self.assertEqual(rep.tradeoffs, [])

    def test_exactly_collinear_pair_is_a_flat_direction_and_a_tradeoff(self):
        J = _independent_jacobian()
        J[:, 1] = 2.0 * J[:, 0]  # b is a scaled copy of a: only a + 2b is determined
        rep = analyze(J, None, ["a", "b", "c"])
        self.assertEqual(len(rep.flat_directions), 1)
        rel_sv, loads = rep.flat_directions[0]
        self.assertLess(rel_sv, 1e-6)
        self.assertEqual(sorted(i for i, _ in loads), [0, 1])
        # the flat direction is (1, -1)/sqrt(2) in unit-column coordinates
        self.assertAlmostEqual(abs(loads[0][1]), abs(loads[1][1]), places=6)
        self.assertEqual([(i, j) for i, j, _ in rep.tradeoffs], [(0, 1)])
        self.assertGreaterEqual(abs(rep.tradeoffs[0][2]), TRADEOFF_CORR)
        self.assertFalse(rep.clean)
        text = "\n".join(rep.lines())
        self.assertIn("flat direction", text)
        self.assertIn("trade-off a <-> b", text)

    def test_nearly_collinear_pair_is_a_tradeoff_without_flat_direction(self):
        rng = np.random.default_rng(3)
        J = rng.standard_normal((400, 3))
        J[:, 1] = J[:, 0] + 0.05 * rng.standard_normal(400)
        rep = analyze(J, None, ["a", "b", "c"])
        self.assertEqual(rep.flat_directions, [])
        self.assertEqual([(i, j) for i, j, _ in rep.tradeoffs], [(0, 1)])
        self.assertGreater(abs(rep.tradeoffs[0][2]), TRADEOFF_CORR)

    def test_noise_level_sets_unpinned(self):
        J = _independent_jacobian() * 1e-3  # weak sensitivity across the board
        r = np.random.default_rng(1).standard_normal(J.shape[0])  # loud noise
        rep = analyze(J, r, ["a", "b", "c"])
        self.assertEqual(rep.unpinned, [0, 1, 2])
        self.assertTrue(all("not pinned" in line for line in rep.lines()))

    def test_bounds_are_flagged(self):
        J = _independent_jacobian()
        rep = analyze(J, None, ["a", "b", "c"], x=[0.0, 0.5, 1.0], lb=[0, 0, 0], ub=[1, 1, 1])
        self.assertEqual(rep.at_lower, [0])
        self.assertEqual(rep.at_upper, [2])

    def test_scale_invariance(self):
        J = _independent_jacobian()
        J[:, 2] = -0.5 * J[:, 1]
        r = np.random.default_rng(1).standard_normal(J.shape[0])
        a = analyze(J, r, ["a", "b", "c"])
        b = analyze(J * 37.0, r * 37.0, ["a", "b", "c"])
        np.testing.assert_allclose(a.std_err, b.std_err, rtol=1e-9)
        self.assertEqual(a.tradeoffs[0][:2], b.tradeoffs[0][:2])
        self.assertAlmostEqual(a.tradeoffs[0][2], b.tradeoffs[0][2], places=9)

    def test_more_parameters_than_residuals(self):
        J = np.random.default_rng(5).standard_normal((2, 4))
        rep = analyze(J, None, list("abcd"))
        # rank 2 of 4: the SVD returns two values, no crash, both live
        self.assertEqual(rep.dead, [])
        self.assertEqual(len(rep.singular_values), 2)

    def test_all_zero_jacobian(self):
        rep = analyze(np.zeros((10, 2)), np.zeros(10), ["a", "b"])
        self.assertEqual(rep.dead, [0, 1])
        self.assertFalse(rep.clean)


class TestEstimatorReport(unittest.TestCase):
    """One estimation step on the example model produces the report."""

    def test_report_attached_to_result(self):
        from twin4build.tests.estimator.example_fixture import (
            EXAMPLE_START,
            STEP_SIZE,
            example_measurements,
            example_parameters,
            load_model,
        )

        model = load_model()
        estimator = tb.Estimator(tb.Simulator(model, execution_mode="functional"))
        start = EXAMPLE_START[0]
        result = estimator.estimate(
            parameters=example_parameters(model),
            measurements=example_measurements(model),
            start_time=[start],
            end_time=[start + datetime.timedelta(hours=12)],
            step_size=STEP_SIZE,
            n_warmup=5,
            method=("scipy", "SLSQP", "ad"),
            options={"maxiter": 1},
            identifiability=True,
        )
        rep = result["identifiability"]
        n_theta = int(estimator._x0_norm.size)
        self.assertEqual(len(rep["names"]), n_theta)
        self.assertTrue(all(rep["names"]))
        self.assertEqual(rep["rel_col_norm"].shape, (n_theta,))
        self.assertEqual(rep["std_err"].shape, (n_theta,))
        for i, j, c in rep["tradeoffs"]:
            self.assertLess(i, j)
            self.assertLessEqual(abs(c), 1.0 + 1e-9)
        self.assertIsInstance(rep["lines"], list)

    def test_report_can_be_switched_off(self):
        from twin4build.tests.estimator.example_fixture import (
            EXAMPLE_START,
            STEP_SIZE,
            example_measurements,
            example_parameters,
            load_model,
        )

        model = load_model()
        estimator = tb.Estimator(tb.Simulator(model, execution_mode="functional"))
        start = EXAMPLE_START[0]
        result = estimator.estimate(
            parameters=example_parameters(model),
            measurements=example_measurements(model),
            start_time=[start],
            end_time=[start + datetime.timedelta(hours=6)],
            step_size=STEP_SIZE,
            n_warmup=5,
            method=("scipy", "SLSQP", "ad"),
            options={"maxiter": 1},
            identifiability=False,
        )
        self.assertNotIn("identifiability", result)


if __name__ == "__main__":
    unittest.main()
