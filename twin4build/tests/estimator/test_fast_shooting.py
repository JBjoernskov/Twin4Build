# Standard library imports
import datetime
import unittest

# Third party imports
import numpy as np
import torch

# Local application imports
import twin4build as tb

tb._IS_TESTING = True

from twin4build.examples.collocation_comparison import (
    EXAMPLE_START,
    STEP_SIZE,
    example_measurements,
    example_parameters,
    load_model,
)


class TestFastSingleShooting(unittest.TestCase):
    """Numerical-equivalence regression check for the fast single-shooting objective.

    The composed-map objective (``options={"fast": True}``) is enabled WITHOUT
    per-run validation against the object-graph objective.  Equivalence holds
    by construction -- each composable component's ``do_step`` is a thin
    port-I/O wrapper delegating to the same ``forward`` the composer threads --
    and THIS test is the tripwire guarding that contract (plus the composer's
    wiring, capture and sensor-lag logic).  It exercises the full 19-parameter
    / 4-sensor estimation example (state-space components, PID controller,
    dampers, the CO2 feedback loop, a lagged pass-through sensor and
    vector-port max -- every composer feature) and asserts value AND gradient
    parity at x0 and at randomly perturbed thetas.
    """

    TOL_VALUE = 1e-5  # relative objective-value tolerance
    TOL_GRAD = 1e-4  # relative gradient tolerance (inf-norm, scaled)

    @classmethod
    def setUpClass(cls):
        model = load_model()
        cls.estimator = tb.Estimator(tb.Simulator(model))
        cls.parameters = example_parameters(model)
        cls.measurements = example_measurements(model)
        start = EXAMPLE_START[0]
        end = start + datetime.timedelta(hours=24)

        # One SLSQP iteration populates the estimator's internal state
        # (normalized bounds, actual_readings, ...) and builds the fast
        # objective through the production code path.
        cls.estimator.estimate(
            parameters=cls.parameters,
            measurements=cls.measurements,
            start_time=[start],
            end_time=[end],
            step_size=STEP_SIZE,
            n_warmup=5,
            method=("scipy", "SLSQP", "ad"),
            options={"maxiter": 1, "fast": True},
        )

    def _eval(self, theta_np, use_fast):
        """(objective value, gradient) at a normalized theta via either path."""
        est = self.estimator
        fast = est._fast_obj
        self.assertIsNotNone(fast, "fast single-shooting objective was not built")
        est._fast_obj = fast if use_fast else None
        est._mse_scaled = 1.0  # identical fixed scaling for both paths
        try:
            z = torch.tensor(theta_np, dtype=torch.float64, requires_grad=True)
            f = est._obj(z, "scalar")
            (g,) = torch.autograd.grad(f, z)
            return float(f.detach()), g.numpy()
        finally:
            est._fast_obj = fast
            est._mse_scaled = None

    def test_fast_objective_matches_object_graph(self):
        est = self.estimator
        x0 = np.asarray(est._x0_norm, dtype=np.float64)
        lbn = np.asarray(est._lb_norm, dtype=np.float64)
        ubn = np.asarray(est._ub_norm, dtype=np.float64)

        rng = np.random.default_rng(42)
        thetas = [x0] + [
            np.clip(x0 + scale * (rng.random(x0.shape) - 0.5), lbn, ubn)
            for scale in (0.05, 0.2)
        ]

        for i, theta in enumerate(thetas):
            f_slow, g_slow = self._eval(theta, use_fast=False)
            f_fast, g_fast = self._eval(theta, use_fast=True)

            rel_val = abs(f_fast - f_slow) / max(1e-12, abs(f_slow))
            self.assertLess(
                rel_val,
                self.TOL_VALUE,
                f"theta[{i}]: objective value mismatch "
                f"(slow={f_slow:.10g}, fast={f_fast:.10g}, rel={rel_val:.3e})",
            )

            gscale = max(1e-12, float(np.abs(g_slow).max()))
            rel_grad = float(np.abs(g_fast - g_slow).max()) / gscale
            self.assertLess(
                rel_grad,
                self.TOL_GRAD,
                f"theta[{i}]: gradient mismatch (rel inf-norm={rel_grad:.3e})",
            )

    def test_per_sensor_rmse_matches(self):
        est = self.estimator
        x0 = np.asarray(est._x0_norm, dtype=np.float64)

        self._eval(x0, use_fast=False)
        rmse_slow = dict(est._last_rmse_per_sensor)
        self._eval(x0, use_fast=True)
        rmse_fast = dict(est._last_rmse_per_sensor)

        self.assertEqual(set(rmse_slow), set(rmse_fast))
        for sensor_id, slow in rmse_slow.items():
            fast = rmse_fast[sensor_id]
            rel = abs(fast - slow) / max(1e-12, abs(slow))
            self.assertLess(
                rel,
                1e-4,
                f"{sensor_id}: per-sensor RMSE mismatch "
                f"(slow={slow:.8g}, fast={fast:.8g}, rel={rel:.3e})",
            )


def shared_example_parameters(model):
    """The example parameter set with the two damper flow rates collapsed
    into ONE shared parameter (a single theta entry driving both dampers)."""
    c = model.components
    supply = c["office_supply_damper"]
    exhaust = c["office_exhaust_damper"]
    params = [p for p in example_parameters(model) if p[0] not in (supply, exhaust)]
    params.append(([supply, exhaust], "nominalAirFlowRate", 0.1, 0.001, 1.0, "shared"))
    return params


class TestFastSingleShootingSharedTheta(TestFastSingleShooting):
    """Same value+gradient parity checks with a SHARED parameter group.

    Shared parameters make theta shorter than the flat (component, attr) list;
    the indexed theta spec (``Estimator._composer_theta_spec``) routes both
    dampers' ``nominalAirFlowRate`` to one theta slot.  Inherits both parity
    tests from :class:`TestFastSingleShooting`.
    """

    @classmethod
    def setUpClass(cls):
        model = load_model()
        cls.estimator = tb.Estimator(tb.Simulator(model))
        cls.parameters = shared_example_parameters(model)
        cls.measurements = example_measurements(model)
        start = EXAMPLE_START[0]
        end = start + datetime.timedelta(hours=24)

        cls.estimator.estimate(
            parameters=cls.parameters,
            measurements=cls.measurements,
            start_time=[start],
            end_time=[end],
            step_size=STEP_SIZE,
            n_warmup=5,
            method=("scipy", "SLSQP", "ad"),
            options={"maxiter": 1, "fast": True},
        )


if __name__ == "__main__":
    unittest.main()
