# Standard library imports
import datetime
import unittest

# Third party imports
import numpy as np
import pandas as pd
import torch
from dateutil import tz

# Local application imports
import twin4build as tb

tb._IS_TESTING = True

START = datetime.datetime(2024, 1, 4, tzinfo=tz.gettz("Europe/Copenhagen"))
STEP_SIZE = 600
N_HOURS = 24


def build_two_zone_model():
    """Two thermal zones coupled by ONE WallTorchSystem.

    Zone A is heated (schedule-driven heat gain); zone B is passive and only
    receives heat through the shared partition wall.  Both zone<->wall loops
    contain a cut-feedback edge (Gauss-Seidel lag), exactly the coupling
    pattern the composed fast paths must reproduce.
    """
    model = tb.Model(id="test_two_zone_wall")

    def make_zone(zone_id):
        return tb.BuildingSpaceThermalTorchSystem(
            C_air=1e6,
            C_wall=5e6,
            R_out=0.01,
            R_in=0.01,
            f_wall=0.0,
            f_air=0.0,
            Q_occ_gain=100.0,
            id=zone_id,
        )

    zone_a = make_zone("ZoneA")
    zone_b = make_zone("ZoneB")
    wall = tb.WallTorchSystem(C=2e5, R_a=0.02, R_b=0.02, id="PartitionWall")

    outdoor = tb.ScheduleSystem(
        weekDayRulesetDict={
            "ruleset_default_value": 5.0,
            "ruleset_start_minute": [0],
            "ruleset_end_minute": [0],
            "ruleset_start_hour": [10],
            "ruleset_end_hour": [18],
            "ruleset_value": [10.0],
        },
        id="Outdoor",
    )
    zero = tb.ScheduleSystem(
        weekDayRulesetDict={"ruleset_default_value": 0.0}, id="Zero"
    )
    supply_air_temp = tb.ScheduleSystem(
        weekDayRulesetDict={"ruleset_default_value": 20.0}, id="SupplyAirTemp"
    )
    heater_a = tb.ScheduleSystem(
        weekDayRulesetDict={
            "ruleset_default_value": 0.0,
            "ruleset_start_minute": [0],
            "ruleset_end_minute": [0],
            "ruleset_start_hour": [6],
            "ruleset_end_hour": [20],
            "ruleset_value": [1500.0],
        },
        id="HeaterA",
    )

    for zone in (zone_a, zone_b):
        model.add_connection(outdoor, zone, "scheduleValue", "outdoorTemperature")
        model.add_connection(zero, zone, "scheduleValue", "supplyAirFlowRate")
        model.add_connection(zero, zone, "scheduleValue", "exhaustAirFlowRate")
        model.add_connection(
            supply_air_temp, zone, "scheduleValue", "supplyAirTemperature"
        )
        model.add_connection(zero, zone, "scheduleValue", "globalIrradiation")
        model.add_connection(zero, zone, "scheduleValue", "numberOfPeople")
    model.add_connection(heater_a, zone_a, "scheduleValue", "heatGain")
    model.add_connection(zero, zone_b, "scheduleValue", "heatGain")

    # The partition wall: both sides fed by the zones' air temperatures, both
    # heat flows returned to the zones' wallHeatGain ports.
    model.add_connection(zone_a, wall, "indoorTemperature", "temperatureA")
    model.add_connection(zone_b, wall, "indoorTemperature", "temperatureB")
    model.add_connection(
        wall, zone_a, "heatFlowRateA", "wallHeatGain", input_port_index=0
    )
    model.add_connection(
        wall, zone_b, "heatFlowRateB", "wallHeatGain", input_port_index=0
    )

    # Calibration sensors on both zone temperatures (synthetic readings are
    # attached in setUpClass after a reference simulation).
    sensor_a = tb.SensorSystem(id="TempSensorA")
    sensor_b = tb.SensorSystem(id="TempSensorB")
    model.add_connection(zone_a, sensor_a, "indoorTemperature", "measuredValue")
    model.add_connection(zone_b, sensor_b, "indoorTemperature", "measuredValue")

    model.load()
    return model, zone_a, zone_b, wall, sensor_a, sensor_b


def attach_synthetic_readings(model, sensor, series, index):
    sensor.df = pd.DataFrame({"value": series}, index=index)


class TestTwoZoneWall(unittest.TestCase):
    """Two-zone integration: interzonal flux consistency in simulation, fast
    single-shooting value+gradient parity with wall parameters estimated, and
    a collocation smoke run."""

    @classmethod
    def setUpClass(cls):
        (
            cls.model,
            cls.zone_a,
            cls.zone_b,
            cls.wall,
            cls.sensor_a,
            cls.sensor_b,
        ) = build_two_zone_model()
        cls.simulator = tb.Simulator(cls.model)
        cls.start = START
        cls.end = START + datetime.timedelta(hours=N_HOURS)

        cls.simulator.simulate(
            start_time=cls.start, end_time=cls.end, step_size=STEP_SIZE
        )

        cls.t_a = cls.zone_a.output["indoorTemperature"].history().detach().flatten()
        cls.t_b = cls.zone_b.output["indoorTemperature"].history().detach().flatten()
        cls.q_a = cls.wall.output["heatFlowRateA"].history().detach().flatten()
        cls.q_b = cls.wall.output["heatFlowRateB"].history().detach().flatten()
        cls.t_w = cls.wall.output["wallTemperature"].history().detach().flatten()

        # Synthetic "measurements": the simulated truth plus deterministic
        # noise, sampled on the simulation grid.
        index = pd.date_range(
            start=cls.start, periods=len(cls.t_a), freq=f"{STEP_SIZE}s"
        )
        rng = np.random.default_rng(0)
        attach_synthetic_readings(
            cls.model, cls.sensor_a,
            cls.t_a.numpy() + 0.05 * rng.standard_normal(len(cls.t_a)), index,
        )
        attach_synthetic_readings(
            cls.model, cls.sensor_b,
            cls.t_b.numpy() + 0.05 * rng.standard_normal(len(cls.t_b)), index,
        )

    # ------------------------------------------------------------------
    # 1. Physics of the coupled simulation
    # ------------------------------------------------------------------

    def test_heat_flows_from_hot_to_cold_zone(self):
        """Zone A is heated, so in the daytime steady stretch the wall must
        extract heat from A (Q_a < 0) and deliver heat to B (Q_b > 0)."""
        # Skip the initial transient (identical initial temperatures).
        sl = slice(len(self.t_a) // 2, None)
        self.assertTrue(bool((self.t_a[sl] > self.t_b[sl]).all()))
        self.assertTrue(bool((self.q_a[sl] < 0).all()))
        self.assertTrue(bool((self.q_b[sl] > 0).all()))
        # The wall temperature lies between the two zone temperatures.
        self.assertTrue(bool((self.t_w[sl] < self.t_a[sl]).all()))
        self.assertTrue(bool((self.t_w[sl] > self.t_b[sl]).all()))

    def test_interzonal_energy_balance(self):
        """Integrated over the run, the heat leaving zone A equals the heat
        entering zone B plus the energy stored in the wall -- the balance the
        single-wall-state design guarantees.  Tolerance covers the ZOH
        sampling error of end-of-step flux outputs."""
        dt = float(STEP_SIZE)
        q_sum = float((self.q_a + self.q_b).sum()) * dt
        stored = float(self.wall.C.get().flatten()[0]) * float(
            self.t_w[-1] - self.t_w[0]
        )
        scale = max(abs(stored), dt * float(self.q_b.abs().sum()))
        self.assertLess(abs(q_sum + stored) / scale, 0.15)

    # ------------------------------------------------------------------
    # 2. Fast single-shooting parity with wall parameters in theta
    # ------------------------------------------------------------------

    def _estimation_setup(self):
        parameters = [
            (self.zone_a, "C_air", 1e6, 1e5, 1e7),
            (self.zone_b, "C_air", 1e6, 1e5, 1e7),
            (self.wall, "C", 2e5, 1e4, 1e7),
            (self.wall, "R_a", 0.02, 1e-3, 1.0),
            (self.wall, "R_b", 0.02, 1e-3, 1.0),
        ]
        measurements = [(self.sensor_a, 0.05), (self.sensor_b, 0.05)]
        return parameters, measurements

    def test_fast_shooting_parity_with_wall_theta(self):
        estimator = tb.Estimator(self.simulator)
        parameters, measurements = self._estimation_setup()
        estimator.estimate(
            parameters=parameters,
            measurements=measurements,
            start_time=[self.start],
            end_time=[self.end],
            step_size=STEP_SIZE,
            n_warmup=0,
            method=("scipy", "SLSQP", "ad"),
            options={"maxiter": 1, "fast": True},
        )
        fast = estimator._fast_obj
        self.assertIsNotNone(
            fast, "fast single-shooting objective was not built for the wall model"
        )

        def eval_obj(theta_np, use_fast):
            estimator._fast_obj = fast if use_fast else None
            estimator._mse_scaled = 1.0
            try:
                z = torch.tensor(theta_np, dtype=torch.float64, requires_grad=True)
                f = estimator._obj(z, "scalar")
                (g,) = torch.autograd.grad(f, z)
                return float(f.detach()), g.numpy()
            finally:
                estimator._fast_obj = fast
                estimator._mse_scaled = None

        x0 = np.asarray(estimator._x0_norm, dtype=np.float64)
        lbn = np.asarray(estimator._lb_norm, dtype=np.float64)
        ubn = np.asarray(estimator._ub_norm, dtype=np.float64)
        rng = np.random.default_rng(3)
        thetas = [x0] + [
            np.clip(x0 + s * (rng.random(x0.shape) - 0.5), lbn, ubn)
            for s in (0.1, 0.3)
        ]
        for i, theta in enumerate(thetas):
            f_slow, g_slow = eval_obj(theta, use_fast=False)
            f_fast, g_fast = eval_obj(theta, use_fast=True)
            rel_val = abs(f_fast - f_slow) / max(1e-12, abs(f_slow))
            self.assertLess(rel_val, 1e-5, f"theta[{i}]: value mismatch {rel_val:.3e}")
            gscale = max(1e-12, float(np.abs(g_slow).max()))
            rel_grad = float(np.abs(g_fast - g_slow).max()) / gscale
            self.assertLess(
                rel_grad, 1e-4, f"theta[{i}]: gradient mismatch {rel_grad:.3e}"
            )

    # ------------------------------------------------------------------
    # 3. Collocation smoke run
    # ------------------------------------------------------------------

    def test_collocation_smoke(self):
        try:
            import casadi  # noqa: F401
        except ImportError:
            self.skipTest("casadi not installed")
        estimator = tb.Estimator(self.simulator)
        parameters, measurements = self._estimation_setup()
        result = estimator.estimate(
            parameters=parameters,
            measurements=measurements,
            start_time=[self.start],
            end_time=[self.end],
            step_size=STEP_SIZE,
            n_warmup=0,
            method=("casadi", "ipopt", "ad", "collocation"),
            options={"maxiter": 10},
        )
        self.assertIn("result_x", result)
        self.assertEqual(len(result["result_x"]), 5)
        self.assertTrue(np.all(np.isfinite(result["result_x"])))


if __name__ == "__main__":
    unittest.main()
