# Standard library imports
import datetime
import unittest

# Third party imports
import numpy as np
import torch
from dateutil import tz

# Local application imports
import twin4build as tb

tb._IS_TESTING = True


def build_model():
    """The optimizer example model (space + heater, bidirectional coupling)
    plus a downstream cost sensor (heater power x electricity price), so the
    test exercises every fast-objective feature: cut-feedback edges, fresh
    loss outputs, an output-cone component that no state depends on, and
    theta-driven captured slots."""
    model = tb.Model(id="test_fast_opt_model")

    building_space = tb.BuildingSpaceThermalSystem(
        C_air=2000000.0,
        C_wall=10000000.0,
        C_boundary=800000.0,
        R_out=0.005,
        R_in=0.005,
        R_boundary=10000,
        f_wall=0,
        f_air=0,
        Q_occ_gain=100.0,
        CO2_occ_gain=0.004,
        CO2_start=400.0,
        infiltrationRate=0.0,
        airVolume=100.0,
        id="BuildingSpace",
    )
    space_heater = tb.SpaceHeaterSystem(
        Q_flow_nominal_sh=2000.0,
        T_a_nominal_sh=60.0,
        T_b_nominal_sh=30.0,
        TAir_nominal_sh=21.0,
        thermalMassHeatCapacity=500000.0,
        nelements=3,
        id="SpaceHeater",
    )

    occupancy = tb.ScheduleSystem(
        weekday_ruleset={"ruleset_default_value": 0}, id="Occupancy"
    )
    outdoor_temp = tb.ScheduleSystem(
        weekday_ruleset={
            "ruleset_default_value": 10.0,
            "ruleset_start_minute": [0, 0],
            "ruleset_end_minute": [0, 0],
            "ruleset_start_hour": [0, 12],
            "ruleset_end_hour": [12, 24],
            "ruleset_value": [5.0, 12.0],
        },
        id="OutdoorTemperature",
    )
    solar = tb.ScheduleSystem(
        weekday_ruleset={"ruleset_default_value": 0.0}, id="Solar"
    )
    supply_flow = tb.ScheduleSystem(
        weekday_ruleset={"ruleset_default_value": 0.0}, id="SupplyFlow"
    )
    exhaust_flow = tb.ScheduleSystem(
        weekday_ruleset={"ruleset_default_value": 0.0}, id="ExhaustFlow"
    )
    supply_air_temp = tb.ScheduleSystem(
        weekday_ruleset={"ruleset_default_value": 20.0}, id="SupplyAirTemp"
    )
    supply_water_temp = tb.ScheduleSystem(
        weekday_ruleset={"ruleset_default_value": 60.0}, id="SupplyWaterTemp"
    )

    mf = 2000.0 / 4180 / (60.0 - 30.0)
    waterflow = tb.ScheduleSystem(
        weekday_ruleset={
            "ruleset_default_value": 0,
            "ruleset_start_minute": [0],
            "ruleset_end_minute": [0],
            "ruleset_start_hour": [8],
            "ruleset_end_hour": [16],
            "ruleset_value": [mf],
        },
        id="Waterflow",
    )

    price = tb.ScheduleSystem(
        weekday_ruleset={
            "ruleset_default_value": 0.5,
            "ruleset_start_minute": [0, 0],
            "ruleset_end_minute": [0, 0],
            "ruleset_start_hour": [6, 17],
            "ruleset_end_hour": [9, 20],
            "ruleset_value": [2.0, 2.5],
        },
        id="Price",
    )
    costs = tb.ScalarProductSystem(scale_factor=2400 / 3600 / 1000, id="Costs")

    model.add_connection(occupancy, building_space, "scheduleValue", "numberOfPeople")
    model.add_connection(
        outdoor_temp, building_space, "scheduleValue", "outdoorTemperature"
    )
    model.add_connection(solar, building_space, "scheduleValue", "globalIrradiation")
    model.add_connection(
        supply_flow, building_space, "scheduleValue", "supplyAirFlowRate"
    )
    model.add_connection(
        exhaust_flow, building_space, "scheduleValue", "exhaustAirFlowRate"
    )
    model.add_connection(
        supply_air_temp, building_space, "scheduleValue", "supplyAirTemperature"
    )
    model.add_connection(
        supply_water_temp, space_heater, "scheduleValue", "supplyWaterTemperature"
    )
    model.add_connection(waterflow, space_heater, "scheduleValue", "waterFlowRate")
    model.add_connection(
        building_space, space_heater, "indoorTemperature", "indoorTemperature"
    )
    model.add_connection(space_heater, building_space, "Power", "heatGain")
    model.add_connection(space_heater, costs, "Power", "input_1")
    model.add_connection(price, costs, "scheduleValue", "input_2")
    model.load()

    heating_setpoint = tb.ScheduleSystem(
        weekday_ruleset={
            "ruleset_default_value": 18.0,
            "ruleset_start_minute": [0],
            "ruleset_end_minute": [0],
            "ruleset_start_hour": [8],
            "ruleset_end_hour": [17],
            "ruleset_value": [21.0],
        },
        id="HeatingSetpoint",
    )
    cooling_setpoint = tb.ScheduleSystem(
        weekday_ruleset={"ruleset_default_value": 26.0}, id="CoolingSetpoint"
    )

    return model, waterflow, space_heater, building_space, costs, (
        heating_setpoint,
        cooling_setpoint,
        mf,
    )


class TestFastControlObjective(unittest.TestCase):
    """Numerical-equivalence regression check for the optimizer's fast
    composed objective (``options={"fast": True}``, the default).

    Like the estimator's ``test_fast_shooting``, this is the tripwire guarding
    the by-construction equivalence contract: value AND gradient parity
    between the composed-map loss and the object-graph loss, at the initial
    iterate and at randomly perturbed decision vectors.
    """

    TOL_VALUE = 1e-8  # relative loss-value tolerance
    TOL_GRAD = 1e-6  # relative gradient tolerance (inf-norm, scaled)

    @classmethod
    def setUpClass(cls):
        (
            model,
            waterflow,
            space_heater,
            building_space,
            costs,
            (heating_setpoint, cooling_setpoint, mf),
        ) = build_model()

        cls.simulator = tb.Simulator(model)
        cls.optimizer = tb.Optimizer(cls.simulator)

        start = datetime.datetime(
            2024, 1, 4, tzinfo=tz.gettz("Europe/Copenhagen")
        )
        end = start + datetime.timedelta(days=1)

        # A couple of SLSQP iterations through the production code path builds
        # the fast objective and populates all solver state.
        cls.optimizer.optimize(
            start_time=start,
            end_time=end,
            step_size=2400,
            variables=[(waterflow, "scheduleValue", 0, mf)],
            objectives=[
                (space_heater, "Power", "min"),
                (costs, "output", "min"),
            ],
            ineq_cons=[
                (building_space, "indoorTemperature", "upper", cooling_setpoint),
                (building_space, "indoorTemperature", "lower", heating_setpoint),
            ],
            method=("scipy", "SLSQP", "ad"),
            options={"maxiter": 2},
        )

    def _bust_caches(self):
        opt = self.optimizer
        opt._theta_obj = 1e6 * torch.ones_like(opt._theta_obj)
        opt._theta_jac = 1e6 * torch.ones_like(opt._theta_jac)

    def _eval(self, theta_np, use_fast):
        """(loss value, gradient) at a decision vector via either path."""
        opt = self.optimizer
        fast = opt._fast_obj
        self.assertIsNotNone(fast, "fast objective was not built")
        opt._fast_obj = fast if use_fast else None
        try:
            self._bust_caches()
            f = float(opt._obj_ad(np.asarray(theta_np, dtype=np.float64)))
            g = np.asarray(
                opt._jac_ad(np.asarray(theta_np, dtype=np.float64))
            )
            return f, g
        finally:
            opt._fast_obj = fast

    def test_fast_loss_matches_object_graph(self):
        opt = self.optimizer
        n = len(opt._theta_obj)

        rng = np.random.default_rng(7)
        thetas = [
            np.full(n, 0.5),
            rng.random(n),
            np.clip(rng.random(n) * 1.2 - 0.1, 0.0, 1.0),  # riding the bounds
        ]

        for i, theta in enumerate(thetas):
            f_slow, g_slow = self._eval(theta, use_fast=False)
            f_fast, g_fast = self._eval(theta, use_fast=True)

            rel_val = abs(f_fast - f_slow) / max(1e-12, abs(f_slow))
            self.assertLess(
                rel_val,
                self.TOL_VALUE,
                f"theta[{i}]: loss value mismatch "
                f"(slow={f_slow:.12g}, fast={f_fast:.12g}, rel={rel_val:.3e})",
            )

            gscale = max(1e-12, float(np.abs(g_slow).max()))
            rel_grad = float(np.abs(g_fast - g_slow).max()) / gscale
            self.assertLess(
                rel_grad,
                self.TOL_GRAD,
                f"theta[{i}]: gradient mismatch (rel inf-norm={rel_grad:.3e})",
            )

    def test_fast_objective_was_built(self):
        """The example model is composable, so the default fast path must not
        have silently fallen back."""
        self.assertIsNotNone(self.optimizer._fast_obj)


if __name__ == "__main__":
    unittest.main()
