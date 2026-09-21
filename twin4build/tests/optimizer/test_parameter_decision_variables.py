"""Parameters as decision variables of the Optimizer.

A decision variable naming a component's ``tps.Parameter`` instead of an
output port is optimized as a handful of numbers, not as a trajectory: the
points of an outdoor-temperature compensation curve (a
:class:`PiecewiseLinearSystem` whose ``Y`` is a parameter) that sets the
supply air temperature.  The functional objective carries the parameter
partition of theta into the composed map exactly as the Estimator carries
estimated parameters, and the object-graph re-simulation after the solve
writes the same values back.
"""

import datetime
import os
import shutil
import unittest

import numpy as np
import torch
from dateutil import tz

import twin4build as tb
from twin4build.systems.utils.piecewise_linear_system import PiecewiseLinearSystem

X_OUT = torch.tensor([-5.0, 0.0, 5.0, 10.0])


def build_model(model_id: str):
    """One heated room; the supply air temperature follows a curve of the
    outdoor temperature.  The outdoor schedule steps between 0 and 10 °C so
    two of the four curve points are exercised."""
    model = tb.Model(id=model_id)
    space = tb.BuildingSpaceThermalTorchSystem(
        C_air=2e6, C_wall=1e7, R_out=0.005, R_in=0.005, f_wall=0, f_air=0,
        Q_occ_gain=100.0, CO2_occ_gain=0.004, CO2_start=400.0, infiltrationRate=0.0,
        airVolume=100.0, id="BuildingSpace",
    )
    heater = tb.SpaceHeaterTorchSystem(
        Q_flow_nominal_sh=2000.0, T_a_nominal_sh=60.0, T_b_nominal_sh=30.0,
        TAir_nominal_sh=21.0, thermalMassHeatCapacity=500000.0, nelements=3, id="SpaceHeater",
    )
    zero = tb.ScheduleSystem(weekDayRulesetDict={"ruleset_default_value": 0.0}, id="Zero")
    outdoor = tb.ScheduleSystem(
        weekDayRulesetDict={
            "ruleset_default_value": 0.0,
            "ruleset_start_minute": [0], "ruleset_end_minute": [0],
            "ruleset_start_hour": [8], "ruleset_end_hour": [16],
            "ruleset_value": [10.0],
        },
        id="Outdoor",
    )
    flow = tb.ScheduleSystem(weekDayRulesetDict={"ruleset_default_value": 0.1}, id="AirFlow")
    supply_water = tb.ScheduleSystem(weekDayRulesetDict={"ruleset_default_value": 60.0}, id="SupplyWater")
    mf = heater.Q_flow_nominal_sh / 4180 / (heater.T_a_nominal_sh - heater.T_b_nominal_sh)
    waterflow = tb.ScheduleSystem(weekDayRulesetDict={"ruleset_default_value": 0.5 * mf}, id="Waterflow")
    setpoint = tb.ScheduleSystem(weekDayRulesetDict={"ruleset_default_value": 21.0}, id="Setpoint")
    # Overheating above the setpoint: warmer supply air lowers the heater's
    # power but raises this, so the two objectives conflict (a Pareto front
    # needs that; with agreeing objectives the sweep is just the anchors).
    discomfort = tb.FunctionSystem(
        inputs=["setpoint", "measured"], fn=lambda d: torch.relu(d["measured"] - d["setpoint"]), id="Discomfort"
    )
    curve = PiecewiseLinearSystem(id="Curve", X=X_OUT, Y=torch.tensor([18.0, 18.0, 18.0, 18.0]), Y_bounds=(12.0, 24.0))

    model.add_connection(zero, space, "scheduleValue", "numberOfPeople")
    model.add_connection(outdoor, space, "scheduleValue", "outdoorTemperature")
    model.add_connection(outdoor, curve, "scheduleValue", "x")
    model.add_connection(curve, space, "y", "supplyAirTemperature")
    model.add_connection(zero, space, "scheduleValue", "globalIrradiation")
    model.add_connection(flow, space, "scheduleValue", "supplyAirFlowRate")
    model.add_connection(flow, space, "scheduleValue", "exhaustAirFlowRate")
    model.add_connection(supply_water, heater, "scheduleValue", "supplyWaterTemperature")
    model.add_connection(waterflow, heater, "scheduleValue", "waterFlowRate")
    model.add_connection(space, heater, "indoorTemperature", "indoorTemperature")
    model.add_connection(heater, space, "Power", "heatGain")
    model.add_connection(setpoint, discomfort, "scheduleValue", "setpoint")
    model.add_connection(space, discomfort, "indoorTemperature", "measured")
    model.load(draw_semantic_model=False, draw_simulation_model=False)
    return model, curve, heater, discomfort


class TestParameterDecisionVariables(unittest.TestCase):
    MODEL_ID = "test_parameter_decision_variables"

    @classmethod
    def setUpClass(cls):
        cls.model, cls.curve, cls.heater, cls.discomfort = build_model(cls.MODEL_ID)
        cls.start = datetime.datetime(2024, 1, 4, tzinfo=tz.gettz("Europe/Copenhagen"))
        cls.end = cls.start + datetime.timedelta(hours=24)

    @classmethod
    def tearDownClass(cls):
        path = os.path.join("generated_files", "models", cls.MODEL_ID)
        if os.path.exists(path):
            shutil.rmtree(path)

    def _optimizer(self):
        return tb.Optimizer(tb.Simulator(self.model, execution_mode="functional"))

    def test_curve_points_are_the_decision_vector(self):
        """Only the two curve points the outdoor temperature visits carry a
        gradient; least heater power wants the room warmer, so they climb to
        the upper bound while the unvisited points stay put."""
        optimizer = self._optimizer()
        optimizer.optimize(
            start_time=self.start, end_time=self.end, step_size=2400,
            variables=[(self.curve, "Y", 12.0, 24.0)],
            objectives=[(self.heater, "Power", "min")],
            method=("scipy", "SLSQP", "ad"),
            options={"maxiter": 30},
        )
        y = self.curve.Y.get().detach().cpu().numpy()
        self.assertGreater(y[1], 23.0, y)  # visited at 0 C
        self.assertGreater(y[3], 23.0, y)  # visited at 10 C
        np.testing.assert_allclose(y[[0, 2]], 18.0, atol=1e-6)  # never visited: no gradient

    def test_functional_and_object_objectives_agree_at_a_point(self):
        optimizer = self._optimizer()
        optimizer.optimize(
            start_time=self.start, end_time=self.end, step_size=2400,
            variables=[(self.curve, "Y", 12.0, 24.0)],
            objectives=[(self.heater, "Power", "min"), (self.discomfort, "output", "min")],
            method=("scipy", "SLSQP", "ad"),
            options={"maxiter": 1},
        )
        theta = torch.tensor([0.25, 0.5, 0.75, 1.0], dtype=torch.float64)
        functional = float(optimizer._functional_objective.loss(theta))
        graph = float(optimizer._Optimizer__obj_ad(theta))
        self.assertAlmostEqual(functional, graph, places=6)
        np.testing.assert_allclose(self.curve.Y.get().detach().cpu().numpy(), [15.0, 18.0, 21.0, 24.0], atol=1e-9)

    def test_pareto_front_over_curve_points(self):
        """The Pareto sweep runs on the parameter partition alone: one curve
        per epsilon value, every point inside its bounds."""
        optimizer = self._optimizer()
        res = optimizer.pareto_front(
            start_time=self.start, end_time=self.end, step_size=2400,
            variables=[(self.curve, "Y", 12.0, 24.0)],
            objective1=(self.heater, "Power", "min"),
            objective2=(self.discomfort, "output", "min"),
            n_points=3,
            options={"maxiter": 5},
        )
        self.assertEqual(len(res.f1), 3)
        self.assertEqual(res.theta.shape, (3, 4))
        self.assertTrue(np.all(res.theta >= -1e-9) and np.all(res.theta <= 1.0 + 1e-9))

    def test_parameters_need_the_functional_objective(self):
        optimizer = tb.Optimizer(tb.Simulator(self.model, execution_mode="object"))
        with self.assertRaises(RuntimeError):
            optimizer.optimize(
                start_time=self.start, end_time=self.end, step_size=2400,
                variables=[(self.curve, "Y", 12.0, 24.0)],
                objectives=[(self.heater, "Power", "min")],
                method=("scipy", "SLSQP", "ad"),
                options={"maxiter": 1},
            )


if __name__ == "__main__":
    unittest.main()
