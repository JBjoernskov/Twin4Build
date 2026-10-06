"""``tb.Variable``: one spec for the variables of the Estimator and the Optimizer (#235).

A variable names its model quantity by the object (a ``tb.Parameter`` or an
output port), takes its bounds from the component's declared ones unless
given, and says how it is treated across components and periods.  The
problem classes resolve it to the tuples they always validated, so a list of
``Variable``s defines exactly the problem the hand-written tuples define.
"""

import datetime
import os
import shutil
import unittest

import numpy as np
import torch
from dateutil import tz

import twin4build as tb
from twin4build.tests.estimator.example_fixture import load_model
from twin4build.tests.optimizer.test_parameter_decision_variables import build_model
from twin4build.utils.problem_variables import estimator_parameters, optimizer_variables


class TestVariableSpec(unittest.TestCase):
    MODEL_ID = "test_variable_spec"

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

    # -- the spec ------------------------------------------------------------
    def test_the_spec_is_validated(self):
        self.assertIs(tb.Variable, tb.types.Variable)
        with self.assertRaises(ValueError):
            tb.Variable(self.curve.Y, components="all")
        with self.assertRaises(ValueError):
            tb.Variable(self.curve.Y, periods="monthly")
        with self.assertRaises(ValueError):  # a trajectory needs its bounds
            tb.Variable(self.heater.output["Power"])
        with self.assertRaises(TypeError):
            tb.Variable(1.0)
        self.assertEqual(tb.Variable(self.curve.Y).periods, "shared")
        self.assertEqual(tb.Variable(self.heater.output["Power"], lb=0, ub=1).periods, "per_period")

    # -- the Estimator ---------------------------------------------------------
    def test_estimator_variables_are_the_tuples(self):
        """Variables and bare parameters resolve to the tuples written by
        hand: the declared bounds where none are given, the given ones
        else, lists of components with their sharing."""
        model = load_model()
        c = model.components
        space, controller = c["office"], c["office_temperature_heating_controller"]
        resolved = estimator_parameters(
            [
                tb.Variable(space.thermal.C_air, x0=2e6, lb=1e6, ub=1e7),
                tb.Variable([controller.kp], lb=1e-5, ub=1.0, components="shared"),
                (space, "thermal.R_out", 0.01, 1e-3, 0.1),  # a tuple passes through
            ],
            model,
        )
        self.assertEqual(resolved[0], (space, "thermal.C_air", 2e6, 1e6, 1e7, "private"))
        self.assertEqual(resolved[1], (controller, "kp", None, 1e-5, 1.0, "shared"))
        self.assertEqual(resolved[2], (space, "thermal.R_out", 0.01, 1e-3, 0.1))
        # a bare parameter: the component's declared bounds, as parameters="auto"
        declared = {attr: (lb, ub) for _c, attr, _x0, lb, ub in space.get_estimable_parameters()}
        (bare,) = estimator_parameters([space.thermal.C_wall], model)
        self.assertEqual(bare[:2], (space, "thermal.C_wall"))
        self.assertEqual((bare[3], bare[4]), declared["thermal.C_wall"])
        with self.assertRaises(NotImplementedError):
            estimator_parameters([tb.Variable(space.thermal.C_air, periods="per_period")], model)

    def test_a_parameter_of_no_component_is_refused(self):
        stray = tb.Parameter(torch.tensor([1.0], dtype=torch.float64), min_value=0.0, max_value=2.0)
        with self.assertRaises(ValueError):
            estimator_parameters([stray], load_model())

    # -- the Optimizer ---------------------------------------------------------
    def test_optimizer_variables_are_the_tuples(self):
        resolved = optimizer_variables(
            [self.curve.Y, tb.Variable(self.model.components["Waterflow"].output["scheduleValue"], lb=0.0, ub=0.02)],
            self.model,
        )
        self.assertEqual(resolved[0], (self.curve, "Y", 12.0, 24.0))  # the curve's declared Y_bounds
        self.assertEqual(resolved[1], (self.model.components["Waterflow"], "scheduleValue", 0.0, 0.02))
        with self.assertRaises(NotImplementedError):
            optimizer_variables([tb.Variable(self.curve.Y, periods="per_period")], self.model)

    def test_a_variable_defines_the_tuples_problem(self):
        """The curve's points as a bare parameter define the same problem as
        the tuple: the same functional objective value at a point."""
        values = []
        for variables in ([(self.curve, "Y", 12.0, 24.0)], [self.curve.Y]):
            optimizer = tb.Optimizer(tb.Simulator(self.model, execution_mode="functional"))
            optimizer.optimize(
                start_time=self.start, end_time=self.end, step_size=2400,
                variables=variables,
                objectives=[(self.heater, "Power", "min"), (self.discomfort, "output", "min")],
                method=("scipy", "SLSQP", "ad"),
                options={"maxiter": 1},
            )
            theta = torch.tensor([0.25, 0.5, 0.75, 1.0], dtype=torch.float64)
            values.append(float(optimizer._functional_objective.loss(theta)))
        np.testing.assert_allclose(values[1], values[0], rtol=1e-12)


if __name__ == "__main__":
    unittest.main()
