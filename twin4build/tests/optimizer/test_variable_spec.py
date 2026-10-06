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
from unittest import mock

import numpy as np
import torch
from dateutil import tz

import twin4build as tb
from twin4build.systems.utils.piecewise_linear_system import PiecewiseLinearSystem
from twin4build.tests.estimator.example_fixture import EXAMPLE_START, STEP_SIZE, example_measurements, load_model
from twin4build.tests.optimizer.test_parameter_decision_variables import X_OUT, build_model
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
        for model_id in (cls.MODEL_ID, cls.MODEL_ID + "_two_curves"):
            path = os.path.join("generated_files", "models", model_id)
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

    def test_private_components_keep_their_own_bounds(self):
        """A private Variable over several components gives each its own
        declared bounds; a shared one refuses bounds that differ."""
        model, curve, _heater, _discomfort = build_model(self.MODEL_ID + "_two_curves")
        other = PiecewiseLinearSystem(id="Curve2", X=X_OUT, Y=torch.tensor([15.0] * 4), Y_bounds=(10.0, 20.0))
        model.add_component(other)
        resolved = estimator_parameters([tb.Variable([curve.Y, other.Y])], model)
        self.assertEqual([(r[0].id, r[1], r[3], r[4], r[5]) for r in resolved], [("Curve", "Y", 12.0, 24.0, "private"), ("Curve2", "Y", 10.0, 20.0, "private")])
        with self.assertRaises(ValueError):
            estimator_parameters([tb.Variable([curve.Y, other.Y], components="shared")], model)
        with self.assertRaises(ValueError):
            estimator_parameters([tb.Variable([curve.Y, curve.Y])], model)

    def test_a_frozen_parameter_resolves(self):
        """The lookup does not depend on a parameter's state: a parameter an
        earlier fit left with requires_grad False is still named."""
        model = load_model()
        controller = model.components["office_temperature_heating_controller"]
        controller.kp.requires_grad_(False)
        (resolved,) = estimator_parameters([tb.Variable(controller.kp, lb=1e-5, ub=1.0)], model)
        self.assertEqual(resolved[:2], (controller, "kp"))

    def test_the_estimator_resolves_before_it_initializes(self):
        """An initialize may replace a parameter object (a multi-branch
        component widens it); a Variable taken before estimate() still names
        it, because the Estimator resolves it to its path first."""
        model = load_model()
        space = model.components["office"]
        handle = space.thermal.C_air
        initialize = type(model).initialize

        def replacing_initialize(self_, *args, **kwargs):
            initialize(self_, *args, **kwargs)
            space.thermal.C_air = tb.Parameter(
                handle.get().detach().clone(), min_value=handle._min_value, max_value=handle._max_value
            )

        est = tb.Estimator(tb.Simulator(model, execution_mode="functional"))
        start = EXAMPLE_START[0]
        with mock.patch.object(type(model), "initialize", replacing_initialize):
            est.estimate(
                parameters=[tb.Variable(handle, x0=2e6, lb=1e6, ub=1e7)],
                measurements=example_measurements(model),
                start_time=[start], end_time=[start + datetime.timedelta(hours=24)], step_size=STEP_SIZE,
                n_warmup=5, method=("scipy", "SLSQP", "ad"), options={"maxiter": 1},
            )
        self.assertIsNot(space.thermal.C_air, handle)  # it was replaced, and the fit ran all the same

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

    def test_the_optimizer_refuses_what_it_cannot_honour(self):
        """The Optimizer starts from the parameter's value (no x0) and a
        trajectory names one port (no sharing); a tuple of Variables is a
        list of them."""
        port = self.model.components["Waterflow"].output["scheduleValue"]
        with self.assertRaises(ValueError):
            optimizer_variables([tb.Variable(self.curve.Y, x0=[18.0] * 4)], self.model)
        with self.assertRaises(ValueError):
            optimizer_variables([tb.Variable(port, lb=0.0, ub=0.02, components="shared")], self.model)
        self.assertEqual(optimizer_variables((tb.Variable(port, lb=0.0, ub=0.02),), self.model), [(self.model.components["Waterflow"], "scheduleValue", 0.0, 0.02)])

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
