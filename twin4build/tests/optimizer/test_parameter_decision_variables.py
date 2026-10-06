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
        for model_id in (cls.MODEL_ID, cls.MODEL_ID + "_gpu"):
            path = os.path.join("generated_files", "models", model_id)
            if os.path.exists(path):
                shutil.rmtree(path)

    def _optimizer(self):
        return tb.Optimizer(tb.Simulator(self.model, execution_mode="functional"))

    def _functional_objective(self, optimizer, curve, heater, discomfort):
        """The functional objective of ``optimizer`` over the two objectives
        (one SLSQP iteration builds it)."""
        optimizer.optimize(
            start_time=self.start, end_time=self.end, step_size=2400,
            variables=[(curve, "Y", 12.0, 24.0)],
            objectives=[(heater, "Power", "min"), (discomfort, "output", "min")],
            method=("scipy", "SLSQP", "ad"),
            options={"maxiter": 1},
        )
        return optimizer._functional_objective

    def _assert_rows(self, objective, batch, theta, gradient, transform_mode):
        """Every row of ``batch`` (and of its gradient) is the row's own
        :meth:`parts` (and gradient)."""
        for i in range(theta.shape[0]):
            z = theta[i].detach().clone().requires_grad_(True)
            row = objective.parts(z, transform_mode=transform_mode)
            for wide, own in zip(batch.objs + batch.phys, row.objs + row.phys):
                # relative: the means span 1e-2 (overheating) to 1e3 W, and a batched rollout sums in another order
                np.testing.assert_allclose(float(wide[i]), float(own), rtol=1e-8, atol=1e-12)
            (own_gradient,) = torch.autograd.grad(sum(row.objs), z)
            np.testing.assert_allclose(
                gradient[i].cpu().numpy(), own_gradient.cpu().numpy(), rtol=1e-8, atol=1e-12
            )

    THETA = [[0.25, 0.5, 0.75, 1.0], [0.6, 0.4, 0.2, 0.1], [1.0, 1.0, 0.0, 0.0]]

    def test_batched_parts_are_the_rows_parts(self):
        """One wide rollout for every row, the post-processing vmapped (the
        Estimator's batched structure), gives each row's parts and gradient;
        ``batched_parts`` on the CPU (row by row) gives the same values."""
        objective = self._functional_objective(self._optimizer(), self.curve, self.heater, self.discomfort)
        theta = torch.tensor(self.THETA, dtype=torch.float64)
        z = theta.clone().requires_grad_(True)
        wide = objective._batched_parts_wide(z)
        (gradient,) = torch.autograd.grad(sum(wide.objs).sum(), z)
        self._assert_rows(objective, wide, theta, gradient, transform_mode=False)
        rows = objective.batched_parts(theta)
        np.testing.assert_allclose(
            torch.stack(rows.objs).detach().numpy(), torch.stack(wide.objs).detach().numpy(), rtol=1e-9
        )

    def _periods(self):
        """Three periods of 24 h starting at 00, 08 and 16 h on consecutive
        days: each meets the outdoor step at another time of its day."""
        starts = [self.start + datetime.timedelta(days=d, hours=8 * d) for d in range(3)]
        return starts, [s + datetime.timedelta(hours=24) for s in starts]

    def _periods_objective(self, n_warmup=0):
        starts, ends = self._periods()
        optimizer = self._optimizer()
        optimizer.optimize(
            start_time=starts, end_time=ends, step_size=2400,
            variables=[(self.curve, "Y", 12.0, 24.0)],
            objectives=[(self.heater, "Power", "min"), (self.discomfort, "output", "min")],
            method=("scipy", "SLSQP", "ad"),
            options={"maxiter": 1},
            n_warmup=n_warmup,
        )
        return optimizer._functional_objective

    def test_periods_roll_out_side_by_side(self):
        """Equal-length periods roll out side by side (the Estimator's
        windows): the same parts and gradients as one period after another,
        for one row and for a batch of rows."""
        objective = self._periods_objective()
        self.assertTrue(objective.parallel_periods)
        theta = torch.tensor(self.THETA, dtype=torch.float64)
        results = {}
        for parallel in (True, False):
            objective.parallel_periods = parallel
            z = theta.clone().requires_grad_(True)
            batch = objective._batched_parts_wide(z)
            (gradient,) = torch.autograd.grad(sum(batch.objs).sum(), z)
            row = objective.parts(theta[0])
            results[parallel] = (
                torch.stack(batch.objs + batch.phys).detach(), gradient, torch.stack(row.objs + row.phys).detach()
            )
        for side, sequence in zip(results[True], results[False]):
            np.testing.assert_allclose(side.numpy(), sequence.numpy(), rtol=1e-8, atol=1e-12)
        # the periods do differ: they meet the outdoor step at other times of their day
        objective.parallel_periods = True
        out = objective._rollout(theta[0])
        j = objective._obj_terms[0][0]
        self.assertFalse(torch.allclose(out[0][j], out[1][j]))

    def test_warmup_steps_leave_the_objectives(self):
        """The first ``n_warmup`` steps of every period are simulated but not
        part of the objectives."""
        objective = self._periods_objective(n_warmup=6)
        theta = torch.tensor(self.THETA[0], dtype=torch.float64)
        out = objective._rollout(theta)
        parts = objective.parts(theta)
        for k, (j, _kind) in enumerate(objective._obj_terms):
            kept = torch.cat([out[p][j][6:].reshape(-1) for p in range(3)]).mean()
            np.testing.assert_allclose(float(parts.phys[k]), float(kept), rtol=1e-12)
        heater = objective._obj_terms[0][0]
        everything = torch.cat([out[p][heater].reshape(-1) for p in range(3)]).mean()
        self.assertGreater(abs(float(parts.phys[0]) - float(everything)), 1e-6 * abs(float(everything)))

    def test_warmup_needs_the_functional_objective(self):
        starts, ends = self._periods()
        optimizer = tb.Optimizer(tb.Simulator(self.model, execution_mode="object"))
        waterflow = self.model.components["Waterflow"]
        with self.assertRaises(RuntimeError):
            optimizer.optimize(
                start_time=starts, end_time=ends, step_size=2400,
                variables=[(waterflow, "scheduleValue", 0.0, 0.02)],
                objectives=[(self.heater, "Power", "min")],
                method=("scipy", "SLSQP", "ad"),
                options={"maxiter": 1},
                n_warmup=6,
            )

    @unittest.skipUnless(torch.cuda.is_available(), "the captured, compiled step needs CUDA")
    def test_batched_parts_on_the_captured_compiled_step(self):
        """On the GPU with per-step CUDA graphs of the compiled step: the
        batch runs as one wide rollout and every row matches its own
        transform-mode :meth:`parts`."""
        model, curve, heater, discomfort = build_model(self.MODEL_ID + "_gpu")
        model.to(device="cuda", dtype=torch.float64)
        simulator = tb.Simulator(
            model, execution_mode="functional", execution_backend="cuda_graph",
            compile_step=True, cuda_graph_scope="step",
        )
        objective = self._functional_objective(tb.Optimizer(simulator), curve, heater, discomfort)
        self.assertTrue(simulator.step_graph_active(torch.device("cuda")))
        theta = torch.tensor(self.THETA, dtype=torch.float64, device="cuda")
        z = theta.clone().requires_grad_(True)
        batch = objective.batched_parts(z)
        (gradient,) = torch.autograd.grad(sum(batch.objs).sum(), z)
        self._assert_rows(objective, batch, theta, gradient, transform_mode=True)

    @unittest.skipUnless(torch.cuda.is_available(), "the captured, compiled step needs CUDA")
    def test_periods_side_by_side_on_the_captured_compiled_step(self):
        """On the GPU with per-step CUDA graphs: periods side by side, for a
        batch of rows, give each row its own sequential parts and gradient."""
        model, curve, heater, discomfort = build_model(self.MODEL_ID + "_gpu")
        model.to(device="cuda", dtype=torch.float64)
        simulator = tb.Simulator(
            model, execution_mode="functional", execution_backend="cuda_graph",
            compile_step=True, cuda_graph_scope="step",
        )
        starts, ends = self._periods()
        optimizer = tb.Optimizer(simulator)
        optimizer.optimize(
            start_time=starts, end_time=ends, step_size=2400,
            variables=[(curve, "Y", 12.0, 24.0)],
            objectives=[(heater, "Power", "min"), (discomfort, "output", "min")],
            method=("scipy", "SLSQP", "ad"),
            options={"maxiter": 1},
        )
        objective = optimizer._functional_objective
        self.assertTrue(objective.parallel_periods and simulator.step_graph_active(torch.device("cuda")))
        theta = torch.tensor(self.THETA, dtype=torch.float64, device="cuda")
        z = theta.clone().requires_grad_(True)
        batch = objective.batched_parts(z)
        (gradient,) = torch.autograd.grad(sum(batch.objs).sum(), z)
        objective.parallel_periods = False  # the reference: one period after another, row by row
        self._assert_rows(objective, batch, theta, gradient, transform_mode=True)

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
