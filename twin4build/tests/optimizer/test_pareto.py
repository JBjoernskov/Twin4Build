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
import torch
from dateutil import tz

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


class TestParetoWithFunctionSystem(unittest.TestCase):
    """Energy-vs-discomfort front where f2 is a FunctionSystem residual
    ``relu(setpoint - T_zone)`` -- the realistic comfort formulation (and a
    regression check that FunctionSystem composes into the fast objective
    and the batched prepass)."""

    @classmethod
    def setUpClass(cls):
        model = tb.Model(id="test_pareto_fnsys_model")
        space = tb.BuildingSpaceThermalTorchSystem(
            C_air=2e6, C_wall=1e7, R_out=0.005, R_in=0.005,
            f_wall=0, f_air=0, Q_occ_gain=100.0, CO2_occ_gain=0.004,
            CO2_start=400.0, infiltrationRate=0.0, airVolume=100.0,
            id="BuildingSpace",
        )
        heater = tb.SpaceHeaterTorchSystem(
            Q_flow_nominal_sh=2000.0, T_a_nominal_sh=60.0,
            T_b_nominal_sh=30.0, TAir_nominal_sh=21.0,
            thermalMassHeatCapacity=500000.0, nelements=3,
            id="SpaceHeater",
        )
        zero = tb.ScheduleSystem(
            weekDayRulesetDict={"ruleset_default_value": 0.0}, id="Zero"
        )
        outdoor = tb.ScheduleSystem(
            weekDayRulesetDict={"ruleset_default_value": 5.0}, id="Outdoor"
        )
        supply_air = tb.ScheduleSystem(
            weekDayRulesetDict={"ruleset_default_value": 20.0}, id="SupplyAir"
        )
        supply_water = tb.ScheduleSystem(
            weekDayRulesetDict={"ruleset_default_value": 60.0}, id="SupplyWater"
        )
        cls.mf = heater.Q_flow_nominal_sh / 4180 / (
            heater.T_a_nominal_sh - heater.T_b_nominal_sh
        )
        # NOTE: the baseline trajectory must VARY -- decision-variable ports
        # normalize with their cached history min/max, and a constant
        # baseline makes that degenerate (denormalize collapses theta to the
        # constant, gradients vanish and the solver stalls at x0).
        waterflow = tb.ScheduleSystem(
            weekDayRulesetDict={
                "ruleset_default_value": 0,
                "ruleset_start_minute": [0], "ruleset_end_minute": [0],
                "ruleset_start_hour": [8], "ruleset_end_hour": [16],
                "ruleset_value": [cls.mf],
            },
            id="Waterflow",
        )
        setpoint = tb.ScheduleSystem(
            weekDayRulesetDict={"ruleset_default_value": 21.0}, id="Setpoint"
        )
        discomfort = tb.FunctionSystem(
            inputs=["setpoint", "measured"],
            fn=lambda d: torch.relu(d["setpoint"] - d["measured"]),
            id="Discomfort",
        )

        model.add_connection(zero, space, "scheduleValue", "numberOfPeople")
        model.add_connection(outdoor, space, "scheduleValue", "outdoorTemperature")
        model.add_connection(zero, space, "scheduleValue", "globalIrradiation")
        model.add_connection(zero, space, "scheduleValue", "supplyAirFlowRate")
        model.add_connection(zero, space, "scheduleValue", "exhaustAirFlowRate")
        model.add_connection(supply_air, space, "scheduleValue", "supplyAirTemperature")
        model.add_connection(supply_water, heater, "scheduleValue", "supplyWaterTemperature")
        model.add_connection(waterflow, heater, "scheduleValue", "waterFlowRate")
        model.add_connection(space, heater, "indoorTemperature", "indoorTemperature")
        model.add_connection(heater, space, "Power", "heatGain")
        model.add_connection(setpoint, discomfort, "scheduleValue", "setpoint")
        model.add_connection(space, discomfort, "indoorTemperature", "measured")
        model.load(draw_semantic_model=False, draw_simulation_model=False)

        cls.model = model
        cls.heater = heater
        cls.waterflow = waterflow
        cls.discomfort = discomfort
        cls.start = datetime.datetime(
            2024, 1, 4, tzinfo=tz.gettz("Europe/Copenhagen")
        )
        cls.end = cls.start + datetime.timedelta(hours=24)

    @classmethod
    def tearDownClass(cls):
        path = "generated_files/models/test_pareto_fnsys_model"
        if os.path.exists(path):
            shutil.rmtree(path)

    def test_energy_vs_discomfort_front(self):
        optimizer = tb.Optimizer(tb.Simulator(self.model))
        res = optimizer.pareto_front(
            start_time=self.start,
            end_time=self.end,
            step_size=2400,
            variables=[(self.waterflow, "scheduleValue", 0.0, self.mf)],
            objective1=(self.heater, "Power", "min"),
            objective2=(self.discomfort, "output", "min"),
            n_points=4,
            options={"maxiter": 20},
        )

        # The FunctionSystem output composed into the fast objective.
        self.assertIsNotNone(optimizer._fast_obj)

        # Physical sanity: discomfort is nonnegative, best at the f2 anchor,
        # and buying comfort costs heater power.
        self.assertTrue(np.all(res.f2 >= -1e-9))
        self.assertLessEqual(res.f2[-1], res.f2.min() + 1e-9)
        self.assertGreater(res.f1[-1], res.f1[0])
        self.assertLess(res.f2[-1], res.f2[0])

        # Epsilon feasibility on the swept points.
        ideal2, nadir2 = res.ideal[1], res.nadir[1]
        f2n = (res.f2_min - ideal2) / (nadir2 - ideal2)
        np.testing.assert_array_less(f2n, res.eps + 1e-3)


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
