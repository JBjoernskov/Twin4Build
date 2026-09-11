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
from types import SimpleNamespace

# Third party imports
import numpy as np
import torch
from dateutil import tz

try:
    import casadi
except ImportError:  # pragma: no cover - depends on optional installation
    casadi = None

# Local application imports
import twin4build as tb
from twin4build.optimizer._pareto import _EpsSubproblem, _pareto_mask
from twin4build.optimizer._pareto_collocation import ParetoCollocation
from twin4build.solvers.ipopt import solve_ipopt_constrained
from benchmarks.common import batch_model

tb._IS_TESTING = True

START = datetime.datetime(2024, 1, 4, tzinfo=tz.gettz("Europe/Copenhagen"))
N_HOURS = 24
OPT_STEP = 3600


def build_chain_model(n_zones, model_id):
    """Build the compact thermal fixture used by Pareto tests."""
    model = tb.Model(id=model_id)
    zones = [
        tb.BuildingSpaceThermalTorchSystem(
            C_air=1e6,
            C_wall=5e6,
            R_out=0.01,
            R_in=0.01,
            f_wall=0.0,
            f_air=0.0,
            Q_occ_gain=100.0,
            id=f"Zone{i}",
        )
        for i in range(n_zones)
    ]
    outdoor = tb.ScheduleSystem(
        weekDayRulesetDict={"ruleset_default_value": 5.0}, id="Outdoor"
    )
    zero = tb.ScheduleSystem(
        weekDayRulesetDict={"ruleset_default_value": 0.0}, id="Zero"
    )
    supply_air = tb.ScheduleSystem(
        weekDayRulesetDict={"ruleset_default_value": 20.0}, id="SupplyAirTemp"
    )
    heaters = [
        tb.ScheduleSystem(
            weekDayRulesetDict={
                "ruleset_default_value": 0.0,
                "ruleset_start_minute": [0],
                "ruleset_end_minute": [0],
                "ruleset_start_hour": [6],
                "ruleset_end_hour": [20],
                "ruleset_value": [1500.0],
            },
            id=f"Heater{i}",
        )
        for i in range(n_zones)
    ]
    for zone, heater in zip(zones, heaters):
        model.add_connection(outdoor, zone, "scheduleValue", "outdoorTemperature")
        model.add_connection(zero, zone, "scheduleValue", "supplyAirFlowRate")
        model.add_connection(zero, zone, "scheduleValue", "exhaustAirFlowRate")
        model.add_connection(supply_air, zone, "scheduleValue", "supplyAirTemperature")
        model.add_connection(zero, zone, "scheduleValue", "globalIrradiation")
        model.add_connection(zero, zone, "scheduleValue", "numberOfPeople")
        model.add_connection(heater, zone, "scheduleValue", "heatGain")
    model.load(draw_semantic_model=False, draw_simulation_model=False)
    return model, zones, [], [], heaters


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


class TestParetoDerivativeBundle(unittest.TestCase):
    """Exact first- and second-order derivatives include every Pareto term."""

    @staticmethod
    def _subproblem():
        class Fast:
            @staticmethod
            def parts(x, *, transform_mode=False):
                f1 = (x[0] - 1.0).square() + 0.3 * x[0] * x[1]
                f2 = x[0].square() * x[1] + torch.sin(x[1])
                soft = 0.2 * (x[0] + x[1]).pow(4)
                return SimpleNamespace(
                    objs=[f1, f2],
                    phys=[f1 + 2.0, f2 - 3.0],
                    eq=[soft],
                    ineq=None,
                )

        opt = SimpleNamespace(
            _functional_objective=Fast(),
            _objectives=[None, None],
            _device=torch.device("cpu"),
            simulator=SimpleNamespace(execution_backend="eager"),
        )
        sub = _EpsSubproblem(opt, delta=0.07)
        sub.set_normalization(-0.4, 1.6)
        return sub

    def test_bundle_cache_and_exact_gradients(self):
        sub = self._subproblem()
        x = np.array([0.35, -0.2])
        first = sub.values(x)
        second = sub.values(x.copy())
        self.assertIs(first, second)
        self.assertEqual(sub.stats["bundle_calls"], 1)

        xt = torch.tensor(x, dtype=torch.float64)
        expected = torch.func.jacrev(lambda z: sub._value_vector(z)[:2])(xt)
        np.testing.assert_allclose(
            np.stack((first["gf"], first["gc"])),
            expected.numpy(),
            rtol=1e-10,
            atol=1e-11,
        )

    def test_exact_lagrangian_hessian_includes_multiplier(self):
        sub = self._subproblem()
        x = np.array([0.35, -0.2])
        sigma, lam = 0.8, -0.45
        got_vals = sub.hessian(x, sigma, np.array([lam]))
        got = np.zeros((2, 2))
        iu = np.triu_indices(2)
        got[iu] = got_vals
        got[(iu[1], iu[0])] = got_vals

        def lag(z):
            values = sub._value_vector(z)
            return sigma * values[0] + lam * values[1]

        expected = torch.func.hessian(lag)(torch.tensor(x, dtype=torch.float64)).numpy()
        np.testing.assert_allclose(got, expected, rtol=1e-9, atol=1e-10)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_direct_capture_replays_changed_inputs(self):
        sub = self._subproblem()
        sub.opt._device = torch.device("cuda")
        sub.capture_derivatives = True
        sub.capture_hessian = True
        sub.stats["bundle_enabled"] = True
        sub.stats["hessian_enabled"] = True
        x1 = np.array([0.35, -0.2])
        x2 = np.array([0.41, -0.13])
        v1 = sub.values(x1)
        v2 = sub.values(x2)
        self.assertTrue(sub.stats["bundle_captured"])
        self.assertGreaterEqual(sub.stats["bundle_replays"], 1)
        self.assertNotAlmostEqual(v1["f"], v2["f"])

        h1 = sub.hessian(x1, 0.8, np.array([-0.45]))
        h2 = sub.hessian(x2, 0.6, np.array([0.25]))
        self.assertTrue(sub.stats["hessian_captured"])
        self.assertGreaterEqual(sub.stats["hessian_replays"], 1)
        self.assertFalse(np.allclose(h1, h2))


class TestIpoptConstraintBounds(unittest.TestCase):
    def test_generic_upper_bound_and_equality_default(self):
        if casadi is None:
            self.skipTest("CasADi is optional")

        common = dict(
            x0=np.array([0.0]),
            lb=np.array([-5.0]),
            ub=np.array([5.0]),
            fun=lambda x: float((x[0] - 2.0) ** 2),
            grad=lambda x: np.array([2.0 * (x[0] - 2.0)]),
            n_g=1,
            g_fun=lambda x: np.array([x[0]]),
            g_jac_vals=lambda x: np.array([1.0]),
            jac_rows=np.array([0]),
            jac_cols=np.array([0]),
            options={"maxiter": 30, "tol": 1e-9},
        )
        bounded = solve_ipopt_constrained(
            **common, lbg=np.array([-np.inf]), ubg=np.array([0.5])
        )
        self.assertLessEqual(bounded.x[0], 0.5 + 1e-6)
        self.assertAlmostEqual(bounded.x[0], 0.5, places=5)

        equality = solve_ipopt_constrained(**common)
        self.assertAlmostEqual(equality.x[0], 0.0, places=6)


class TestParetoFront(unittest.TestCase):
    """End-to-end sweeps on the 1-zone chain model."""

    N_POINTS = 5
    MAXITER = 30

    @classmethod
    def setUpClass(cls):
        cls.model, cls.zones, _, _, cls.heaters = build_chain_model(
            1, "test_pareto_model"
        )
        cls.simulator = tb.Simulator(cls.model, execution_mode="functional")
        cls.end = START + datetime.timedelta(hours=N_HOURS)

    @classmethod
    def tearDownClass(cls):
        path = "generated_files/models/test_pareto_model"
        if os.path.exists(path):
            shutil.rmtree(path)

    def _sweep(
        self,
        batched_prepass: bool,
        method=("scipy", "SLSQP", "ad"),
        n_points=None,
    ):
        optimizer = tb.Optimizer(self.simulator)
        return optimizer.pareto_front(
            start_time=START,
            end_time=self.end,
            step_size=OPT_STEP,
            variables=[(self.heaters[0], "scheduleValue", 0.0, 3000.0)],
            objective1=(self.heaters[0], "scheduleValue", "min"),
            objective2=(self.zones[0], "indoorTemperature", "max"),
            n_points=n_points or self.N_POINTS,
            method=method,
            batched_prepass=batched_prepass,
            options={"maxiter": self.MAXITER},
        )

    def _collocation_callbacks(self, *, soft_comfort=False):
        optimizer = tb.Optimizer(self.simulator)
        optimizer._variables = [(self.heaters[0], "scheduleValue", 0.0, 3000.0)]
        optimizer._objectives = [
            (self.heaters[0], "scheduleValue", "min"),
            (self.zones[0], "indoorTemperature", "max"),
        ]
        optimizer._eq_cons = []
        optimizer._ineq_cons = (
            [(self.zones[0], "indoorTemperature", "lower", 100.0)]
            if soft_comfort
            else []
        )
        optimizer._start_time = [START + datetime.timedelta(hours=5)]
        optimizer._end_time = [START + datetime.timedelta(hours=8)]
        optimizer._stepSize = [OPT_STEP]
        (
            optimizer._second_time_steps,
            optimizer._date_time_steps,
            optimizer._max_timesteps,
            optimizer._n_timesteps,
        ) = tb.Simulator.get_simulation_timesteps(
            optimizer._start_time, optimizer._end_time, optimizer._stepSize
        )
        optimizer._timestep_mask = torch.ones(
            optimizer._max_timesteps, 1, dtype=torch.bool
        )
        optimizer._max_values = {}
        controls, bounds = optimizer._prepare_scipy_problem(
            ("casadi", "ipopt", "ad"), {}
        )
        problem = ParetoCollocation(
            optimizer,
            controls,
            bounds,
            delta=0.07,
            options={"hessian": "exact"},
        )
        problem.configure(0, 1, 0.07, -0.2, 0.8)
        return problem

    def test_collocation_defects_and_sparse_jacobian_match_fd(self):
        problem = self._collocation_callbacks()
        z = problem.z0.copy()
        z[1] += 0.03
        values = problem.g(z)
        dense = np.zeros((len(values), len(z)))
        dense[problem.jac_rows, problem.jac_cols] = problem.jac(z)
        eps = 1e-6
        fd = np.column_stack(
            [
                (
                    problem.g(z + eps * np.eye(len(z))[j])
                    - problem.g(z - eps * np.eye(len(z))[j])
                )
                / (2 * eps)
                for j in range(len(z))
            ]
        )
        np.testing.assert_allclose(dense, fd, rtol=2e-4, atol=2e-5)

    def test_exact_sparse_lagrangian_hessian_matches_gradient_fd(self):
        problem = self._collocation_callbacks(soft_comfort=True)
        z = problem.z0.copy()
        z[1] += 0.02
        sigma = 0.73
        lam = np.linspace(-0.3, 0.4, problem.n_links * problem.Da + 1)
        values = problem.hessian(z, sigma, lam)
        dense = np.zeros((len(z), len(z)))
        dense[problem.hess_rows, problem.hess_cols] = values
        dense = dense + np.triu(dense, 1).T

        def lag_grad(x):
            jac = np.zeros((len(lam), len(x)))
            jac[problem.jac_rows, problem.jac_cols] = problem.jac(x)
            return sigma * problem.grad(x) + jac.T @ lam

        eps = 2e-6
        fd = np.column_stack(
            [
                (
                    lag_grad(z + eps * np.eye(len(z))[j])
                    - lag_grad(z - eps * np.eye(len(z))[j])
                )
                / (2 * eps)
                for j in range(len(z))
            ]
        )
        np.testing.assert_allclose(dense, fd, rtol=3e-3, atol=2e-4)

    def test_front_properties(self):
        res = self._sweep(batched_prepass=True)

        n = len(res.eps)
        self.assertEqual(n, self.N_POINTS)
        for arr in (
            res.f1,
            res.f2,
            res.f1_min,
            res.f2_min,
            res.slope,
            res.success,
            res.nit,
            res.pareto_mask,
        ):
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

    def test_ipopt_front_feasibility_and_slsqp_parity(self):
        if casadi is None:
            self.skipTest("CasADi is optional")
        slsqp = self._sweep(False, n_points=3)
        ipopt = self._sweep(
            False, method=("casadi", "ipopt", "ad", "collocation"), n_points=3
        )
        ipopt_f2n = (ipopt.f2_min - ipopt.ideal[1]) / (ipopt.nadir[1] - ipopt.ideal[1])
        np.testing.assert_array_less(ipopt_f2n, ipopt.eps + 2e-3)
        np.testing.assert_allclose(ipopt.f1_min, slsqp.f1_min, rtol=5e-2, atol=5e-2)
        self.assertEqual(ipopt.hessian["kind"], "exact")
        self.assertEqual(
            ipopt.callback_shapes["constraints"],
            (N_HOURS - 1) * ipopt.callback_shapes["state_width"] + 1,
        )
        self.assertLess(float(np.max(ipopt.max_defect)), 1e-6)
        self.assertTrue(
            all(max(row["objective_abs"]) < 2e-3 for row in ipopt.rollout_parity)
        )

    def test_three_tuple_ipopt_is_rejected_as_ambiguous(self):
        with self.assertRaisesRegex(ValueError, "collocation"):
            self._sweep(False, method=("casadi", "ipopt", "ad"), n_points=2)

    def test_apply_writes_solution(self):
        res = self._sweep(batched_prepass=True)
        res.apply(len(res.eps) - 1)  # the f2 anchor: heater at full power
        applied = (
            self.heaters[0]
            .output["scheduleValue"]
            .history(i_s=0)
            .detach()
            .cpu()
            .numpy()
        )
        self.assertGreater(applied.mean(), 2000.0)

    def test_object_graph_fallback(self):
        optimizer = tb.Optimizer(tb.Simulator(self.model))
        res = optimizer.pareto_front(
            start_time=START,
            end_time=self.end,
            step_size=OPT_STEP,
            variables=[(self.heaters[0], "scheduleValue", 0.0, 3000.0)],
            objective1=(self.heaters[0], "scheduleValue", "min"),
            objective2=(self.zones[0], "indoorTemperature", "max"),
            n_points=2,
            batched_prepass=False,
            options={"maxiter": 3},
        )
        self.assertIsNone(optimizer._functional_objective)
        self.assertEqual(len(res.eps), 2)

    def test_removed_capture_option_is_rejected(self):
        optimizer = tb.Optimizer(tb.Simulator(self.model))
        with self.assertRaisesRegex(TypeError, "Removed Pareto option"):
            optimizer.pareto_front(
                start_time=START,
                end_time=self.end,
                step_size=OPT_STEP,
                variables=[(self.heaters[0], "scheduleValue", 0.0, 3000.0)],
                objective1=(self.heaters[0], "scheduleValue", "min"),
                objective2=(self.zones[0], "indoorTemperature", "max"),
                n_points=2,
                batched_prepass=False,
                options={"maxiter": 1, "capture_derivatives": True},
            )

    def test_pareto_rejects_obsolete_fast_option(self):
        optimizer = tb.Optimizer(self.simulator)
        with self.assertRaisesRegex(TypeError, "execution_mode='functional'"):
            optimizer.pareto_front(
                start_time=START,
                end_time=self.end,
                step_size=OPT_STEP,
                variables=[(self.heaters[0], "scheduleValue", 0.0, 3000.0)],
                objective1=(self.heaters[0], "scheduleValue", "min"),
                objective2=(self.zones[0], "indoorTemperature", "max"),
                n_points=2,
                options={"fast": True},
            )


class TestParetoWithFunctionSystem(unittest.TestCase):
    """Energy-vs-discomfort front where f2 is a FunctionSystem residual
    ``relu(setpoint - T_zone)`` -- the realistic comfort formulation (and a
    regression check that FunctionSystem composes into the fast objective
    and the batched prepass)."""

    @classmethod
    def setUpClass(cls):
        model = tb.Model(id="test_pareto_fnsys_model")
        space = tb.BuildingSpaceThermalTorchSystem(
            C_air=2e6,
            C_wall=1e7,
            R_out=0.005,
            R_in=0.005,
            f_wall=0,
            f_air=0,
            Q_occ_gain=100.0,
            CO2_occ_gain=0.004,
            CO2_start=400.0,
            infiltrationRate=0.0,
            airVolume=100.0,
            id="BuildingSpace",
        )
        heater = tb.SpaceHeaterTorchSystem(
            Q_flow_nominal_sh=2000.0,
            T_a_nominal_sh=60.0,
            T_b_nominal_sh=30.0,
            TAir_nominal_sh=21.0,
            thermalMassHeatCapacity=500000.0,
            nelements=3,
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
        cls.mf = (
            heater.Q_flow_nominal_sh
            / 4180
            / (heater.T_a_nominal_sh - heater.T_b_nominal_sh)
        )
        # NOTE: the baseline trajectory must VARY -- decision-variable ports
        # normalize with their cached history min/max, and a constant
        # baseline makes that degenerate (denormalize collapses theta to the
        # constant, gradients vanish and the solver stalls at x0).
        waterflow = tb.ScheduleSystem(
            weekDayRulesetDict={
                "ruleset_default_value": 0,
                "ruleset_start_minute": [0],
                "ruleset_end_minute": [0],
                "ruleset_start_hour": [8],
                "ruleset_end_hour": [16],
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
        model.add_connection(
            supply_water, heater, "scheduleValue", "supplyWaterTemperature"
        )
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
        cls.start = datetime.datetime(2024, 1, 4, tzinfo=tz.gettz("Europe/Copenhagen"))
        cls.end = cls.start + datetime.timedelta(hours=24)

    @classmethod
    def tearDownClass(cls):
        path = "generated_files/models/test_pareto_fnsys_model"
        if os.path.exists(path):
            shutil.rmtree(path)

    def test_energy_vs_discomfort_front(self):
        optimizer = tb.Optimizer(tb.Simulator(self.model, execution_mode="functional"))
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
        self.assertIsNotNone(optimizer._functional_objective)

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

        # Regression (weak anchor): relu discomfort is flat at 0, so the RAW
        # f2 anchor stops at an arbitrary over-heated profile (its f1 is the
        # nadir estimate).  The eps=0 endpoint must be solved as a sweep
        # subproblem instead, recovering a clearly cheaper f2-optimal point.
        f1_range = res.nadir[0] - res.ideal[0]
        self.assertGreater(
            (res.nadir[0] - res.f1_min[-1]) / f1_range,
            0.05,
            "eps=0 endpoint is no cheaper than the raw f2 anchor -- "
            "weakly Pareto-optimal endpoint is back",
        )


class TestBatchedParetoCollocation(unittest.TestCase):
    """The one-step sparse callbacks preserve a same-class ``n_c=2`` batch."""

    def test_batched_defects_jacobian_and_hessian(self):
        self._check_batched_defects_jacobian_and_hessian(2)

    def test_three_replica_defects_jacobian_and_hessian(self):
        self._check_batched_defects_jacobian_and_hessian(3)

    def _check_batched_defects_jacobian_and_hessian(self, n_c):

        source, zones, _, _, heaters = build_chain_model(
            n_c, f"test_pareto_batched_nc{n_c}"
        )
        batched, _ = batch_model(source, measure=False)
        zone_meta, _ = source.get_batched_component_info(zones[0].id)
        zone = batched.components[zone_meta.id]
        batched_heaters = [
            batched.components[source.get_batched_component_info(h.id)[0].id]
            for h in heaters
        ]
        optimizer = tb.Optimizer(tb.Simulator(batched, execution_mode="functional"))
        optimizer._variables = [
            (heater, "scheduleValue", 0.0, 3000.0) for heater in batched_heaters
        ]
        optimizer._objectives = [
            (batched_heaters[0], "scheduleValue", "min"),
            (zone, "indoorTemperature", "max"),
        ]
        optimizer._eq_cons = []
        optimizer._ineq_cons = [(zone, "indoorTemperature", "upper", 0.0)]
        optimizer._start_time = [START]
        optimizer._end_time = [START + datetime.timedelta(hours=3)]
        optimizer._stepSize = [OPT_STEP]
        (
            optimizer._second_time_steps,
            optimizer._date_time_steps,
            optimizer._max_timesteps,
            optimizer._n_timesteps,
        ) = tb.Simulator.get_simulation_timesteps(
            optimizer._start_time, optimizer._end_time, optimizer._stepSize
        )
        optimizer._timestep_mask = torch.ones(
            optimizer._max_timesteps, 1, dtype=torch.bool
        )
        optimizer._max_values = {}
        controls, bounds = optimizer._prepare_scipy_problem(
            ("casadi", "ipopt", "ad"), {}
        )
        problem = ParetoCollocation(
            optimizer,
            controls,
            bounds,
            delta=0.04,
            options={"hessian": "exact"},
        )
        problem.configure(0, 1, 0.04, -0.2, 0.8)
        self.assertIn((n_c, 2), problem.fast.layout.shapes)
        self.assertEqual(problem.n_replicas, n_c)
        dense_jac_nnz = (
            problem.n_links * problem.Da * (problem.n_vars + problem.Da + 1)
            + problem.n_z
        )
        dense_hess_nnz = problem.n_seg * (
            problem.n_vars * (problem.n_vars + 1) // 2
            + problem.n_vars * problem.Da
            + problem.Da * (problem.Da + 1) // 2
        )
        self.assertLess(len(problem.jac_rows), dense_jac_nnz)
        self.assertLess(len(problem.hess_rows), dense_hess_nnz)
        self.assertLess(np.max(np.abs(problem.g(problem.z0)[:-1])), 1e-10)

        z = problem.z0.copy()
        z[1] += 0.02
        eps = 1e-6
        eye = np.eye(problem.n_z)
        jac = np.zeros((problem.n_links * problem.Da + 1, problem.n_z))
        jac[problem.jac_rows, problem.jac_cols] = problem.jac(z)
        jac_fd = np.column_stack(
            [
                (problem.g(z + eps * eye[j]) - problem.g(z - eps * eye[j])) / (2 * eps)
                for j in range(problem.n_z)
            ]
        )
        np.testing.assert_allclose(jac, jac_fd, rtol=3e-4, atol=3e-5)
        zt = torch.tensor(z, dtype=torch.float64)
        jac_ad = torch.func.jacrev(
            lambda value: torch.cat(
                [
                    problem._values_tensor(value)[2].reshape(-1),
                    problem._values_tensor(value)[1].reshape(1),
                ]
            )
        )(zt)
        np.testing.assert_allclose(jac, jac_ad.numpy(), rtol=1e-9, atol=1e-10)

        sigma = 0.77
        lam = np.linspace(-0.3, 0.4, jac.shape[0])
        hess = np.zeros((problem.n_z, problem.n_z))
        hess[problem.hess_rows, problem.hess_cols] = problem.hessian(z, sigma, lam)
        hess += np.triu(hess, 1).T

        def lag_grad(x):
            j = np.zeros_like(jac)
            j[problem.jac_rows, problem.jac_cols] = problem.jac(x)
            return sigma * problem.grad(x) + j.T @ lam

        hess_fd = np.column_stack(
            [
                (lag_grad(z + 2 * eps * eye[j]) - lag_grad(z - 2 * eps * eye[j]))
                / (4 * eps)
                for j in range(problem.n_z)
            ]
        )
        np.testing.assert_allclose(hess, hess_fd, rtol=5e-3, atol=4e-4)

        def dense_lagrangian(value):
            objective, f2n, defect, _objs, _phys = problem._values_tensor(value)
            lam_t = torch.as_tensor(lam, dtype=value.dtype, device=value.device)
            return (
                sigma * objective
                + lam_t[-1] * f2n
                + (lam_t[:-1] * defect.reshape(-1)).sum()
            )

        hess_ad = torch.func.hessian(dense_lagrangian)(zt)
        np.testing.assert_allclose(hess, hess_ad.numpy(), rtol=1e-9, atol=1e-10)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_bundle_and_hessian_replay_changed_inputs(self):
        model, zones, _, _, heaters = build_chain_model(
            1, "test_pareto_collocation_cuda"
        )
        nonlinear = tb.FunctionSystem(
            inputs=["x"],
            fn=lambda values: values["x"].square(),
            id="SquaredTemperature",
        )
        model.add_connection(zones[0], nonlinear, "indoorTemperature", "x")
        model.load(draw_semantic_model=False, draw_simulation_model=False)
        model.to("cuda", torch.float64)
        optimizer = tb.Optimizer(
            tb.Simulator(
                model,
                execution_mode="functional",
                execution_backend="cuda_graph",
            )
        )
        optimizer._variables = [(heaters[0], "scheduleValue", 0.0, 3000.0)]
        optimizer._objectives = [
            (heaters[0], "scheduleValue", "min"),
            (nonlinear, "output", "max"),
        ]
        optimizer._eq_cons = []
        optimizer._ineq_cons = []
        optimizer._start_time = [START + datetime.timedelta(hours=5)]
        optimizer._end_time = [START + datetime.timedelta(hours=8)]
        optimizer._stepSize = [OPT_STEP]
        (
            optimizer._second_time_steps,
            optimizer._date_time_steps,
            optimizer._max_timesteps,
            optimizer._n_timesteps,
        ) = tb.Simulator.get_simulation_timesteps(
            optimizer._start_time, optimizer._end_time, optimizer._stepSize
        )
        optimizer._timestep_mask = torch.ones(
            optimizer._max_timesteps, 1, dtype=torch.bool
        )
        optimizer._max_values = {}
        controls, bounds = optimizer._prepare_scipy_problem(
            ("casadi", "ipopt", "ad"), {}
        )
        problem = ParetoCollocation(
            optimizer,
            controls,
            bounds,
            delta=0.05,
            options={"hessian": "exact"},
        )
        problem.configure(0, 1, 0.05, -0.2, 0.8)
        z1 = problem.z0.copy()
        z2 = z1.copy()
        z2[: problem.n_u] = np.clip(
            z2[: problem.n_u] + 0.2,
            problem.lb[: problem.n_u],
            problem.ub[: problem.n_u],
        )
        z2[problem.n_u + problem.Da] += 0.05
        self.assertFalse(np.array_equal(z1[: problem.n_u], z2[: problem.n_u]))
        self.assertFalse(np.array_equal(z1[problem.n_u :], z2[problem.n_u :]))
        eager1 = problem._bundle_tensor(
            torch.as_tensor(z1, dtype=torch.float64, device="cuda")
        )[0]
        eager2 = problem._bundle_tensor(
            torch.as_tensor(z2, dtype=torch.float64, device="cuda")
        )[0]
        self.assertNotAlmostEqual(float(eager1), float(eager2))
        f1 = problem.fun(z1)
        f2 = problem.fun(z2)
        self.assertNotAlmostEqual(f1, f2)
        self.assertTrue(problem.stats["bundle_captured"])
        self.assertGreaterEqual(problem.stats["bundle_replays"], 1)

        problem.capture_derivatives = False
        eager_j1 = problem.jac(z1)
        eager_j2 = problem.jac(z2)
        problem.capture_derivatives = True
        graph_j1 = problem.jac(z1)
        graph_j2 = problem.jac(z2)
        np.testing.assert_allclose(graph_j1, eager_j1, rtol=1e-9, atol=1e-10)
        np.testing.assert_allclose(graph_j2, eager_j2, rtol=1e-9, atol=1e-10)
        self.assertTrue(problem.stats["jacobian_captured"])
        self.assertGreaterEqual(problem.stats["jacobian_replays"], 1)

        n_g = problem.n_links * problem.Da + 1
        h1 = problem.hessian(z1, 0.8, np.linspace(-0.2, 0.3, n_g))
        h2 = problem.hessian(z2, 0.6, np.linspace(0.25, -0.1, n_g))
        self.assertTrue(problem.stats["hessian_captured"])
        self.assertGreaterEqual(problem.stats["hessian_replays"], 1)
        self.assertFalse(np.allclose(h1, h2))


class TestOptimizeReturnsResult(unittest.TestCase):
    """Regression: the SciPy solve path must return its result object."""

    def test_optimize_returns_scipy_result(self):
        model, zones, _, _, heaters = build_chain_model(1, "test_pareto_optret_model")
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

    def test_obsolete_fast_option_is_rejected(self):
        model, _, _, _, heaters = build_chain_model(1, "test_pareto_fast_option_model")
        try:
            optimizer = tb.Optimizer(tb.Simulator(model))
            end = START + datetime.timedelta(hours=N_HOURS)
            with self.assertRaisesRegex(TypeError, "execution_mode='functional'"):
                optimizer.optimize(
                    start_time=START,
                    end_time=end,
                    step_size=OPT_STEP,
                    variables=[(heaters[0], "scheduleValue", 0.0, 3000.0)],
                    objectives=[(heaters[0], "scheduleValue", "min")],
                    options={"fast": True, "maxiter": 1},
                )
        finally:
            path = "generated_files/models/test_pareto_fast_option_model"
            if os.path.exists(path):
                shutil.rmtree(path)


if __name__ == "__main__":
    unittest.main()
