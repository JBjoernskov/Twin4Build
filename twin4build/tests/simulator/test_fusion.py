"""Tests for compile-time fusion of connected state-space clusters
(:class:`FusedStateSpaceSystem`).

Covers, on a two-zone + partition-wall model:

1. Exactness: the fused one-step map equals an independently hand-derived
   monolithic RC model discretized with exact ZOH.
2. Consistency: fused and unfused simulations agree at moderate parameters
   when the step size is small (the unfused path's Gauss-Seidel lag error
   vanishes as dt -> 0).
3. Stability: at a stiff coupling (R = 1e-4, 20-minute steps) where the
   unfused explicit coupling diverges, the fused simulation stays finite and
   physically tight (zone/wall temperatures pinned together).
"""

# Standard library imports
import datetime
import unittest

# Third party imports
import torch
from dateutil import tz

# Local application imports
import twin4build as tb

tb._IS_TESTING = True

START = datetime.datetime(2024, 1, 4, tzinfo=tz.gettz("Europe/Copenhagen"))


def build_model(r_a=0.02, r_b=0.02, fuse=True, model_id="test_fusion"):
    """Two thermal zones coupled by one WallSystem; zone A heated."""
    model = tb.Model(id=model_id)

    def make_zone(zone_id):
        return tb.BuildingSpaceThermalSystem(
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
    wall = tb.WallSystem(C=2e5, R_a=r_a, R_b=r_b, id="PartitionWall")

    outdoor = tb.ScheduleSystem(
        weekday_ruleset={"ruleset_default_value": 5.0}, id="Outdoor"
    )
    zero = tb.ScheduleSystem(
        weekday_ruleset={"ruleset_default_value": 0.0}, id="Zero"
    )
    supply_air_temp = tb.ScheduleSystem(
        weekday_ruleset={"ruleset_default_value": 20.0}, id="SupplyAirTemp"
    )
    heater_a = tb.ScheduleSystem(
        weekday_ruleset={
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

    model.add_connection(zone_a, wall, "indoorTemperature", "temperatureA")
    model.add_connection(zone_b, wall, "indoorTemperature", "temperatureB")
    model.add_connection(
        wall, zone_a, "heatFlowRateA", "wallHeatGain", input_port_index=0
    )
    model.add_connection(
        wall, zone_b, "heatFlowRateB", "wallHeatGain", input_port_index=0
    )

    model.load(
        draw_semantic_model=False,
        draw_simulation_model=False,
        enable_fusion=fuse,
    )
    return model, zone_a, zone_b, wall


def simulate(model, hours, step_size):
    sim = tb.Simulator(model)
    sim.simulate(
        start_time=START,
        end_time=START + datetime.timedelta(hours=hours),
        step_size=step_size,
        show_progress_bar=False,
    )
    return {
        "t_a": model.components["ZoneA"]
        .output["indoorTemperature"].history().detach().flatten(),
        "t_b": model.components["ZoneB"]
        .output["indoorTemperature"].history().detach().flatten(),
        "t_w": model.components["PartitionWall"]
        .output["wallTemperature"].history().detach().flatten(),
    }


def observed_support(matrix):
    """Matrix indices that are nonzero in at least one component batch."""
    mask = (matrix != 0).any(dim=0)
    return {tuple(int(v) for v in row) for row in mask.nonzero().tolist()}


class TestFusionExactness(unittest.TestCase):
    """The fused one-step map must equal the hand-derived monolithic model."""

    def test_matches_hand_derived_monolithic_model(self):
        model, zone_a, zone_b, wall = build_model()
        sim_model = model.simulation_model
        self.assertEqual(len(sim_model._fused_components), 1)
        fused = next(iter(sim_model._fused_components.values()))
        self.assertEqual(len(fused.members), 3)
        simulate(model, 1, 600)  # initializes the fused layout

        dt = 600.0
        # Joint state order follows the members' model insertion order:
        # [ZoneA.T_air, ZoneA.T_wall_ext, ZoneB.T_air, ZoneB.T_wall_ext, T_p].
        p = {
            "C_air": 1e6, "C_wall": 5e6, "R_in": 0.01, "R_out": 0.01,
            "C_p": 2e5, "R_a": 0.02, "R_b": 0.02,
        }
        # Hand-derived continuous dynamics (f_air = f_wall = 0, zero flows):
        #   C_air  dTa/dt = (Tw - Ta)/R_in + Q_heat + (Tp - Ta)/R_side
        #   C_wall dTw/dt = (Tout - Tw)/R_out + (Ta - Tw)/R_in
        #   C_p    dTp/dt = (TaA - Tp)/R_a + (TaB - Tp)/R_b
        # Inputs (constant over the step): [Tout, Q_heat_A].
        A = torch.zeros((5, 5), dtype=torch.float64)
        B = torch.zeros((5, 2), dtype=torch.float64)
        for base, r_side in ((0, p["R_a"]), (2, p["R_b"])):
            A[base, base] = -1 / (p["R_in"] * p["C_air"]) - 1 / (
                r_side * p["C_air"]
            )
            A[base, base + 1] = 1 / (p["R_in"] * p["C_air"])
            A[base, 4] = 1 / (r_side * p["C_air"])
            A[base + 1, base] = 1 / (p["R_in"] * p["C_wall"])
            A[base + 1, base + 1] = (
                -1 / (p["R_out"] * p["C_wall"]) - 1 / (p["R_in"] * p["C_wall"])
            )
            B[base + 1, 0] = 1 / (p["R_out"] * p["C_wall"])
            A[4, base] = 1 / ((p["R_a"], p["R_b"])[base // 2] * p["C_p"])
        A[4, 4] = -1 / (p["R_a"] * p["C_p"]) - 1 / (p["R_b"] * p["C_p"])
        B[0, 1] = 1 / p["C_air"]

        # Exact ZOH via the augmented-matrix exponential.
        n, m = 5, 2
        M = torch.zeros((n + m, n + m), dtype=torch.float64)
        M[:n, :n] = A * dt
        M[:n, n:] = B * dt
        eM = torch.matrix_exp(M)
        Ad, Bd = eM[:n, :n], eM[:n, n:]

        x0 = torch.tensor([21.0, 18.0, 19.0, 17.5, 20.0], dtype=torch.float64)
        t_out, q_heat = 5.0, 1500.0
        x_ref = Ad @ x0 + Bd @ torch.tensor([t_out, q_heat], dtype=torch.float64)

        # Fused one-step from the same state and held inputs.
        zero = torch.zeros((1, 1), dtype=torch.float64)
        inputs = {}
        for zid in ("ZoneA", "ZoneB"):
            inputs[f"{zid}.outdoorTemperature"] = zero + t_out
            for port in (
                "supplyAirFlowRate", "exhaustAirFlowRate", "globalIrradiation",
                "numberOfPeople",
            ):
                inputs[f"{zid}.{port}"] = zero
            inputs[f"{zid}.supplyAirTemperature"] = zero + 20.0
        inputs["ZoneA.heatGain"] = zero + q_heat
        inputs["ZoneB.heatGain"] = zero
        x_next, outs = fused.forward(
            x0.reshape(1, 1, 5), inputs, fused._forward_params(), dt
        )

        self.assertTrue(
            torch.allclose(x_next.flatten(), x_ref, rtol=1e-10, atol=1e-9),
            f"fused step {x_next.flatten().tolist()} != reference {x_ref.tolist()}",
        )
        # Published outputs: zone air temperatures are the first joint states.
        self.assertAlmostEqual(
            float(outs["ZoneA.indoorTemperature"]), float(x_ref[0]), places=9
        )
        self.assertAlmostEqual(
            float(outs["ZoneB.indoorTemperature"]), float(x_ref[2]), places=9
        )
        self.assertAlmostEqual(
            float(outs["PartitionWall.wallTemperature"]), float(x_ref[4]), places=9
        )


class TestFusionConsistency(unittest.TestCase):
    """Fused and unfused runs converge to each other as dt -> 0."""

    def test_matches_unfused_at_small_step(self):
        res_f = simulate(build_model(fuse=True, model_id="fuse_on")[0], 6, 30)
        res_u = simulate(build_model(fuse=False, model_id="fuse_off")[0], 6, 30)
        for key in ("t_a", "t_b", "t_w"):
            err = float((res_f[key] - res_u[key]).abs().max())
            self.assertLess(err, 0.05, f"{key}: fused vs unfused max err {err}")

    def test_unfused_escape_hatch(self):
        model, *_ = build_model(fuse=False, model_id="fuse_off2")
        self.assertEqual(len(model.simulation_model._fused_components), 0)
        order = [c.id for g in model.simulation_model._execution_order for c in g]
        self.assertIn("PartitionWall", order)


class TestFusionStability(unittest.TestCase):
    """Stiff coupling (R = 1e-4 at 20-minute steps): the explicit port
    coupling diverges; the fused block is unconditionally stable."""

    def test_stiff_coupling_stays_finite_and_tight(self):
        model, *_ = build_model(r_a=1e-4, r_b=1e-4, model_id="fuse_stiff")
        res = simulate(model, 24, 1200)
        for key in ("t_a", "t_b", "t_w"):
            self.assertTrue(
                bool(torch.isfinite(res[key]).all()), f"{key} not finite"
            )
        # With R ~ 0 the three temperatures are essentially one node.
        self.assertLess(float((res["t_a"] - res["t_w"]).abs().max()), 0.5)
        self.assertLess(float((res["t_b"] - res["t_w"]).abs().max()), 0.5)


class TestFusionStructuralSupport(unittest.TestCase):
    """Static matrix support must be safe at every parameter iterate."""

    def setUp(self):
        self.model, *_ = build_model(model_id="support_contract")
        simulate(self.model, 1, 600)
        self.fused = next(
            iter(self.model.simulation_model._fused_components.values())
        )

    def assert_support_covers(self, unit, matrices):
        declared = unit._ss_support()
        for name, matrix in zip(("D", "E", "F"), matrices[3:]):
            self.assertTrue(
                observed_support(matrix) <= set(declared[name]),
                f"{type(unit).__name__} {name} has an undeclared nonzero",
            )

    def test_declared_support_covers_zero_and_random_interior_values(self):
        generator = torch.Generator().manual_seed(126)
        for entry in self.fused._units:
            unit = entry["unit"]
            defaults = {
                name: getattr(unit, name).get() for name in unit.PARAM_NAMES
            }
            cases = [defaults]

            zero_case = dict(defaults)
            for name in ("f_air", "f_wall"):
                if name in zero_case:
                    zero_case[name] = torch.zeros_like(zero_case[name])
            cases.append(zero_case)

            for _ in range(5):
                params = {}
                for name, value in defaults.items():
                    factor = 0.6 + 0.8 * torch.rand((), generator=generator)
                    params[name] = (
                        value * factor
                        if bool((value != 0).any())
                        else torch.full_like(value, float(factor) * 0.1)
                    )
                cases.append(params)

            for params in cases:
                self.assert_support_covers(unit, unit._build_matrices(params))

    def test_composite_mass_unit_support_is_conservative(self):
        mass = tb.BuildingSpaceMassSystem(id="support_mass")
        params = {
            "V": torch.tensor([65.0], dtype=torch.float64),
            "G_occ": torch.tensor([3e-6], dtype=torch.float64),
            "m_inf": torch.tensor([0.001], dtype=torch.float64),
        }
        self.assert_support_covers(mass, mass._build_matrices(params))

    def test_assemble_is_differentiable_and_fullgraph_traceable(self):
        base = self.fused._forward_params()
        target_names = ("C_air", "R_in", "R_a")
        targets = []
        for entry in self.fused._units:
            for name in target_names:
                key = f"{entry['param_prefix']}.{name}"
                if key in base and key not in targets:
                    targets.append(key)
        targets = targets[:3]
        self.assertEqual(len(targets), 3)
        theta0 = torch.stack([base[key].reshape(-1)[0] for key in targets])

        def packed_matrices(theta):
            params = dict(base)
            for i, key in enumerate(targets):
                params[key] = theta[i].reshape(1)
            return torch.cat(
                [matrix.reshape(-1) for matrix in self.fused._assemble(params)]
            )

        value = packed_matrices(theta0)
        jacobian = torch.func.jacrev(packed_matrices)(theta0)
        curvature = torch.func.hessian(
            lambda theta: packed_matrices(theta).square().mean()
        )(theta0)
        self.assertTrue(bool(torch.isfinite(value).all()))
        self.assertTrue(bool(torch.isfinite(jacobian).all()))
        self.assertTrue(bool(torch.isfinite(curvature).all()))

        if hasattr(torch, "compile"):
            compiled = torch.compile(
                packed_matrices, backend="eager", fullgraph=True
            )
            self.assertTrue(
                torch.allclose(compiled(theta0), value, rtol=1e-10, atol=1e-12)
            )
            hessian_fn = torch.func.hessian(
                lambda theta: packed_matrices(theta).square().mean()
            )
            compiled_hessian = torch.compile(
                hessian_fn, backend="aot_eager", fullgraph=True
            )
            compiled_curvature = compiled_hessian(theta0)
            error = (compiled_curvature - curvature).abs()
            max_index = int(error.argmax())
            reference_at_max = float(curvature.reshape(-1)[max_index])
            compiled_at_max = float(compiled_curvature.reshape(-1)[max_index])
            self.assertTrue(
                torch.allclose(
                    compiled_curvature,
                    curvature,
                    rtol=1e-8,
                    atol=1e-10,
                ),
                "compiled Hessian max-error entry: "
                f"{compiled_at_max:.12g} != {reference_at_max:.12g}",
            )

    def test_transform_path_matches_eager_and_does_not_mutate_cache(self):
        state = self.fused.get_state()
        inputs = {name: port.get() for name, port in self.fused.input.items()}
        params = self.fused._forward_params()
        target = next(key for key in params if key.endswith(".R_a"))
        theta0 = params[target].reshape(-1)[0]

        def transformed(theta):
            live_params = dict(params)
            live_params[target] = theta.reshape(1)
            state_next, _ = self.fused.forward(
                state,
                inputs,
                live_params,
                600.0,
                transform_mode=True,
            )
            return state_next

        self.fused._fwd_mat_cache = None
        transformed_value = transformed(theta0)
        self.assertIsNone(self.fused._fwd_mat_cache)

        eager_params = dict(params)
        eager_params[target] = theta0.reshape(1)
        eager_value, _ = self.fused.forward(
            state, inputs, eager_params, 600.0, transform_mode=False
        )
        self.assertTrue(
            torch.allclose(
                transformed_value, eager_value, rtol=2e-8, atol=2e-9
            )
        )

        jacobian = torch.func.jacrev(transformed)(theta0)
        curvature = torch.func.hessian(
            lambda theta: transformed(theta).square().mean()
        )(theta0)
        self.assertTrue(bool(torch.isfinite(jacobian).all()))
        self.assertTrue(bool(torch.isfinite(curvature).all()))
        if hasattr(torch, "compile"):
            transformed_hessian = torch.func.hessian(
                lambda theta: transformed(theta).square().mean()
            )
            compiled_hessian = torch.compile(
                transformed_hessian, backend="aot_eager", fullgraph=True
            )
            self.assertTrue(
                torch.allclose(
                    compiled_hessian(theta0),
                    curvature,
                    rtol=1e-7,
                    atol=1e-9,
                )
            )


if __name__ == "__main__":
    unittest.main()
