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
    """Two thermal zones coupled by one WallTorchSystem; zone A heated."""
    model = tb.Model(id=model_id)

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
    wall = tb.WallTorchSystem(C=2e5, R_a=r_a, R_b=r_b, id="PartitionWall")

    outdoor = tb.ScheduleSystem(
        weekDayRulesetDict={"ruleset_default_value": 5.0}, id="Outdoor"
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

    model.add_connection(zone_a, wall, "indoorTemperature", "temperatureA")
    model.add_connection(zone_b, wall, "indoorTemperature", "temperatureB")
    model.add_connection(
        wall, zone_a, "heatFlowRateA", "wallHeatGain", input_port_index=0
    )
    model.add_connection(
        wall, zone_b, "heatFlowRateB", "wallHeatGain", input_port_index=0
    )

    model.simulation_model.enable_fusion = fuse
    model.load(draw_semantic_model=False, draw_simulation_model=False)
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


if __name__ == "__main__":
    unittest.main()
