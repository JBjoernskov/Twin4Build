"""A zone served by several AHU branches (issue #179).

The zone's ``supplyAirFlowRate`` / ``exhaustAirFlowRate`` are Vector ports,
one slot per branch, summed inside the zone; the AHU keeps one branch per
damper and maps each exhaust branch to its room's exhaust temperature.  One
``add_connection`` per branch on the same output -> input pair merges into a
single connection with tensor slot indices, which both engines route.
"""

# Standard library imports
import datetime
import unittest

# Third party imports
import torch
from dateutil import tz

# Local application imports
import twin4build as tb
from twin4build.systems.air_handling_unit.air_handling_unit_system import (
    AirHandlingUnitSystem,
)
from twin4build.systems.building_space.building_space_system import BuildingSpaceSystem
from twin4build.systems.schedule.schedule_system import ScheduleSystem

tb._IS_TESTING = True


def _schedule(id_, value):
    return ScheduleSystem(
        id=id_,
        weekday_ruleset={
            "ruleset_start_minute": [], "ruleset_end_minute": [], "ruleset_start_hour": [],
            "ruleset_end_hour": [], "ruleset_value": [], "ruleset_default_value": value,
        },
    )


def _zone(id_):
    return BuildingSpaceSystem(
        id=id_, C_wall=1e6, C_air=1e4, C_boundary=5e5, R_out=0.01, R_in=0.02, R_boundary=0.03,
        f_wall=0.5, f_air=0.3, Q_occ_gain=100.0, CO2_occ_gain=0.004, CO2_start=400.0, airVolume=100.0,
        T_wall_start=20.0, T_air_start=20.0, T_int_start=20.0, T_boundary_start=18.0,
    )


def build_model(model_id):
    model = tb.Model(id=model_id)
    ahu = AirHandlingUnitSystem(
        id="ahu",
        damper_kwargs={"a": 1.0, "nominalAirFlowRate": 0.5},
        heat_recovery_kwargs={
            "eps_75_h": 0.7, "eps_100_h": 0.75, "eps_75_c": 0.65, "eps_100_c": 0.7,
            "primaryAirFlowRateMax": 1.0, "secondaryAirFlowRateMax": 1.0,
        },
        supply_fan_kwargs={
            "nominalPowerRate": 1000.0, "nominalAirFlowRate": 1, "c1": 0, "c2": 0.2, "c3": 0.8, "c4": 0, "f_total": 1.0,
        },
        exhaust_fan_kwargs={
            "nominalPowerRate": 1000.0, "nominalAirFlowRate": 1, "c1": 0, "c2": 0.2, "c3": 0.8, "c4": 0, "f_total": 1.0,
        },
        n_branches=3,
    )
    # Three dampers: branch 0 serves zone A, branches 1 and 2 serve zone B.
    for slot, value in ((0, 0.8), (1, 0.4), (2, 0.6)):
        damper = _schedule(f"damper_{slot}", value)
        model.add_connection(damper, ahu, "scheduleValue", "supplyDamperPosition", input_port_index=slot)
        model.add_connection(damper, ahu, "scheduleValue", "exhaustDamperPosition", input_port_index=slot)
    model.add_connection(_schedule("t_sup", 18.0), ahu, "scheduleValue", "supplyAirTemperatureSetpoint")
    model.add_connection(_schedule("t_out", 5.0), ahu, "scheduleValue", "outdoorAirTemperature")
    zones = {"A": _zone("zone_a"), "B": _zone("zone_b")}
    # Zone A: one branch, no index needed on the scalar side; zone B: two
    # branches into its Vector flow ports, one add_connection per branch.
    model.add_connection(ahu, zones["A"], "supplyAirFlowRate", "supplyAirFlowRate", output_port_index=0, input_port_index=0)
    model.add_connection(ahu, zones["A"], "exhaustAirFlowRate", "exhaustAirFlowRate", output_port_index=0, input_port_index=0)
    for k, branch in enumerate((1, 2)):
        model.add_connection(ahu, zones["B"], "supplyAirFlowRate", "supplyAirFlowRate", output_port_index=branch, input_port_index=k)
        model.add_connection(ahu, zones["B"], "exhaustAirFlowRate", "exhaustAirFlowRate", output_port_index=branch, input_port_index=k)
    # Exhaust temperature: one slot per zone.
    for slot, zone in enumerate(zones.values()):
        model.add_connection(zone, ahu, "indoorTemperature", "exhaustTemperature", input_port_index=slot)
    for name, zone in zones.items():
        model.add_connection(ahu, zone, "supplyAirTemperature", "supplyAirTemperature")
        model.add_connection(_schedule(f"t_out_{name}", 5.0), zone, "scheduleValue", "outdoorTemperature")
        model.add_connection(_schedule(f"people_{name}", 2.0), zone, "scheduleValue", "numberOfPeople")
        model.add_connection(_schedule(f"sun_{name}", 0.0), zone, "scheduleValue", "globalIrradiation")
        model.add_connection(_schedule(f"gain_{name}", 0.0), zone, "scheduleValue", "heatGain")
    model.load()
    return model, ahu, zones


class TestVectorFlowPorts(unittest.TestCase):
    START = datetime.datetime(2023, 1, 1, tzinfo=tz.UTC)

    def _kwargs(self):
        return dict(start_time=self.START, end_time=self.START + datetime.timedelta(hours=2), step_size=600, show_progress_bar=False)

    def test_merged_connection_and_branch_map(self):
        model, ahu, zones = build_model("vector_flow_ports_topology")
        (cp,) = [cp for cp in zones["B"].connects_at if cp.input_port == "supplyAirFlowRate"]
        (conn,) = cp.connects_system_through
        self.assertEqual(cp.input_port_index[conn].tolist(), [0, 1])
        self.assertEqual(cp.output_port_index[conn].tolist(), [1, 2])
        tb.Simulator(model, execution_mode="object").simulate(**self._kwargs())
        self.assertEqual(zones["B"].input["supplyAirFlowRate"].n_v, 2)
        self.assertEqual(ahu._branch_room_index.tolist(), [0, 1, 1])

    def test_fan_speed_gates_the_branch_flows(self):
        model, ahu, zones = build_model("vector_flow_ports_fan_gate")
        tb.Simulator(model, execution_mode="object").simulate(**self._kwargs())
        ungated = ahu.output["supplyAirFlowRate"]._history.detach().clone()
        self.assertGreater(float(ungated.mean()), 0.0)
        # Fan stopped: no branch moves air, whatever the dampers say.
        model.add_connection(_schedule("fan_off", 0.0), ahu, "scheduleValue", "supplyFanSpeed")
        model.load()
        for mode in ("object", "functional"):
            tb.Simulator(model, execution_mode=mode).simulate(**self._kwargs())
            torch.testing.assert_close(
                ahu.output["supplyAirFlowRate"]._history, torch.zeros_like(ungated), msg=mode
            )

    def test_zone_sums_its_branches_and_engines_agree(self):
        model, ahu, zones = build_model("vector_flow_ports_parity")
        tb.Simulator(model, execution_mode="object").simulate(**self._kwargs())
        branch_flows = ahu.output["supplyAirFlowRate"]._history.detach().clone()  # (n_t, n_s, n_c, 3)
        zone_b_slots = zones["B"].input["supplyAirFlowRate"]._history.detach().clone()
        # zone -> AHU (exhaust temperature) -> zone (flow) is a cycle; the
        # zone executes first and reads the AHU's previous step (Gauss-Seidel
        # lag), so slot k of zone B at step t is branch k+1 at step t-1.
        torch.testing.assert_close(zone_b_slots[1:, ..., 0], branch_flows[:-1, ..., 1])
        torch.testing.assert_close(zone_b_slots[1:, ..., 1], branch_flows[:-1, ..., 2])
        # The submodels see the total.
        torch.testing.assert_close(
            zones["B"].thermal.input["supplyAirFlowRate"]._history.reshape(zone_b_slots.shape[:-1]),
            zone_b_slots.sum(dim=-1),
        )
        # Branches 1 and 2 carry more air than branch 0 (dampers 0.4 + 0.6 > 0.8),
        # so zone B must be ventilated harder than zone A.
        self.assertGreater(float(zone_b_slots.sum(dim=-1).mean()), float(branch_flows[..., 0].mean()))
        t_object = {k: z.output["indoorTemperature"]._history.detach().clone() for k, z in zones.items()}
        co2_object = {k: z.output["indoorCO2"]._history.detach().clone() for k, z in zones.items()}

        tb.Simulator(model, execution_mode="functional").simulate(**self._kwargs())
        for k, z in zones.items():
            torch.testing.assert_close(z.output["indoorTemperature"]._history, t_object[k], rtol=1e-6, atol=1e-6)
            torch.testing.assert_close(z.output["indoorCO2"]._history, co2_object[k], rtol=1e-6, atol=1e-4)
        # The functional engine materialises input histories same-step (no
        # Gauss-Seidel lag), so compare the routing, not the timing.
        functional_slots = zones["B"].input["supplyAirFlowRate"]._history
        self.assertEqual(tuple(functional_slots.shape), tuple(zone_b_slots.shape))
        torch.testing.assert_close(functional_slots[1:], zone_b_slots[1:], rtol=1e-6, atol=1e-8)


if __name__ == "__main__":
    unittest.main()
