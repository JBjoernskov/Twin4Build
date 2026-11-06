"""A zone served by several AHU branches (issue #179).

The zone's ``supplyAirFlowRate`` / ``exhaustAirFlowRate`` are Vector ports,
one slot per branch, summed inside the zone; the AHU keeps one branch per
damper and maps each exhaust branch to its room's exhaust temperature.  One
``add_connection`` per branch on the same output -> input pair merges into a
single connection with tensor slot indices, which both engines route.
"""

# Standard library imports
import datetime
import os
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
        id=id_,
        thermal=dict(C_wall=1e6, C_air=1e4, R_out=0.01, R_in=0.02, f_wall=0.5, f_air=0.3, Q_occ_gain=100.0),
        mass=dict(V=100.0),
    )


def build_model(model_id, exhaust_per_zone=False):
    """Each zone's temperature on the exhaust slot of every branch serving it
    (zone B's on 1 and 2, one connection).  ``exhaust_per_zone``: one slot
    per zone instead, which the AHU refuses (it keeps no branch -> room map)."""
    model = tb.Model(id=model_id)
    ahu = AirHandlingUnitSystem(
        id="ahu",
        supply_damper={"a": 1.0, "nominalAirFlowRate": 0.5}, exhaust_damper={"a": 1.0, "nominalAirFlowRate": 0.5},
        heat_recovery={
            "eps_75_h": 0.7, "eps_100_h": 0.75, "eps_75_c": 0.65, "eps_100_c": 0.7,
            "primaryAirFlowRateMax": 1.0, "secondaryAirFlowRateMax": 1.0,
        },
        supply_fan={
            "beta0": 0.0, "beta1": 200.0 / 3.0, "beta2": 400.0, "beta3": 1000.0, "f_total": 1.0,
        },
        exhaust_fan={
            "beta0": 0.0, "beta1": 200.0 / 3.0, "beta2": 400.0, "beta3": 1000.0, "f_total": 1.0,
        },
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
    if exhaust_per_zone:
        for slot, zone in enumerate(zones.values()):
            model.add_connection(zone, ahu, "indoorTemperature", "exhaustTemperature", input_port_index=slot)
    else:
        # Exhaust temperature: one slot per branch, a zone on each of its branches.
        model.add_connection(zones["A"], ahu, "indoorTemperature", "exhaustTemperature", input_port_index=0)
        model.add_connection(zones["B"], ahu, "indoorTemperature", "exhaustTemperature", input_port_index=[1, 2])
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

    def test_merged_connection_and_a_temperature_per_branch(self):
        model, ahu, zones = build_model("vector_flow_ports_topology")
        (cp,) = [cp for cp in zones["B"].connects_at if cp.input_port == "supplyAirFlowRate"]
        (conn,) = cp.connects_system_through
        self.assertEqual(cp.input_port_index[conn].tolist(), [0, 1])
        self.assertEqual(cp.output_port_index[conn].tolist(), [1, 2])
        tb.Simulator(model, execution_mode="object").simulate(**self._kwargs())
        self.assertEqual(zones["B"].input["supplyAirFlowRate"].n_v, 2)
        self.assertEqual(ahu.input["exhaustTemperature"].n_v, 3)

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

    def test_per_branch_exhaust_ratio_takes_the_wired_branch_count(self):
        """A reloaded model carries the constructor's ``n_branches`` (1); the
        per-branch ratio must still get one entry per WIRED branch, like the
        dampers."""
        model, ahu, zones = build_model("vector_flow_ports_ratio_width")
        ahu.exhaust_follows_supply = True
        ahu.exhaust_ratio_per_branch = True
        ahu.n_branches = 1
        tb.Simulator(model, execution_mode="object").simulate(**self._kwargs())
        self.assertEqual(tuple(ahu.exhaustFlowRatio.get().shape), (3,))
        self.assertEqual(tuple(ahu.supply_damper.a.get().shape), (3,))
        reference = ahu.output["exhaustAirFlowRate"]._history.detach().clone()
        # The functional engine applies the per-branch ratio on the branch
        # axis too (it used to broadcast it as a batch axis and grow the
        # state by n_branches^2 - n_branches entries).
        tb.Simulator(model, execution_mode="functional").simulate(**self._kwargs())
        torch.testing.assert_close(ahu.output["exhaustAirFlowRate"]._history, reference)

    def test_untied_damper_offset_survives_serialization(self):
        """An AHU whose branch dampers keep a minimum flow (``set_c``)
        reloads from its TTL with ``c`` free and the same flows, in both
        engines."""
        import shutil

        model, ahu, zones = build_model("vector_flow_ports_offset")
        ahu.supply_damper.c = 0.02 - float(ahu.supply_damper.a.get().reshape(-1)[0])
        model.load()
        tb.Simulator(model, execution_mode="object").simulate(**self._kwargs())
        reference = ahu.output["supplyAirFlowRate"]._history.detach().clone()
        self.assertGreater(float(reference.min()), 0.0)  # closed branches still pass the minimum
        self.assertIn("supply_damper.c", [e[1] for e in ahu.get_estimable_parameters()])
        self.assertNotIn("exhaust_damper.c", [e[1] for e in ahu.get_estimable_parameters()])
        model.serialize()
        path, _ = model._simulation_model._semantic_model.get_dir(filename="instance_graph.ttl")
        try:
            reloaded = tb.Model(id="vector_flow_ports_offset_reloaded")
            reloaded.load(filename=path)
            ahu2 = reloaded.components["ahu"]
            self.assertIsNotNone(ahu2.supply_damper.c)
            self.assertIsNone(ahu2.exhaust_damper.c)
            for mode in ("object", "functional"):
                tb.Simulator(reloaded, execution_mode=mode).simulate(**self._kwargs())
                torch.testing.assert_close(ahu2.output["supplyAirFlowRate"]._history, reference, msg=mode)
        finally:
            for mid in ("vector_flow_ports_offset", "vector_flow_ports_offset_reloaded"):
                shutil.rmtree(os.path.join("generated_files", "models", mid), ignore_errors=True)

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
        # The submodels see the total: the room's forward sums the slots for
        # them (a child's own ports are never written).
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


def _exhaust_slots(ahu):
    """``{sender id: (input slots, output slot)}`` on the AHU's exhaustTemperature."""
    (cp,) = [cp for cp in ahu.connects_at if cp.input_port == "exhaustTemperature"]
    out = {}
    for conn in cp.connects_system_through:
        in_v = cp.input_port_index[conn]
        out[conn.connects_system.id] = (torch.as_tensor(in_v).reshape(-1).tolist(), cp.output_port_index[conn])
    return out


def _histories(model, ahu, zones):
    out = {f"{k}.{p}": z.output[p]._history.detach().clone()
           for k, z in zones.items() for p in ("indoorTemperature", "indoorCO2")}
    out.update({f"ahu.{p}": ahu.output[p]._history.detach().clone()
                for p in ("supplyAirTemperature", "supplyAirFlowRate", "exhaustAirFlowRate")})
    return out


class TestScalarIntoSeveralSlots(unittest.TestCase):
    """One scalar output filling several slots of a Vector input: a zone's
    temperature on the exhaust slot of each AHU branch serving it."""

    START = datetime.datetime(2023, 1, 1, tzinfo=tz.UTC)

    def _kwargs(self):
        return dict(start_time=self.START, end_time=self.START + datetime.timedelta(hours=2), step_size=600, show_progress_bar=False)

    def test_one_connection_carries_the_slots(self):
        model, ahu, zones = build_model("scalar_fan_out_topology")
        self.assertEqual(_exhaust_slots(ahu), {"zone_a": ([0], None), "zone_b": ([1, 2], None)})
        self.assertEqual(ahu.get_n_v_from_connections("exhaustTemperature"), 3)

    def test_wiring_the_pair_again_on_other_slots_merges(self):
        model, ahu, zones = build_model("scalar_fan_out_merge", exhaust_per_zone=True)
        # zone B on slot 1 (per zone); add slot 2 and it is per branch
        model.add_connection(zones["B"], ahu, "indoorTemperature", "exhaustTemperature", input_port_index=2)
        self.assertEqual(_exhaust_slots(ahu)["zone_b"], ([1, 2], None))

    def test_a_slot_twice_is_rejected(self):
        model, ahu, zones = build_model("scalar_fan_out_twice")
        with self.assertRaises(AssertionError):
            model.add_connection(zones["B"], ahu, "indoorTemperature", "exhaustTemperature", input_port_index=1)
        with self.assertRaises(AssertionError):
            model.add_connection(zones["A"], ahu, "indoorTemperature", "exhaustTemperature", input_port_index=[3, 3])

    def test_one_temperature_slot_per_zone_raises_and_names_the_wiring(self):
        """The AHU reads one exhaust temperature per branch and keeps no
        branch -> room map: one slot per zone is refused, with the fix."""
        model, ahu, zones = build_model("scalar_fan_out_per_zone", exhaust_per_zone=True)
        with self.assertRaisesRegex(ValueError, "input_port_index"):
            tb.Simulator(model, execution_mode="object").simulate(**self._kwargs())

    def test_a_zone_on_two_branches_fills_both_slots_in_both_engines(self):
        for mode in ("object", "functional"):
            model, ahu, zones = build_model(f"scalar_fan_out_per_branch_{mode}")
            tb.Simulator(model, execution_mode=mode).simulate(**self._kwargs())
            self.assertEqual(ahu.input["exhaustTemperature"].n_v, 3)
            exhaust = ahu.input["exhaustTemperature"]._history
            torch.testing.assert_close(exhaust[..., 1], exhaust[..., 2], rtol=0, atol=0, msg=mode)

    def test_the_slots_survive_serialization(self):
        import shutil

        model, ahu, zones = build_model("scalar_fan_out_saved")
        tb.Simulator(model, execution_mode="object").simulate(**self._kwargs())
        reference = zones["B"].output["indoorTemperature"]._history.detach().clone()
        model.serialize()
        path, _ = model._simulation_model._semantic_model.get_dir(filename="instance_graph.ttl")
        try:
            reloaded = tb.Model(id="scalar_fan_out_saved_reloaded")
            reloaded.load(filename=path)
            ahu2 = reloaded.components["ahu"]
            self.assertEqual(_exhaust_slots(ahu2), {"zone_a": ([0], None), "zone_b": ([1, 2], None)})
            for mode in ("object", "functional"):
                tb.Simulator(reloaded, execution_mode=mode).simulate(**self._kwargs())
                torch.testing.assert_close(
                    reloaded.components["zone_b"].output["indoorTemperature"]._history, reference, msg=mode
                )
        finally:
            for mid in ("scalar_fan_out_saved", "scalar_fan_out_saved_reloaded"):
                shutil.rmtree(os.path.join("generated_files", "models", mid), ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
