"""Keeping some groups of a measured partition: ``keep_groups``.

A factored unit feeding three rooms through three terminals, as in
``test_partition``.  A few rooms of it are a model of their own: the groups
kept, the groups that send into them, and the leaves that feed what is
kept.  The return junction, which maps every branch to the slot of its
room's temperature, follows the connections that survive.
"""
import datetime
import shutil
import tempfile
import unittest

import torch

import twin4build as tb
from twin4build.model.partition import cut_measured_edges, keep_groups, measured_partition
from twin4build.systems.air_handling_unit.air_handling_unit_core_system import (
    AirHandlingUnitCoreSystem,
)
from twin4build.systems.damper.damper_system import DamperSystem
from twin4build.systems.junction.return_flow_junction_system import (
    ReturnFlowJunctionSystem,
)
from twin4build.systems.junction.supply_flow_junction_system import (
    SupplyFlowJunctionSystem,
)
from twin4build.utils.get_main_dir import get_main_dir, set_main_dir

tb._IS_TESTING = True

from twin4build.tests.model.test_partition import END, FAN_KWARGS, RATIO, START, STEP, Room, data_sensor, series

N_ROOMS = 3

POSITIONS = ([1.0, 0.5, 0.3, 0.7, 1.0, 0.6, 0.2, 0.9], [0.2, 0.4, 0.6, 0.8, 1.0, 0.8, 0.6, 0.4], [0.9, 0.9, 0.1, 0.1, 0.5, 0.5, 0.7, 0.7])


def setUpModule():
    """The models of this module keep their files in a temporary folder."""
    global _MAIN_DIR, _FILES
    _MAIN_DIR = get_main_dir()
    _FILES = tempfile.mkdtemp()
    set_main_dir(_FILES)


def tearDownModule():
    set_main_dir(_MAIN_DIR)
    shutil.rmtree(_FILES, ignore_errors=True)


def build(model_id="keep_groups"):
    model = tb.Model(id=model_id)
    outdoor = series([5.0] * 8, id="outdoor")
    setpoint = series([18.0] * 8, id="setpoint")
    sj = SupplyFlowJunctionSystem(id="supply_junction")
    rj = ReturnFlowJunctionSystem(id="return_junction", branch_temperature_slots=list(range(N_ROOMS)))
    unit = AirHandlingUnitCoreSystem(id="unit", supply_fan_kwargs=dict(FAN_KWARGS), exhaust_fan_kwargs=dict(FAN_KWARGS))
    for k in range(N_ROOMS):
        d = DamperSystem(id=f"terminal{k}", a=1.0, nominalAirFlowRate=0.5, exhaustFlowRatio=RATIO)
        r = Room(id=f"room{k}", k=0.5 + 0.25 * k)
        model.add_connection(series(POSITIONS[k], id=f"pos{k}"), d, "measuredValue", "damperPosition")
        model.add_connection(d, r, "airFlowRate", "flow", input_port_index=0)
        model.add_connection(d, sj, "airFlowRate", "airFlowRateOut", input_port_index=k)
        model.add_connection(d, rj, "exhaustAirFlowRate", "airFlowRateIn", input_port_index=k)
        model.add_connection(r, rj, "T", "airTemperatureIn", input_port_index=k)
        model.add_connection(unit, r, "supplyAirTemperature", "supplyT")
        model.add_connection(d, data_sensor(f"flow_sensor{k}"), "airFlowRate", "measuredValue")
        model.add_connection(r, data_sensor(f"T_sensor{k}"), "T", "measuredValue")
    model.add_connection(sj, unit, "airFlowRateIn", "totalSupplyAirFlowRate")
    model.add_connection(rj, unit, "airFlowRateOut", "totalExhaustAirFlowRate")
    model.add_connection(rj, unit, "airTemperatureOut", "returnAirTemperature")
    model.add_connection(outdoor, unit, "measuredValue", "outdoorAirTemperature")
    model.add_connection(setpoint, unit, "measuredValue", "supplyAirTemperatureSetpoint")
    model.add_connection(unit, data_sensor("supplyT_sensor"), "supplyAirTemperature", "measuredValue")
    model.load(draw_semantic_model=False, draw_simulation_model=False)
    return model


def partition_of(model, derived=True):
    """The measured partition; with ``derived`` the terminals' exhaust flows
    are a fixed ratio of their measured supply flows, which gives every
    terminal a group of its own."""
    mapping = None
    if derived:
        mapping = {
            (f"terminal{k}", "exhaustAirFlowRate"): (model.components[f"flow_sensor{k}"], lambda s: RATIO * s)
            for k in range(N_ROOMS)
        }
    return measured_partition(model, derived=mapping)


def simulate(model):
    model.load(draw_semantic_model=False, draw_simulation_model=False)
    simulator = tb.Simulator(model)
    simulator.simulate(start_time=START, end_time=END, step_size=STEP, show_progress_bar=False)
    return simulator


def trajectory(model, cid, port="T"):
    return model.components[cid].output[port].history().reshape(-1).detach().clone()


def senders_of(component):
    return {conn.connects_system.id for cp in component.connects_at for conn in cp.connects_system_through}


class TestKeepGroups(unittest.TestCase):
    def test_rooms_of_the_cut_model_with_what_sends_into_them(self):
        """After the cut the groups are independent: two rooms with their
        terminals simulate as they do in the whole model."""
        reference = build("keep_cut_reference")
        cut_measured_edges(reference, partition_of(reference))
        simulate(reference)

        model = build("keep_cut")
        p = partition_of(model)
        cut_measured_edges(model, p)
        rooms = [p.group_of["room0"], p.group_of["room2"]]
        report = keep_groups(model, p, rooms, with_senders=True, stop_at=[p.group_of["unit"]])
        self.assertEqual(
            set(report["kept"]),
            {
                "room0", "T_sensor0", "room2", "T_sensor2",  # the groups named
                "terminal0", "flow_sensor0", "terminal2", "flow_sensor2",  # the groups that send into them
                "pos0", "pos2",  # the leaves of the terminals
                "flow_sensor0__replay", "flow_sensor2__replay", "supplyT_sensor__replay",  # and of the rooms
            },
        )
        self.assertEqual(set(report["kept"]), set(model.components))
        self.assertEqual(report["groups"], sorted(rooms + [p.group_of["terminal0"], p.group_of["terminal2"]]))
        for cid in ("room1", "terminal1", "unit", "supply_junction", "return_junction", "outdoor", "T_sensor1__replay"):
            self.assertIn(cid, report["removed"])
        self.assertEqual(report["repaired"], [])
        simulate(model)
        for cid in ("room0", "room2"):
            torch.testing.assert_close(trajectory(model, cid), trajectory(reference, cid))

    def test_without_senders_the_groups_and_their_leaves(self):
        model = build("keep_plain")
        p = partition_of(model)
        cut_measured_edges(model, p)
        report = keep_groups(model, p, [p.group_of["room1"]], with_senders=False)
        self.assertEqual(
            set(report["kept"]), {"room1", "T_sensor1", "flow_sensor1__replay", "supplyT_sensor__replay"}
        )
        self.assertEqual(report["groups"], [p.group_of["room1"]])
        self.assertEqual(senders_of(model.components["room1"]), {"flow_sensor1__replay", "supplyT_sensor__replay"})
        simulate(model)
        self.assertTrue(torch.isfinite(trajectory(model, "room1")).all())

    def test_senders_are_followed_through_the_unit_unless_stopped(self):
        """Every room receives from the unit and the unit from every room:
        through it the rooms' senders are the whole model."""
        model = build("keep_all")
        p = partition_of(model)
        before = set(model.components)
        report = keep_groups(model, p, [p.group_of["room0"]], with_senders=True)
        self.assertEqual(set(report["kept"]), before)
        self.assertEqual(report["removed"], [])
        self.assertEqual(report["groups"], sorted(set(p.group_of.values()) - {p.group_of[c] for c in ("outdoor", "setpoint", "pos0", "pos1", "pos2")}))
        # a group named is kept even when the senders stop at it
        model = build("keep_stop_named")
        p = partition_of(model)
        unit = p.group_of["unit"]
        report = keep_groups(model, p, [p.group_of["room0"], unit], with_senders=False, stop_at=[unit])
        self.assertIn("unit", report["kept"])
        self.assertIn("return_junction", report["kept"])

    def test_the_return_junction_follows_a_room_that_is_gone(self):
        """The terminals stay with the unit (their exhaust flows are not
        measured), one room goes: its branch still carries flow and takes
        the temperature of a room that is left."""
        model = build("keep_junction_room")
        p = partition_of(model, derived=False)
        self.assertIn("terminal1", p.groups[p.group_of["unit"]])
        report = keep_groups(model, p, [p.group_of["room0"], p.group_of["room2"], p.group_of["unit"]], with_senders=False)
        self.assertEqual(report["removed"], ["room1", "T_sensor1"])
        self.assertEqual(report["repaired"], ["return_junction"])
        junction = model.components["return_junction"]
        self.assertEqual(junction.branch_temperature_slots, [0, 0, 2])
        self.assertEqual(senders_of(junction), {"terminal0", "terminal1", "terminal2", "room0", "room2"})
        simulate(model)
        self.assertEqual(junction.input["airFlowRateIn"].get().shape[-1], 3)
        # the return temperature mixes the temperatures of the rooms that are left
        mixed = trajectory(model, "return_junction", "airTemperatureOut")
        rooms = torch.stack([trajectory(model, "room0"), trajectory(model, "room2")])
        self.assertTrue(torch.isfinite(mixed).all())
        self.assertGreaterEqual(float(mixed.min()), float(rooms.min()) - 1e-9)
        self.assertLessEqual(float(mixed.max()), float(rooms.max()) + 1e-9)

    def test_the_return_junction_follows_a_branch_that_is_gone(self):
        """The last room goes with its terminal: the junction's map is cut
        to the branches that are left."""
        model = build("keep_junction_branch")
        p = partition_of(model)
        groups = [p.group_of[c] for c in ("room0", "room1", "terminal0", "terminal1", "unit")]
        report = keep_groups(model, p, groups, with_senders=False)
        self.assertEqual(
            set(report["removed"]), {"room2", "T_sensor2", "terminal2", "flow_sensor2", "pos2"}
        )
        self.assertEqual(report["repaired"], ["return_junction"])
        junction = model.components["return_junction"]
        self.assertEqual(junction.branch_temperature_slots, [0, 1])
        simulate(model)
        self.assertEqual(junction.input["airFlowRateIn"].get().shape[-1], 2)
        self.assertTrue(torch.isfinite(trajectory(model, "return_junction", "airTemperatureOut")).all())
        self.assertTrue(torch.isfinite(trajectory(model, "room0")).all())

    def test_a_junction_whose_connections_all_survive_is_left_alone(self):
        model = build("keep_junction_whole")
        p = partition_of(model, derived=False)
        everything = sorted(set(p.group_of.values()))
        report = keep_groups(model, p, everything, with_senders=False)
        self.assertEqual(report["removed"], [])
        self.assertEqual(report["repaired"], [])
        self.assertEqual(model.components["return_junction"].branch_temperature_slots, [0, 1, 2])

    def test_a_group_the_partition_does_not_have(self):
        model = build("keep_unknown")
        p = partition_of(model)
        before = set(model.components)
        with self.assertRaises(ValueError):
            keep_groups(model, p, [0, len(p.groups)])
        self.assertEqual(set(model.components), before)


if __name__ == "__main__":
    unittest.main()
