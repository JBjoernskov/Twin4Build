"""Worlds by signal (3 Nov 2026): a world copies what the signals it needs depend on, not whole components.

Two rooms A and B (temperature and CO2 scored), each supplied by a damper driven by a schedule; no damper has a
flow meter.  On top of them, one of:

* a hallway H (a room) whose supply is the two rooms' surplus air, its own surplus metered and scored: H's world
  needs A's and B's surplus, which depend on their dampers only -- so it gets the rooms' projections
  (:class:`~twin4build.AirImbalanceSystem`) and copies of the two dampers (their parameters in consensus), and A and B
  stay worlds of their own;
* a junction J summing the two dampers' flows, metered and scored (a unit's supply meter): J's world gets copies of
  the dampers, A and B stay apart.

By component (the rule before), either would put A, B and everything else in one group.
"""

import datetime
import unittest

import pandas as pd
from dateutil import tz

import twin4build as tb
from twin4build.model.partition import cut_measured_edges, measured_partition, split_worlds
from twin4build.tests.model.test_split_worlds import START, STEP, _sched, _series, senders

tb._IS_TESTING = True


def _room(model, rid):
    room = tb.BuildingSpaceSystem(
        thermal_kwargs=dict(C_air=2e6, C_wall=5e6, R_out=0.01, R_in=0.01, f_wall=0.0, f_air=0.0, Q_occ_gain=100.0),
        mass_kwargs=dict(V=100.0, G_occ=5e-6, m_inf=0.01),
        id=rid,
    )
    for port, value in (("outdoorTemperature", 0.0), ("supplyAirTemperature", 20.0), ("globalIrradiation", 0.0),
                        ("numberOfPeople", 0.0), ("heatGain", 500.0), ("outdoorCo2Concentration", 400.0)):
        model.add_connection(_sched(value, f"{rid}_{port}"), room, "scheduleValue", port)
    model.add_connection(_sched(0.02, f"{rid}_exh"), room, "scheduleValue", "exhaustAirFlowRate", input_port_index=0)
    damper = tb.DamperSystem(nominalAirFlowRate=0.1, id=f"{rid}_damper")
    model.add_connection(_sched(0.6, f"{rid}_command"), damper, "scheduleValue", "damperPosition")
    model.add_connection(damper, room, "airFlowRate", "supplyAirFlowRate", input_port_index=0)
    sensors = [_series(20.0, f"{rid}_T"), _series(400.0, f"{rid}_CO2")]
    model.add_connection(room, sensors[0], "indoorTemperature", "measuredValue")
    model.add_connection(room, sensors[1], "indoorCO2", "measuredValue")
    return room, damper, sensors


def build_hallway(model_id):
    model = tb.Model(id=model_id)
    sensors = []
    for k, rid in enumerate(("A", "B")):
        room, _, s = _room(model, rid)
        sensors += s
    hallway = tb.BuildingSpaceSystem(id="H")
    for port, value in (("outdoorTemperature", 0.0), ("supplyAirTemperature", 20.0), ("globalIrradiation", 0.0),
                        ("numberOfPeople", 0.0), ("heatGain", 0.0), ("outdoorCo2Concentration", 400.0)):
        model.add_connection(_sched(value, f"H_{port}"), hallway, "scheduleValue", port)
    for k, rid in enumerate(("A", "B")):
        model.add_connection(model.components[rid], hallway, "surplusAirFlowRate", "supplyAirFlowRate", input_port_index=k)
    model.add_connection(_sched(0.01, "H_exh"), hallway, "scheduleValue", "exhaustAirFlowRate", input_port_index=0)
    meter = _series(0.05, "H_surplus")
    model.add_connection(hallway, meter, "surplusAirFlowRate", "measuredValue")
    return model, sensors + [meter]


def build_junction(model_id):
    model = tb.Model(id=model_id)
    sensors = []
    for rid in ("A", "B"):
        _, _, s = _room(model, rid)
        sensors += s
    junction = tb.SupplyFlowJunctionSystem(id="J")
    for k, rid in enumerate(("A", "B")):
        model.add_connection(model.components[f"{rid}_damper"], junction, "airFlowRate", "airFlowRateOut",
                             input_port_index=k)
    meter = _series(0.12, "J_flow")
    model.add_connection(junction, meter, "airFlowRateIn", "measuredValue")
    return model, sensors + [meter]


class TestHallway(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.sensors = build_hallway("signal_worlds_hallway")
        cls.worlds = split_worlds(cls.model, measured=cls.sensors, scored=cls.sensors)
        cls.partition = measured_partition(cls.model, measured=cls.sensors, scored=cls.sensors)

    def test_the_rooms_are_projected_and_their_dampers_copied(self):
        self.assertEqual(self.worlds.merged, {})
        self.assertEqual(self.worlds.projections, {"A": ["A__projection_in_H"], "B": ["B__projection_in_H"]})
        self.assertEqual(self.worlds.copies, {"A_damper": ["A_damper__in_H"], "B_damper": ["B_damper__in_H"]})
        self.assertIsInstance(self.model.components["A__projection_in_H"], tb.AirImbalanceSystem)

    def test_the_hallway_reads_the_projections_and_they_read_the_copies(self):
        self.assertEqual({s.id for s in senders(self.model.components["H"], "supplyAirFlowRate")},
                         {"A__projection_in_H", "B__projection_in_H"})
        self.assertEqual([s.id for s in senders(self.model.components["A__projection_in_H"], "supplyAirFlowRate")],
                         ["A_damper__in_H"])
        # the original damper still supplies its room
        self.assertEqual([s.id for s in senders(self.model.components["A"], "supplyAirFlowRate")], ["A_damper"])

    def test_three_worlds(self):
        g = self.partition.group_of
        self.assertEqual(len({g["A"], g["B"], g["H"]}), 3)
        self.assertEqual(g["A_damper"], g["A"])
        self.assertEqual(g["A_damper__in_H"], g["H"])
        self.assertEqual(g["A__projection_in_H"], g["H"])

    def test_only_what_the_copied_signal_depends_on_is_shared(self):
        shared = {a for (cid, a) in self.worlds.shared if cid == "A_damper"}
        self.assertEqual(shared, {"nominalAirFlowRate", "a"})  # c tied, the exhaust ratio not needed
        self.assertEqual(self.worlds.shared[("A_damper", "a")], [("A_damper", "a"), ("A_damper__in_H", "a")])

    def test_the_split_model_simulates_and_the_projection_matches_the_room(self):
        model, sensors = build_hallway("signal_worlds_hallway_sim")
        split_worlds(model, measured=sensors, scored=sensors)
        model.load(draw_semantic_model=False, draw_simulation_model=False)
        tb.Simulator(model).simulate(start_time=START, end_time=START + datetime.timedelta(hours=1), step_size=STEP,
                                     show_progress_bar=False)
        room = model.components["A"].output["surplusAirFlowRate"].history()
        projected = model.components["A__projection_in_H"].output["surplusAirFlowRate"].history()
        self.assertTrue(bool(((room - projected).abs() < 1e-12).all()))


class TestJunction(unittest.TestCase):
    def test_the_meter_s_world_copies_the_dampers(self):
        model, sensors = build_junction("signal_worlds_junction")
        worlds = split_worlds(model, measured=sensors, scored=sensors)
        self.assertEqual(worlds.merged, {})
        self.assertEqual(worlds.copies, {"A_damper": ["A_damper__in_J"], "B_damper": ["B_damper__in_J"]})
        self.assertEqual(worlds.projections, {})
        p = measured_partition(model, measured=sensors, scored=sensors)
        g = p.group_of
        self.assertEqual(len({g["A"], g["B"], g["J"]}), 3)
        self.assertEqual({s.id for s in senders(model.components["J"], "airFlowRateOut")},
                         {"A_damper__in_J", "B_damper__in_J"})

    def test_the_split_model_simulates(self):
        model, sensors = build_junction("signal_worlds_junction_sim")
        worlds = split_worlds(model, measured=sensors, scored=sensors)
        cut_measured_edges(model, measured_partition(model, measured=sensors, scored=sensors))
        model.load(draw_semantic_model=False, draw_simulation_model=False)
        tb.Simulator(model).simulate(start_time=START, end_time=START + datetime.timedelta(hours=1), step_size=STEP,
                                     show_progress_bar=False)
        total = model.components["J"].output["airFlowRateIn"].history()
        one = model.components["A_damper"].output["airFlowRate"].history()
        self.assertTrue(bool(((total - 2 * one).abs() < 1e-12).all()))  # two equal dampers, copies equal originals


if __name__ == "__main__":
    unittest.main()
