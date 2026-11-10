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

A lagged hub (a building's hidden interior node) that one world reads only through measurements is provided from
that world: the other worlds read it through a gradient barrier instead of copies of it and its walls.
"""

import datetime
import unittest

import pandas as pd
import torch
from dateutil import tz

import twin4build as tb
from twin4build.model.partition import cut_measured_edges, measured_partition, split_worlds
from twin4build.systems.utils.gradient_barrier_system import GradientBarrierSystem
from twin4build.tests.model.test_split_worlds import START, STEP, _sched, _series, senders

tb._IS_TESTING = True


def _room(model, rid):
    room = tb.BuildingSpaceSystem(
        thermal=dict(C_air=2e6, C_wall=5e6, R_out=0.01, R_in=0.01, f_wall=0.0, f_air=0.0, Q_occ_gain=100.0),
        mass=dict(V=100.0, G_occ=5e-6, m_inf=0.01),
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

    def test_the_batched_projections_keep_one_value_per_copy(self):
        # The two rooms' projections batch into one AirImbalanceSystem: every port of it holds
        # both copies (a downstream run once saw the exhaust vector allocated for one) and each copy its room's
        # imbalance.  B's damper is larger, so the copies differ.
        def split(model_id):
            model, sensors = build_hallway(model_id)
            model.components["B_damper"].nominalAirFlowRate.set(0.2, normalized=False)
            split_worlds(model, measured=sensors, scored=sensors)
            model.load(draw_semantic_model=False, draw_simulation_model=False)
            return model

        span = dict(start_time=START, end_time=START + datetime.timedelta(hours=1), step_size=STEP,
                    show_progress_bar=False)
        model = split("signal_worlds_hallway_unbatched")
        tb.Simulator(model).simulate(**span)
        source = split("signal_worlds_hallway_batched")
        batched = source.batch_components()
        batched.load(draw_semantic_model=False, draw_simulation_model=False)
        meta, _ = source.get_batched_component_info("A__projection_in_H")
        self.assertIsInstance(meta, tb.AirImbalanceSystem)
        self.assertEqual(meta._n_c_batched, 2)
        for mode in ("object", "functional"):
            with self.subTest(mode=mode):
                tb.Simulator(batched, execution_mode=mode).simulate(**span)
                for name, port in {**meta.input, **meta.output}.items():
                    self.assertEqual(port.n_c, 2, name)
                for rid in ("A", "B"):
                    _, i_c = source.get_batched_component_info(f"{rid}__projection_in_H")
                    room = model.components[rid].output["surplusAirFlowRate"]._history[..., 0]
                    torch.testing.assert_close(meta.output["surplusAirFlowRate"]._history[..., i_c], room)


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


def build_node(model_id, with_meter=True, with_draw=False):
    """Rooms A and B (scored), each with a core wall to one hidden node N (a lagged :class:`ThermalMassNodeSystem`
    with a free capacity); with ``with_meter`` a space M, scored, takes N's temperature as its supply air (a unit's
    return air drawn from a building's hidden interior): M's world reads N only through A's and B's measurements.
    With ``with_draw`` N also reads A's surplus air (unmeasured, state-free: as the node reads a box's draw)."""
    model = tb.Model(id=model_id)
    sensors = []
    node = tb.ThermalMassNodeSystem(C=1e8, T_init=20.0, id="N")
    for k, rid in enumerate(("A", "B")):
        room, _, s = _room(model, rid)
        sensors += s
        wall = tb.WallSystem(C=2e5, R_a=0.02, R_b=0.02, id=f"W_{rid}")
        model.add_connection(room, wall, "indoorTemperature", "temperatureA")
        model.add_connection(node, wall, "temperature", "temperatureB")
        model.add_connection(wall, room, "heatFlowRateA", "wallHeatGain", input_port_index=0)
        model.add_connection(wall, node, "heatFlowRateB", "heatFlowRate", input_port_index=k)
    if with_draw:
        model.add_connection(model.components["A"], node, "surplusAirFlowRate", "heatFlowRate", input_port_index=2)
    if with_meter:
        meter = tb.BuildingSpaceSystem(id="M")
        for port, value in (("outdoorTemperature", 0.0), ("globalIrradiation", 0.0), ("numberOfPeople", 0.0),
                            ("heatGain", 0.0), ("outdoorCo2Concentration", 400.0)):
            model.add_connection(_sched(value, f"M_{port}"), meter, "scheduleValue", port)
        model.add_connection(_sched(0.1, "M_sup"), meter, "scheduleValue", "supplyAirFlowRate", input_port_index=0)
        model.add_connection(_sched(0.1, "M_exh"), meter, "scheduleValue", "exhaustAirFlowRate", input_port_index=0)
        model.add_connection(node, meter, "temperature", "supplyAirTemperature")
        sensors.append(_series(20.0, "M_T"))
        model.add_connection(meter, sensors[-1], "indoorTemperature", "measuredValue")
    return model, sensors


class TestProvidedNode(unittest.TestCase):
    """7 Nov 2026: a lagged hub that one world reads only through measurements lives there (provided); the rooms'
    worlds read it through a gradient barrier, not copies of it and of every wall it has."""

    @classmethod
    def setUpClass(cls):
        cls.model, cls.sensors = build_node("signal_worlds_node")
        cls.worlds = split_worlds(cls.model, measured=cls.sensors, scored=cls.sensors)
        cls.partition = measured_partition(cls.model, measured=cls.sensors, scored=cls.sensors)

    def test_the_node_is_provided_from_the_meter_s_world(self):
        self.assertEqual(self.worlds.merged, {})
        self.assertEqual(self.worlds.provided, {"N": ["N__temperature__barrier"]})
        self.assertNotIn("N", self.worlds.copies)
        # each core wall twice: in its room's world (the original) and in the node's (a copy on the measured room)
        self.assertEqual(self.worlds.copies, {"W_A": ["W_A__in_M"], "W_B": ["W_B__in_M"]})

    def test_the_rooms_read_the_barrier_and_the_node_reads_the_measured_rooms(self):
        c = self.model.components
        self.assertEqual([s.id for s in senders(c["W_A"], "temperatureB")], ["N__temperature__barrier"])
        self.assertEqual([s.id for s in senders(c["N__temperature__barrier"], "value")], ["N"])
        self.assertEqual([s.id for s in senders(c["W_A__in_M"], "temperatureB")], ["N"])
        self.assertEqual([s.id for s in senders(c["W_A__in_M"], "temperatureA")], ["A_T__world_replay"])
        self.assertEqual({s.id for s in senders(c["N"], "heatFlowRate")}, {"W_A__in_M", "W_B__in_M"})

    def test_three_groups_and_the_barrier_ties_none(self):
        g = self.partition.group_of
        self.assertEqual(len({g["A"], g["B"], g["M"]}), 3)
        self.assertEqual(g["N"], g["M"])
        self.assertEqual(g["W_A__in_M"], g["M"])
        self.assertEqual(g["W_A"], g["A"])
        self.assertEqual(g["N__temperature__barrier"], g["N"])  # with what it passes on; its readers apart

    def test_the_split_model_simulates_and_the_barrier_passes_the_node_on(self):
        model, sensors = build_node("signal_worlds_node_sim")
        split_worlds(model, measured=sensors, scored=sensors)
        cut_measured_edges(model, measured_partition(model, measured=sensors, scored=sensors))
        model.load(draw_semantic_model=False, draw_simulation_model=False)
        tb.Simulator(model).simulate(start_time=START, end_time=START + datetime.timedelta(hours=1), step_size=STEP,
                                     show_progress_bar=False)
        node = model.components["N"].output["temperature"].history()
        passed = model.components["N__temperature__barrier"].output["value"].history()
        self.assertTrue(bool(torch.isfinite(passed).all()))
        self.assertLess(float((node - passed).abs().max()), 1.0)  # the node's value (start of step), not another signal

    def test_the_solver_s_blocks_stay_apart(self):
        """The estimator's structure walk (index_coupling) does not join the rooms' blocks to the node's through the
        barrier: no parameter reaches what a barrier sends."""
        model, sensors = build_node("signal_worlds_node_blocks")
        split_worlds(model, measured=sensors, scored=sensors)
        cut_measured_edges(model, measured_partition(model, measured=sensors, scored=sensors))
        # unfused: the theta spec names the rooms and walls themselves (fused, a block would hold them)
        model.load(draw_semantic_model=False, draw_simulation_model=False, enable_fusion=False)
        model.initialize(start_time=[START], end_time=[START + datetime.timedelta(hours=1)], step_size=STEP)
        # every estimable parameter
        theta_spec = [(comp, attr) for comp in model.components.values() if hasattr(comp, "get_estimable_parameters")
                      for _, attr, _, _, _ in comp.get_estimable_parameters()]
        measurements = [s for s in sensors if s.id in model.components]
        simulator = tb.Simulator(model, execution_mode="functional", execution_backend="eager")
        _, functional_model = simulator.build_functional_model(theta_spec=theta_spec, measurements=measurements,
                                                               step_size=STEP)
        functional_model.prepare_routes(torch.device("cpu"))
        theta_block, _column_block, _n_blocks = functional_model.index_coupling()
        # A with its wall, B with its wall, and the node's world (N, M, the walls' copies): three blocks
        self.assertEqual(len({int(b) for b in theta_block}), 3, sorted(functional_model.theta_by_comp))

    def test_the_barrier_stops_the_gradient(self):
        x = torch.tensor([[21.0]], requires_grad=True)
        _, out = GradientBarrierSystem(id="b").forward(None, {"value": x}, {}, 600.0)
        self.assertEqual(float(out["value"]), 21.0)
        self.assertFalse(out["value"].requires_grad)

    def test_another_world_s_state_free_output_keeps_the_home(self):
        """7 Nov 2026: a large building's node reads the central box's draw (unmeasured, the box a world of its own
        by its meter); the box's draw is computed in the home from what it reads, so the home stays clean."""
        model, sensors = build_node("signal_worlds_node_draw", with_draw=True)
        worlds = split_worlds(model, measured=sensors, scored=sensors)
        self.assertEqual(worlds.merged, {})
        self.assertEqual(worlds.provided, {"N": ["N__temperature__barrier"]})
        self.assertNotIn("N", worlds.copies)

    def test_without_a_world_that_reads_it_through_measurements_it_is_copied(self):
        model, sensors = build_node("signal_worlds_node_alone", with_meter=False)
        worlds = split_worlds(model, measured=sensors, scored=sensors)
        self.assertEqual(worlds.provided, {})
        self.assertIn("N", worlds.copies)


if __name__ == "__main__":
    unittest.main()
