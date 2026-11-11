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
that world: the other worlds read it through cut connections (the value without its gradient) instead of copies of
it and its walls.
"""

import datetime
import unittest

import pandas as pd
import torch
from dateutil import tz

import twin4build as tb
from twin4build.model.partition import cut_measured_edges, measured_partition, split_worlds
from twin4build.utils.rgetattr import rgetattr
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


#: The node's cut connections: the rooms' walls read its temperature without its gradient.
CUTS = [("temperature", "W_A", "temperatureB"), ("temperature", "W_B", "temperatureB")]


def _cut(receiver, port):
    """``ConnectionPoint.detached_of`` of every connection into ``receiver.port``."""
    return [cp.detached_of(conn) for cp in receiver.connects_at if cp.input_port == port
            for conn in cp.connects_system_through]


class TestProvidedNode(unittest.TestCase):
    """7 Nov 2026: a lagged hub that one world reads only through measurements lives there (provided); the rooms'
    worlds read it through cut connections (the value without its gradient), not copies of it and of every wall it
    has."""

    @classmethod
    def setUpClass(cls):
        cls.model, cls.sensors = build_node("signal_worlds_node")
        cls.worlds = split_worlds(cls.model, measured=cls.sensors, scored=cls.sensors)
        cls.partition = measured_partition(cls.model, measured=cls.sensors, scored=cls.sensors)

    def test_the_node_is_provided_from_the_meter_s_world(self):
        self.assertEqual(self.worlds.merged, {})
        self.assertEqual(sorted(self.worlds.provided["N"]), CUTS)
        self.assertNotIn("N", self.worlds.copies)
        # each core wall twice: in its room's world (the original) and in the node's (a copy on the measured room)
        self.assertEqual(self.worlds.copies, {"W_A": ["W_A__in_M"], "W_B": ["W_B__in_M"]})

    def test_the_rooms_read_the_node_cut_and_the_node_reads_the_measured_rooms(self):
        c = self.model.components
        self.assertEqual([s.id for s in senders(c["W_A"], "temperatureB")], ["N"])
        self.assertEqual(_cut(c["W_A"], "temperatureB"), [True])
        # the node's own world reads it with its gradient
        self.assertEqual([s.id for s in senders(c["W_A__in_M"], "temperatureB")], ["N"])
        self.assertEqual(_cut(c["W_A__in_M"], "temperatureB"), [False])
        self.assertEqual([s.id for s in senders(c["W_A__in_M"], "temperatureA")], ["A_T__world_replay"])
        self.assertEqual({s.id for s in senders(c["N"], "heatFlowRate")}, {"W_A__in_M", "W_B__in_M"})

    def test_three_groups_and_the_cut_ties_none(self):
        g = self.partition.group_of
        self.assertEqual(len({g["A"], g["B"], g["M"]}), 3)
        self.assertEqual(g["N"], g["M"])
        self.assertEqual(g["W_A__in_M"], g["M"])
        self.assertEqual(g["W_A"], g["A"])  # it reads the node as data

    def test_the_split_model_simulates_and_the_cut_passes_the_node_on(self):
        model, sensors = build_node("signal_worlds_node_sim")
        split_worlds(model, measured=sensors, scored=sensors)
        cut_measured_edges(model, measured_partition(model, measured=sensors, scored=sensors))
        model.load(draw_semantic_model=False, draw_simulation_model=False)
        tb.Simulator(model).simulate(start_time=START, end_time=START + datetime.timedelta(hours=1), step_size=STEP,
                                     show_progress_bar=False)
        node = model.components["N"].output["temperature"].history()
        passed = model.components["W_A"].input["temperatureB"].history()
        self.assertTrue(bool(torch.isfinite(passed).all()))
        torch.testing.assert_close(passed, node, rtol=0, atol=0)  # the node's value (start of step), every step

    def _functional(self, model_id, batched=False):
        """The split model's functional map over every estimable parameter (unfused: the theta spec names the rooms
        and walls themselves; fused, a block would hold them), its routes prepared; with ``batched``, of the model
        batched (the walls and their copies in one meta, which reads the node cut for two instances and not for
        the other two)."""
        model, sensors = build_node(model_id)
        split_worlds(model, measured=sensors, scored=sensors)
        cut_measured_edges(model, measured_partition(model, measured=sensors, scored=sensors))
        model.load(draw_semantic_model=False, draw_simulation_model=False, enable_fusion=False)
        if batched:
            source, model = model, model.batch_components()
            model.load(draw_semantic_model=False, draw_simulation_model=False, enable_fusion=False)
        model.initialize(start_time=[START], end_time=[START + datetime.timedelta(hours=1)], step_size=STEP)
        theta_spec, t = [], 0
        for comp in model.components.values():
            for _, attr, _, _, _ in (comp.get_estimable_parameters() if hasattr(comp, "get_estimable_parameters") else ()):
                n = int(rgetattr(comp, attr).get().numel())  # a meta's: one per instance
                theta_spec.append((comp, attr, t if n == 1 else list(range(t, t + n))))
                t += n
        measurements = [s for s in model.components.values() if isinstance(s, tb.SensorSystem) and s.has_data]
        simulator = tb.Simulator(model, execution_mode="functional", execution_backend="eager")
        layout, functional_model = simulator.build_functional_model(theta_spec=theta_spec, measurements=measurements,
                                                                    step_size=STEP)
        functional_model.prepare_routes(torch.device("cpu"))
        return model, simulator, layout, functional_model, theta_spec

    def test_the_solver_s_blocks_stay_apart(self):
        """The estimator's structure walk (index_coupling) does not join the rooms' blocks to the node's through the
        cut connections: no parameter reaches the rooms through them."""
        _, _, _, functional_model, _ = self._functional("signal_worlds_node_blocks")
        theta_block, _column_block, _n_blocks = functional_model.index_coupling()
        # A with its wall, B with its wall, and the node's world (N, M, the walls' copies): three blocks
        self.assertEqual(len({int(b) for b in theta_block}), 3, sorted(functional_model.theta_by_comp))

    def test_the_cut_stops_the_gradient(self):
        """Through the functional engine: the rooms' walls read the node's temperature, but the node's capacity
        does not reach them; the walls' copies, in the node's world, get its gradient."""
        model, sensors = build_node("signal_worlds_node_gradient")
        model.components["N"].T_init = 15.0  # colder than the walls: heat flows, its capacity matters
        split_worlds(model, measured=sensors, scored=sensors)
        cut_measured_edges(model, measured_partition(model, measured=sensors, scored=sensors))
        model.load(draw_semantic_model=False, draw_simulation_model=False, enable_fusion=False)
        end = START + datetime.timedelta(hours=1)
        model.initialize([START], [end], [STEP])
        c = model.components
        simulator = tb.Simulator(model, execution_mode="functional")
        layout, functional = simulator.build_functional_model(
            theta_spec=[(c["N"], "C", 0)], outputs=[(c[w], "heatFlowRateB") for w in ("W_A", "W_A__in_M")],
            step_size=[STEP])
        tape = simulator.record_exogenous_inputs(functional, [START], [end], [STEP], layout=layout)
        theta = c["N"].C.get().detach().reshape(-1)[:1].to(torch.float64).clone().requires_grad_(True)
        out = simulator.rollout_functional(functional, tape.Y0[0], theta, tape.exogenous_tape[0], transform_mode=True)
        columns = out.reshape(out.shape[0], -1)
        grads = [torch.autograd.grad(columns[:, j].sum(), theta, retain_graph=True, allow_unused=True)[0]
                 for j in range(2)]
        self.assertTrue(grads[0] is None or float(grads[0].abs().sum()) == 0.0)  # the room's wall: cut
        self.assertNotEqual(float(grads[1].abs().sum()), 0.0)  # the copy in the node's world: it reaches it

    def test_the_batched_split_model_steps_as_the_unbatched_one(self):
        """Batched, the rooms' walls are one meta that reads the node cut, their copies another that reads it with
        its gradient: both engines step as the unbatched model, and the blocks stay apart."""
        model, sensors = build_node("signal_worlds_node_batched_cut")
        split_worlds(model, measured=sensors, scored=sensors)
        cut_measured_edges(model, measured_partition(model, measured=sensors, scored=sensors))
        model.load(draw_semantic_model=False, draw_simulation_model=False, enable_fusion=False)
        span = dict(start_time=START, end_time=START + datetime.timedelta(hours=1), step_size=STEP,
                    show_progress_bar=False)
        tb.Simulator(model).simulate(**span)
        batched = model.batch_components()
        batched.load(draw_semantic_model=False, draw_simulation_model=False, enable_fusion=False)
        self.assertEqual(_cut(model.get_batched_component_info("W_A")[0], "temperatureB"), [True])
        self.assertEqual(_cut(model.get_batched_component_info("W_A__in_M")[0], "temperatureB"), [False])
        for mode in ("object", "functional"):
            with self.subTest(mode=mode):
                tb.Simulator(batched, execution_mode=mode).simulate(**span)
                for w in ("W_A", "W_A__in_M"):
                    meta_w, i_c = model.get_batched_component_info(w)
                    torch.testing.assert_close(meta_w.output["heatFlowRateA"].history(i_c=i_c).reshape(-1),
                                               model.components[w].output["heatFlowRateA"].history().reshape(-1))
        _, _, _, functional_model, _ = self._functional("signal_worlds_node_batched_blocks", batched=True)
        theta_block, _column_block, _n_blocks = functional_model.index_coupling()
        self.assertEqual(len({int(b) for b in theta_block}), 3)

    def test_a_batched_meta_holds_one_flag_per_pair(self):
        """Two walls on one node, one reading it cut, one not, batch into one meta: its connection from the node
        carries one flag per pair; both engines step as the unbatched model, and the node's capacity reaches the
        uncut wall only."""
        def build(model_id):
            model = tb.Model(id=model_id)
            node = tb.ThermalMassNodeSystem(C=1e8, T_init=20.0, id="N")
            for k, (wid, cut) in enumerate((("W1", True), ("W2", False))):
                wall = tb.WallSystem(C=2e5, R_a=0.02, R_b=0.02, id=wid)
                model.add_connection(_sched(22.0 + 2.0 * k, f"{wid}_a"), wall, "scheduleValue", "temperatureA")
                model.add_connection(node, wall, "temperature", "temperatureB", detached=cut)
                model.add_connection(wall, node, "heatFlowRateB", "heatFlowRate", input_port_index=k)
            model.load(draw_semantic_model=False, draw_simulation_model=False, enable_fusion=False)
            return model

        end = START + datetime.timedelta(hours=1)
        span = dict(start_time=START, end_time=end, step_size=STEP, show_progress_bar=False)
        model = build("signal_worlds_cut_pair")
        tb.Simulator(model).simulate(**span)
        batched = build("signal_worlds_cut_pair_source").batch_components()
        source = batched._batch_source_model
        batched.load(draw_semantic_model=False, draw_simulation_model=False, enable_fusion=False)
        meta, _ = source.get_batched_component_info("W1")
        self.assertEqual(meta._n_c_batched, 2)
        (cut,) = _cut(meta, "temperatureB")
        index = next(cp.input_component_index[conn] for cp in meta.connects_at if cp.input_port == "temperatureB"
                     for conn in cp.connects_system_through)
        flags = {int(i): bool(f) for i, f in zip(index.tolist(), cut.tolist())}
        self.assertEqual(flags, {source.get_batched_component_info("W1")[1]: True,
                                 source.get_batched_component_info("W2")[1]: False})
        for mode in ("object", "functional"):
            with self.subTest(mode=mode):
                tb.Simulator(batched, execution_mode=mode).simulate(**span)
                for w in ("W1", "W2"):
                    i_c = source.get_batched_component_info(w)[1]
                    torch.testing.assert_close(meta.output["heatFlowRateA"].history(i_c=i_c).reshape(-1),
                                               model.components[w].output["heatFlowRateA"].history().reshape(-1))
        batched.initialize([START], [end], [STEP])
        simulator = tb.Simulator(batched, execution_mode="functional")
        node = batched.components["N"]
        layout, functional = simulator.build_functional_model(
            theta_spec=[(node, "C", 0)], outputs=[(meta, "heatFlowRateA")], step_size=[STEP])
        tape = simulator.record_exogenous_inputs(functional, [START], [end], [STEP], layout=layout)
        theta = node.C.get().detach().reshape(-1)[:1].to(torch.float64).clone().requires_grad_(True)
        out = simulator.rollout_functional(functional, tape.Y0[0], theta, tape.exogenous_tape[0], transform_mode=True)
        columns = out.reshape(out.shape[0], -1)
        grads = {}
        for w in ("W1", "W2"):
            g = torch.autograd.grad(columns[:, source.get_batched_component_info(w)[1]].sum(), theta,
                                    retain_graph=True, allow_unused=True)[0]
            grads[w] = 0.0 if g is None else float(g.abs().sum())
        self.assertEqual(grads["W1"], 0.0)
        self.assertNotEqual(grads["W2"], 0.0)

    def test_a_saved_model_keeps_the_cut(self):
        model, sensors = build_node("signal_worlds_node_saved")
        split_worlds(model, measured=sensors, scored=sensors)
        model.load(draw_semantic_model=False, draw_simulation_model=False, enable_fusion=False)
        path = model.serialize()
        again = tb.Model(id="signal_worlds_node_saved_again")
        again.load(filename=path, draw_semantic_model=False, draw_simulation_model=False, enable_fusion=False)
        c = again.components
        self.assertEqual(_cut(c["W_A"], "temperatureB"), [True])
        self.assertEqual(_cut(c["W_A__in_M"], "temperatureB"), [False])

    def test_another_world_s_state_free_output_keeps_the_home(self):
        """7 Nov 2026: a large building's node reads the central box's draw (unmeasured, the box a world of its own
        by its meter); the box's draw is computed in the home from what it reads, so the home stays clean."""
        model, sensors = build_node("signal_worlds_node_draw", with_draw=True)
        worlds = split_worlds(model, measured=sensors, scored=sensors)
        self.assertEqual(worlds.merged, {})
        self.assertEqual(sorted(worlds.provided["N"]), CUTS)
        self.assertNotIn("N", worlds.copies)

    def test_without_a_world_that_reads_it_through_measurements_it_is_copied(self):
        model, sensors = build_node("signal_worlds_node_alone", with_meter=False)
        worlds = split_worlds(model, measured=sensors, scored=sensors)
        self.assertEqual(worlds.provided, {})
        self.assertIn("N", worlds.copies)


if __name__ == "__main__":
    unittest.main()
