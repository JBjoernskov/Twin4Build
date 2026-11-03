"""Worlds (29 Oct 2026): a component that reaches the scored sensors of several groups is evaluated once per group.

Three rooms with scored sensors -- A and B with temperature and CO2, C with CO2 only -- and three walls:

* ``W_AB`` between A and B: both temperatures measured, so it is split into two worlds (A's, B's), each reading its
  own room simulated (a fusable arc) and the other room's measured temperature; its three parameters are shared by the
  two copies;
* ``W_AC`` between A and C: C's temperature is not measured, so the two worlds cannot be told apart and merge (the wall
  stays whole, in the merged group);
* ``W_out`` fed by schedules only, heating A and B: split too (it reads nothing from either world).

After the split, :func:`measured_partition` keeps A (with C) and B apart, and the model loads and simulates.
"""
import datetime
import unittest

import pandas as pd
from dateutil import tz

import twin4build as tb
from twin4build.model.partition import measured_partition, split_worlds

tb._IS_TESTING = True
START = datetime.datetime(2024, 1, 1, tzinfo=tz.UTC)
STEP = 600


def _sched(value, sid):
    return tb.ScheduleSystem(weekday_ruleset={"ruleset_default_value": value}, id=sid)


def _series(value, sid):
    index = pd.DatetimeIndex([START + datetime.timedelta(seconds=STEP * k) for k in range(12)], name="time")
    return tb.SensorSystem(df=pd.DataFrame({"value": [float(value)] * 12}, index=index), use_df=True, id=sid)


def _room(model, rid, temperature_sensor=True, skip=()):
    room = tb.BuildingSpaceSystem(
        thermal_kwargs=dict(C_air=2e6, C_wall=5e6, R_out=0.01, R_in=0.01, f_wall=0.0, f_air=0.0, Q_occ_gain=100.0),
        mass_kwargs=dict(V=100.0, G_occ=5e-6, m_inf=0.01),
        id=rid,
    )
    for port, value in (("outdoorTemperature", 0.0), ("supplyAirTemperature", 20.0), ("globalIrradiation", 0.0),
                        ("numberOfPeople", 0.0), ("heatGain", 500.0), ("outdoorCO2", 400.0)):
        if port in skip:
            continue
        model.add_connection(_sched(value, f"{rid}_{port}"), room, "scheduleValue", port)
    model.add_connection(_sched(0.1, f"{rid}_sup"), room, "scheduleValue", "supplyAirFlowRate", input_port_index=0)
    model.add_connection(_sched(0.1, f"{rid}_exh"), room, "scheduleValue", "exhaustAirFlowRate", input_port_index=0)
    sensors = []
    if temperature_sensor:
        sensors.append(_series(20.0, f"{rid}_T"))
        model.add_connection(room, sensors[-1], "indoorTemperature", "measuredValue")
    sensors.append(_series(400.0, f"{rid}_CO2"))
    model.add_connection(room, sensors[-1], "indoorCO2", "measuredValue")
    return room, sensors


def _wall(model, wid, a, b, slot_a, slot_b, temperature_a=None, temperature_b=None):
    wall = tb.WallSystem(C=2e5, R_a=0.02, R_b=0.02, id=wid)
    model.add_connection(temperature_a or a, wall, "scheduleValue" if temperature_a else "indoorTemperature", "temperatureA")
    model.add_connection(temperature_b or b, wall, "scheduleValue" if temperature_b else "indoorTemperature", "temperatureB")
    model.add_connection(wall, a, "heatFlowRateA", "wallHeatGain", input_port_index=slot_a)
    model.add_connection(wall, b, "heatFlowRateB", "wallHeatGain", input_port_index=slot_b)
    return wall


def build(model_id):
    model = tb.Model(id=model_id)
    a, sa = _room(model, "A")
    b, sb = _room(model, "B")
    c, sc = _room(model, "C", temperature_sensor=False)
    _wall(model, "W_AB", a, b, 0, 0)
    _wall(model, "W_AC", a, c, 1, 0)
    _wall(model, "W_out", a, b, 2, 1, temperature_a=_sched(5.0, "Out_A"), temperature_b=_sched(5.0, "Out_B"))
    return model, sa + sb + sc


def senders(comp, port):
    return [conn.connects_system for cp in comp.connects_at if cp.input_port == port for conn in cp.connects_system_through]


class TestSplitWorlds(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.sensors = build("split_worlds")
        cls.worlds = split_worlds(cls.model, measured=cls.sensors, scored=cls.sensors)
        cls.partition = measured_partition(cls.model, measured=cls.sensors, scored=cls.sensors)

    def test_what_is_split_and_what_merges(self):
        self.assertEqual(set(self.worlds.copies), {"W_AB", "W_out"})
        self.assertEqual(self.worlds.copies["W_AB"], ["W_AB__in_B"])
        self.assertEqual(self.worlds.merged, {"W_AC": ["A", "C"]})  # C's temperature unmeasured: one world, the wall whole
        self.assertNotIn("W_AC", self.worlds.copies)

    def test_the_groups_after_the_split(self):
        g = self.partition.group_of
        self.assertNotEqual(g["A"], g["B"])
        self.assertEqual(g["A"], g["C"])  # merged through W_AC
        self.assertEqual(g["W_AB"], g["A"])
        self.assertEqual(g["W_AB__in_B"], g["B"])
        self.assertEqual(g["W_out__in_B"], g["B"])

    def test_each_member_reads_its_own_room_and_the_other_measured(self):
        wall, copy_ = self.model.components["W_AB"], self.model.components["W_AB__in_B"]
        self.assertEqual([s.id for s in senders(wall, "temperatureA")], ["A"])
        self.assertEqual([s.id for s in senders(wall, "temperatureB")], ["B_T__world_replay"])
        self.assertEqual([s.id for s in senders(copy_, "temperatureA")], ["A_T__world_replay"])
        self.assertEqual([s.id for s in senders(copy_, "temperatureB")], ["B"])
        # each member heats its own room only
        self.assertEqual({s.id for s in senders(self.model.components["A"], "wallHeatGain")}, {"W_AB", "W_AC", "W_out"})
        self.assertEqual({s.id for s in senders(self.model.components["B"], "wallHeatGain")}, {"W_AB__in_B", "W_out__in_B"})

    def test_the_shared_parameters(self):
        shared = {k: v for k, v in self.worlds.shared.items() if k[0] == "W_AB"}
        self.assertEqual({a for _, a in shared}, {"C", "R_a", "R_b"})
        for (cid, attr), members in shared.items():
            self.assertEqual(members, [("W_AB", attr), ("W_AB__in_B", attr)])
        # the copies are copies: their own parameter objects, the same values
        self.assertIsNot(self.model.components["W_AB"].R_a, self.model.components["W_AB__in_B"].R_a)

    def test_the_split_model_loads_and_simulates(self):
        model, sensors = build("split_worlds_sim")
        split_worlds(model, measured=sensors, scored=sensors)
        model.load(draw_semantic_model=False, draw_simulation_model=False)
        tb.Simulator(model).simulate(start_time=START, end_time=START + datetime.timedelta(hours=1), step_size=STEP,
                                     show_progress_bar=False)
        t = model.components["B"].output["indoorTemperature"].history()
        self.assertTrue(bool((t == t).all()))


def build_data(model_id):
    """Rooms A and B (measured) take their heat gain from M, the max of room C's measured CO2 and a schedule: M has
    no parameters and reads only data."""
    model = tb.Model(id=model_id)
    a, sa = _room(model, "A", skip=("heatGain",))
    b, sb = _room(model, "B", skip=("heatGain",))
    c, sc = _room(model, "C", temperature_sensor=False)
    m = tb.MaxSystem(id="M")
    model.add_connection(c, m, "indoorCO2", "inputs", input_port_index=0)
    model.add_connection(_sched(500.0, "M_floor"), m, "scheduleValue", "inputs", input_port_index=1)
    model.add_connection(m, a, "value", "heatGain")
    model.add_connection(m, b, "value", "heatGain")
    return model, sa + sb + sc


class TestDataIsNotSplit(unittest.TestCase):
    """2 Nov 2026: a component without free parameters that reads only data (here a measured signal and a schedule)
    is exogenous: evaluated once and read by every world, not copied per world, and it ties no rooms together."""

    def test_a_function_of_data_is_one_component_read_by_every_world(self):
        model, sensors = build_data("data_not_split")
        worlds = split_worlds(model, measured=sensors, scored=sensors)
        self.assertEqual(worlds.once, ["M"])
        self.assertNotIn("M", worlds.copies)
        self.assertNotIn("M__in_B", model.components)
        # it reads the replay of C's measured CO2 (and its schedule), and both rooms read it
        self.assertEqual({s.id for s in senders(model.components["M"], "inputs")}, {"C_CO2__world_replay", "M_floor"})
        self.assertEqual({s.id for s in senders(model.components["B"], "heatGain")}, {"M"})
        p = measured_partition(model, measured=sensors, scored=sensors)
        g = p.group_of
        self.assertEqual(len({g["A"], g["B"], g["C"], g["M"]}), 4)

    def test_the_split_model_with_a_function_of_data_simulates(self):
        model, sensors = build_data("data_not_split_sim")
        split_worlds(model, measured=sensors, scored=sensors)
        model.load(draw_semantic_model=False, draw_simulation_model=False)
        tb.Simulator(model).simulate(start_time=START, end_time=START + datetime.timedelta(hours=1), step_size=STEP,
                                     show_progress_bar=False)
        t = model.components["A"].output["indoorTemperature"].history()
        self.assertTrue(bool((t == t).all()))

    def test_a_wall_without_free_parameters_between_measured_rooms_is_still_split(self):
        # fusable inputs are never data: each world keeps its own room simulated
        model, sensors = build("wall_without_parameters")
        worlds = split_worlds(model, free=["A", "B", "C"], measured=sensors, scored=sensors)
        self.assertIn("W_AB", worlds.copies)
        self.assertNotIn("W_out", worlds.copies)  # reads two schedules only: exogenous
        self.assertEqual(worlds.shared, {})


def _first_entry(functional_model, cid):
    idx = next(iter(functional_model.theta_by_comp[cid].values()))
    return idx.start if isinstance(idx, slice) else (int(idx[0]) if isinstance(idx, (list, tuple)) else int(idx))


class TestDataCouplesNoBlocks(unittest.TestCase):
    """2 Nov 2026: the estimator's structure walk (index_coupling) leaves out what a component no parameter reaches
    sends on: M, evaluated once on data, feeds A and B, and A and B stay separate blocks of the solver."""

    def test_rooms_fed_by_one_function_of_data_are_separate_blocks(self):
        import torch

        from twin4build.model.partition import cut_measured_edges

        model, sensors = build_data("data_blocks")
        split_worlds(model, measured=sensors, scored=sensors)
        cut_measured_edges(model, measured_partition(model, measured=sensors, scored=sensors))
        model.load(draw_semantic_model=False, draw_simulation_model=False)
        model.initialize(start_time=[START], end_time=[START + datetime.timedelta(hours=1)], step_size=STEP)
        theta_spec = [(model.components[cid], attr) for cid in ("A", "B")
                      for _, attr, _, _, _ in model.components[cid].get_estimable_parameters()]
        measurements = [s for s in sensors if s.id in model.components]
        simulator = tb.Simulator(model, execution_mode="functional", execution_backend="eager")
        _, functional_model = simulator.build_functional_model(theta_spec=theta_spec, measurements=measurements,
                                                               step_size=STEP)
        functional_model.prepare_routes(torch.device("cpu"))
        theta_block, _column_block, n_blocks = functional_model.index_coupling()
        self.assertGreaterEqual(n_blocks, 2)
        self.assertNotEqual(theta_block[_first_entry(functional_model, "A")],
                            theta_block[_first_entry(functional_model, "B")])


if __name__ == "__main__":
    unittest.main()
