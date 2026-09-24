"""The measured partition: groups bounded by measured signals, and the cut.

A factored unit feeding two rooms through two terminals, with data sensors
on the terminals' flows, the rooms' temperatures and the unit's supply
temperature.  Every connection between a terminal, a room and the unit is
then measured, so the minimal groups are one per terminal, one per room and
one for the unit with its junctions; the cut replaces the crossing edges by
replay leaves and the estimator's structure walk finds one block per group
that carries parameters.  With the terminals' exhaust flows declared as a
fixed ratio of their measured supply flows (``derived``), the terminals
leave the unit's group too.
"""
import datetime
import unittest

import pandas as pd
import torch
from dateutil import tz

import twin4build as tb
import twin4build.core as core
import twin4build.utils.types as tps
from twin4build.model.partition import cut_measured_edges, measured_partition
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

tb._IS_TESTING = True

START = datetime.datetime(2024, 1, 1, tzinfo=tz.UTC)
END = START + datetime.timedelta(minutes=40)
STEP = 600
RATIO = 0.9
FAN_KWARGS = {
    "nominalPowerRate": 1000.0, "nominalAirFlowRate": 1.0, "c1": 0.0, "c2": 0.2,
    "c3": 0.8, "c4": 0.0, "f_total": 1.0,
}


def _n_t(start_time, end_time, step_size):
    _, _, n_t, _ = core.Simulator.get_simulation_timesteps(start_time, end_time, step_size)
    return n_t


def series(values, **kwargs):
    values = [float(v) for v in values]
    index = pd.DatetimeIndex([START + datetime.timedelta(seconds=STEP * k) for k in range(len(values))], name="time")
    return tb.SensorSystem(df=pd.DataFrame({"value": values}, index=index), use_df=True, **kwargs)


def data_sensor(id_):
    """A data-bearing sensor that also reads a port (a scored measurement)."""
    return series([20.0] * 8, id=id_)


class Room(core.System):
    """``T <- T + k (sum flows) + 0.01 (T_supply - T)``: a state, a parameter."""

    def __init__(self, k=1.0, **kwargs):
        super().__init__(**kwargs)
        self.input = {"flow": tps.Vector(), "supplyT": tps.Scalar()}
        self.output = {"T": tps.Scalar()}
        self.k = tps.Parameter(torch.tensor(float(k)), min_value=0.0, max_value=10.0)
        self.parameter = {"k": {"lb": 0.0, "ub": 10.0}}
        self.T = tps.State(n_v=1, init_value=20.0, names=[f"{self.id}.T"])
        self._config = {"parameters": ["k"]}

    @property
    def config(self):
        return self._config

    def initialize(self, start_time, end_time, step_size):
        n_t = _n_t(start_time, end_time, step_size)
        indices = [
            int(i)
            for cp in self.connects_at
            if cp.input_port == "flow"
            for conn in cp.connects_system_through
            for i in torch.as_tensor(cp.input_port_index[conn]).reshape(-1).tolist()
        ]
        n_v = max(indices, default=-1) + 1
        self.k = self.k.expand_to_n_c(self.n_c)
        self.input["flow"].initialize(n_t=n_t, n_s=len(start_time), n_c=self.n_c, n_v=n_v)
        self.input["supplyT"].initialize(n_t=n_t, n_s=len(start_time), n_c=self.n_c)
        self.output["T"].initialize(n_t=n_t, n_s=len(start_time), n_c=self.n_c)
        self.T.initialize(n_s=len(start_time), n_c=self.n_c, n_v=1, force=True)

    PARAM_NAMES = ("k",)

    def forward(self, x, inputs, params, sample_time):
        k = params["k"].reshape(-1, 1)
        x_next = x + k * inputs["flow"].sum(dim=-1, keepdim=True) + 0.01 * (inputs["supplyT"].reshape(-1, 1) - x)
        return x_next, {"T": x_next[..., 0]}

    def do_step(self, second_time, date_time, step_size, step_index):
        inputs = {"flow": self.input["flow"].get(), "supplyT": self.input["supplyT"].get()}
        x_next, outs = self.forward(self.get_state(), inputs, self._forward_params(), step_size)
        self.set_state(x_next)
        self.output["T"]._set(outs["T"], i_t=step_index)


class Gain(core.System):
    """``u = g (setpoint - T)``: a proportional controller, one parameter."""

    def __init__(self, g=0.1, **kwargs):
        super().__init__(**kwargs)
        self.input = {"T": tps.Scalar()}
        self.output = {"u": tps.Scalar()}
        self.g = tps.Parameter(torch.tensor(float(g)), min_value=0.0, max_value=10.0)
        self.parameter = {"g": {"lb": 0.0, "ub": 10.0}}
        self._config = {"parameters": ["g"]}

    @property
    def config(self):
        return self._config

    def initialize(self, start_time, end_time, step_size):
        n_t = _n_t(start_time, end_time, step_size)
        self.g = self.g.expand_to_n_c(self.n_c)
        self.input["T"].initialize(n_t=n_t, n_s=len(start_time), n_c=self.n_c)
        self.output["u"].initialize(n_t=n_t, n_s=len(start_time), n_c=self.n_c)

    PARAM_NAMES = ("g",)

    def forward(self, x, inputs, params, sample_time):
        u = params["g"].reshape(-1) * (21.0 - inputs["T"].reshape(-1))
        return x, {"u": u}

    def do_step(self, second_time, date_time, step_size, step_index):
        _, outs = self.forward(None, {"T": self.input["T"].get()}, self._forward_params(), step_size)
        self.output["u"]._set(outs["u"], i_t=step_index)


def build():
    model = tb.Model(id="partition_toy")
    pos = [series([1.0, 0.5, 0.3, 0.7, 1.0, 0.6, 0.2, 0.9], id=f"pos{k}") for k in range(2)]
    outdoor = series([5.0] * 8, id="outdoor")
    setpoint = series([18.0] * 8, id="setpoint")
    dampers = [DamperSystem(id=f"terminal{k}", a=1.0, nominalAirFlowRate=0.5, exhaustFlowRatio=RATIO) for k in range(2)]
    rooms = [Room(id=f"room{k}") for k in range(2)]
    sj = SupplyFlowJunctionSystem(id="supply_junction")
    rj = ReturnFlowJunctionSystem(id="return_junction")
    unit = AirHandlingUnitCoreSystem(id="unit", supply_fan_kwargs=dict(FAN_KWARGS), exhaust_fan_kwargs=dict(FAN_KWARGS))
    for k in range(2):
        d, r = dampers[k], rooms[k]
        model.add_connection(pos[k], d, "measuredValue", "damperPosition")
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


def _blocks(model):
    """The estimator's own structure walk on ``model``."""
    model.initialize(start_time=[START], end_time=[END], step_size=STEP)
    theta_spec = [
        (comp, attr)
        for comp in model.components.values()
        if hasattr(comp, "get_estimable_parameters")
        for _, attr, _, _, _ in comp.get_estimable_parameters()
    ]
    measurements = [
        c for c in model.components.values()
        if isinstance(c, tb.SensorSystem) and c.has_data and any(cp.connects_system_through for cp in c.connects_at)
    ]
    simulator = tb.Simulator(model, execution_mode="functional", execution_backend="eager")
    _, functional_model = simulator.build_functional_model(
        theta_spec=theta_spec, measurements=measurements, step_size=STEP
    )
    functional_model.prepare_routes(torch.device("cpu"))
    theta_block, column_block, n_blocks = functional_model.index_coupling()
    return theta_block, column_block, n_blocks


def _group(partition, cid):
    return set(partition.groups[partition.group_of[cid]])


class TestMeasuredPartition(unittest.TestCase):
    def test_groups_follow_the_measurements(self):
        model = build()
        p = measured_partition(model)
        # the rooms are bounded by their measured flow and the measured supply temperature
        self.assertEqual(_group(p, "room0"), {"room0", "T_sensor0"})
        self.assertEqual(_group(p, "room1"), {"room1", "T_sensor1"})
        # the terminals' exhaust flows are not measured: they bind the terminals to the return junction and the unit
        self.assertEqual(
            _group(p, "unit"),
            {"unit", "supply_junction", "return_junction", "terminal0", "terminal1", "flow_sensor0", "flow_sensor1", "supplyT_sensor"},
        )
        crossing = {(e.sender.id, e.output_port, e.receiver.id, e.input_port) for e in p.crossing}
        self.assertIn(("terminal0", "airFlowRate", "room0", "flow"), crossing)
        self.assertIn(("unit", "supplyAirTemperature", "room1", "supplyT"), crossing)
        self.assertIn(("room0", "T", "return_junction", "airTemperatureIn"), crossing)
        # the measured flow into the supply junction stays inside the unit's group
        self.assertIn(("terminal0", "airFlowRate", "supply_junction", "airFlowRateOut"), {
            (e.sender.id, e.output_port, e.receiver.id, e.input_port) for e in p.internal
        })

    def test_derived_exhaust_frees_the_terminals(self):
        model = build()
        derived = {
            (f"terminal{k}", "exhaustAirFlowRate"): (model.components[f"flow_sensor{k}"], lambda s: RATIO * s)
            for k in range(2)
        }
        p = measured_partition(model, derived=derived)
        self.assertEqual(_group(p, "terminal0"), {"terminal0", "flow_sensor0"})
        self.assertEqual(_group(p, "terminal1"), {"terminal1", "flow_sensor1"})
        self.assertEqual(_group(p, "unit"), {"unit", "supply_junction", "return_junction", "supplyT_sensor"})
        self.assertEqual(sum(1 for g in p.groups if any(c in p.free for c in g)), 5)

    def test_cut_gives_the_estimator_one_block_per_group(self):
        model = build()
        _, _, n_before = _blocks(model)
        self.assertEqual(n_before, 1)
        model = build()
        derived = {
            (f"terminal{k}", "exhaustAirFlowRate"): (model.components[f"flow_sensor{k}"], lambda s: RATIO * s)
            for k in range(2)
        }
        p = measured_partition(model, derived=derived)
        leaves = cut_measured_edges(model, p)
        self.assertTrue(all(leaf.has_data for leaf in leaves))
        model.load(draw_semantic_model=False, draw_simulation_model=False)
        theta_block, column_block, n_blocks = _blocks(model)
        self.assertEqual(n_blocks, 5)
        # every measurement column belongs to a block, and the rooms' T sensors to the rooms' blocks
        self.assertTrue((column_block >= 0).all())
        # the room still receives a flow and a supply temperature: from the leaves
        room = model.components["room0"]
        senders = {conn.connects_system.id for cp in room.connects_at for conn in cp.connects_system_through}
        self.assertEqual(senders, {"flow_sensor0__replay", "supplyT_sensor__replay"})
        # the return junction's exhaust slots replay the ratio times the measured flow
        junction = model.components["return_junction"]
        senders = {conn.connects_system.id for cp in junction.connects_at for conn in cp.connects_system_through}
        self.assertEqual(senders, {"flow_sensor0__replay1", "flow_sensor1__replay1", "T_sensor0__replay", "T_sensor1__replay"})

    def test_cut_model_simulates_with_the_replayed_series(self):
        model = build()
        p = measured_partition(model)
        cut_measured_edges(model, p)
        model.load(draw_semantic_model=False, draw_simulation_model=False)
        sim = tb.Simulator(model)
        sim.simulate(start_time=START, end_time=END, step_size=STEP, show_progress_bar=False)
        # the rooms integrate the replayed constant 20 (flow sensor series) instead of the damper's flow
        hist = model.components["room0"].output["T"].history().reshape(-1)
        self.assertTrue(torch.isfinite(hist).all())
        self.assertGreater(float(hist[-1]), float(hist[0]))

    def test_fusable_arcs_bind_even_when_measured(self):
        """Two zones joined by a wall, both temperatures measured: the wall's
        arcs are fusable, so the pair stays one group and the cut leaves the
        wall's edges alone (replayed, the explicit exchange would diverge)."""
        model = tb.Model(id="partition_wall")
        zones = [
            tb.BuildingSpaceThermalSystem(C_air=1e6, C_wall=5e6, R_out=0.01, R_in=0.01, f_wall=0.0, f_air=0.0, Q_occ_gain=100.0, id=f"zone{k}")
            for k in range(2)
        ]
        wall = tb.WallSystem(C=2e5, R_a=0.02, R_b=0.02, id="wall")
        outdoor = series([5.0] * 8, id="outdoor")
        zero = series([0.0] * 8, id="zero")
        for z in zones:
            for port in ("outdoorTemperature", "supplyAirFlowRate", "exhaustAirFlowRate", "supplyAirTemperature", "globalIrradiation", "numberOfPeople", "heatGain"):
                model.add_connection(outdoor if port == "outdoorTemperature" else zero, z, "measuredValue", port)
            model.add_connection(z, data_sensor(f"{z.id}_T"), "indoorTemperature", "measuredValue")
        model.add_connection(zones[0], wall, "indoorTemperature", "temperatureA")
        model.add_connection(zones[1], wall, "indoorTemperature", "temperatureB")
        model.add_connection(wall, zones[0], "heatFlowRateA", "wallHeatGain", input_port_index=0)
        model.add_connection(wall, zones[1], "heatFlowRateB", "wallHeatGain", input_port_index=0)
        model.load(draw_semantic_model=False, draw_simulation_model=False)
        p = measured_partition(model)
        self.assertEqual(_group(p, "zone0"), {"zone0", "zone1", "wall", "zone0_T", "zone1_T"})
        self.assertEqual(p.crossing, [])
        self.assertTrue(any(e.sender.id == "zone0" and e.receiver.id == "wall" for e in p.binding))
        cut_measured_edges(model, p)
        self.assertEqual(len(model.get_components_by_class(tb.SensorSystem)), 4)

    def test_a_sensor_feeding_a_controller_opens_the_loop(self):
        """A room whose measured temperature feeds a controller that drives
        the room: the sensor's outgoing edge is measured by the sensor's own
        series, so the cut hands the controller the replayed measurement.
        The group is unchanged (the controller still drives the room), but
        the loop is open."""
        model = tb.Model(id="partition_loop")
        room = Room(k=0.5, id="room")
        gain = Gain(g=0.1, id="gain")
        supply = series([18.0] * 8, id="supplyT")
        sensor = data_sensor("T_sensor")
        model.add_connection(supply, room, "measuredValue", "supplyT")
        model.add_connection(room, sensor, "T", "measuredValue")
        model.add_connection(sensor, gain, "measuredValue", "T")
        model.add_connection(gain, room, "u", "flow", input_port_index=0)
        model.load(draw_semantic_model=False, draw_simulation_model=False)
        p = measured_partition(model)
        self.assertEqual(_group(p, "room"), {"room", "T_sensor", "gain"})
        self.assertEqual(p.crossing, [])
        internal = {(e.sender.id, e.receiver.id) for e in p.internal}
        self.assertEqual(internal, {("T_sensor", "gain")})
        self.assertTrue(any(e.sender.id == "gain" and e.receiver.id == "room" for e in p.binding))
        leaves = cut_measured_edges(model, p)
        self.assertEqual([leaf.id for leaf in leaves], ["T_sensor__replay"])
        senders = {conn.connects_system.id for cp in gain.connects_at for conn in cp.connects_system_through}
        self.assertEqual(senders, {"T_sensor__replay"})
        self.assertTrue(any(conn.connects_system.id == "gain" for cp in room.connects_at for conn in cp.connects_system_through))
        # left alone when asked
        model2 = tb.Model(id="partition_loop2")
        room2, gain2, sensor2 = Room(k=0.5, id="room"), Gain(g=0.1, id="gain"), data_sensor("T_sensor")
        model2.add_connection(series([18.0] * 8, id="supplyT"), room2, "measuredValue", "supplyT")
        model2.add_connection(room2, sensor2, "T", "measuredValue")
        model2.add_connection(sensor2, gain2, "measuredValue", "T")
        model2.add_connection(gain2, room2, "u", "flow", input_port_index=0)
        model2.load(draw_semantic_model=False, draw_simulation_model=False)
        self.assertEqual(cut_measured_edges(model2, measured_partition(model2), internal=False), [])


if __name__ == "__main__":
    unittest.main()
