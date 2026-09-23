"""The unit factored into terminals, junctions and the device on its totals.

The composite ``AirHandlingUnitSystem`` (dampers and junctions inside,
vector ports per branch) and the factored wiring (one ``DamperSystem`` per
terminal, gated by the fan and carrying its exhaust flow, a
``SupplyFlowJunctionSystem`` and a ``ReturnFlowJunctionSystem``, and the
``AirHandlingUnitCoreSystem`` on the totals) are the same device: they give
the same branch flows, supply temperature and powers step for step, in
object mode, in the functional engine, and with the terminals batched into
one meta component.  Factored, the estimation problem's coupling structure
follows the wiring: with the unit's supply temperature and the rooms'
states measured, every room is its own block.
"""
import datetime
import unittest

import pandas as pd
import torch
from dateutil import tz

import twin4build as tb
import twin4build.utils.types as tps
from twin4build.systems.air_handling_unit.air_handling_unit_core_system import (
    AirHandlingUnitCoreSystem,
)
from twin4build.systems.air_handling_unit.air_handling_unit_system import (
    AirHandlingUnitSystem,
)
from twin4build.systems.damper.damper_system import DamperSystem
from twin4build.systems.junction.return_flow_junction_system import (
    ReturnFlowJunctionSystem,
)
from twin4build.systems.junction.supply_flow_junction_system import (
    SupplyFlowJunctionSystem,
)
from twin4build.tests.model.test_batching_vector_slots import Sink

tb._IS_TESTING = True

START = datetime.datetime(2024, 1, 1, tzinfo=tz.UTC)
END = START + datetime.timedelta(hours=1)
STEP = 600
RATIO = 0.9
FAN_KWARGS = {
    "nominalPowerRate": 1000.0, "nominalAirFlowRate": 1.0, "c1": 0.0, "c2": 0.2,
    "c3": 0.8, "c4": 0.0, "f_total": 1.0,
}
HR_KWARGS = {
    "eps_75_h": 0.7, "eps_100_h": 0.75, "eps_75_c": 0.65, "eps_100_c": 0.7,
    "primaryAirFlowRateMax": 1.0, "secondaryAirFlowRateMax": 1.0,
}
DAMPER_KWARGS = {"a": 1.0, "nominalAirFlowRate": 0.5}
#: Per step: the two branches' damper positions, the fan speed, the two
#: rooms' temperatures, the outdoor temperature and the supply setpoint.
POSITIONS = [(1.0, 0.5), (0.3, 0.7), (0.0, 1.0), (0.6, 0.6), (1.0, 1.0), (0.2, 0.9)]
FAN_SPEED = [1.0, 1.0, 0.05, 0.0, 1.0, 0.5]
ROOM_T = [(22.0, 21.0), (22.5, 21.0), (23.0, 20.5), (22.0, 22.0), (21.0, 24.0), (22.0, 21.5)]
OUTDOOR = [5.0, 6.0, 4.0, 3.0, 8.0, 10.0]
SETPOINT = 18.0


def Series(values, **kwargs):
    """An exogenous series as a data-bearing sensor (replayed by every engine)."""
    values = [float(v) for v in values]
    index = pd.DatetimeIndex([START + datetime.timedelta(seconds=STEP * k) for k in range(len(values) + 2)], name="time")
    values = values + [values[-1]] * 2
    return tb.SensorSystem(df=pd.DataFrame({"value": values}, index=index), use_df=True, **kwargs)


def _leaves(model, prefix=""):
    """The exogenous inputs both wirings share."""
    pos = [Series([p[k] for p in POSITIONS], id=f"{prefix}pos{k}") for k in range(2)]
    fan = Series(FAN_SPEED, id=f"{prefix}fan")
    room = [Series([r[k] for r in ROOM_T], id=f"{prefix}room{k}") for k in range(2)]
    outdoor = Series(OUTDOOR, id=f"{prefix}outdoor")
    setpoint = Series([SETPOINT] * len(POSITIONS), id=f"{prefix}setpoint")
    for c in pos + [fan] + room + [outdoor, setpoint]:
        model.add_component(c)
    return pos, fan, room, outdoor, setpoint


def build_composite():
    model = tb.Model(id="composite_unit")
    pos, fan, room, outdoor, setpoint = _leaves(model)
    unit = AirHandlingUnitSystem(
        id="unit",
        supply_damper_kwargs=dict(DAMPER_KWARGS),
        heat_recovery_kwargs=dict(HR_KWARGS),
        supply_fan_kwargs=dict(FAN_KWARGS),
        exhaust_fan_kwargs=dict(FAN_KWARGS),
        n_branches=2,
        exhaust_follows_supply=True,
        exhaustFlowRatio=RATIO,
    )
    for k in range(2):
        model.add_connection(pos[k], unit, "measuredValue", "supplyDamperPosition", input_port_index=k)
        model.add_connection(room[k], unit, "measuredValue", "exhaustTemperature", input_port_index=k)
    model.add_connection(fan, unit, "measuredValue", "supplyFanSpeed")
    model.add_connection(outdoor, unit, "measuredValue", "outdoorAirTemperature")
    model.add_connection(setpoint, unit, "measuredValue", "supplyAirTemperatureSetpoint")
    sink = Sink(id="sink")  # a state for the functional engine to roll
    model.add_connection(unit, sink, "supplyAirTemperature", "u", input_port_index=0)
    # readers of the branch flows: a room-like sink on both branches and a
    # flow sensor on branch 1 (what the factoring must reroute to the terminals)
    flows = Sink(id="flows")
    model.add_connection(
        unit, flows, "supplyAirFlowRate", "u",
        output_port_index=torch.tensor([0, 1]), input_port_index=torch.tensor([0, 1]),
    )
    flow_sensor = tb.SensorSystem(id="flow_sensor1")
    model.add_connection(unit, flow_sensor, "supplyAirFlowRate", "measuredValue", output_port_index=1)
    model.load(draw_semantic_model=False, draw_simulation_model=False)
    return model


def build_factored():
    model = tb.Model(id="factored_unit")
    pos, fan, room, outdoor, setpoint = _leaves(model)
    dampers = [
        DamperSystem(id=f"terminal{k}", exhaustFlowRatio=RATIO, **DAMPER_KWARGS)
        for k in range(2)
    ]
    supply_junction = SupplyFlowJunctionSystem(id="supply_junction")
    return_junction = ReturnFlowJunctionSystem(id="return_junction")
    unit = AirHandlingUnitCoreSystem(
        id="unit",
        heat_recovery_kwargs=dict(HR_KWARGS),
        supply_fan_kwargs=dict(FAN_KWARGS),
        exhaust_fan_kwargs=dict(FAN_KWARGS),
    )
    for k, d in enumerate(dampers):
        model.add_connection(pos[k], d, "measuredValue", "damperPosition")
        model.add_connection(fan, d, "measuredValue", "fanSpeed")
        model.add_connection(d, supply_junction, "airFlowRate", "airFlowRateOut", input_port_index=k)
        model.add_connection(d, return_junction, "exhaustAirFlowRate", "airFlowRateIn", input_port_index=k)
        model.add_connection(room[k], return_junction, "measuredValue", "airTemperatureIn", input_port_index=k)
    model.add_connection(supply_junction, unit, "airFlowRateIn", "totalSupplyAirFlowRate")
    model.add_connection(return_junction, unit, "airFlowRateOut", "totalExhaustAirFlowRate")
    model.add_connection(return_junction, unit, "airTemperatureOut", "returnAirTemperature")
    model.add_connection(outdoor, unit, "measuredValue", "outdoorAirTemperature")
    model.add_connection(setpoint, unit, "measuredValue", "supplyAirTemperatureSetpoint")
    sink = Sink(id="sink")  # a state for the functional engine to roll
    model.add_connection(unit, sink, "supplyAirTemperature", "u", input_port_index=0)
    model.load(draw_semantic_model=False, draw_simulation_model=False)
    return model


DEVICE_OUTPUTS = (
    "supplyAirTemperature", "preheatSupplyAirTemperature", "exhaustAirTemperatureOut",
    "totalSupplyAirFlowRate", "totalExhaustAirFlowRate",
    "heatingPower", "coolingPower", "supplyFanPower", "exhaustFanPower",
)


def _history(model, cid, port, batched=None):
    if batched is None:
        hist = model.components[cid].output[port].history()
        return hist.reshape(hist.shape[0], -1).detach().cpu().clone()
    meta, i_c = model._component_to_meta[cid]
    hist = meta.output[port].history()
    n_t = hist.shape[0]
    return hist.reshape(n_t, meta.n_c, -1)[:, i_c, :].detach().cpu().clone()


def _simulate(model, **simulator_kwargs):
    sim = tb.Simulator(model, **simulator_kwargs)
    sim.simulate(start_time=START, end_time=END, step_size=STEP, show_progress_bar=False)


def _device_histories(model, batched=None):
    return {p: _history(model, "unit", p, batched) for p in DEVICE_OUTPUTS}


class TestFactoredUnit(unittest.TestCase):
    def assert_same_device(self, composite, factored):
        for port in DEVICE_OUTPUTS:
            torch.testing.assert_close(factored[port], composite[port], msg=port)

    def test_object_mode_parity(self):
        composite, factored = build_composite(), build_factored()
        _simulate(composite)
        _simulate(factored)
        self.assert_same_device(_device_histories(composite), _device_histories(factored))
        # the terminals' flows are the unit's branch flows, gated by the fan
        branch = _history(composite, "unit", "supplyAirFlowRate")
        exhaust = _history(composite, "unit", "exhaustAirFlowRate")
        for k in range(2):
            torch.testing.assert_close(_history(factored, f"terminal{k}", "airFlowRate")[:, 0], branch[:, k])
            torch.testing.assert_close(
                _history(factored, f"terminal{k}", "exhaustAirFlowRate")[:, 0], exhaust[:, k]
            )
        # the fan stops the flow at step 3 (speed 0) and halves it at step 2 (0.05)
        flow0 = _history(factored, "terminal0", "airFlowRate")[:, 0]
        self.assertEqual(float(flow0[3]), 0.0)
        self.assertGreater(float(flow0[0]), 0.0)

    def test_functional_parity(self):
        composite, factored = build_composite(), build_factored()
        _simulate(composite)
        reference = _device_histories(composite)
        _simulate(factored, execution_mode="functional", execution_backend="eager")
        self.assert_same_device(reference, _device_histories(factored))

    def test_batched_terminals_parity(self):
        composite, factored = build_composite(), build_factored()
        _simulate(composite)
        reference = _device_histories(composite)
        batched = factored.batch_components()
        self.assertIs(
            factored._component_to_meta["terminal0"][0], factored._component_to_meta["terminal1"][0]
        )
        batched.load(draw_semantic_model=False, draw_simulation_model=False)
        _simulate(batched)
        self.assert_same_device(reference, _device_histories(factored, batched))
        _simulate(batched, execution_mode="functional", execution_backend="eager")
        self.assert_same_device(reference, _device_histories(factored, batched))

    def test_junction_width_from_slots(self):
        """One connection carrying several slots (a batched meta) sizes the
        junction by its highest slot, not by the number of connections."""
        factored = build_factored()
        batched = factored.batch_components()
        batched.load(draw_semantic_model=False, draw_simulation_model=False)
        batched.initialize(start_time=[START], end_time=[END], step_size=STEP)
        self.assertEqual(batched.components["supply_junction"].n_input_ports, 2)
        self.assertEqual(batched.components["return_junction"].n_input_ports, 2)

    def test_estimable_parameters(self):
        """The terminal owns its ratio; the composite unit keeps its own and
        does not report its internal dampers' ratios."""
        terminal = DamperSystem(id="t", exhaustFlowRatio=RATIO, **DAMPER_KWARGS)
        attrs = {e[1] for e in terminal.get_estimable_parameters()}
        self.assertIn("exhaustFlowRatio", attrs)
        self.assertIn("a", attrs)
        composite = build_composite().components["unit"]
        attrs = {e[1] for e in composite.get_estimable_parameters()}
        self.assertIn("exhaustFlowRatio", attrs)
        self.assertNotIn("supply_damper.exhaustFlowRatio", attrs)
        self.assertNotIn("exhaust_damper.exhaustFlowRatio", attrs)
        self.assertNotIn("supply_damper.exhaustFlowRatio", composite.config["parameters"])
        core_unit = build_factored().components["unit"]
        attrs = {e[1] for e in core_unit.get_estimable_parameters()}
        self.assertTrue(all(a.split(".")[0] in ("coil", "heat_recovery", "supply_fan", "exhaust_fan") for a in attrs))
        self.assertTrue(attrs)

    def test_damper_forward_without_fan_speed_is_ungated(self):
        """A caller that passes no ``fanSpeed`` (the composite unit's internal
        dampers gate on their own) gets the plain characteristic."""
        d = DamperSystem(id="d", **DAMPER_KWARGS)
        d.initialize(start_time=[START], end_time=[END], step_size=STEP)
        params = d._forward_params()
        u = torch.tensor([0.5], dtype=tps.float_dtype())
        _, plain = d.forward(None, {"damperPosition": u}, params, STEP)
        _, gated = d.forward(None, {"damperPosition": u, "fanSpeed": torch.tensor([0.05])}, params, STEP)
        torch.testing.assert_close(gated["airFlowRate"], 0.5 * plain["airFlowRate"])
        torch.testing.assert_close(plain["exhaustAirFlowRate"], plain["airFlowRate"])


class TestFactoringTheComposite(unittest.TestCase):
    """``Model.factor_air_handling_units`` turns the translated composite
    into the factored form and keeps every history."""

    def _reference(self):
        composite = build_composite()
        _simulate(composite)
        histories = _device_histories(composite)
        histories["flows"] = _history(composite, "flows", "w")
        histories["flow_sensor1"] = _history(composite, "flow_sensor1", "measuredValue")
        histories["branch"] = _history(composite, "unit", "supplyAirFlowRate")
        histories["exhaust"] = _history(composite, "unit", "exhaustAirFlowRate")
        return histories

    def _factored(self):
        model = build_composite()
        (parts,) = model.factor_air_handling_units()
        model.load(draw_semantic_model=False, draw_simulation_model=False)
        return model, parts

    def _check(self, model, reference, batched=None):
        for port in DEVICE_OUTPUTS:
            torch.testing.assert_close(_history(model, "unit", port, batched), reference[port], msg=port)
        torch.testing.assert_close(_history(model, "flows", "w", batched), reference["flows"])
        torch.testing.assert_close(_history(model, "flow_sensor1", "measuredValue", batched), reference["flow_sensor1"])
        for k in range(2):
            torch.testing.assert_close(
                _history(model, f"unit_terminal{k}", "airFlowRate", batched)[:, 0], reference["branch"][:, k]
            )
            torch.testing.assert_close(
                _history(model, f"unit_terminal{k}", "exhaustAirFlowRate", batched)[:, 0], reference["exhaust"][:, k]
            )

    def test_structure(self):
        model, parts = self._factored()
        self.assertEqual(model.get_components_by_class(AirHandlingUnitSystem), [])
        self.assertIsInstance(model.components["unit"], AirHandlingUnitCoreSystem)
        self.assertEqual([d.id for d in parts["dampers"]], ["unit_terminal0", "unit_terminal1"])
        self.assertEqual(parts["exhaust_dampers"], [])
        self.assertAlmostEqual(float(parts["dampers"][0].exhaustFlowRatio.get().reshape(-1)[0]), RATIO)
        self.assertIn("unit_supply_junction", model.components)
        self.assertIn("unit_return_junction", model.components)

    def test_object_mode(self):
        reference = self._reference()
        model, _ = self._factored()
        _simulate(model)
        self._check(model, reference)

    def test_functional_mode(self):
        reference = self._reference()
        model, _ = self._factored()
        _simulate(model, execution_mode="functional", execution_backend="eager")
        self._check(model, reference)

    def test_batched(self):
        reference = self._reference()
        model, _ = self._factored()
        batched = model.batch_components()
        batched.load(draw_semantic_model=False, draw_simulation_model=False)
        _simulate(batched, execution_mode="functional", execution_backend="eager")
        self._check(model, reference, batched)

    def test_exhaust_dampers_are_factored_too(self):
        model = build_composite()
        unit = model.components["unit"]
        unit.exhaust_follows_supply = False
        for k in range(2):
            model.add_connection(model.components[f"pos{k}"], unit, "measuredValue", "exhaustDamperPosition", input_port_index=k)
        model.load(draw_semantic_model=False, draw_simulation_model=False)
        _simulate(model)
        reference = _device_histories(model)
        exhaust = _history(model, "unit", "exhaustAirFlowRate")
        (parts,) = model.factor_air_handling_units()
        self.assertEqual(len(parts["exhaust_dampers"]), 2)
        model.load(draw_semantic_model=False, draw_simulation_model=False)
        _simulate(model)
        for port in DEVICE_OUTPUTS:
            torch.testing.assert_close(_history(model, "unit", port), reference[port], msg=port)
        for k in range(2):
            torch.testing.assert_close(
                _history(model, f"unit_exhaust_terminal{k}", "airFlowRate")[:, 0], exhaust[:, k]
            )


if __name__ == "__main__":
    unittest.main()
