"""A unit with an explicit exhaust side: the return air comes from exhaust
terminals (``returnAirFlowRate`` / ``returnAirTemperature``), not from the
branches' exhaust.

The composite and its factored form give the same device step for step; the
branches carry no exhaust; the exhaust ratios (the unit's and the
terminals') are not estimated, since nothing they set is read.
"""
import unittest

import torch

import twin4build as tb
from twin4build.systems.air_handling_unit.air_handling_unit_system import AirHandlingUnitSystem
from twin4build.systems.damper.damper_system import DamperSystem
from twin4build.tests.model.test_batching_vector_slots import Sink
from twin4build.tests.systems.air_handling_unit.test_factored_unit import (
    DAMPER_KWARGS,
    DEVICE_OUTPUTS,
    FAN_KWARGS,
    HR_KWARGS,
    POSITIONS,
    RATIO,
    Series,
    _device_histories,
    _history,
    _leaves,
    _simulate,
)

tb._IS_TESTING = True

#: Per step: the two return streams' flows [kg/s] and temperatures [C].
STREAM_FLOWS = [(0.4, 0.2), (0.3, 0.5), (0.0, 0.0), (0.2, 0.2), (0.6, 0.1), (0.1, 0.4)]
STREAM_T = [(21.0, 24.0), (21.5, 23.0), (20.0, 22.0), (22.0, 22.0), (20.0, 25.0), (21.0, 21.0)]


def build_explicit():
    model = tb.Model(id="explicit_return_unit")
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
        flow = Series([f[k] for f in STREAM_FLOWS], id=f"stream_flow{k}")
        temp = Series([t[k] for t in STREAM_T], id=f"stream_t{k}")
        model.add_component(flow)
        model.add_component(temp)
        model.add_connection(flow, unit, "measuredValue", "returnAirFlowRate", input_port_index=k)
        model.add_connection(temp, unit, "measuredValue", "returnAirTemperature", input_port_index=k)
    model.add_connection(fan, unit, "measuredValue", "supplyFanSpeed")
    model.add_connection(outdoor, unit, "measuredValue", "outdoorAirTemperature")
    model.add_connection(setpoint, unit, "measuredValue", "supplyAirTemperatureSetpoint")
    sink = Sink(id="sink")
    model.add_connection(unit, sink, "supplyAirTemperature", "u", input_port_index=0)
    flows = Sink(id="flows")
    model.add_connection(
        unit, flows, "supplyAirFlowRate", "u",
        output_port_index=torch.tensor([0, 1]), input_port_index=torch.tensor([0, 1]),
    )
    model.load(draw_semantic_model=False, draw_simulation_model=False)
    return model


class TestExplicitReturn(unittest.TestCase):
    def _expected_return(self):
        f = torch.tensor(STREAM_FLOWS, dtype=torch.float64)
        t = torch.tensor(STREAM_T, dtype=torch.float64)
        total = f.sum(dim=1)
        mixed = torch.where(total > 1e-5, (f * t).sum(dim=1) / total.clamp_min(1e-5), torch.full_like(total, 20.0))
        return total, mixed

    def test_composite_returns_the_streams(self):
        model = build_explicit()
        unit = model.components["unit"]
        self.assertTrue(unit.explicit_return)
        _simulate(model)
        total, _ = self._expected_return()
        torch.testing.assert_close(_history(model, "unit", "totalExhaustAirFlowRate")[:, 0], total)
        # no exhaust on the branches
        self.assertEqual(float(_history(model, "unit", "exhaustAirFlowRate").abs().max()), 0.0)
        # the supply side is the same as without the streams
        self.assertGreater(float(_history(model, "unit", "totalSupplyAirFlowRate")[0, 0]), 0.0)

    def test_factored_matches_composite(self):
        model = build_explicit()
        _simulate(model)
        reference = _device_histories(model)
        (parts,) = model.factor_air_handling_units()
        self.assertEqual(parts["return_junction"].branch_temperature_slots, [])
        for d in parts["dampers"]:
            self.assertEqual(float(d.exhaustFlowRatio.get().reshape(-1)[0]), 0.0)
        model.load(draw_semantic_model=False, draw_simulation_model=False)
        for kwargs in ({}, {"execution_mode": "functional", "execution_backend": "eager"}):
            _simulate(model, **kwargs)
            for port in DEVICE_OUTPUTS:
                torch.testing.assert_close(_history(model, "unit", port), reference[port], msg=f"{port} {kwargs}")
            _, mixed = self._expected_return()
            torch.testing.assert_close(_history(model, "unit_return_junction", "airTemperatureOut")[:, 0], mixed)

    def test_functional_composite_matches_object_mode(self):
        model = build_explicit()
        _simulate(model)
        reference = _device_histories(model)
        _simulate(model, execution_mode="functional", execution_backend="eager")
        for port in DEVICE_OUTPUTS:
            torch.testing.assert_close(_history(model, "unit", port), reference[port], msg=port)

    def test_ratios_are_not_estimated(self):
        unit = build_explicit().components["unit"]
        attrs = {e[1] for e in unit.get_estimable_parameters()}
        self.assertNotIn("exhaustFlowRatio", attrs)
        self.assertFalse(any(a.startswith("exhaust_damper.") for a in attrs))
        self.assertIn("exhaustFlowRatio", unit._inactive_parameters())
        model = build_explicit()
        (parts,) = model.factor_air_handling_units()
        for d in parts["dampers"]:
            attrs = {e[1] for e in d.get_estimable_parameters()}
            self.assertNotIn("exhaustFlowRatio", attrs)  # nothing reads its exhaust
            self.assertIn("a", attrs)

    def test_a_terminal_keeps_its_ratio_unless_its_exhaust_is_elsewhere(self):
        d = DamperSystem(id="d", **DAMPER_KWARGS)
        self.assertIn("exhaustFlowRatio", {e[1] for e in d.get_estimable_parameters()})
        d = DamperSystem(id="d", exhaust_modelled=False, **DAMPER_KWARGS)
        self.assertNotIn("exhaustFlowRatio", {e[1] for e in d.get_estimable_parameters()})
        self.assertIn("exhaust_modelled", d.config["parameters"])

    def test_streams_share_a_source_temperature(self):
        """Three streams, two sources: stream 2 is drawn from stream 0's source
        (``return_temperature_slots``), in the composite and factored."""
        model = build_explicit()
        unit = model.components["unit"]
        extra = Series([0.3] * len(POSITIONS), id="stream_flow2")
        model.add_component(extra)
        model.add_connection(extra, unit, "measuredValue", "returnAirFlowRate", input_port_index=2)
        unit.return_temperature_slots = [0, 1, 0]
        model.load(draw_semantic_model=False, draw_simulation_model=False)
        _simulate(model)
        f = torch.tensor([[a, b, 0.3] for a, b in STREAM_FLOWS], dtype=torch.float64)
        t = torch.tensor([[a, b, a] for a, b in STREAM_T], dtype=torch.float64)
        expected = (f * t).sum(dim=1) / f.sum(dim=1)
        reference = _device_histories(model)
        (parts,) = model.factor_air_handling_units()
        self.assertEqual(parts["return_junction"].branch_temperature_slots, [0, 1, 0])
        model.load(draw_semantic_model=False, draw_simulation_model=False)
        _simulate(model)
        torch.testing.assert_close(_history(model, "unit_return_junction", "airTemperatureOut")[:, 0], expected)
        for port in DEVICE_OUTPUTS:
            torch.testing.assert_close(_history(model, "unit", port), reference[port], msg=port)
        self.assertEqual(AirHandlingUnitSystem(id="u", return_temperature_slots="0,1,0").return_temperature_slots, [0, 1, 0])
        # one slot: a saved model's loader turns the literal "0" into the integer 0
        from twin4build.model.simulation_model.simulation_model import _convert_literal_value

        one = AirHandlingUnitSystem(id="u1", return_temperature_slots=_convert_literal_value("0"))
        self.assertEqual(one.return_temperature_slots, [0])

    def test_unequal_streams_are_an_error(self):
        model = build_explicit()
        unit = model.components["unit"]
        extra = Series([0.1] * len(POSITIONS), id="stream_flow2")
        model.add_component(extra)
        model.add_connection(extra, unit, "measuredValue", "returnAirFlowRate", input_port_index=2)
        with self.assertRaises(ValueError):
            model.factor_air_handling_units()


if __name__ == "__main__":
    unittest.main()
