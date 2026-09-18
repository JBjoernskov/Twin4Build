"""``exhaust_follows_supply``: the AHU's (and the occupancy inversion's)
exhaust flow is ``exhaustFlowRatio`` times the supply flow.

With one exhaust meter per AHU the per-branch exhaust dampers are not
identifiable (only their total is); the ratio is pinned by that meter.
"""

# Standard library imports
import datetime
import unittest

# Third party imports
import torch
from dateutil import tz

# Local application imports
import twin4build
import twin4build as tb
from twin4build.systems.air_handling_unit.air_handling_unit_system import (
    AirHandlingUnitSystem,
)

twin4build._IS_TESTING = True

START = datetime.datetime(2023, 1, 1, tzinfo=tz.UTC)


def _ahu(follows: bool, ratio: float = 0.93, n_branches: int = 2) -> AirHandlingUnitSystem:
    fan_kwargs = {
        "nominalPowerRate": 1000.0, "nominalAirFlowRate": 1, "c1": 0, "c2": 0.2,
        "c3": 0.8, "c4": 0, "f_total": 1.0,
    }
    return AirHandlingUnitSystem(
        id="ahu",
        supply_damper_kwargs={"nominalAirFlowRate": 0.5},
        exhaust_damper_kwargs={"nominalAirFlowRate": 0.2},  # deliberately different
        supply_fan_kwargs=fan_kwargs,
        exhaust_fan_kwargs=dict(fan_kwargs),
        n_branches=n_branches,
        exhaust_follows_supply=follows,
        exhaustFlowRatio=ratio,
    )


def _step(ahu: AirHandlingUnitSystem, supply_pos, exhaust_pos):
    ahu.initialize(start_time=[START], end_time=[START + datetime.timedelta(hours=1)], step_size=[600])
    ahu.input["supplyDamperPosition"].set(torch.tensor([supply_pos], dtype=torch.float64), i_t=0)
    ahu.input["exhaustDamperPosition"].set(torch.tensor([exhaust_pos], dtype=torch.float64), i_t=0)
    ahu.input["exhaustTemperature"].set(torch.tensor([[22.0] * len(supply_pos)], dtype=torch.float64), i_t=0)
    ahu.input["supplyAirTemperatureSetpoint"].set(torch.tensor([18.0], dtype=torch.float64), i_t=0)
    ahu.input["outdoorAirTemperature"].set(torch.tensor([5.0], dtype=torch.float64), i_t=0)
    ahu.do_step(second_time=0, date_time=START, step_size=[600], step_index=0)
    return ahu.output["supplyAirFlowRate"].get(), ahu.output["exhaustAirFlowRate"].get()


class TestExhaustFollowsSupply(unittest.TestCase):
    def test_do_step_scales_supply_by_the_ratio(self):
        sup, exh = _step(_ahu(True, 0.93), [1.0, 0.5], [0.0, 0.0])  # exhaust dampers closed
        torch.testing.assert_close(exh, 0.93 * sup)

    def test_default_uses_the_exhaust_damper(self):
        sup, exh = _step(_ahu(False), [1.0, 0.5], [1.0, 1.0])
        self.assertFalse(torch.allclose(exh, 0.93 * sup))
        # fully open exhaust dampers deliver their own nominal flow
        torch.testing.assert_close(exh, torch.full_like(exh, 0.2))

    def test_forward_matches_do_step_and_reads_the_ratio_from_params(self):
        ahu = _ahu(True, 0.93)
        sup, exh = _step(ahu, [1.0, 0.5], [0.0, 0.0])
        inputs = {
            "supplyDamperPosition": torch.tensor([[1.0, 0.5]], dtype=torch.float64),
            "exhaustDamperPosition": torch.tensor([[0.0, 0.0]], dtype=torch.float64),
            "exhaustTemperature": torch.tensor([[22.0, 22.0]], dtype=torch.float64),
            "supplyAirTemperatureSetpoint": torch.tensor([18.0], dtype=torch.float64),
            "outdoorAirTemperature": torch.tensor([5.0], dtype=torch.float64),
        }
        _, out = ahu.forward(None, inputs, {}, 600.0)
        torch.testing.assert_close(out["exhaustAirFlowRate"], exh[0])
        # an estimated ratio arrives through params
        _, out2 = ahu.forward(None, inputs, {"exhaustFlowRatio": torch.tensor([0.5], dtype=torch.float64)}, 600.0)
        torch.testing.assert_close(out2["exhaustAirFlowRate"], 0.5 * out2["supplyAirFlowRate"])

    def test_preheat_temperature_output(self):
        ahu = _ahu(False)
        _step(ahu, [1.0, 1.0], [1.0, 1.0])
        preheat = ahu.output["preheatSupplyAirTemperature"].get()
        supply = ahu.output["supplyAirTemperature"].get()
        # heat-recovery outlet lies between outdoor (5) and the setpoint (18)
        self.assertTrue(bool((preheat > 5.0).all()) and bool((preheat <= 18.0 + 1e-9).all()))
        self.assertTrue(bool((supply >= preheat - 1e-9).all()))

    def test_estimable_parameters_follow_the_mode(self):
        attrs_on = {attr for _, attr, *_ in _ahu(True).get_estimable_parameters()}
        attrs_off = {attr for _, attr, *_ in _ahu(False).get_estimable_parameters()}
        self.assertIn("exhaustFlowRatio", attrs_on)
        self.assertFalse(any(a.startswith("exhaust_damper.") for a in attrs_on))
        self.assertNotIn("exhaustFlowRatio", attrs_off)
        self.assertTrue(any(a.startswith("exhaust_damper.") for a in attrs_off))
        self.assertIn("supply_damper.nominalAirFlowRate", attrs_on)

    def test_config_round_trips_the_option(self):
        model = tb.Model(id="ahu_ratio_round_trip")
        ahu = _ahu(True, 0.9)
        model.add_component(ahu)
        cfg = ahu.config["parameters"]
        self.assertIn("exhaust_follows_supply", cfg)
        self.assertIn("exhaustFlowRatio", cfg)


class TestOccupancyFollowsSupply(unittest.TestCase):
    def _occ(self, follows):
        occ = tb.OccupancySystem(
            V=100.0, G_occ=5e-6, m_inf=0.001,
            supply_damper_nominalAirFlowRate=0.5,
            exhaust_damper_nominalAirFlowRate=0.2,
            exhaust_follows_supply=follows,
            exhaustFlowRatio=0.93,
            id="occ",
        )
        return occ

    def _people(self, occ, ratio_param=None):
        def t(v):
            return torch.tensor([float(v)], dtype=torch.float64)

        params = {
            "mass.V": t(100.0), "mass.G_occ": t(5e-6),
            "mass.m_inf": t(0.001),
            "supply_damper.a": t(1.0), "supply_damper.nominalAirFlowRate": t(0.5),
            "exhaust_damper.a": t(1.0), "exhaust_damper.nominalAirFlowRate": t(0.2),
        }
        if ratio_param is not None:
            params["exhaustFlowRatio"] = t(ratio_param)
        inputs = {
            "indoorCo2Measured": t(900.0),
            "previousIndoorCo2Measured": t(880.0),
            "damperPositionMeasured": t(1.0),
            "outdoorCo2Concentration": t(400.0),
        }
        _, out = occ.forward(None, inputs, params, 600.0)
        return float(out["scheduleValue"])

    def test_ratio_changes_the_booked_people(self):
        n_surplus = self._people(self._occ(True))
        n_deficit = self._people(self._occ(True), ratio_param=1.2)
        # Make-up flow is max(m_exh - m_sup, 0): a ratio below 1 does not
        # change m_tot, a ratio above 1 does.
        self.assertNotAlmostEqual(n_surplus, n_deficit)
        self.assertAlmostEqual(n_surplus, self._people(self._occ(False)), places=5)

    def test_estimable_parameters_follow_the_mode(self):
        attrs_on = {attr for _, attr, *_ in self._occ(True).get_estimable_parameters()}
        attrs_off = {attr for _, attr, *_ in self._occ(False).get_estimable_parameters()}
        self.assertIn("exhaustFlowRatio", attrs_on)
        self.assertFalse(any(a.startswith("exhaust_damper.") for a in attrs_on))
        self.assertNotIn("exhaustFlowRatio", attrs_off)


if __name__ == "__main__":
    unittest.main()
