"""Balanced ventilation in the room air models (``air_balance``).

Every air stream entering the room is balanced by room air leaving at room
state, so

* supply in excess of the exhaust leaves through the envelope and the
  exhaust flow drops out of the dynamics;
* exhaust in excess of the supply draws outdoor make-up air through the
  envelope;

on the object (``do_step``), functional (``forward``) and fused paths alike.
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
import twin4build.utils.constants as constants
from twin4build.systems.building_space.air_balance import (
    balanced_flow_inputs,
    make_up_air_flow,
)

twin4build._IS_TESTING = True

START = datetime.datetime(2024, 1, 4, tzinfo=tz.gettz("Europe/Copenhagen"))
DT = 0.01  # tiny step: (x_next - x) / dt is the continuous right-hand side to O(dt)


def _t(v):
    return torch.tensor([float(v)], dtype=torch.float64)


class TestMakeUpFlow(unittest.TestCase):
    def test_make_up_is_exhaust_deficit_clamped_at_zero(self):
        sup = torch.tensor([0.10, 0.10, 0.10], dtype=torch.float64)
        exh = torch.tensor([0.05, 0.10, 0.16], dtype=torch.float64)
        torch.testing.assert_close(
            make_up_air_flow(sup, exh),
            torch.tensor([0.0, 0.0, 0.06], dtype=torch.float64),
        )

    def test_transform_replaces_only_the_exhaust_slot(self):
        out = balanced_flow_inputs(
            {"supplyAirFlowRate": _t(0.1), "exhaustAirFlowRate": _t(0.3), "x": _t(1)}
        )
        self.assertEqual(set(out), {"exhaustAirFlowRate"})
        torch.testing.assert_close(out["exhaustAirFlowRate"], _t(0.2))
        self.assertEqual(balanced_flow_inputs({"supplyAirFlowRate": _t(0.1)}), {})


class TestThermalBalance(unittest.TestCase):
    """Right-hand side of the air node with the wall at air temperature (no
    conduction) and no other gains: only the ventilation terms remain."""

    C_AIR = 1e5
    T_I, T_O, T_SUP = 20.0, 5.0, 30.0

    def setUp(self):
        self.zone = tb.BuildingSpaceThermalSystem(
            C_air=self.C_AIR,
            C_wall=5e6,
            R_out=0.01,
            R_in=0.01,
            f_wall=0.0,
            f_air=0.0,
            Q_occ_gain=0.0,
            T_air_start=self.T_I,
            T_wall_start=self.T_I,
            id="zone",
        )
        self.zone.initialize(
            start_time=[START], end_time=[START + datetime.timedelta(hours=1)], step_size=[DT]
        )
        self.params = self.zone._forward_params()
        self.x0 = self.zone.ss_model.get_state()

    def _dTdt(self, m_sup, m_exh):
        inputs = {
            "outdoorTemperature": _t(self.T_O),
            "supplyAirFlowRate": _t(m_sup),
            "exhaustAirFlowRate": _t(m_exh),
            "supplyAirTemperature": _t(self.T_SUP),
            "globalIrradiation": _t(0.0),
            "numberOfPeople": _t(0.0),
            "heatGain": _t(0.0),
        }
        x_next, _ = self.zone.forward(self.x0, inputs, self.params, DT)
        return float((x_next[..., 0] - self.x0[..., 0]) / DT)

    def _expected(self, m_sup, m_exh):
        m_mu = max(m_exh - m_sup, 0.0)
        return (
            constants.CP_AIR
            * (m_sup * (self.T_SUP - self.T_I) + m_mu * (self.T_O - self.T_I))
            / self.C_AIR
        )

    def test_balanced_flows_match_the_classic_balance(self):
        self.assertAlmostEqual(self._dTdt(0.1, 0.1), self._expected(0.1, 0.1), places=6)

    def test_surplus_supply_is_independent_of_exhaust(self):
        # Excess supply leaves at room state: no fictitious storage term.
        a, b, c = self._dTdt(0.1, 0.0), self._dTdt(0.1, 0.05), self._dTdt(0.1, 0.1)
        self.assertAlmostEqual(a, b, places=9)
        self.assertAlmostEqual(b, c, places=9)
        self.assertAlmostEqual(a, self._expected(0.1, 0.0), places=6)

    def test_exhaust_deficit_draws_outdoor_air(self):
        # 0.06 kg/s of make-up air at T_o cools the room on top of the supply.
        got = self._dTdt(0.1, 0.16)
        self.assertAlmostEqual(got, self._expected(0.1, 0.16), places=6)
        self.assertLess(got, self._dTdt(0.1, 0.1))

    def test_do_step_matches_forward(self):
        z = self.zone
        for port, value in (
            ("outdoorTemperature", self.T_O),
            ("supplyAirFlowRate", 0.1),
            ("exhaustAirFlowRate", 0.16),
            ("supplyAirTemperature", self.T_SUP),
            ("globalIrradiation", 0.0),
            ("numberOfPeople", 0.0),
            ("heatGain", 0.0),
        ):
            z.input[port].set(_t(value), i_t=0)
        z.do_step(second_time=0, date_time=START, step_size=[DT], step_index=0)
        got = float((z.output["indoorTemperature"].get() - self.T_I) / DT)
        self.assertAlmostEqual(got, self._expected(0.1, 0.16), places=6)


class TestMassBalance(unittest.TestCase):
    V = 100.0
    C_I, C_OUT = 1000.0, 400.0

    def setUp(self):
        self.room = tb.BuildingSpaceMassSystem(V=self.V, G_occ=0.0, m_inf=0.0, id="room")
        self.room.initialize(
            start_time=[START], end_time=[START + datetime.timedelta(hours=1)], step_size=[DT]
        )
        self.params = self.room._forward_params()
        self.x0 = torch.full_like(self.room.ss_model.get_state(), self.C_I)

    def _dCdt(self, m_sup, m_exh):
        inputs = {
            "supplyAirFlowRate": _t(m_sup),
            "exhaustAirFlowRate": _t(m_exh),
            "outdoorCO2": _t(self.C_OUT),
            "numberOfPeople": _t(0.0),
        }
        x_next, _ = self.room.forward(self.x0, inputs, self.params, DT)
        return float((x_next[..., 0] - self.x0[..., 0]) / DT)

    def _expected(self, m_sup, m_exh):
        m_mu = max(m_exh - m_sup, 0.0)
        return (m_sup + m_mu) * (self.C_OUT - self.C_I) / (constants.RHO_AIR * self.V)

    def _assert_rhs(self, got, expected):
        self.assertAlmostEqual(got / expected, 1.0, places=4)

    def test_balanced_flows_match_the_classic_balance(self):
        self._assert_rhs(self._dCdt(0.1, 0.1), self._expected(0.1, 0.1))

    def test_surplus_supply_is_independent_of_exhaust(self):
        a, b = self._dCdt(0.1, 0.0), self._dCdt(0.1, 0.1)
        self.assertAlmostEqual(a, b, places=9)
        self._assert_rhs(a, self._expected(0.1, 0.0))

    def test_exhaust_deficit_dilutes_with_outdoor_air(self):
        got = self._dCdt(0.1, 0.16)
        self._assert_rhs(got, self._expected(0.1, 0.16))
        self.assertLess(got, self._dCdt(0.1, 0.1))

    def test_ventilation_never_pushes_below_outdoor(self):
        # Room at outdoor level: no ventilation combination changes it.
        self.x0 = torch.full_like(self.x0, self.C_OUT)
        for m_sup, m_exh in ((0.1, 0.0), (0.0, 0.1), (0.1, 0.3)):
            self.assertAlmostEqual(self._dCdt(m_sup, m_exh), 0.0, places=9)


def _two_zone_model(fuse, model_id, m_sup, m_exh):
    """Two thermal zones through a partition wall, both ventilated with the
    given (unbalanced) supply/exhaust flows."""
    model = tb.Model(id=model_id)

    def make_zone(zone_id):
        return tb.BuildingSpaceThermalSystem(
            C_air=1e6, C_wall=5e6, R_out=0.01, R_in=0.01,
            f_wall=0.0, f_air=0.0, Q_occ_gain=100.0, id=zone_id,
        )

    zone_a, zone_b = make_zone("ZoneA"), make_zone("ZoneB")
    wall = tb.WallSystem(C=2e5, R_a=0.02, R_b=0.02, id="PartitionWall")
    const = {}
    for name, value in (
        ("Outdoor", 5.0), ("Zero", 0.0), ("SupplyAirTemp", 20.0),
        ("Supply", m_sup), ("Exhaust", m_exh),
    ):
        const[name] = tb.ScheduleSystem(
            weekday_ruleset={"ruleset_default_value": value}, id=name
        )
    heater = tb.ScheduleSystem(
        weekday_ruleset={"ruleset_default_value": 1500.0}, id="HeaterA"
    )
    for zone in (zone_a, zone_b):
        model.add_connection(const["Outdoor"], zone, "scheduleValue", "outdoorTemperature")
        model.add_connection(const["Supply"], zone, "scheduleValue", "supplyAirFlowRate")
        model.add_connection(const["Exhaust"], zone, "scheduleValue", "exhaustAirFlowRate")
        model.add_connection(const["SupplyAirTemp"], zone, "scheduleValue", "supplyAirTemperature")
        model.add_connection(const["Zero"], zone, "scheduleValue", "globalIrradiation")
        model.add_connection(const["Zero"], zone, "scheduleValue", "numberOfPeople")
    model.add_connection(heater, zone_a, "scheduleValue", "heatGain")
    model.add_connection(const["Zero"], zone_b, "scheduleValue", "heatGain")
    model.add_connection(zone_a, wall, "indoorTemperature", "temperatureA")
    model.add_connection(zone_b, wall, "indoorTemperature", "temperatureB")
    model.add_connection(wall, zone_a, "heatFlowRateA", "wallHeatGain", input_port_index=0)
    model.add_connection(wall, zone_b, "heatFlowRateB", "wallHeatGain", input_port_index=0)
    model.load(draw_semantic_model=False, draw_simulation_model=False, enable_fusion=fuse)
    return model


def _simulate(model, hours, step_size):
    tb.Simulator(model).simulate(
        start_time=START,
        end_time=START + datetime.timedelta(hours=hours),
        step_size=step_size,
        show_progress_bar=False,
    )
    return {
        zid: model.components[zid].output["indoorTemperature"].history().detach().flatten()
        for zid in ("ZoneA", "ZoneB")
    }


class TestFusedPathAppliesTheTransform(unittest.TestCase):
    """The fused block stacks the members' raw inputs; the make-up transform
    must be applied there too, or fused and unfused rooms disagree."""

    def test_fused_matches_unfused_with_exhaust_deficit(self):
        res_f = _simulate(_two_zone_model(True, "bal_fuse_on", 0.05, 0.12), 6, 30)
        res_u = _simulate(_two_zone_model(False, "bal_fuse_off", 0.05, 0.12), 6, 30)
        for zid in ("ZoneA", "ZoneB"):
            err = float((res_f[zid] - res_u[zid]).abs().max())
            self.assertLess(err, 0.05, f"{zid}: fused vs unfused max err {err}")

    def test_fused_surplus_supply_ignores_exhaust(self):
        res_0 = _simulate(_two_zone_model(True, "bal_surplus_0", 0.1, 0.0), 6, 30)
        res_1 = _simulate(_two_zone_model(True, "bal_surplus_1", 0.1, 0.08), 6, 30)
        for zid in ("ZoneA", "ZoneB"):
            torch.testing.assert_close(res_0[zid], res_1[zid], rtol=0, atol=1e-6)

    def test_fused_deficit_cools_more_than_balanced(self):
        res_b = _simulate(_two_zone_model(True, "bal_def_b", 0.05, 0.05), 6, 30)
        res_d = _simulate(_two_zone_model(True, "bal_def_d", 0.05, 0.12), 6, 30)
        self.assertLess(float(res_d["ZoneA"][-1]), float(res_b["ZoneA"][-1]))


if __name__ == "__main__":
    unittest.main()
