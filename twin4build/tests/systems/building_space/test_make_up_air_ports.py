"""``makeUpAirCO2`` / ``makeUpAirTemperature``: unwired, the port reads the
outdoor value (``Input(default=...)``), the classic balance; wired, the
make-up stream carries the port's state and only the exhaust-deficit case
can see it.  ``supplyAirCO2`` the same way for the supply stream."""

# Standard library imports
import datetime
import functools
import unittest

# Third party imports
import torch

# Local application imports
import twin4build
import twin4build as tb
from twin4build.systems.utils.smooth_saturation import saturation_mode
from twin4build.systems.building_space.building_space_mass_system import (
    ACTIVE_INPUT_SLOTS,
    MAKE_UP_CO2_SLOT,
    N_INPUTS,
    OCCUPANCY_SLOT,
    OUTDOOR_CO2_SLOT,
    SUPPLY_CO2_SLOT,
    mass_matrices,
)
from twin4build.utils import constants

twin4build._IS_TESTING = True


def _exact(test):
    """The exact balance (the hard saturation mode): the smooth mode's
    make-up flow is small but not zero where the hard max is."""

    @functools.wraps(test)
    def run(self):
        with saturation_mode("hard"):
            return test(self)

    return run

DT = 600.0
START = datetime.datetime(2024, 3, 4, 0, 0)


def _t(v):
    return torch.tensor([float(v)], dtype=torch.float64)


class TestContract(unittest.TestCase):
    def test_slots(self):
        self.assertEqual(N_INPUTS, 5)
        self.assertEqual((OUTDOOR_CO2_SLOT, OCCUPANCY_SLOT, MAKE_UP_CO2_SLOT), (2, 3, 4))
        self.assertEqual(ACTIVE_INPUT_SLOTS, (2, 3, 4))


class TestMassMakeUpPort(unittest.TestCase):
    def setUp(self):
        self.room = tb.BuildingSpaceMassSystem(V=120.0, G_occ=5e-6, m_inf=0.004, id="room")
        self.room.n_c = 1
        self.params = {"V": _t(120.0), "G_occ": _t(5e-6), "m_inf": _t(0.004)}
        self.x0 = torch.full((1, 1), 900.0, dtype=torch.float64)

    def _step(self, m_sup, m_exh, make_up=None, wired=False):
        inputs = {
            "supplyAirFlowRate": _t(m_sup),
            "exhaustAirFlowRate": _t(m_exh),
            "outdoorCo2Concentration": _t(400.0),
            "numberOfPeople": _t(0.0),
        }
        if make_up is not None and wired:  # unwired, the engine passes the outdoor value
            inputs["makeUpAirCO2"] = _t(make_up)
        x1, _ = self.room.forward(self.x0, self.room.with_defaults(inputs), self.room.full_params(dict(self.params), DT), DT)
        return float(x1[0, 0])

    def test_unwired_ignores_the_port(self):
        self.assertEqual(self._step(0.1, 0.18), self._step(0.1, 0.18, make_up=800.0, wired=False))

    def test_wired_at_outdoor_equals_unwired(self):
        self.assertAlmostEqual(
            self._step(0.1, 0.18), self._step(0.1, 0.18, make_up=400.0, wired=True), places=12
        )

    def test_wired_deficit_draws_the_make_up_state(self):
        # Corridor air at 800 ppm dilutes less than outdoor air at 400.
        c_out = self._step(0.1, 0.18, make_up=400.0, wired=True)
        c_corr = self._step(0.1, 0.18, make_up=800.0, wired=True)
        self.assertGreater(c_corr, c_out)
        # And the effect is exactly the make-up fraction of the difference:
        # only the m_mu = 0.08 kg/s stream changed its source.
        c_mid = self._step(0.1, 0.18, make_up=600.0, wired=True)
        self.assertAlmostEqual(c_mid - c_out, (c_corr - c_out) / 2, places=9)

    @_exact
    def test_wired_surplus_never_sees_the_port(self):
        self.assertAlmostEqual(
            self._step(0.1, 0.03, make_up=400.0, wired=True),
            self._step(0.1, 0.03, make_up=800.0, wired=True),
            places=12,
        )


class TestMassSupplyPort(unittest.TestCase):
    """``supplyAirCO2``: unwired the supply stream brings outdoor air; wired,
    it brings the port's concentration."""

    def setUp(self):
        self.room = tb.BuildingSpaceMassSystem(V=120.0, G_occ=5e-6, m_inf=0.004, id="room")
        self.room.n_c = 1
        self.params = {"V": _t(120.0), "G_occ": _t(5e-6), "m_inf": _t(0.004)}
        self.x0 = torch.full((1, 1), 900.0, dtype=torch.float64)

    def _step(self, m_sup, m_exh, supply=None, wired=False):
        inputs = {
            "supplyAirFlowRate": _t(m_sup),
            "exhaustAirFlowRate": _t(m_exh),
            "outdoorCo2Concentration": _t(400.0),
            "numberOfPeople": _t(0.0),
        }
        if supply is not None and wired:  # unwired, the engine passes the outdoor value
            inputs["supplyAirCO2"] = _t(supply)
        x1, _ = self.room.forward(self.x0, self.room.with_defaults(inputs), self.room.full_params(dict(self.params), DT), DT)
        return float(x1[0, 0])

    def test_unwired_ignores_the_port(self):
        self.assertEqual(self._step(0.1, 0.05), self._step(0.1, 0.05, supply=800.0, wired=False))

    def test_wired_at_outdoor_equals_unwired(self):
        self.assertAlmostEqual(self._step(0.1, 0.05), self._step(0.1, 0.05, supply=400.0, wired=True), places=12)

    def test_wired_supply_brings_its_concentration(self):
        c_out = self._step(0.1, 0.05, supply=400.0, wired=True)
        c_room = self._step(0.1, 0.05, supply=800.0, wired=True)
        self.assertGreater(c_room, c_out)
        # linear in the supply concentration: only the supply stream changed its source
        c_mid = self._step(0.1, 0.05, supply=600.0, wired=True)
        self.assertAlmostEqual(c_mid - c_out, (c_room - c_out) / 2, places=9)
        # no supply flow, no effect
        self.assertAlmostEqual(
            self._step(0.0, 0.05, supply=400.0, wired=True), self._step(0.0, 0.05, supply=800.0, wired=True), places=12
        )

    def test_layout_support_and_matrices(self):
        self.room.n_exchanges = 1
        built = self.room._build_matrices(dict(self.params))
        # the make-up and supply CO2 columns are always there (unwired, they
        # read the outdoor CO2); the envelope inflow's column is last
        expected = mass_matrices(
            _t(120.0), _t(5e-6), _t(0.004), 1, make_up_co2_slot=MAKE_UP_CO2_SLOT, n_exchanges=1, supply_co2=True,
            infiltration_column=True,
        )
        for a, b in zip(built, expected):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

        u = [name for name, _ in self.room._ss_layout()["u"]]
        self.assertEqual(u[SUPPLY_CO2_SLOT : SUPPLY_CO2_SLOT + 2], ["supplyAirCO2", "exchangeCO2Gain"])
        self.assertIn((0, 0, SUPPLY_CO2_SLOT), self.room._ss_support()["F"])
        A, B, C, D, E, F = self.room._build_matrices(dict(self.params))
        self.assertEqual(B.shape[-1], N_INPUTS + 3)
        self.assertEqual(float(F[0, 0, 0, OUTDOOR_CO2_SLOT]), 0.0)
        self.assertAlmostEqual(float(F[0, 0, 0, SUPPLY_CO2_SLOT]), 1.0 / (120.0 * float(constants.RHO_AIR)), places=12)
        # the exchange slot keeps its linear source, now one column later
        self.assertAlmostEqual(float(B[0, 0, SUPPLY_CO2_SLOT + 1]), 1.0 / (120.0 * float(constants.RHO_AIR)), places=12)
        self.assertEqual(float(B[0, 0, SUPPLY_CO2_SLOT]), 0.0)


class TestCompositeWiring(unittest.TestCase):
    """A ``BuildingSpaceSystem``'s submodels see the optional ports wired on
    the composite (the connections are the composite's, not theirs)."""

    def _room(self, wired):
        model = tb.Model(id=f"composite_wiring_{wired}")
        room = tb.BuildingSpaceSystem(id="Room")
        ports = [
            ("outdoorTemperature", 0.0),
            ("supplyAirTemperature", 20.0),
            ("globalIrradiation", 0.0),
            ("numberOfPeople", 0.0),
            ("heatGain", 0.0),
            ("outdoorCo2Concentration", 400.0),
        ]
        if wired:
            ports += [("makeUpAirTemperature", 18.0), ("makeUpAirCO2", 700.0), ("supplyAirCO2", 600.0)]
        for port, value in ports:
            schedule = tb.ScheduleSystem(weekday_ruleset={"ruleset_default_value": value}, id=f"S_{port}")
            model.add_connection(schedule, room, "scheduleValue", port)
        for port, value in (("supplyAirFlowRate", 0.1), ("exhaustAirFlowRate", 0.2)):
            schedule = tb.ScheduleSystem(weekday_ruleset={"ruleset_default_value": value}, id=f"S_{port}")
            model.add_connection(schedule, room, "scheduleValue", port, input_port_index=0)
        model.load(draw_semantic_model=False, draw_simulation_model=False)
        start = START.replace(tzinfo=datetime.timezone.utc)
        tb.Simulator(model).simulate(
            start_time=start, end_time=start + datetime.timedelta(hours=6), step_size=int(DT), show_progress_bar=False
        )
        return room

    def test_the_submodels_see_the_wired_ports(self):
        room = self._room(wired=True)
        self.assertEqual(room.input_defaults(), {})
        # 0.1 kg/s at 600 ppm and 0.1 kg/s at 700 ppm (plus a little outdoor
        # infiltration): the room settles near 650, not at the outdoor 400
        co2 = float(room.output["indoorCO2"].get().reshape(-1)[0])
        self.assertGreater(co2, 600.0)
        unwired = self._room(wired=False)
        self.assertEqual(
            unwired.input_defaults(),
            {"makeUpAirTemperature": "outdoorTemperature", "makeUpAirCO2": "outdoorCo2Concentration",
             "supplyAirCO2": "outdoorCo2Concentration"},
        )
        self.assertLess(float(unwired.output["indoorCO2"].get().reshape(-1)[0]), 450.0)


class TestThermalMakeUpPort(unittest.TestCase):
    T_I, T_O, T_SUP = 20.0, 5.0, 30.0

    def setUp(self):
        self.zone = tb.BuildingSpaceThermalSystem(
            C_air=1e5, C_wall=5e6, R_out=0.01, R_in=0.01, f_wall=0.0, f_air=0.0,
            Q_occ_gain=0.0, id="zone",
        )
        self.zone.initialize([START], [START + datetime.timedelta(hours=1)], [DT])
        self.params = self.zone._forward_params()
        self.x0 = self.zone.get_state()

    def _step(self, m_sup, m_exh, make_up=None, wired=False):
        inputs = {
            "outdoorTemperature": _t(self.T_O),
            "supplyAirFlowRate": _t(m_sup),
            "exhaustAirFlowRate": _t(m_exh),
            "supplyAirTemperature": _t(self.T_SUP),
            "globalIrradiation": _t(0.0),
            "numberOfPeople": _t(0.0),
            "heatGain": _t(0.0),
        }
        if make_up is not None and wired:  # unwired, the engine passes the outdoor value
            inputs["makeUpAirTemperature"] = _t(make_up)
        x1, _ = self.zone.forward(self.x0, self.zone.with_defaults(inputs), self.zone.full_params(dict(self.params), DT), DT)
        return float(x1[..., 0].reshape(-1)[0])

    def test_unwired_ignores_the_port(self):
        self.assertEqual(self._step(0.1, 0.18), self._step(0.1, 0.18, make_up=22.0, wired=False))

    def test_wired_at_outdoor_equals_unwired(self):
        self.assertAlmostEqual(
            self._step(0.1, 0.18), self._step(0.1, 0.18, make_up=self.T_O, wired=True), places=12
        )

    def test_wired_deficit_draws_the_make_up_state(self):
        # Corridor air at 22 C cools the room less than outdoor air at 5 C.
        t_out = self._step(0.1, 0.18, make_up=self.T_O, wired=True)
        t_corr = self._step(0.1, 0.18, make_up=22.0, wired=True)
        self.assertGreater(t_corr, t_out)

    @_exact
    def test_wired_surplus_never_sees_the_port(self):
        self.assertAlmostEqual(
            self._step(0.1, 0.03, make_up=self.T_O, wired=True),
            self._step(0.1, 0.03, make_up=22.0, wired=True),
            places=12,
        )

    def test_layout_and_support_carry_the_new_slot(self):
        u = [name for name, _ in self.zone._ss_layout()["u"]]
        self.assertEqual(u[self.zone.MAKE_UP_TEMPERATURE_SLOT], "makeUpAirTemperature")
        self.assertIn((2, 0, self.zone.MAKE_UP_TEMPERATURE_SLOT), self.zone._ss_support()["F"])
        self.assertIn((2, 0, 0), self.zone._ss_support()["F"])


class TestOutdoorAirRecovery(TestThermalMakeUpPort):
    """``eta_inf``: the outdoor air arrives at ``T_o + eta (T_i - T_o)`` -- the infiltration and an outdoor make-up
    stream carry ``1 - eta`` of their heat; a make-up stream from a neighbouring zone (``makeUpAirTemperature``
    wired) is not touched."""

    def setUp(self):
        super().setUp()
        # the same zone with its make-up air wired (a neighbouring zone's temperature)
        model = tb.Model(id="outdoor_air_recovery_wired")
        self.wired_zone = tb.BuildingSpaceThermalSystem(
            C_air=1e5, C_wall=5e6, R_out=0.01, R_in=0.01, f_wall=0.0, f_air=0.0, Q_occ_gain=0.0, id="zone",
        )
        neighbour = tb.ScheduleSystem(weekday_ruleset={"ruleset_default_value": 22.0}, id="Neighbour")
        model.add_connection(neighbour, self.wired_zone, "scheduleValue", "makeUpAirTemperature")
        self.wired_zone.initialize([START], [START + datetime.timedelta(hours=1)], [DT])

    def _step(self, m_sup, m_exh, make_up=None, wired=False):
        # a wired step runs the zone whose make-up air is wired
        zone, self.zone = self.zone, self.wired_zone if wired else self.zone
        try:
            return super()._step(m_sup, m_exh, make_up=make_up, wired=wired)
        finally:
            self.zone = zone

    def _with(self, eta, m_inf, m_sup, m_exh, make_up=None, wired=False):
        self.params = dict(self.zone._forward_params(), eta_inf=_t(eta), m_inf=_t(m_inf))
        return self._step(m_sup, m_exh, make_up=make_up, wired=wired)

    def test_the_default_is_outdoor_air(self):
        self.assertIn("eta_inf", self.zone._forward_params())
        self.assertEqual(float(self.zone._forward_params()["eta_inf"].reshape(-1)[0]), 0.0)
        self.assertEqual(self.zone.parameter["eta_inf"], {"lb": 0.0, "ub": 1.0})
        self.assertTrue(self.zone._make_up_is_outdoor_air())
        self.assertFalse(self.wired_zone._make_up_is_outdoor_air())

    @_exact
    def test_full_recovery_removes_the_outdoor_air_heat(self):
        # eta 1: neither the infiltration nor the outdoor make-up stream (0.08 kg/s) carries heat -- as no
        # infiltration and an exhaust that the supply covers
        self.assertAlmostEqual(self._with(1.0, 0.01, 0.1, 0.18), self._with(0.0, 0.0, 0.1, 0.1), places=12)

    def test_recovery_warms_monotonically(self):
        cold, half, full = (self._with(eta, 0.01, 0.1, 0.18) for eta in (0.0, 0.5, 1.0))
        self.assertLess(cold, half)
        self.assertLess(half, full)

    @_exact
    def test_a_wired_make_up_stream_is_not_touched(self):
        # wired: only the infiltration takes the factor
        self.assertAlmostEqual(
            self._with(1.0, 0.01, 0.1, 0.18, make_up=22.0, wired=True),
            self._with(0.0, 0.0, 0.1, 0.18, make_up=22.0, wired=True),
            places=12,
        )

    def test_a_room_hands_its_wiring_to_its_thermal_model(self):
        for wired in (False, True):
            with self.subTest(wired=wired):
                room = TestCompositeWiring._room(self, wired)
                self.assertEqual(room.thermal._make_up_is_outdoor_air(), not wired)


if __name__ == "__main__":
    unittest.main()
