"""``makeUpAirCO2`` / ``makeUpAirTemperature``: unwired is the classic
balance bit for bit; wired, the make-up stream carries the port's state and
only the exhaust-deficit case can see it.  ``supplyAirCO2`` the same way for
the supply stream."""

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
        self.room._make_up_wired = wired
        self.room._fwd_mat_cache = None  # matrices are cached per params identity
        inputs = {
            "supplyAirFlowRate": _t(m_sup),
            "exhaustAirFlowRate": _t(m_exh),
            "outdoorCo2Concentration": _t(400.0),
            "numberOfPeople": _t(0.0),
        }
        if make_up is not None:
            inputs["makeUpAirCO2"] = _t(make_up)
        x1, _ = self.room.forward(self.x0, inputs, dict(self.params), DT)
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
    """``supplyAirCO2``: unwired the supply stream brings outdoor air (the
    columns bit for bit); wired, it brings the port's concentration."""

    def setUp(self):
        self.room = tb.BuildingSpaceMassSystem(V=120.0, G_occ=5e-6, m_inf=0.004, id="room")
        self.room.n_c = 1
        self.params = {"V": _t(120.0), "G_occ": _t(5e-6), "m_inf": _t(0.004)}
        self.x0 = torch.full((1, 1), 900.0, dtype=torch.float64)

    def _step(self, m_sup, m_exh, supply=None, wired=False):
        self.room._supply_wired = wired
        self.room._fwd_mat_cache = None
        inputs = {
            "supplyAirFlowRate": _t(m_sup),
            "exhaustAirFlowRate": _t(m_exh),
            "outdoorCO2": _t(400.0),
            "numberOfPeople": _t(0.0),
        }
        if supply is not None:
            inputs["supplyAirCO2"] = _t(supply)
        x1, _ = self.room.forward(self.x0, inputs, dict(self.params), DT)
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
        unwired = self.room._build_matrices(dict(self.params))
        for a, b in zip(unwired, mass_matrices(_t(120.0), _t(5e-6), _t(0.004), 1, n_exchanges=1)):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.assertEqual(unwired[1].shape[-1], N_INPUTS + 1)

        self.room._supply_wired = True
        u = [name for name, _ in self.room._ss_layout()["u"]]
        self.assertEqual(u[SUPPLY_CO2_SLOT : SUPPLY_CO2_SLOT + 2], ["supplyAirCO2", "exchangeCO2Gain"])
        self.assertIn((0, 0, SUPPLY_CO2_SLOT), self.room._ss_support()["F"])
        A, B, C, D, E, F = self.room._build_matrices(dict(self.params))
        self.assertEqual(B.shape[-1], N_INPUTS + 2)
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
            ("outdoorCO2", 400.0),
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
        self.assertTrue(room.thermal._make_up_wired)
        self.assertTrue(room.mass._make_up_wired)
        self.assertTrue(room.mass._supply_wired)
        # 0.1 kg/s at 600 ppm and 0.1 kg/s at 700 ppm (plus a little outdoor
        # infiltration): the room settles near 650, not at the outdoor 400
        co2 = float(room.output["indoorCO2"].get().reshape(-1)[0])
        self.assertGreater(co2, 600.0)
        unwired = self._room(wired=False)
        self.assertFalse(unwired.thermal._make_up_wired or unwired.mass._make_up_wired or unwired.mass._supply_wired)
        self.assertLess(float(unwired.output["indoorCO2"].get().reshape(-1)[0]), 450.0)


class TestThermalMakeUpPort(unittest.TestCase):
    T_I, T_O, T_SUP = 20.0, 5.0, 30.0

    def setUp(self):
        self.zone = tb.BuildingSpaceThermalSystem(
            C_air=1e5, C_wall=5e6, R_out=0.01, R_in=0.01, f_wall=0.0, f_air=0.0,
            Q_occ_gain=0.0, T_air_start=self.T_I, T_wall_start=self.T_I, id="zone",
        )
        self.zone.initialize([START], [START + datetime.timedelta(hours=1)], [DT])
        self.params = self.zone._forward_params()
        self.x0 = self.zone.ss_model.get_state()

    def _step(self, m_sup, m_exh, make_up=None, wired=False):
        self.zone._make_up_wired = wired
        self.zone._fwd_mat_cache = None
        inputs = {
            "outdoorTemperature": _t(self.T_O),
            "supplyAirFlowRate": _t(m_sup),
            "exhaustAirFlowRate": _t(m_exh),
            "supplyAirTemperature": _t(self.T_SUP),
            "globalIrradiation": _t(0.0),
            "numberOfPeople": _t(0.0),
            "heatGain": _t(0.0),
        }
        if make_up is not None:
            inputs["makeUpAirTemperature"] = _t(make_up)
        x1, _ = self.zone.forward(self.x0, inputs, dict(self.params), DT)
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


if __name__ == "__main__":
    unittest.main()
