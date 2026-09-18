"""``makeUpAirCO2`` / ``makeUpAirTemperature``: unwired is the classic
balance bit for bit; wired, the make-up stream carries the port's state and
only the exhaust-deficit case can see it."""

# Standard library imports
import datetime
import unittest

# Third party imports
import torch

# Local application imports
import twin4build
import twin4build as tb
from twin4build.systems.building_space.building_space_mass_system import (
    ACTIVE_INPUT_SLOTS,
    MAKE_UP_CO2_SLOT,
    N_INPUTS,
    OCCUPANCY_SLOT,
    OUTDOOR_CO2_SLOT,
)

twin4build._IS_TESTING = True

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
            "outdoorCO2": _t(400.0),
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

    def test_wired_surplus_never_sees_the_port(self):
        self.assertAlmostEqual(
            self._step(0.1, 0.03, make_up=400.0, wired=True),
            self._step(0.1, 0.03, make_up=800.0, wired=True),
            places=12,
        )


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
