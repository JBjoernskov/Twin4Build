"""The occupancy inversion must be the exact inverse of the CO2 mass model.

``OccupancySystem.forward`` books the people that explain a measured CO2
step; ``BuildingSpaceMassSystem.forward`` propagates CO2 from people and
flows.  Both must use the balanced ventilation form (``air_balance``): a
known occupancy pushed through the forward model and inverted again must
come back, also when supply and exhaust differ.
"""

# Standard library imports
import unittest

# Third party imports
import torch

# Local application imports
import twin4build
import twin4build as tb
import twin4build.utils.constants as constants

twin4build._IS_TESTING = True

DT = 1.0  # the inversion is a forward difference; keep the step small
V, G_OCC, M_INF = 120.0, 5e-6, 0.004
C_OUT = 400.0


def _t(v):
    return torch.tensor([float(v)], dtype=torch.float64)


def _invert(m_sup, m_exh, c_prev, c_now):
    occ = tb.OccupancySystem(
        V=V, G_occ=G_OCC, m_inf=M_INF,
        supply_damper_nominalAirFlowRate=m_sup,
        exhaust_damper_nominalAirFlowRate=m_exh,
        id="occ",
    )
    occ._co2_from_sensor = False
    params = {
        "mass.V": _t(V), "mass.G_occ": _t(G_OCC), "mass.m_inf": _t(M_INF),
        "supply_damper.a": _t(1.0), "supply_damper.nominalAirFlowRate": _t(m_sup),
        "exhaust_damper.a": _t(1.0), "exhaust_damper.nominalAirFlowRate": _t(m_exh),
    }
    inputs = {
        "indoorCo2Measured": _t(c_now),
        "previousIndoorCo2Measured": _t(c_prev),
        "damperPositionMeasured": _t(1.0),  # fully open: flow = nominal
        "outdoorCo2Concentration": _t(C_OUT),
    }
    _, out = occ.forward(torch.zeros((1, 0), dtype=torch.float64), inputs, params, DT)
    return float(out["scheduleValue"])


def _forward_step(m_sup, m_exh, c_prev, n_people):
    room = tb.BuildingSpaceMassSystem(V=V, G_occ=G_OCC, m_inf=M_INF, id="room")
    room.n_c = 1
    params = {"V": _t(V), "G_occ": _t(G_OCC), "m_inf": _t(M_INF)}
    inputs = {
        "supplyAirFlowRate": _t(m_sup), "exhaustAirFlowRate": _t(m_exh),
        "outdoorCO2": _t(C_OUT), "numberOfPeople": _t(n_people),
    }
    x0 = torch.full((1, 1), c_prev, dtype=torch.float64)
    x1, _ = room.forward(x0, inputs, params, DT)
    return float(x1[0, 0])


class TestInversionRoundTrip(unittest.TestCase):
    def _round_trip(self, m_sup, m_exh, n_people=12.0, c_prev=900.0):
        c_now = _forward_step(m_sup, m_exh, c_prev, n_people)
        got = _invert(m_sup, m_exh, c_prev, c_now)
        # forward difference vs exact ZOH step: O(lambda*dt) ~ 1e-3 relative
        self.assertAlmostEqual(got / n_people, 1.0, places=2, msg=(m_sup, m_exh, got))

    def test_balanced(self):
        self._round_trip(0.10, 0.10)

    def test_surplus_supply(self):
        self._round_trip(0.10, 0.03)

    def test_exhaust_deficit(self):
        self._round_trip(0.10, 0.18)

    def test_surplus_supply_ignores_exhaust(self):
        # The forward model drops the exhaust when the supply covers it; so
        # must the inversion (same people for any smaller exhaust).
        c_now = _forward_step(0.10, 0.02, 900.0, 12.0)
        self.assertAlmostEqual(_invert(0.10, 0.02, 900.0, c_now), _invert(0.10, 0.08, 900.0, c_now), places=6)

    def test_old_unbalanced_form_would_disagree(self):
        # Guard against regressing to (m_inf+m_exh)*C - (m_inf+m_sup)*C_out.
        m_sup, m_exh, c_prev = 0.10, 0.18, 900.0
        c_now = _forward_step(m_sup, m_exh, c_prev, 12.0)
        alpha = G_OCC * (constants.M_AIR / constants.M_CO2) * 1e6
        old = (
            constants.RHO_AIR * V * (c_now - c_prev) / DT
            + (M_INF + m_exh) * c_prev
            - (M_INF + m_sup) * C_OUT
        ) / alpha
        self.assertGreater(abs(old - 12.0) / 12.0, 0.05)


if __name__ == "__main__":
    unittest.main()
