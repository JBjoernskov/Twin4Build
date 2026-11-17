"""The in-zone boundary wall is gone in 2.0 (``C_boundary``, ``R_boundary``,
``boundaryTemperature``): a WallSystem on ``wallHeatGain`` takes its place.
Each old name raises and says so, and no room parameter is left without
effect in theta.
"""

# Standard library imports
import unittest

# Local application imports
import twin4build
import twin4build as tb

twin4build._IS_TESTING = True


def _attrs(entries):
    return {attr for _, attr, *_ in entries}


class TestBoundaryWallGone(unittest.TestCase):
    def test_the_old_values_raise_and_name_the_wall(self):
        with self.assertRaisesRegex(TypeError, "WallSystem"):
            tb.BuildingSpaceThermalSystem(id="zone", C_boundary=5e5)
        with self.assertRaisesRegex(TypeError, "WallSystem"):
            tb.BuildingSpaceSystem(id="room", thermal=dict(R_boundary=0.01))
        zone = tb.BuildingSpaceThermalSystem(id="zone")
        with self.assertRaisesRegex(AttributeError, "WallSystem"):
            zone.n_boundary_temperature = 1
        with self.assertRaisesRegex(AttributeError, "WallSystem"):
            zone.C_boundary

    def test_every_room_parameter_acts(self):
        space = tb.BuildingSpaceSystem(id="room", mass=dict(V=100.0))
        attrs = _attrs(space.get_estimable_parameters())
        self.assertEqual(
            attrs,
            {"thermal.C_air", "thermal.C_wall", "thermal.R_in", "thermal.R_out", "thermal.f_air",
             "thermal.f_wall", "thermal.Q_occ_gain", "thermal.eta_inf", "mass.V", "mass.G_occ", "mass.m_inf"},
        )
        self.assertNotIn("boundaryTemperature", space.input)


if __name__ == "__main__":
    unittest.main()
