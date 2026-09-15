"""Parameters the wiring leaves without effect stay out of theta.

The in-zone boundary wall (``C_boundary`` / ``R_boundary``) only acts when
a ``boundaryTemperature`` is connected; otherwise the residual Jacobian
column is identically zero and the estimator would carry two dead entries
per room.  ``System.get_estimable_parameters`` now honours an owner's
``_inactive_parameters()``.
"""

# Standard library imports
import unittest

# Local application imports
import twin4build
import twin4build as tb

twin4build._IS_TESTING = True


def _attrs(entries):
    return {attr for _, attr, *_ in entries}


class TestInactiveBoundaryParameters(unittest.TestCase):
    def test_unwired_boundary_is_skipped_on_the_composite(self):
        space = tb.BuildingSpaceSystem(id="room", airVolume=100.0)
        attrs = _attrs(space.get_estimable_parameters())
        self.assertIn("thermal.C_air", attrs)
        self.assertIn("thermal.R_out", attrs)
        self.assertNotIn("thermal.C_boundary", attrs)
        self.assertNotIn("thermal.R_boundary", attrs)

    def test_unwired_boundary_is_skipped_on_the_thermal_unit(self):
        zone = tb.BuildingSpaceThermalSystem(id="zone")
        attrs = _attrs(zone.get_estimable_parameters())
        self.assertIn("C_wall", attrs)
        self.assertNotIn("C_boundary", attrs)
        self.assertNotIn("R_boundary", attrs)

    def test_connected_boundary_keeps_the_parameters(self):
        model = tb.Model(id="boundary_model")
        zone = tb.BuildingSpaceThermalSystem(id="zone")
        boundary = tb.ScheduleSystem(
            weekday_ruleset={"ruleset_default_value": 15.0}, id="Boundary"
        )
        model.add_connection(boundary, zone, "scheduleValue", "boundaryTemperature")
        attrs = _attrs(zone.get_estimable_parameters())
        self.assertIn("C_boundary", attrs)
        self.assertIn("R_boundary", attrs)

    def test_manual_setup_keeps_the_parameters(self):
        zone = tb.BuildingSpaceThermalSystem(id="zone")
        zone.n_boundary_temperature = 1
        attrs = _attrs(zone.get_estimable_parameters())
        self.assertIn("C_boundary", attrs)
        zone.n_boundary_temperature = 0
        self.assertNotIn("C_boundary", _attrs(zone.get_estimable_parameters()))


if __name__ == "__main__":
    unittest.main()
