"""``OutdoorEnvironmentSystem.set_transformation_by_type`` routes a
per-Brick-class unit conversion to the matching feed.

Before, only the outdoor-temperature feed could receive a transformation
(``set_transformation``), so a rule keyed by ``Solar_Irradiance_Sensor``
(e.g. clipping the 1e8 W/m2 spikes a BMS irradiance sensor logs) was
silently dropped and ``Model.set_transformations`` picked a single rule per
component even when the component models sensors of different classes.
"""

# Standard library imports
import unittest

# Local application imports
import twin4build
import twin4build.core as core
from twin4build.model.semantic_model.semantic_model import SemanticModel
from twin4build.systems.outdoor_environment.outdoor_environment_system import (
    OutdoorEnvironmentSystem,
)

twin4build._IS_TESTING = True

BRICK = core.namespace.BRICK


class TestFeedTransformations(unittest.TestCase):
    def test_rules_reach_their_feeds(self):
        sm = SemanticModel(id="feed_transformations")
        outdoor = OutdoorEnvironmentSystem(id="outdoor")
        clip = lambda x: min(x, 1500.0)  # noqa: E731
        f2c = lambda x: (x - 32) * 5 / 9  # noqa: E731

        outdoor.set_transformation_by_type(sm.get_type(BRICK.Solar_Irradiance_Sensor), clip)
        outdoor.set_transformation_by_type(sm.get_type(BRICK.Outside_Air_Temperature_Sensor), f2c)
        self.assertIs(outdoor._transformation_globalIrradiation, clip)
        self.assertIs(outdoor._transformation_outdoorTemperature, f2c)

        # A rule keyed by the generic Temperature_Sensor still reaches the
        # temperature feed (subclass dispatch happens in Model).
        outdoor.set_transformation_by_type(sm.get_type(BRICK.Temperature_Sensor), clip)
        self.assertIs(outdoor._transformation_outdoorTemperature, clip)
        # Unrelated classes are ignored.
        outdoor.set_transformation_by_type(sm.get_type(BRICK.CO2_Sensor), f2c)
        self.assertIs(outdoor._transformation_globalIrradiation, clip)


if __name__ == "__main__":
    unittest.main()
