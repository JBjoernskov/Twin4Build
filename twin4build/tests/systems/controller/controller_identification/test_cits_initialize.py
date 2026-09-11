"""``ControllerIdentificationSystem.initialize`` must size its ``onOffSignal``
Vector with ``n_v`` (regression: it passed ``size=``, a leftover from an
API rename, and every Stage-1 controller identification run failed with
``TypeError: Vector.initialize() got an unexpected keyword argument 'size'``).
"""

# Standard library imports
import datetime
import unittest
from zoneinfo import ZoneInfo

# Local application imports
import twin4build
from twin4build.systems.controller.controller_identification.controller_identification_pi_system import (
    ControllerIdentificationPISystem,
)

twin4build._IS_TESTING = True


class TestCitsInitialize(unittest.TestCase):
    def test_initialize_sizes_on_off_vector(self):
        tz = ZoneInfo("Europe/Copenhagen")
        cits = ControllerIdentificationPISystem(
            n_sensors=1, n_setpoints=1, n_actuators=1, n_on_off_signals=1, id="cits"
        )
        start = datetime.datetime(2024, 3, 4, tzinfo=tz)
        end = datetime.datetime(2024, 3, 5, tzinfo=tz)
        cits.initialize(start_time=[start], end_time=[end], step_size=[600])
        self.assertEqual(cits.input["onOffSignal"].n_v, 1)
        self.assertEqual(cits.input["sensorValue"].n_v, 1)
        self.assertEqual(cits.output["inputSignal"].n_v, 1)


if __name__ == "__main__":
    unittest.main()
