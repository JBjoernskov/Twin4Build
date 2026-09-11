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

    def test_pi_gains_are_estimable(self):
        """``parameters="auto"`` must see kp / Ti of every PI candidate
        (regression: they were created frozen and silently skipped, so only
        the gate parameters were fitted)."""
        tz = ZoneInfo("Europe/Copenhagen")
        cits = ControllerIdentificationPISystem(
            n_sensors=1, n_setpoints=1, n_actuators=1, n_on_off_signals=1, id="cits"
        )
        start = datetime.datetime(2024, 3, 4, tzinfo=tz)
        end = datetime.datetime(2024, 3, 5, tzinfo=tz)
        cits.initialize(start_time=[start], end_time=[end], step_size=[600])
        names = [t[1] for t in cits.get_estimable_parameters()]
        self.assertIn("candidate_0_0.kp", names)
        self.assertIn("candidate_0_0.Ti", names)
        self.assertNotIn("candidate_0_0.Td", names)

    def test_estimable_bounds_follow_the_parameter(self):
        """Bounds written onto ``kp`` / ``Ti`` by the rewire seeding must be
        what the estimator sees (regression: the class constants were
        returned, so a rewired Ti = 7200 s > 1800 s was rejected as x0 > ub)."""
        import torch

        tz = ZoneInfo("Europe/Copenhagen")
        cits = ControllerIdentificationPISystem(
            n_sensors=1, n_setpoints=1, n_actuators=1, n_on_off_signals=1, id="cits"
        )
        start = datetime.datetime(2024, 3, 4, tzinfo=tz)
        end = datetime.datetime(2024, 3, 5, tzinfo=tz)
        cits.initialize(start_time=[start], end_time=[end], step_size=[600])
        ti = cits.candidate_0_0.Ti
        ti.min_value = torch.tensor(600.0, dtype=torch.float64)
        ti.max_value = torch.tensor(7200.0, dtype=torch.float64)
        ti.set(torch.tensor(7200.0, dtype=torch.float64), normalized=False)
        (entry,) = [t for t in cits.get_estimable_parameters() if t[1] == "candidate_0_0.Ti"]
        _, _, x0, lb, ub = entry
        self.assertAlmostEqual(x0, 7200.0, places=3)
        self.assertAlmostEqual(lb, 600.0)
        self.assertAlmostEqual(ub, 7200.0)


if __name__ == "__main__":
    unittest.main()
