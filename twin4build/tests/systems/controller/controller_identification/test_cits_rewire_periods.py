"""A rewire over periods of different lengths scores each period's own steps.

A sensor holds every period padded to the longest with its last value.  The
rewire used to score that padding: weeks of frozen values reward the loop that
matches them (a proportional loop on a constant setpoint) and penalise a loop
with integral action, which winds up over them.  A valve loop on a room's
scheduled heating setpoint then lost to a proportional loop on a flat one."""

# Standard library imports
import datetime
import unittest

# Third party imports
import numpy as np
import pandas as pd

# Local application imports
import twin4build as tb
from twin4build.systems.controller.controller_identification.controller_identification_pi_system import (
    ControllerIdentificationPISystem,
)
from twin4build.systems.controller.controller_identification.loop_fit import pi_outputs
from twin4build.systems.sensor.sensor_system import SensorSystem
from twin4build.utils.data_loaders.sources import DataFrameSource

tb._IS_TESTING = True

STEP = 600
DAY = 144


def _df(index, values):
    return pd.DataFrame({"value": np.asarray(values, dtype=float)}, index=index)


def _loop(n_setpoints):
    model = tb.Model(id=f"test_cits_rewire_periods_{n_setpoints}")
    cits = ControllerIdentificationPISystem(id="cits", n_sensors=1, n_setpoints=n_setpoints, n_on_off_signals=1)
    return model, cits


class TestRewireOverUnequalPeriods(unittest.TestCase):
    def setUp(self):
        # one day, then five days, two weeks apart: the first period is padded by four days
        self.periods = [
            (datetime.datetime(2021, 1, 4, tzinfo=datetime.timezone.utc), 1),
            (datetime.datetime(2021, 1, 18, tzinfo=datetime.timezone.utc), 5),
        ]
        self.index = pd.DatetimeIndex(
            np.concatenate([pd.date_range(s, periods=d * DAY, freq="10min") for s, d in self.periods])
        )
        n = len(self.index)
        hour = self.index.hour.to_numpy()
        # the room drifts on its own (a 23 h swing, not the schedule's day); the heating setpoint is 19 C at
        # night and 21 C by day; the valve is a PI on that setpoint, with integral action
        self.room = 20.0 + 0.6 * np.sin(2 * np.pi * np.arange(n) / 138) + 0.3 * np.sin(2 * np.pi * np.arange(n) / 31)
        self.heating = np.where((hour >= 6) & (hour < 18), 21.0, 19.0)
        # the short period ends with the room 0.8 K below its setpoint, as on a cold Sunday night: its padding
        # holds that, with the valve part open, for four days
        self.room[DAY - 12 : DAY] = self.heating[DAY - 12 : DAY] - 0.8
        self.command = pi_outputs(self.heating - self.room, np.array([0.1]), np.array([7200.0]), STEP)[:, 0]

    def _rewire(self, setpoints):
        model, cits = _loop(len(setpoints))
        sensor = SensorSystem(id="room", source=DataFrameSource(_df(self.index, self.room)))
        gate = SensorSystem(id="gate", source=DataFrameSource(_df(self.index, np.full(len(self.index), 1.0))))
        command = SensorSystem(id="valve", source=DataFrameSource(_df(self.index, self.command)))
        model.add_connection(sensor, cits, "measuredValue", "sensorValue", input_port_index=0)
        for k, (name, values) in enumerate(setpoints):
            sp = SensorSystem(id=name, source=DataFrameSource(_df(self.index, values)))
            model.add_connection(sp, cits, "measuredValue", "setpointValue", input_port_index=k)
        model.add_connection(gate, cits, "measuredValue", "onOffSignal", input_port_index=0)
        model.add_connection(cits, command, "inputSignal", "measuredValue", output_port_index=0)
        model.rewire(
            start_time=[s for s, _ in self.periods],
            end_time=[s + datetime.timedelta(days=d) for s, d in self.periods],
            step_size=STEP,
            mode="train",
        )
        return model.simulation_model.rewire_reports["cits"]

    def test_only_each_periods_own_steps_are_scored(self):
        report = self._rewire([("heating", self.heating)])
        own = sum(d * DAY for _, d in self.periods)  # 6 days, not 2 x 5 days with the padding
        self.assertEqual(report.score.n_active, own)

    def test_the_scheduled_setpoint_and_the_integral_action_are_found(self):
        report = self._rewire([("flat", np.full(len(self.index), 22.0)), ("heating", self.heating)])
        self.assertEqual(report.winner[1], "heating")
        self.assertTrue(report.is_reverse)
        self.assertLess(report.Ti_x0, 1e5)


if __name__ == "__main__":
    unittest.main()
