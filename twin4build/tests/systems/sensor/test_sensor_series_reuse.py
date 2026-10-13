"""A sensor keeps its series input while its source is unchanged.

``TimeSeriesInputSystem`` caches its series by window (start, end, step) and
skips the load when initialized again over the same windows.  The sensor
built a new input at every ``initialize``, so the cache was always empty:
every simulate of a model loaded every sensor again (from the database, the
disk cache or an in-memory frame).
"""

# Standard library imports
import datetime
import unittest
from unittest import mock
from zoneinfo import ZoneInfo

# Third party imports
import pandas as pd
import torch

# Local application imports
import twin4build
import twin4build.systems.utils.time_series_input_system as tsi_mod
from twin4build.systems.sensor.sensor_system import SensorSystem

twin4build._IS_TESTING = True

DBCONFIG = {
    "table_name": "measurements",
    "db_host": "localhost",
    "db_port": 5432,
    "db_name": "db",
    "db_user": "u",
    "db_password": "p",
}
TZ = ZoneInfo("Europe/Copenhagen")
STARTS = [datetime.datetime(2024, 3, 4, tzinfo=TZ), datetime.datetime(2024, 3, 6, tzinfo=TZ)]
ENDS = [datetime.datetime(2024, 3, 5, tzinfo=TZ), datetime.datetime(2024, 3, 7, tzinfo=TZ)]
STEPS = [600, 600]


def _series(start, end, step, value):
    index = pd.date_range(start, end, freq=f"{step}s", inclusive="left", name="time")
    return pd.Series(float(value), index=index, name="value")


def _database(**kwargs):
    """The database load of one window: a constant series."""
    return _series(kwargs["start_time"], kwargs["end_time"], kwargs["step_size"], 21.0)


def _values(sensor):
    return sensor.output["measuredValue"].history().detach().clone()


class TestSensorSeriesReuse(unittest.TestCase):
    def test_same_windows_load_once(self):
        sensor = SensorSystem(id="room_temperature", uuid="R01_TRU01", dbconfig=DBCONFIG)
        with mock.patch.object(tsi_mod, "load_from_database", side_effect=_database) as load:
            sensor.initialize(start_time=STARTS, end_time=ENDS, step_size=STEPS)
            first, kept = _values(sensor), sensor.time_series_input
            sensor.initialize(start_time=STARTS, end_time=ENDS, step_size=STEPS)
        self.assertEqual(load.call_count, len(STARTS))  # one load per window, none the second time
        self.assertIs(sensor.time_series_input, kept)
        torch.testing.assert_close(_values(sensor), first)

    def test_new_windows_load_again(self):
        sensor = SensorSystem(id="room_temperature", uuid="R01_TRU01", dbconfig=DBCONFIG)
        later = [t + datetime.timedelta(days=7) for t in STARTS], [t + datetime.timedelta(days=7) for t in ENDS]
        with mock.patch.object(tsi_mod, "load_from_database", side_effect=_database) as load:
            sensor.initialize(start_time=STARTS, end_time=ENDS, step_size=STEPS)
            sensor.initialize(start_time=later[0], end_time=later[1], step_size=STEPS)
        self.assertEqual(load.call_count, 2 * len(STARTS))

    def test_new_point_loads_again(self):
        sensor = SensorSystem(id="room_temperature", uuid="R01_TRU01", dbconfig=DBCONFIG)
        with mock.patch.object(tsi_mod, "load_from_database", side_effect=_database) as load:
            sensor.initialize(start_time=STARTS, end_time=ENDS, step_size=STEPS)
            sensor.uuid = "R01_TRU02"
            sensor.initialize(start_time=STARTS, end_time=ENDS, step_size=STEPS)
        self.assertEqual(load.call_count, 2 * len(STARTS))
        self.assertEqual(load.call_args.kwargs["sensor_id"], "R01_TRU02")

    def test_new_series_builds_a_new_input(self):
        # A cleaned series replaces the frame (``set_series``): the next initialize must sample the new one.
        sensor = SensorSystem(id="room_temperature")
        sensor.set_series(_series(STARTS[0], ENDS[-1], 600, 21.0))
        with mock.patch.object(tsi_mod, "sample_from_df", wraps=tsi_mod.sample_from_df) as sample:
            sensor.initialize(start_time=STARTS, end_time=ENDS, step_size=STEPS)
            sensor.initialize(start_time=STARTS, end_time=ENDS, step_size=STEPS)
            self.assertEqual(sample.call_count, len(STARTS))
            sensor.set_series(_series(STARTS[0], ENDS[-1], 600, 7.0))
            sensor.initialize(start_time=STARTS, end_time=ENDS, step_size=STEPS)
        self.assertEqual(sample.call_count, 2 * len(STARTS))
        torch.testing.assert_close(_values(sensor), torch.full_like(_values(sensor), 7.0))


if __name__ == "__main__":
    unittest.main()
