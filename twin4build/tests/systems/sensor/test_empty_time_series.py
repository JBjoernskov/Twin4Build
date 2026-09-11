"""A sensor whose database query returns no rows in the simulated window
must fail with a message naming the sensor, not with numpy's
``could not broadcast input array from shape (0,0)``."""

# Standard library imports
import datetime
import unittest
from unittest import mock
from zoneinfo import ZoneInfo

# Third party imports
import pandas as pd

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


class TestEmptyTimeSeriesError(unittest.TestCase):
    def test_empty_database_result_raises_named_error(self):
        tz = ZoneInfo("Europe/Copenhagen")
        sensor = SensorSystem(
            id="lonely_sensor", uuid="POINT_WITHOUT_DATA", dbconfig=DBCONFIG
        )
        start = datetime.datetime(2024, 3, 4, tzinfo=tz)
        end = datetime.datetime(2024, 3, 5, tzinfo=tz)

        empty = pd.DataFrame(
            {"POINT_WITHOUT_DATA": []},
            index=pd.DatetimeIndex([], tz=tz, name="time"),
        )
        with mock.patch.object(tsi_mod, "load_from_database", return_value=empty):
            with self.assertRaises(ValueError) as ctx:
                sensor.initialize(
                    start_time=[start], end_time=[end], step_size=[600]
                )
        msg = str(ctx.exception)
        self.assertIn("lonely_sensor", msg)
        self.assertIn("POINT_WITHOUT_DATA", msg)


if __name__ == "__main__":
    unittest.main()
