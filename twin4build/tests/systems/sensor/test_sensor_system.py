# Standard library imports
import datetime
import unittest
import warnings

# Third party imports
import pandas as pd
from dateutil import tz

# Local application imports
from twin4build.systems.sensor.sensor_system import SensorSystem


class TestSensorSystem(unittest.TestCase):
    def setUp(self):
        dates = pd.date_range(
            start=datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=tz.UTC),
            periods=10,
            freq="10min",
        )
        df = pd.DataFrame({"value": [20.0] * 10}, index=dates)
        self.sensor = SensorSystem(id="test_sensor", df=df)

    def test_initialization(self):
        """Test sensor system initialization."""
        self.assertIsNotNone(self.sensor)
        self.assertEqual(self.sensor.id, "test_sensor")

    def test_do_step(self):
        """Test sensor system do_step method."""
        start_time = [datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=tz.UTC)]
        end_time = [datetime.datetime(2023, 1, 1, 1, 40, 0, tzinfo=tz.UTC)]
        step_size = [600]
        self.sensor.initialize(
            start_time=start_time, end_time=end_time, step_size=step_size
        )

        # Execute a time step
        datetime_val = datetime.datetime(2023, 1, 1, 0, 10, 0, tzinfo=tz.UTC)
        self.sensor.do_step(
            second_time=600, date_time=datetime_val, step_size=600, step_index=1
        )

        # Check that measured value was set
        measured = self.sensor.output["measuredValue"].get()
        self.assertIsNotNone(measured)
        self.assertAlmostEqual(measured.item(), 20.0, places=1)

    def test_do_step_batch(self):
        """Test sensor system do_step method with batch size > 1."""
        dates = pd.date_range(
            start=datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=tz.UTC),
            periods=10,
            freq="10min",
        )
        df = pd.DataFrame({"value": [20.0] * 10}, index=dates)
        sensor_batch = SensorSystem(id="test_sensor_batch", df=df)

        start_time = [datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=tz.UTC)]
        end_time = [datetime.datetime(2023, 1, 1, 1, 40, 0, tzinfo=tz.UTC)]
        step_size = [600]
        sensor_batch.initialize(
            start_time=start_time, end_time=end_time, step_size=step_size
        )

        # Execute a time step
        datetime_val = datetime.datetime(2023, 1, 1, 0, 10, 0, tzinfo=tz.UTC)
        sensor_batch.do_step(
            second_time=600, date_time=datetime_val, step_size=600, step_index=1
        )

        # Check that measured value was set
        measured = sensor_batch.output["measuredValue"].get()
        self.assertIsNotNone(measured)

    def test_auto_detect_use_df(self):
        """Test that use_df is auto-detected when df is provided."""
        self.assertTrue(self.sensor.use_df)
        self.assertFalse(self.sensor.use_spreadsheet)
        self.assertFalse(self.sensor.use_database)

    def test_deprecated_usedf_constructor_warning(self):
        """Test that usedf in constructor shows deprecation warning."""
        dates = pd.date_range(
            start=datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=tz.UTC),
            periods=10,
            freq="10min",
        )
        df = pd.DataFrame({"value": [20.0] * 10}, index=dates)
        with self.assertWarns(DeprecationWarning):
            sensor = SensorSystem(id="test_deprecated_usedf", df=df, usedf=True)
        self.assertTrue(sensor.use_df)

    def test_deprecated_usedf_property_warning(self):
        """Test that accessing usedf property shows deprecation warning."""
        with self.assertWarns(DeprecationWarning):
            _ = self.sensor.usedf

    def test_deprecated_usedf_setter_warning(self):
        """Test that setting usedf property shows deprecation warning."""
        with self.assertWarns(DeprecationWarning):
            self.sensor.usedf = False
        self.assertFalse(self.sensor.use_df)

    def test_deprecated_useSpreadsheet_property_warning(self):
        """Test that accessing useSpreadsheet property shows deprecation warning."""
        with self.assertWarns(DeprecationWarning):
            _ = self.sensor.useSpreadsheet

    def test_deprecated_useDatabase_property_warning(self):
        """Test that accessing useDatabase property shows deprecation warning."""
        with self.assertWarns(DeprecationWarning):
            _ = self.sensor.useDatabase

    def test_only_one_flag_allowed(self):
        """Test that only one of use_spreadsheet, use_database, use_df can be True."""
        dates = pd.date_range(
            start=datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=tz.UTC),
            periods=10,
            freq="10min",
        )
        df = pd.DataFrame({"value": [20.0] * 10}, index=dates)
        with self.assertRaises(AssertionError):
            SensorSystem(
                id="test_multiple_flags",
                df=df,
                use_df=True,
                use_spreadsheet=True,
            )

    def test_df_property_auto_sets_use_df(self):
        """Test that setting df property auto-sets use_df=True."""
        dates = pd.date_range(
            start=datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=tz.UTC),
            periods=10,
            freq="10min",
        )
        df = pd.DataFrame({"value": [20.0] * 10}, index=dates)
        sensor = SensorSystem(id="test_df_setter")
        sensor.df = df
        self.assertTrue(sensor.use_df)
        self.assertFalse(sensor.use_spreadsheet)
        self.assertFalse(sensor.use_database)

    def test_backward_compat_camelcase(self):
        """Test that old camelCase code still works."""
        dates = pd.date_range(
            start=datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=tz.UTC),
            periods=10,
            freq="10min",
        )
        df = pd.DataFrame({"value": [20.0] * 10}, index=dates)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            sensor = SensorSystem(id="test_backward_compat", df=df, usedf=True)
            self.assertTrue(sensor.usedf)
            sensor.useSpreadsheet = False
            self.assertFalse(sensor.useSpreadsheet)


if __name__ == "__main__":
    unittest.main()
