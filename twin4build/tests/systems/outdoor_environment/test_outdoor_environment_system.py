# Standard library imports
import datetime
import unittest
import warnings

# Third party imports
import pandas as pd
from dateutil import tz
import torch

# Local application imports
from twin4build.systems.outdoor_environment.outdoor_environment_system import (
    OutdoorEnvironmentSystem,
)

# Set test flag
import twin4build
twin4build._IS_TESTING = True


class TestOutdoorEnvironmentSystem(unittest.TestCase):
    def setUp(self):
        self.outdoor_env = OutdoorEnvironmentSystem(id="test_outdoor")

    def test_initialization(self):
        """Test outdoor environment system initialization."""
        self.assertIsNotNone(self.outdoor_env)
        self.assertEqual(self.outdoor_env.id, "test_outdoor")

    def test_initialization_with_df(self):
        """Test outdoor environment system with DataFrame."""
        dates = pd.date_range(
            start=datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=tz.UTC),
            periods=10,
            freq="10min",
        )
        df = pd.DataFrame(
            {
                "outdoorTemperature": [5.0] * 10,
                "globalIrradiation": [100.0] * 10,
                "outdoorCo2Concentration": [400.0] * 10,
            },
            index=dates,
        )

        outdoor_env = OutdoorEnvironmentSystem(id="test_outdoor_df", df=df)
        self.assertIsNotNone(outdoor_env)
        self.assertIsNotNone(outdoor_env.df)

    def test_initialization_with_correction(self):
        """Test outdoor environment system with linear correction."""
        outdoor_env = OutdoorEnvironmentSystem(
            id="test_outdoor_correction", a=1.1, b=0.5, apply_correction=True
        )
        self.assertEqual(outdoor_env.a.get().item(), 1.1)
        self.assertEqual(outdoor_env.b.get().item(), 0.5)
        self.assertTrue(outdoor_env.apply_correction)

    def test_input_output_properties(self):
        """Test input and output property accessors."""
        self.assertIsInstance(self.outdoor_env.input, dict)
        self.assertIsInstance(self.outdoor_env.output, dict)

        # Check outputs exist
        self.assertIn("outdoorTemperature", self.outdoor_env.output)
        self.assertIn("globalIrradiation", self.outdoor_env.output)
        self.assertIn("outdoorCo2Concentration", self.outdoor_env.output)

    def test_do_step_without_correction(self):
        """Test outdoor environment system do_step method without correction."""
        dates = pd.date_range(
            start=datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=tz.UTC),
            periods=10,
            freq="10min",
        )
        df = pd.DataFrame(
            {
                "outdoorTemperature": [5.0] * 10,
                "globalIrradiation": [100.0] * 10,
                "outdoorCo2Concentration": [400.0] * 10,
            },
            index=dates,
        )

        outdoor_env = OutdoorEnvironmentSystem(id="test_outdoor_step", df=df)

        start_time = [datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=tz.UTC)]
        end_time = [datetime.datetime(2023, 1, 1, 1, 0, 0, tzinfo=tz.UTC)]
        step_size = [600]

        outdoor_env.initialize(
            start_time=start_time, end_time=end_time, step_size=step_size
        )

        # Execute a time step
        datetime_val = datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=tz.UTC)
        outdoor_env.do_step(
            second_time=0, date_time=datetime_val, step_size=600, step_index=0
        )

        # Check that outputs were set
        temp = outdoor_env.output["outdoorTemperature"].get()
        irrad = outdoor_env.output["globalIrradiation"].get()
        co2 = outdoor_env.output["outdoorCo2Concentration"].get()

        self.assertIsNotNone(temp)
        self.assertIsNotNone(irrad)
        self.assertIsNotNone(co2)

    def test_do_step_with_correction(self):
        """Test outdoor environment system do_step with linear correction applied."""
        dates = pd.date_range(
            start=datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=tz.UTC),
            periods=10,
            freq="10min",
        )
        df = pd.DataFrame(
            {
                "outdoorTemperature": [10.0] * 10,
                "globalIrradiation": [100.0] * 10,
                "outdoorCo2Concentration": [400.0] * 10,
            },
            index=dates,
        )

        # Create with correction: y = 1.1 * x + 0.5
        outdoor_env = OutdoorEnvironmentSystem(
            id="test_outdoor_correction_step",
            df=df,
            a=1.1,
            b=0.5,
            apply_correction=True,
        )

        start_time = [datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=tz.UTC)]
        end_time = [datetime.datetime(2023, 1, 1, 1, 0, 0, tzinfo=tz.UTC)]
        step_size = [600]

        outdoor_env.initialize(
            start_time=start_time, end_time=end_time, step_size=step_size
        )

        # Execute a time step
        datetime_val = datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=tz.UTC)
        outdoor_env.do_step(
            second_time=0, date_time=datetime_val, step_size=step_size, step_index=0
        )

        # Check that correction was applied to temperature
        temp = outdoor_env.output["outdoorTemperature"].get()
        # Expected: 1.1 * 10.0 + 0.5 = 11.5
        self.assertAlmostEqual(temp.item(), 11.5, places=1)

    def test_initialize_without_data_raises_error(self):
        """Test that initialize raises error when no data source is provided."""
        outdoor_env = OutdoorEnvironmentSystem(id="test_no_data")

        start_time = [datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=tz.UTC)]
        end_time = [datetime.datetime(2023, 1, 1, 1, 0, 0, tzinfo=tz.UTC)]
        step_size = [600]

        with self.assertRaises(ValueError):
            outdoor_env.initialize(
                start_time=start_time, end_time=end_time, step_size=step_size
            )

    def test_apply_method(self):
        """Test the _apply method for linear correction."""
        outdoor_env = OutdoorEnvironmentSystem(
            id="test_apply", a=2.0, b=1.0, apply_correction=True
        )

        # Test _apply method directly
        result = outdoor_env._apply(torch.tensor(5.0))
        self.assertAlmostEqual(result.item(), 11.0, places=5)  # 2.0 * 5.0 + 1.0 = 11.0


if __name__ == "__main__":
    unittest.main()
