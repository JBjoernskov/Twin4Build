# Standard library imports
import datetime
import unittest
import warnings

# Third party imports
from dateutil import tz

# Local application imports
# Set test flag
import twin4build
from twin4build.systems.schedule.schedule_system import ScheduleSystem

twin4build._IS_TESTING = True


class TestScheduleSystem(unittest.TestCase):
    def setUp(self):
        self.schedule = ScheduleSystem(
            weekDayRulesetDict={
                "ruleset_start_minute": [0],
                "ruleset_end_minute": [0],
                "ruleset_start_hour": [0],
                "ruleset_end_hour": [1],
                "ruleset_value": [20],
                "ruleset_default_value": 0,
            },
            id="test_schedule",
        )

    def test_initialization(self):
        """Test schedule system initialization."""
        self.assertIsNotNone(self.schedule)
        self.assertEqual(self.schedule.id, "test_schedule")

    def test_do_step(self):
        """Test schedule system do_step method."""
        # Initialize
        start_time = [datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=tz.UTC)]
        end_time = [datetime.datetime(2023, 1, 1, 1, 40, 0, tzinfo=tz.UTC)]
        step_size = [600]
        self.schedule.initialize(
            start_time=start_time, end_time=end_time, step_size=step_size
        )

        # Execute a time step
        datetime_val = datetime.datetime(2023, 1, 1, 0, 30, 0, tzinfo=tz.UTC)
        self.schedule.do_step(
            second_time=1800, date_time=datetime_val, step_size=600, step_index=3
        )

        # Check that output was set
        self.assertIsNotNone(self.schedule.output["scheduleValue"].get())

    def test_do_step_batch(self):
        """Test schedule system do_step method with batch size > 1."""
        schedule_batch = ScheduleSystem(
            weekDayRulesetDict={
                "ruleset_start_minute": [0],
                "ruleset_end_minute": [0],
                "ruleset_start_hour": [0],
                "ruleset_end_hour": [1],
                "ruleset_value": [20],
                "ruleset_default_value": 0,
            },
            id="test_schedule_batch",
        )

        start_time = [datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=tz.UTC)]
        end_time = [datetime.datetime(2023, 1, 1, 1, 40, 0, tzinfo=tz.UTC)]
        step_size = [600]
        schedule_batch.initialize(
            start_time=start_time, end_time=end_time, step_size=step_size
        )

        # Execute a time step
        datetime_val = datetime.datetime(2023, 1, 1, 0, 30, 0, tzinfo=tz.UTC)
        schedule_batch.do_step(
            second_time=1800, date_time=datetime_val, step_size=600, step_index=3
        )

        # Check that output was set
        output = schedule_batch.output["scheduleValue"].get()
        self.assertIsNotNone(output)

    def test_auto_detect_use_dict(self):
        """Test that use_dict is auto-detected when weekDayRulesetDict is provided."""
        schedule = ScheduleSystem(
            id="test_auto_dict",
            weekDayRulesetDict={
                "ruleset_start_minute": [0],
                "ruleset_end_minute": [0],
                "ruleset_start_hour": [0],
                "ruleset_end_hour": [1],
                "ruleset_value": [20],
                "ruleset_default_value": 0,
            },
        )
        self.assertTrue(schedule.use_dict)
        self.assertFalse(schedule.use_spreadsheet)
        self.assertFalse(schedule.use_database)

    def test_deprecated_usedict_constructor_warning(self):
        """Test that usedict in constructor shows deprecation warning."""
        with self.assertWarns(DeprecationWarning):
            schedule = ScheduleSystem(
                id="test_deprecated_usedict",
                weekDayRulesetDict={
                    "ruleset_start_minute": [0],
                    "ruleset_end_minute": [0],
                    "ruleset_start_hour": [0],
                    "ruleset_end_hour": [1],
                    "ruleset_value": [20],
                    "ruleset_default_value": 0,
                },
                usedict=True,
            )
        self.assertTrue(schedule.use_dict)

    def test_deprecated_usedict_property_warning(self):
        """Test that accessing usedict property shows deprecation warning."""
        with self.assertWarns(DeprecationWarning):
            _ = self.schedule.usedict

    def test_deprecated_usedict_setter_warning(self):
        """Test that setting usedict property shows deprecation warning."""
        with self.assertWarns(DeprecationWarning):
            self.schedule.usedict = False
        self.assertFalse(self.schedule.use_dict)

    def test_deprecated_useSpreadsheet_property_warning(self):
        """Test that accessing useSpreadsheet property shows deprecation warning."""
        with self.assertWarns(DeprecationWarning):
            _ = self.schedule.useSpreadsheet

    def test_deprecated_useDatabase_property_warning(self):
        """Test that accessing useDatabase property shows deprecation warning."""
        with self.assertWarns(DeprecationWarning):
            _ = self.schedule.useDatabase

    def test_only_one_flag_allowed(self):
        """Test that only one of use_spreadsheet, use_database, use_dict can be True."""
        with self.assertRaises(AssertionError):
            ScheduleSystem(
                id="test_multiple_flags",
                use_dict=True,
                use_spreadsheet=True,
            )

    def test_backward_compat_camelcase(self):
        """Test that old camelCase code still works."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            schedule = ScheduleSystem(
                id="test_backward_compat",
                weekDayRulesetDict={
                    "ruleset_start_minute": [0],
                    "ruleset_end_minute": [0],
                    "ruleset_start_hour": [0],
                    "ruleset_end_hour": [1],
                    "ruleset_value": [20],
                    "ruleset_default_value": 0,
                },
                usedict=True,
            )
            self.assertTrue(schedule.usedict)
            schedule.useSpreadsheet = False
            self.assertFalse(schedule.useSpreadsheet)

    def test_caching_with_different_datetime_instances(self):
        """Test that caching works reliably with different datetime instances having same values."""
        # Standard library imports
        import os
        import shutil

        # Third party imports
        import torch

        # Local application imports
        import twin4build as tb
        from twin4build.model.model import Model

        # Set test flag
        tb._IS_TESTING = True

        # Create model with schedule system that has noise
        model = Model(id="test_cache_model")
        schedule = ScheduleSystem(
            weekDayRulesetDict={
                "ruleset_start_minute": [0],
                "ruleset_end_minute": [0],
                "ruleset_start_hour": [0],
                "ruleset_end_hour": [1],
                "ruleset_value": [20],
                "ruleset_default_value": 0,
            },
            add_noise=True,
            id="test_schedule_cache",
        )
        model.add_component(schedule)
        model.load()

        simulator = tb.Simulator(model)

        # First simulation with datetime objects
        start_time1 = datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=tz.UTC)
        end_time1 = datetime.datetime(2023, 1, 1, 1, 0, 0, tzinfo=tz.UTC)
        step_size1 = 600

        simulator.simulate(
            start_time=start_time1,
            end_time=end_time1,
            step_size=step_size1,
            show_progress_bar=False,
        )

        # Get results from first simulation
        history1 = schedule.output["scheduleValue"].history().clone()

        # Second simulation with different datetime instances but same values
        start_time2 = datetime.datetime(
            2023, 1, 1, 0, 0, 0, tzinfo=tz.UTC
        )  # Different object, same value
        end_time2 = datetime.datetime(
            2023, 1, 1, 1, 0, 0, tzinfo=tz.UTC
        )  # Different object, same value
        step_size2 = 600

        # Verify they are different objects but equal values
        self.assertIsNot(start_time1, start_time2)
        self.assertEqual(start_time1, start_time2)
        self.assertIsNot(end_time1, end_time2)
        self.assertEqual(end_time1, end_time2)

        simulator.simulate(
            start_time=start_time2,
            end_time=end_time2,
            step_size=step_size2,
            show_progress_bar=False,
        )

        # Get results from second simulation
        history2 = schedule.output["scheduleValue"].history()

        # Verify outputs are identical (caching works with different datetime instances)
        torch.testing.assert_close(
            history1,
            history2,
            msg="Simulation results differ with different datetime instances: caching not working reliably",
        )

        # Cleanup
        if os.path.exists("generated_files/models/test_cache_model"):
            shutil.rmtree("generated_files/models/test_cache_model")


if __name__ == "__main__":
    unittest.main()
