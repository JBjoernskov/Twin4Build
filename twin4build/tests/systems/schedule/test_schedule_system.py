# Standard library imports
import datetime
import unittest
import warnings

# Third party imports
import pytz

# Local application imports
from twin4build.systems.schedule.schedule_system import ScheduleSystem


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
        start_time = [datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)]
        end_time = [datetime.datetime(2023, 1, 1, 1, 40, 0, tzinfo=pytz.UTC)]
        step_size = [600]
        self.schedule.initialize(
            start_time=start_time, end_time=end_time, step_size=step_size
        )

        # Execute a time step
        datetime_val = datetime.datetime(2023, 1, 1, 0, 30, 0, tzinfo=pytz.UTC)
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

        start_time = [datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)]
        end_time = [datetime.datetime(2023, 1, 1, 1, 40, 0, tzinfo=pytz.UTC)]
        step_size = [600]
        schedule_batch.initialize(
            start_time=start_time, end_time=end_time, step_size=step_size
        )

        # Execute a time step
        datetime_val = datetime.datetime(2023, 1, 1, 0, 30, 0, tzinfo=pytz.UTC)
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


if __name__ == "__main__":
    unittest.main()

