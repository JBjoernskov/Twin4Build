# Standard library imports
import datetime
import unittest
import os
import shutil

# Third party imports
from dateutil import tz
import pandas as pd
import torch

# Local application imports
# Set test flag
import twin4build
from twin4build.systems.schedule.schedule_system import ScheduleSystem
from twin4build.model.model import Model
import twin4build as tb
from twin4build.utils.data_loaders.sources import DataFrameSource

twin4build._IS_TESTING = True



RULES = {
    "ruleset_start_minute": [0],
    "ruleset_end_minute": [0],
    "ruleset_start_hour": [0],
    "ruleset_end_hour": [1],
    "ruleset_value": [20],
    "ruleset_default_value": 0,
}

class TestScheduleSystem(unittest.TestCase):
    def setUp(self):
        self.schedule = ScheduleSystem(
            weekday_ruleset={
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
            weekday_ruleset={
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

    def test_a_ruleset_schedule_has_no_source(self):
        schedule = ScheduleSystem(id="test_rules", weekday_ruleset=RULES)
        self.assertIsNone(schedule.source)

    def test_removed_usedict_constructor_raises(self):
        """Old camelCase constructor kwargs are hard-removed."""
        with self.assertRaises(TypeError):
            ScheduleSystem(
                id="test_removed_usedict",
                weekday_ruleset={
                    "ruleset_start_minute": [0],
                    "ruleset_end_minute": [0],
                    "ruleset_start_hour": [0],
                    "ruleset_end_hour": [1],
                    "ruleset_value": [20],
                    "ruleset_default_value": 0,
                },
                usedict=True,
            )

    def test_removed_usedict_property_raises(self):
        """Old camelCase properties are hard-removed."""
        with self.assertRaises(AttributeError):
            _ = self.schedule.usedict
        with self.assertRaises(AttributeError):
            _ = self.schedule.useSpreadsheet
        with self.assertRaises(AttributeError):
            _ = self.schedule.useDatabase

    def test_the_old_source_flags_are_unknown_keywords(self):
        for flag in ("use_dict", "use_spreadsheet", "use_database"):
            with self.subTest(flag), self.assertRaisesRegex(TypeError, flag):
                ScheduleSystem(id="test_flags", weekday_ruleset=RULES, **{flag: True})

    def test_a_schedule_from_a_reader(self):
        index = pd.date_range("2023-01-01", periods=7, freq="10min", tz="UTC")
        schedule = ScheduleSystem(id="test_reader", source=DataFrameSource(pd.Series(range(7), index=index, dtype=float)))
        start = datetime.datetime(2023, 1, 1, tzinfo=tz.UTC)
        schedule.initialize([start], [start + datetime.timedelta(hours=1)], [600])
        torch.testing.assert_close(
            schedule.output["scheduleValue"].history().reshape(-1), torch.arange(6, dtype=torch.float64)
        )

    def test_a_day_without_a_ruleset_raises(self):
        schedule = ScheduleSystem(id="test_weekend_only", weekend_ruleset=RULES)
        start = datetime.datetime(2023, 1, 2, tzinfo=tz.UTC)  # a Monday
        with self.assertRaisesRegex(ValueError, "monday"):
            schedule.initialize([start], [start + datetime.timedelta(hours=1)], [600])

    def test_the_rulesets_are_left_as_given(self):
        rules = {"ruleset_default_value": 3.0}
        schedule = ScheduleSystem(id="test_unchanged", weekday_ruleset=rules)
        start = datetime.datetime(2023, 1, 2, tzinfo=tz.UTC)
        schedule.initialize([start], [start + datetime.timedelta(hours=1)], [600])
        self.assertEqual(rules, {"ruleset_default_value": 3.0})
        self.assertIsNone(schedule.monday_ruleset)
        schedule.weekday_ruleset = {"ruleset_default_value": 5.0}
        schedule.initialize([start], [start + datetime.timedelta(hours=1)], [600])
        self.assertEqual(float(schedule.output["scheduleValue"].history().reshape(-1)[0]), 5.0)

    def test_a_deprecated_keyword_still_works_with_a_warning(self):
        with self.assertWarns(DeprecationWarning):
            schedule = ScheduleSystem(id="test_old_name", weekDayRulesetDict=RULES)
        self.assertEqual(schedule.weekday_ruleset, RULES)

    def test_caching_with_different_datetime_instances(self):
        """Test that caching works reliably with different datetime instances having same values."""
        # Standard library imports

        # Third party imports

        # Local application imports

        # Set test flag
        tb._IS_TESTING = True

        # Create model with schedule system that has noise
        model = Model(id="test_cache_model")
        schedule = ScheduleSystem(
            weekday_ruleset={
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


class TestOldAttributeNames(unittest.TestCase):
    def test_an_old_name_sets_and_reads_the_current_one(self):
        """``schedule.weekDayRulesetDict = {...}`` set an attribute nothing
        read (the example models' setup did it); it sets the ruleset now."""
        schedule = tb.ScheduleSystem(id="old_name")
        ruleset = {"ruleset_default_value": 900.0}
        with self.assertWarns(DeprecationWarning):
            schedule.weekDayRulesetDict = ruleset
        self.assertEqual(schedule.weekday_ruleset, ruleset)
        self.assertNotIn("weekDayRulesetDict", schedule.__dict__)
        with self.assertWarns(DeprecationWarning):
            self.assertEqual(schedule.weekDayRulesetDict, ruleset)

if __name__ == "__main__":
    unittest.main()
