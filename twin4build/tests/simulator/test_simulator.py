# Standard library imports
import datetime
import os
import shutil
import unittest

# Third party imports
import numpy as np
import pytz
import torch

# Local application imports
from twin4build.model.model import Model
from twin4build.simulator.simulator import Simulator
from twin4build.systems.damper.damper_torch_system import DamperTorchSystem
from twin4build.systems.schedule.schedule_system import ScheduleSystem


class TestSimulator(unittest.TestCase):
    def setUp(self):
        self.model = Model(id="test_sim_model")

        schedule = ScheduleSystem(
            weekDayRulesetDict={
                "ruleset_start_minute": [0],
                "ruleset_end_minute": [0],
                "ruleset_start_hour": [0],
                "ruleset_end_hour": [1],
                "ruleset_value": [0.5],
                "ruleset_default_value": 0,
            },
            id="schedule",
        )
        damper = DamperTorchSystem(id="damper")

        self.model.add_component(schedule)
        self.model.add_component(damper)
        self.model.add_connection(schedule, damper, "scheduleValue", "damperPosition")
        self.model.load()

        self.simulator = Simulator(self.model)

    def tearDown(self):
        # Cleanup generated files
        if os.path.exists("generated_files/models/test_sim_model"):
            shutil.rmtree("generated_files/models/test_sim_model")

    def test_simulate(self):
        """Test basic single period simulation."""
        start_time = datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)
        end_time = datetime.datetime(2023, 1, 1, 1, 0, 0, tzinfo=pytz.UTC)
        step_size = 600

        self.simulator.simulate(
            start_time=start_time, end_time=end_time, step_size=step_size
        )

        # Verify that simulation populated the history
        schedule = self.model.components["schedule"]
        damper = self.model.components["damper"]

        # Check that output history was populated
        self.assertTrue(schedule.output["scheduleValue"]._history_is_populated)
        self.assertTrue(damper.output["airFlowRate"]._history_is_populated)

        # Check that history has the correct shape
        # 6 timesteps for 1 hour with 600s step size: 0, 600, 1200, 1800, 2400, 3000
        expected_timesteps = 6
        self.assertEqual(
            schedule.output["scheduleValue"].history.shape[1], expected_timesteps
        )

    def test_simulate_batched(self):
        """Test batched simulation with multiple time periods."""
        # Define multiple simulation periods
        periods = [
            (
                datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC),
                datetime.datetime(2023, 1, 1, 1, 0, 0, tzinfo=pytz.UTC),
            ),
            (
                datetime.datetime(2023, 1, 2, 0, 0, 0, tzinfo=pytz.UTC),
                datetime.datetime(2023, 1, 2, 1, 0, 0, tzinfo=pytz.UTC),
            ),
            (
                datetime.datetime(2023, 1, 3, 0, 0, 0, tzinfo=pytz.UTC),
                datetime.datetime(2023, 1, 3, 1, 0, 0, tzinfo=pytz.UTC),
            ),
        ]
        step_size = 600

        start_times = [p[0] for p in periods]
        end_times = [p[1] for p in periods]

        # Simulate with list of periods
        self.simulator.simulate(
            start_time=start_times, end_time=end_times, step_size=step_size
        )

        # Check that batch dimension is correct
        schedule = self.model.components["schedule"]
        history = schedule.output["scheduleValue"].history

        # Batch size should be 3
        self.assertEqual(history.shape[0], 3)

    def test_simulate_invalid_time_period(self):
        """Test simulation with invalid time period (start >= end)."""
        start_time = datetime.datetime(2023, 1, 1, 1, 0, 0, tzinfo=pytz.UTC)
        end_time = datetime.datetime(
            2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC
        )  # Before start
        step_size = 600

        # Should raise an error
        with self.assertRaises((AssertionError, ValueError, IndexError)):
            self.simulator.simulate(
                start_time=start_time, end_time=end_time, step_size=step_size
            )

    def test_simulate_zero_step_size(self):
        """Test simulation with zero step size."""
        start_time = datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)
        end_time = datetime.datetime(2023, 1, 1, 1, 0, 0, tzinfo=pytz.UTC)
        step_size = 0

        # Should raise an error
        with self.assertRaises((AssertionError, ValueError, ZeroDivisionError)):
            self.simulator.simulate(
                start_time=start_time, end_time=end_time, step_size=step_size
            )

    def test_simulate_negative_step_size(self):
        """Test simulation with negative step size."""
        start_time = datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)
        end_time = datetime.datetime(2023, 1, 1, 1, 0, 0, tzinfo=pytz.UTC)
        step_size = -600

        # Should raise an error
        with self.assertRaises((AssertionError, ValueError, IndexError)):
            self.simulator.simulate(
                start_time=start_time, end_time=end_time, step_size=step_size
            )

    def test_simulate_very_short_period(self):
        """Test simulation with very short time period."""
        start_time = datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)
        end_time = datetime.datetime(
            2023, 1, 1, 0, 0, 10, tzinfo=pytz.UTC
        )  # 10 seconds
        step_size = 5  # 5 second steps

        self.simulator.simulate(
            start_time=start_time, end_time=end_time, step_size=step_size
        )

        # Should have 2 timesteps
        schedule = self.model.components["schedule"]
        history = schedule.output["scheduleValue"].history
        self.assertEqual(history.shape[1], 2)

    def test_simulate_multiple_runs(self):
        """Test running multiple simulations sequentially."""
        start_time1 = datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)
        end_time1 = datetime.datetime(2023, 1, 1, 1, 0, 0, tzinfo=pytz.UTC)

        start_time2 = datetime.datetime(2023, 1, 2, 0, 0, 0, tzinfo=pytz.UTC)
        end_time2 = datetime.datetime(2023, 1, 2, 1, 0, 0, tzinfo=pytz.UTC)

        step_size = 600

        # First simulation
        self.simulator.simulate(
            start_time=start_time1, end_time=end_time1, step_size=step_size
        )

        # Second simulation should work without issues
        self.simulator.simulate(
            start_time=start_time2, end_time=end_time2, step_size=step_size
        )

        # Verify second simulation results
        schedule = self.model.components["schedule"]
        self.assertTrue(schedule.output["scheduleValue"]._history_is_populated)

    def test_simulate_with_different_step_sizes(self):
        """Test simulations with different step sizes."""
        start_time = datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)
        end_time = datetime.datetime(2023, 1, 1, 1, 0, 0, tzinfo=pytz.UTC)

        # Test with 300 second steps
        self.simulator.simulate(start_time=start_time, end_time=end_time, step_size=300)

        schedule = self.model.components["schedule"]
        history1 = schedule.output["scheduleValue"].history

        # Should have 12 timesteps (3600 / 300)
        self.assertEqual(history1.shape[1], 12)

    def test_simulate_result_caching(self):
        """Test that simulation results are properly cached."""
        start_time = datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)
        end_time = datetime.datetime(2023, 1, 1, 1, 0, 0, tzinfo=pytz.UTC)
        step_size = 600

        # Run simulation
        self.simulator.simulate(
            start_time=start_time, end_time=end_time, step_size=step_size
        )

        # Get results
        schedule = self.model.components["schedule"]
        history1 = schedule.output["scheduleValue"].history.clone()

        # Run same simulation again
        self.simulator.simulate(
            start_time=start_time, end_time=end_time, step_size=step_size
        )

        history2 = schedule.output["scheduleValue"].history

        # Results should be identical
        torch.testing.assert_close(history1, history2)

    def test_simulate_without_progress_bar(self):
        """Test simulation without progress bar."""
        start_time = datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)
        end_time = datetime.datetime(2023, 1, 1, 1, 0, 0, tzinfo=pytz.UTC)
        step_size = 600

        # Run simulation without progress bar
        self.simulator.simulate(
            start_time=start_time,
            end_time=end_time,
            step_size=step_size,
            show_progress_bar=False,
        )

        # Verify simulation completed
        schedule = self.model.components["schedule"]
        self.assertTrue(schedule.output["scheduleValue"]._history_is_populated)

    def test_set_simulation_timesteps(self):
        """Test set_simulation_timesteps method."""
        start_time = datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)
        end_time = datetime.datetime(2023, 1, 1, 1, 0, 0, tzinfo=pytz.UTC)
        step_size = 600

        self.simulator.set_simulation_timesteps(start_time, end_time, step_size)

        # Check that timesteps were set
        self.assertIsNotNone(self.simulator.second_time_steps)
        self.assertIsNotNone(self.simulator.date_time_steps)

        # Verify shape
        self.assertEqual(self.simulator.second_time_steps.shape[1], 6)  # 6 timesteps

    def test_get_simulation_timesteps_static(self):
        """Test the static get_simulation_timesteps method."""
        start_time = datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)
        end_time = datetime.datetime(2023, 1, 1, 1, 0, 0, tzinfo=pytz.UTC)
        step_size = 600

        second_steps, date_steps, max_timesteps, n_timesteps = (
            Simulator.get_simulation_timesteps(start_time, end_time, step_size)
        )

        self.assertEqual(max_timesteps, 6)
        self.assertEqual(n_timesteps, [6])
        self.assertEqual(len(second_steps[0]), 6)

    def test_get_simulation_timesteps_batched(self):
        """Test get_simulation_timesteps with multiple periods of different lengths."""
        start_times = [
            datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC),
            datetime.datetime(2023, 1, 2, 0, 0, 0, tzinfo=pytz.UTC),
        ]
        end_times = [
            datetime.datetime(2023, 1, 1, 1, 0, 0, tzinfo=pytz.UTC),  # 6 steps
            datetime.datetime(2023, 1, 2, 0, 30, 0, tzinfo=pytz.UTC),  # 3 steps
        ]
        step_size = [600, 600]

        second_steps, date_steps, max_timesteps, n_timesteps = (
            Simulator.get_simulation_timesteps(start_times, end_times, step_size)
        )

        self.assertEqual(max_timesteps, 6)  # Max of 6 and 3
        self.assertEqual(n_timesteps, [6, 3])
        # Second period should be padded with NaN
        self.assertTrue(np.isnan(second_steps[1, 3]))


if __name__ == "__main__":
    unittest.main()
