# Standard library imports
import datetime
import os
import shutil
import unittest

# Third party imports
import numpy as np
import torch
from dateutil import tz

# Local application imports
import twin4build as tb
import twin4build.examples.utils as utils
from twin4build.model.model import Model
from twin4build.simulator.simulator import Simulator
from twin4build.systems.damper.damper_torch_system import DamperTorchSystem
from twin4build.systems.schedule.schedule_system import ScheduleSystem

# Set test flag
tb._IS_TESTING = True


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
        start_time = datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=tz.UTC)
        end_time = datetime.datetime(2023, 1, 1, 1, 0, 0, tzinfo=tz.UTC)
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
        # History uses time-first layout (n_t, n_s, n_c), so shape[0] is n_t
        expected_timesteps = 6
        self.assertEqual(
            schedule.output["scheduleValue"].history().shape[0], expected_timesteps
        )

    def test_simulate_batched(self):
        """Test batched simulation with multiple time periods."""
        # Define multiple simulation periods
        periods = [
            (
                datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=tz.UTC),
                datetime.datetime(2023, 1, 1, 1, 0, 0, tzinfo=tz.UTC),
            ),
            (
                datetime.datetime(2023, 1, 2, 0, 0, 0, tzinfo=tz.UTC),
                datetime.datetime(2023, 1, 2, 1, 0, 0, tzinfo=tz.UTC),
            ),
            (
                datetime.datetime(2023, 1, 3, 0, 0, 0, tzinfo=tz.UTC),
                datetime.datetime(2023, 1, 3, 1, 0, 0, tzinfo=tz.UTC),
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
        history = schedule.output["scheduleValue"].history()  # Now a method

        # History shape is (n_t, n_s, n_c) - time-first layout
        # Batch size (n_s) should be 3, which is shape[1]
        self.assertEqual(history.shape[1], 3)

    def test_simulate_invalid_time_period(self):
        """Test simulation with invalid time period (start >= end)."""
        start_time = datetime.datetime(2023, 1, 1, 1, 0, 0, tzinfo=tz.UTC)
        end_time = datetime.datetime(
            2023, 1, 1, 0, 0, 0, tzinfo=tz.UTC
        )  # Before start
        step_size = 600

        # Should raise an error
        with self.assertRaises((AssertionError, ValueError, IndexError)):
            self.simulator.simulate(
                start_time=start_time, end_time=end_time, step_size=step_size
            )

    def test_simulate_zero_step_size(self):
        """Test simulation with zero step size."""
        start_time = datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=tz.UTC)
        end_time = datetime.datetime(2023, 1, 1, 1, 0, 0, tzinfo=tz.UTC)
        step_size = 0

        # Should raise an error
        with self.assertRaises((AssertionError, ValueError, ZeroDivisionError)):
            self.simulator.simulate(
                start_time=start_time, end_time=end_time, step_size=step_size
            )

    def test_simulate_negative_step_size(self):
        """Test simulation with negative step size."""
        start_time = datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=tz.UTC)
        end_time = datetime.datetime(2023, 1, 1, 1, 0, 0, tzinfo=tz.UTC)
        step_size = -600

        # Should raise an error
        with self.assertRaises((AssertionError, ValueError, IndexError)):
            self.simulator.simulate(
                start_time=start_time, end_time=end_time, step_size=step_size
            )

    def test_simulate_very_short_period(self):
        """Test simulation with very short time period."""
        start_time = datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=tz.UTC)
        end_time = datetime.datetime(
            2023, 1, 1, 0, 0, 10, tzinfo=tz.UTC
        )  # 10 seconds
        step_size = 5  # 5 second steps

        self.simulator.simulate(
            start_time=start_time, end_time=end_time, step_size=step_size
        )

        # Should have 2 timesteps
        # History shape is (n_t, n_s, n_c) - time-first layout, so shape[0] is n_t
        schedule = self.model.components["schedule"]
        history = schedule.output["scheduleValue"].history()  # Now a method
        self.assertEqual(history.shape[0], 2)

    def test_simulate_multiple_runs(self):
        """Test running multiple simulations sequentially."""
        start_time1 = datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=tz.UTC)
        end_time1 = datetime.datetime(2023, 1, 1, 1, 0, 0, tzinfo=tz.UTC)

        start_time2 = datetime.datetime(2023, 1, 2, 0, 0, 0, tzinfo=tz.UTC)
        end_time2 = datetime.datetime(2023, 1, 2, 1, 0, 0, tzinfo=tz.UTC)

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
        start_time = datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=tz.UTC)
        end_time = datetime.datetime(2023, 1, 1, 1, 0, 0, tzinfo=tz.UTC)

        # Test with 300 second steps
        self.simulator.simulate(start_time=start_time, end_time=end_time, step_size=300)

        schedule = self.model.components["schedule"]
        history1 = schedule.output["scheduleValue"].history()

        # Should have 12 timesteps (3600 / 300)
        # History uses time-first layout (n_t, n_s, n_c), so shape[0] is n_t
        self.assertEqual(history1.shape[0], 12)

    def test_simulate_result_caching(self):
        """Test that simulation results are properly cached."""
        start_time = datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=tz.UTC)
        end_time = datetime.datetime(2023, 1, 1, 1, 0, 0, tzinfo=tz.UTC)
        step_size = 600

        # Run simulation
        self.simulator.simulate(
            start_time=start_time, end_time=end_time, step_size=step_size
        )

        # Get results
        schedule = self.model.components["schedule"]
        history1 = schedule.output["scheduleValue"].history().clone()

        # Run same simulation again
        self.simulator.simulate(
            start_time=start_time, end_time=end_time, step_size=step_size
        )

        history2 = schedule.output["scheduleValue"].history()

        # Results should be identical
        torch.testing.assert_close(history1, history2)

    def test_simulate_without_progress_bar(self):
        """Test simulation without progress bar."""
        start_time = datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=tz.UTC)
        end_time = datetime.datetime(2023, 1, 1, 1, 0, 0, tzinfo=tz.UTC)
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
        start_time = datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=tz.UTC)
        end_time = datetime.datetime(2023, 1, 1, 1, 0, 0, tzinfo=tz.UTC)
        step_size = 600

        self.simulator.set_simulation_timesteps(start_time, end_time, step_size)

        # Check that timesteps were set
        self.assertIsNotNone(self.simulator.second_time_steps)
        self.assertIsNotNone(self.simulator.date_time_steps)

        # Verify shape
        self.assertEqual(self.simulator.second_time_steps.shape[1], 6)  # 6 timesteps

    def test_get_simulation_timesteps_static(self):
        """Test the static get_simulation_timesteps method."""
        start_time = datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=tz.UTC)
        end_time = datetime.datetime(2023, 1, 1, 1, 0, 0, tzinfo=tz.UTC)
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
            datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=tz.UTC),
            datetime.datetime(2023, 1, 2, 0, 0, 0, tzinfo=tz.UTC),
        ]
        end_times = [
            datetime.datetime(2023, 1, 1, 1, 0, 0, tzinfo=tz.UTC),  # 6 steps
            datetime.datetime(2023, 1, 2, 0, 30, 0, tzinfo=tz.UTC),  # 3 steps
        ]
        step_size = [600, 600]

        second_steps, date_steps, max_timesteps, n_timesteps = (
            Simulator.get_simulation_timesteps(start_times, end_times, step_size)
        )

        self.assertEqual(max_timesteps, 6)  # Max of 6 and 3
        self.assertEqual(n_timesteps, [6, 3])
        # Second period should be padded with NaN
        self.assertTrue(np.isnan(second_steps[1, 3]))

    def _setup_estimator_example_model(self, model_id: str) -> tb.Model:
        """
        Set up a model with the estimator_example configuration.

        This uses the same complex model structure as the estimator example,
        which includes thermal zones, sensors, controllers, and HVAC components.

        Args:
            model_id: Unique identifier for the model.

        Returns:
            Configured model ready for simulation.
        """
        model = tb.Model(id=model_id)

        # Load the model from semantic file
        filename_simulation = utils.get_path(
            ["estimator_example", "instance_graph.ttl"]
        )
        model.load(simulation_model_filename=filename_simulation, verbose=0)

        # Configure file paths and column indices for sensors
        # (instance_graph uses datecolumn=2/valuecolumn=4 for original large CSVs;
        #  the test CSVs have only 2 columns: datecolumn=0, valuecolumn=1)
        for sensor_id, csv_name in [
            ("office_temperature_sensor", "temperature_sensor.csv"),
            ("office_co2_sensor", "co2_sensor.csv"),
            ("office_valve_position_sensor", "valve_position_sensor.csv"),
            ("office_damper_position_sensor", "damper_position_sensor.csv"),
            ("supply_air_temperature_sensor", "supply_air_temperature.csv"),
        ]:
            model.components[sensor_id].filename = utils.get_path(["estimator_example", csv_name])
            model.components[sensor_id].datecolumn = 0
            model.components[sensor_id].valuecolumn = 1

        model.components["office_temperature_heating_setpoint"].filename = utils.get_path(
            ["estimator_example", "temperature_heating_setpoint.csv"]
        )
        model.components["office_temperature_heating_setpoint"].datecolumn = 0
        model.components["office_temperature_heating_setpoint"].valuecolumn = 1
        model.components["outdoor_environment"].filename_outdoorTemperature = (
            utils.get_path(["estimator_example", "outdoor_environment.csv"])
        )
        model.components["outdoor_environment"].filename_globalIrradiation = (
            utils.get_path(["estimator_example", "outdoor_environment.csv"])
        )
        model.components["outdoor_environment"].valuecolumn_outdoorCo2Concentration = 2
        model.components["outdoor_environment"].filename_outdoorCo2Concentration = (
            utils.get_path(["estimator_example", "outdoor_environment.csv"])
        )

        return model

    def test_jacobi_vs_gauss_seidel_small_step(self):
        """
        Test that jacobi and gauss-seidel produce nearly identical results
        with a small step size.

        With a sufficiently small step size, the difference between the two
        iteration methods should be negligible, as both methods converge to
        the same solution.
        """
        # Use a small step size (5 minutes = 300 seconds)
        step_size = 1

        # Define simulation period (2 hours for faster testing)
        start_time = datetime.datetime(
            year=2023,
            month=11,
            day=27,
            hour=8,
            minute=0,
            second=0,
            tzinfo=tz.gettz("Europe/Copenhagen"),
        )
        end_time = datetime.datetime(
            year=2023,
            month=11,
            day=28,
            hour=10,
            minute=0,
            second=0,
            tzinfo=tz.gettz("Europe/Copenhagen"),
        )

        # Set up and run gauss-seidel simulation
        model_gs = self._setup_estimator_example_model("test_gauss_seidel")
        simulator_gs = tb.Simulator(model_gs)
        simulator_gs.simulate(
            step_size=step_size,
            start_time=start_time,
            end_time=end_time,
            show_progress_bar=False,
            iteration_method="gauss-seidel",
        )

        # Set up and run jacobi simulation
        model_jacobi = self._setup_estimator_example_model("test_jacobi")
        simulator_jacobi = tb.Simulator(model_jacobi)
        simulator_jacobi.simulate(
            step_size=step_size,
            start_time=start_time,
            end_time=end_time,
            show_progress_bar=False,
            iteration_method="jacobi",
        )

        # Compare results for key outputs
        space_gs = model_gs.components["office"]
        space_jacobi = model_jacobi.components["office"]

        # Get indoor temperature outputs
        temp_gs = space_gs.output["indoorTemperature"].history().detach().numpy()
        temp_jacobi = space_jacobi.output["indoorTemperature"].history().detach().numpy()

        # Get CO2 concentration outputs
        co2_gs = space_gs.output["indoorCO2"].history().detach().numpy()
        co2_jacobi = space_jacobi.output["indoorCO2"].history().detach().numpy()

        # Plotting for debugging
        # tb.plot.plot(simulator_gs.date_time_steps,
        # [tb.plot.Entry(temp_gs, label="Gauss-Seidel",),
        # tb.plot.Entry(temp_jacobi, label="Jacobi")], ylabel_1axis="Temperature [°C]")
        # tb.plot.plot(simulator_gs.date_time_steps,
        # [tb.plot.Entry(co2_gs, label="Gauss-Seidel"),
        # tb.plot.Entry(co2_jacobi, label="Jacobi")], ylabel_1axis="CO2 Concentration [ppmv]", show=True)

        # With a small step size, results should be very close
        # Allow for small numerical differences (relative tolerance of 1%)
        np.testing.assert_allclose(
            temp_gs,
            temp_jacobi,
            rtol=0.01,
            atol=0.1,  # 0.1 degree absolute tolerance
            err_msg="Indoor temperature results differ too much between gauss-seidel and jacobi",
        )

        np.testing.assert_allclose(
            co2_gs,
            co2_jacobi,
            rtol=0.01,
            atol=5,  # 5 ppm absolute tolerance
            err_msg="CO2 concentration results differ too much between gauss-seidel and jacobi",
        )

        # Also verify that both simulations actually produced results
        self.assertTrue(
            len(temp_gs) > 0, "Gauss-Seidel simulation should produce results"
        )
        self.assertTrue(
            len(temp_jacobi) > 0, "Jacobi simulation should produce results"
        )

    def test_jacobi_runs_successfully(self):
        """
        Test that the jacobi iteration method runs without errors.

        This is a basic smoke test to ensure the jacobi option works.
        """
        step_size = 600  # 10 minutes

        start_time = datetime.datetime(
            year=2023,
            month=11,
            day=27,
            hour=8,
            minute=0,
            second=0,
            tzinfo=tz.gettz("Europe/Copenhagen"),
        )
        end_time = datetime.datetime(
            year=2023,
            month=11,
            day=27,
            hour=9,
            minute=0,
            second=0,
            tzinfo=tz.gettz("Europe/Copenhagen"),
        )

        model = self._setup_estimator_example_model("test_jacobi_smoke")
        simulator = tb.Simulator(model)

        # This should complete without raising any exceptions
        simulator.simulate(
            step_size=step_size,
            start_time=start_time,
            end_time=end_time,
            show_progress_bar=False,
            iteration_method="jacobi",
        )

        # Verify the iteration method was set correctly
        self.assertEqual(
            simulator.iteration_method,
            "jacobi",
            "Iteration method should be set to 'jacobi'",
        )

        # Verify simulation produced outputs
        space = model.components["office"]
        self.assertIn(
            "indoorTemperature",
            space.output,
            "Space should have indoorTemperature output",
        )

    def test_gauss_seidel_runs_successfully(self):
        """
        Test that gauss-seidel runs successfully.
        """
        step_size = 600

        start_time = datetime.datetime(
            year=2023,
            month=11,
            day=27,
            hour=8,
            minute=0,
            second=0,
            tzinfo=tz.gettz("Europe/Copenhagen"),
        )
        end_time = datetime.datetime(
            year=2023,
            month=11,
            day=27,
            hour=9,
            minute=0,
            second=0,
            tzinfo=tz.gettz("Europe/Copenhagen"),
        )

        model = self._setup_estimator_example_model("test_default_method")
        simulator = tb.Simulator(model)

        # Run without specifying iteration_method
        simulator.simulate(
            step_size=step_size,
            start_time=start_time,
            end_time=end_time,
            show_progress_bar=False,
        )

        # Verify gauss-seidel is the default
        self.assertEqual(
            simulator.iteration_method,
            "gauss-seidel",
            "Default iteration method should be 'gauss-seidel'",
        )


if __name__ == "__main__":
    unittest.main()
