# Standard library imports
import datetime
import os
import shutil
import tempfile
import unittest

# Third party imports
import numpy as np
import pandas as pd
from dateutil import tz

# Local application imports
import twin4build as tb
import twin4build.examples.utils as utils
from twin4build.estimator.estimator import Estimator
from twin4build.model.model import Model
from twin4build.simulator.simulator import Simulator
from twin4build.systems.damper.damper_torch_system import DamperTorchSystem

# Set test flag
tb._IS_TESTING = True
# Local application imports
from twin4build.systems.schedule.schedule_system import ScheduleSystem
from twin4build.systems.sensor.sensor_system import SensorSystem


class TestEstimator(unittest.TestCase):
    def setUp(self):
        self.model = Model(id="test_est_model")

        self.schedule = ScheduleSystem(
            weekDayRulesetDict={
                "ruleset_start_minute": [0],
                "ruleset_end_minute": [0],
                "ruleset_start_hour": [0],
                "ruleset_end_hour": [24],
                "ruleset_value": [0.5],
                "ruleset_default_value": 0,
            },
            id="schedule",
        )
        self.damper = DamperTorchSystem(id="damper", a=1.0, nominalAirFlowRate=0.5)

        self.model.add_component(self.schedule)
        self.model.add_component(self.damper)
        self.model.add_connection(
            self.schedule, self.damper, "scheduleValue", "damperPosition"
        )

    def tearDown(self):
        if os.path.exists("generated_files/models/test_est_model"):
            shutil.rmtree("generated_files/models/test_est_model")

    def test_estimator_initialization(self):
        """Test that estimator can be initialized with a simulator."""
        self.model.load()
        simulator = Simulator(self.model)
        estimator = Estimator(simulator)

        self.assertIsNotNone(estimator)
        self.assertEqual(estimator.simulator, simulator)

    def test_estimate_with_sensor_data(self):
        """Test estimation with sensor measurement data."""
        start_time = datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=tz.UTC)
        end_time = datetime.datetime(2023, 1, 1, 1, 0, 0, tzinfo=tz.UTC)
        step_size = 600

        # First, run simulation to generate "true" data with known parameters
        true_a = 1.0
        self.damper.a.set(true_a, normalized=False)

        self.model.load()
        simulator = Simulator(self.model)
        simulator.simulate(
            start_time=start_time, end_time=end_time, step_size=step_size
        )

        # Extract the simulated air flow rate as "measurement"
        # History uses time-first layout (n_t, n_s, n_c), use history(i_s=0, i_c=0) to get (n_t,)
        true_airflow = (
            self.damper.output["airFlowRate"].history(i_s=0, i_c=0).detach().numpy()
        )

        # Create a sensor with this "measured" data
        # Note: simulation produces n_timesteps which may not include the end time
        dates = pd.date_range(
            start=start_time, periods=len(true_airflow), freq=f"{step_size}s"
        )
        df = pd.DataFrame({"value": true_airflow}, index=dates)

        sensor = SensorSystem(
            id="airflow_sensor",
            df=df,
        )

        # Create new model for estimation with different initial parameter
        model_est = Model(id="test_est_model_2")
        schedule_est = ScheduleSystem(
            weekDayRulesetDict={
                "ruleset_start_minute": [0],
                "ruleset_end_minute": [0],
                "ruleset_start_hour": [0],
                "ruleset_end_hour": [24],
                "ruleset_value": [0.5],
                "ruleset_default_value": 0,
            },
            id="schedule",
        )
        damper_est = DamperTorchSystem(
            id="damper", a=0.5, nominalAirFlowRate=0.5
        )  # Wrong initial value
        model_est.add_connection(
            schedule_est, damper_est, "scheduleValue", "damperPosition"
        )
        model_est.add_connection(damper_est, sensor, "airFlowRate", "measuredValue")
        model_est.load()

        # Create estimator
        sim_est = Simulator(model_est)
        estimator = Estimator(sim_est)

        # Define parameters to estimate - new list format
        parameters = [
            (
                damper_est,
                "a",
                0.5,
                0.1,
                2.0,
                "private",
            ),  # (component, attr, x0, lb, ub, type)
        ]

        # Run estimation
        # measurements is a list of tuples: [(sensor, standard_deviation), ...]
        result = estimator.estimate(
            parameters=parameters,
            measurements=[(sensor, 0.01)],  # sensor with 0.01 standard deviation
            start_time=start_time,
            end_time=end_time,
            step_size=step_size,
            method=("scipy", "SLSQP", "ad"),
            n_warmup=0,
        )

        # Check that estimation converged close to true value
        estimated_a = damper_est.a.get().item()
        self.assertAlmostEqual(estimated_a, true_a, places=2)

        # Cleanup
        if os.path.exists("generated_files/models/test_est_model_2"):
            shutil.rmtree("generated_files/models/test_est_model_2")

    def test_estimate_with_invalid_bounds(self):
        """Test that estimation raises error when lower bound > upper bound."""
        start_time = datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=tz.UTC)
        end_time = datetime.datetime(2023, 1, 1, 0, 30, 0, tzinfo=tz.UTC)
        step_size = 600

        # Create simple model with sensor
        dates = pd.date_range(start=start_time, end=end_time, freq=f"{step_size}s")
        df = pd.DataFrame({"value": np.ones(len(dates)) * 0.5}, index=dates)
        sensor = SensorSystem(id="sensor", df=df)

        self.model.add_component(sensor)
        self.model.add_connection(self.damper, sensor, "airFlowRate", "measuredValue")
        self.model.load()

        simulator = Simulator(self.model)
        estimator = Estimator(simulator)

        # Define parameters with invalid bounds (lb > ub)
        parameters = [
            (self.damper, "a", 1.0, 2.0, 0.5, "private"),  # lb=2.0 > ub=0.5
        ]

        # Should raise an assertion or value error
        with self.assertRaises((AssertionError, ValueError)):
            estimator.estimate(
                parameters=parameters,
                measurements=[(sensor, 0.01)],
                start_time=start_time,
                end_time=end_time,
                step_size=step_size,
                method=("scipy", "SLSQP", "ad"),
            )

    def test_estimate_without_measurements(self):
        """Test that estimation raises error when no measurements provided."""
        start_time = datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=tz.UTC)
        end_time = datetime.datetime(2023, 1, 1, 0, 30, 0, tzinfo=tz.UTC)
        step_size = 600

        self.model.load()
        simulator = Simulator(self.model)
        estimator = Estimator(simulator)

        parameters = [
            (self.damper, "a", 1.0, 0.5, 2.0, "private"),
        ]

        # Should raise error when measurements is empty or None
        with self.assertRaises(
            (AssertionError, ValueError, TypeError, UnboundLocalError)
        ):
            estimator.estimate(
                parameters=parameters,
                measurements=[],  # Empty measurements
                start_time=start_time,
                end_time=end_time,
                step_size=step_size,
                method=("scipy", "SLSQP", "ad"),
            )

    def test_estimate_with_empty_parameters(self):
        """Test that estimation raises error when no parameters to estimate."""
        start_time = datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=tz.UTC)
        end_time = datetime.datetime(2023, 1, 1, 0, 30, 0, tzinfo=tz.UTC)
        step_size = 600

        dates = pd.date_range(start=start_time, end=end_time, freq=f"{step_size}s")
        df = pd.DataFrame({"value": np.ones(len(dates)) * 0.5}, index=dates)
        sensor = SensorSystem(id="sensor", df=df)

        self.model.add_component(sensor)
        self.model.add_connection(self.damper, sensor, "airFlowRate", "measuredValue")
        self.model.load()

        simulator = Simulator(self.model)
        estimator = Estimator(simulator)

        # Should raise error when parameters is empty
        with self.assertRaises((AssertionError, ValueError, IndexError)):
            estimator.estimate(
                parameters=[],  # Empty parameters
                measurements=[(sensor, 0.01)],
                start_time=start_time,
                end_time=end_time,
                step_size=step_size,
                method=("scipy", "SLSQP", "ad"),
            )

    def test_estimate_with_nonexistent_attribute(self):
        """Test that estimation raises error when parameter attribute doesn't exist."""
        start_time = datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=tz.UTC)
        end_time = datetime.datetime(2023, 1, 1, 0, 30, 0, tzinfo=tz.UTC)
        step_size = 600

        dates = pd.date_range(start=start_time, end=end_time, freq=f"{step_size}s")
        df = pd.DataFrame({"value": np.ones(len(dates)) * 0.5}, index=dates)
        sensor = SensorSystem(id="sensor", df=df)

        self.model.add_component(sensor)
        self.model.add_connection(self.damper, sensor, "airFlowRate", "measuredValue")
        self.model.load()

        simulator = Simulator(self.model)
        estimator = Estimator(simulator)

        # Define parameters with non-existent attribute
        parameters = [
            (self.damper, "nonexistent_param", 1.0, 0.5, 2.0, "private"),
        ]

        # Should raise AttributeError
        with self.assertRaises((AttributeError, KeyError)):
            estimator.estimate(
                parameters=parameters,
                measurements=[(sensor, 0.01)],
                start_time=start_time,
                end_time=end_time,
                step_size=step_size,
                method=("scipy", "SLSQP", "ad"),
            )

    def test_estimate_with_invalid_time_range(self):
        """Test that estimation raises error when start_time >= end_time."""
        start_time = datetime.datetime(2023, 1, 1, 1, 0, 0, tzinfo=tz.UTC)
        end_time = datetime.datetime(
            2023, 1, 1, 0, 0, 0, tzinfo=tz.UTC
        )  # Before start_time
        step_size = 600

        dates = pd.date_range(
            start=datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=tz.UTC),
            end=datetime.datetime(2023, 1, 1, 1, 0, 0, tzinfo=tz.UTC),
            freq=f"{step_size}s",
        )
        df = pd.DataFrame({"value": np.ones(len(dates)) * 0.5}, index=dates)
        sensor = SensorSystem(id="sensor", df=df)

        self.model.add_component(sensor)
        self.model.add_connection(self.damper, sensor, "airFlowRate", "measuredValue")
        self.model.load()

        simulator = Simulator(self.model)
        estimator = Estimator(simulator)

        parameters = [
            (self.damper, "a", 1.0, 0.5, 2.0, "private"),
        ]

        # Should raise error when start_time >= end_time
        with self.assertRaises((AssertionError, ValueError)):
            estimator.estimate(
                parameters=parameters,
                measurements=[(sensor, 0.01)],
                start_time=start_time,
                end_time=end_time,
                step_size=step_size,
                method=("scipy", "SLSQP", "ad"),
            )

    def test_estimator_save_and_load(self):
        """
        Test that estimation results can be saved and loaded, and that
        the loaded results can be used to run a simulation.
        """
        # Create a temporary directory for test outputs
        test_dir = tempfile.mkdtemp(prefix="twin4build_test_")

        try:
            # Step 1: Load the model (same as in estimator_example.ipynb)
            model = tb.Model(id="estimator_example")

            # Load the model from semantic file
            filename_simulation = utils.get_path(
                ["estimator_example", "instance_graph.ttl"]
            )

            model.load(simulation_model_filename=filename_simulation, verbose=0)

            # Configure file paths for sensors
            for sensor_id, csv_name in [
                ("office_temperature_sensor", "temperature_sensor.csv"),
                ("office_co2_sensor", "co2_sensor.csv"),
                ("office_valve_position_sensor", "valve_position_sensor.csv"),
                ("office_damper_position_sensor", "damper_position_sensor.csv"),
                ("supply_air_temperature_sensor", "supply_air_temperature.csv"),
            ]:
                model.components[sensor_id].filename = utils.get_path(
                    ["estimator_example", csv_name]
                )
                model.components[sensor_id].datecolumn = 0
                model.components[sensor_id].valuecolumn = 1
            model.components["office_temperature_heating_setpoint"].filename = (
                utils.get_path(
                    ["estimator_example", "temperature_heating_setpoint.csv"]
                )
            )
            model.components["office_temperature_heating_setpoint"].datecolumn = 0
            model.components["office_temperature_heating_setpoint"].valuecolumn = 1
            model.components["outdoor_environment"].filename_outdoorTemperature = (
                utils.get_path(["estimator_example", "outdoor_environment.csv"])
            )
            model.components["outdoor_environment"].filename_globalIrradiation = (
                utils.get_path(["estimator_example", "outdoor_environment.csv"])
            )
            model.components[
                "outdoor_environment"
            ].valuecolumn_outdoorCo2Concentration = 2
            model.components["outdoor_environment"].filename_outdoorCo2Concentration = (
                utils.get_path(["estimator_example", "outdoor_environment.csv"])
            )

            # Set up a temporary directory for saving estimation results
            # This ensures files are cleaned up and works in GitHub Actions
            model.dir_conf = [test_dir, "model_parameters"]

            # Step 2: Set up simulation parameters (using shorter time periods for faster testing)
            simulator = tb.Simulator(model)
            step_size = 1200  # 20 minutes in seconds
            start_time = [
                datetime.datetime(
                    year=2023,
                    month=11,
                    day=27,
                    hour=0,
                    minute=0,
                    second=0,
                    tzinfo=tz.gettz("Europe/Copenhagen"),
                ),
                datetime.datetime(
                    year=2023,
                    month=12,
                    day=2,
                    hour=0,
                    minute=0,
                    second=0,
                    tzinfo=tz.gettz("Europe/Copenhagen"),
                ),
            ]
            end_time = [
                datetime.datetime(
                    year=2023,
                    month=12,
                    day=1,
                    hour=0,
                    minute=0,
                    second=0,
                    tzinfo=tz.gettz("Europe/Copenhagen"),
                ),
                datetime.datetime(
                    year=2023,
                    month=12,
                    day=5,
                    hour=0,
                    minute=0,
                    second=0,
                    tzinfo=tz.gettz("Europe/Copenhagen"),
                ),
            ]

            # Step 3: Identify key components
            space = model.components["office"]
            space_heater = model.components["office_space_heater"]
            heating_controller = model.components[
                "office_temperature_heating_controller"
            ]

            # Step 4: Define parameters for estimation (simplified set for faster testing)
            parameters = [
                # Thermal parameters
                (space, "thermal.C_air", 2e6, 1e6, 1e8),
                (space, "thermal.C_wall", 2e6, 1e6, 1e8),
                # Space heater parameters
                (space_heater, "thermalMassHeatCapacity", 10000, 1000, 50000),
                # Controller parameters
                (heating_controller, "kp", 0.001, 1e-5, 1, "private"),
            ]

            # Step 5: Configure measuring devices
            percentile = 2
            measurements = [
                (model.components["office_temperature_sensor"], 0.1 / percentile),
                (model.components["office_co2_sensor"], 30 / percentile),
            ]

            # Step 6: Create estimator
            estimator = tb.Estimator(simulator)

            # Step 7: Run parameter estimation
            # Use a very small number of iterations for faster testing
            # In a real scenario, you'd use more iterations
            result = estimator.estimate(
                start_time=start_time,
                end_time=end_time,
                step_size=step_size,
                parameters=parameters,
                measurements=measurements,
                n_warmup=5,  # Reduced for faster testing
                method=("scipy", "SLSQP", "ad"),
                n_cores=1,  # Use 1 core to avoid issues in CI
                options={"maxiter": 2},  # Limit iterations for testing
            )

            # Step 8: Verify that result was returned
            self.assertIsNotNone(result, "Estimation result should not be None")
            self.assertIn("result_x", result, "Result should contain 'result_x'")
            self.assertIn(
                "component_id", result, "Result should contain 'component_id'"
            )
            self.assertIn(
                "component_attr", result, "Result should contain 'component_attr'"
            )

            # Step 9: Verify that the result file was saved
            saved_file_path = estimator.result_savedir_pickle
            self.assertIsNotNone(saved_file_path, "Result file path should be set")
            self.assertTrue(
                os.path.exists(saved_file_path),
                f"Result file should exist at {saved_file_path}",
            )
            self.assertTrue(
                saved_file_path.endswith(".pickle"),
                "Result file should have .pickle extension",
            )

            # Step 10: Create a new model for loading the results
            model2 = tb.Model(id="estimator_example_loaded")
            model2.load(simulation_model_filename=filename_simulation, verbose=0)

            # Configure file paths for the new model
            for sensor_id, csv_name in [
                ("office_temperature_sensor", "temperature_sensor.csv"),
                ("office_co2_sensor", "co2_sensor.csv"),
                ("office_valve_position_sensor", "valve_position_sensor.csv"),
                ("office_damper_position_sensor", "damper_position_sensor.csv"),
                ("supply_air_temperature_sensor", "supply_air_temperature.csv"),
            ]:
                model2.components[sensor_id].filename = utils.get_path(
                    ["estimator_example", csv_name]
                )
                model2.components[sensor_id].datecolumn = 0
                model2.components[sensor_id].valuecolumn = 1
            model2.components["office_temperature_heating_setpoint"].filename = (
                utils.get_path(
                    ["estimator_example", "temperature_heating_setpoint.csv"]
                )
            )
            model2.components["office_temperature_heating_setpoint"].datecolumn = 0
            model2.components["office_temperature_heating_setpoint"].valuecolumn = 1
            model2.components["outdoor_environment"].filename_outdoorTemperature = (
                utils.get_path(["estimator_example", "outdoor_environment.csv"])
            )
            model2.components["outdoor_environment"].filename_globalIrradiation = (
                utils.get_path(["estimator_example", "outdoor_environment.csv"])
            )
            model2.components[
                "outdoor_environment"
            ].valuecolumn_outdoorCo2Concentration = 2
            model2.components[
                "outdoor_environment"
            ].filename_outdoorCo2Concentration = utils.get_path(
                ["estimator_example", "outdoor_environment.csv"]
            )

            # Step 11: Load the estimation result from file
            model2.load_estimation_result(filename=saved_file_path)

            # Step 12: Verify that parameters were loaded
            # Check that the loaded result is accessible
            self.assertIsNotNone(
                model2.simulation_model._result,
                "Model should have a result after loading",
            )
            self.assertIn(
                "result_x",
                model2.simulation_model._result,
                "Loaded result should contain 'result_x'",
            )

            # Step 13: Run a simulation with the loaded parameters to verify they work
            simulator2 = tb.Simulator(model2)
            # Use a very short time period for faster testing
            test_start = datetime.datetime(
                year=2023,
                month=11,
                day=27,
                hour=0,
                minute=0,
                second=0,
                tzinfo=tz.gettz("Europe/Copenhagen"),
            )
            test_end = datetime.datetime(
                year=2023,
                month=11,
                day=27,
                hour=1,
                minute=0,
                second=0,
                tzinfo=tz.gettz("Europe/Copenhagen"),
            )

            # This should complete without errors if the parameters were loaded correctly
            simulator2.simulate(
                step_size=step_size,
                start_time=test_start,
                end_time=test_end,
                show_progress_bar=False,
            )

            # Step 14: Verify that the simulation produced results
            # Check that the space component has output values
            space_loaded = model2.components["office"]
            self.assertIsNotNone(
                space_loaded.output,
                "Space component should have output after simulation",
            )
            self.assertIn(
                "indoorTemperature",
                space_loaded.output,
                "Space should have indoorTemperature output",
            )

            # Step 15: Verify that the parameters are the same for the loaded model
            self.assertAlmostEqual(
                space.thermal.C_air,
                space_loaded.thermal.C_air,
                delta=0.1 * space.thermal.C_air,
            )
            self.assertAlmostEqual(
                space.thermal.C_wall,
                space_loaded.thermal.C_wall,
                delta=0.1 * space.thermal.C_wall,
            )

            heater_loaded = model2.components["office_space_heater"]
            self.assertAlmostEqual(
                heater_loaded.thermalMassHeatCapacity,
                space_heater.thermalMassHeatCapacity,
                delta=0.1 * heater_loaded.thermalMassHeatCapacity,
            )

            # Test passed if we got here without exceptions
            self.assertTrue(
                True, "Successfully saved, loaded, and used estimation results"
            )

        finally:
            # Clean up the temporary directory
            if os.path.exists(test_dir):
                shutil.rmtree(test_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
