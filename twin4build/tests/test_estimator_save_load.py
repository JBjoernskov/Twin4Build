# Standard library imports
import datetime
import os
import shutil
import tempfile
import unittest

# Third party imports
from dateutil import tz

# Local application imports
import twin4build as tb
import twin4build.examples.utils as utils


class TestEstimatorSaveLoad(unittest.TestCase):
    """
    Test that parameter estimation results can be saved and loaded correctly.

    This test extends the estimator_example.ipynb to verify that:
    1. Estimation results are saved to disk
    2. Results can be loaded from file
    3. Loaded results can be used to configure a model for simulation
    4. Simulation runs successfully with loaded parameters
    """

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a temporary directory for test outputs
        self.test_dir = tempfile.mkdtemp(prefix="twin4build_test_")

    def tearDown(self):
        """Clean up after each test method."""
        # Remove the temporary directory and its contents
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_estimator_save_and_load(self):
        """
        Test that estimation results can be saved and loaded, and that
        the loaded results can be used to run a simulation.
        """
        # Step 1: Load the model (same as in estimator_example.ipynb)
        model = tb.Model(id="estimator_example")

        # Load the model from semantic file
        filename_simulation = utils.get_path(
            ["estimator_example", "instance_graph.ttl"]
        )
        model.load(simulation_model_filename=filename_simulation, verbose=0)

        # Configure file paths for sensors
        model.components["020B_temperature_sensor"].filename = utils.get_path(
            ["estimator_example", "temperature_sensor.csv"]
        )
        model.components["020B_co2_sensor"].filename = utils.get_path(
            ["estimator_example", "co2_sensor.csv"]
        )
        model.components["020B_valve_position_sensor"].filename = utils.get_path(
            ["estimator_example", "valve_position_sensor.csv"]
        )
        model.components["020B_damper_position_sensor"].filename = utils.get_path(
            ["estimator_example", "damper_position_sensor.csv"]
        )
        model.components["BTA004"].filename = utils.get_path(
            ["estimator_example", "supply_air_temperature.csv"]
        )
        model.components["020B_temperature_heating_setpoint"].filename = utils.get_path(
            ["estimator_example", "temperature_heating_setpoint.csv"]
        )
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

        # Set up a temporary directory for saving estimation results
        # This ensures files are cleaned up and works in GitHub Actions
        model.dir_conf = [self.test_dir, "model_parameters"]

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
        model.initialize(start_time, end_time, step_size, simulator)

        # Step 3: Identify key components
        space = model.components["020B"]
        space_heater = model.components["020B_space_heater"]
        heating_controller = model.components["020B_temperature_heating_controller"]
        co2_controller = model.components["020B_co2_controller"]
        space_heater_valve = model.components["020B_space_heater_valve"]
        supply_damper = model.components["020B_room_supply_damper"]
        exhaust_damper = model.components["020B_room_exhaust_damper"]

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
            (model.components["020B_temperature_sensor"], 0.1 / percentile),
            (model.components["020B_co2_sensor"], 30 / percentile),
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
        self.assertIn("component_id", result, "Result should contain 'component_id'")
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
        model2.components["020B_temperature_sensor"].filename = utils.get_path(
            ["estimator_example", "temperature_sensor.csv"]
        )
        model2.components["020B_co2_sensor"].filename = utils.get_path(
            ["estimator_example", "co2_sensor.csv"]
        )
        model2.components["020B_valve_position_sensor"].filename = utils.get_path(
            ["estimator_example", "valve_position_sensor.csv"]
        )
        model2.components["020B_damper_position_sensor"].filename = utils.get_path(
            ["estimator_example", "damper_position_sensor.csv"]
        )
        model2.components["BTA004"].filename = utils.get_path(
            ["estimator_example", "supply_air_temperature.csv"]
        )
        model2.components["020B_temperature_heating_setpoint"].filename = (
            utils.get_path(["estimator_example", "temperature_heating_setpoint.csv"])
        )
        model2.components["outdoor_environment"].filename_outdoorTemperature = (
            utils.get_path(["estimator_example", "outdoor_environment.csv"])
        )
        model2.components["outdoor_environment"].filename_globalIrradiation = (
            utils.get_path(["estimator_example", "outdoor_environment.csv"])
        )
        model2.components["outdoor_environment"].valuecolumn_outdoorCo2Concentration = 2
        model2.components["outdoor_environment"].filename_outdoorCo2Concentration = (
            utils.get_path(["estimator_example", "outdoor_environment.csv"])
        )

        # Step 11: Load the estimation result from file
        model2.load_estimation_result(filename=saved_file_path)

        # Step 12: Verify that parameters were loaded
        # Check that the loaded result is accessible
        self.assertIsNotNone(
            model2.simulation_model._result, "Model should have a result after loading"
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
        space_loaded = model2.components["020B"]
        self.assertIsNotNone(
            space_loaded.output, "Space component should have output after simulation"
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

        heater_loaded = model2.components["020B_space_heater"]
        self.assertAlmostEqual(
            heater_loaded.thermalMassHeatCapacity,
            space_heater.thermalMassHeatCapacity,
            delta=0.1 * heater_loaded.thermalMassHeatCapacity,
        )

        # Test passed if we got here without exceptions
        self.assertTrue(True, "Successfully saved, loaded, and used estimation results")


if __name__ == "__main__":
    unittest.main()
