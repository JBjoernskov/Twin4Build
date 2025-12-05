# Standard library imports
import datetime
import unittest

# Third party imports
import numpy as np
from dateutil import tz

# Local application imports
import twin4build as tb
import twin4build.examples.utils as utils


class TestIterationMethods(unittest.TestCase):
    """
    Test that both 'gauss-seidel' and 'jacobi' iteration methods in the Simulator
    produce consistent results.

    With a small enough step_size, both methods should converge to nearly identical
    results. This test verifies that the jacobi iteration method is working correctly
    by comparing its output to the gauss-seidel method.
    """

    def _setup_model(self, model_id: str) -> tb.Model:
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
        model_gs = self._setup_model("test_gauss_seidel")
        simulator_gs = tb.Simulator(model_gs)
        simulator_gs.simulate(
            step_size=step_size,
            start_time=start_time,
            end_time=end_time,
            show_progress_bar=False,
            iteration_method="gauss-seidel",
        )

        # Set up and run jacobi simulation
        model_jacobi = self._setup_model("test_jacobi")
        simulator_jacobi = tb.Simulator(model_jacobi)
        simulator_jacobi.simulate(
            step_size=step_size,
            start_time=start_time,
            end_time=end_time,
            show_progress_bar=False,
            iteration_method="jacobi",
        )

        # Compare results for key outputs
        space_gs = model_gs.components["020B"]
        space_jacobi = model_jacobi.components["020B"]

        # Get indoor temperature outputs
        temp_gs = space_gs.output["indoorTemperature"].history.detach().numpy()
        temp_jacobi = space_jacobi.output["indoorTemperature"].history.detach().numpy()

        # Get CO2 concentration outputs
        co2_gs = space_gs.output["indoorCO2"].history.detach().numpy()
        co2_jacobi = space_jacobi.output["indoorCO2"].history.detach().numpy()


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
        self.assertTrue(len(temp_gs) > 0, "Gauss-Seidel simulation should produce results")
        self.assertTrue(len(temp_jacobi) > 0, "Jacobi simulation should produce results")

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

        model = self._setup_model("test_jacobi_smoke")
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
        space = model.components["020B"]
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

        model = self._setup_model("test_default_method")
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

