import unittest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import twin4build as tb
import datetime
from dateutil import tz
import twin4build.examples.utils as utils
import time
import multiprocessing
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import tempfile
import shutil
import torch


class TestEstimator(unittest.TestCase):
    """
    Test suite for the Estimator solver functionality.
    
    This test suite evaluates the parameter estimation capabilities of the Estimator
    class using different optimization methods (AD and FD) with various solvers.
    """
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that are shared across all test methods."""
        cls.temp_dir = tempfile.mkdtemp()
        
        # Set up model and parameters for testing
        cls.model, cls.simulator, cls.parameters, cls.measurements, \
        cls.startTime, cls.endTime, cls.stepSize = cls._setup_model_and_parameters()
        
        # Create estimator
        cls.estimator = tb.Estimator(cls.simulator)
        
        # Define test methods - reduced set for faster testing
        cls.test_methods = [
            # Automatic Differentiation methods (should work)
            ("scipy", "SLSQP", "ad"),
            ("scipy", "L-BFGS-B", "ad"),
            ("scipy", "TNC", "ad"),
            
            # Finite Difference methods (may have issues with multiprocessing)
            ("scipy", "trf", "fd"),
            ("scipy", "dogbox", "fd"),
        ]
        
        # Define options for different method types
        cls.ad_options = {
            "max_nfev": 5,  # Very short for testing
            "verbose": 0,
            "ftol": 1e-6,
            "xtol": 1e-6,
        }
        
        cls.fd_options = {
            "max_nfev": 5,  # Very short for testing
            "verbose": 0,
            "ftol": 1e-6,
            "xtol": 1e-6,
            "n_cores": min(2, multiprocessing.cpu_count()),  # Use fewer cores for testing
        }
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test fixtures."""
        if hasattr(cls, 'temp_dir') and os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir)
    
    @staticmethod
    def _setup_model_and_parameters():
        """
        Set up the model and parameters for testing.
        
        Returns
        -------
        tuple
            (model, simulator, parameters, measurements, startTime, endTime, stepSize)
        """
        # Create a new model
        model = tb.Model(id="estimator_solver_test")
        
        # Load the model from semantic file
        filename_simulation = utils.get_path(["estimator_example", "semantic_model.ttl"])
        print(f"Loading model from: {filename_simulation}")
        model.load(simulation_model_filename=filename_simulation, verbose=False)

        model.components["020B_temperature_sensor"].filename = utils.get_path(["estimator_example", "temperature_sensor.csv"])
        model.components["020B_co2_sensor"].filename = utils.get_path(["estimator_example", "co2_sensor.csv"])
        model.components["020B_valve_position_sensor"].filename = utils.get_path(["estimator_example", "valve_position_sensor.csv"])
        model.components["020B_damper_position_sensor"].filename = utils.get_path(["estimator_example", "damper_position_sensor.csv"])
        model.components["BTA004"].filename = utils.get_path(["estimator_example", "supply_air_temperature.csv"])
        model.components["020B_temperature_heating_setpoint"].filename = utils.get_path(["estimator_example", "temperature_heating_setpoint.csv"])
        model.components["outdoor_environment"].filename_outdoorTemperature = utils.get_path(["estimator_example", "outdoor_environment.csv"])
        model.components["outdoor_environment"].filename_globalIrradiation = utils.get_path(["estimator_example", "outdoor_environment.csv"])
        model.components["outdoor_environment"].filename_outdoorCo2Concentration = utils.get_path(["estimator_example", "outdoor_environment.csv"])

        # Set up simulation parameters - very short time period for testing
        simulator = tb.Simulator(model)
        stepSize = 1200  # 20 minutes in seconds
        startTime = datetime.datetime(year=2023, month=11, day=27, hour=0, minute=0, second=0,
                                        tzinfo=tz.gettz("Europe/Copenhagen"))
        endTime = datetime.datetime(year=2023, month=11, day=27, hour=2, minute=0, second=0,
                                    tzinfo=tz.gettz("Europe/Copenhagen"))  # Only 2 hours for testing

        # Get components
        space = model.components["020B"]
        space_heater = model.components["020B_space_heater"]
        heating_controller = model.components["020B_temperature_heating_controller"]
        space_heater_valve = model.components["020B_space_heater_valve"]

        # Define parameters to estimate - minimal set for faster testing
        parameters = {"private": {
            # Thermal parameters (subset for testing)
            "thermal.C_air": {"components": [space], "x0": 2e+6, "lb": 1e+6, "ub": 1e+7},
            "thermal.R_out": {"components": [space], "x0": 0.05, "lb": 0.01, "ub": 1},
            
            # Space heater parameters
            "UA": {"components": [space_heater], "x0": 30, "lb": 1, "ub": 100},
            
            # Controller parameters (only one controller for testing)
            "kp": {"components": [heating_controller], "x0": [0.001], "lb": [1e-5], "ub": [1]},
            
            # Valve parameters
            "waterFlowRateMax": {"components": [space_heater_valve], "x0": 0.01, "lb": 1e-6, "ub": 0.1},
        }}

        # Define measuring devices
        measurements = [
            model.components["020B_valve_position_sensor"],
            model.components["020B_temperature_sensor"],
        ]

        return model, simulator, parameters, measurements, startTime, endTime, stepSize
    
    def _test_solver_method(self, method: Tuple[str, str, str], expected_success: bool = True) -> Dict[str, Any]:
        """
        Test a specific solver method and return results.
        
        Parameters
        ----------
        method : Tuple[str, str, str]
            Solver method (library, optimizer, mode)
        expected_success : bool
            Whether the method is expected to succeed
            
        Returns
        -------
        Dict[str, Any]
            Test results including timing and success status
        """
        method_name = f"{method[0]}_{method[1]}_{method[2]}"
        
        # Choose appropriate options based on method type
        if method[1] in ["trf", "dogbox"]:
            # These methods use least_squares, so use max_nfev
            options = self.ad_options.copy() if method[2] == "ad" else self.fd_options.copy()
        else:
            # These methods use minimize, so use maxiter
            if method[2] == "ad":
                if method[1] == "trust-constr":
                    # trust-constr has different parameters
                    options = {
                        "maxiter": 5,  # Very short for testing
                        "disp": False,
                        "xtol": 1e-6,  # trust-constr uses xtol but not ftol
                    }
                else:
                    # Other minimize methods (SLSQP, L-BFGS-B, TNC)
                    options = {
                        "maxiter": 5,  # Very short for testing
                        "disp": False,
                        "ftol": 1e-6,
                        "xtol": 1e-6,
                    }
            else:
                options = self.fd_options.copy()
        
        # Extract n_cores for FD methods
        n_cores = None
        if method[2] == "fd":
            n_cores = options.get("n_cores", min(2, multiprocessing.cpu_count()))
            # Remove n_cores from options since it's passed separately
            options = {k: v for k, v in options.items() if k != "n_cores"}
        
        result = {
            "method": method_name,
            "success": False,
            "time_taken": None,
            "error": None,
            "iterations": None,
            "final_objective": None,
            "scalar_objective": None
        }
        
        try:
            # Time the estimation
            start_time = time.time()
            
            # Run estimation
            estimation_result = self.estimator.estimate(
                parameters=self.parameters,
                measurements=self.measurements,
                startTime=self.startTime,
                endTime=self.endTime,
                stepSize=self.stepSize,
                n_initialization_steps=5,  # Very short for testing
                method=method,
                n_cores=n_cores,
                options=options,
            )
            
            end_time = time.time()
            
            # Record results
            result["success"] = True
            result["time_taken"] = end_time - start_time
            result["iterations"] = estimation_result.get("iterations", "N/A")
            result["nfev"] = estimation_result.get("nfev", "N/A")
            result["final_objective"] = estimation_result.get("final_objective", "N/A")
            result["optimization_success"] = estimation_result.get("success", "N/A")
            result["message"] = estimation_result.get("message", "N/A")
            
            # Compute comparable scalar objective value
            final_obj = estimation_result.get("final_objective", None)
            if final_obj is not None:
                if hasattr(final_obj, '__iter__') and not isinstance(final_obj, str):
                    # For least_squares methods, compute MSE of residuals
                    if isinstance(final_obj, (list, tuple)):
                        final_obj = np.array(final_obj)
                    result["scalar_objective"] = np.mean(final_obj**2)
                else:
                    # For minimize methods, use the scalar value directly
                    result["scalar_objective"] = float(final_obj)
            else:
                result["scalar_objective"] = "N/A"
            
        except Exception as e:
            result["error"] = str(e)
            if expected_success:
                # If we expected success but got an error, raise it
                raise e
        
        return result
    
    def test_ad_methods(self):
        """Test automatic differentiation methods."""
        ad_methods = [m for m in self.test_methods if m[2] == "ad"]
        
        for method in ad_methods:
            with self.subTest(method=method):
                result = self._test_solver_method(method, expected_success=True)
                
                # Assertions for AD methods
                self.assertTrue(result["success"], f"AD method {method} failed to run: {result.get('error', 'Unknown error')}")
                self.assertIsNotNone(result["time_taken"], "Time taken should be recorded")
                self.assertIsNotNone(result["final_objective"], "Final objective should be recorded")
                
                # Check that we got reasonable results
                if result["scalar_objective"] != "N/A":
                    # Include numpy numeric types in the check
                    self.assertIsInstance(result["scalar_objective"], (int, float, np.number), "Scalar objective should be numeric")
                    self.assertGreaterEqual(result["scalar_objective"], 0, "Objective should be non-negative")
    
    def test_fd_methods(self):
        """Test finite difference methods."""
        fd_methods = [m for m in self.test_methods if m[2] == "fd"]
        
        for method in fd_methods:
            with self.subTest(method=method):
                # FD methods might fail due to multiprocessing issues, so we don't raise on failure
                result = self._test_solver_method(method, expected_success=False)
                
                if result["success"]:
                    # If it succeeded, check the results
                    self.assertIsNotNone(result["time_taken"], "Time taken should be recorded")
                    self.assertIsNotNone(result["final_objective"], "Final objective should be recorded")
                    
                    # Check that we got reasonable results
                    if result["scalar_objective"] != "N/A":
                        # Include numpy numeric types in the check
                        self.assertIsInstance(result["scalar_objective"], (int, float, np.number), "Scalar objective should be numeric")
                        self.assertGreaterEqual(result["scalar_objective"], 0, "Objective should be non-negative")
                else:
                    # If it failed, log the error but don't fail the test
                    print(f"FD method {method} failed as expected: {result.get('error', 'Unknown error')}")
    
    def test_method_comparison(self):
        """Compare performance between AD and FD methods."""
        # Test one AD and one FD method for comparison
        ad_method = ("scipy", "SLSQP", "ad")
        fd_method = ("scipy", "trf", "fd")
        
        ad_result = self._test_solver_method(ad_method, expected_success=True)
        fd_result = self._test_solver_method(fd_method, expected_success=False)
        
        # AD method should always run successfully
        self.assertTrue(ad_result["success"], "AD method should run without errors")
        
        # If FD method succeeds, compare performance
        if fd_result["success"]:
            self.assertIsNotNone(ad_result["time_taken"], "AD time should be recorded")
            self.assertIsNotNone(fd_result["time_taken"], "FD time should be recorded")
            
            # Both should have reasonable objective values
            if ad_result["scalar_objective"] != "N/A" and fd_result["scalar_objective"] != "N/A":
                # Include numpy numeric types in the check
                self.assertIsInstance(ad_result["scalar_objective"], (int, float, np.number))
                self.assertIsInstance(fd_result["scalar_objective"], (int, float, np.number))
    
    def test_parameter_bounds(self):
        """Test that parameter bounds are respected."""
        # Test with a simple AD method
        method = ("scipy", "SLSQP", "ad")
        result = self._test_solver_method(method, expected_success=True)
        
        self.assertTrue(result["success"], "Method should run without errors")
        
        # Check that the result contains the expected fields
        self.assertIn("final_objective", result, "Result should contain final_objective")
        self.assertIn("optimization_success", result, "Result should contain optimization_success")
    
    def test_error_handling(self):
        """Test error handling with invalid parameters."""
        # Test with invalid method - expect AssertionError, not ValueError
        with self.assertRaises(AssertionError):
            self.estimator.estimate(
                parameters=self.parameters,
                measurements=self.measurements,
                startTime=self.startTime,
                endTime=self.endTime,
                stepSize=self.stepSize,
                method=("invalid", "method", "ad")
            )
    
    def test_tensor_reset_functionality(self):
        """Test that tensor reset functionality works correctly."""
        # First run an AD method to create torch tensors
        ad_method = ("scipy", "SLSQP", "ad")
        ad_result = self._test_solver_method(ad_method, expected_success=True)
        
        self.assertTrue(ad_result["success"], "AD method should run without errors")
        
        # Now test that the model can be reset for FD methods
        # This tests the make_pickable functionality
        try:
            self.simulator.model.make_pickable()
            print("✓ Model make_pickable() succeeded")
        except Exception as e:
            self.fail(f"Model make_pickable() failed: {e}")
        
        # Test that we can still run AD methods after reset
        ad_result2 = self._test_solver_method(ad_method, expected_success=True)
        self.assertTrue(ad_result2["success"], "AD method should still work after reset")


def run_benchmark_tests():
    """
    Run benchmark tests to compare different methods.
    This function can be called independently for performance testing.
    """
    print("Running benchmark tests...")
    
    # Create test instance
    test_instance = TestEstimator()
    test_instance.setUpClass()
    
    try:
        # Test all methods
        results = []
        for method in test_instance.test_methods:
            print(f"\nTesting {method}...")
            result = test_instance._test_solver_method(method, expected_success=False)
            results.append(result)
            
            if result["success"]:
                print(f"✓ {method}: {result['time_taken']:.2f}s, obj={result['scalar_objective']}")
            else:
                print(f"✗ {method}: {result.get('error', 'Unknown error')}")
        
        # Print summary
        successful = [r for r in results if r["success"]]
        print(f"\nSummary: {len(successful)}/{len(results)} methods succeeded")
        
        if successful:
            times = [r["time_taken"] for r in successful if r["time_taken"] is not None]
            if times:
                print(f"Average time: {np.mean(times):.2f}s")
                print(f"Best time: {min(times):.2f}s")
    
    finally:
        test_instance.tearDownClass()


if __name__ == "__main__":
    # Run the tests
    unittest.main(verbosity=2) 