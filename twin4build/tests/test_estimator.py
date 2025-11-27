import unittest
import datetime
import shutil
import os
import pandas as pd
import numpy as np
import pytz
from twin4build.model.model import Model
from twin4build.simulator.simulator import Simulator
from twin4build.estimator.estimator import Estimator
from twin4build.systems.schedule.schedule_system import ScheduleSystem
from twin4build.systems.damper.damper_torch_system import DamperTorchSystem
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
                "ruleset_default_value": 0
            },
            id="schedule"
        )
        self.damper = DamperTorchSystem(id="damper", a=1.0, nominalAirFlowRate=0.5)
        
        self.model.add_component(self.schedule)
        self.model.add_component(self.damper)
        self.model.add_connection(self.schedule, self.damper, "scheduleValue", "damperPosition")

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
        start_time = datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)
        end_time = datetime.datetime(2023, 1, 1, 1, 0, 0, tzinfo=pytz.UTC)
        step_size = 600
        
        # First, run simulation to generate "true" data with known parameters
        true_a = 1.0
        self.damper.a.set(true_a, normalized=False)
        
        self.model.load()
        simulator = Simulator(self.model)
        simulator.simulate(start_time=start_time, end_time=end_time, step_size=step_size)
        
        # Extract the simulated air flow rate as "measurement"
        true_airflow = self.damper.output["airFlowRate"].history.detach().numpy()[0]
        
        # Create a sensor with this "measured" data
        # Note: simulation produces n_timesteps which may not include the end time
        dates = pd.date_range(start=start_time, periods=len(true_airflow), freq=f'{step_size}s')
        df = pd.DataFrame({
            'value': true_airflow
        },
        index=dates
        )
        
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
                "ruleset_default_value": 0
            },
            id="schedule"
        )
        damper_est = DamperTorchSystem(id="damper", a=0.5, nominalAirFlowRate=0.5)  # Wrong initial value
        model_est.add_connection(schedule_est, damper_est, "scheduleValue", "damperPosition")
        model_est.add_connection(damper_est, sensor, "airFlowRate", "measuredValue")
        model_est.load()
        
        # Create estimator
        sim_est = Simulator(model_est)
        estimator = Estimator(sim_est)
        
        # Define parameters to estimate - new list format
        parameters = [
            (damper_est, "a", 0.5, 0.1, 2.0, "private"),  # (component, attr, x0, lb, ub, type)
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
            n_warmup=0
        )
        
        # Check that estimation converged close to true value
        estimated_a = damper_est.a.get().item()
        self.assertAlmostEqual(estimated_a, true_a, places=2)
        
        # Cleanup
        if os.path.exists("generated_files/models/test_est_model_2"):
            shutil.rmtree("generated_files/models/test_est_model_2")

    def test_estimate_with_invalid_bounds(self):
        """Test that estimation raises error when lower bound > upper bound."""
        start_time = datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)
        end_time = datetime.datetime(2023, 1, 1, 0, 30, 0, tzinfo=pytz.UTC)
        step_size = 600
        
        # Create simple model with sensor
        dates = pd.date_range(start=start_time, end=end_time, freq=f'{step_size}s')
        df = pd.DataFrame({'value': np.ones(len(dates)) * 0.5}, index=dates)
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
        start_time = datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)
        end_time = datetime.datetime(2023, 1, 1, 0, 30, 0, tzinfo=pytz.UTC)
        step_size = 600
        
        self.model.load()
        simulator = Simulator(self.model)
        estimator = Estimator(simulator)
        
        parameters = [
            (self.damper, "a", 1.0, 0.5, 2.0, "private"),
        ]
        
        # Should raise error when measurements is empty or None
        with self.assertRaises((AssertionError, ValueError, TypeError, UnboundLocalError)):
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
        start_time = datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)
        end_time = datetime.datetime(2023, 1, 1, 0, 30, 0, tzinfo=pytz.UTC)
        step_size = 600
        
        dates = pd.date_range(start=start_time, end=end_time, freq=f'{step_size}s')
        df = pd.DataFrame({'value': np.ones(len(dates)) * 0.5}, index=dates)
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
        start_time = datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)
        end_time = datetime.datetime(2023, 1, 1, 0, 30, 0, tzinfo=pytz.UTC)
        step_size = 600
        
        dates = pd.date_range(start=start_time, end=end_time, freq=f'{step_size}s')
        df = pd.DataFrame({'value': np.ones(len(dates)) * 0.5}, index=dates)
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
        start_time = datetime.datetime(2023, 1, 1, 1, 0, 0, tzinfo=pytz.UTC)
        end_time = datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)  # Before start_time
        step_size = 600
        
        dates = pd.date_range(start=datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC), 
                             end=datetime.datetime(2023, 1, 1, 1, 0, 0, tzinfo=pytz.UTC), 
                             freq=f'{step_size}s')
        df = pd.DataFrame({'value': np.ones(len(dates)) * 0.5}, index=dates)
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

if __name__ == '__main__':
    unittest.main()
