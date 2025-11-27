import unittest
import datetime
import torch
import pytz
from twin4build.systems.schedule.schedule_system import ScheduleSystem
from twin4build.systems.damper.damper_torch_system import DamperTorchSystem
from twin4build.systems.fan.fan_torch_system import FanTorchSystem
from twin4build.systems.valve.valve_torch_system import ValveTorchSystem
from twin4build.systems.controller.rulebased_controller.on_off_controller.on_off_controller_system import OnOffControllerSystem
from twin4build.systems.controller.setpoint_controller.pid_controller.pid_controller_system import PIDControllerSystem
from twin4build.systems.outdoor_environment.outdoor_environment_system import OutdoorEnvironmentSystem
from twin4build.systems.sensor.sensor_system import SensorSystem
from twin4build.systems.junction.supply_flow_junction_system import SupplyFlowJunctionSystem
from twin4build.systems.junction.return_flow_junction_system import ReturnFlowJunctionSystem
from twin4build.systems.utils.time_series_input_system import TimeSeriesInputSystem
import pandas as pd
import numpy as np


class TestScheduleSystem(unittest.TestCase):
    def setUp(self):
        self.schedule = ScheduleSystem(
            weekDayRulesetDict={
                "ruleset_start_minute": [0],
                "ruleset_end_minute": [0],
                "ruleset_start_hour": [0],
                "ruleset_end_hour": [1],
                "ruleset_value": [20],
                "ruleset_default_value": 0
            },
            id="test_schedule"
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
        self.schedule.initialize(start_time=start_time, end_time=end_time, step_size=step_size)
        
        # Execute a time step
        datetime_val = datetime.datetime(2023, 1, 1, 0, 30, 0, tzinfo=pytz.UTC)
        self.schedule.do_step(second_time=1800, date_time=datetime_val, step_size=600, step_index=3)
        
        # Check that output was set
        self.assertIsNotNone(self.schedule.output["scheduleValue"].get())


class TestDamperTorchSystem(unittest.TestCase):
    def setUp(self):
        self.damper = DamperTorchSystem(
            id="test_damper",
            a=1.0,
            nominalAirFlowRate=0.5
        )

    def test_initialization(self):
        """Test damper system initialization."""
        self.assertIsNotNone(self.damper)
        self.assertEqual(self.damper.id, "test_damper")

    def test_do_step(self):
        """Test damper system do_step method."""
        # Initialize
        start_time = [datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)]
        end_time = [datetime.datetime(2023, 1, 1, 1, 40, 0, tzinfo=pytz.UTC)]
        step_size = [600]
        self.damper.initialize(start_time=start_time, end_time=end_time, step_size=step_size)
        
        # Set input
        self.damper.input["damperPosition"].set(torch.tensor([0.5]), step_index=0)
        
        # Execute a time step
        datetime_val = datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)
        self.damper.do_step(second_time=0, date_time=datetime_val, step_size=600, step_index=0)
        
        # Check that output was calculated
        airflow = self.damper.output["airFlowRate"].get()
        self.assertIsNotNone(airflow)
        self.assertGreater(airflow.item(), 0)

    def test_airflow_calculation(self):
        """Test that damper calculates airflow correctly."""
        start_time = [datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)]
        end_time = [datetime.datetime(2023, 1, 1, 1, 40, 0, tzinfo=pytz.UTC)]
        step_size = [600]
        self.damper.initialize(start_time=start_time, end_time=end_time, step_size=step_size)
        
        # At 100% position, should give nominal airflow
        self.damper.input["damperPosition"].set(torch.tensor([1.0]), step_index=0)
        self.damper.do_step(second_time=0, date_time=datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC), step_size=600, step_index=0)
        
        airflow = self.damper.output["airFlowRate"].get().item()
        self.assertAlmostEqual(airflow, 0.5, places=2)


class TestFanTorchSystem(unittest.TestCase):
    def setUp(self):
        self.fan = FanTorchSystem(id="test_fan", nominalPowerRate=1000.0, nominalAirFlowRate=1.0, c1=0, c2=0.8, c3=0.2, c4=0.0, f_total=0.9)

    def test_initialization(self):
        """Test fan system initialization."""
        start_time = [datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)]
        end_time = [datetime.datetime(2023, 1, 1, 1, 40, 0, tzinfo=pytz.UTC)]
        step_size = [600]
        self.fan.initialize(start_time=start_time, end_time=end_time, step_size=step_size)
        self.assertIsNotNone(self.fan)
        self.assertEqual(self.fan.id, "test_fan")

    def test_do_step(self):
        """Test fan system do_step method."""
        start_time = [datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)]
        end_time = [datetime.datetime(2023, 1, 1, 1, 40, 0, tzinfo=pytz.UTC)]
        step_size = [600]
        self.fan.initialize(start_time=start_time, end_time=end_time, step_size=step_size)

        # Set inputs
        self.fan.input["airFlowRate"].set(torch.tensor([1.0]), step_index=0)
        self.fan.input["inletAirTemperature"].set(torch.tensor([20.0]), step_index=0)
        
        # Set required inputs (may vary based on implementation)
        # Execute a time step
        datetime_val = datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)
        self.fan.do_step(second_time=0, date_time=datetime_val, step_size=600, step_index=0)
        
        self.assertIsNotNone(self.fan.output["outletAirTemperature"].get())
        self.assertIsNotNone(self.fan.output["Power"].get())
        self.assertGreater(self.fan.output["outletAirTemperature"].get().item(), 20.0)
        self.assertGreater(self.fan.output["Power"].get().item(), 0.0)


class TestValveTorchSystem(unittest.TestCase):
    def setUp(self):
        self.valve = ValveTorchSystem(id="test_valve", valveAuthority=0.5, waterFlowRateMax=1.0)

    def test_initialization(self):
        """Test valve system initialization."""
        self.assertIsNotNone(self.valve)
        self.assertEqual(self.valve.id, "test_valve")

    def test_do_step(self):
        """Test valve system do_step method."""
        start_time = [datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)]
        end_time = [datetime.datetime(2023, 1, 1, 1, 40, 0, tzinfo=pytz.UTC)]
        step_size = [600]
        self.valve.initialize(start_time=start_time, end_time=end_time, step_size=step_size)

        # Set inputs
        self.valve.input["valvePosition"].set(torch.tensor([0.5]), step_index=0)
        
        # Execute a time step
        datetime_val = datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)
        self.valve.do_step(second_time=0, date_time=datetime_val, step_size=600, step_index=0)

        output = self.valve.output["waterFlowRate"].get()
        self.assertIsNotNone(output)
        self.assertGreater(output.item(), 0)


class TestOnOffControllerSystem(unittest.TestCase):
    def setUp(self):
        self.controller = OnOffControllerSystem(id="test_controller", offValue=0, onValue=1, isReverse=False)

    def test_initialization(self):
        """Test on/off controller initialization."""
        self.assertIsNotNone(self.controller)
        self.assertEqual(self.controller.id, "test_controller")

    def test_do_step(self):
        """Test on/off controller do_step method."""
        start_time = [datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)]
        end_time = [datetime.datetime(2023, 1, 1, 1, 40, 0, tzinfo=pytz.UTC)]
        step_size = [600]
        self.controller.initialize(start_time=start_time, end_time=end_time, step_size=step_size)
        
        # Set inputs
        self.controller.input["actualValue"].set(torch.tensor([20.0]), step_index=0)
        self.controller.input["setpointValue"].set(torch.tensor([22.0]), step_index=0)
        
        # Execute a time step
        datetime_val = datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)
        self.controller.do_step(second_time=0, date_time=datetime_val, step_size=600, step_index=0)
        
        # Check output
        output = self.controller.output["inputSignal"].get()
        self.assertIsNotNone(output)


class TestPIDControllerSystem(unittest.TestCase):
    def setUp(self):
        self.controller = PIDControllerSystem(id="test_pid", Kp=1.0, Ki=0.1, Kd=0.01)

    def test_initialization(self):
        """Test PID controller initialization."""
        self.assertIsNotNone(self.controller)
        self.assertEqual(self.controller.id, "test_pid")

    def test_do_step(self):
        """Test PID controller do_step method."""
        start_time = [datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)]
        end_time = [datetime.datetime(2023, 1, 1, 1, 40, 0, tzinfo=pytz.UTC)]
        step_size = [600]
        self.controller.initialize(start_time=start_time, end_time=end_time, step_size=step_size)
        
        # Set inputs
        self.controller.input["actualValue"].set(torch.tensor([20.0]), step_index=0)
        self.controller.input["setpointValue"].set(torch.tensor([22.0]), step_index=0)
        
        # Execute a time step
        datetime_val = datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)
        self.controller.do_step(second_time=0, date_time=datetime_val, step_size=600, step_index=0)
        
        # Check output
        output = self.controller.output["inputSignal"].get()
        self.assertIsNotNone(output)


class TestSensorSystem(unittest.TestCase):
    def setUp(self):
        dates = pd.date_range(
            start=datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC),
            periods=10,
            freq='10min'
        )
        df = pd.DataFrame({'value': [20.0] * 10}, index=dates)
        self.sensor = SensorSystem(id="test_sensor", df=df)

    def test_initialization(self):
        """Test sensor system initialization."""
        self.assertIsNotNone(self.sensor)
        self.assertEqual(self.sensor.id, "test_sensor")

    def test_do_step(self):
        """Test sensor system do_step method."""
        start_time = [datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)]
        end_time = [datetime.datetime(2023, 1, 1, 1, 40, 0, tzinfo=pytz.UTC)]
        step_size = [600]
        self.sensor.initialize(start_time=start_time, end_time=end_time, step_size=step_size)
        
        # Execute a time step
        datetime_val = datetime.datetime(2023, 1, 1, 0, 10, 0, tzinfo=pytz.UTC)
        self.sensor.do_step(second_time=600, date_time=datetime_val, step_size=600, step_index=1)
        
        # Check that measured value was set
        measured = self.sensor.output["measuredValue"].get()
        self.assertIsNotNone(measured)
        self.assertAlmostEqual(measured.item(), 20.0, places=1)


class TestOutdoorEnvironmentSystem(unittest.TestCase):
    def setUp(self):
        self.outdoor_env = OutdoorEnvironmentSystem(id="test_outdoor")

    def test_initialization(self):
        """Test outdoor environment system initialization."""
        self.assertIsNotNone(self.outdoor_env)
        self.assertEqual(self.outdoor_env.id, "test_outdoor")


class TestFlowJunctions(unittest.TestCase):
    def test_supply_flow_junction_initialization(self):
        """Test supply flow junction initialization."""
        junction = SupplyFlowJunctionSystem(id="test_supply_junction")
        self.assertIsNotNone(junction)
        self.assertEqual(junction.id, "test_supply_junction")

    def test_return_flow_junction_initialization(self):
        """Test return flow junction initialization."""
        junction = ReturnFlowJunctionSystem(id="test_return_junction")
        self.assertIsNotNone(junction)
        self.assertEqual(junction.id, "test_return_junction")


class TestTimeSeriesInputSystem(unittest.TestCase):
    def setUp(self):
        dates = pd.date_range(
            start=datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC),
            periods=10,
            freq='10min'
        )
        df = pd.DataFrame({'value': [0.5] * 10}, index=dates)
        self.ts_input = TimeSeriesInputSystem(id="test_timeseries", df=df)

    def test_initialization(self):
        """Test time series input system initialization."""
        self.assertIsNotNone(self.ts_input)
        self.assertEqual(self.ts_input.id, "test_timeseries")

    def test_do_step(self):
        """Test time series input system do_step method."""
        start_time = [datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)]
        end_time = [datetime.datetime(2023, 1, 1, 1, 40, 0, tzinfo=pytz.UTC)]
        step_size = [600]
        self.ts_input.initialize(start_time=start_time, end_time=end_time, step_size=step_size)
        
        # Execute a time step
        datetime_val = datetime.datetime(2023, 1, 1, 0, 10, 0, tzinfo=pytz.UTC)
        self.ts_input.do_step(second_time=600, date_time=datetime_val, step_size=600, step_index=1)
        
        # Check that value was set
        value = self.ts_input.output["value"].get()
        self.assertIsNotNone(value)
        self.assertAlmostEqual(value.item(), 0.5, places=5)



if __name__ == '__main__':
    unittest.main()

