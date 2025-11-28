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
from twin4build.systems.space_heater.space_heater_torch_system import SpaceHeaterTorchSystem
from twin4build.systems.coil.coil_torch_system import CoilTorchSystem
from twin4build.systems.air_to_air_heat_recovery.air_to_air_heat_recovery_system import AirToAirHeatRecoverySystem
from twin4build.systems.utils.piecewise_linear_system import PiecewiseLinearSystem
from twin4build.systems.utils.discrete_statespace_system import DiscreteStatespaceSystem
from twin4build.systems.controller.rulebased_controller.rulebased_controller_system import RulebasedControllerSystem
import pandas as pd
import numpy as np


class TestSpaceHeaterTorchSystem(unittest.TestCase):
    def setUp(self):
        self.heater = SpaceHeaterTorchSystem(
            id="test_heater",
            Q_flow_nominal_sh=1000.0,
            T_a_nominal_sh=60.0,
            T_b_nominal_sh=40.0,
            TAir_nominal_sh=20.0,
            thermalMassHeatCapacity=5000.0,
            nelements=2
        )

    def test_initialization(self):
        """Test space heater system initialization."""
        self.assertIsNotNone(self.heater)
        self.assertEqual(self.heater.id, "test_heater")
        self.assertEqual(self.heater.nelements, 2)

    def test_do_step(self):
        """Test space heater system do_step method."""
        start_time = [datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)]
        end_time = [datetime.datetime(2023, 1, 1, 1, 40, 0, tzinfo=pytz.UTC)]
        step_size = [600]
        self.heater.initialize(start_time=start_time, end_time=end_time, step_size=step_size)
        
        # Set inputs
        self.heater.input["supplyWaterTemperature"].set(torch.tensor([60.0]), step_index=0)
        self.heater.input["waterFlowRate"].set(torch.tensor([0.1]), step_index=0)
        self.heater.input["indoorTemperature"].set(torch.tensor([20.0]), step_index=0)
        
        # Execute a time step
        datetime_val = datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)
        self.heater.do_step(second_time=0, date_time=datetime_val, step_size=step_size, step_index=0)
        
        # Check outputs
        outlet_temp = self.heater.output["outletWaterTemperature"].get()
        radiator_power = self.heater.output["Power"].get()
        
        self.assertIsNotNone(outlet_temp)
        self.assertIsNotNone(radiator_power)


class TestCoilTorchSystem(unittest.TestCase):
    def setUp(self):
        self.coil = CoilTorchSystem(id="test_coil")

    def test_initialization(self):
        """Test coil system initialization."""
        self.assertIsNotNone(self.coil)
        self.assertEqual(self.coil.id, "test_coil")

    def test_do_step(self):
        """Test coil system do_step method."""
        start_time = [datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)]
        end_time = [datetime.datetime(2023, 1, 1, 1, 40, 0, tzinfo=pytz.UTC)]
        step_size = [600]
        self.coil.initialize(start_time=start_time, end_time=end_time, step_size=step_size)
        
        # Set inputs
        self.coil.input["inletAirTemperature"].set(torch.tensor([20.0]), step_index=0)
        self.coil.input["outletAirTemperatureSetpoint"].set(torch.tensor([22.0]), step_index=0)
        self.coil.input["airFlowRate"].set(torch.tensor([1.0]), step_index=0)
        
        # Execute a time step
        datetime_val = datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)
        self.coil.do_step(second_time=0, date_time=datetime_val, step_size=600, step_index=0)
        
        # Check outputs
        heating_power = self.coil.output["heatingPower"].get()
        cooling_power = self.coil.output["coolingPower"].get()
        
        self.assertIsNotNone(heating_power)
        self.assertIsNotNone(cooling_power)
        # Should be heating
        self.assertGreater(heating_power.item(), 0)
        self.assertEqual(cooling_power.item(), 0)


class TestAirToAirHeatRecoverySystem(unittest.TestCase):
    def setUp(self):
        self.hr = AirToAirHeatRecoverySystem(
            id="test_hr",
            eps_75_h=0.8,
            eps_100_h=0.7,
            eps_75_c=0.8,
            eps_100_c=0.7,
            primaryAirFlowRateMax=1.0,
            secondaryAirFlowRateMax=1.0
        )

    def test_initialization(self):
        """Test heat recovery system initialization."""
        self.assertIsNotNone(self.hr)
        self.assertEqual(self.hr.id, "test_hr")

    def test_do_step(self):
        """Test heat recovery system do_step method."""
        start_time = [datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)]
        end_time = [datetime.datetime(2023, 1, 1, 1, 40, 0, tzinfo=pytz.UTC)]
        step_size = [600]
        self.hr.initialize(start_time=start_time, end_time=end_time, step_size=step_size)
        
        # Set inputs
        # Supply side (outdoor to indoor)
        self.hr.input["primaryTemperatureIn"].set(torch.tensor([0.0]), step_index=0) # Cold outdoor
        self.hr.input["primaryAirFlowRate"].set(torch.tensor([1.0]), step_index=0)
        self.hr.input["primaryTemperatureOutSetpoint"].set(torch.tensor([20.0]), step_index=0)

        # Exhaust side (indoor to outdoor)
        self.hr.input["secondaryTemperatureIn"].set(torch.tensor([20.0]), step_index=0) # Warm return
        self.hr.input["secondaryAirFlowRate"].set(torch.tensor([1.0]), step_index=0)
        
        # Execute a time step
        datetime_val = datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)
        self.hr.do_step(second_time=0, date_time=datetime_val, step_size=600, step_index=0)


        primary_in = self.hr.input["primaryTemperatureIn"].get()
        secondary_in = self.hr.input["secondaryTemperatureIn"].get()
        
        # Check outputs
        primary_out = self.hr.output["primaryTemperatureOut"].get()
        secondary_out = self.hr.output["secondaryTemperatureOut"].get()
        
        self.assertIsNotNone(primary_out)
        self.assertIsNotNone(secondary_out)
        
        # Expect heat recovery: primary temp should increase
        self.assertGreater(primary_out.item(), primary_in.item())
        # Secondary temp should decrease
        self.assertLess(secondary_out.item(), secondary_in.item())



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



class TestRulebasedControllerSystem(unittest.TestCase):
    def setUp(self):
        from twin4build.systems.controller.rulebased_controller.rulebased_controller_system import RulebasedControllerSystem
        self.controller = RulebasedControllerSystem(id="test_rulebased")

    def test_initialization(self):
        """Test rulebased controller initialization."""
        self.assertIsNotNone(self.controller)
        self.assertEqual(self.controller.id, "test_rulebased")
        self.assertEqual(self.controller.interval, 99)  # Default interval

    def test_do_step_low_value(self):
        """Test rulebased controller with low actual value (< 600)."""
        start_time = [datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)]
        end_time = [datetime.datetime(2023, 1, 1, 1, 40, 0, tzinfo=pytz.UTC)]
        step_size = [600]
        self.controller.initialize(start_time=start_time, end_time=end_time, step_size=step_size)
        
        # Set low input value
        self.controller.input["actualValue"].set(torch.tensor([500.0]), step_index=0)
        
        # Execute a time step
        datetime_val = datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)
        self.controller.do_step(second_time=0, date_time=datetime_val, step_size=600, step_index=0)
        
        # Check output - should be 0 for values below 600
        output = self.controller.output["inputSignal"].get()
        self.assertIsNotNone(output)
        self.assertEqual(output.item(), 0)

    def test_do_step_medium_value(self):
        """Test rulebased controller with medium actual value (600-750)."""
        start_time = [datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)]
        end_time = [datetime.datetime(2023, 1, 1, 1, 40, 0, tzinfo=pytz.UTC)]
        step_size = [600]
        self.controller.initialize(start_time=start_time, end_time=end_time, step_size=step_size)
        
        # Set medium input value
        self.controller.input["actualValue"].set(torch.tensor([650.0]), step_index=0)
        
        # Execute a time step
        datetime_val = datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)
        self.controller.do_step(second_time=0, date_time=datetime_val, step_size=600, step_index=0)
        
        # Check output - should be 0.45 for values between 600-750
        output = self.controller.output["inputSignal"].get()
        self.assertIsNotNone(output)
        self.assertAlmostEqual(output.item(), 0.45, places=2)

    def test_do_step_high_value(self):
        """Test rulebased controller with high actual value (> 900)."""
        start_time = [datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)]
        end_time = [datetime.datetime(2023, 1, 1, 1, 40, 0, tzinfo=pytz.UTC)]
        step_size = [600]
        self.controller.initialize(start_time=start_time, end_time=end_time, step_size=step_size)
        
        # Set high input value
        self.controller.input["actualValue"].set(torch.tensor([950.0]), step_index=0)
        
        # Execute a time step
        datetime_val = datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)
        self.controller.do_step(second_time=0, date_time=datetime_val, step_size=600, step_index=0)
        
        # Check output - should be 1.0 for values above 900
        output = self.controller.output["inputSignal"].get()
        self.assertIsNotNone(output)
        self.assertEqual(output.item(), 1.0)


class TestReturnFlowJunctionSystem(unittest.TestCase):
    def setUp(self):
        self.junction = ReturnFlowJunctionSystem(id="test_return_junction", airFlowRateBias=0.1)

    def test_initialization(self):
        """Test return flow junction initialization."""
        self.assertIsNotNone(self.junction)
        self.assertEqual(self.junction.id, "test_return_junction")
        self.assertEqual(self.junction.airFlowRateBias, 0.1)

    def test_initialization_default_bias(self):
        """Test return flow junction with default bias."""
        junction = ReturnFlowJunctionSystem(id="test_default")
        self.assertEqual(junction.airFlowRateBias, 0)

    def test_do_step(self):
        """Test return flow junction do_step method."""
        start_time = [datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)]
        end_time = [datetime.datetime(2023, 1, 1, 1, 40, 0, tzinfo=pytz.UTC)]
        step_size = [600]
        self.junction.initialize(start_time=start_time, end_time=end_time, step_size=step_size)
        
        # Set inputs - vector inputs for multiple flows
        self.junction.input["airFlowRateIn"].set(torch.tensor([[0.5, 0.5]]), step_index=0)
        self.junction.input["airTemperatureIn"].set(torch.tensor([[20.0, 22.0]]), step_index=0)
        
        # Execute a time step
        datetime_val = datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)
        self.junction.do_step(second_time=0, date_time=datetime_val, step_size=600, step_index=0)
        
        # Check outputs
        flow_out = self.junction.output["airFlowRateOut"].get()
        temp_out = self.junction.output["airTemperatureOut"].get()
        
        self.assertIsNotNone(flow_out)
        self.assertIsNotNone(temp_out)
        # Total flow = 0.5 + 0.5 + 0.1 (bias) = 1.1
        self.assertAlmostEqual(flow_out.item(), 1.1, places=2)
        # Weighted avg temp = (20*0.5 + 22*0.5) / 1.1 ≈ 19.09
        self.assertAlmostEqual(temp_out.item(), 21.0/1.1, places=1)

    def test_do_step_zero_flow(self):
        """Test return flow junction with zero flow."""
        start_time = [datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)]
        end_time = [datetime.datetime(2023, 1, 1, 1, 40, 0, tzinfo=pytz.UTC)]
        step_size = [600]
        
        junction = ReturnFlowJunctionSystem(id="test_zero_flow")
        junction.initialize(start_time=start_time, end_time=end_time, step_size=step_size)
        
        # Set inputs with zero flow
        junction.input["airFlowRateIn"].set(torch.tensor([[0.0, 0.0]]), step_index=0)
        junction.input["airTemperatureIn"].set(torch.tensor([[20.0, 22.0]]), step_index=0)
        
        # Execute a time step
        datetime_val = datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)
        junction.do_step(second_time=0, date_time=datetime_val, step_size=600, step_index=0)
        
        # Check outputs - should be 0 flow and default temperature
        flow_out = junction.output["airFlowRateOut"].get()
        temp_out = junction.output["airTemperatureOut"].get()
        
        self.assertEqual(flow_out.item(), 0)
        self.assertEqual(temp_out.item(), 20)  # Default temperature when flow is 0


class TestSupplyFlowJunctionSystemExtended(unittest.TestCase):
    def setUp(self):
        self.junction = SupplyFlowJunctionSystem(id="test_supply_junction", airFlowRateBias=0.05)

    def test_initialization(self):
        """Test supply flow junction initialization."""
        self.assertIsNotNone(self.junction)
        self.assertEqual(self.junction.id, "test_supply_junction")
        self.assertEqual(self.junction.airFlowRateBias, 0.05)

    def test_initialization_default_bias(self):
        """Test supply flow junction with default bias."""
        junction = SupplyFlowJunctionSystem(id="test_default")
        self.assertEqual(junction.airFlowRateBias, 0)

    def test_do_step(self):
        """Test supply flow junction do_step method."""
        start_time = [datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)]
        end_time = [datetime.datetime(2023, 1, 1, 1, 40, 0, tzinfo=pytz.UTC)]
        step_size = [600]
        self.junction.initialize(start_time=start_time, end_time=end_time, step_size=step_size)
        
        # Set inputs - vector inputs for multiple flows
        self.junction.input["airFlowRateOut"].set(torch.tensor([[0.3, 0.4, 0.3]]), step_index=0)
        
        # Execute a time step
        datetime_val = datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)
        self.junction.do_step(second_time=0, date_time=datetime_val, step_size=600, step_index=0)
        
        # Check output - sum of flows + bias
        flow_in = self.junction.output["airFlowRateIn"].get()
        
        self.assertIsNotNone(flow_in)
        # Total = 0.3 + 0.4 + 0.3 + 0.05 = 1.05
        self.assertAlmostEqual(flow_in.item(), 1.05, places=2)


class TestDiscreteStatespaceSystem(unittest.TestCase):
    def setUp(self):
        from twin4build.systems.utils.discrete_statespace_system import DiscreteStatespaceSystem
        # Simple first-order system: dx/dt = -x + u, y = x
        A = torch.tensor([[0.9]], dtype=torch.float64)  # Already discrete
        B = torch.tensor([[0.1]], dtype=torch.float64)
        C = torch.tensor([[1.0]], dtype=torch.float64)
        D = torch.tensor([[0.0]], dtype=torch.float64)
        x0 = torch.tensor([0.0], dtype=torch.float64)
        
        self.system = DiscreteStatespaceSystem(
            id="test_ss",
            A=A, B=B, C=C, D=D,
            x0=x0,
            is_discrete=True
        )

    def test_initialization(self):
        """Test discrete statespace system initialization."""
        self.assertIsNotNone(self.system)
        self.assertEqual(self.system.id, "test_ss")

    def test_state_property(self):
        """Test get_state and set_state methods."""
        state = self.system.get_state()
        self.assertIsNotNone(state)
        
        # Set a new state
        new_state = torch.tensor([[1.5]], dtype=torch.float64)
        self.system.set_state(new_state)
        retrieved_state = self.system.get_state()
        self.assertAlmostEqual(retrieved_state[0, 0].item(), 1.5, places=5)


class TestPiecewiseLinearSystem(unittest.TestCase):
    def setUp(self):
        from twin4build.systems.utils.piecewise_linear_system import PiecewiseLinearSystem
        # Create a simple piecewise linear function: y = x for x in [0, 1], y = 2-x for x in [1, 2]
        X = np.array([0.0, 1.0, 2.0])
        Y = np.array([0.0, 1.0, 0.0])
        self.system = PiecewiseLinearSystem(id="test_piecewise", X=X, Y=Y)

    def test_initialization(self):
        """Test piecewise linear system initialization."""
        self.assertIsNotNone(self.system)
        self.assertEqual(self.system.id, "test_piecewise")

    def test_get_Y_within_range(self):
        """Test interpolation within range."""
        # At x=0.5, y should be 0.5 (on the first segment)
        y = self.system._get_Y(0.5)
        self.assertAlmostEqual(y, 0.5, places=5)
        
        # At x=1.5, y should be 0.5 (on the second segment, y = 2 - x)
        y = self.system._get_Y(1.5)
        self.assertAlmostEqual(y, 0.5, places=5)

    def test_get_Y_at_boundaries(self):
        """Test interpolation at boundary points."""
        # At x=0, y should be 0
        y = self.system._get_Y(0)
        self.assertAlmostEqual(y, 0.0, places=5)
        
        # At x=1, y should be 1
        y = self.system._get_Y(1.0)
        self.assertAlmostEqual(y, 1.0, places=5)
        
        # At x=2, y should be 0
        y = self.system._get_Y(2.0)
        self.assertAlmostEqual(y, 0.0, places=5)

    def test_get_Y_outside_range(self):
        """Test interpolation outside range (extrapolation/clamping)."""
        # Below range, should return first Y value
        y = self.system._get_Y(-1.0)
        self.assertAlmostEqual(y, 0.0, places=5)
        
        # Above range, should return last Y value
        y = self.system._get_Y(3.0)
        self.assertAlmostEqual(y, 0.0, places=5)


class TestOnOffControllerSystemExtended(unittest.TestCase):
    def test_reverse_mode(self):
        """Test on/off controller in reverse mode."""
        controller = OnOffControllerSystem(id="test_reverse", offValue=0, onValue=1, isReverse=True)
        
        start_time = [datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)]
        end_time = [datetime.datetime(2023, 1, 1, 1, 40, 0, tzinfo=pytz.UTC)]
        step_size = [600]
        controller.initialize(start_time=start_time, end_time=end_time, step_size=step_size)
        
        # In reverse mode, when actual > setpoint, output should be ON
        controller.input["actualValue"].set(torch.tensor([25.0]), step_index=0)
        controller.input["setpointValue"].set(torch.tensor([22.0]), step_index=0)
        
        datetime_val = datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)
        controller.do_step(second_time=0, date_time=datetime_val, step_size=600, step_index=0)
        
        output = controller.output["inputSignal"].get()
        self.assertIsNotNone(output)


if __name__ == '__main__':
    unittest.main()

