# Standard library imports
import datetime
import unittest

# Third party imports
import numpy as np
import pandas as pd
import pytz
import torch

# Local application imports
from twin4build.systems.air_to_air_heat_recovery.air_to_air_heat_recovery_system import (
    AirToAirHeatRecoverySystem,
)
from twin4build.systems.building_space.building_space_torch_system import (
    BuildingSpaceTorchSystem,
)
from twin4build.systems.coil.coil_torch_system import CoilTorchSystem
from twin4build.systems.controller.rulebased_controller.on_off_controller.on_off_controller_system import (
    OnOffControllerSystem,
)
from twin4build.systems.controller.setpoint_controller.pid_controller.pid_controller_system import (
    PIDControllerSystem,
)
from twin4build.systems.damper.damper_torch_system import DamperTorchSystem
from twin4build.systems.fan.fan_torch_system import FanTorchSystem
from twin4build.systems.junction.return_flow_junction_system import (
    ReturnFlowJunctionSystem,
)
from twin4build.systems.junction.supply_flow_junction_system import (
    SupplyFlowJunctionSystem,
)
from twin4build.systems.outdoor_environment.outdoor_environment_system import (
    OutdoorEnvironmentSystem,
)
from twin4build.systems.schedule.piecewise_linear_schedule_system import (
    PiecewiseLinearScheduleSystem,
)
from twin4build.systems.schedule.schedule_system import ScheduleSystem
from twin4build.systems.sensor.sensor_system import SensorSystem
from twin4build.systems.space_heater.space_heater_torch_system import (
    SpaceHeaterTorchSystem,
)
from twin4build.systems.utils.discrete_statespace_system import DiscreteStatespaceSystem
from twin4build.systems.utils.max_system import MaxSystem
from twin4build.systems.utils.on_off_system import OnOffSystem
from twin4build.systems.utils.pass_input_to_output import PassInputToOutput
from twin4build.systems.utils.piecewise_linear_system import PiecewiseLinearSystem
from twin4build.systems.utils.time_series_input_system import TimeSeriesInputSystem
from twin4build.systems.valve.valve_torch_system import ValveTorchSystem


class TestSpaceHeaterTorchSystem(unittest.TestCase):
    def setUp(self):
        self.heater = SpaceHeaterTorchSystem(
            id="test_heater",
            Q_flow_nominal_sh=1000.0,
            T_a_nominal_sh=60.0,
            T_b_nominal_sh=40.0,
            TAir_nominal_sh=20.0,
            thermalMassHeatCapacity=5000.0,
            nelements=2,
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
        self.heater.initialize(
            start_time=start_time, end_time=end_time, step_size=step_size
        )

        # Set inputs
        self.heater.input["supplyWaterTemperature"].set(
            torch.tensor([60.0]), step_index=0
        )
        self.heater.input["waterFlowRate"].set(torch.tensor([0.1]), step_index=0)
        self.heater.input["indoorTemperature"].set(torch.tensor([20.0]), step_index=0)

        # Execute a time step
        datetime_val = datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)
        self.heater.do_step(
            second_time=0, date_time=datetime_val, step_size=step_size, step_index=0
        )

        # Check outputs
        outlet_temp = self.heater.output["outletWaterTemperature"].get()
        radiator_power = self.heater.output["Power"].get()

        self.assertIsNotNone(outlet_temp)
        self.assertIsNotNone(radiator_power)

    def test_do_step_batch(self):
        """Test space heater system do_step method with batch size > 1."""
        heater_batch = SpaceHeaterTorchSystem(
            id="test_heater_batch",
            Q_flow_nominal_sh=1000.0,
            T_a_nominal_sh=60.0,
            T_b_nominal_sh=40.0,
            TAir_nominal_sh=20.0,
            thermalMassHeatCapacity=5000.0,
            nelements=2,
        )

        batch_size = 3

        # Batch size is determined by the length of start_time/end_time/step_size lists
        start_time = [
            datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)
        ] * batch_size
        end_time = [
            datetime.datetime(2023, 1, 1, 1, 40, 0, tzinfo=pytz.UTC)
        ] * batch_size
        step_size = [600] * batch_size
        heater_batch.initialize(
            start_time=start_time, end_time=end_time, step_size=step_size
        )

        # Set inputs with batch size 3 (inputs already initialized with correct batch_size by initialize())
        heater_batch.input["supplyWaterTemperature"].set(
            torch.tensor([60.0, 65.0, 55.0]), step_index=0
        )
        heater_batch.input["waterFlowRate"].set(
            torch.tensor([0.1, 0.15, 0.08]), step_index=0
        )
        heater_batch.input["indoorTemperature"].set(
            torch.tensor([20.0, 22.0, 18.0]), step_index=0
        )

        # Execute a time step
        datetime_val = datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)
        heater_batch.do_step(
            second_time=0, date_time=datetime_val, step_size=step_size, step_index=0
        )

        # Check outputs - verify all outputs have consistent batch shape
        outlet_temp = heater_batch.output["outletWaterTemperature"].get()
        radiator_power = heater_batch.output["Power"].get()

        self.assertIsNotNone(outlet_temp)
        self.assertIsNotNone(radiator_power)
        self.assertEqual(
            outlet_temp.shape[0], batch_size
        )  # Output batch matches input batch
        self.assertEqual(
            radiator_power.shape[0], batch_size
        )  # Output batch matches input batch


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
        self.coil.initialize(
            start_time=start_time, end_time=end_time, step_size=step_size
        )

        # Set inputs
        self.coil.input["inletAirTemperature"].set(torch.tensor([20.0]), step_index=0)
        self.coil.input["outletAirTemperatureSetpoint"].set(
            torch.tensor([22.0]), step_index=0
        )
        self.coil.input["airFlowRate"].set(torch.tensor([1.0]), step_index=0)

        # Execute a time step
        datetime_val = datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)
        self.coil.do_step(
            second_time=0, date_time=datetime_val, step_size=600, step_index=0
        )

        # Check outputs
        heating_power = self.coil.output["heatingPower"].get()
        cooling_power = self.coil.output["coolingPower"].get()

        self.assertIsNotNone(heating_power)
        self.assertIsNotNone(cooling_power)
        # Should be heating
        self.assertGreater(heating_power.item(), 0)
        self.assertEqual(cooling_power.item(), 0)

    def test_do_step_batch(self):
        """Test coil system do_step method with batch size > 1."""
        coil_batch = CoilTorchSystem(id="test_coil_batch")

        batch_size = 2

        start_time = [datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)] * 2
        end_time = [datetime.datetime(2023, 1, 1, 1, 40, 0, tzinfo=pytz.UTC)] * 2
        step_size = [600] * 2
        coil_batch.initialize(
            start_time=start_time, end_time=end_time, step_size=step_size
        )

        # Initialize inputs with batch size
        # coil_batch.input["inletAirTemperature"].initialize(n_timesteps=1, batch_size=batch_size, size=1)
        # coil_batch.input["outletAirTemperatureSetpoint"].initialize(n_timesteps=1, batch_size=batch_size, size=1)
        # coil_batch.input["airFlowRate"].initialize(n_timesteps=1, batch_size=batch_size, size=1)

        # Set inputs with batch size 2
        coil_batch.input["inletAirTemperature"].set(
            torch.tensor([20.0, 25.0]), step_index=0
        )
        coil_batch.input["outletAirTemperatureSetpoint"].set(
            torch.tensor([22.0, 23.0]), step_index=0
        )
        coil_batch.input["airFlowRate"].set(torch.tensor([1.0, 1.2]), step_index=0)

        # Execute a time step
        datetime_val = datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)
        coil_batch.do_step(
            second_time=0, date_time=datetime_val, step_size=step_size, step_index=0
        )

        # Check outputs - verify all outputs have consistent batch shape
        heating_power = coil_batch.output["heatingPower"].get()
        cooling_power = coil_batch.output["coolingPower"].get()

        self.assertIsNotNone(heating_power)
        self.assertIsNotNone(cooling_power)
        self.assertEqual(
            heating_power.shape[0], batch_size
        )  # Output batch matches input batch
        self.assertEqual(
            cooling_power.shape[0], batch_size
        )  # Output batch matches input batch


class TestAirToAirHeatRecoverySystem(unittest.TestCase):
    def setUp(self):
        self.hr = AirToAirHeatRecoverySystem(
            id="test_hr",
            eps_75_h=0.8,
            eps_100_h=0.7,
            eps_75_c=0.8,
            eps_100_c=0.7,
            primaryAirFlowRateMax=1.0,
            secondaryAirFlowRateMax=1.0,
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
        self.hr.initialize(
            start_time=start_time, end_time=end_time, step_size=step_size
        )

        # Set inputs
        # Supply side (outdoor to indoor)
        self.hr.input["primaryTemperatureIn"].set(
            torch.tensor([0.0]), step_index=0
        )  # Cold outdoor
        self.hr.input["primaryAirFlowRate"].set(torch.tensor([1.0]), step_index=0)
        self.hr.input["primaryTemperatureOutSetpoint"].set(
            torch.tensor([20.0]), step_index=0
        )

        # Exhaust side (indoor to outdoor)
        self.hr.input["secondaryTemperatureIn"].set(
            torch.tensor([20.0]), step_index=0
        )  # Warm return
        self.hr.input["secondaryAirFlowRate"].set(torch.tensor([1.0]), step_index=0)

        # Execute a time step
        datetime_val = datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)
        self.hr.do_step(
            second_time=0, date_time=datetime_val, step_size=600, step_index=0
        )

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

    def test_do_step_batch(self):
        """Test heat recovery system do_step method with batch size > 1."""
        hr_batch = AirToAirHeatRecoverySystem(
            id="test_hr_batch",
            eps_75_h=0.8,
            eps_100_h=0.7,
            eps_75_c=0.8,
            eps_100_c=0.7,
            primaryAirFlowRateMax=1.0,
            secondaryAirFlowRateMax=1.0,
        )

        batch_size = 2

        start_time = [datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)] * 2
        end_time = [datetime.datetime(2023, 1, 1, 1, 40, 0, tzinfo=pytz.UTC)] * 2
        step_size = [600] * 2
        hr_batch.initialize(
            start_time=start_time, end_time=end_time, step_size=step_size
        )

        # Initialize inputs with batch size
        # hr_batch.input["primaryTemperatureIn"].initialize(n_timesteps=1, batch_size=batch_size, size=1)
        # hr_batch.input["primaryAirFlowRate"].initialize(n_timesteps=1, batch_size=batch_size, size=1)
        # hr_batch.input["primaryTemperatureOutSetpoint"].initialize(n_timesteps=1, batch_size=batch_size, size=1)
        # hr_batch.input["secondaryTemperatureIn"].initialize(n_timesteps=1, batch_size=batch_size, size=1)
        # hr_batch.input["secondaryAirFlowRate"].initialize(n_timesteps=1, batch_size=batch_size, size=1)

        # Set inputs with batch size 2
        hr_batch.input["primaryTemperatureIn"].set(
            torch.tensor([0.0, -5.0]), step_index=0
        )
        hr_batch.input["primaryAirFlowRate"].set(torch.tensor([1.0, 0.8]), step_index=0)
        hr_batch.input["primaryTemperatureOutSetpoint"].set(
            torch.tensor([20.0, 20.0]), step_index=0
        )
        hr_batch.input["secondaryTemperatureIn"].set(
            torch.tensor([20.0, 22.0]), step_index=0
        )
        hr_batch.input["secondaryAirFlowRate"].set(
            torch.tensor([1.0, 0.8]), step_index=0
        )

        # Execute a time step
        datetime_val = datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)
        hr_batch.do_step(
            second_time=0, date_time=datetime_val, step_size=step_size, step_index=0
        )

        # Check outputs - verify all outputs have consistent batch shape
        primary_out = hr_batch.output["primaryTemperatureOut"].get()
        secondary_out = hr_batch.output["secondaryTemperatureOut"].get()

        self.assertIsNotNone(primary_out)
        self.assertIsNotNone(secondary_out)
        self.assertEqual(
            primary_out.shape[0], batch_size
        )  # Output batch matches input batch
        self.assertEqual(
            secondary_out.shape[0], batch_size
        )  # Output batch matches input batch


class TestScheduleSystem(unittest.TestCase):
    def setUp(self):
        self.schedule = ScheduleSystem(
            weekDayRulesetDict={
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
        start_time = [datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)]
        end_time = [datetime.datetime(2023, 1, 1, 1, 40, 0, tzinfo=pytz.UTC)]
        step_size = [600]
        self.schedule.initialize(
            start_time=start_time, end_time=end_time, step_size=step_size
        )

        # Execute a time step
        datetime_val = datetime.datetime(2023, 1, 1, 0, 30, 0, tzinfo=pytz.UTC)
        self.schedule.do_step(
            second_time=1800, date_time=datetime_val, step_size=600, step_index=3
        )

        # Check that output was set
        self.assertIsNotNone(self.schedule.output["scheduleValue"].get())

    def test_do_step_batch(self):
        """Test schedule system do_step method with batch size > 1."""
        schedule_batch = ScheduleSystem(
            weekDayRulesetDict={
                "ruleset_start_minute": [0],
                "ruleset_end_minute": [0],
                "ruleset_start_hour": [0],
                "ruleset_end_hour": [1],
                "ruleset_value": [20],
                "ruleset_default_value": 0,
            },
            id="test_schedule_batch",
        )

        start_time = [datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)]
        end_time = [datetime.datetime(2023, 1, 1, 1, 40, 0, tzinfo=pytz.UTC)]
        step_size = [600]
        schedule_batch.initialize(
            start_time=start_time, end_time=end_time, step_size=step_size
        )

        # Execute a time step
        datetime_val = datetime.datetime(2023, 1, 1, 0, 30, 0, tzinfo=pytz.UTC)
        schedule_batch.do_step(
            second_time=1800, date_time=datetime_val, step_size=600, step_index=3
        )

        # Check that output was set
        output = schedule_batch.output["scheduleValue"].get()
        self.assertIsNotNone(output)


class TestDamperTorchSystem(unittest.TestCase):
    def setUp(self):
        self.damper = DamperTorchSystem(id="test_damper", a=1.0, nominalAirFlowRate=0.5)

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
        self.damper.initialize(
            start_time=start_time, end_time=end_time, step_size=step_size
        )

        # Set input
        self.damper.input["damperPosition"].set(torch.tensor([0.5]), step_index=0)

        # Execute a time step
        datetime_val = datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)
        self.damper.do_step(
            second_time=0, date_time=datetime_val, step_size=600, step_index=0
        )

        # Check that output was calculated
        airflow = self.damper.output["airFlowRate"].get()
        self.assertIsNotNone(airflow)
        self.assertGreater(airflow.item(), 0)

    def test_airflow_calculation(self):
        """Test that damper calculates airflow correctly."""
        start_time = [datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)]
        end_time = [datetime.datetime(2023, 1, 1, 1, 40, 0, tzinfo=pytz.UTC)]
        step_size = [600]
        self.damper.initialize(
            start_time=start_time, end_time=end_time, step_size=step_size
        )

        # At 100% position, should give nominal airflow
        self.damper.input["damperPosition"].set(torch.tensor([1.0]), step_index=0)
        self.damper.do_step(
            second_time=0,
            date_time=datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC),
            step_size=600,
            step_index=0,
        )

        airflow = self.damper.output["airFlowRate"].get().item()
        self.assertAlmostEqual(airflow, 0.5, places=2)

    def test_do_step_batch(self):
        """Test damper system do_step method with batch size > 1."""
        damper_batch = DamperTorchSystem(
            id="test_damper_batch", a=1.0, nominalAirFlowRate=0.5
        )

        batch_size = 3

        start_time = [datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)]
        end_time = [datetime.datetime(2023, 1, 1, 1, 40, 0, tzinfo=pytz.UTC)]
        step_size = [600]
        damper_batch.initialize(
            start_time=start_time, end_time=end_time, step_size=step_size
        )
        damper_batch.input["damperPosition"].initialize(
            n_timesteps=1, batch_size=batch_size, size=1
        )
        damper_batch.output["damperPosition"].initialize(
            n_timesteps=1, batch_size=batch_size, size=1
        )
        damper_batch.output["airFlowRate"].initialize(
            n_timesteps=1, batch_size=batch_size, size=1
        )

        # Set input with batch size 3
        damper_batch.input["damperPosition"].set(
            torch.tensor([0.5, 0.7, 0.3]), step_index=0
        )

        # Execute a time step
        datetime_val = datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)
        damper_batch.do_step(
            second_time=0, date_time=datetime_val, step_size=600, step_index=0
        )

        # Check that output was calculated - verify batch shape consistency
        airflow = damper_batch.output["airFlowRate"].get()
        self.assertIsNotNone(airflow)
        self.assertEqual(
            airflow.shape[0], batch_size
        )  # Output batch matches input batch


class TestFanTorchSystem(unittest.TestCase):
    def setUp(self):
        self.fan = FanTorchSystem(
            id="test_fan",
            nominalPowerRate=1000.0,
            nominalAirFlowRate=1.0,
            c1=0,
            c2=0.8,
            c3=0.2,
            c4=0.0,
            f_total=0.9,
        )

    def test_initialization(self):
        """Test fan system initialization."""
        start_time = [datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)]
        end_time = [datetime.datetime(2023, 1, 1, 1, 40, 0, tzinfo=pytz.UTC)]
        step_size = [600]
        self.fan.initialize(
            start_time=start_time, end_time=end_time, step_size=step_size
        )
        self.assertIsNotNone(self.fan)
        self.assertEqual(self.fan.id, "test_fan")

    def test_do_step(self):
        """Test fan system do_step method."""
        start_time = [datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)]
        end_time = [datetime.datetime(2023, 1, 1, 1, 40, 0, tzinfo=pytz.UTC)]
        step_size = [600]
        self.fan.initialize(
            start_time=start_time, end_time=end_time, step_size=step_size
        )

        # Set inputs
        self.fan.input["airFlowRate"].set(torch.tensor([1.0]), step_index=0)
        self.fan.input["inletAirTemperature"].set(torch.tensor([20.0]), step_index=0)

        # Set required inputs (may vary based on implementation)
        # Execute a time step
        datetime_val = datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)
        self.fan.do_step(
            second_time=0, date_time=datetime_val, step_size=600, step_index=0
        )

        self.assertIsNotNone(self.fan.output["outletAirTemperature"].get())
        self.assertIsNotNone(self.fan.output["Power"].get())
        self.assertGreater(self.fan.output["outletAirTemperature"].get().item(), 20.0)
        self.assertGreater(self.fan.output["Power"].get().item(), 0.0)

    def test_do_step_batch(self):
        """Test fan system do_step method with batch size > 1."""
        fan_batch = FanTorchSystem(
            id="test_fan_batch",
            nominalPowerRate=1000.0,
            nominalAirFlowRate=1.0,
            c1=0,
            c2=0.8,
            c3=0.2,
            c4=0.0,
            f_total=0.9,
        )

        batch_size = 2

        start_time = [datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)]
        end_time = [datetime.datetime(2023, 1, 1, 1, 40, 0, tzinfo=pytz.UTC)]
        step_size = [600]
        fan_batch.initialize(
            start_time=start_time, end_time=end_time, step_size=step_size
        )

        # Initialize inputs with batch size
        fan_batch.input["airFlowRate"].initialize(
            n_timesteps=1, batch_size=batch_size, size=1
        )
        fan_batch.input["inletAirTemperature"].initialize(
            n_timesteps=1, batch_size=batch_size, size=1
        )

        # Set inputs with batch size 2
        fan_batch.input["airFlowRate"].set(torch.tensor([1.0, 0.8]), step_index=0)
        fan_batch.input["inletAirTemperature"].set(
            torch.tensor([20.0, 22.0]), step_index=0
        )

        fan_batch.output["outletAirTemperature"].initialize(
            n_timesteps=1, batch_size=batch_size, size=1
        )
        fan_batch.output["Power"].initialize(
            n_timesteps=1, batch_size=batch_size, size=1
        )

        # Execute a time step
        datetime_val = datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)
        fan_batch.do_step(
            second_time=0, date_time=datetime_val, step_size=600, step_index=0
        )

        # Check outputs - verify all outputs have consistent batch shape
        outlet_temp = fan_batch.output["outletAirTemperature"].get()
        power = fan_batch.output["Power"].get()

        self.assertIsNotNone(outlet_temp)
        self.assertIsNotNone(power)
        self.assertEqual(
            outlet_temp.shape[0], batch_size
        )  # Output batch matches input batch
        self.assertEqual(power.shape[0], batch_size)  # Output batch matches input batch


class TestValveTorchSystem(unittest.TestCase):
    def setUp(self):
        self.valve = ValveTorchSystem(
            id="test_valve", valveAuthority=0.5, waterFlowRateMax=1.0
        )

    def test_initialization(self):
        """Test valve system initialization."""
        self.assertIsNotNone(self.valve)
        self.assertEqual(self.valve.id, "test_valve")

    def test_do_step(self):
        """Test valve system do_step method."""
        start_time = [datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)]
        end_time = [datetime.datetime(2023, 1, 1, 1, 40, 0, tzinfo=pytz.UTC)]
        step_size = [600]
        self.valve.initialize(
            start_time=start_time, end_time=end_time, step_size=step_size
        )

        # Set inputs
        self.valve.input["valvePosition"].set(torch.tensor([0.5]), step_index=0)

        # Execute a time step
        datetime_val = datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)
        self.valve.do_step(
            second_time=0, date_time=datetime_val, step_size=600, step_index=0
        )

        output = self.valve.output["waterFlowRate"].get()
        self.assertIsNotNone(output)
        self.assertGreater(output.item(), 0)

    def test_do_step_batch(self):
        """Test valve system do_step method with batch size > 1."""
        valve_batch = ValveTorchSystem(
            id="test_valve_batch", valveAuthority=0.5, waterFlowRateMax=1.0
        )

        batch_size = 3

        start_time = [datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)]
        end_time = [datetime.datetime(2023, 1, 1, 1, 40, 0, tzinfo=pytz.UTC)]
        step_size = [600]
        valve_batch.initialize(
            start_time=start_time, end_time=end_time, step_size=step_size
        )
        valve_batch.input["valvePosition"].initialize(
            n_timesteps=1, batch_size=batch_size, size=1
        )
        valve_batch.output["waterFlowRate"].initialize(
            n_timesteps=1, batch_size=batch_size, size=1
        )
        valve_batch.output["valvePosition"].initialize(
            n_timesteps=1, batch_size=batch_size, size=1
        )

        # Set inputs with batch size 3
        valve_batch.input["valvePosition"].set(
            torch.tensor([0.5, 0.7, 0.3]), step_index=0
        )

        # Execute a time step
        datetime_val = datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)
        valve_batch.do_step(
            second_time=0, date_time=datetime_val, step_size=600, step_index=0
        )

        # Check output - verify batch shape consistency
        output = valve_batch.output["waterFlowRate"].get()
        self.assertIsNotNone(output)
        self.assertEqual(
            output.shape[0], batch_size
        )  # Output batch matches input batch


class TestOnOffControllerSystem(unittest.TestCase):
    def setUp(self):
        self.controller = OnOffControllerSystem(
            id="test_controller", offValue=0, onValue=1, isReverse=False
        )

    def test_initialization(self):
        """Test on/off controller initialization."""
        self.assertIsNotNone(self.controller)
        self.assertEqual(self.controller.id, "test_controller")

    def test_do_step(self):
        """Test on/off controller do_step method."""
        start_time = [datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)]
        end_time = [datetime.datetime(2023, 1, 1, 1, 40, 0, tzinfo=pytz.UTC)]
        step_size = [600]
        self.controller.initialize(
            start_time=start_time, end_time=end_time, step_size=step_size
        )

        # Set inputs
        self.controller.input["actualValue"].set(torch.tensor([20.0]), step_index=0)
        self.controller.input["setpointValue"].set(torch.tensor([22.0]), step_index=0)

        # Execute a time step
        datetime_val = datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)
        self.controller.do_step(
            second_time=0, date_time=datetime_val, step_size=600, step_index=0
        )

        # Check output
        output = self.controller.output["inputSignal"].get()
        self.assertIsNotNone(output)

    def test_do_step_batch(self):
        """Test on/off controller do_step method with batch size > 1."""
        controller_batch = OnOffControllerSystem(
            id="test_controller_batch", offValue=0, onValue=1, isReverse=False
        )

        batch_size = 2

        start_time = [
            datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)
        ] * batch_size
        end_time = [
            datetime.datetime(2023, 1, 1, 1, 40, 0, tzinfo=pytz.UTC)
        ] * batch_size
        step_size = [600] * batch_size
        controller_batch.initialize(
            start_time=start_time, end_time=end_time, step_size=step_size
        )

        # Set inputs with batch size 2
        controller_batch.input["actualValue"].set(
            torch.tensor([20.0, 23.0]), step_index=0
        )
        controller_batch.input["setpointValue"].set(
            torch.tensor([22.0, 22.0]), step_index=0
        )

        # Execute a time step
        datetime_val = datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)
        controller_batch.do_step(
            second_time=0, date_time=datetime_val, step_size=600, step_index=0
        )

        # Check output - verify batch shape consistency
        output = controller_batch.output["inputSignal"].get()
        self.assertIsNotNone(output)
        self.assertEqual(
            output.shape[0], batch_size
        )  # Output batch matches input batch


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
        self.controller.initialize(
            start_time=start_time, end_time=end_time, step_size=step_size
        )

        # Set inputs
        self.controller.input["actualValue"].set(torch.tensor([20.0]), step_index=0)
        self.controller.input["setpointValue"].set(torch.tensor([22.0]), step_index=0)

        # Execute a time step
        datetime_val = datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)
        self.controller.do_step(
            second_time=0, date_time=datetime_val, step_size=600, step_index=0
        )

        # Check output
        output = self.controller.output["inputSignal"].get()
        self.assertIsNotNone(output)

    def test_do_step_batch(self):
        """Test PID controller do_step method with batch size > 1."""
        controller_batch = PIDControllerSystem(
            id="test_pid_batch", Kp=1.0, Ki=0.1, Kd=0.01
        )

        batch_size = 3

        start_time = [
            datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)
        ] * batch_size
        end_time = [
            datetime.datetime(2023, 1, 1, 1, 40, 0, tzinfo=pytz.UTC)
        ] * batch_size
        step_size = [600] * batch_size
        controller_batch.initialize(
            start_time=start_time, end_time=end_time, step_size=step_size
        )

        # Set inputs with batch size 3
        controller_batch.input["actualValue"].set(
            torch.tensor([20.0, 21.0, 19.0]), step_index=0
        )
        controller_batch.input["setpointValue"].set(
            torch.tensor([22.0, 22.0, 22.0]), step_index=0
        )

        # Execute a time step
        datetime_val = datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)
        controller_batch.do_step(
            second_time=0, date_time=datetime_val, step_size=600, step_index=0
        )

        # Check output - verify batch shape consistency
        output = controller_batch.output["inputSignal"].get()
        self.assertIsNotNone(output)
        self.assertEqual(
            output.shape[0], batch_size
        )  # Output batch matches input batch


class TestSensorSystem(unittest.TestCase):
    def setUp(self):
        dates = pd.date_range(
            start=datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC),
            periods=10,
            freq="10min",
        )
        df = pd.DataFrame({"value": [20.0] * 10}, index=dates)
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
        self.sensor.initialize(
            start_time=start_time, end_time=end_time, step_size=step_size
        )

        # Execute a time step
        datetime_val = datetime.datetime(2023, 1, 1, 0, 10, 0, tzinfo=pytz.UTC)
        self.sensor.do_step(
            second_time=600, date_time=datetime_val, step_size=600, step_index=1
        )

        # Check that measured value was set
        measured = self.sensor.output["measuredValue"].get()
        self.assertIsNotNone(measured)
        self.assertAlmostEqual(measured.item(), 20.0, places=1)

    def test_do_step_batch(self):
        """Test sensor system do_step method with batch size > 1."""
        dates = pd.date_range(
            start=datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC),
            periods=10,
            freq="10min",
        )
        df = pd.DataFrame({"value": [20.0] * 10}, index=dates)
        sensor_batch = SensorSystem(id="test_sensor_batch", df=df)

        start_time = [datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)]
        end_time = [datetime.datetime(2023, 1, 1, 1, 40, 0, tzinfo=pytz.UTC)]
        step_size = [600]
        sensor_batch.initialize(
            start_time=start_time, end_time=end_time, step_size=step_size
        )

        # Execute a time step
        datetime_val = datetime.datetime(2023, 1, 1, 0, 10, 0, tzinfo=pytz.UTC)
        sensor_batch.do_step(
            second_time=600, date_time=datetime_val, step_size=600, step_index=1
        )

        # Check that measured value was set
        measured = sensor_batch.output["measuredValue"].get()
        self.assertIsNotNone(measured)


class TestOutdoorEnvironmentSystem(unittest.TestCase):
    def setUp(self):
        self.outdoor_env = OutdoorEnvironmentSystem(id="test_outdoor")

    def test_initialization(self):
        """Test outdoor environment system initialization."""
        self.assertIsNotNone(self.outdoor_env)
        self.assertEqual(self.outdoor_env.id, "test_outdoor")

    def test_initialization_with_df(self):
        """Test outdoor environment system with DataFrame."""
        dates = pd.date_range(
            start=datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC),
            periods=10,
            freq="10min",
        )
        df = pd.DataFrame(
            {
                "outdoorTemperature": [5.0] * 10,
                "globalIrradiation": [100.0] * 10,
                "outdoorCo2Concentration": [400.0] * 10,
            },
            index=dates,
        )

        outdoor_env = OutdoorEnvironmentSystem(id="test_outdoor_df", df=df)
        self.assertIsNotNone(outdoor_env)
        self.assertIsNotNone(outdoor_env.df)

    def test_initialization_with_correction(self):
        """Test outdoor environment system with linear correction."""
        outdoor_env = OutdoorEnvironmentSystem(
            id="test_outdoor_correction", a=1.1, b=0.5, apply_correction=True
        )
        self.assertEqual(outdoor_env.a.get().item(), 1.1)
        self.assertEqual(outdoor_env.b.get().item(), 0.5)
        self.assertTrue(outdoor_env.apply_correction)

    def test_input_output_properties(self):
        """Test input and output property accessors."""
        self.assertIsInstance(self.outdoor_env.input, dict)
        self.assertIsInstance(self.outdoor_env.output, dict)

        # Check outputs exist
        self.assertIn("outdoorTemperature", self.outdoor_env.output)
        self.assertIn("globalIrradiation", self.outdoor_env.output)
        self.assertIn("outdoorCo2Concentration", self.outdoor_env.output)

    def test_config_property(self):
        """Test config property accessor."""
        config = self.outdoor_env.config
        self.assertIsInstance(config, dict)
        self.assertIn("parameters", config)
        self.assertIn("spreadsheet", config)
        self.assertIn("database", config)

    def test_initialize_with_df(self):
        """Test outdoor environment system initialization with DataFrame."""
        dates = pd.date_range(
            start=datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC),
            periods=10,
            freq="10min",
        )
        df = pd.DataFrame(
            {
                "outdoorTemperature": [5.0] * 10,
                "globalIrradiation": [100.0] * 10,
                "outdoorCo2Concentration": [400.0] * 10,
            },
            index=dates,
        )

        outdoor_env = OutdoorEnvironmentSystem(id="test_outdoor_init", df=df)

        start_time = [datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)]
        end_time = [datetime.datetime(2023, 1, 1, 1, 0, 0, tzinfo=pytz.UTC)]
        step_size = [600]

        outdoor_env.initialize(
            start_time=start_time, end_time=end_time, step_size=step_size
        )

        # Check outputs are initialized
        self.assertIsNotNone(outdoor_env.output["outdoorTemperature"].history)

    def test_do_step_without_correction(self):
        """Test outdoor environment system do_step method without correction."""
        dates = pd.date_range(
            start=datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC),
            periods=10,
            freq="10min",
        )
        df = pd.DataFrame(
            {
                "outdoorTemperature": [5.0] * 10,
                "globalIrradiation": [100.0] * 10,
                "outdoorCo2Concentration": [400.0] * 10,
            },
            index=dates,
        )

        outdoor_env = OutdoorEnvironmentSystem(id="test_outdoor_step", df=df)

        start_time = [datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)]
        end_time = [datetime.datetime(2023, 1, 1, 1, 0, 0, tzinfo=pytz.UTC)]
        step_size = [600]

        outdoor_env.initialize(
            start_time=start_time, end_time=end_time, step_size=step_size
        )

        # Execute a time step
        datetime_val = datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)
        outdoor_env.do_step(
            second_time=0, date_time=datetime_val, step_size=600, step_index=0
        )

        # Check that outputs were set
        temp = outdoor_env.output["outdoorTemperature"].get()
        irrad = outdoor_env.output["globalIrradiation"].get()
        co2 = outdoor_env.output["outdoorCo2Concentration"].get()

        self.assertIsNotNone(temp)
        self.assertIsNotNone(irrad)
        self.assertIsNotNone(co2)

    def test_do_step_with_correction(self):
        """Test outdoor environment system do_step with linear correction applied."""
        dates = pd.date_range(
            start=datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC),
            periods=10,
            freq="10min",
        )
        df = pd.DataFrame(
            {
                "outdoorTemperature": [10.0] * 10,
                "globalIrradiation": [100.0] * 10,
                "outdoorCo2Concentration": [400.0] * 10,
            },
            index=dates,
        )

        # Create with correction: y = 1.1 * x + 0.5
        outdoor_env = OutdoorEnvironmentSystem(
            id="test_outdoor_correction_step",
            df=df,
            a=1.1,
            b=0.5,
            apply_correction=True,
        )

        start_time = [datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)]
        end_time = [datetime.datetime(2023, 1, 1, 1, 0, 0, tzinfo=pytz.UTC)]
        step_size = [600]

        outdoor_env.initialize(
            start_time=start_time, end_time=end_time, step_size=step_size
        )

        # Execute a time step
        datetime_val = datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)
        outdoor_env.do_step(
            second_time=0, date_time=datetime_val, step_size=600, step_index=0
        )

        # Check that correction was applied to temperature
        temp = outdoor_env.output["outdoorTemperature"].get()
        # Expected: 1.1 * 10.0 + 0.5 = 11.5
        self.assertAlmostEqual(temp.item(), 11.5, places=1)

    def test_initialize_without_data_raises_error(self):
        """Test that initialize raises error when no data source is provided."""
        outdoor_env = OutdoorEnvironmentSystem(id="test_no_data")

        start_time = [datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)]
        end_time = [datetime.datetime(2023, 1, 1, 1, 0, 0, tzinfo=pytz.UTC)]
        step_size = [600]

        with self.assertRaises(ValueError):
            outdoor_env.initialize(
                start_time=start_time, end_time=end_time, step_size=step_size
            )

    def test_apply_method(self):
        """Test the _apply method for linear correction."""
        outdoor_env = OutdoorEnvironmentSystem(
            id="test_apply", a=2.0, b=1.0, apply_correction=True
        )

        # Test _apply method directly
        result = outdoor_env._apply(torch.tensor(5.0))
        self.assertAlmostEqual(result.item(), 11.0, places=5)  # 2.0 * 5.0 + 1.0 = 11.0


class TestTimeSeriesInputSystem(unittest.TestCase):
    def setUp(self):
        dates = pd.date_range(
            start=datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC),
            periods=10,
            freq="10min",
        )
        df = pd.DataFrame({"value": [0.5] * 10}, index=dates)
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
        self.ts_input.initialize(
            start_time=start_time, end_time=end_time, step_size=step_size
        )

        # Execute a time step
        datetime_val = datetime.datetime(2023, 1, 1, 0, 10, 0, tzinfo=pytz.UTC)
        self.ts_input.do_step(
            second_time=600, date_time=datetime_val, step_size=600, step_index=1
        )

        # Check that value was set
        value = self.ts_input.output["value"].get()
        self.assertIsNotNone(value)
        self.assertAlmostEqual(value.item(), 0.5, places=5)

    def test_do_step_batch(self):
        """Test time series input system do_step method with batch size > 1."""
        dates = pd.date_range(
            start=datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC),
            periods=10,
            freq="10min",
        )
        df = pd.DataFrame({"value": [0.5] * 10}, index=dates)
        ts_input_batch = TimeSeriesInputSystem(id="test_timeseries_batch", df=df)

        start_time = [datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)]
        end_time = [datetime.datetime(2023, 1, 1, 1, 40, 0, tzinfo=pytz.UTC)]
        step_size = [600]
        ts_input_batch.initialize(
            start_time=start_time, end_time=end_time, step_size=step_size
        )

        # Execute a time step
        datetime_val = datetime.datetime(2023, 1, 1, 0, 10, 0, tzinfo=pytz.UTC)
        ts_input_batch.do_step(
            second_time=600, date_time=datetime_val, step_size=600, step_index=1
        )

        # Check that value was set
        value = ts_input_batch.output["value"].get()
        self.assertIsNotNone(value)


class TestReturnFlowJunctionSystem(unittest.TestCase):
    def setUp(self):
        self.junction = ReturnFlowJunctionSystem(
            id="test_return_junction", airFlowRateBias=0.1
        )

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
        self.junction.initialize(
            start_time=start_time, end_time=end_time, step_size=step_size
        )
        self.junction.input["airFlowRateIn"].initialize(
            n_timesteps=1, batch_size=1, size=2
        )
        self.junction.input["airTemperatureIn"].initialize(
            n_timesteps=1, batch_size=1, size=2
        )
        self.junction.output["airFlowRateOut"].initialize(n_timesteps=1, batch_size=1)
        self.junction.output["airTemperatureOut"].initialize(
            n_timesteps=1, batch_size=1
        )

        # Set inputs - vector inputs for multiple flows
        self.junction.input["airFlowRateIn"].set(
            torch.tensor([[0.5, 0.5]]), step_index=0
        )
        self.junction.input["airTemperatureIn"].set(
            torch.tensor([[20.0, 22.0]]), step_index=0
        )

        # Execute a time step
        datetime_val = datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)
        self.junction.do_step(
            second_time=0, date_time=datetime_val, step_size=600, step_index=0
        )

        # Check outputs
        flow_out = self.junction.output["airFlowRateOut"].get()
        temp_out = self.junction.output["airTemperatureOut"].get()

        self.assertIsNotNone(flow_out)
        self.assertIsNotNone(temp_out)
        # Total flow = 0.5 + 0.5 + 0.1 (bias) = 1.1
        self.assertAlmostEqual(flow_out.item(), 1.1, places=2)
        # Weighted avg temp = (20*0.5 + 22*0.5) / 1.1 ≈ 19.09
        self.assertAlmostEqual(temp_out.item(), 21.0 / 1.1, places=1)

    def test_do_step_zero_flow(self):
        """Test return flow junction with zero flow."""
        start_time = [datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)]
        end_time = [datetime.datetime(2023, 1, 1, 1, 40, 0, tzinfo=pytz.UTC)]
        step_size = [600]

        junction = ReturnFlowJunctionSystem(id="test_zero_flow")
        junction.initialize(
            start_time=start_time, end_time=end_time, step_size=step_size
        )

        # Set inputs with zero flow
        junction.input["airFlowRateIn"].set(torch.tensor([[0.0, 0.0]]), step_index=0)
        junction.input["airTemperatureIn"].set(
            torch.tensor([[20.0, 22.0]]), step_index=0
        )

        # Execute a time step
        datetime_val = datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)
        junction.do_step(
            second_time=0, date_time=datetime_val, step_size=600, step_index=0
        )

        # Check outputs - should be 0 flow and default temperature
        flow_out = junction.output["airFlowRateOut"].get()
        temp_out = junction.output["airTemperatureOut"].get()

        self.assertEqual(flow_out.item(), 0)
        self.assertEqual(temp_out.item(), 20)  # Default temperature when flow is 0

    def test_do_step_batch(self):
        """Test return flow junction do_step method with batch size > 1."""
        junction_batch = ReturnFlowJunctionSystem(
            id="test_return_junction_batch", airFlowRateBias=0.1
        )

        start_time = [datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)]
        end_time = [datetime.datetime(2023, 1, 1, 1, 40, 0, tzinfo=pytz.UTC)]
        step_size = [600]
        batch_size = 2
        junction_batch.initialize(
            start_time=start_time, end_time=end_time, step_size=step_size
        )
        junction_batch.input["airFlowRateIn"].initialize(
            n_timesteps=1, batch_size=batch_size, size=2
        )
        junction_batch.input["airTemperatureIn"].initialize(
            n_timesteps=1, batch_size=batch_size, size=2
        )
        junction_batch.output["airFlowRateOut"].initialize(
            n_timesteps=1, batch_size=batch_size
        )
        junction_batch.output["airTemperatureOut"].initialize(
            n_timesteps=1, batch_size=batch_size
        )

        # Set inputs - vector inputs for multiple flows with batch size 2
        junction_batch.input["airFlowRateIn"].set(
            torch.tensor([[0.5, 0.5], [0.6, 0.4]]), step_index=0
        )
        junction_batch.input["airTemperatureIn"].set(
            torch.tensor([[20.0, 22.0], [21.0, 23.0]]), step_index=0
        )

        # Execute a time step
        datetime_val = datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)
        junction_batch.do_step(
            second_time=0, date_time=datetime_val, step_size=600, step_index=0
        )

        # Check outputs - verify all outputs have consistent batch shape
        flow_out = junction_batch.output["airFlowRateOut"].get()
        temp_out = junction_batch.output["airTemperatureOut"].get()

        self.assertIsNotNone(flow_out)
        self.assertIsNotNone(temp_out)
        self.assertEqual(
            flow_out.shape[0], batch_size
        )  # Output batch matches input batch
        self.assertEqual(
            temp_out.shape[0], batch_size
        )  # Output batch matches input batch


class TestSupplyFlowJunctionSystem(unittest.TestCase):
    def setUp(self):
        self.junction = SupplyFlowJunctionSystem(
            id="test_supply_junction", airFlowRateBias=0.05
        )

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
        self.junction.initialize(
            start_time=start_time, end_time=end_time, step_size=step_size
        )
        self.junction.input["airFlowRateOut"].initialize(
            n_timesteps=1, batch_size=1, size=3
        )
        self.junction.output["airFlowRateIn"].initialize(n_timesteps=1, batch_size=1)

        # Set inputs - vector inputs for multiple flows
        self.junction.input["airFlowRateOut"].set(
            torch.tensor([[0.3, 0.4, 0.3]]), step_index=0
        )

        # Execute a time step
        datetime_val = datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)
        self.junction.do_step(
            second_time=0, date_time=datetime_val, step_size=600, step_index=0
        )

        # Check output - sum of flows + bias
        flow_in = self.junction.output["airFlowRateIn"].get()

        self.assertIsNotNone(flow_in)
        # Total = 0.3 + 0.4 + 0.3 + 0.05 = 1.05
        self.assertAlmostEqual(flow_in.item(), 1.05, places=2)

    def test_do_step_batch(self):
        """Test supply flow junction do_step method with batch size > 1."""
        junction_batch = SupplyFlowJunctionSystem(
            id="test_supply_junction_batch", airFlowRateBias=0.05
        )

        start_time = [datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)]
        end_time = [datetime.datetime(2023, 1, 1, 1, 40, 0, tzinfo=pytz.UTC)]
        step_size = [600]
        batch_size = 2
        junction_batch.initialize(
            start_time=start_time, end_time=end_time, step_size=step_size
        )
        junction_batch.input["airFlowRateOut"].initialize(
            n_timesteps=1, batch_size=batch_size, size=3
        )
        junction_batch.output["airFlowRateIn"].initialize(
            n_timesteps=1, batch_size=batch_size
        )

        # Set inputs - vector inputs for multiple flows with batch size 2
        junction_batch.input["airFlowRateOut"].set(
            torch.tensor([[0.3, 0.4, 0.3], [0.2, 0.5, 0.3]]), step_index=0
        )

        # Execute a time step
        datetime_val = datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)
        junction_batch.do_step(
            second_time=0, date_time=datetime_val, step_size=600, step_index=0
        )

        # Check output - sum of flows + bias, verify batch shape consistency
        flow_in = junction_batch.output["airFlowRateIn"].get()

        self.assertIsNotNone(flow_in)
        self.assertEqual(
            flow_in.shape[0], batch_size
        )  # Output batch matches input batch


class TestDiscreteStatespaceSystem(unittest.TestCase):
    def setUp(self):
        # Local application imports
        from twin4build.systems.utils.discrete_statespace_system import (
            DiscreteStatespaceSystem,
        )

        # Simple first-order system: dx/dt = -x + u, y = x
        A = torch.tensor([[0.9]], dtype=torch.float64)  # Already discrete
        B = torch.tensor([[0.1]], dtype=torch.float64)
        C = torch.tensor([[1.0]], dtype=torch.float64)
        D = torch.tensor([[0.0]], dtype=torch.float64)
        x0 = torch.tensor([0.0], dtype=torch.float64)

        self.system = DiscreteStatespaceSystem(
            id="test_ss", A=A, B=B, C=C, D=D, x0=x0, is_discrete=True
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

    def test_do_step(self):
        """Test discrete statespace system do_step method."""
        start_time = [datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)]
        end_time = [datetime.datetime(2023, 1, 1, 1, 40, 0, tzinfo=pytz.UTC)]
        step_size = [600]
        self.system.initialize(
            start_time=start_time, end_time=end_time, step_size=step_size
        )

        # Set input
        self.system.input["u"].set(
            torch.tensor([[1.0]], dtype=torch.float64), step_index=0
        )

        # Execute a time step
        datetime_val = datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)
        self.system.do_step(
            second_time=0, date_time=datetime_val, step_size=step_size, step_index=0
        )

        # Check output
        output = self.system.output["y"].get()
        self.assertIsNotNone(output)

    def test_do_step_batch(self):
        """Test discrete statespace system do_step method with batch size > 1."""
        A = torch.tensor([[0.9]], dtype=torch.float64)
        B = torch.tensor([[0.1]], dtype=torch.float64)
        C = torch.tensor([[1.0]], dtype=torch.float64)
        D = torch.tensor([[0.0]], dtype=torch.float64)
        x0 = torch.tensor([0.0], dtype=torch.float64)

        system_batch = DiscreteStatespaceSystem(
            id="test_ss_batch", A=A, B=B, C=C, D=D, x0=x0, is_discrete=True
        )

        batch_size = 2

        start_time = [datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)] * 2
        end_time = [datetime.datetime(2023, 1, 1, 1, 40, 0, tzinfo=pytz.UTC)] * 2
        step_size = [600] * 2
        system_batch.initialize(
            start_time=start_time, end_time=end_time, step_size=step_size
        )

        # Initialize inputs with batch size (state space input is 2D: batch x state_dim)
        system_batch.input["u"].initialize(n_timesteps=1, batch_size=batch_size, size=1)

        # Set input with batch size 2
        system_batch.input["u"].set(
            torch.tensor([[1.0], [2.0]], dtype=torch.float64), step_index=0
        )

        # Execute a time step
        datetime_val = datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)
        system_batch.do_step(
            second_time=0, date_time=datetime_val, step_size=step_size, step_index=0
        )

        # Check output - verify batch shape consistency
        output = system_batch.output["y"].get()
        self.assertIsNotNone(output)
        self.assertEqual(
            output.shape[0], batch_size
        )  # Output batch matches input batch


class TestPiecewiseLinearSystem(unittest.TestCase):
    def setUp(self):
        # Local application imports
        from twin4build.systems.utils.piecewise_linear_system import (
            PiecewiseLinearSystem,
        )

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
        controller = OnOffControllerSystem(
            id="test_reverse", offValue=0, onValue=1, isReverse=True
        )

        start_time = [datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)]
        end_time = [datetime.datetime(2023, 1, 1, 1, 40, 0, tzinfo=pytz.UTC)]
        step_size = [600]
        controller.initialize(
            start_time=start_time, end_time=end_time, step_size=step_size
        )

        # In reverse mode, when actual > setpoint, output should be ON
        controller.input["actualValue"].set(torch.tensor([25.0]), step_index=0)
        controller.input["setpointValue"].set(torch.tensor([22.0]), step_index=0)

        datetime_val = datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)
        controller.do_step(
            second_time=0, date_time=datetime_val, step_size=600, step_index=0
        )

        output = controller.output["inputSignal"].get()
        self.assertIsNotNone(output)


class TestPiecewiseLinearSystem(unittest.TestCase):
    def setUp(self):
        """Set up a PiecewiseLinearSystem for testing."""
        # Create simple linear points: (0, 0), (5, 10), (10, 5)
        X = torch.tensor([0.0, 5.0, 10.0])
        Y = torch.tensor([0.0, 10.0, 5.0])

        self.piecewise = PiecewiseLinearSystem(id="test_piecewise", X=X, Y=Y)

    def test_initialization(self):
        """Test PiecewiseLinearSystem initialization."""
        self.assertIsNotNone(self.piecewise)
        self.assertEqual(self.piecewise.id, "test_piecewise")

    def test_config_property(self):
        """Test config property returns correct parameters."""
        config = self.piecewise.config
        self.assertIsInstance(config, dict)
        self.assertIn("parameters", config)

    def test_do_step(self):
        """Test do_step method with piecewise linear interpolation."""
        # Initialize
        start_time = [datetime.datetime(2023, 1, 2, 8, 0, 0, tzinfo=pytz.UTC)]
        end_time = [datetime.datetime(2023, 1, 2, 18, 0, 0, tzinfo=pytz.UTC)]
        step_size = [3600]

        self.piecewise.initialize(
            start_time=start_time, end_time=end_time, step_size=step_size
        )

        # Set input (x value for interpolation)
        self.piecewise.input["x"].set(
            torch.tensor([2.5]), step_index=0
        )  # Midpoint between 0 and 5

        # Execute a time step
        datetime_val = datetime.datetime(2023, 1, 2, 10, 0, 0, tzinfo=pytz.UTC)
        self.piecewise.do_step(
            second_time=7200, date_time=datetime_val, step_size=3600, step_index=0
        )

        # Check output is set (should be interpolated: 2.5 * 2 = 5.0)
        output = self.piecewise.output["y"].get()
        self.assertIsNotNone(output)
        self.assertAlmostEqual(output.item(), 5.0, places=1)

    def test_do_step_batch(self):
        """Test do_step method with piecewise linear interpolation and batch size > 1."""
        piecewise_batch = PiecewiseLinearSystem(
            id="test_piecewise_batch",
            X=torch.tensor([0.0, 5.0, 10.0]),
            Y=torch.tensor([0.0, 10.0, 5.0]),
        )

        batch_size = 2

        # Initialize
        start_time = [datetime.datetime(2023, 1, 2, 8, 0, 0, tzinfo=pytz.UTC)]
        end_time = [datetime.datetime(2023, 1, 2, 18, 0, 0, tzinfo=pytz.UTC)]
        step_size = [3600]

        piecewise_batch.initialize(
            start_time=start_time, end_time=end_time, step_size=step_size
        )

        # Initialize inputs with batch size
        piecewise_batch.input["x"].initialize(
            n_timesteps=1, batch_size=batch_size, size=1
        )
        piecewise_batch.output["y"].initialize(
            n_timesteps=1, batch_size=batch_size, size=1
        )

        # Set input with batch size 2
        piecewise_batch.input["x"].set(torch.tensor([2.5, 7.5]), step_index=0)

        # Execute a time step
        datetime_val = datetime.datetime(2023, 1, 2, 10, 0, 0, tzinfo=pytz.UTC)
        piecewise_batch.do_step(
            second_time=7200, date_time=datetime_val, step_size=3600, step_index=0
        )

        # Check output - verify batch shape consistency
        output = piecewise_batch.output["y"].get()
        self.assertIsNotNone(output)
        self.assertEqual(
            output.shape[0], batch_size
        )  # Output batch matches input batch


class TestMaxSystem(unittest.TestCase):
    def setUp(self):
        self.max_system = MaxSystem(id="test_max")

    def test_initialization(self):
        """Test MaxSystem initialization."""
        self.assertIsNotNone(self.max_system)
        self.assertEqual(self.max_system.id, "test_max")

    def test_do_step(self):
        """Test MaxSystem do_step method."""
        start_time = [datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)]
        end_time = [datetime.datetime(2023, 1, 1, 1, 0, 0, tzinfo=pytz.UTC)]
        step_size = [600]
        self.max_system.input["inputs"].initialize(n_timesteps=1, batch_size=1, size=3)
        self.max_system.initialize(
            start_time=start_time, end_time=end_time, step_size=step_size
        )

        # MaxSystem uses "inputs" as a Vector
        input_values = torch.tensor([5.0, 3.0, 2.0])
        self.max_system.input["inputs"].set(input_values, step_index=0)

        # Execute a time step
        datetime_val = datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)
        self.max_system.do_step(
            second_time=0, date_time=datetime_val, step_size=600, step_index=0
        )

        # Check output is maximum
        output = self.max_system.output["value"].get()
        self.assertIsNotNone(output)
        self.assertEqual(output.item(), 5.0)

    def test_do_step_different_max(self):
        """Test MaxSystem with different maximum value."""
        start_time = [datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)]
        end_time = [datetime.datetime(2023, 1, 1, 1, 0, 0, tzinfo=pytz.UTC)]
        step_size = [600]
        self.max_system.input["inputs"].initialize(n_timesteps=1, batch_size=1, size=3)
        self.max_system.initialize(
            start_time=start_time, end_time=end_time, step_size=step_size
        )

        input_values = torch.tensor([[2.0, 8.0, 4.0]])
        self.max_system.input["inputs"].set(input_values, step_index=0)

        datetime_val = datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)
        self.max_system.do_step(
            second_time=0, date_time=datetime_val, step_size=600, step_index=0
        )

        output = self.max_system.output["value"].get()
        self.assertEqual(output.item(), 8.0)

    def test_do_step_batch(self):
        """Test MaxSystem with batch size > 1."""
        max_system_batch = MaxSystem(id="test_max_batch")

        start_time = [datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)] * 2
        end_time = [datetime.datetime(2023, 1, 1, 1, 0, 0, tzinfo=pytz.UTC)] * 2
        step_size = [600] * 2
        batch_size = 2
        max_system_batch.input["inputs"].initialize(n_timesteps=1, size=3)
        max_system_batch.initialize(
            start_time=start_time, end_time=end_time, step_size=step_size
        )

        # Set inputs with batch size 2
        input_values = torch.tensor([[5.0, 3.0, 2.0], [1.0, 9.0, 4.0]])
        max_system_batch.input["inputs"].set(input_values, step_index=0)

        # Execute a time step
        datetime_val = datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)
        max_system_batch.do_step(
            second_time=0, date_time=datetime_val, step_size=step_size, step_index=0
        )

        # Check output is maximum - verify batch shape consistency
        output = max_system_batch.output["value"].get()
        self.assertIsNotNone(output)
        self.assertEqual(
            output.shape[0], batch_size
        )  # Output batch matches input batch
        self.assertAlmostEqual(output[0].item(), 5.0)
        self.assertAlmostEqual(output[1].item(), 9.0)


class TestOnOffSystem(unittest.TestCase):
    def setUp(self):
        self.onoff_system = OnOffSystem(
            id="test_onoff", threshold=0.5, is_off_value=0.0
        )

    def test_initialization(self):
        """Test OnOffSystem initialization."""
        self.assertIsNotNone(self.onoff_system)
        self.assertEqual(self.onoff_system.id, "test_onoff")

    def test_do_step_above_threshold(self):
        """Test OnOffSystem when criteria is above threshold."""
        start_time = [datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)]
        end_time = [datetime.datetime(2023, 1, 1, 1, 0, 0, tzinfo=pytz.UTC)]
        step_size = [600]
        self.onoff_system.initialize(
            start_time=start_time, end_time=end_time, step_size=step_size
        )

        # Set inputs - criteriaValue above threshold
        self.onoff_system.input["criteriaValue"].set(
            torch.tensor([1.0]), step_index=0
        )  # Above 0.5
        self.onoff_system.input["value"].set(torch.tensor([100.0]), step_index=0)

        # Execute a time step
        datetime_val = datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)
        self.onoff_system.do_step(
            second_time=0, date_time=datetime_val, step_size=600, step_index=0
        )

        # Check output equals input value when above threshold
        output = self.onoff_system.output["value"].get()
        self.assertIsNotNone(output)
        self.assertEqual(output.item(), 100.0)

    def test_do_step_below_threshold(self):
        """Test OnOffSystem when criteria is below threshold."""
        start_time = [datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)]
        end_time = [datetime.datetime(2023, 1, 1, 1, 0, 0, tzinfo=pytz.UTC)]
        step_size = [600]
        self.onoff_system.initialize(
            start_time=start_time, end_time=end_time, step_size=step_size
        )

        # Set inputs - criteriaValue below threshold
        self.onoff_system.input["criteriaValue"].set(
            torch.tensor([0.2]), step_index=0
        )  # Below 0.5
        self.onoff_system.input["value"].set(torch.tensor([100.0]), step_index=0)

        datetime_val = datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)
        self.onoff_system.do_step(
            second_time=0, date_time=datetime_val, step_size=600, step_index=0
        )

        # Check output is is_off_value when below threshold
        output = self.onoff_system.output["value"].get()
        self.assertEqual(output.item(), 0.0)

    def test_do_step_batch(self):
        """Test OnOffSystem with batch size > 1."""
        onoff_system_batch = OnOffSystem(
            id="test_onoff_batch", threshold=0.5, is_off_value=0.0
        )

        batch_size = 3

        start_time = [
            datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)
        ] * batch_size
        end_time = [
            datetime.datetime(2023, 1, 1, 1, 0, 0, tzinfo=pytz.UTC)
        ] * batch_size
        step_size = [600] * batch_size
        onoff_system_batch.initialize(
            start_time=start_time, end_time=end_time, step_size=step_size
        )

        # Initialize inputs with batch size
        onoff_system_batch.input["criteriaValue"].initialize(
            n_timesteps=1, batch_size=batch_size, size=1
        )
        onoff_system_batch.input["value"].initialize(
            n_timesteps=1, batch_size=batch_size, size=1
        )

        # Set inputs with batch size 3 - some above, some below threshold
        onoff_system_batch.input["criteriaValue"].set(
            torch.tensor([1.0, 0.2, 0.8]), step_index=0
        )
        onoff_system_batch.input["value"].set(
            torch.tensor([100.0, 200.0, 150.0]), step_index=0
        )

        # Execute a time step
        datetime_val = datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)
        onoff_system_batch.do_step(
            second_time=0, date_time=datetime_val, step_size=600, step_index=0
        )

        # Check output - verify batch shape consistency and values
        output = onoff_system_batch.output["value"].get()
        self.assertIsNotNone(output)
        self.assertEqual(
            output.shape[0], batch_size
        )  # Output batch matches input batch
        self.assertEqual(output[0].item(), 100.0)  # Above threshold
        self.assertEqual(output[1].item(), 0.0)  # Below threshold
        self.assertEqual(output[2].item(), 150.0)  # Above threshold


class TestPassInputToOutput(unittest.TestCase):
    def setUp(self):
        self.pass_system = PassInputToOutput(id="test_pass")

    def test_initialization(self):
        """Test PassInputToOutput initialization."""
        self.assertIsNotNone(self.pass_system)
        self.assertEqual(self.pass_system.id, "test_pass")

    def test_do_step(self):
        """Test PassInputToOutput passes value correctly."""
        start_time = [datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)]
        end_time = [datetime.datetime(2023, 1, 1, 1, 0, 0, tzinfo=pytz.UTC)]
        step_size = [600]
        self.pass_system.initialize(
            start_time=start_time, end_time=end_time, step_size=step_size
        )

        # Set input (correct key is "value")
        test_value = 42.5
        self.pass_system.input["value"].set(torch.tensor([test_value]), step_index=0)

        # Execute a time step
        datetime_val = datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)
        self.pass_system.do_step(
            second_time=0, date_time=datetime_val, step_size=600, step_index=0
        )

        # Check output equals input
        output = self.pass_system.output["value"].get()
        self.assertIsNotNone(output)
        self.assertEqual(output.item(), test_value)

    def test_do_step_batch(self):
        """Test PassInputToOutput passes value correctly with batch size > 1."""
        pass_system_batch = PassInputToOutput(id="test_pass_batch")

        batch_size = 2

        start_time = [datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)]
        end_time = [datetime.datetime(2023, 1, 1, 1, 0, 0, tzinfo=pytz.UTC)]
        step_size = [600]
        pass_system_batch.initialize(
            start_time=start_time, end_time=end_time, step_size=step_size
        )

        # Initialize inputs with batch size
        pass_system_batch.input["value"].initialize(
            n_timesteps=1, batch_size=batch_size, size=1
        )
        pass_system_batch.output["value"].initialize(
            n_timesteps=1, batch_size=batch_size, size=1
        )

        # Set input with batch size 2
        test_values = torch.tensor([42.5, 55.3])
        pass_system_batch.input["value"].set(test_values, step_index=0)

        # Execute a time step
        datetime_val = datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)
        pass_system_batch.do_step(
            second_time=0, date_time=datetime_val, step_size=600, step_index=0
        )

        # Check output equals input - verify batch shape consistency
        output = pass_system_batch.output["value"].get()
        self.assertIsNotNone(output)
        self.assertEqual(
            output.shape[0], batch_size
        )  # Output batch matches input batch
        self.assertAlmostEqual(output[0].item(), 42.5, places=5)
        self.assertAlmostEqual(output[1].item(), 55.3, places=5)


class TestBuildingSpaceTorchSystem(unittest.TestCase):
    def setUp(self):
        """Set up a BuildingSpaceTorchSystem for testing."""
        # BuildingSpaceTorchSystem is a composite system with thermal and mass subsystems
        thermal_kwargs = {
            "C_air": 1000.0,
            "C_wall": 5000.0,
            "C_int": 3000.0,
            "C_boundary": 10000.0,
            "R_out": 0.01,
            "R_in": 0.005,
            "R_int": 0.003,
            "R_boundary": 0.008,
            "f_wall": 0.3,
            "f_air": 0.7,
            "Q_occ_gain": 100.0,
        }
        mass_kwargs = {"V": 100.0, "G_occ": 0.01, "m_inf": 0.001}
        self.building_space = BuildingSpaceTorchSystem(
            id="test_building_space",
            thermal_kwargs=thermal_kwargs,
            mass_kwargs=mass_kwargs,
        )

    def test_initialization(self):
        """Test BuildingSpaceTorchSystem initialization."""
        self.assertIsNotNone(self.building_space)
        self.assertEqual(self.building_space.id, "test_building_space")
        # Check subsystems exist
        self.assertIsNotNone(self.building_space.thermal)
        self.assertIsNotNone(self.building_space.mass)

    def test_config_parameters(self):
        """Test that config parameters are properly set."""
        config = self.building_space.config
        self.assertIsInstance(config, dict)
        self.assertIn("parameters", config)
        # Parameters are prefixed with "thermal." or "mass."
        self.assertIn("thermal.C_air", config["parameters"])
        self.assertIn("thermal.R_out", config["parameters"])
        self.assertIn("mass.V", config["parameters"])

    def test_subsystem_parameters(self):
        """Test that subsystem parameters are accessible."""
        # Access through subsystems
        self.assertEqual(self.building_space.thermal.C_air.get(), 1000.0)
        self.assertEqual(self.building_space.thermal.C_wall.get(), 5000.0)
        self.assertAlmostEqual(self.building_space.thermal.R_out.get(), 0.01)

    def test_input_output_structure(self):
        """Test that input/output ports exist."""
        self.assertIsInstance(self.building_space.input, dict)
        self.assertIsInstance(self.building_space.output, dict)
        # Check for actual input keys from the implementation
        self.assertIn("supplyAirFlowRate", self.building_space.input)
        self.assertIn("exhaustAirFlowRate", self.building_space.input)
        self.assertIn("outdoorTemperature", self.building_space.input)
        # Check for actual output keys
        self.assertIn("indoorTemperature", self.building_space.output)
        self.assertIn("indoorCO2", self.building_space.output)

    def test_do_step(self):
        """Test BuildingSpaceTorchSystem do_step method."""
        start_time = [datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)]
        end_time = [datetime.datetime(2023, 1, 1, 1, 40, 0, tzinfo=pytz.UTC)]
        step_size = [600]
        self.building_space.initialize(
            start_time=start_time, end_time=end_time, step_size=step_size
        )

        # Set required inputs
        self.building_space.input["supplyAirFlowRate"].set(
            torch.tensor([0.5]), step_index=0
        )
        self.building_space.input["exhaustAirFlowRate"].set(
            torch.tensor([0.5]), step_index=0
        )
        self.building_space.input["outdoorTemperature"].set(
            torch.tensor([5.0]), step_index=0
        )
        self.building_space.input["supplyAirTemperature"].set(
            torch.tensor([20.0]), step_index=0
        )
        self.building_space.input["outdoorCO2"].set(torch.tensor([400.0]), step_index=0)
        self.building_space.input["numberOfPeople"].set(
            torch.tensor([2.0]), step_index=0
        )
        self.building_space.input["globalIrradiation"].set(
            torch.tensor([0.0]), step_index=0
        )

        # Execute a time step
        datetime_val = datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)
        self.building_space.do_step(
            second_time=0, date_time=datetime_val, step_size=step_size, step_index=0
        )

        # Check outputs
        indoor_temp = self.building_space.output["indoorTemperature"].get()
        indoor_co2 = self.building_space.output["indoorCO2"].get()

        self.assertIsNotNone(indoor_temp)
        self.assertIsNotNone(indoor_co2)

    def test_do_step_batch(self):
        """Test BuildingSpaceTorchSystem do_step method with batch size > 1."""
        thermal_kwargs = {
            "C_air": 1000.0,
            "C_wall": 5000.0,
            "C_int": 3000.0,
            "C_boundary": 10000.0,
            "R_out": 0.01,
            "R_in": 0.005,
            "R_int": 0.003,
            "R_boundary": 0.008,
            "f_wall": 0.3,
            "f_air": 0.7,
            "Q_occ_gain": 100.0,
        }
        mass_kwargs = {"V": 100.0, "G_occ": 0.01, "m_inf": 0.001}
        building_space_batch = BuildingSpaceTorchSystem(
            id="test_building_space_batch",
            thermal_kwargs=thermal_kwargs,
            mass_kwargs=mass_kwargs,
        )

        batch_size = 2

        start_time = [datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)] * 2
        end_time = [datetime.datetime(2023, 1, 1, 1, 40, 0, tzinfo=pytz.UTC)] * 2
        step_size = [600] * 2
        building_space_batch.initialize(
            start_time=start_time, end_time=end_time, step_size=step_size
        )

        # Initialize inputs with batch size
        # building_space_batch.input["supplyAirFlowRate"].initialize(n_timesteps=1, batch_size=batch_size, size=1)
        # building_space_batch.input["exhaustAirFlowRate"].initialize(n_timesteps=1, batch_size=batch_size, size=1)
        # building_space_batch.input["outdoorTemperature"].initialize(n_timesteps=1, batch_size=batch_size, size=1)
        # building_space_batch.input["supplyAirTemperature"].initialize(n_timesteps=1, batch_size=batch_size, size=1)
        # building_space_batch.input["outdoorCO2"].initialize(n_timesteps=1, batch_size=batch_size, size=1)
        # building_space_batch.input["numberOfPeople"].initialize(n_timesteps=1, batch_size=batch_size, size=1)
        # building_space_batch.input["globalIrradiation"].initialize(n_timesteps=1, batch_size=batch_size, size=1)

        # Set required inputs with batch size 2
        building_space_batch.input["supplyAirFlowRate"].set(
            torch.tensor([0.5, 0.6]), step_index=0
        )
        building_space_batch.input["exhaustAirFlowRate"].set(
            torch.tensor([0.5, 0.6]), step_index=0
        )
        building_space_batch.input["outdoorTemperature"].set(
            torch.tensor([5.0, 3.0]), step_index=0
        )
        building_space_batch.input["supplyAirTemperature"].set(
            torch.tensor([20.0, 21.0]), step_index=0
        )
        building_space_batch.input["outdoorCO2"].set(
            torch.tensor([400.0, 400.0]), step_index=0
        )
        building_space_batch.input["numberOfPeople"].set(
            torch.tensor([2.0, 3.0]), step_index=0
        )
        building_space_batch.input["globalIrradiation"].set(
            torch.tensor([0.0, 100.0]), step_index=0
        )

        # Execute a time step
        datetime_val = datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)
        building_space_batch.do_step(
            second_time=0, date_time=datetime_val, step_size=step_size, step_index=0
        )

        # Check outputs - verify all outputs have consistent batch shape
        indoor_temp = building_space_batch.output["indoorTemperature"].get()
        indoor_co2 = building_space_batch.output["indoorCO2"].get()

        self.assertIsNotNone(indoor_temp)
        self.assertIsNotNone(indoor_co2)
        self.assertEqual(
            indoor_temp.shape[0], batch_size
        )  # Output batch matches input batch
        self.assertEqual(
            indoor_co2.shape[0], batch_size
        )  # Output batch matches input batch


if __name__ == "__main__":
    unittest.main()
