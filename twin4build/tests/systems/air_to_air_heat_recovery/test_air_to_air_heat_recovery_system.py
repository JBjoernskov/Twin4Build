# Standard library imports
import datetime
import unittest

# Third party imports
import pytz
import torch

# Local application imports
from twin4build.systems.air_to_air_heat_recovery.air_to_air_heat_recovery_system import (
    AirToAirHeatRecoverySystem,
)


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
            torch.tensor([0.0]), i_t=0
        )  # Cold outdoor
        self.hr.input["primaryAirFlowRate"].set(torch.tensor([1.0]), i_t=0)
        self.hr.input["primaryTemperatureOutSetpoint"].set(
            torch.tensor([20.0]), i_t=0
        )

        # Exhaust side (indoor to outdoor)
        self.hr.input["secondaryTemperatureIn"].set(
            torch.tensor([20.0]), i_t=0
        )  # Warm return
        self.hr.input["secondaryAirFlowRate"].set(torch.tensor([1.0]), i_t=0)

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

        # Set inputs with batch size 2
        hr_batch.input["primaryTemperatureIn"].set(
            torch.tensor([0.0, -5.0]), i_t=0
        )
        hr_batch.input["primaryAirFlowRate"].set(torch.tensor([1.0, 0.8]), i_t=0)
        hr_batch.input["primaryTemperatureOutSetpoint"].set(
            torch.tensor([20.0, 20.0]), i_t=0
        )
        hr_batch.input["secondaryTemperatureIn"].set(
            torch.tensor([20.0, 22.0]), i_t=0
        )
        hr_batch.input["secondaryAirFlowRate"].set(
            torch.tensor([1.0, 0.8]), i_t=0
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


if __name__ == "__main__":
    unittest.main()
