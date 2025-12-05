# Standard library imports
import datetime
import unittest

# Third party imports
import pytz
import torch

# Local application imports
from twin4build.systems.fan.fan_torch_system import FanTorchSystem


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


if __name__ == "__main__":
    unittest.main()

