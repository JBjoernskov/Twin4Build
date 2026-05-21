# Standard library imports
import datetime
import unittest

# Third party imports
from dateutil import tz
import torch

# Local application imports
from twin4build.systems.coil.coil_torch_system import CoilTorchSystem


class TestCoilTorchSystem(unittest.TestCase):
    def setUp(self):
        self.coil = CoilTorchSystem(id="test_coil")

    def test_initialization(self):
        """Test coil system initialization."""
        self.assertIsNotNone(self.coil)
        self.assertEqual(self.coil.id, "test_coil")

    def test_do_step(self):
        """Test coil system do_step method."""
        start_time = [datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=tz.UTC)]
        end_time = [datetime.datetime(2023, 1, 1, 1, 40, 0, tzinfo=tz.UTC)]
        step_size = [600]
        self.coil.initialize(
            start_time=start_time, end_time=end_time, step_size=step_size
        )

        # Set inputs
        self.coil.input["inletAirTemperature"].set(torch.tensor([20.0]), i_t=0)
        self.coil.input["outletAirTemperatureSetpoint"].set(
            torch.tensor([22.0]), i_t=0
        )
        self.coil.input["airFlowRate"].set(torch.tensor([1.0]), i_t=0)

        # Execute a time step
        datetime_val = datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=tz.UTC)
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

        start_time = [datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=tz.UTC)] * 2
        end_time = [datetime.datetime(2023, 1, 1, 1, 40, 0, tzinfo=tz.UTC)] * 2
        step_size = [600] * 2
        coil_batch.initialize(
            start_time=start_time, end_time=end_time, step_size=step_size
        )

        # Set inputs with batch size 2
        coil_batch.input["inletAirTemperature"].set(
            torch.tensor([20.0, 25.0]), i_t=0
        )
        coil_batch.input["outletAirTemperatureSetpoint"].set(
            torch.tensor([22.0, 23.0]), i_t=0
        )
        coil_batch.input["airFlowRate"].set(torch.tensor([1.0, 1.2]), i_t=0)

        # Execute a time step
        datetime_val = datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=tz.UTC)
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


if __name__ == "__main__":
    unittest.main()
