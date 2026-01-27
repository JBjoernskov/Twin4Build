# Standard library imports
import datetime
import unittest

# Third party imports
import pytz
import torch

# Local application imports
from twin4build.systems.valve.valve_torch_system import ValveTorchSystem

# Set test flag
import twin4build
twin4build._IS_TESTING = True


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
        self.valve.input["valvePosition"].set(torch.tensor([0.5]), i_t=0)

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
            n_t=1, n_s=batch_size, n_v=1
        )
        valve_batch.output["waterFlowRate"].initialize(
            n_t=1, n_s=batch_size, n_v=1
        )
        valve_batch.output["valvePosition"].initialize(
            n_t=1, n_s=batch_size, n_v=1
        )

        # Set inputs with batch size 3
        valve_batch.input["valvePosition"].set(
            torch.tensor([0.5, 0.7, 0.3]), i_t=0
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


if __name__ == "__main__":
    unittest.main()
