# Standard library imports
import datetime
import unittest

# Third party imports
import pytz
import torch

# Local application imports
from twin4build.systems.controller.setpoint_controller.pid_controller.pid_controller_system import (
    PIDControllerSystem,
)


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


if __name__ == "__main__":
    unittest.main()
