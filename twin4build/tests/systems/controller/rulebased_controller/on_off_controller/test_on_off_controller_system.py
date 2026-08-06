# Standard library imports
import datetime
import unittest

# Third party imports
import torch
from dateutil import tz

# Local application imports
# Set test flag
import twin4build
from twin4build.systems.controller.rulebased_controller.on_off_controller.on_off_controller_system import (
    OnOffControllerSystem,
)

twin4build._IS_TESTING = True


class TestOnOffControllerSystem(unittest.TestCase):
    def setUp(self):
        self.controller = OnOffControllerSystem(
            id="test_controller", off_value=0, on_value=1, is_reverse=False
        )

    def test_initialization(self):
        """Test on/off controller initialization."""
        self.assertIsNotNone(self.controller)
        self.assertEqual(self.controller.id, "test_controller")

    def test_do_step(self):
        """Test on/off controller do_step method."""
        start_time = [datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=tz.UTC)]
        end_time = [datetime.datetime(2023, 1, 1, 1, 40, 0, tzinfo=tz.UTC)]
        step_size = [600]
        self.controller.initialize(
            start_time=start_time, end_time=end_time, step_size=step_size
        )

        # Set inputs
        self.controller.input["actualValue"].set(torch.tensor([20.0]), i_t=0)
        self.controller.input["setpointValue"].set(torch.tensor([22.0]), i_t=0)

        # Execute a time step
        datetime_val = datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=tz.UTC)
        self.controller.do_step(
            second_time=0, date_time=datetime_val, step_size=600, step_index=0
        )

        # Check output
        output = self.controller.output["inputSignal"].get()
        self.assertIsNotNone(output)

    def test_do_step_batch(self):
        """Test on/off controller do_step method with batch size > 1."""
        controller_batch = OnOffControllerSystem(
            id="test_controller_batch", off_value=0, on_value=1, is_reverse=False
        )

        batch_size = 2

        start_time = [
            datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=tz.UTC)
        ] * batch_size
        end_time = [datetime.datetime(2023, 1, 1, 1, 40, 0, tzinfo=tz.UTC)] * batch_size
        step_size = [600] * batch_size
        controller_batch.initialize(
            start_time=start_time, end_time=end_time, step_size=step_size
        )

        # Set inputs with batch size 2
        controller_batch.input["actualValue"].set(torch.tensor([20.0, 23.0]), i_t=0)
        controller_batch.input["setpointValue"].set(torch.tensor([22.0, 22.0]), i_t=0)

        # Execute a time step
        datetime_val = datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=tz.UTC)
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
