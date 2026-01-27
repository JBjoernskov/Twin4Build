# Standard library imports
import datetime
import unittest

# Third party imports
import pytz
import torch

# Local application imports
from twin4build.systems.damper.damper_torch_system import DamperTorchSystem

# Set test flag
import twin4build
twin4build._IS_TESTING = True


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
        self.damper.input["damperPosition"].set(torch.tensor([0.5]), i_t=0)

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
        self.damper.input["damperPosition"].set(torch.tensor([1.0]), i_t=0)
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
            n_t=1, n_s=batch_size, n_v=1
        )
        damper_batch.output["damperPosition"].initialize(
            n_t=1, n_s=batch_size, n_v=1
        )
        damper_batch.output["airFlowRate"].initialize(
            n_t=1, n_s=batch_size, n_v=1
        )

        # Set input with batch size 3
        damper_batch.input["damperPosition"].set(
            torch.tensor([0.5, 0.7, 0.3]), i_t=0
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


if __name__ == "__main__":
    unittest.main()
