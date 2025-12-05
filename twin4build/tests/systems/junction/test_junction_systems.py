# Standard library imports
import datetime
import unittest

# Third party imports
import pytz
import torch

# Local application imports
from twin4build.systems.junction.return_flow_junction_system import (
    ReturnFlowJunctionSystem,
)
from twin4build.systems.junction.supply_flow_junction_system import (
    SupplyFlowJunctionSystem,
)


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


if __name__ == "__main__":
    unittest.main()

