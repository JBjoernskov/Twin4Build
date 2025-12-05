# Standard library imports
import datetime
import unittest
import warnings

# Third party imports
import numpy as np
import pandas as pd
import pytz
import torch

# Local application imports
from twin4build.systems.utils.discrete_statespace_system import DiscreteStatespaceSystem
from twin4build.systems.utils.max_system import MaxSystem
from twin4build.systems.utils.on_off_system import OnOffSystem
from twin4build.systems.utils.pass_input_to_output import PassInputToOutput
from twin4build.systems.utils.piecewise_linear_system import PiecewiseLinearSystem
from twin4build.systems.utils.time_series_input_system import TimeSeriesInputSystem


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


class TestDiscreteStatespaceSystem(unittest.TestCase):
    def setUp(self):
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


class TestPiecewiseLinearSystem(unittest.TestCase):
    def setUp(self):
        # Create a simple piecewise linear function
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


class TestMaxSystem(unittest.TestCase):
    def setUp(self):
        self.system = MaxSystem(id="test_max")

    def test_initialization(self):
        """Test max system initialization."""
        self.assertIsNotNone(self.system)
        self.assertEqual(self.system.id, "test_max")

    def test_do_step(self):
        """Test max system do_step method."""
        start_time = [datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)]
        end_time = [datetime.datetime(2023, 1, 1, 1, 0, 0, tzinfo=pytz.UTC)]
        step_size = [600]
        self.system.initialize(
            start_time=start_time, end_time=end_time, step_size=step_size
        )

        # Set inputs
        self.system.input["inputSignal"].initialize(n_timesteps=1, batch_size=1, size=3)
        self.system.input["inputSignal"].set(
            torch.tensor([[1.0, 5.0, 3.0]]), step_index=0
        )

        # Execute
        datetime_val = datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)
        self.system.do_step(
            second_time=0, date_time=datetime_val, step_size=600, step_index=0
        )

        # Check output
        output = self.system.output["outputSignal"].get()
        self.assertIsNotNone(output)
        self.assertEqual(output.item(), 5.0)


class TestOnOffSystem(unittest.TestCase):
    def setUp(self):
        self.system = OnOffSystem(id="test_onoff", offValue=0, onValue=100)

    def test_initialization(self):
        """Test on/off system initialization."""
        self.assertIsNotNone(self.system)
        self.assertEqual(self.system.id, "test_onoff")

    def test_do_step_on(self):
        """Test on/off system when signal is ON (> 0.5)."""
        start_time = [datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)]
        end_time = [datetime.datetime(2023, 1, 1, 1, 0, 0, tzinfo=pytz.UTC)]
        step_size = [600]
        self.system.initialize(
            start_time=start_time, end_time=end_time, step_size=step_size
        )

        # Set input > 0.5
        self.system.input["inputSignal"].set(torch.tensor([0.8]), step_index=0)

        # Execute
        datetime_val = datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)
        self.system.do_step(
            second_time=0, date_time=datetime_val, step_size=600, step_index=0
        )

        # Check output should be onValue
        output = self.system.output["outputSignal"].get()
        self.assertEqual(output.item(), 100)


class TestPassInputToOutput(unittest.TestCase):
    def setUp(self):
        self.system = PassInputToOutput(id="test_passthrough")

    def test_initialization(self):
        """Test pass input to output system initialization."""
        self.assertIsNotNone(self.system)
        self.assertEqual(self.system.id, "test_passthrough")

    def test_do_step(self):
        """Test pass input to output system do_step method."""
        start_time = [datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)]
        end_time = [datetime.datetime(2023, 1, 1, 1, 0, 0, tzinfo=pytz.UTC)]
        step_size = [600]
        self.system.initialize(
            start_time=start_time, end_time=end_time, step_size=step_size
        )

        # Set input
        self.system.input["inputSignal"].set(torch.tensor([42.0]), step_index=0)

        # Execute
        datetime_val = datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)
        self.system.do_step(
            second_time=0, date_time=datetime_val, step_size=600, step_index=0
        )

        # Check output equals input
        output = self.system.output["outputSignal"].get()
        self.assertEqual(output.item(), 42.0)


if __name__ == "__main__":
    unittest.main()

