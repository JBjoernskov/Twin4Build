# Standard library imports
import datetime
import unittest
import warnings

# Third party imports
import numpy as np
import pandas as pd
import torch
from dateutil import tz

# Local application imports
# Set test flag
import twin4build
from twin4build.systems.utils.discrete_statespace_system import DiscreteStatespaceSystem
from twin4build.systems.utils.max_system import MaxSystem
from twin4build.systems.utils.weighted_sum_system import WeightedSumSystem
from twin4build.systems.utils.on_off_system import OnOffSystem
from twin4build.systems.utils.pass_input_to_output import PassInputToOutput
from twin4build.systems.utils.piecewise_linear_system import PiecewiseLinearSystem
from twin4build.systems.utils.time_series_input_system import TimeSeriesInputSystem
from twin4build.utils.data_loaders.sources import DataFrameSource

twin4build._IS_TESTING = True


class TestTimeSeriesInputSystem(unittest.TestCase):
    def setUp(self):
        dates = pd.date_range(
            start=datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=tz.UTC),
            periods=10,
            freq="10min",
        )
        df = pd.DataFrame({"value": [0.5] * 10}, index=dates)
        self.ts_input = TimeSeriesInputSystem(id="test_timeseries", source=DataFrameSource(df))

    def test_initialization(self):
        """Test time series input system initialization."""
        self.assertIsNotNone(self.ts_input)
        self.assertEqual(self.ts_input.id, "test_timeseries")

    def test_do_step(self):
        """Test time series input system do_step method."""
        start_time = [datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=tz.UTC)]
        end_time = [datetime.datetime(2023, 1, 1, 1, 40, 0, tzinfo=tz.UTC)]
        step_size = [600]
        self.ts_input.initialize(
            start_time=start_time, end_time=end_time, step_size=step_size
        )

        # Execute a time step
        datetime_val = datetime.datetime(2023, 1, 1, 0, 10, 0, tzinfo=tz.UTC)
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
            id="test_ss", A=A, B=B, C=C, D=D, x0=x0
        )

    def test_initialization(self):
        """Test discrete statespace system initialization."""
        self.assertIsNotNone(self.system)
        self.assertEqual(self.system.id, "test_ss")

    def test_state_property(self):
        """Test get_state and set_state methods."""
        # Standard library imports

        # Must initialize before get_state/set_state
        start_time = [datetime.datetime(2024, 1, 1)]
        end_time = [datetime.datetime(2024, 1, 2)]
        self.system.initialize(start_time, end_time, step_size=3600)

        state = self.system.get_state()
        self.assertIsNotNone(state)

        # Set a new state - shape (n_s, n_c, n_states) = (1, 1, 1)
        new_state = torch.tensor([[[1.5]]], dtype=torch.float64)
        self.system.set_state(new_state)
        retrieved_state = self.system.get_state()
        self.assertAlmostEqual(retrieved_state[0, 0, 0].item(), 1.5, places=5)

    def test_the_matrices_are_settings_that_size_the_ports(self):
        """2.0: declared; the matrices are nested tuples (saved as JSON, compared by value when copies batch)."""
        system = DiscreteStatespaceSystem(id="sized", A=[[-1e-4, 0.0], [0.0, -2e-4]], B=[[1e-4], [0.0]],
                                          C=[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], x0=[20.0, 18.0])
        self.assertEqual(system.A, ((-1e-4, 0.0), (0.0, -2e-4)))
        self.assertEqual(system.declared_size("output", "y"), 3)
        self.assertEqual(system.declared_size("state", "x"), 2)

    def test_a_removed_keyword_names_its_replacement(self):
        with self.assertRaisesRegex(TypeError, "the run's step size"):
            DiscreteStatespaceSystem(id="stepped", A=[[0.9]], B=[[0.1]], C=[[1.0]], step_size=3600)

    def test_an_input_of_another_width_raises_at_the_run_start(self):
        model = twin4build.Model(id="test_ss_width")
        system = DiscreteStatespaceSystem(id="ss_width", A=[[-1e-4]], B=[[1e-4, 1e-4]], C=[[1.0]])
        source = twin4build.ScheduleSystem(id="ss_width_u", weekday_ruleset={"ruleset_default_value": 1.0})
        model.add_connection(source, system, "scheduleValue", "u", input_port_index=0)
        model.load(draw_semantic_model=False, draw_simulation_model=False)
        start = datetime.datetime(2024, 1, 1, tzinfo=datetime.timezone.utc)
        with self.assertRaisesRegex(ValueError, "u is 1 wide, but B has 2 columns"):
            twin4build.Simulator(model).simulate(start_time=start, end_time=start + datetime.timedelta(hours=1),
                                                step_size=600, show_progress_bar=False)


class TestPiecewiseLinearSystem(unittest.TestCase):
    def setUp(self):
        # Create a simple piecewise linear function
        X = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
        Y = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
        self.system = PiecewiseLinearSystem(id="test_piecewise", X=X, Y=Y)

    def test_initialization(self):
        """Test piecewise linear system initialization."""
        self.assertIsNotNone(self.system)
        self.assertEqual(self.system.id, "test_piecewise")

    def _y(self, x, Y=None):
        x = torch.as_tensor(x, dtype=torch.float64)
        _, out = self.system.forward(None, {"x": x}, {"Y": self.system.Y.get() if Y is None else Y}, 600)
        return out["y"]

    def test_get_Y_within_range(self):
        """Linear between two points, constant beyond the first and last."""
        torch.testing.assert_close(self._y([0.5, 1.5, -1.0, 3.0]), torch.tensor([0.5, 0.5, 0.0, 0.0], dtype=torch.float64))

    def test_one_row_of_points_per_copy(self):
        """Batched copies hold their points flat, one copy after another;
        every simulation of a copy reads its row."""
        Y = torch.tensor([0.0, 1.0, 0.0, 2.0, 2.0, 4.0], dtype=torch.float64)
        x = torch.tensor([0.5, 1.5, 0.5, 1.5], dtype=torch.float64)  # 2 simulations x 2 copies
        torch.testing.assert_close(self._y(x, Y), torch.tensor([0.5, 3.0, 0.5, 3.0], dtype=torch.float64))

    def test_the_points_are_checked(self):
        with self.assertRaisesRegex(ValueError, "increase"):
            PiecewiseLinearSystem(id="unsorted", X=[0.0, 2.0, 1.0], Y=[0.0, 1.0, 2.0])
        with self.assertRaisesRegex(ValueError, "3 values, but its size is 2"):
            PiecewiseLinearSystem(id="miscounted", X=[0.0, 1.0], Y=[0.0, 1.0, 2.0])
        with self.assertRaisesRegex(TypeError, "needs the parameter 'Y'"):
            PiecewiseLinearSystem(id="pointless", X=[0.0, 1.0])
        with self.assertRaisesRegex(TypeError, "tb.Variable"):
            PiecewiseLinearSystem(id="bounded", X=[0.0, 1.0], Y=[0.0, 1.0], Y_bounds=(0.0, 1.0))
        flat = PiecewiseLinearSystem(id="flat", X=[0.0, 1.0, 2.0], Y=18.0)
        self.assertEqual(flat.Y.get().tolist(), [18.0, 18.0, 18.0])  # one value fills every point


class TestMaxSystem(unittest.TestCase):
    def setUp(self):
        self.system = MaxSystem(id="test_max")

    def test_initialization(self):
        """Test max system initialization."""
        self.assertIsNotNone(self.system)
        self.assertEqual(self.system.id, "test_max")

    def test_do_step(self):
        """Test max system do_step method."""
        start_time = [datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=tz.UTC)]
        end_time = [datetime.datetime(2023, 1, 1, 1, 0, 0, tzinfo=tz.UTC)]
        step_size = [600]

        # Set inputs - initialize with size parameter for Vector type
        self.system.input["inputs"].initialize(n_t=1, n_s=1, n_v=3)

        self.system.initialize(
            start_time=start_time, end_time=end_time, step_size=step_size
        )
        self.system.input["inputs"].set(torch.tensor([[1.0, 5.0, 3.0]]), i_t=0)

        # Execute
        datetime_val = datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=tz.UTC)
        self.system.do_step(
            second_time=0, date_time=datetime_val, step_size=600, step_index=0
        )

        # Check output
        output = self.system.output["value"].get()
        self.assertIsNotNone(output)
        self.assertEqual(output.item(), 5.0)


class TestWeightedSumSystem(unittest.TestCase):
    def _step(self, system, values):
        start_time = [datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=tz.UTC)]
        end_time = [datetime.datetime(2023, 1, 1, 1, 0, 0, tzinfo=tz.UTC)]
        system.input["inputs"].initialize(n_t=1, n_s=1, n_v=len(values))
        system.initialize(start_time=start_time, end_time=end_time, step_size=[600])
        system.input["inputs"].set(torch.tensor([values]), i_t=0)
        system.do_step(second_time=0, date_time=start_time[0], step_size=600, step_index=0)
        return system.output["value"].get().item()

    def test_plain_sum_without_weights(self):
        self.assertAlmostEqual(self._step(WeightedSumSystem(id="sum"), [1.0, 5.0, 3.0]), 9.0)

    def test_weighted_sum(self):
        system = WeightedSumSystem(weights=[0.5, 0.25, 0.25], id="mean")
        self.assertAlmostEqual(self._step(system, [1.0, 5.0, 3.0]), 2.5)

    def test_one_weight_round_trips_as_a_scalar(self):
        """A saved one-slot weight list comes back as a scalar literal."""
        system = WeightedSumSystem(weights=2.0, id="single")
        self.assertAlmostEqual(self._step(system, [4.0]), 8.0)

    def test_weight_count_must_match_the_slots(self):
        system = WeightedSumSystem(weights=[1.0, 2.0], id="mismatch")
        with self.assertRaises(ValueError):
            self._step(system, [1.0, 2.0, 3.0])

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_the_compiled_forward_is_captured_without_a_host_copy(self):
        """A step captured as a CUDA graph cannot copy from the host: the
        weights are one tensor on the values' device, also when compiled."""
        system = WeightedSumSystem(weights=[0.5, 0.25, 0.25], id="captured")

        def step(values):
            return system.forward(None, {"inputs": values}, None, 600.0)[1]["value"]

        compiled = torch.compile(step, fullgraph=True, dynamic=False)
        values = torch.tensor([[1.0, 5.0, 3.0], [2.0, 2.0, 2.0]], device="cuda")
        compiled(values)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = compiled(values)
        graph.replay()
        torch.testing.assert_close(out.cpu(), torch.tensor([2.5, 2.0]))

    def test_forward_keeps_batch_dims(self):
        system = WeightedSumSystem(weights=[1.0, 2.0], id="batched")
        system.input["inputs"].initialize(n_t=1, n_s=1, n_v=2)
        system.initialize(
            start_time=[datetime.datetime(2023, 1, 1, tzinfo=tz.UTC)],
            end_time=[datetime.datetime(2023, 1, 1, 1, tzinfo=tz.UTC)],
            step_size=[600],
        )
        vals = torch.tensor([[[1.0, 1.0], [2.0, 3.0]]], dtype=torch.float64)  # (n_s=1, n_c=2, n_v=2)
        _, outs = system.forward(None, {"inputs": vals}, {}, 600)
        self.assertEqual(tuple(outs["value"].shape), (1, 2))
        self.assertEqual(outs["value"].tolist(), [[3.0, 8.0]])


class TestOnOffSystem(unittest.TestCase):
    def setUp(self):
        self.system = OnOffSystem(
            id="test_onoff", threshold=0.5, is_off_value=0, is_on_value=100
        )

    def test_initialization(self):
        """Test on/off system initialization."""
        self.assertIsNotNone(self.system)
        self.assertEqual(self.system.id, "test_onoff")

    def test_do_step_on(self):
        """Test on/off system when signal is ON (> 0.5)."""
        start_time = [datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=tz.UTC)]
        end_time = [datetime.datetime(2023, 1, 1, 1, 0, 0, tzinfo=tz.UTC)]
        step_size = [600]
        self.system.initialize(
            start_time=start_time, end_time=end_time, step_size=step_size
        )

        # Set inputs: value is the on value, criteriaValue is what we compare
        self.system.input["value"].set(torch.tensor([100.0]), i_t=0)
        self.system.input["criteriaValue"].set(torch.tensor([0.8]), i_t=0)

        # Execute
        datetime_val = datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=tz.UTC)
        self.system.do_step(
            second_time=0, date_time=datetime_val, step_size=600, step_index=0
        )

        # Check output should be value (100) since criteriaValue (0.8) >= threshold (0.5)
        output = self.system.output["value"].get()
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
        start_time = [datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=tz.UTC)]
        end_time = [datetime.datetime(2023, 1, 1, 1, 0, 0, tzinfo=tz.UTC)]
        step_size = [600]
        self.system.initialize(
            start_time=start_time, end_time=end_time, step_size=step_size
        )

        # Set input
        self.system.input["value"].set(torch.tensor([42.0]), i_t=0)

        # Execute
        datetime_val = datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=tz.UTC)
        self.system.do_step(
            second_time=0, date_time=datetime_val, step_size=600, step_index=0
        )

        # Check output equals input
        output = self.system.output["value"].get()
        self.assertEqual(output.item(), 42.0)


if __name__ == "__main__":
    unittest.main()
