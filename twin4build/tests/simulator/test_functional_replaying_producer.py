"""Functional simulation with a replaying component in a loop.

A controller in playback mode (``replays_data()``) outputs its recorded
command whatever its inputs say; the functional engine keeps it out of the
traced cone and records its output on the exogenous tape.  The public
functional simulation recorded that tape by stepping the producer
*closure* of the replaying component, walked into its producers -- the
zone whose temperature it reads, a functional component -- and refused
("exogenous producer closure reaches functional component ...").  The
closure of a replaying component is the component itself.
"""

# Standard library imports
import datetime
import unittest

# Third party imports
import torch
from dateutil import tz

# Local application imports
import twin4build as tb
import twin4build.core as core
import twin4build.utils.types as tps
from twin4build.tests.model.test_batching_vector_slots import END, START, STEP, Sink, _n_t

tb._IS_TESTING = True


class Replayer(core.System):
    """Reads the sink's state (a loop) but outputs a recorded series."""

    def __init__(self, series, **kwargs):
        super().__init__(**kwargs)
        self.input = {"x": tps.Scalar()}
        self.output = {"y": tps.Vector()}
        self.series = torch.as_tensor(series, dtype=torch.float64)
        self._config = {"parameters": []}

    @property
    def config(self):
        return self._config

    def replays_data(self) -> bool:
        return True

    def initialize(self, start_time, end_time, step_size):
        n_t = _n_t(start_time, end_time, step_size)
        self.input["x"].initialize(n_t=n_t, n_s=len(start_time), n_c=self.n_c)
        self.output["y"].initialize(n_t=n_t, n_s=len(start_time), n_c=self.n_c, n_v=1)

    PARAM_NAMES = ()

    def forward(self, x, inputs, params, sample_time):  # never traced: replays
        raise AssertionError("a replaying component is not traced")

    def do_step(self, second_time, date_time, step_size, step_index):
        value = self.series[step_index].reshape(1, 1, 1)
        self.output["y"]._set(value, i_t=step_index)


class TestFunctionalReplayingProducer(unittest.TestCase):
    def test_replaying_producer_in_a_loop(self):
        series = [1.0, 2.0, 0.5, 3.0]
        model = tb.Model(id="functional_replaying_producer")
        sink = Sink(id="sink")
        replayer = Replayer(series, id="replayer")
        model.add_connection(replayer, sink, "y", "u", output_port_index=0, input_port_index=0)
        model.add_connection(sink, replayer, "w", "x")  # the loop the closure walk hit
        model.load(draw_semantic_model=False, draw_simulation_model=False)
        simulator = tb.Simulator(model, execution_mode="functional", execution_backend="eager")
        simulator.simulate(start_time=START, end_time=END, step_size=STEP, show_progress_bar=False)
        hist = sink.output["w"].history()
        got = hist.reshape(hist.shape[0], -1)[:, 0].detach().cpu().clone()
        # The recorded value of step t drives step t, as on the estimator's
        # tape: acc <- acc + 3 * y, the cumulative sum of the series, tripled.
        # (The object engine cuts the loop with a one-step delay and lags
        # by one sample; that is its cycle rule, not the tape's.)
        expected = 3.0 * torch.cumsum(torch.tensor(series[: len(got)], dtype=got.dtype), dim=0)
        torch.testing.assert_close(got, expected)


if __name__ == "__main__":
    unittest.main()
