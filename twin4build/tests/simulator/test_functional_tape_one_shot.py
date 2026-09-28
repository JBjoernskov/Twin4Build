"""The exogenous tape is recorded in one shot where the window is on hand.

The public functional simulation filled the tape one time step at a
time: for every exogenous key and step, a history lookup or a stepped
producer closure and a routed write, 270 000 stepped calls and minutes on
a translated building.  A source whose history covers the window (a data
sensor, a schedule, a replaying controller handing its replayed history
over through ``replayed_output_history``) now fills its column with one
routed tensor; only sources without one are still stepped.
"""

# Standard library imports
import unittest
from unittest import mock

# Third party imports
import torch

# Local application imports
import twin4build as tb
from twin4build.simulator import _functional_simulation as fs
from twin4build.tests.model.test_batching_vector_slots import END, START, STEP, Sink
from twin4build.tests.simulator.test_functional_replaying_producer import Replayer

tb._IS_TESTING = True


class HandingReplayer(Replayer):
    """Hands its replayed series over, so the tape needs no stepping."""

    def replayed_output_history(self, output_name, n_t):
        if output_name != "y":
            return None
        return self.series[:n_t].reshape(n_t, 1, 1, 1).to(torch.float64)


def run(replayer_cls):
    model = tb.Model(id=f"tape_one_shot_{replayer_cls.__name__}")
    sink = Sink(id="sink")
    replayer = replayer_cls([1.0, 2.0, 0.5, 3.0], id="replayer")
    model.add_connection(replayer, sink, "y", "u", output_port_index=0, input_port_index=0)
    model.add_connection(sink, replayer, "w", "x")
    model.load(draw_semantic_model=False, draw_simulation_model=False)
    simulator = tb.Simulator(model, execution_mode="functional", execution_backend="eager")
    calls = {"stepped": 0}
    original = fs.FunctionalSimulationSession._step_external

    def counting(self, component, step, done, visiting):
        calls["stepped"] += 1
        return original(self, component, step, done, visiting)

    with mock.patch.object(fs.FunctionalSimulationSession, "_step_external", counting):
        simulator.simulate(start_time=START, end_time=END, step_size=STEP, show_progress_bar=False)
    hist = sink.output["w"].history()
    return hist.reshape(hist.shape[0], -1)[:, 0].detach().cpu().clone(), calls["stepped"]


class TestTapeOneShot(unittest.TestCase):
    def test_handed_over_history_needs_no_stepping_and_matches(self):
        stepped_result, stepped_calls = run(Replayer)
        one_shot_result, one_shot_calls = run(HandingReplayer)
        torch.testing.assert_close(one_shot_result, stepped_result)
        self.assertGreater(stepped_calls, 0)
        self.assertEqual(one_shot_calls, 0)


if __name__ == "__main__":
    unittest.main()
