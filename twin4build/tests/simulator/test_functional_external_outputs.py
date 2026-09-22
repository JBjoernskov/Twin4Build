"""Requested outputs outside the traced cone cost one kernel per step, not one each.

The public functional simulation asks the functional model for every
output port of every component.  A port of a component outside the
traced cone (a data sensor, a controller in playback) is "external": the
step returns zeros there and the session fills the history in afterwards.
The step used to allocate one zeros tensor per external output per step,
thousands on a translated building, which dominated its kernel count and
its captured graph.  All external outputs are one zeros tensor now.
"""

# Standard library imports
import collections
import unittest

# Third party imports
import torch
from torch.utils._python_dispatch import TorchDispatchMode

# Local application imports
import twin4build as tb
from twin4build.tests.model.test_batching_vector_slots import END, START, STEP, Sink, _n_t
from twin4build.tests.simulator.test_functional_replaying_producer import Replayer

tb._IS_TESTING = True


class OpCounter(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.kinds = collections.Counter()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.kinds[str(func)] += 1
        return func(*args, **(kwargs or {}))


class TestExternalOutputs(unittest.TestCase):
    def test_external_outputs_are_one_zeros_block(self):
        model = tb.Model(id="functional_external_outputs")
        sink = Sink(id="sink")
        replayers = [Replayer([1.0, 2.0, 0.5, 3.0], id=f"replayer{i}") for i in range(3)]
        model.add_connection(replayers[0], sink, "y", "u", output_port_index=0, input_port_index=0)
        for r in replayers:
            model.add_connection(sink, r, "w", "x")
        model.load(draw_semantic_model=False, draw_simulation_model=False)
        model.initialize(start_time=[START], end_time=[END], step_size=STEP)
        simulator = tb.Simulator(model, execution_mode="functional", execution_backend="eager")

        def zeros_per_step(outputs):
            _, fm = simulator.build_functional_model(step_size=STEP, outputs=outputs)
            fm.prepare_routes(torch.device("cpu"))
            y0 = torch.zeros(fm.D + int(fm._n_feedback), dtype=torch.float64)
            theta = torch.zeros(0, dtype=torch.float64)
            exo = torch.zeros(int(sum(fm._exogenous_widths)), dtype=torch.float64)
            counter = OpCounter()
            with counter:
                _, meas = fm.F_aug(y0, theta, exo, transform_mode=True)
            return fm, meas, counter.kinds["aten.zeros.default"]

        _, _, zeros_without = zeros_per_step([(sink, "w")])
        fm, meas, zeros_with = zeros_per_step([(r, "y") for r in replayers] + [(sink, "w")])
        kinds = [m[0] for m in fm.meas_sources]
        self.assertEqual(kinds.count("external"), 3)
        self.assertEqual(kinds.count("fresh"), 1)
        blocks, perm = fm._meas_groups
        self.assertEqual([b[0] for b in blocks].count("external_all"), 1)
        self.assertEqual(next(b for b in blocks if b[0] == "external_all")[1], 3)
        self.assertEqual(meas.numel(), fm.n_meas)
        self.assertEqual(meas[:3].tolist(), [0.0, 0.0, 0.0])  # externals, in order, zeros
        # three external outputs cost one zeros kernel, not three
        self.assertLessEqual(zeros_with - zeros_without, 1)


if __name__ == "__main__":
    unittest.main()
