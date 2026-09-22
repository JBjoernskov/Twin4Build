"""The public functional simulation captures its rollout on CUDA.

``functional_rollout_tape`` (the rollout ``Simulator.simulate`` captures
whole under ``execution_backend="cuda_graph"``) did not prepare the
routes for the device, so the captured step copied index tensors from the
host: "Cannot copy between CPU and CUDA tensors during CUDA graph
capture".  Runs only where CUDA is available.
"""

# Standard library imports
import unittest

# Third party imports
import torch

# Local application imports
import twin4build as tb
from twin4build.tests.model.test_batching_vector_slots import END, START, STEP, build, sink_histories

tb._IS_TESTING = True


@unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
class TestFunctionalSimulationCudaGraph(unittest.TestCase):
    def test_captured_rollout_matches_eager(self):
        model, _ = build()
        batched = model.batch_components()
        batched.load(draw_semantic_model=False, draw_simulation_model=False)
        batched.to(device="cuda", dtype=torch.float64)
        kwargs = dict(start_time=START, end_time=END, step_size=STEP, show_progress_bar=False)
        tb.Simulator(batched, execution_mode="functional", execution_backend="eager").simulate(**kwargs)
        eager = sink_histories(model, batched)
        tb.Simulator(batched, execution_mode="functional", execution_backend="cuda_graph").simulate(**kwargs)
        captured = sink_histories(model, batched)
        for cid in eager:
            torch.testing.assert_close(captured[cid], eager[cid], msg=cid)


if __name__ == "__main__":
    unittest.main()
