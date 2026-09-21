"""The routes prepared for CUDA can still be applied to CPU tags.

``prepare_routes(device)`` moves the routes' index tensors, including a
route's target order, to the rollout's device.  The coupling-structure walk
(``index_coupling``, the block trust-region solver's first step) applies
the routes to a CPU tag tensor and failed at ``selected[order]`` with
"indices should be either on cpu or on the same device as the indexed
tensor".  Runs only where CUDA is available.
"""

# Standard library imports
import unittest

# Third party imports
import torch

# Local application imports
import twin4build as tb
from twin4build.simulator._functional import FunctionalModel
from twin4build.tests.model.test_batching_vector_slots import END, START, STEP, build


@unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
class TestRoutesPreparedForCuda(unittest.TestCase):
    def test_permuted_route_applies_to_cpu_tags(self):
        """A route whose pairs cover the target in another order carries a
        prepared order tensor on the device; CPU tags must still route."""
        cuda = torch.device("cuda")
        target = torch.tensor([2, 1, 0])
        route = (
            slice(None),  # the whole (scalar) output
            torch.arange(3, device=cuda),  # source instances
            target.to(cuda),  # target instances, permuted
            3,
            False,
            FunctionalModel._target_order_of(target, 3, cuda),
        )
        tags = torch.arange(1.0, 4.0)[:, None]
        mapped = FunctionalModel._apply_routes(FunctionalModel.__new__(FunctionalModel), tags, [route])
        self.assertEqual(mapped.device.type, "cpu")
        self.assertEqual(mapped.reshape(-1).tolist(), [3.0, 2.0, 1.0])

    def test_index_coupling_after_routes_on_cuda(self):
        """The solver's order on a CUDA run: a rollout prepares the routes
        for the device, then the structure walk runs; same blocks as on CPU."""
        model, _ = build()
        batched = model.batch_components()
        batched.load(draw_semantic_model=False, draw_simulation_model=False)
        batched.to(device="cuda", dtype=torch.float64)
        batched.initialize(start_time=[START], end_time=[END], step_size=STEP)
        simulator = tb.Simulator(batched, execution_mode="functional", execution_backend="eager")
        _, reference = simulator.build_functional_model(step_size=STEP)
        reference.prepare_routes(torch.device("cpu"))
        theta_block, column_block, n_blocks = reference.index_coupling()
        _, functional_model = simulator.build_functional_model(step_size=STEP)
        functional_model.prepare_routes(torch.device("cuda"))
        theta_again, column_again, n_again = functional_model.index_coupling()
        self.assertEqual(n_blocks, n_again)
        self.assertEqual(list(theta_block), list(theta_again))
        self.assertEqual(list(column_block), list(column_again))


if __name__ == "__main__":
    unittest.main()
