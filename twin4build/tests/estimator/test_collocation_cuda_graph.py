"""CUDA Graph regression coverage for collocation callback bundles."""

# Standard library imports
import unittest

# Third party imports
import torch

# Local application imports
from benchmarks import common


class TestCollocationCudaGraph(unittest.TestCase):
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA unavailable")
    def test_collocation_captures_and_replays_all_callback_bundles(self):
        config = common.BenchmarkConfig(mode="smoke")
        setup = common.batched_estimation_problem(1, config)
        result, _seconds, _estimator = common._run_estimation(
            setup,
            config,
            "cuda",
            common.ESTIMATION_METHODS["ipopt-collocation"],
            1,
        )

        stats = result["collocation_audit"]["cuda_graph"]
        for name in (
            "forward_bundle",
            "gradient_jacobian_bundle",
            "lagrangian_hessian",
        ):
            bundle = stats[name]
            self.assertTrue(bundle["captured"], (name, bundle))
            self.assertTrue(bundle["enabled"], (name, bundle))
            self.assertIsNone(bundle["fallback_reason"], (name, bundle))
        # The wrapper's capture setup checks eager/captured output parity for both
        # the initial and a perturbed input before reporting captured=True.
        self.assertGreaterEqual(stats["gradient_jacobian_bundle"]["replays"], 1)


if __name__ == "__main__":
    unittest.main()
