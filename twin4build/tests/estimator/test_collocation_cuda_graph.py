"""CUDA Graph regression coverage for collocation callback bundles."""

import pytest
import torch

from benchmarks import common


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_collocation_captures_and_replays_all_callback_bundles():
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
        assert bundle["captured"], (name, bundle)
        assert bundle["enabled"], (name, bundle)
        assert bundle["fallback_reason"] is None, (name, bundle)
    # The wrapper's capture setup checks eager/captured output parity for both
    # the initial and a perturbed input before reporting captured=True.
    assert stats["gradient_jacobian_bundle"]["replays"] >= 1
