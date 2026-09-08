"""The compiled functional step (issue #134).

The fused matrix-exponential variant and the ``compile_step`` option are
checked everywhere; the compiled rollout itself needs Inductor's CUDA backend
(Triton), which Windows torch wheels do not ship, so that part skips there.
"""

import os

import pytest
import torch

import twin4build as tb
from twin4build.simulator.simulator import _has_triton
from twin4build.systems.utils.discrete_statespace_system import (
    _expm_ss,
    _expm_ss_fused,
)

os.environ.setdefault("T4B_BENCHMARK_MODE", "smoke")


def _rc_block(scale):
    return torch.tensor(
        [[-scale, 0.2 * scale, 0.1], [0.0, -0.5 * scale, 0.3], [0.0, 0.0, 0.0]],
        dtype=torch.float64,
    )


def test_fused_exponential_matches_bmm_form_across_the_scaling_range():
    for scale in (0.02, 0.27, 2.4, 27.0, 3e3, 2.6e5):
        M = _rc_block(scale)
        torch.testing.assert_close(_expm_ss_fused(M), _expm_ss(M), rtol=2e-12, atol=2e-14)
        torch.testing.assert_close(_expm_ss_fused(M), torch.matrix_exp(M), rtol=2e-12, atol=2e-14)
    # batched leading dims, as the components use it
    Mb = torch.stack([_rc_block(0.27), _rc_block(27.0)]).unsqueeze(1)  # (2, 1, 3, 3)
    torch.testing.assert_close(_expm_ss_fused(Mb), _expm_ss(Mb), rtol=2e-12, atol=2e-14)


def test_fused_exponential_derivatives_match():
    M = _rc_block(0.27)
    direction = torch.tensor(
        [[0.3, -0.2, 0.1], [0.0, -0.4, 0.2], [0.0, 0.0, 0.0]], dtype=torch.float64
    )
    _, jvp_fused = torch.func.jvp(_expm_ss_fused, (M,), (direction,))
    _, jvp_ref = torch.func.jvp(torch.matrix_exp, (M,), (direction,))
    torch.testing.assert_close(jvp_fused, jvp_ref, rtol=2e-10, atol=2e-12)


def test_expm_is_not_fused_outside_compile():
    # Plain eager keeps the cuBLAS form; the fused form is a compile-only path.
    assert torch.compiler.is_compiling() is False
    M = _rc_block(2.4)
    torch.testing.assert_close(_expm_ss(M), torch.matrix_exp(M), rtol=2e-12, atol=2e-14)


def test_compile_step_option_validation():
    with pytest.raises(ValueError, match="compile_step"):
        tb.Simulator(None, compile_step="yes")
    if not _has_triton():
        with pytest.raises(ValueError, match="Triton"):
            tb.Simulator(None, compile_step=True)
    sim = tb.Simulator(None, execution_mode="functional", execution_backend="eager")
    assert sim.compile_step == "auto"
    assert sim.step_compilation_active("cpu") is False
    sim_off = tb.Simulator(None, compile_step=False)
    assert sim_off.step_compilation_active("cuda") is False
    if torch.cuda.is_available():
        assert sim.step_compilation_active("cuda") is _has_triton()


@pytest.mark.skipif(
    not (torch.cuda.is_available() and _has_triton()),
    reason="needs CUDA and a Triton-capable torch (Inductor CUDA backend)",
)
def test_compiled_rollout_matches_eager_step():
    """Values and gradients of a compiled shooting rollout equal the eager ones."""
    common = pytest.importorskip("benchmarks.common")
    BenchmarkConfig = common.BenchmarkConfig
    _estimation_window = common._estimation_window
    batched_estimation_problem = common.batched_estimation_problem
    seed_everything = common.seed_everything
    import twin4build.utils.types as tps

    config = BenchmarkConfig(mode="smoke")
    results = {}

    class Done(Exception):
        pass

    def _probe(self, method, n_cores, options):
        obj = self._functional_objective
        theta = torch.as_tensor(
            self._x0_norm, dtype=tps.float_dtype(), device=self._device
        ).unsqueeze(0)
        results[self.simulator.compile_step] = obj.batched_value_and_grad(theta + 0.05)
        raise Done

    original_dispatch = tb.Estimator._dispatch_solve
    tb.Estimator._dispatch_solve = _probe
    try:
        for compile_step in (False, True):
            # A fresh problem per pass: estimate() consumes the parameter objects.
            seed_everything(config.seed)
            setup = batched_estimation_problem(1, config)
            model = setup["model"]
            model.to("cuda", torch.float64)
            sim = tb.Simulator(
                model, execution_mode="functional", execution_backend="eager", compile_step=compile_step
            )
            try:
                tb.Estimator(sim).estimate(
                    parameters=setup["parameters"],
                    measurements=setup["measurements"],
                    method=("scipy", "SLSQP", "ad"),
                    options={"maxiter": 1},
                    **_estimation_window(config),
                )
            except Done:
                pass
    finally:
        tb.Estimator._dispatch_solve = original_dispatch
    v0, g0 = results[False]
    v1, g1 = results[True]
    torch.testing.assert_close(v1, v0, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(g1, g0, rtol=1e-11, atol=1e-11)
