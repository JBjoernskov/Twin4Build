"""Focused lifecycle and replay tests for direct CUDA Graph wrappers."""

import gc
import weakref
from unittest import mock

import pytest
import torch

import twin4build.utils._cuda_graph as cuda_graph
from twin4build.estimator import _collocation


class _FakeGraph:
    def __init__(self):
        self.reset_calls = 0

    def reset(self):
        self.reset_calls += 1


def test_bound_method_does_not_retain_owner():
    class Owner:
        def call(self, value):
            return value

    owner = Owner()
    wrapper = cuda_graph.CudaGraphCallable(owner.call)
    owner_ref = weakref.ref(owner)
    del owner
    gc.collect()

    assert owner_ref() is None
    with pytest.raises(ReferenceError):
        wrapper.fn
    wrapper.close()
    wrapper.close()


def test_close_defers_reset_during_capture_and_is_idempotent():
    cuda_graph._deferred_graphs.clear()
    wrapper = cuda_graph.CudaGraphCallable(lambda value: value)
    graph = _FakeGraph()
    wrapper.graph = graph

    with mock.patch.object(cuda_graph, "_is_capturing", return_value=True):
        wrapper.close()
        wrapper.close()
    assert graph.reset_calls == 0
    assert graph in cuda_graph._deferred_graphs

    with (
        mock.patch.object(cuda_graph, "_is_capturing", return_value=False),
        mock.patch.object(torch.cuda, "synchronize") as synchronize,
    ):
        cuda_graph._drain_deferred_graphs()
    assert graph.reset_calls == 1
    synchronize.assert_called_once()
    assert not cuda_graph._deferred_graphs


def test_direct_bundle_never_falls_back_after_capture_invalidation():
    calls = {"eager": 0}

    def eager(value):
        calls["eager"] += 1
        return value

    class InvalidGraph:
        def __init__(self, fn):
            pass

        def __call__(self, *inputs):
            raise cuda_graph.CudaGraphCaptureInvalidated("invalidated")

        def close(self):
            pass

    bundle = _collocation._DirectCudaGraph(eager, True, "test")
    with (
        mock.patch.object(_collocation, "CudaGraphCallable", InvalidGraph),
        pytest.raises(cuda_graph.CudaGraphCaptureInvalidated),
    ):
        bundle(torch.tensor(1.0))
    assert calls["eager"] == 0


def test_direct_bundle_ordinary_setup_error_falls_back_once():
    calls = {"eager": 0}

    def eager(value):
        calls["eager"] += 1
        return value + 1

    class IncompatibleGraph:
        def __init__(self, fn):
            pass

        def __call__(self, *inputs):
            raise RuntimeError("capture unsupported")

        def close(self):
            pass

    bundle = _collocation._DirectCudaGraph(eager, True, "test")
    with mock.patch.object(_collocation, "CudaGraphCallable", IncompatibleGraph):
        assert bundle(torch.tensor(1.0)) == 2.0
    assert calls["eager"] == 1
    assert not bundle.stats["enabled"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_default_capture_is_one_warmup_and_one_record():
    calls = {"n": 0}

    def fn(value):
        calls["n"] += 1
        return value * 2

    wrapper = cuda_graph.CudaGraphCallable(fn)
    x = torch.tensor([0.25, -0.5], device="cuda", dtype=torch.float64)
    with wrapper:
        torch.testing.assert_close(wrapper(x), x * 2)
    assert calls["n"] == 2
    wrapper_probe = cuda_graph.CudaGraphCallable(fn, verify_input_tracking=True)
    with wrapper_probe:
        wrapper_probe(x)
    assert calls["n"] == 5


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_capture_replay_parity_and_owned_snapshot():
    wrapper = cuda_graph.CudaGraphCallable(
        lambda x, scale: (x.sin() * scale, x.square() + scale)
    )
    x1 = torch.linspace(-0.7, 0.9, 17, device="cuda", dtype=torch.float64)
    x2 = x1 + 0.13
    scale1 = torch.tensor(1.4, device="cuda", dtype=torch.float64)
    scale2 = torch.tensor(0.6, device="cuda", dtype=torch.float64)

    with wrapper:
        first = tuple(value.clone() for value in wrapper(x1, scale1))
        second = tuple(value.clone() for value in wrapper(x2, scale2))
        expected = (x2.sin() * scale2, x2.square() + scale2)
        torch.testing.assert_close(second, expected)
        torch.testing.assert_close(first, (x1.sin() * scale1, x1.square() + scale1))
        assert wrapper.capture_count == 1
        assert wrapper.replay_count == 1
        wrapper.reset()
        assert not wrapper.closed
        recaptured = tuple(value.clone() for value in wrapper(x1, scale1))
        torch.testing.assert_close(
            recaptured, (x1.sin() * scale1, x1.square() + scale1)
        )
        assert wrapper.capture_count == 2
    assert wrapper.closed


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_fixed_shape_collocation_bundle_capture_replay_parity():
    dtype, device = torch.float64, torch.device("cuda")
    A = torch.tensor([[0.3, -0.2, 0.7], [0.5, 0.4, -0.1]], dtype=dtype, device=device)

    def callbacks(z, sigma, lam):
        az = A @ z
        objective = z.pow(4).sum()
        gradient = 4.0 * z.pow(3)
        defects = az.sin()
        jacobian = az.cos().unsqueeze(1) * A
        hessian = torch.diag(12.0 * sigma * z.square())
        hessian = hessian + A.T @ torch.diag(-lam * az.sin()) @ A
        return torch.cat(
            (
                objective.reshape(1),
                gradient,
                defects,
                jacobian.reshape(-1),
                hessian.reshape(-1),
            )
        )

    bundle = _collocation._DirectCudaGraph(callbacks, True, "collocation test")
    z1 = torch.tensor([0.2, -0.4, 0.7], dtype=dtype, device=device)
    z2 = torch.tensor([-0.3, 0.1, 0.9], dtype=dtype, device=device)
    sigma1 = torch.tensor(0.8, dtype=dtype, device=device)
    sigma2 = torch.tensor(1.1, dtype=dtype, device=device)
    lam1 = torch.tensor([0.2, -0.5], dtype=dtype, device=device)
    lam2 = torch.tensor([-0.1, 0.6], dtype=dtype, device=device)

    first = bundle(z1, sigma1, lam1).clone()
    second = bundle(z2, sigma2, lam2).clone()
    torch.testing.assert_close(first, callbacks(z1, sigma1, lam1))
    torch.testing.assert_close(second, callbacks(z2, sigma2, lam2))
    assert bundle.stats["captured"]
    assert bundle.stats["replays"] == 1
    assert bundle.stats["capture_seconds"] > 0
    assert bundle.stats["replay_seconds"] >= 0
    bundle.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_capture_records_phases_and_notes_the_failing_phase(monkeypatch):
    monkeypatch.setenv(cuda_graph.SYNC_PHASES_ENV, "1")
    wrapper = cuda_graph.CudaGraphCallable(lambda x: x * 2.0)
    x = torch.ones(4, device="cuda", dtype=torch.float64)
    wrapper(x)
    assert wrapper.last_phase == "done"
    assert cuda_graph.LAST_PHASE == "capture:done"

    calls = {"n": 0}

    def flaky(x):
        # Warmup passes; the record phase raises, so the note must say "record".
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("boom during record")
        return x * 2.0

    wrapper = cuda_graph.CudaGraphCallable(flaky)
    with pytest.raises(RuntimeError, match="boom") as info:
        wrapper(x)
    assert wrapper.last_phase == "record"
    notes = getattr(info.value, "__notes__", [])
    assert any("phase: capture:record" in note for note in notes)
    assert cuda_graph.LAST_PHASE == "capture:record"
