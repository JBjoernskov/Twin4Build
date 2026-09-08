"""Small reusable direct-CUDA-Graph callable wrapper."""

from __future__ import annotations

import inspect
import os
import time
import weakref

import torch

# Set T4B_CUDA_GRAPH_SYNC_PHASES=1 to synchronize the device after every phase
# of a capture (warmup, record, replay, parity check).  CUDA reports an
# asynchronous fault at the next synchronizing call, so without this an
# illegal memory access raised by, say, the replay only surfaces at the
# caller's synchronize and the phase is lost.  Costs one extra sync per phase.
SYNC_PHASES_ENV = "T4B_CUDA_GRAPH_SYNC_PHASES"


def _sync_phases_enabled() -> bool:
    return os.environ.get(SYNC_PHASES_ENV, "").strip().lower() in {"1", "true", "yes"}


class CudaGraphCaptureInvalidated(RuntimeError):
    """The CUDA context can no longer safely run an eager fallback."""


_capture_invalidated = False
# A graph must not be destroyed/reset while *any* stream is being captured.
# Keep deferred graphs alive until a later, safe lifecycle operation.
_deferred_graphs = []


def is_cuda_graph_capture_invalidated(exc: BaseException) -> bool:
    """Recognize CUDA's invalidated/prior-capture error spellings."""
    text = f"{type(exc).__name__}: {exc}".lower()
    markers = (
        "cudaerrorstreamcaptureinvalidated",
        "stream capture invalidated",
        "capture status: invalidated",
        "capture was invalidated",
        "prior error during capture",
        "previous error during capture",
    )
    return isinstance(exc, CudaGraphCaptureInvalidated) or any(
        marker in text for marker in markers
    )


def cuda_graph_capture_is_invalidated() -> bool:
    """Whether this process has observed an invalidated CUDA capture."""
    return _capture_invalidated


def _is_capturing() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        return bool(torch.cuda.is_current_stream_capturing())
    except Exception:
        # A failed status query is not evidence that resetting is safe.
        return True


def _reset_graph(graph, *, synchronize: bool) -> bool:
    """Reset one graph iff no stream capture is active."""
    global _capture_invalidated
    if graph is None or _is_capturing():
        return False
    try:
        if synchronize:
            torch.cuda.synchronize()
        graph.reset()
    except Exception as exc:
        if is_cuda_graph_capture_invalidated(exc):
            _capture_invalidated = True
        # Lifecycle methods and destructors are intentionally best effort.
        return False
    return True


def _drain_deferred_graphs() -> None:
    if not _deferred_graphs or _is_capturing():
        return
    pending = list(_deferred_graphs)
    _deferred_graphs.clear()
    for graph in pending:
        if not _reset_graph(graph, synchronize=True):
            _deferred_graphs.append(graph)


class CudaGraphCallable:
    """Capture and replay a fixed-shape tensor-only callable.

    The object owns its graph and is an idempotent context manager. ``close``
    synchronizes and resets only outside capture. Bound methods are held
    weakly, avoiding ``owner -> wrapper -> bound method -> owner`` cycles.

    First-call cost is one eager warmup plus the capture recording, then a
    replay-vs-eager parity check. Extra warmups and a second eager probe at a
    perturbed input are optional because each is a full reverse-mode rollout
    on estimation problems.
    """

    def __init__(self, fn, warmup_calls=1, verify_input_tracking=False):
        if inspect.ismethod(fn) and fn.__self__ is not None:
            self._fn = None
            self._weak_method = weakref.WeakMethod(fn)
        else:
            self._fn = fn
            self._weak_method = None
        if int(warmup_calls) < 1:
            raise ValueError("warmup_calls must be at least 1")
        self.warmup_calls = int(warmup_calls)
        self.verify_input_tracking = bool(verify_input_tracking)
        self.graph = None
        self.static_inputs = None
        self.static_output = None
        self.closed = False
        self.capture_count = 0
        self.replay_count = 0
        self.capture_seconds = 0.0
        self.replay_seconds = 0.0
        # Last capture phase entered; on failure the phase the fault surfaced in.
        self.last_phase: str | None = None

    @property
    def fn(self):
        fn = self._fn if self._weak_method is None else self._weak_method()
        if fn is None:
            raise ReferenceError("CUDA Graph callable owner has been released")
        return fn

    def _capture(self, inputs):
        global _capture_invalidated
        if _capture_invalidated:
            raise CudaGraphCaptureInvalidated(
                "A prior CUDA stream capture was invalidated; eager execution "
                "and additional capture are unsafe in this process."
            )
        _drain_deferred_graphs()
        started = time.perf_counter()
        sync_phases = _sync_phases_enabled()

        def phase(name: str) -> None:
            # Synchronizing here attributes an asynchronous fault to the
            # phase that launched it (the previous one) instead of to the
            # caller's next synchronize.
            if sync_phases:
                torch.cuda.synchronize()
            self.last_phase = name

        try:
            phase("static-inputs")
            self.static_inputs = tuple(torch.empty_like(value) for value in inputs)
            for target, value in zip(self.static_inputs, inputs):
                target.copy_(value)
            phase("warmup")
            warmup_stream = torch.cuda.Stream()
            warmup_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(warmup_stream):
                for _ in range(self.warmup_calls):
                    reference_output = self.fn(*self.static_inputs)
            torch.cuda.current_stream().wait_stream(warmup_stream)
            phase("record")
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.static_output = self.fn(*self.static_inputs)
            phase("replay")
            self.graph.replay()
            phase("parity-check")
            torch.testing.assert_close(
                self.static_output,
                reference_output,
                rtol=1e-9,
                atol=1e-11,
                equal_nan=True,
                msg="Direct CUDA Graph replay differs from eager output",
            )
            # One warmup + one capture record is enough to certify replay
            # parity. A second eager AD at a perturbed input (the old
            # input-tracking probe) doubled first-call cost on long
            # reverse-mode rollouts and is covered by later solver steps
            # and by tests that call the wrapper twice.
            if self.verify_input_tracking:
                probe_inputs = tuple(
                    (
                        value + (index + 1) * 1e-4
                        if value.is_floating_point() or value.is_complex()
                        else value.clone()
                    )
                    for index, value in enumerate(inputs)
                )
                probe_output = self.fn(*probe_inputs)
                for target, value in zip(self.static_inputs, probe_inputs):
                    target.copy_(value)
                self.graph.replay()
                torch.testing.assert_close(
                    self.static_output,
                    probe_output,
                    rtol=1e-9,
                    atol=1e-11,
                    equal_nan=True,
                    msg="Direct CUDA Graph replay does not track changed inputs",
                )
                for target, value in zip(self.static_inputs, inputs):
                    target.copy_(value)
                self.graph.replay()
            phase("done")
        except Exception as exc:
            if hasattr(exc, "add_note"):
                exc.add_note(
                    f"CUDA graph capture phase: {self.last_phase} "
                    f"(per-phase device sync {'on' if sync_phases else 'off'}; "
                    f"set {SYNC_PHASES_ENV}=1 to attribute asynchronous faults)"
                )
            if is_cuda_graph_capture_invalidated(exc):
                _capture_invalidated = True
            self.close()
            if _capture_invalidated:
                raise CudaGraphCaptureInvalidated(
                    "CUDA stream capture was invalidated; eager fallback is "
                    "unsafe in this process."
                ) from exc
            raise
        self.capture_count += 1
        self.capture_seconds += time.perf_counter() - started
        return self.static_output

    def __call__(self, *inputs):
        if self.closed:
            raise RuntimeError("CudaGraphCallable is closed")
        if self.graph is None:
            return self._capture(inputs)
        if len(inputs) != len(self.static_inputs):
            raise ValueError("CUDA Graph input count changed after capture")
        for target, value in zip(self.static_inputs, inputs):
            if (
                target.shape != value.shape
                or target.dtype != value.dtype
                or target.device != value.device
            ):
                raise ValueError("CUDA Graph input shape, dtype, or device changed")
            target.copy_(value)
        started = time.perf_counter()
        self.graph.replay()
        self.replay_count += 1
        self.replay_seconds += time.perf_counter() - started
        return self.static_output

    def _release_graph(self) -> None:
        graph, self.graph = self.graph, None
        if graph is not None and not _reset_graph(graph, synchronize=True):
            _deferred_graphs.append(graph)
        self.static_inputs = None
        self.static_output = None
        _drain_deferred_graphs()

    def reset(self) -> None:
        """Release a captured graph, allowing the next call to recapture."""
        if self.closed:
            return
        self._release_graph()

    def close(self) -> None:
        """Permanently release this callable; repeated calls are harmless."""
        if self.closed:
            return
        self._release_graph()
        self.closed = True
        self._fn = None
        self._weak_method = None

    def __enter__(self):
        if self.closed:
            raise RuntimeError("CudaGraphCallable is closed")
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
        return False

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
