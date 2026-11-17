"""Small reusable direct-CUDA-Graph callable wrapper."""

from __future__ import annotations

import inspect
import time
import warnings
import weakref

import torch
from torch.utils import _pytree as pytree

from twin4build import settings as _settings
from twin4build.utils.logger import LOGGER


# The ``cuda_graph_sync_phases`` setting (T4B_CUDA_GRAPH_SYNC_PHASES=1)
# synchronizes the device after every phase of a capture (warmup, record,
# replay, parity check).  CUDA reports an asynchronous fault at the next
# synchronizing call, so without this an illegal memory access raised by, say,
# the replay only surfaces at the caller's synchronize and the phase is lost.
# Costs one extra sync per phase.
def _sync_phases_enabled() -> bool:
    return _settings.value("cuda_graph_sync_phases")


# Most recent phase entered by any capture wrapper or bundle caller in this
# process.  Read it after a CUDA fault to learn which step launched the
# faulting kernel (with cuda_graph_sync_phases on, the preceding phases
# completed without a device error).
LAST_PHASE: str | None = None


#: Above this many component-steps (functional components x time steps) a
#: capture of the *eager* step is warned about: it records every kernel of
#: the rollout (thousands per time step for the eager step, a few hundred
#: for the compiled one) at roughly 8 kB of host memory each; a six-day pass
#: of a 2000-component model recorded about three million nodes, 43 GB.
LARGE_EAGER_CAPTURE_COMPONENT_STEPS = 500_000


def warn_if_large_eager_capture(n_components: int, n_steps: int, compiled: bool) -> bool:
    """Warn when a CUDA-graph capture of the eager step is large.

    Capturing the eager step is legitimate where nothing can compile it (a
    Windows torch without Triton) and cheap on small models; on a large one
    the record runs to tens of gigabytes of host memory.  Returns whether a
    warning was issued.
    """
    if compiled:
        return False
    size = int(n_components) * int(n_steps)
    if size <= LARGE_EAGER_CAPTURE_COMPONENT_STEPS:
        return False
    warnings.warn(
        "Capturing the eager functional step of a large model (%d functional "
        "components x %d time steps): the CUDA graph records every kernel of "
        "the rollout and its host memory grows with them (about 8 kB per "
        "kernel; a 2000-component, 864-step pass recorded 43 GB). On a "
        "Triton-capable torch set compile_step='auto' or True so the capture "
        "records the compiled step instead; otherwise consider "
        "execution_backend='eager' or a shorter window." % (n_components, n_steps),
        RuntimeWarning,
        stacklevel=3,
    )
    return True


def mark_phase(name: str) -> None:
    """Record ``name`` as the current phase, syncing first if enabled."""
    global LAST_PHASE
    if _sync_phases_enabled():
        torch.cuda.synchronize()
    LAST_PHASE = name


def phase_note() -> str:
    return (
        f"CUDA graph phase: {LAST_PHASE} (per-phase device sync "
        f"{'on' if _sync_phases_enabled() else 'off'}; set "
        f"T4B_CUDA_GRAPH_SYNC_PHASES=1 to attribute asynchronous faults)"
    )


#: ``c`` in the bound ``c * n * u * scale`` of :func:`check_replay_matches_eager`.
REPLAY_BOUND_FACTOR = 8


def largest_dimension(*trees) -> int:
    """The largest dimension of any tensor in the pytrees ``trees``, at least 1.

    The ``n`` the callers of :func:`check_replay_matches_eager` pass: the
    longest sum a function computes runs along an axis of the tensors it
    reads and returns (the state width, the input width, a batch or window
    length, a captured rollout's number of steps).
    """
    n = 1
    for tree in trees:
        for leaf in pytree.tree_leaves(tree):
            if isinstance(leaf, torch.Tensor) and leaf.dim() > 0:
                n = max(n, *leaf.shape)
    return n


def check_replay_matches_eager(replay, eager, *, n: int, label: str):
    """Check a CUDA-graph replay against an eager run of the same function.

    The replay and the eager run do the same arithmetic but may sum in
    different orders: an atomic accumulation (``index_add_``,
    ``scatter_add_``, the backward of an indexing) adds in the order its
    threads arrive, and the capture may pick other kernels than the eager
    call.  In float32 the two then differ by a few ulp in some elements.

    The difference allowed is the standard bound on floating-point
    summation: two orders of a sum of ``n`` terms differ by at most about
    ``2 * n * u * sum(|terms|)``, ``u = finfo(dtype).eps / 2`` the unit
    roundoff.  Element-wise, in each output tensor::

        |replay - eager| <= c * n * u * scale,    c = REPLAY_BOUND_FACTOR = 8

    ``n`` is the length of the longest sum the function computes; the
    callers pass :func:`largest_dimension` of its input and output tensors.
    ``scale`` is the largest finite magnitude of that eager output tensor,
    at least the dtype's smallest normal.  The assumptions:

    * an output is at most four such sums deep (``c = 8`` is four times the
      factor 2 of one sum);
    * a sum's terms are at most of the size of the largest value of the
      tensor it ends up in, so ``scale`` stands for ``sum(|terms|)``; taking
      the tensor's largest value instead of each element's own covers an
      element whose terms cancel to near zero;
    * no sum is longer than ``n``.

    Relative to ``scale`` the bound is ``c * n * u``: at ``n = 100`` about
    ``4.8e-5`` in float32 and ``8.9e-14`` in float64.  A wrong capture (a
    stale address, an input it does not track) is off by about the values
    themselves, orders of magnitude above the bound while ``n`` stays well
    below ``1 / (c * u)``.  NaNs must sit at the same elements, infinities
    must be equal, and so must integer and boolean outputs.

    Raises :class:`AssertionError` naming the largest difference observed
    and the one allowed.  Otherwise logs both at debug level, for the output
    closest to its bound, and returns them as ``(observed, allowed)``.
    """
    replay_leaves, replay_spec = pytree.tree_flatten(replay)
    eager_leaves, eager_spec = pytree.tree_flatten(eager)
    headline = f"{label} differs from eager output"
    if replay_spec != eager_spec:
        raise AssertionError(
            f"{headline}: the outputs are structured {replay_spec}, eager {eager_spec}"
        )
    n = max(1, int(n))
    # the output closest to its bound: (observed / allowed, observed, allowed, index)
    closest = (0.0, 0.0, 0.0, None)
    for index, (out, ref) in enumerate(zip(replay_leaves, eager_leaves)):
        where = f"{headline}: output {index}"
        tensors = isinstance(out, torch.Tensor), isinstance(ref, torch.Tensor)
        if not all(tensors):
            if any(tensors) or out != ref:
                raise AssertionError(f"{where} is {out!r}, eager {ref!r}")
            continue
        if out.shape != ref.shape or out.dtype != ref.dtype:
            raise AssertionError(
                f"{where} is {out.dtype} {tuple(out.shape)}, "
                f"eager {ref.dtype} {tuple(ref.shape)}"
            )
        if out.numel() == 0:
            continue
        if not (ref.is_floating_point() or ref.is_complex()):
            if not torch.equal(out, ref):
                raise AssertionError(f"{where} ({ref.dtype}) is not equal")
            continue
        finfo = torch.finfo(ref.dtype)
        u = finfo.eps / 2
        largest = float(torch.where(torch.isfinite(ref), ref.abs(), 0).max())
        scale = max(largest, finfo.tiny)
        allowed = REPLAY_BOUND_FACTOR * n * u * scale
        same = (out == ref) | (torch.isnan(out) & torch.isnan(ref))
        diff = torch.where(same, 0, (out - ref).abs()).nan_to_num(nan=float("inf"))
        observed = float(diff.max())
        if observed > allowed:
            over = int((diff.double() > allowed).sum())
            raise AssertionError(
                f"{where} ({ref.dtype}, shape {tuple(ref.shape)}): "
                f"max |replay - eager| = {observed:.3e} exceeds the allowed "
                f"{allowed:.3e} (= {REPLAY_BOUND_FACTOR} * n {n} * u {u:.3e} "
                f"* scale {scale:.3e}); {over} of {ref.numel()} elements over"
            )
        if observed / allowed >= closest[0]:
            closest = (observed / allowed, observed, allowed, index)
    ratio, observed, allowed, index = closest
    LOGGER.debug(
        "%s matches eager output: max |replay - eager| %.3e, allowed %.3e "
        "(output %s, %.1e of the bound, n %d)",
        label,
        observed,
        allowed,
        index,
        ratio,
        n,
    )
    return observed, allowed


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
        def phase(name: str) -> None:
            # Synchronizing here attributes an asynchronous fault to the
            # phase that launched it (the previous one) instead of to the
            # caller's next synchronize.
            mark_phase(f"capture:{name}")
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
            n = largest_dimension(self.static_inputs, reference_output)
            check_replay_matches_eager(
                self.static_output,
                reference_output,
                n=n,
                label="Direct CUDA Graph replay",
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
                check_replay_matches_eager(
                    self.static_output,
                    probe_output,
                    n=n,
                    label="Direct CUDA Graph replay at changed inputs",
                )
                for target, value in zip(self.static_inputs, inputs):
                    target.copy_(value)
                self.graph.replay()
            phase("done")
        except Exception as exc:
            if hasattr(exc, "add_note"):
                exc.add_note(phase_note())
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
