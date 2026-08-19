"""CUDA-graph capture for a repeatedly-called tensor function.

WHY
---
The collocation exact-Hessian callback is **CPU-dispatch-bound, not GPU-bound**
(see issue #122).  Measured on an A100 at ``maxiter=100``, per Hessian call:

    CPU dispatch  1259 ms
    GPU compute    156 ms   -> the GPU is busy 12.4% of the call

and the exact Hessian achieves 12.6% of its Amdahl ceiling.  Those being the
same number is the point: the speedup shortfall *is* the idle fraction.  The
GPU work itself is healthy -- 58.5% of device time is fp64 GEMM on tensor cores
-- there is just too little of it against ~18,000 ATen dispatches per call.

``torch.compile`` cannot help: Dynamo fails to trace ``vmap(jacfwd(jacrev(.)))``
on doubly-wrapped ``GradTrackingTensor(BatchedTensor(...))`` even with
``backend="eager"``, so this is a tracing limitation rather than a missing
compiler.  A CUDA graph sidesteps tracing entirely: it records the eager kernel
stream once and replays it as a single launch, collapsing every dispatch.

SAFETY
------
Replay executes the *identical recorded kernels*, so results are bit-identical
by construction -- not merely close.  That is asserted on the first replay and
capture is abandoned if it does not hold, so a silent numerical change is not
among the failure modes.  Any capture failure falls back to eager permanently
with one warning.

REQUIREMENTS on the wrapped function
------------------------------------
* static shapes and dtypes across calls (checked);
* no host synchronisation inside -- ``.item()``, ``.cpu()``, ``float()``, or a
  Python branch on a tensor value.  Any of those raises during capture, which
  is caught and falls back;
* no data-dependent shapes;
* it must read its inputs from the tensors it is given (they become static
  buffers whose *contents* change between replays, never their addresses).
"""

from __future__ import annotations

import os
import warnings as _warnings
from typing import Callable, Dict, List, Optional

import torch

from twin4build.utils.logger import LOGGER


#: Every runner appends its outcome here, so a caller can ask what actually
#: happened instead of inferring it from timings.  Capture silently falling
#: back to eager is indistinguishable from "the graph did not help", which is
#: precisely the ambiguity that made the first A/B unreadable.
CAPTURE_LOG: List[str] = []


#: Populated only when TWIN4BUILD_GRAPH_DEBUG is set: the individual pieces of
#: the Hessian region, so a diagnostic can try capturing them SEPARATELY.
#: cudaErrorStreamCaptureInvalidated names the symptom, never the offending op,
#: so bisection is the only way to localise it.
DEBUG_PARTS: Dict[str, object] = {}


def capture_status() -> List[str]:
    """What each runner did this process: 'captured' or why it fell back."""
    return list(CAPTURE_LOG)


def cuda_graphs_enabled() -> bool:
    """``TWIN4BUILD_CUDA_GRAPH=0`` disables capture everywhere."""
    return os.environ.get("TWIN4BUILD_CUDA_GRAPH", "1") not in ("0", "false", "False")


class CudaGraphRunner:
    """Run ``fn`` eagerly a few times, then capture and replay it.

    Parameters
    ----------
    fn : callable
        ``fn(**tensors) -> Tensor``.  Must satisfy the requirements above.
    name : str
        Used in log messages only.
    warmup : int
        Eager calls (on a side stream) before capture.  CUDA graph capture
        requires the allocator and any lazily-initialised kernels to have
        settled; 3 is the conventional number.
    enabled : bool
        Set False to force the eager path (used for A/B measurement).

    Notes
    -----
    The returned tensor is the graph's **static output buffer** -- it is
    overwritten by the next replay.  Callers must consume it (copy, or move to
    host) before calling again.  This is deliberate: cloning per call would add
    back a device allocation on the path whose overhead is the point.
    """

    def __init__(
        self,
        fn: Callable[..., torch.Tensor],
        *,
        name: str = "fn",
        warmup: int = 3,
        enabled: bool = True,
        capture_error_mode: str = "global",
    ):
        self._fn = fn
        self._name = name
        self._warmup_target = max(1, int(warmup))
        self._enabled = bool(enabled) and cuda_graphs_enabled()
        self._capture_error_mode = capture_error_mode
        self._graph: Optional[torch.cuda.CUDAGraph] = None
        self._static_in: Optional[Dict[str, torch.Tensor]] = None
        self._static_out: Optional[torch.Tensor] = None
        self._warmups = 0
        self._disabled = False
        self._replays = 0

    # -- introspection, for tests and logging ------------------------------
    @property
    def captured(self) -> bool:
        return self._graph is not None

    @property
    def disabled(self) -> bool:
        return self._disabled

    @property
    def replays(self) -> int:
        return self._replays

    @property
    def static_inputs(self) -> Optional[Dict[str, torch.Tensor]]:
        """The real inputs from the last call -- what a diagnostic replays."""
        return self._static_in

    def _fallback(self, reason: str, **tensors) -> torch.Tensor:
        if not self._disabled:
            self._disabled = True
            msg = (
                f"CUDA graph capture unavailable for {self._name} ({reason}) "
                "-- running eager. This costs speed, not correctness."
            )
            CAPTURE_LOG.append(f"{self._name}: FELL BACK -- {reason}")
            # warnings.warn, NOT LOGGER: LOGGER output is suppressed by default,
            # so logging this would make a silent fallback indistinguishable
            # from a graph that captured but did not help -- the exact
            # ambiguity that wasted a benchmark run.
            _warnings.warn(msg, RuntimeWarning, stacklevel=3)
            LOGGER.warning(msg)
        return self._fn(**tensors)

    def _bind(self, tensors: Dict[str, torch.Tensor]) -> None:
        """Allocate the static input buffers from the first call's shapes."""
        self._static_in = {
            k: torch.empty_like(v) for k, v in tensors.items()
        }

    def _copy_in(self, tensors: Dict[str, torch.Tensor]) -> bool:
        """Copy call values into the static buffers; False on a shape change."""
        for k, v in tensors.items():
            buf = self._static_in.get(k)
            if buf is None or buf.shape != v.shape or buf.dtype != v.dtype:
                return False
            buf.copy_(v)
        return True

    def __call__(self, **tensors: torch.Tensor) -> torch.Tensor:
        if self._disabled or not self._enabled:
            return self._fn(**tensors)

        dev = next(iter(tensors.values())).device
        if dev.type != "cuda":
            # Not an error: CPU runs have nothing to capture.  Silent, because
            # this is the common case and a warning would be noise.
            self._disabled = True
            CAPTURE_LOG.append(f"{self._name}: skipped (device is {dev.type})")
            return self._fn(**tensors)

        if self._static_in is None:
            self._bind(tensors)
        if not self._copy_in(tensors):
            return self._fallback("input shape or dtype changed", **tensors)

        # -- warm-up phase: run eager on a side stream ----------------------
        if self._graph is None and self._warmups < self._warmup_target:
            self._warmups += 1
            try:
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    out = self._fn(**self._static_in)
                torch.cuda.current_stream().wait_stream(s)
                return out
            except Exception as exc:  # noqa: BLE001
                return self._fallback(f"warm-up failed: {exc}", **tensors)

        # -- capture --------------------------------------------------------
        if self._graph is None:
            try:
                # Reference computed BEFORE capture, from the same static
                # inputs, so the post-capture replay can be checked against it.
                reference = self._fn(**self._static_in).clone()
                torch.cuda.synchronize()

                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(
                    graph, capture_error_mode=self._capture_error_mode
                ):
                    static_out = self._fn(**self._static_in)
                graph.replay()
                torch.cuda.synchronize()

                if static_out.shape != reference.shape:
                    return self._fallback("captured output shape differs", **tensors)
                # Bit-identical, not approximately equal: replay runs the very
                # kernels that were recorded.  Anything else means the capture
                # did not record what we think it did.
                if not torch.equal(static_out, reference):
                    delta = (static_out - reference).abs().max().item()
                    return self._fallback(
                        f"replay differs from eager (max |delta| = {delta:.3e})",
                        **tensors,
                    )
                self._graph = graph
                self._static_out = static_out
                CAPTURE_LOG.append(f"{self._name}: captured")
                _warnings.warn(
                    f"CUDA graph CAPTURED for {self._name} "
                    f"({static_out.numel()} outputs) -- replay is one launch. "
                    "Reported so a no-op speedup can be told apart from a "
                    "silent fallback.",
                    UserWarning, stacklevel=3,
                )
                LOGGER.config(
                    "CUDA graph captured for %s: %d-element output, replay is "
                    "one launch instead of the eager dispatch stream.",
                    self._name, static_out.numel(),
                )
            except Exception as exc:  # noqa: BLE001
                return self._fallback(f"capture failed: {exc}", **tensors)

        self._graph.replay()
        self._replays += 1
        return self._static_out
