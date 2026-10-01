"""Per-step CUDA graphs: capture one step, replay it along the rollout.

``execution_backend="cuda_graph"`` captures a whole rollout (and, in the
estimator, its backward pass) as one graph.  Inside a capture every kernel
launch becomes a node, so a loop of 864 calls of the compiled step records
864 copies of its kernels, forward and backward: the graph's own memory
grows with the number of steps and sits outside the allocator (about 6 GB
for 864 steps of a 2700-meta building on top of 2 GB of tensors).

With ``cuda_graph_scope="step"`` one graph holds ONE step's forward kernels
and one graph its reverse-mode kernels (the step's vector-Jacobian product
with the cotangents of the next state and of the step's outputs).  The
rollout replays the forward graph from a Python loop and keeps the states;
its gradient is an adjoint sweep that replays the backward graph from the
last step to the first.  Both are wrapped in one :class:`torch.autograd.Function`
so the estimator's objectives differentiate through the rollout exactly as
before, without capturing anything at rollout level.  Memory is one step's
two graphs plus the saved states; time is one graph launch per step.

The step is the compiled transform-mode step (:meth:`FunctionalModel.compiled_step`,
or the compiled batched step for a batch of parameter vectors); the
theta-only matrices the rollout hoists (:meth:`FunctionalModel.step_constants`)
enter the graphs as static inputs set once per rollout, and their cotangents
come back out, so the outer autograd continues from them to theta.
"""
from __future__ import annotations

import time
from typing import Callable, List, Optional, Sequence, Tuple

import torch
from torch.utils import _pytree as pytree


class StepGraph:
    """One captured graph for a fixed-shape, tensor-only ``fn(*static, *dynamic)``.

    ``static`` inputs are set once per rollout (:meth:`set_static`: the
    parameters and the hoisted matrices), ``dynamic`` inputs change every
    call (the state, the exogenous row, the cotangents).  The first call
    after :meth:`set_static` captures: an eager warm-up on a side stream,
    the recording, one replay and a replay-against-eager parity check.
    Outputs are the graph's static buffers: the caller clones what it keeps.
    """

    def __init__(self, fn: Callable, n_static: int, name: str = "step"):
        self.fn = fn
        self.n_static = int(n_static)
        self.name = name
        self.graph: Optional[torch.cuda.CUDAGraph] = None
        self.static: Optional[Tuple[torch.Tensor, ...]] = None
        self.dynamic: Optional[Tuple[torch.Tensor, ...]] = None
        self.outputs: Optional[Tuple[torch.Tensor, ...]] = None
        self.capture_seconds = 0.0
        self.replay_count = 0

    def set_static(self, *static: torch.Tensor) -> None:
        if len(static) != self.n_static:
            raise ValueError(f"{self.name}: expected {self.n_static} static inputs, got {len(static)}")
        if self.static is None:
            self.static = tuple(torch.empty_like(s) for s in static)
        for target, value in zip(self.static, static):
            if target.shape != value.shape or target.dtype != value.dtype or target.device != value.device:
                raise ValueError(f"{self.name}: static input shape, dtype or device changed after capture")
            target.copy_(value.detach())

    def _capture(self, dynamic: Sequence[torch.Tensor]) -> None:
        started = time.perf_counter()
        self.dynamic = tuple(torch.empty_like(d) for d in dynamic)
        for target, value in zip(self.dynamic, dynamic):
            target.copy_(value.detach())
        inputs = tuple(self.static) + tuple(self.dynamic)
        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup_stream):
            reference = tuple(o.detach().clone() for o in self.fn(*inputs))
        torch.cuda.current_stream().wait_stream(warmup_stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.outputs = tuple(self.fn(*inputs))
        self.graph.replay()
        torch.cuda.synchronize()
        for out, ref in zip(self.outputs, reference):
            torch.testing.assert_close(
                out, ref, rtol=1e-9, atol=1e-11, equal_nan=True,
                msg=f"{self.name}: CUDA graph replay differs from eager output",
            )
        self.capture_seconds += time.perf_counter() - started

    def __call__(self, *dynamic: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        if self.static is None:
            raise RuntimeError(f"{self.name}: set_static before calling")
        if self.graph is None:
            self._capture(dynamic)
            return self.outputs
        if len(dynamic) != len(self.dynamic):
            raise ValueError(f"{self.name}: dynamic input count changed after capture")
        for target, value in zip(self.dynamic, dynamic):
            if target.shape != value.shape or target.dtype != value.dtype:
                raise ValueError(f"{self.name}: dynamic input shape or dtype changed after capture")
            target.copy_(value)
        self.graph.replay()
        self.replay_count += 1
        return self.outputs


class StepGraphs:
    """The forward and the adjoint graph of one step function.

    ``step(y, theta, u, constants)`` is the (compiled) transform-mode step;
    ``constants`` is the pytree the rollout hoists (``None`` for the batched
    step, which rebuilds its matrices).  ``static`` = ``(theta, *leaves)``.
    """

    def __init__(self, step: Callable, spec, n_leaves: int, name: str = "step"):
        self.step = step
        self.spec = spec
        self.n_leaves = int(n_leaves)
        n_static = 1 + self.n_leaves

        def unflatten(leaves):
            return pytree.tree_unflatten(list(leaves), spec) if spec is not None else None

        def fwd_fn(theta, *rest):
            leaves, (y, u) = rest[: self.n_leaves], rest[self.n_leaves :]
            y_next, meas = step(y, theta, u, unflatten(leaves))
            return y_next, meas

        def adj_fn(theta, *rest):
            leaves, (y, u, ybar_next, mbar) = rest[: self.n_leaves], rest[self.n_leaves :]
            y_ = y.detach().requires_grad_(True)
            theta_ = theta.detach().requires_grad_(True)
            leaves_ = [l.detach().requires_grad_(True) for l in leaves]
            with torch.enable_grad():
                y_next, meas = step(y_, theta_, u, unflatten(leaves_))
                grads = torch.autograd.grad(
                    (y_next, meas), (y_, theta_, *leaves_), grad_outputs=(ybar_next, mbar), allow_unused=True
                )
            return tuple(g if g is not None else torch.zeros_like(x) for g, x in zip(grads, (y_, theta_, *leaves_)))

        self.fwd = StepGraph(fwd_fn, n_static, name=f"{name}:forward")
        self.adj = StepGraph(adj_fn, n_static, name=f"{name}:adjoint")

    def set_static(self, theta: torch.Tensor, leaves: Sequence[torch.Tensor]) -> None:
        self.fwd.set_static(theta, *leaves)
        self.adj.set_static(theta, *leaves)


class _StepGraphRollout(torch.autograd.Function):
    """The rollout as one autograd node: forward replays the step graph and
    keeps the states, backward sweeps the adjoint graph from the last step
    to the first, accumulating the cotangents of theta and of the hoisted
    matrices."""

    @staticmethod
    def forward(ctx, graphs: StepGraphs, y0, theta, tape, *leaves):
        graphs.set_static(theta, leaves)
        y = y0.detach()
        states = [y]
        rows = []
        with torch.no_grad():
            for t in range(tape.shape[0]):
                y_next, meas = graphs.fwd(y, tape[t])
                y = y_next.clone()
                states.append(y)
                rows.append(meas.clone())
        states_t = torch.stack(states)
        outputs = torch.stack(rows) if rows else torch.zeros((0,) + tuple(graphs.fwd.outputs[1].shape), dtype=tape.dtype, device=tape.device)
        ctx.graphs = graphs
        ctx.save_for_backward(theta, tape, states_t, *leaves)
        return states_t, outputs

    @staticmethod
    def backward(ctx, sbar, mbar):
        theta, tape, states, *leaves = ctx.saved_tensors
        graphs = ctx.graphs
        graphs.set_static(theta, leaves)
        n_t = tape.shape[0]
        ybar = torch.zeros_like(states[0])
        thetabar = torch.zeros_like(theta)
        leafbars = [torch.zeros_like(l) for l in leaves]
        with torch.no_grad():
            for t in range(n_t - 1, -1, -1):
                if sbar is not None:
                    ybar = ybar + sbar[t + 1]
                out = graphs.adj(states[t], tape[t], ybar, mbar[t])
                ybar = out[0].clone()
                thetabar += out[1]
                for i in range(len(leaves)):
                    leafbars[i] += out[2 + i]
            if sbar is not None:
                ybar = ybar + sbar[0]
        return (None, ybar, thetabar, None, *leafbars)


def step_graph_rollout(functional_model, y0, theta, tape, *, batched: bool = False, kind: Optional[str] = None):
    """Roll ``functional_model`` over ``tape`` with per-step CUDA graphs.

    Returns ``(states (n_t + 1, ...), outputs (n_t, ...))``, both
    differentiable w.r.t. ``theta`` (and ``y0``).  ``kind``:

    * ``"scalar"`` (default): ``y0 (D_aug,)``, ``theta (n_theta,)``,
      ``tape (n_t, n_exogenous)``; the compiled step with the hoisted
      matrices.
    * ``"batched"`` (or ``batched=True``): a batch of parameter starts,
      ``y0 (B, D_aug)``, ``theta (B, n_theta)``, one shared tape; the
      eager vmapped batched step.
    * ``"windows"``: one parameter vector over a batch of windows, ``y0 (P,
      D_aug)``, ``theta (n_theta,)``, ``tape (n_t, P, n_exogenous)``; the
      compiled window step with the hoisted matrices shared.
    * ``"rows"``: rows with their own state, parameters and inputs, ``y0
      (N, D_aug)``, ``theta (N, n_theta)``, ``tape (n_t, N, n_exogenous)``.

    The graphs are cached on the model per kind and input shapes.
    """
    functional_model.prepare_routes(theta.device)
    kind = kind or ("batched" if batched else "scalar")
    if kind == "batched":
        # the eager vmap, not ``compiled_batched_step``: ``torch.compile``
        # over ``vmap`` is numerically wrong on torch 2.11 (see
        # ``FunctionalModel.window_step``); captured per step, its kernel
        # count costs nothing at replay
        bstep = functional_model.batched_step

        def step(y, th, u, constants):
            return bstep(y, th, u)

        leaves, spec = [], None
    elif kind == "rows":
        rstep = functional_model.rows_step

        def step(y, th, u, constants):
            return rstep(y, th, u)

        leaves, spec = [], None
    elif kind == "windows":
        step = functional_model.window_step
        constants = functional_model.step_constants(theta)
        leaves, spec = pytree.tree_flatten(constants)
    elif kind == "scalar":
        step = functional_model.compiled_step
        constants = functional_model.step_constants(theta)
        leaves, spec = pytree.tree_flatten(constants)
    else:
        raise ValueError(f"unknown step kind {kind!r}")
    key = (kind, tuple(y0.shape), tuple(theta.shape), tuple(tape.shape[1:]), tuple(tuple(l.shape) for l in leaves))
    cache = functional_model.__dict__.setdefault("_step_graphs", {})
    graphs = cache.get(key)
    if graphs is None:
        graphs = StepGraphs(step, spec, len(leaves), name=f"{kind} step")
        cache[key] = graphs
    return _StepGraphRollout.apply(graphs, y0, theta, tape, *leaves)
