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
When the exogenous tape requires grad (the optimizer's trajectory decision
variables override columns of it), the rollout returns the tape's gradient
from a second adjoint graph that also returns each step's input-row
cotangent (#236); the forward graph is the same either way.

The step is the compiled transform-mode step (:meth:`FunctionalModel.compiled_step`,
or the compiled batched step for a batch of parameter vectors); the
theta-only matrices the rollout hoists (:meth:`FunctionalModel.derived_values`)
enter the graphs as static inputs set once per rollout, and their cotangents
come back out, so the outer autograd continues from them to theta.
"""
from __future__ import annotations

import time
from typing import Callable, List, Optional, Sequence, Tuple

import torch
from torch.utils import _pytree as pytree

from twin4build.simulator._functional import STEP_KINDS
from twin4build.utils._cuda_graph import check_replay_matches_eager, largest_dimension


def initialization_count(functional_model) -> int:
    """How often the model ``functional_model`` steps has been initialized
    (``SimulationModel.initialization_count``)."""
    model = functional_model.model
    model = getattr(model, "_simulation_model", None) or model
    return int(getattr(model, "initialization_count", 0))


class StepGraph:
    """One captured graph for a fixed-shape, tensor-only ``fn(*static, *dynamic)``.

    ``static`` inputs are set once per rollout (:meth:`set_static`: the
    parameters and the hoisted matrices), ``dynamic`` inputs change every
    call (the state, the exogenous row, the cotangents).  The first call
    after :meth:`set_static` captures: an eager warm-up on a side stream,
    the recording, one replay and a replay-against-eager parity check
    (:func:`~twin4build.utils._cuda_graph.check_replay_matches_eager`).
    Outputs are the graph's static buffers: the caller clones what it keeps.
    ``joined`` names the outputs that are several tensors laid end to end:
    their length is no sum's length, and the parity check's ``n`` leaves it
    out (the tensors they join have their own shapes among the inputs).
    """

    def __init__(self, fn: Callable, n_static: int, name: str = "step", joined: Sequence[int] = ()):
        self.fn = fn
        self.n_static = int(n_static)
        self.name = name
        self.joined = frozenset(joined)
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
        # the replay may sum in another order than the eager run (atomic
        # accumulation, other kernels): it must agree within the bound on
        # reordered sums, n the widest dimension of the step's tensors
        summed = [r for i, r in enumerate(reference) if i not in self.joined]
        check_replay_matches_eager(
            self.outputs, reference, n=largest_dimension(inputs, summed),
            label=f"{self.name}: CUDA graph replay",
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
    The adjoint's outputs are ``(ybar, thetabar, leafbar)``, and with
    ``tape_grad`` the cotangent of the step's input row ``u`` after them;
    ``leafbar`` is the floating leaves' cotangents as one vector, in the
    leaves' order (:meth:`leafbars` splits it), so the sweep adds one tensor
    per step, not one per leaf.  :meth:`adjoint` captures the variant a
    backward needs, once.
    """

    def __init__(self, step: Callable, spec, n_leaves: int, name: str = "step"):
        self.step = step
        self.spec = spec
        self.n_leaves = int(n_leaves)
        self.name = name
        n_static = 1 + self.n_leaves

        def unflatten(leaves):
            return pytree.tree_unflatten(list(leaves), spec) if spec is not None else None

        def fwd_fn(theta, *rest):
            leaves, (y, u) = rest[: self.n_leaves], rest[self.n_leaves :]
            y_next, meas = step(y, theta, u, unflatten(leaves))
            return y_next, meas

        def adj_fn(theta, *rest, tape_grad=False):
            leaves, (y, u, ybar_next, mbar) = rest[: self.n_leaves], rest[self.n_leaves :]
            y_ = y.detach().requires_grad_(True)
            theta_ = theta.detach().requires_grad_(True)
            u_ = u.detach().requires_grad_(True) if tape_grad else u
            # An integer leaf (a sparse bilinear term's positions) takes no
            # gradient: only the floating leaves are differentiated.
            floating = [l.is_floating_point() for l in leaves]
            leaves_ = [l.detach().requires_grad_(True) if f else l for l, f in zip(leaves, floating)]
            diff = [l for l, f in zip(leaves_, floating) if f]
            wrt = (y_, theta_, *diff) + ((u_,) if tape_grad else ())
            with torch.enable_grad():
                y_next, meas = step(y_, theta_, u_, unflatten(leaves_))
                grads = torch.autograd.grad((y_next, meas), wrt, grad_outputs=(ybar_next, mbar), allow_unused=True)
            grads = [g if g is not None else torch.zeros_like(x) for g, x in zip(grads, wrt)]
            leafbar = torch.cat([g.reshape(-1) for g in grads[2 : 2 + len(diff)]]) if diff else theta.new_zeros(0)
            return (grads[0], grads[1], leafbar) + ((grads[-1],) if tape_grad else ())

        self.fwd = StepGraph(fwd_fn, n_static, name=f"{name}:forward")
        self._adj_fn = adj_fn
        self._n_static = n_static
        self._adjoints = {}

    def adjoint(self, tape_grad: bool) -> StepGraph:
        """The adjoint graph, with (``tape_grad``) or without the input-row cotangent."""
        graph = self._adjoints.get(bool(tape_grad))
        if graph is None:
            fn = self._adj_fn if not tape_grad else (lambda *a: self._adj_fn(*a, tape_grad=True))
            # leafbar (output 2) is the leaves' cotangents laid end to end
            graph = StepGraph(fn, self._n_static, name=f"{self.name}:adjoint{':tape' if tape_grad else ''}", joined=(2,))
            self._adjoints[bool(tape_grad)] = graph
        return graph

    def set_static(self, theta: torch.Tensor, leaves: Sequence[torch.Tensor]) -> None:
        self.fwd.set_static(theta, *leaves)

    @staticmethod
    def leafbars(leafbar: Optional[torch.Tensor], leaves: Sequence[torch.Tensor]) -> List[Optional[torch.Tensor]]:
        """The adjoint's ``leafbar`` vector as one cotangent per leaf
        (``None`` for an integer leaf, zeros when nothing was swept)."""
        floating = [l for l in leaves if l.is_floating_point()]
        if leafbar is None:
            pieces = iter([torch.zeros_like(l) for l in floating])
        else:
            pieces = iter([p.reshape(l.shape) for p, l in zip(leafbar.split([l.numel() for l in floating]), floating)])
        return [next(pieces) if l.is_floating_point() else None for l in leaves]


class _StepGraphRollout(torch.autograd.Function):
    """The rollout as one autograd node: forward replays the step graph and
    keeps the states, backward sweeps the adjoint graph from the last step
    to the first, accumulating the cotangents of theta and of the hoisted
    matrices, and, when the tape needs a gradient, collecting each step's
    input-row cotangent as the tape's gradient."""

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
        tape_grad = bool(ctx.needs_input_grad[3])
        adj = ctx.graphs.adjoint(tape_grad)
        adj.set_static(theta, *leaves)
        n_t = tape.shape[0]
        ybar = torch.zeros_like(states[0])
        thetabar = torch.zeros_like(theta)
        tapebar = torch.zeros_like(tape) if tape_grad else None
        leafbar = None
        with torch.no_grad():
            for t in range(n_t - 1, -1, -1):
                if sbar is not None:
                    ybar = ybar + sbar[t + 1]
                out = adj(states[t], tape[t], ybar, mbar[t])
                ybar = out[0].clone()
                thetabar += out[1]
                leafbar = out[2].clone() if leafbar is None else leafbar.add_(out[2])
                if tape_grad:
                    tapebar[t] = out[-1]
            if sbar is not None:
                ybar = ybar + sbar[0]
        return (None, ybar, thetabar, tapebar, *StepGraphs.leafbars(leafbar, leaves))


def step_graph_rollout(functional_model, y0, theta, tape, *, batched: bool = False, kind: Optional[str] = None):
    """Roll ``functional_model`` over ``tape`` with per-step CUDA graphs.

    Returns ``(states (n_t + 1, ...), outputs (n_t, ...))``, both
    differentiable w.r.t. ``theta`` and ``y0``, and w.r.t. ``tape`` when it
    requires grad (the backward then replays the adjoint graph that also
    returns the input rows' cotangents).  ``kind``:

    * ``"scalar"`` (default): ``y0 (D_aug,)``, ``theta (n_theta,)``,
      ``tape (n_t, n_exogenous)``.
    * ``"batched"`` (or ``batched=True``): a batch of parameter starts,
      ``y0 (B, D_aug)``, ``theta (B, n_theta)``, one shared tape.
    * ``"windows"``: one parameter vector over a batch of windows, ``y0 (P,
      D_aug)``, ``theta (n_theta,)``, ``tape (n_t, P, n_exogenous)``.
    * ``"rows"``: rows with their own state, parameters and inputs, ``y0
      (N, D_aug)``, ``theta (N, n_theta)``, ``tape (n_t, N, n_exogenous)``.

    Every kind captures the compiled step of its kind
    (:meth:`FunctionalModel.step`) with the values hoisted out of it
    (:meth:`FunctionalModel.derived_values`: shared for one theta, per row for
    a batch) as the graph's static inputs.

    The graphs are cached on the model per kind and input shapes.
    """
    functional_model.prepare_routes(theta.device)
    kind = kind or ("batched" if batched else "scalar")
    # Step graphs run only with the compiled step (``step_graph_active``), so
    # every kind captures a compiled step: the batched kinds compile the vmap
    # of ``F_aug`` (correct since #241's scatter assembly), a few fused
    # kernels per step instead of the eager vmap's thousands
    if kind not in STEP_KINDS:
        raise ValueError(f"unknown step kind {kind!r}")
    step = functional_model.step(kind, compiled=True)
    if kind == "rows":
        rows_step = step

        def step(y, th, u, constants):
            # a step's rows may come as (B, P, n_exogenous), one broadcast
            # tape for B rows over P windows
            return rows_step(y, th, u.reshape(-1, u.shape[-1]), constants)

    # hoisted once per rollout: shared for one theta, per row for a batch
    constants = functional_model.derived_values(theta)
    leaves, spec = pytree.tree_flatten(constants)
    key = (kind, tuple(y0.shape), tuple(theta.shape), tuple(tape.shape[1:]), tuple(tuple(l.shape) for l in leaves))
    # A captured step reads every tensor it touches at its address at capture;
    # an initialize() of the model reallocates the components' tensors, and a
    # graph from before reads freed memory (the second simulate() of one
    # simulator ran from garbage).  The graphs hold for one initialization.
    count = initialization_count(functional_model)
    if functional_model.__dict__.get("_step_graphs_initialization") != count:
        functional_model.__dict__["_step_graphs"] = {}
        functional_model.__dict__["_step_graphs_initialization"] = count
    cache = functional_model.__dict__.setdefault("_step_graphs", {})
    graphs = cache.get(key)
    if graphs is None:
        graphs = StepGraphs(step, spec, len(leaves), name=f"{kind} step")
        cache[key] = graphs
    return _StepGraphRollout.apply(graphs, y0, theta, tape, *leaves)
