"""Collocation (simultaneous) transcription for :class:`Estimator`.

Single-shooting estimation evaluates the objective by simulating the *entire*
horizon from one fixed initial condition, then backpropagating through the full
unrolled trajectory.  The gradient is a product of per-step Jacobians; for
unstable or oscillatory dynamics that product is badly conditioned on long
horizons (the exploding/vanishing-gradient problem of backprop through time).
Dissipative building models are largely insensitive to this -- their per-step
Jacobians contract -- but every single-shooting iterate is still a full
sequential rollout.

**Collocation** promotes the state at every timestep boundary ``s_i`` to a
decision variable, stacked alongside the physical parameters ``theta``.  The
dynamics are enforced as hard equality *continuity defects*

    d_i = x(t_{i+1}; s_i, theta) - s_{i+1}      (i = 0 .. K-2)

where the first term is one step of the model from ``s_i`` and ``s_{i+1}`` is
read directly off the decision vector.  Gradients only ever flow through a
single step.  The practical benefits are robustness to poor initial parameter
guesses (the state variables can stay close to the data while ``theta`` is
still far off), the sparse block-bidiagonal NLP structure IPOPT exploits, and
the estimated initial state coming out of the fit for free.
The boundary states are *nuisance* variables: they are discarded after the fit
(``EstimationResult`` reports only ``theta``; the per-period initial states are
additionally returned as ``estimated_initial_state``).

The solve (:func:`_solve_sparse_collocation`) hands IPOPT the defects as sparse
equality constraints with an explicit block-bidiagonal Jacobian, a Gauss-Newton
Hessian of the least-squares objective, and patience-based early stopping.
When the model is composable, the objective/constraints/derivatives all come
from the pure one-step map built by
:class:`~twin4build.simulator._composed.OneStepComposer` (via
:meth:`Simulator.compose`) -- no per-eval
object-graph simulate; otherwise an exact-but-slow finite-difference fallback
runs the object graph.  Requires the CasADi/IPOPT backend.

A soft-penalty multiple-shooting form (``MSE + lambda * ||d||^2`` on a coarse
segmentation, first-order solver) used to live here; it was removed after
benchmarking showed it wanders without converging on exactly the problems the
hard-constraint solve handles.
"""

from __future__ import annotations

import datetime
import traceback
import os as _os
import time as _time
from collections import defaultdict
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.func import jacrev, vmap

import twin4build.utils.types as tps
import twin4build.core as core
from twin4build.simulator._functional import StateLayout as _StateLayout
from twin4build.simulator._functional import collect_stateful as _collect_stateful
from twin4build.simulator._replica_layout import (
    ReplicaLayout,
    ThetaReplicaLayout,
    colored_hessian_rows,
    colored_jacobian,
    validate_replica_influence,
)
from twin4build.utils._cuda_graph import (
    CudaGraphCallable,
    is_cuda_graph_capture_invalidated,
)
from twin4build.utils.logger import LOGGER
from twin4build.solvers.ipopt import solve_ipopt_constrained
from twin4build.utils.types import denormalize_unit, theta_bound_tensors


class _PhaseProfiler:
    """Opt-in synchronized wall-clock timings for collocation phases."""

    def __init__(self, enabled: bool, device: torch.device):
        self.enabled = enabled
        self.device = device
        self.timings: Dict[str, float] = {}

    def _sync(self) -> None:
        if self.enabled and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def start(self):
        if not self.enabled:
            return None
        self._sync()
        return _time.perf_counter()

    def stop(self, name: str, started) -> None:
        if started is None:
            return
        self._sync()
        self.timings[name] = self.timings.get(name, 0.0) + (
            _time.perf_counter() - started
        )

    @contextmanager
    def phase(self, name: str):
        started = self.start()
        try:
            yield
        finally:
            self.stop(name, started)


def _segment_boundaries(n_t: int, n_segments: int) -> List[int]:
    """Partition ``n_t`` timesteps into ``n_segments`` contiguous blocks.

    Returns boundary indices ``[b_0=0, b_1, ..., b_K=n_t]`` (length K+1), so
    segment ``i`` spans steps ``[b_i, b_{i+1})``.  Blocks are as even as
    possible; ``n_segments >= n_t`` degenerates to one step per segment
    (collocation).
    """
    n_segments = max(1, min(n_segments, n_t))
    base, extra = divmod(n_t, n_segments)
    bounds = [0]
    for i in range(n_segments):
        bounds.append(bounds[-1] + base + (1 if i < extra else 0))
    return bounds


def _aggregate_objective_targets(
    actual: torch.Tensor,
    included: torch.Tensor,
    previous_of: torch.Tensor,
    measurement_lag: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Map scored measurement targets onto their producing segments.

    Returns a per-(producer, sensor) target count and target mean. Multiple
    targets may map to a period's first producer when warmup is disabled.
    """
    n_seg, n_meas = actual.shape
    seg_ix = torch.arange(n_seg, dtype=torch.long, device=actual.device)
    producer_of = torch.where(
        measurement_lag.unsqueeze(0),
        previous_of.unsqueeze(1),
        seg_ix.unsqueeze(1),
    )
    scored = torch.nonzero(included, as_tuple=False).flatten()
    count = torch.zeros_like(actual)
    target_sum = torch.zeros_like(actual)
    ones = torch.ones(len(scored), dtype=actual.dtype, device=actual.device)
    for m in range(n_meas):
        source = producer_of[scored, m]
        count[:, m].index_add_(0, source, ones)
        target_sum[:, m].index_add_(0, source, actual[scored, m])
    target_mean = target_sum / count.clamp_min(1.0)
    return count, target_mean


def _fixed_basis_hessian_two_args(grad_fn, x, y, *rest):
    """Hessian blocks from explicit, statically sized VJP bases.

    ``torch.func.hessian`` builds its forward-mode basis with data-dependent
    diagonal offsets that Dynamo cannot specialize in current Colab PyTorch.
    A JVP-over-gradient workaround still hits a CUDA fake/inference-tensor bug.
    This reverse-over-reverse formulation supplies fixed cotangent bases
    directly and avoids both compiler failures.
    """
    primals = (x, y, *rest)
    x_zero = torch.zeros_like(x)
    y_zero = torch.zeros_like(y)
    _, pullback = torch.func.vjp(grad_fn, *primals)

    hxx, hxy = vmap(lambda basis: pullback((basis, y_zero))[:2])(
        torch.eye(x.numel(), dtype=x.dtype, device=x.device).reshape(
            x.numel(), *x.shape
        )
    )
    hyx, hyy = vmap(lambda basis: pullback((x_zero, basis))[:2])(
        torch.eye(y.numel(), dtype=y.dtype, device=y.device).reshape(
            y.numel(), *y.shape
        )
    )
    return (hxx, hxy), (hyx, hyy)


class _IterateCache:
    """One-iterate cache shared by all composable IPOPT callbacks."""

    def __init__(self):
        self._key = None
        self._values = {}
        self.stats = {
            "forward_evaluations": 0,
            "forward_cache_hits": 0,
            "derivative_evaluations": 0,
            "derivative_cache_hits": 0,
        }

    @staticmethod
    def _normalize(z):
        array = np.ascontiguousarray(np.asarray(z, dtype=np.float64))
        return array, (array.shape, array.tobytes())

    def _get(self, name, z, compute):
        array, key = self._normalize(z)
        if key != self._key:
            self._key = key
            self._values.clear()
        if name in self._values:
            self.stats[f"{name}_cache_hits"] += 1
            return self._values[name]
        value = compute(array)
        self._values[name] = value
        self.stats[f"{name}_evaluations"] += 1
        return value

    def forward(self, z, compute):
        return self._get("forward", z, compute)

    def derivatives(self, z, compute):
        return self._get("derivative", z, compute)


class _DirectCudaGraph:
    """Optional graph with safe pre-invalidation eager fallback and statistics."""

    def __init__(self, fn, requested: bool, name: str):
        self.fn = fn
        self.requested = bool(requested)
        self.enabled = bool(requested)
        self.name = name
        self.graph = None
        self.stats = {
            "requested": self.requested,
            "enabled": self.enabled,
            "captured": False,
            "calls": 0,
            "replays": 0,
            "capture_seconds": 0.0,
            "replay_seconds": 0.0,
            "fallback_reason": None,
        }

    def __call__(self, *inputs):
        self.stats["calls"] += 1
        if not self.enabled:
            return self.fn(*inputs)
        try:
            if self.graph is None:
                LOGGER.config("Capturing CUDA Graph %s", self.name)
                LOGGER._flush_log_buffer()
                self.graph = CudaGraphCallable(self.fn)
                output = self.graph(*inputs)
                self.stats["captured"] = True
                LOGGER.config(
                    "Captured CUDA Graph %s in %.3fs",
                    self.name,
                    float(self.graph.capture_seconds),
                )
                LOGGER._flush_log_buffer()
            else:
                output = self.graph(*inputs)
                self.stats["replays"] += 1
            self.stats["capture_seconds"] = self.graph.capture_seconds
            self.stats["replay_seconds"] = self.graph.replay_seconds
            return output
        except Exception as exc:
            if is_cuda_graph_capture_invalidated(exc):
                # CUDA documents subsequent work in this process/context as
                # unsafe. In particular, do not execute the eager callback.
                raise
            self.stats["fallback_reason"] = f"{type(exc).__name__}: {exc}"
            self.enabled = False
            self.stats["enabled"] = False
            if self.graph is not None:
                self.graph.close()
                self.graph = None
            LOGGER.warning(
                "Direct CUDA Graph %s unavailable (%s) -- using eager.",
                self.name,
                self.stats["fallback_reason"],
            )
            return self.fn(*inputs)

    def close(self):
        if self.graph is not None:
            self.graph.close()
            self.graph = None


def _assemble_objective_gradient(
    Jt_meas,
    Jx_meas,
    measurements,
    target_count,
    target_mean,
    measurement_sd,
    n_objective_terms,
):
    """Assemble the exact least-squares gradient from shared measurement rows."""
    weighted_residual = target_count * (measurements - target_mean) / measurement_sd
    scale = 2.0 / float(n_objective_terms)
    grad_theta = scale * torch.einsum("gmt,gm->t", Jt_meas, weighted_residual)
    grad_states = scale * torch.einsum("gmd,gm->gd", Jx_meas, weighted_residual)
    return torch.cat([grad_theta, grad_states.reshape(-1)])


def _pack_colored_jacobian_vals(Jtg, Jtl, Jxg, Jxl, cp_i, row_includes_local):
    """Pack colored Jacobian blocks into COO values with one host sync.

    Matches the historical ``(link, row)`` loop order: global theta colors,
    replica-local theta colors when the output row is private, global state
    colors, replica-local state colors when private, then ``-1`` for the
    defect ``-I`` entry.  Global output rows omit the local-color blocks.
    """
    Jtg_l = Jtg.index_select(0, cp_i)
    Jtl_l = Jtl.index_select(0, cp_i)
    Jxg_l = Jxg.index_select(0, cp_i)
    Jxl_l = Jxl.index_select(0, cp_i)
    n_links, n_rows = Jtg_l.shape[:2]
    packed = torch.cat(
        [
            Jtg_l,
            Jtl_l,
            Jxg_l,
            Jxl_l,
            Jtg_l.new_full((n_links, n_rows, 1), -1.0),
        ],
        dim=-1,
    )
    n_tg = Jtg_l.shape[-1]
    n_tl = Jtl_l.shape[-1]
    n_xg = Jxg_l.shape[-1]
    n_xl = Jxl_l.shape[-1]
    keep = packed.new_ones((n_rows, packed.shape[-1]), dtype=torch.bool)
    global_rows = ~row_includes_local.to(device=keep.device, dtype=torch.bool)
    if n_tl:
        keep[global_rows, n_tg : n_tg + n_tl] = False
    if n_xl:
        keep[
            global_rows,
            n_tg + n_tl + n_xg : n_tg + n_tl + n_xg + n_xl,
        ] = False
    return packed.reshape(-1)[keep.expand(n_links, -1, -1).reshape(-1)]


def solve_collocation(estimator, method: tuple, options: Dict) -> SimpleNamespace:
    """Run a collocation (simultaneous transcription) estimation solve.

    Parameters
    ----------
    estimator : Estimator
        The calling estimator, fully set up by ``estimate`` (parameters,
        measurements, normalized bounds ``_x0_norm``/``_lb_norm``/``_ub_norm``,
        ``actual_readings``, and a single time period in ``_start_time`` etc.).
    method : tuple
        ``(library, optimizer, mode)`` -- must be the CasADi backend
        (the sparse hard-constraint NLP needs IPOPT).
    options : dict
        Solver options, forwarded to :func:`_solve_sparse_collocation`
        (``hessian``, ``early_stopping``, ``pin_initial_state``,
        ``boundary_state_init``, plus raw
        IPOPT options such as ``maxiter``).

    Returns
    -------
    types.SimpleNamespace
        SciPy-``OptimizeResult``-like (``x`` = optimal **theta_norm** only,
        plus ``fun``/``success``/``nit``/``nfev``/``message``) so the
        Estimator's result-building tail consumes it unchanged.
    """
    self = estimator
    options = dict(options or {})
    # IPOPT/CasADi stays on the CPU (small numpy vectors); the torch rollouts
    # and Jacobians run on the model's device.  Inbound z vectors are placed
    # on `dev`, outbound values go through .cpu().numpy().
    dev = self.simulator.model.device
    profiler = _PhaseProfiler(bool(options.pop("profile_phases", False)), dev)
    transcription_started = profiler.start()

    if method[0] != "casadi":
        raise ValueError(
            "Transcription solves require the CasADi/IPOPT backend -- use "
            "method=('casadi', 'ipopt', 'ad', 'collocation')."
        )

    start_times = list(self._start_time)
    end_times = list(self._end_time)
    step_sizes = list(self._stepSize)

    n_theta = len(self._x0_norm)

    # ---- Warm start: one forward sim at theta0 over every period -----------
    # Builds the state layout and seeds each segment's initial state from a
    # dynamically-consistent trajectory.
    x0_param_values = self._theta_to_param_values(
        torch.tensor(self._x0_norm, dtype=tps.float_dtype(), device=dev)
    )
    self.simulator.model.set_parameters(
        x0_param_values,
        self._flat_components,
        self._parameter_names,
        normalized=True,
        overwrite=True,
    )
    with profiler.phase("warm_start_initial_simulation_seconds"):
        self.simulator.simulate(
            start_time=start_times,
            end_time=end_times,
            step_size=step_sizes,
            show_progress_bar=False,
        )
    stateful = _collect_stateful(self.simulator.model)
    assert stateful, (
        "No StatefulSystem components found -- multiple-shooting has no boundary "
        "states to introduce. Use single-shooting for purely algebraic models."
    )
    layout = _StateLayout(stateful)
    D = layout.width

    # ---- Flatten every period's segments onto a single n_s batch axis ------
    # Every period is split into one-step segments (full simultaneous
    # transcription); the segments of *all* periods live together on the
    # simulator's n_s axis.  Continuity defects link only *consecutive
    # segments within the same period* -- disjoint training windows are
    # independent experiments (no cross-window stitching).
    seg_starts: List[datetime.datetime] = []
    seg_ends: List[datetime.datetime] = []
    seg_steps: List[int] = []
    seg_len: List[int] = []
    warm_states: List[torch.Tensor] = []
    continuity_pairs: List[Tuple[int, int]] = []
    seg_actual: Dict[str, List[torch.Tensor]] = {
        md.id: [] for md, _ in self._measurements
    }
    seg_is_warmup: List[bool] = (
        []
    )  # first n_warmup segments of each period -> excluded from the data fit

    g = 0
    for p, (s_p, e_p, step_p) in enumerate(zip(start_times, end_times, step_sizes)):
        _, _, n_t_p, _ = core.Simulator.get_simulation_timesteps(s_p, e_p, step_p)
        bounds_p = _segment_boundaries(n_t_p, n_t_p)
        Kp = len(bounds_p) - 1
        with profiler.phase("warm_start_boundary_collection_seconds"):
            state0_p = _warmstart_segment_states(
                self, layout, bounds_p, s_p, e_p, step_p
            )
        actual_p = {
            md.id: np.asarray(
                self.actual_readings[md.id][p].to_numpy(), dtype=np.float64
            ).flatten()
            for md, _ in self._measurements
        }
        for i in range(Kp):
            seg_starts.append(
                s_p + datetime.timedelta(seconds=int(bounds_p[i] * step_p))
            )
            seg_ends.append(
                s_p + datetime.timedelta(seconds=int(bounds_p[i + 1] * step_p))
            )
            seg_steps.append(step_p)
            seg_len.append(bounds_p[i + 1] - bounds_p[i])
            warm_states.append(state0_p[i])
            seg_is_warmup.append(i < self._n_warmup)
            for md, _ in self._measurements:
                seg_actual[md.id].append(
                    torch.tensor(
                        actual_p[md.id][bounds_p[i] : bounds_p[i + 1]],
                        dtype=tps.float_dtype(),
                        device=dev,
                    )
                )
            if i < Kp - 1:
                continuity_pairs.append((g, g + 1))
            g += 1
    n_seg = g
    if not continuity_pairs:
        raise ValueError(
            "Collocation requires more than one timestep per period -- there "
            "are no continuity links to constrain.  Use single-shooting."
        )

    LOGGER.config(
        "Transcription: %s | %d periods | %d segments total | %d continuity links",
        self._transcription,
        len(start_times),
        n_seg,
        len(continuity_pairs),
    )

    # Per-dimension state normalization (O(1) decision vars regardless of units).
    seg_state0 = torch.stack(warm_states, dim=0)  # (n_seg, D)
    center = seg_state0.mean(dim=0)
    scale = seg_state0.std(dim=0)
    scale = torch.where(scale < 1e-6, torch.ones_like(scale), scale)

    def s_to_norm(s_phys: torch.Tensor) -> torch.Tensor:
        return (s_phys - center) / scale

    def s_from_norm(s_norm: torch.Tensor) -> torch.Tensor:
        return s_norm * scale + center

    seg_state0_norm = s_to_norm(seg_state0)  # (n_seg, D)

    # ---- Decision vector z = [theta_norm | s_norm.flatten()] ---------------
    z0 = np.concatenate(
        [
            np.asarray(self._x0_norm, dtype=np.float64),
            seg_state0_norm.reshape(-1).detach().cpu().numpy(),
        ]
    )
    # Box on the boundary states: the warm-start trajectory's own per-dimension
    # range, widened by ``_DEFAULT_STATE_MARGIN`` normalized units (a std each).
    # A fixed +/-6 box used to clip the initial transient of the warm start
    # (which sits many std from the trajectory mean), making a feasible warm
    # start infeasible and excluding the true trajectory from the feasible
    # set.  The composer path below recomputes this from the augmented y0.
    state_lb, state_ub = _trajectory_box(seg_state0_norm, _DEFAULT_STATE_MARGIN)
    lb = np.concatenate([np.asarray(self._lb_norm, dtype=np.float64), state_lb])
    ub = np.concatenate([np.asarray(self._ub_norm, dtype=np.float64), state_ub])

    self._eval_count = 0
    LOGGER.config(
        "Decision variables: %d (theta=%d, states=%d)", len(z0), n_theta, n_seg * D
    )

    # Sparse, hard-constraint collocation: the dynamics are hard equality
    # *defect constraints* with an explicit block-bidiagonal Jacobian; IPOPT's
    # sparse linear solver exploits the structure (each defect row touches
    # only s_i, s_{i+1}, theta).  The objective is data-fit only.
    result = _solve_sparse_collocation(
        self,
        method,
        options,
        n_theta,
        D,
        n_seg,
        layout,
        seg_starts,
        seg_ends,
        seg_steps,
        seg_len,
        seg_actual,
        continuity_pairs,
        s_to_norm,
        s_from_norm,
        z0,
        lb,
        ub,
        seg_is_warmup,
        profiler=profiler,
    )
    profiler.stop("transcription_total_seconds", transcription_started)
    if profiler.enabled:
        t = profiler.timings
        t["pre_ipopt_other_seconds"] = max(
            0.0,
            t.get("pre_ipopt_total_seconds", 0.0)
            - t.get("composer_build_seconds", 0.0)
            - t.get("capture_rollout_seconds", 0.0),
        )
        t["audit_other_seconds"] = max(
            0.0,
            t.get("postsolve_audit_total_seconds", 0.0)
            - t.get("audit_defect_seconds", 0.0)
            - t.get("audit_nlp_forward_seconds", 0.0)
            - t.get("audit_composed_rollout_seconds", 0.0)
            - t.get("audit_object_graph_rollout_seconds", 0.0),
        )
        t["transcription_orchestration_other_seconds"] = max(
            0.0,
            t["transcription_total_seconds"]
            - t.get("warm_start_initial_simulation_seconds", 0.0)
            - t.get("warm_start_boundary_collection_seconds", 0.0)
            - t.get("pre_ipopt_total_seconds", 0.0)
            - t.get("ipopt_total_seconds", 0.0)
            - t.get("postsolve_attach_state_seconds", 0.0)
            - t.get("postsolve_audit_total_seconds", 0.0),
        )
        result.collocation_timing = dict(profiler.timings)
        LOGGER.result("COLLOCATION PHASE TIMINGS (s): %s", result.collocation_timing)
    return result


_DEFAULT_STATE_MARGIN = 6.0  # normalized units beyond the warm-start range

# IPOPT options applied (as defaults) when the initial point is feasible, i.e.
# ``boundary_state_init`` resolved to "rollout": start the barrier small and
# keep the initial point where it is instead of pushing it into the interior.
_IPOPT_WARM_START_DEFAULTS = {
    "ipopt.warm_start_init_point": "yes",
    "ipopt.mu_init": 1e-6,
    "ipopt.bound_push": 1e-8,
    "ipopt.bound_frac": 1e-8,
    "ipopt.warm_start_bound_push": 1e-8,
    "ipopt.warm_start_bound_frac": 1e-8,
    "ipopt.warm_start_slack_bound_push": 1e-8,
    "ipopt.warm_start_slack_bound_frac": 1e-8,
    "ipopt.warm_start_mult_bound_push": 1e-8,
}


def _trajectory_box(y_norm: torch.Tensor, margin: float):
    """Per-dimension box bounds that CONTAIN the warm-start trajectory.

    ``y_norm`` is ``(n_seg, dim)`` in normalized units (std-scaled, so
    ``margin`` reads in standard deviations of that dimension).  Returns the
    flattened ``(n_seg * dim,)`` lower and upper bounds: each dimension's
    warm-start min/max widened by ``margin`` on both sides, tiled over the
    segments.  Any box that does NOT contain the warm start is a bug: IPOPT
    projects the initial point into the box, which silently turns a
    feasible warm start into an infeasible one (and, worse, excludes the
    true trajectory when the fit has excursions outside the box).
    """
    y = y_norm.detach().to(dtype=torch.float64).cpu().numpy()
    n_seg = y.shape[0]
    lo = np.tile(y.min(axis=0) - float(margin), n_seg)
    hi = np.tile(y.max(axis=0) + float(margin), n_seg)
    return lo, hi


def _warmstart_segment_states(
    self, layout, bounds_idx, start_time, end_time, step_size
):
    """Capture the state trajectory at each segment boundary from one rollout.

    Runs a per-step simulation over the full window, snapshotting each stateful
    component's state at the segment-boundary steps.  Returns ``(K, D)``.
    """
    K = len(bounds_idx) - 1
    boundary_set = set(bounds_idx[:-1])  # states at the START of each segment
    snapshots: Dict[int, torch.Tensor] = {}

    # We drive the simulator step-by-step by re-running the full window with an
    # after_initialize no-op, then reading get_state is insufficient (only final
    # state).  Instead reconstruct via the components' recorded output histories
    # is model-specific, so we step manually here.
    self.simulator.get_simulation_timesteps([start_time], [end_time], [step_size])
    self.simulator.model.initialize([start_time], [end_time], [step_size])
    second_time_steps, date_time_steps, max_timesteps, _ = (
        core.Simulator.get_simulation_timesteps([start_time], [end_time], [step_size])
    )
    for step_index in range(max_timesteps):
        if step_index in boundary_set:
            snapshots[step_index] = layout.gather(0).detach().clone()
        self.simulator._do_system_time_step(
            self.simulator.model,
            second_time_steps[:, step_index],
            date_time_steps[:, step_index],
            [step_size],
            step_index,
            "gauss-seidel",
        )
    # Any boundary at/after the final step: use the last available state.
    for i, b in enumerate(bounds_idx[:-1]):
        if b not in snapshots:
            snapshots[b] = layout.gather(0).detach().clone()
    return torch.stack([snapshots[b] for b in bounds_idx[:-1]], dim=0)  # (K, D)


def _solve_sparse_collocation(
    self,
    method,
    options,
    n_theta,
    D,
    n_seg,
    layout,
    seg_starts,
    seg_ends,
    seg_steps,
    seg_len,
    seg_actual,
    continuity_pairs,
    s_to_norm,
    s_from_norm,
    z0,
    lb,
    ub,
    seg_is_warmup=None,
    profiler=None,
):
    """Hard-constraint collocation with a block-bidiagonal sparse Jacobian.

    Objective = data-fit MSE (continuity moved into constraints).  Constraints
    ``g`` are the per-link defects ``end_norm[i] - s_norm[j]`` (j = i+1 within a
    period).  The constraint Jacobian is sparse:

    * ``d defect_l / d s_norm[i]`` (D x D): the segment's *state* sensitivity --
      exact via ``D`` reverse-mode passes, isolated per segment because the
      simulator's n_s batch elements are independent.
    * ``d defect_l / d theta`` (D x n_theta): shared across segments, so it can't
      be isolated in one backward pass -- computed by ``n_theta`` finite
      differences (theta is low-dimensional, so this is cheap).
    * ``d defect_l / d s_norm[j]`` = ``-I``.
    """
    # IPOPT itself stays on the CPU; the torch evaluations run on the model's
    # device (inbound z -> dev, outbound -> .cpu().numpy()).
    dev = self.simulator.model.device
    profiler = profiler or _PhaseProfiler(False, dev)
    pre_ipopt_started = profiler.start()
    cp = [(int(a), int(b)) for a, b in continuity_pairs]
    n_links = len(cp)
    n_g = n_links * D
    # Segments that start a period (no incoming continuity link): their boundary
    # state is the trajectory's *initial condition*.
    period_starts = sorted(set(range(n_seg)) - {j for _, j in cp})
    # ``pin_initial_state``: fix each period's initial augmented state at its
    # warm-start value via bound equality (lb == ub).  This removes the extra
    # initial-condition freedom relative to single-shooting, so with tight
    # defect tolerances the feasible set is exactly the single-shooting
    # trajectory manifold (equivalence/stationarity testing).
    pin_initial_state = bool(options.pop("pin_initial_state", False))
    legacy_hessian_options = {
        "gauss_newton",
        "exact_hessian",
        "capture_hessian",
        "capture_derivatives",
        "capture",
    }.intersection(options)
    if legacy_hessian_options:
        names = ", ".join(sorted(legacy_hessian_options))
        raise TypeError(
            f"Removed collocation option(s): {names}. Use "
            "hessian='exact'|'gauss_newton'|'limited_memory'; CUDA graph "
            "capture is selected by Simulator.execution_backend."
        )
    hessian = str(options.pop("hessian", "exact")).lower()
    if hessian not in {"exact", "gauss_newton", "limited_memory"}:
        raise ValueError("hessian must be 'exact', 'gauss_newton', or 'limited_memory'")
    gauss_newton = hessian in {"exact", "gauss_newton"}
    # Exact mode adds the constraint-curvature term sum(lam_g * d2g) that
    # plain Gauss-Newton drops.  GN is exact only for least squares with SMALL
    # residuals and LINEAR constraints; here the constraints are the nonlinear
    # dynamics, so the dropped term is significant -- it is why the dual
    # infeasibility plateaus, IPOPT's real convergence test is unreachable, and
    # the acceptable_* heuristics below are needed to stop the solve at all.
    # Costs ~3x the constraint Jacobian per iteration (Da+n_theta forward
    # tangents over one reverse pass, vs Da cotangents) in exchange for a
    # reachable KKT test and Newton-rate convergence.
    exact_hessian = hessian == "exact"
    # ``early_stopping``: patience-based stagnation stop + best-feasible-iterate
    # checkpoint (see solve_ipopt_constrained).  False disables; a dict
    # overrides the patience/tolerance defaults.  Default: on whenever the GN
    # Hessian is on (that is the configuration whose dual criteria plateau).
    # ``boundary_state_init``: WHERE the collocation boundary states start.
    # This used to be a ``data_warmstart`` boolean, which was a bad interface:
    # the correct value depended on whether the CALLER had already produced a
    # converged fit, nothing checked it, and getting it wrong failed silently
    # (a cold start with the "refinement" setting simply returns its own input).
    #
    #   "rollout" -- start on the warm-start trajectory itself.  Right when
    #                REFINING an already-converged fit: it begins ON the
    #                trajectory manifold at that fit, and because only a
    #                FEASIBLE x0 can become the best-iterate incumbent, it is
    #                also what lets early stopping guarantee the solve never
    #                returns worse than what it was handed.
    #   "data"     -- seed the directly-observed states from the MEASUREMENTS.
    #                A COLD-START device: it keeps a simultaneous method out of
    #                bad local minima when theta is far off, at the cost of
    #                violating the continuity defects (measured on the
    #                full-workflow example from a converged fit: max|defect|
    #                4.9 seeded vs 3.4e-5 unseeded).
    #   "auto"     -- (default) decide by MEASURING the warm start instead of
    #                asking the caller to remember.  See _AUTO_REFINE_TOL.
    _AUTO_REFINE_TOL = 25.0  # mean weighted squared residual, i.e. ~5 sd
    boundary_state_init = str(options.pop("boundary_state_init", "auto")).lower()
    # ``boundary_state_margin``: how far (in normalized units, i.e. std of each
    # dimension over the warm-start trajectory) the boundary-state box extends
    # beyond the warm start's own per-dimension min/max.  See _trajectory_box.
    boundary_state_margin = float(
        options.pop("boundary_state_margin", _DEFAULT_STATE_MARGIN)
    )
    if not boundary_state_margin > 0:
        raise ValueError("boundary_state_margin must be positive")
    if "data_warmstart" in options:  # older boolean spelling
        boundary_state_init = (
            "data" if bool(options.pop("data_warmstart")) else "rollout"
        )
    if _os.environ.get("TWIN4BUILD_NO_DATA_WARMSTART") is not None:
        boundary_state_init = "rollout"  # legacy escape hatch, retained
    _ws_fit = None  # set when "auto" measures the warm start; reported in the audit
    if boundary_state_init not in ("auto", "data", "rollout"):
        raise ValueError(
            "boundary_state_init must be 'auto', 'data' or 'rollout' (got "
            f"{boundary_state_init!r}).  'rollout' refines a converged fit; "
            "'data' cold-starts from the measurements; 'auto' measures the "
            "warm start and picks."
        )
    early_stopping = options.pop("early_stopping", None)
    if early_stopping is None:
        early_stopping = hessian != "limited_memory"

    # -- Stage-3 fast Jacobian via the functorch composer --------------------
    # Try to build a pure one-step map F(states, theta_phys, captured) from the
    # components' forward() methods.  If it works, the constraint Jacobian comes
    # from a single vmap(jacrev(F)) call instead of D reverse passes + n_theta
    # finite-difference re-simulations. Shared and multi-branch parameters
    # compose via indexed theta selectors. On ordinary incompatibilities,
    # fall back to the exact-but-slow FD path.
    composer = None
    state_replica_layout = theta_replica_layout = None
    composer_started = profiler.start()
    try:
        # Indexed theta spec: shared parameters route several (comp, attr)
        # entries to one selector and compiled parameters use branch slices.
        theta_spec, unique_parameters = self._composer_theta_spec()
        composed_layout, comp = self.simulator.build_functional_model(
            theta_spec=theta_spec,
            measurements=[md for md, _ in self._measurements],
            step_size=seg_steps[0],
        )
        LOGGER.config(
            "Composer captured (frozen exogenous) inputs: %s | "
            "cut-feedback edges: %s",
            comp._exogenous_keys,
            comp._feedback_keys,
        )
        # Plain (functorch-safe) denormalization from the parameters'
        # physical bounds + scaling (tps.Parameter.denormalize is a
        # Tensor-subclass method and breaks under functorch) -- one
        # representative parameter per unique theta entry.
        lb_t, ub_t, log_mask = theta_bound_tensors(unique_parameters, device=dev)
        composer = comp
        state_replica_layout = ReplicaLayout.from_functional_model(composer)
        theta_replica_layout = ThetaReplicaLayout.from_estimator(self, composer)
        if state_replica_layout.n_replicas != theta_replica_layout.n_replicas:
            raise NotImplementedError(
                "Theta and augmented-state replica counts disagree: "
                f"{theta_replica_layout.n_replicas} versus "
                f"{state_replica_layout.n_replicas}."
            )
    except Exception as exc:  # noqa: BLE001
        if (
            isinstance(exc, NotImplementedError)
            and self.simulator.execution_mode == "functional"
        ):
            raise
        LOGGER.warning(
            "Composer unavailable (%s) -- using finite-difference Jacobian.", exc
        )
        composer = None
    finally:
        profiler.stop("composer_build_seconds", composer_started)

    def _denorm(th_norm):
        # Single source of truth for the normalized->physical map
        # (tps.Parameter.denormalize routes through the same function).
        return denormalize_unit(th_norm, lb_t, ub_t, log_mask)

    def _simulate(theta, s_norm):
        """Run all segments; return (scaled_mse, raw_mse, end_norm)."""
        s_phys = s_from_norm(s_norm)
        param_values = self._theta_to_param_values(theta)
        self.simulator.model.set_parameters(
            param_values,
            self._flat_components,
            self._parameter_names,
            normalized=True,
            overwrite=True,
        )
        self.simulator.simulate(
            start_time=seg_starts,
            end_time=seg_ends,
            step_size=seg_steps,
            show_progress_bar=False,
            after_initialize=lambda: layout.scatter(s_phys),
        )
        res_terms, res_raw = [], []
        for md, sd in self._measurements:
            for gi in range(n_seg):
                L = seg_len[gi]
                y = md.input["measuredValue"].history(i_t=slice(0, L), i_s=gi, i_c=0)
                raw = seg_actual[md.id][gi] - y
                res_raw.append(raw)
                res_terms.append(raw / sd)
        mse = torch.mean(torch.cat(res_terms) ** 2)
        raw_mse = torch.mean(torch.cat(res_raw) ** 2)
        end_norm = s_to_norm(layout.end_states(n_seg))
        return mse, raw_mse, end_norm

    # Objective value+grad cache (data-fit only).
    self._eval_count = 0
    _c = {"key": None, "f": None, "gf": None}

    def _obj_compute(z):
        key = z.tobytes()
        if _c["key"] == key:
            return _c["f"], _c["gf"]
        self._eval_count += 1
        zt = torch.tensor(z, dtype=tps.float_dtype(), device=dev, requires_grad=True)
        mse, raw_mse, _ = _simulate(zt[:n_theta], zt[n_theta:].reshape(n_seg, D))
        (gf,) = torch.autograd.grad(mse, zt)
        self._last_rmse = float(raw_mse.detach()) ** 0.5
        _c.update(key=key, f=float(mse.detach()), gf=gf.detach().cpu().numpy())
        if self._eval_count % 10 == 1:
            LOGGER.iter(
                "eval=%d | obj=%.6f | rmse=%.4f",
                self._eval_count,
                _c["f"],
                self._last_rmse,
            )
        return _c["f"], _c["gf"]

    def obj_fun(z):
        return _obj_compute(np.asarray(z, dtype=np.float64))[0]

    def obj_grad(z):
        return _obj_compute(np.asarray(z, dtype=np.float64))[1]

    def _defect_from_end(end_norm, s_norm):
        rows = [end_norm[i] - s_norm[j] for (i, j) in cp]  # each (D,)
        return torch.stack(rows, dim=0)  # (n_links, D)

    def g_fun(z):
        zt = torch.tensor(z, dtype=tps.float_dtype(), device=dev)
        with torch.no_grad():
            _, _, end_norm = _simulate(zt[:n_theta], zt[n_theta:].reshape(n_seg, D))
        s_norm = zt[n_theta:].reshape(n_seg, D)
        return _defect_from_end(end_norm, s_norm).reshape(-1).cpu().numpy()

    # Fixed sparsity pattern (rows, cols), in the exact order g_jac_vals fills.
    jac_rows, jac_cols = [], []
    for l, (i, j) in enumerate(cp):
        for r in range(D):
            row = l * D + r
            for c in range(n_theta):  # d/dtheta block
                jac_rows.append(row)
                jac_cols.append(c)
            for c in range(D):  # d/ds[i] block
                jac_rows.append(row)
                jac_cols.append(n_theta + i * D + c)
            jac_rows.append(row)
            jac_cols.append(n_theta + j * D + r)  # -I on s[j]
    jac_rows = np.asarray(jac_rows, dtype=np.int64)
    jac_cols = np.asarray(jac_cols, dtype=np.int64)

    def _assemble_vals(J_theta, J_s):
        """Pack per-segment blocks into the (jac_rows, jac_cols) order.

        ``J_theta[c][i, r]`` = d end_norm[i, r]/d theta[c];
        ``J_s[r][i, c]``     = d end_norm[i, r]/d s_norm[i, c].
        """
        vals = []
        for i, j in cp:
            for r in range(D):
                for c in range(n_theta):
                    vals.append(float(J_theta[c][i, r]))
                for c in range(D):
                    vals.append(float(J_s[r][i, c]))
                vals.append(-1.0)  # d defect_r / d s_norm[j, r]
        return np.asarray(vals, dtype=np.float64)

    cp_i = torch.tensor([i for i, _ in cp], dtype=torch.long, device=dev)
    cp_j = torch.tensor([j for _, j in cp], dtype=torch.long, device=dev)

    # ===== Augmented-state vmap-F path ======================================
    # Compute the objective, defects AND Jacobian from a single vmap(F_aug) with
    # inputs captured once -- no per-eval object-graph simulate (the profiled
    # bottleneck: model.initialize re-read every CSV on every evaluation).
    #
    # Cut feedback edges (e.g. office.heatGain <- space_heater.Power) are one-step
    # LAG variables -- state in a discrete-time sense.  We append them to the
    # state (y = [state | feedback], width Da = D + n_fb); F_aug maps
    # y_t -> [F(s_t, w_t), producer_output(s_t, w_t)], so feedback continuity IS
    # ordinary state continuity and matches do_step's one-step-delayed feedback
    # exactly.  Only truly-exogenous inputs (weather, schedules) stay frozen.
    n_fb = composer.n_feedback if composer is not None else 0
    Da = D + n_fb
    CAP = None
    ACT = SD_meas = y_to_norm = y_from_norm = None
    z0_a, lb_a, ub_a = z0, lb, ub
    jac_rows_a = jac_cols_a = None
    n_g_a = n_links * Da
    if composer is not None and composer.meas_sources:
        theta0_phys = _denorm(
            torch.tensor(z0[:n_theta], dtype=tps.float_dtype(), device=dev)
        )
        s0_phys = s_from_norm(
            torch.tensor(z0[n_theta:], dtype=tps.float_dtype(), device=dev).reshape(
                n_seg, D
            )
        )
        # Continuity chains: segment -> next segment within the same period.
        _next_of = dict(cp)
        _chains = []
        for _s0 in period_starts:
            chain = [_s0]
            while chain[-1] in _next_of:
                chain.append(_next_of[chain[-1]])
            _chains.append(chain)
        # Capture the exogenous inputs AND the delayed-feedback warm start from
        # one CONTINUOUS batched do_step rollout over all periods at theta0
        # (the shared Simulator.capture_rollout; see
        # simulator/_composed.py::capture_reference_rollout for why this must
        # be a continuous run: stateful exogenous drivers like OccupancySystem,
        # and Gauss-Seidel consumption semantics for the feedback warm start).
        # Segments are one step each, so the rollout's per-timestep rows map
        # 1:1 onto the chain's segment indices.
        CAP = torch.zeros(
            (n_seg, composer._n_exogenous),
            dtype=tps.float_dtype(),
            device=dev,
        )
        fb0 = torch.zeros((n_seg, n_fb), dtype=tps.float_dtype(), device=dev)
        with profiler.phase("capture_rollout_seconds"):
            R = self.simulator.record_exogenous_inputs(
                composer,
                [seg_starts[chain[0]] for chain in _chains],
                [seg_ends[chain[-1]] for chain in _chains],
                [seg_steps[chain[0]] for chain in _chains],
            )
        for p, chain in enumerate(_chains):
            idx = torch.tensor(chain[: R.n_timesteps[p]], dtype=torch.long, device=dev)
            CAP[idx] = R.exogenous_tape[p][: len(idx)]
            fb0[idx] = R.feedback_tape[p][: len(idx)]
        fb_center = fb0.mean(dim=0)
        # Robust scale: a ~constant feedback has tiny std; scaling by it would blow
        # the (bounded) decision variable up.  Floor by a fraction of the magnitude.
        # (std needs >= 2 samples and a non-empty feedback dim, else it warns.)
        fb_floor = 0.1 * fb_center.abs() + 1e-3
        if n_fb and n_seg > 1:
            fb_scale = torch.maximum(fb0.std(dim=0), fb_floor)
        else:
            fb_scale = fb_floor

        md_list = [md for md, _ in self._measurements]
        SD_meas = torch.tensor(
            [float(sd) for _, sd in self._measurements],
            dtype=tps.float_dtype(),
            device=dev,
        )
        ACT = torch.zeros((n_seg, len(md_list)), dtype=tps.float_dtype(), device=dev)
        for m, md in enumerate(md_list):
            for gi in range(n_seg):
                ACT[gi, m] = float(
                    torch.as_tensor(seg_actual[md.id][gi]).reshape(-1)[0]
                )
        # Warmup mask: exclude each period's first n_warmup segments from the data
        # fit, exactly as single-shooting does -- otherwise the collocation scores
        # the initial transient (e.g. CO2 settling from the default init, ~300 ppm)
        # that single-shooting throws away, which dominates the (all-sensor)
        # objective and drags the optimum off the good (temperature) solution.
        _incl = torch.tensor(
            [not w for w in (seg_is_warmup or [False] * n_seg)],
            dtype=torch.bool,
            device=dev,
        )
        if not bool(_incl.any()):
            _incl = torch.ones(n_seg, dtype=torch.bool, device=dev)
        LOGGER.config(
            "Collocation objective: scoring %d/%d segments (%d warmup excluded)",
            int(_incl.sum()),
            n_seg,
            n_seg - int(_incl.sum()),
        )

        # One-step sensor lag -- the SAME correction FunctionalEstimationObjective applies
        # (see _shooting.py).  A pass-through sensor that executes BEFORE its
        # producer in the Gauss-Seidel order reads the producer's PREVIOUS-step
        # output, so ``do_step`` (and single-shooting, which shifts to match)
        # scores a one-step-lagged signal for it, while ``F_aug`` returns the
        # current step's.  Without the same shift here the collocation
        # objective scores a DIFFERENT quantity than stage 1 for those sensors:
        # the fit looks better inside the NLP than the model actually is, and a
        # collocation "refinement" of a single-shooting optimum is not even
        # minimizing the same function.
        meas_lag = [
            bool(spec[0] == "fresh" and composer.pos[md.id] < composer.pos[spec[1]])
            for md, spec in zip(md_list, composer.meas_sources)
        ]
        # Predecessor segment of each segment (itself for period starts, which
        # have no predecessor; those sit inside the n_warmup mask in every
        # practical configuration, so their value is not scored).
        _prev_of = torch.arange(n_seg, dtype=torch.long, device=dev)
        _next_of = torch.arange(n_seg, dtype=torch.long, device=dev)
        for _i, _j in cp:
            _prev_of[_j] = _i
            _next_of[_i] = _j
        _lag_mask = torch.tensor(meas_lag, dtype=torch.bool, device=dev).reshape(1, -1)
        _any_lag = any(meas_lag)
        _obj_mask, _ACT_eff = _aggregate_objective_targets(
            ACT,
            _incl,
            _prev_of,
            torch.tensor(meas_lag, dtype=torch.bool, device=dev),
        )
        _n_objective_terms = int(_incl.sum()) * len(md_list)
        if _any_lag:
            LOGGER.config(
                "Collocation objective: one-step sensor lag on %s",
                [md.id for md, l in zip(md_list, meas_lag) if l],
            )

        def _apply_meas_lag(Meas):
            """Score lagged sensors against their predecessor segment's value."""
            if not _any_lag:
                return Meas
            return torch.where(_lag_mask, Meas[_prev_of], Meas)

        def y_to_norm(y_phys):
            s_n = s_to_norm(y_phys[..., :D])
            if not n_fb:
                return s_n
            return torch.cat([s_n, (y_phys[..., D:] - fb_center) / fb_scale], dim=-1)

        def y_from_norm(y_norm):
            s_p = s_from_norm(y_norm[..., :D])
            if not n_fb:
                return s_p
            return torch.cat([s_p, y_norm[..., D:] * fb_scale + fb_center], dim=-1)

        # Data-informed warm start: seed the directly-observed boundary states from
        # the measurements instead of the default-parameter rollout.  Collocation's
        # state variables let us plant the *measured* trajectory as the initial
        # guess -- the standard trick to keep simultaneous methods out of bad local
        # minima (single-shooting cannot do this: it has no state variables).  We
        # discover which state dim each measurement reads from d(meas)/d(y) (a
        # near-unit-gain readout, e.g. temp->T_air, co2->CO2, valve/damper->PID
        # memory), then overwrite that dim with the data across all segments.
        y0_phys = torch.cat([s0_phys, fb0], dim=1) if n_fb else s0_phys  # (n_seg, Da)
        # "auto": decide by measuring the warm start rather than trusting the
        # caller to remember whether they warm started.  The objective is the
        # MEAN WEIGHTED SQUARED RESIDUAL, so it reads in units of measurement
        # standard deviations: ~1 means the rollout already sits in the noise,
        # >> 1 means theta is far off.  A converged fit on the full-workflow
        # example scores ~8; its uncalibrated x0 scores ~1140.  Anything under
        # _AUTO_REFINE_TOL (~5 sd) is treated as a refinement worth preserving.
        if boundary_state_init == "auto":
            with torch.no_grad():
                _, _M_ws = vmap(
                    lambda yi, ci: composer.F_aug(
                        yi, theta0_phys, ci, transform_mode=True
                    )
                )(y0_phys, CAP)
                _ws_fit = float(
                    (((ACT - _apply_meas_lag(_M_ws)) / SD_meas) ** 2)[_incl].mean()
                )
            boundary_state_init = "rollout" if _ws_fit <= _AUTO_REFINE_TOL else "data"
            LOGGER.config(
                "Boundary-state init: auto -> '%s'.  The warm start scores "
                "%.4g (mean weighted squared residual; <= %.4g means it is "
                "already within ~%.0f sd of the data and worth preserving, "
                "otherwise the observed states are seeded from measurements).",
                boundary_state_init,
                _ws_fit,
                _AUTO_REFINE_TOL,
                _AUTO_REFINE_TOL**0.5,
            )
        if boundary_state_init == "rollout":
            LOGGER.config(
                "Boundary-state init: 'rollout' -- the boundary states ARE the "
                "warm-start trajectory, so the initial point lies on the "
                "continuity manifold and early stopping can adopt it as the "
                "best-feasible incumbent (the solve cannot return worse)."
            )
        if boundary_state_init == "data":
            Jm = jacrev(
                lambda y: composer.F_aug(y, theta0_phys, CAP[0], transform_mode=True)[1]
            )(y0_phys[0].clone())
            # Measurement predicted AT the warm start, needed for the correction
            # below (one cheap vmap over the segments).
            with torch.no_grad():
                _, M0 = vmap(
                    lambda yi, ci: composer.F_aug(
                        yi, theta0_phys, ci, transform_mode=True
                    )
                )(y0_phys, CAP)
            seeded, allmap, _seeded_dims = [], [], set()
            for m in range(len(md_list)):
                j = int(Jm[m].abs().argmax())
                coeff = float(Jm[m, j])
                allmap.append((md_list[m].id, j, round(coeff, 3)))
                if j in _seeded_dims:
                    # Two measurements reading the same state dim: applying both
                    # corrections would double-count the same residual.
                    continue
                if (
                    abs(coeff) > 0.2
                ):  # a state readout (unit-gain, or attenuated by a clamp)
                    # First-order correction TOWARD the data, not a rescaling of
                    # it.  ``coeff`` is d(meas_t)/d(y_t) -- for a state with
                    # dynamics that is a one-step transition factor (0.79 for a
                    # room-air temperature at 20 min steps), NOT a readout gain,
                    # so the old ``y = ACT / coeff`` was only valid when the
                    # measurement passed through the origin.  It did not: it
                    # seeded every segment's air temperature at 21.5/0.79 ~ 27 C
                    # and handed IPOPT a point 225x worse in objective and
                    # grossly infeasible (measured: f=1875.8, max|defect|=7.7,
                    # against 8.2 / 7e-5 for the same warm start unseeded).
                    #
                    #     meas(y) ~ meas(y0) + coeff * (y - y0)
                    #  => y = y0 + (ACT - meas(y0)) / coeff
                    #
                    # This is exact to first order regardless of offset, reduces
                    # to plain data seeding for a true unit-gain readout, and --
                    # crucially -- leaves an ALREADY-GOOD warm start essentially
                    # untouched, because the residual it corrects by is small.
                    # Pair each state with the data value actually scored
                    # against it: for a lagged sensor that is the NEXT
                    # segment's sample, since meas(y_g) is compared to
                    # ACT[next(g)].
                    _tgt = ACT[_next_of, m] if meas_lag[m] else ACT[:, m]
                    y0_phys[:, j] = y0_phys[:, j] + (_tgt - M0[:, m]) / coeff
                    _seeded_dims.add(j)
                    seeded.append((md_list[m].id, j, round(coeff, 3)))
            LOGGER.config(
                "Data-informed warm start: readouts %s | seeded %s", allmap, seeded
            )
        y0_norm = y_to_norm(y0_phys)
        if bool(options.pop("validate_replica_sparsity", __debug__)):
            _theta_probe = torch.tensor(
                z0[:n_theta], dtype=tps.float_dtype(), device=dev
            )
            _state_probe = y0_norm[0]
            _cap_probe = CAP[0]
            validate_replica_influence(
                lambda value: y_to_norm(
                    composer.F_aug(
                        y_from_norm(value),
                        _denorm(_theta_probe),
                        _cap_probe,
                        transform_mode=True,
                    )[0]
                ),
                _state_probe,
                state_replica_layout,
                state_replica_layout,
            )
            validate_replica_influence(
                lambda value: y_to_norm(
                    composer.F_aug(
                        y_from_norm(_state_probe),
                        _denorm(value),
                        _cap_probe,
                        transform_mode=True,
                    )[0]
                ),
                _theta_probe,
                theta_replica_layout,
                state_replica_layout,
            )

            # Every measurement row may belong to at most one replica.  Rows
            # independent of all private inputs are genuinely global/shared.
            # This check catches pooled/cross-zone outputs before their scalar
            # objective is assigned compact arrowhead curvature.
            _meas_support = [set() for _ in md_list]
            for _input, _input_layout, _fn in (
                (
                    _state_probe,
                    state_replica_layout,
                    lambda value: composer.F_aug(
                        y_from_norm(value),
                        _denorm(_theta_probe),
                        _cap_probe,
                        transform_mode=True,
                    )[1],
                ),
                (
                    _theta_probe,
                    theta_replica_layout,
                    lambda value: composer.F_aug(
                        y_from_norm(_state_probe),
                        _denorm(value),
                        _cap_probe,
                        transform_mode=True,
                    )[1],
                ),
            ):
                for _replica, _columns in enumerate(_input_layout.replica_indices):
                    for _column in _columns:
                        _direction = torch.zeros_like(_input)
                        _direction[int(_column)] = 1
                        _, _dm = torch.func.jvp(_fn, (_input,), (_direction,))
                        for _m in (
                            torch.nonzero(_dm.abs() > 1e-9, as_tuple=False)
                            .flatten()
                            .tolist()
                        ):
                            _meas_support[_m].add(_replica)
            _mixed = {
                md_list[m].id: sorted(support)
                for m, support in enumerate(_meas_support)
                if len(support) > 1
            }
            if _mixed:
                raise NotImplementedError(
                    "Replica-colored estimator collocation detected "
                    "cross-replica measurement outputs: "
                    f"{_mixed}. Pooled measurements require an explicit "
                    "global output mapping."
                )
        z0_a = np.concatenate(
            [
                np.asarray(z0[:n_theta], dtype=np.float64),
                y0_norm.reshape(-1).detach().cpu().numpy(),
            ]
        )
        # Box on the boundary variables: the warm start's own per-dimension
        # range (AFTER any data seeding) widened by ``boundary_state_margin``
        # std on each side, so the initial point is inside by construction.
        # (A fixed +/-6 box used to clip the initial transient -- 11-19 std
        # from the trajectory mean on the canonical 1-zone case -- which made
        # every "rollout" start infeasible and excluded the true trajectory.)
        y_lb, y_ub = _trajectory_box(y0_norm, boundary_state_margin)
        lb_a = np.concatenate([np.asarray(lb[:n_theta], dtype=np.float64), y_lb])
        ub_a = np.concatenate([np.asarray(ub[:n_theta], dtype=np.float64), y_ub])
        LOGGER.config(
            "Boundary-state box: warm-start range +/- %.1f std per dimension "
            "(widest dimension spans %.1f normalized units).",
            boundary_state_margin,
            float((y_ub - y_lb).max()) if y_ub.size else 0.0,
        )
        if pin_initial_state:
            for s0 in period_starts:
                a = n_theta + s0 * Da
                lb_a[a : a + Da] = z0_a[a : a + Da]
                ub_a[a : a + Da] = z0_a[a : a + Da]
            LOGGER.config(
                "Pinned the initial augmented state of %d period(s) at the "
                "warm-start value (bound equality).",
                len(period_starts),
            )

        # Exact replica block-arrowhead Jacobian.  A global output may depend
        # only on shared theta/state slots; a replica output additionally
        # depends on that same replica's private slots.
        _state_row_replica = np.full(Da, -1, dtype=np.int64)
        for _replica, _indices in enumerate(state_replica_layout.replica_indices):
            _state_row_replica[_indices] = _replica
        jr, jcc = [], []
        for l, (i, j) in enumerate(cp):
            for r in range(Da):
                row = l * Da + r
                theta_cols = theta_replica_layout.global_indices.tolist()
                state_cols = state_replica_layout.global_indices.tolist()
                replica = int(_state_row_replica[r])
                if replica >= 0:
                    theta_cols += theta_replica_layout.replica_indices[replica].tolist()
                    state_cols += state_replica_layout.replica_indices[replica].tolist()
                jr.extend([row] * len(theta_cols))
                jcc.extend(theta_cols)
                jr.extend([row] * len(state_cols))
                jcc.extend((n_theta + i * Da + np.asarray(state_cols)).tolist())
                jr.append(row)
                jcc.append(n_theta + j * Da + r)
        jac_rows_a = np.asarray(jr, dtype=np.int64)
        jac_cols_a = np.asarray(jcc, dtype=np.int64)

    def _fwd_all_raw(theta_norm, y_norm):
        """Evaluate all segments once, before applying measurement lag."""
        theta_phys = _denorm(theta_norm)
        y_phys = y_from_norm(y_norm)
        Yn, Meas = vmap(
            lambda yi, ci: composer.F_aug(yi, theta_phys, ci, transform_mode=True)
        )(y_phys, CAP)
        return y_to_norm(Yn), Meas

    def _fwd_all(theta_norm, y_norm):
        """Evaluate all segments and apply lag to measurements only."""
        Y_next, Meas_raw = _fwd_all_raw(theta_norm, y_norm)
        return Y_next, _apply_meas_lag(Meas_raw)

    _callback_cache = _IterateCache()
    _callback_counts = {
        "objective_values": 0,
        "objective_gradients": 0,
        "constraints": 0,
        "constraint_jacobians": 0,
    }
    _capture_requested = self.simulator.execution_backend == "cuda_graph"

    def _shared_forward_tensor(zt):
        y_norm = zt[n_theta:].reshape(n_seg, Da)
        Y_next, Meas_raw = _fwd_all_raw(zt[:n_theta], y_norm)
        Meas = _apply_meas_lag(Meas_raw)
        included = _incl.to(dtype=Meas.dtype).unsqueeze(1)
        mse = (
            included * ((ACT - Meas) / SD_meas).square()
        ).sum() / float(_n_objective_terms)
        rmse = (
            (included * (ACT - Meas).square()).sum()
            / float(_n_objective_terms)
        ).sqrt()
        defect = Y_next[cp_i] - y_norm[cp_j]
        return mse, rmse, Meas_raw, Meas, defect

    _forward_graph = _DirectCudaGraph(
        _shared_forward_tensor, _capture_requested, "collocation forward bundle"
    )

    def _compute_shared_forward(z):
        zt = torch.tensor(z, dtype=tps.float_dtype(), device=dev)
        with torch.no_grad():
            mse, rmse, Meas_raw, Meas, defect = _forward_graph(zt)
        return {
            "mse": float(mse),
            "rmse": float(rmse),
            "measurements_raw": Meas_raw.detach().clone(),
            "measurements": Meas.detach().clone(),
            "defects": defect.detach().clone(),
            "objective_recorded": False,
        }

    def _shared_forward(z):
        return _callback_cache.forward(z, _compute_shared_forward)

    def _record_objective_evaluation(data):
        if data["objective_recorded"]:
            return
        data["objective_recorded"] = True
        self._eval_count += 1
        self._last_rmse = data["rmse"]
        if self._eval_count % 10 == 1:
            LOGGER.iter(
                "eval=%d | obj=%.6f | rmse=%.4f",
                self._eval_count,
                data["mse"],
                data["rmse"],
            )

    def _obj_value_fast(z):
        _callback_counts["objective_values"] += 1
        data = _shared_forward(z)
        _record_objective_evaluation(data)
        return data["mse"]

    def _obj_grad_fast(z):
        _callback_counts["objective_gradients"] += 1
        z = np.asarray(z, dtype=np.float64)
        data = _shared_forward(z)
        _record_objective_evaluation(data)
        d = _derivs(z)
        gradient = torch.cat([d["grad_theta"], d["grad_y"].reshape(-1)])
        return gradient.cpu().numpy().astype(np.float64)

    def g_fun_fast(z):
        _callback_counts["constraints"] += 1
        return (
            _shared_forward(z)["defects"].reshape(-1).cpu().numpy().astype(np.float64)
        )

    def _end_norm_fn(y_norm_i, theta_norm, captured_i):
        Yn, _ = composer.F_aug(
            y_from_norm(y_norm_i),
            _denorm(theta_norm),
            captured_i,
            transform_mode=True,
        )
        return y_to_norm(Yn)

    def _fixed_global_basis(replica_layout):
        basis = torch.zeros(
            (replica_layout.shared_width, replica_layout.width),
            dtype=tps.float_dtype(),
            device=dev,
        )
        if replica_layout.shared_width:
            indices, _ = replica_layout.device_indices(dev)
            basis[
                torch.arange(replica_layout.shared_width, device=dev),
                indices,
            ] = 1
        return basis

    # Construct every NumPy-backed color index/basis before entering capture.
    # The derivative graph closes over these stable CUDA allocations.
    theta_global_basis = _fixed_global_basis(theta_replica_layout)
    state_global_basis = _fixed_global_basis(state_replica_layout)
    theta_local_basis = theta_replica_layout.colored_local_basis(
        dtype=tps.float_dtype(), device=dev
    )
    state_local_basis = state_replica_layout.colored_local_basis(
        dtype=tps.float_dtype(), device=dev
    )

    def _shared_derivatives_tensor(zt):
        theta_norm = zt[:n_theta]
        y_norm = zt[n_theta:].reshape(n_seg, Da)

        def one(yi, ci):
            transition_y = lambda value: _end_norm_fn(value, theta_norm, ci)
            transition_t = lambda value: _end_norm_fn(yi, value, ci)
            Jxg, Jxl = colored_jacobian(
                transition_y,
                yi,
                state_replica_layout,
                global_basis=state_global_basis,
                local_basis=state_local_basis,
            )
            Jtg, Jtl = colored_jacobian(
                transition_t,
                theta_norm,
                theta_replica_layout,
                global_basis=theta_global_basis,
                local_basis=theta_local_basis,
            )
            return Jxg, Jxl, Jtg, Jtl

        Jxg, Jxl, Jtg, Jtl = vmap(one)(y_norm, CAP)

        def objective(th, states):
            _, measurements = _fwd_all_raw(th, states)
            residual = (measurements - _ACT_eff) / SD_meas
            return (_obj_mask * residual.square()).sum() / float(_n_objective_terms)

        grad_theta, grad_y = torch.func.grad(objective, argnums=(0, 1))(
            theta_norm, y_norm
        )
        return Jxg, Jxl, Jtg, Jtl, grad_theta, grad_y

    _derivative_graph = _DirectCudaGraph(
        _shared_derivatives_tensor,
        _capture_requested,
        "collocation gradient/Jacobian bundle",
    )

    def _compute_shared_derivatives(z):
        zt = torch.tensor(z, dtype=tps.float_dtype(), device=dev)
        Jxg, Jxl, Jtg, Jtl, grad_theta, grad_y = _derivative_graph(zt)
        return {
            "Jxg": Jxg.detach().clone(),
            "Jxl": Jxl.detach().clone(),
            "Jtg": Jtg.detach().clone(),
            "Jtl": Jtl.detach().clone(),
            "grad_theta": grad_theta.detach().clone(),
            "grad_y": grad_y.detach().clone(),
        }

    def _derivs(z):
        return _callback_cache.derivatives(z, _compute_shared_derivatives)

    _row_includes_local = (
        torch.as_tensor(_state_row_replica >= 0, dtype=torch.bool, device=dev)
        if composer is not None and composer.meas_sources
        else None
    )
    _jac_pack_logged = {"done": False}

    def g_jac_vals_fast(z):
        """Exact block-arrowhead Jacobian from colored JVP directions."""
        _callback_counts["constraint_jacobians"] += 1
        started = None
        if not _jac_pack_logged["done"]:
            LOGGER.config("Assembling collocation Jacobian values")
            LOGGER._flush_log_buffer()
            started = _time.perf_counter()
        d = _derivs(np.asarray(z, dtype=np.float64))
        vals = _pack_colored_jacobian_vals(
            d["Jtg"],
            d["Jtl"],
            d["Jxg"],
            d["Jxl"],
            cp_i,
            _row_includes_local,
        )
        packed = vals.detach().cpu().numpy().astype(np.float64)
        if started is not None:
            LOGGER.config(
                "Assembled collocation Jacobian values in %.3fs (%d nnz)",
                _time.perf_counter() - started,
                int(packed.size),
            )
            LOGGER._flush_log_buffer()
            _jac_pack_logged["done"] = True
        return packed

    def g_jac_vals_fd(z):
        z = np.asarray(z, dtype=np.float64)
        zt = torch.tensor(z, dtype=tps.float_dtype(), device=dev, requires_grad=True)
        _, _, end_norm = _simulate(zt[:n_theta], zt[n_theta:].reshape(n_seg, D))
        J_s = []
        for r in range(D):
            go = torch.zeros_like(end_norm)
            go[:, r] = 1.0
            (gz,) = torch.autograd.grad(
                end_norm, zt, grad_outputs=go, retain_graph=True
            )
            J_s.append(gz[n_theta:].reshape(n_seg, D).detach())
        end0 = end_norm.detach()
        eps = 1e-6
        J_theta = []
        for c in range(n_theta):
            zp = z.copy()
            zp[c] += eps
            ztp = torch.tensor(zp, dtype=tps.float_dtype(), device=dev)
            with torch.no_grad():
                _, _, end_p = _simulate(ztp[:n_theta], ztp[n_theta:].reshape(n_seg, D))
            J_theta.append(((end_p - end0) / eps))
        return _assemble_vals(J_theta, J_s)

    _jac_state = {"use_fast": composer is not None}

    def g_jac_vals(z):
        if _jac_state["use_fast"]:
            try:
                return g_jac_vals_fast(z)
            except Exception as exc:  # noqa: BLE001
                if is_cuda_graph_capture_invalidated(exc):
                    raise
                LOGGER.warning(
                    "Composer Jacobian failed (%s) -- falling back to "
                    "finite-difference.\n%s",
                    exc,
                    traceback.format_exc(),
                )
                _jac_state["use_fast"] = False
        return g_jac_vals_fd(z)

    def _jac_selfcheck(z0_use, jr, jc, ng, gfun, gjac, dim):
        """Compare the composer Jacobian to a finite-difference of ``gfun`` on a
        few columns -- ground-truth correctness check (env TWIN4BUILD_JAC_CHECK)."""
        if not _os.environ.get("TWIN4BUILD_JAC_CHECK"):
            return
        z0_use = np.asarray(z0_use, dtype=np.float64)
        g0 = gfun(z0_use)
        vals = gjac(z0_use)
        colmap = defaultdict(dict)
        for k in range(len(jr)):
            colmap[int(jc[k])][int(jr[k])] = vals[k]
        eps = 1e-6
        cols = list(range(n_theta)) + [n_theta + 0, n_theta + n_seg // 2 * dim]
        worst = 0.0
        for c in cols:
            zp = z0_use.copy()
            zp[c] += eps
            fd = (gfun(zp) - g0) / eps
            ana = np.zeros(ng)
            for r, v in colmap[c].items():
                ana[r] = v
            denom = max(1.0, float(np.abs(fd).max()))
            d = float(np.abs(ana - fd).max()) / denom
            worst = max(worst, d)
            LOGGER.config(
                "JAC-CHECK col %d (%s): max|ana-fd|/scale = %.3e  (|fd|max=%.3e)",
                c,
                "theta" if c < n_theta else "state",
                d,
                float(np.abs(fd).max()),
            )
        LOGGER.config("JAC-CHECK worst relative column error = %.3e", worst)

    def _attach_x0(result, z_full, dim, from_norm):
        """Record the optimised *initial state* (each period's first boundary
        state, physical units) per stateful component.  ``z_full`` may be
        augmented with feedback lag variables (width ``dim`` per segment); the
        component states are the leading ``D`` entries.  Shapes are
        ``(n_periods, n_c, state_size)`` so the dict can seed a batched
        multi-period ``simulate`` directly via ``component.set_state``."""
        z_full = np.asarray(z_full, dtype=np.float64)
        y = from_norm(
            torch.tensor(z_full[n_theta:], dtype=tps.float_dtype(), device=dev).reshape(
                n_seg, dim
            )
        )
        x0 = y[list(period_starts), :D]  # (n_periods, D); feedback lag vars trail
        d = {}
        for comp, (start, stop), (n_c, ss) in zip(
            layout.components, layout.slices, layout.shapes
        ):
            d[comp.id] = (
                x0[:, start:stop].reshape(len(period_starts), n_c, ss).detach().clone()
            )
        result.estimated_initial_state = d
        return result

    LOGGER.config(
        "Sparse collocation: %d constraints, %d nonzeros | Jacobian=%s | n_feedback=%d",
        n_g_a if composer is not None else n_g,
        len(jac_rows_a) if jac_rows_a is not None else len(jac_rows),
        "composer(augmented)" if composer is not None else "finite-diff",
        n_fb,
    )

    # Timestepping micro-benchmark (env TWIN4BUILD_BENCH_TIMESTEP): compute every
    # segment's one-step transition both ways -- the new vmap(F_aug) map vs the old
    # object-graph batched simulate (do_step traversal) -- and report wall time and
    # agreement.  This isolates *why* collocation is fast: the per-evaluation
    # forward cost that both the objective and the defects pay each iteration.
    if (
        _os.environ.get("TWIN4BUILD_BENCH_TIMESTEP")
        and composer is not None
        and CAP is not None
    ):
        th0 = torch.tensor(z0_a[:n_theta], dtype=tps.float_dtype(), device=dev)
        y0 = torch.tensor(z0_a[n_theta:], dtype=tps.float_dtype(), device=dev).reshape(
            n_seg, Da
        )
        s0 = torch.tensor(z0[n_theta:], dtype=tps.float_dtype(), device=dev).reshape(
            n_seg, D
        )
        with torch.no_grad():
            Yn, _ = _fwd_all(th0, y0)  # warm up vmap path
            _, _, end_old = _simulate(th0, s0)  # warm up object-graph path
        agree = float((Yn[:, :D] - end_old).abs().max())
        reps = 5
        t0 = _time.time()
        for _ in range(reps):
            with torch.no_grad():
                _fwd_all(th0, y0)
        t_new = (_time.time() - t0) / reps
        t0 = _time.time()
        for _ in range(reps):
            with torch.no_grad():
                _simulate(th0, s0)
        t_old = (_time.time() - t0) / reps
        LOGGER.result(
            "TIMESTEP BENCH (%d segments x 1 step): vmap(F_aug)=%.1f ms | "
            "object-graph simulate=%.1f ms | speedup=%.1fx | max|Δstate|=%.2e",
            n_seg,
            t_new * 1e3,
            t_old * 1e3,
            (t_old / t_new if t_new else float("nan")),
            agree,
        )

    def _audit_fast(result):
        """Post-solve feasibility / self-consistency audit (fast path).

        Reports (a) the max defect violation at the returned solution, (b) how
        many boundary variables sit on their box bounds, and (c) a per-sensor
        comparison of THREE fits:

        * **NLP-internal** -- measurements evaluated at the free boundary
          variables (what the optimizer scored);
        * **sequential F_aug rollout** -- from each period's estimated initial
          state (params + init incl. feedback lags).  A gap to (NLP) means the
          solution leans on defect slack;
        * **do_step rollout** -- the real object-graph model simulated from the
          same estimated initial component states.  A gap to (F_aug) isolates
          model mismatch between the composed map and ``do_step`` -- e.g.
          captured inputs frozen from the reference simulation that actually
          depend on theta / the states (cut control loops).
        """
        z = np.asarray(result.x, dtype=np.float64)
        zt = torch.tensor(z, dtype=tps.float_dtype(), device=dev)
        theta_norm = zt[:n_theta]
        y_norm = zt[n_theta:].reshape(n_seg, Da)
        with profiler.phase("audit_defect_seconds"):
            g_vals = np.asarray(g_fun_fast(z), dtype=np.float64)
        max_defect = float(np.abs(g_vals).max()) if g_vals.size else 0.0
        next_of = dict(cp)
        with torch.no_grad():
            with profiler.phase("audit_nlp_forward_seconds"):
                _, Meas_nlp = _fwd_all(theta_norm, y_norm)
            theta_phys = _denorm(theta_norm)
            y_phys = y_from_norm(y_norm)
            Meas_roll = torch.zeros_like(Meas_nlp)
            with profiler.phase("audit_composed_rollout_seconds"):
                for s0 in period_starts:
                    gi, y = s0, y_phys[s0]
                    while True:
                        y_next, meas = composer.F_aug(y, theta_phys, CAP[gi])
                        Meas_roll[gi] = meas
                        nxt = next_of.get(gi)
                        if nxt is None:
                            break
                        gi, y = nxt, y_next
                # Same one-step sensor-lag shift the objective applies, so this
                # rollout is comparable to Meas_nlp and to the do_step rollout.
                Meas_roll = _apply_meas_lag(Meas_roll)
            # Real object-graph (do_step) rollout from the same estimated
            # initial component states and theta.  Divergence from the F_aug
            # rollout is *model mismatch* between the composed map and do_step
            # (e.g. control loops cut by frozen "captured" inputs).
            Meas_step = torch.zeros_like(Meas_nlp)
            param_values = self._theta_to_param_values(zt[:n_theta])
            self.simulator.model.set_parameters(
                param_values,
                self._flat_components,
                self._parameter_names,
                normalized=True,
                overwrite=True,
            )
            with profiler.phase("audit_object_graph_rollout_seconds"):
                for s0 in period_starts:
                    chain = [s0]
                    while chain[-1] in next_of:
                        chain.append(next_of[chain[-1]])
                    x0 = y_phys[s0, :D]

                    def _seed(x0=x0):
                        for comp, (a, b), (n_c, ss) in zip(
                            layout.components, layout.slices, layout.shapes
                        ):
                            comp.set_state(x0[a:b].reshape(1, n_c, ss))

                    self.simulator.simulate(
                        start_time=[seg_starts[s0]],
                        end_time=[seg_ends[chain[-1]]],
                        step_size=[seg_steps[s0]],
                        show_progress_bar=False,
                        after_initialize=_seed,
                        execution_mode="object",
                        execution_backend="eager",
                    )
                    for m, (md, _) in enumerate(self._measurements):
                        vals = md.input["measuredValue"].history(
                            i_t=slice(0, len(chain)), i_s=0, i_c=0
                        )
                        Meas_step[chain, m] = vals.reshape(-1).detach()
        audit = {
            "return_status": str(getattr(result, "status", "")),
            "max_defect": max_defect,
            # What the boundary states were actually initialised from, AFTER
            # "auto" resolved.  Reported rather than left in a log line so the
            # choice is inspectable from the result -- a "rollout" init paired
            # with a large warm_start_fit is the signature of the misuse where
            # a cold start is told to refine and gets its own x0 handed back.
            "boundary_state_init": boundary_state_init,
            "warm_start_fit": _ws_fit,
            "per_sensor": {},
        }
        for m, md in enumerate(md_list):
            e_nlp = float(
                torch.sqrt((((ACT[:, m] - Meas_nlp[:, m]) ** 2)[_incl]).mean())
            )
            e_roll = float(
                torch.sqrt((((ACT[:, m] - Meas_roll[:, m]) ** 2)[_incl]).mean())
            )
            e_step = float(
                torch.sqrt((((ACT[:, m] - Meas_step[:, m]) ** 2)[_incl]).mean())
            )
            audit["per_sensor"][md.id] = {
                "nlp_rmse": e_nlp,
                "rollout_rmse": e_roll,
                "do_step_rmse": e_step,
            }
        # Active box bounds on the (non-pinned) boundary variables.
        lb_m = torch.tensor(
            lb_a[n_theta:], dtype=tps.float_dtype(), device=dev
        ).reshape(n_seg, Da)
        ub_m = torch.tensor(
            ub_a[n_theta:], dtype=tps.float_dtype(), device=dev
        ).reshape(n_seg, Da)
        free = (ub_m - lb_m) > 1e-12
        at_bound = ((y_norm - lb_m).abs() < 1e-6) | ((ub_m - y_norm).abs() < 1e-6)
        audit["n_active_state_bounds"] = int(at_bound[free].sum())
        audit["n_free_state_vars"] = int(free.sum())
        th_lb = torch.tensor(lb_a[:n_theta], dtype=tps.float_dtype(), device=dev)
        th_ub = torch.tensor(ub_a[:n_theta], dtype=tps.float_dtype(), device=dev)
        audit["n_theta_at_bounds"] = int(
            (
                ((theta_norm - th_lb).abs() < 1e-6)
                | ((th_ub - theta_norm).abs() < 1e-6)
            ).sum()
        )
        LOGGER.result(
            "AUDIT: status=%s | max|defect|=%.3e (normalized units) | boundary "
            "vars at box bound: %d/%d | theta at bound: %d/%d",
            audit["return_status"],
            max_defect,
            audit["n_active_state_bounds"],
            audit["n_free_state_vars"],
            audit["n_theta_at_bounds"],
            n_theta,
        )
        for mid, e in audit["per_sensor"].items():
            LOGGER.result(
                "AUDIT %-32s NLP-internal RMSE=%.4f | sequential F_aug rollout "
                "RMSE=%.4f | do_step rollout RMSE=%.4f (raw units)",
                mid,
                e["nlp_rmse"],
                e["rollout_rmse"],
                e["do_step_rmse"],
            )
        audit["callback_cache"] = {
            **_callback_counts,
            **_callback_cache.stats,
        }
        audit["cuda_graph"] = {
            "forward_bundle": dict(_forward_graph.stats),
            "gradient_jacobian_bundle": dict(_derivative_graph.stats),
            "lagrangian_hessian": (
                dict(_hessian_graph.stats)
                if _hessian_graph is not None
                else {
                    "requested": False,
                    "enabled": False,
                    "captured": False,
                    "reason": "limited-memory Hessian selected",
                }
            ),
        }
        result.collocation_audit = audit
        return audit

    # Augmented feedback-as-state collocation: the cut feedback signals are
    # decision variables (extra state) tied to their producer outputs by ordinary
    # continuity, so there is no frozen carry and no outer re-capture -- one solve.
    if composer is not None and CAP is not None:
        obj_fun = _obj_value_fast
        obj_grad = _obj_grad_fast

        # -- Gauss-Newton Hessian of the Lagrangian ---------------------------
        # The objective is a plain least-squares MSE, so its GN Hessian is
        # sigma * (2/N) * sum_g J_g^T J_g with J_g = d(meas_g / SD)/d(theta, y_g)
        # -- computable from ONE vmap(jacrev) over the measurement map (n_meas
        # reverse passes vs Da for the constraint Jacobian, so it is cheaper
        # than the Jacobian we already evaluate every iteration).  Constraint
        # curvature (lam_g * d2g) is dropped, the classic GN approximation;
        # IPOPT's inertia correction covers the indefiniteness gap.  Structure:
        # a dense theta x theta block, theta x y_g strips and per-segment
        # y_g x y_g diagonal blocks -- an arrowhead pattern IPOPT factorizes
        # in ~linear time.  Only segments scored by the objective contribute.
        hess_rows = hess_cols = hess_vals_fn = None
        _hessian_graph = None
        if gauss_newton:
            options = dict(options or {})
            tg = theta_replica_layout.global_indices
            tl = theta_replica_layout.replica_indices
            yg = state_replica_layout.global_indices
            yl = state_replica_layout.replica_indices
            p_global, p_zone = len(tg), theta_replica_layout.local_width
            d_global, d_zone = len(yg), state_replica_layout.local_width

            theta_entries = []
            segment_entries = []

            def add_entry(target, a, b, kind, source, column):
                target.append(
                    (
                        min(int(a), int(b)),
                        max(int(a), int(b)),
                        kind,
                        int(source),
                        int(column),
                    )
                )

            # Theta-theta entries occur once in IPOPT's structure and receive
            # the sum of all segment curvature contributions.
            for a_pos, a in enumerate(tg):
                for b in tg[a_pos:]:
                    add_entry(theta_entries, a, b, 0, a_pos, b)
                for block in tl:
                    for b in block:
                        add_entry(theta_entries, a, b, 0, a_pos, b)
            for block in tl:
                for a_pos, a in enumerate(block):
                    for b in block[a_pos:]:
                        add_entry(theta_entries, a, b, 1, a_pos, b)

            # Segment-local theta/state and state/state arrowhead entries.
            for a_pos, a in enumerate(tg):
                for b in yg:
                    add_entry(segment_entries, a, n_theta + b, 0, a_pos, n_theta + b)
                for block in yl:
                    for b in block:
                        add_entry(
                            segment_entries, a, n_theta + b, 0, a_pos, n_theta + b
                        )
            for replica, block in enumerate(tl):
                for a_pos, a in enumerate(block):
                    for b in yg:
                        add_entry(
                            segment_entries, a, n_theta + b, 1, a_pos, n_theta + b
                        )
                    for b in yl[replica]:
                        add_entry(
                            segment_entries, a, n_theta + b, 1, a_pos, n_theta + b
                        )
            for a_pos, a in enumerate(yg):
                source = p_global + a_pos
                for b in yg[a_pos:]:
                    add_entry(
                        segment_entries,
                        n_theta + a,
                        n_theta + b,
                        0,
                        source,
                        n_theta + b,
                    )
                for block in yl:
                    for b in block:
                        add_entry(
                            segment_entries,
                            n_theta + a,
                            n_theta + b,
                            0,
                            source,
                            n_theta + b,
                        )
            for block in yl:
                for a_pos, a in enumerate(block):
                    source = p_zone + a_pos
                    for b in block[a_pos:]:
                        add_entry(
                            segment_entries,
                            n_theta + a,
                            n_theta + b,
                            1,
                            source,
                            n_theta + b,
                        )

            rows_h = [e[0] for e in theta_entries]
            cols_h = [e[1] for e in theta_entries]
            for segment in range(n_seg):
                base = segment * Da
                for row, col, *_ in segment_entries:
                    rows_h.append(row if row < n_theta else row + base)
                    cols_h.append(col if col < n_theta else col + base)
            hess_rows = np.asarray(rows_h, dtype=np.int64)
            hess_cols = np.asarray(cols_h, dtype=np.int64)

            def descriptors(entries):
                return (
                    torch.as_tensor(
                        [e[2] for e in entries], dtype=torch.bool, device=dev
                    ),
                    torch.as_tensor(
                        [e[3] for e in entries], dtype=torch.long, device=dev
                    ),
                    torch.as_tensor(
                        [e[4] for e in entries], dtype=torch.long, device=dev
                    ),
                )

            th_kind, th_source, th_col = descriptors(theta_entries)
            sg_kind, sg_source, sg_col = descriptors(segment_entries)
            objective_scale = 2.0 / float(_n_objective_terms)

            def segment_scalar(th, yi, ci, li, mi, ai, scale):
                Yn, meas = composer.F_aug(
                    y_from_norm(yi), _denorm(th), ci, transform_mode=True
                )
                residual = (meas - ai) / SD_meas
                return (li * y_to_norm(Yn)).sum() + 0.5 * scale * (
                    mi * residual.square()
                ).sum()

            segment_grad = torch.func.grad(segment_scalar, argnums=(0, 1))

            def one_curvature(yi, th, ci, li, mi, ai, scale):
                return colored_hessian_rows(
                    segment_grad,
                    th,
                    yi,
                    theta_replica_layout,
                    state_replica_layout,
                    ci,
                    li,
                    mi,
                    ai,
                    scale,
                    x_global_basis=theta_global_basis,
                    x_local_basis=theta_local_basis,
                    y_global_basis=state_global_basis,
                    y_local_basis=state_local_basis,
                )

            def compact_curvature(theta_norm, y_norm, lam_by_seg, scale):
                shared, local = vmap(
                    lambda yi, ci, li, mi, ai: one_curvature(
                        yi, theta_norm, ci, li, mi, ai, scale
                    )
                )(y_norm, CAP, lam_by_seg, _obj_mask, _ACT_eff)
                # A one-replica layout has no shared colors.  Pad both color
                # axes so torch.where never has to index an empty inactive
                # branch (important inside direct CUDA-graph capture).
                shared = torch.cat(
                    [shared.new_zeros((n_seg, 1, n_theta + Da)), shared], dim=1
                )
                local = torch.cat(
                    [local.new_zeros((n_seg, 1, n_theta + Da)), local], dim=1
                )
                th = torch.where(
                    th_kind.unsqueeze(0),
                    local[
                        :,
                        torch.where(th_kind, th_source + 1, 0),
                        th_col,
                    ],
                    shared[
                        :,
                        torch.where(th_kind, 0, th_source + 1),
                        th_col,
                    ],
                ).sum(0)
                sg = torch.where(
                    sg_kind.unsqueeze(0),
                    local[
                        :,
                        torch.where(sg_kind, sg_source + 1, 0),
                        sg_col,
                    ],
                    shared[
                        :,
                        torch.where(sg_kind, 0, sg_source + 1),
                        sg_col,
                    ],
                )
                return torch.cat([th, sg.reshape(-1)])

            _hessian_graph = _DirectCudaGraph(
                compact_curvature,
                _capture_requested,
                "collocation Lagrangian Hessian",
            )

            def hess_vals_fn(z, sigma, lam_g):
                zt = torch.as_tensor(
                    np.asarray(z, dtype=np.float64), dtype=tps.float_dtype(), device=dev
                )
                lam_values = np.asarray(lam_g, dtype=np.float64).reshape(n_links, Da)
                if not exact_hessian:
                    lam_values = np.zeros_like(lam_values)
                lam = torch.as_tensor(lam_values, dtype=tps.float_dtype(), device=dev)
                lam_by_seg = torch.zeros(
                    (n_seg, Da), dtype=tps.float_dtype(), device=dev
                ).index_add(0, cp_i, lam)
                values = _hessian_graph(
                    zt[:n_theta],
                    zt[n_theta:].reshape(n_seg, Da),
                    lam_by_seg,
                    torch.as_tensor(
                        float(sigma) * objective_scale,
                        dtype=tps.float_dtype(),
                        device=dev,
                    ),
                )
                return values.detach().clone().cpu().numpy().astype(np.float64)

            LOGGER.config(
                "%s replica-arrowhead Hessian: %d nonzeros; "
                "colored directions per segment=%d (global=%d, private=%d).",
                "Exact Lagrangian" if exact_hessian else "Gauss-Newton",
                len(hess_rows),
                p_global + d_global + p_zone + d_zone,
                p_global + d_global,
                p_zone + d_zone,
            )
        # Warm-start feasibility: if the initial defects are far from zero the
        # optimizer starts OFF the trajectory manifold (either the warm start is
        # inconsistent or F_aug does not reproduce do_step) and its first moves
        # are feasibility restoration, not descent.
        g0 = np.asarray(g_fun_fast(z0_a), dtype=np.float64)
        LOGGER.config(
            "Warm-start feasibility: max|defect(z0)| = %.3e | objective(z0) = %.6f",
            float(np.abs(g0).max()) if g0.size else 0.0,
            obj_fun(z0_a),
        )
        if g0.size:
            G0 = np.abs(g0).reshape(n_links, Da)
            worst = G0.max(axis=0)
            dim_labels = []
            for _comp, (_a, _b) in zip(layout.components, layout.slices):
                dim_labels += [f"{_comp.id}[{k}]" for k in range(_b - _a)]
            dim_labels += [
                "fb:" + ".".join(map(str, k)) for k in composer._feedback_keys
            ]
            top = np.argsort(-worst)[:5]
            LOGGER.config(
                "Warm-start worst defect dims (normalized): %s",
                [
                    (
                        dim_labels[i],
                        round(float(worst[i]), 3),
                        f"link={int(G0[:, i].argmax())}",
                    )
                    for i in top
                ],
            )
        _jac_selfcheck(
            z0_a, jac_rows_a, jac_cols_a, n_g_a, g_fun_fast, g_jac_vals_fast, Da
        )
        es_cfg = None
        if early_stopping:
            es_cfg = dict(early_stopping) if isinstance(early_stopping, dict) else {}
            es_cfg.setdefault("n_theta", n_theta)
            # The stagnation rule's right aggressiveness is the SAME question
            # the boundary states just answered, so let one detected regime
            # drive both instead of making the caller keep two knobs in sync.
            #
            # Refining ("rollout"): the incumbent IS the converged fit we were
            # handed, so bailing early is free -- we cannot do worse than it.
            # Cold start ("data"): there is no good incumbent to protect, and
            # an interior-point method's objective legitimately plateaus for
            # long stretches while mu decreases, so patience-10 strangles the
            # solve mid-descent.  Only defaults move; anything the caller set
            # explicitly is left alone.
            if boundary_state_init == "data":
                es_cfg.setdefault("patience", 50)
                es_cfg.setdefault("min_delta_rel", 1e-4)
            LOGGER.config(
                "Early stopping enabled (%s regime): feas_tol=%s patience=%s "
                "min_delta_rel=%s theta_tol=%s",
                boundary_state_init,
                es_cfg.get("feas_tol", 1e-2),
                es_cfg.get("patience", 10),
                es_cfg.get("min_delta_rel", 1e-3),
                es_cfg.get("theta_tol", 1e-4),
            )
        if boundary_state_init == "rollout":
            # The start is ON the trajectory manifold, so treat it as a warm
            # start for IPOPT's barrier too.  The default mu_init=0.1 pushes
            # every bounded variable toward the interior first, which on the
            # canonical case lifted a converged fit from 1.00 to 1.38 and
            # dragged unidentified theta (PID Td, damper flow) across their
            # ranges before descending again.  Callers can override any key.
            for _key, _val in _IPOPT_WARM_START_DEFAULTS.items():
                options.setdefault(_key, _val)
        profiler.stop("pre_ipopt_total_seconds", pre_ipopt_started)
        try:
            with profiler.phase("ipopt_total_seconds"):
                result = solve_ipopt_constrained(
                    z0_a,
                    lb_a,
                    ub_a,
                    obj_fun,
                    obj_grad,
                    n_g_a,
                    g_fun_fast,
                    g_jac_vals_fast,
                    jac_rows_a,
                    jac_cols_a,
                    options=options,
                    hess_vals=hess_vals_fn,
                    hess_rows=hess_rows,
                    hess_cols=hess_cols,
                    early_stopping=es_cfg,
                )
            with profiler.phase("postsolve_attach_state_seconds"):
                _attach_x0(result, result.x, Da, y_from_norm)
            with profiler.phase("postsolve_audit_total_seconds"):
                _audit_fast(result)
            result.x = np.asarray(result.x, dtype=np.float64)[:n_theta]
            result.nfev = self._eval_count
            return result
        finally:
            _forward_graph.close()
            _derivative_graph.close()
            if _hessian_graph is not None:
                _hessian_graph.close()

    if pin_initial_state and (composer is None or CAP is None):
        for s0 in period_starts:
            a = n_theta + s0 * D
            lb[a : a + D] = z0[a : a + D]
            ub[a : a + D] = z0[a : a + D]
        LOGGER.config(
            "Pinned the initial state of %d period(s) at the warm-start value "
            "(bound equality).",
            len(period_starts),
        )
    _jac_selfcheck(z0, jac_rows, jac_cols, n_g, g_fun, g_jac_vals, D)
    profiler.stop("pre_ipopt_total_seconds", pre_ipopt_started)
    with profiler.phase("ipopt_total_seconds"):
        result = solve_ipopt_constrained(
            z0,
            lb,
            ub,
            obj_fun,
            obj_grad,
            n_g,
            g_fun,
            g_jac_vals,
            jac_rows,
            jac_cols,
            options=options,
        )
    with profiler.phase("postsolve_attach_state_seconds"):
        _attach_x0(result, result.x, D, s_from_norm)
    result.x = np.asarray(result.x, dtype=np.float64)[:n_theta]
    result.nfev = self._eval_count
    return result
