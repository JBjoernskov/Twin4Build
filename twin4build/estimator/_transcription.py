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
import os as _os
import time
from types import SimpleNamespace
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.func import jacrev, vmap

import twin4build.utils.types as tps
from twin4build.simulator._composed import StateLayout as _StateLayout
from twin4build.simulator._composed import collect_stateful as _collect_stateful
from twin4build.estimator._cuda_graph import CudaGraphRunner as _CudaGraphRunner
from twin4build.utils.logger import LOGGER
from twin4build.utils.types import denormalize_unit, theta_bound_tensors


def _normalize_hessian_stages(value):
    """Validate and fill the opt-in GN-to-exact staging configuration."""
    if value in (None, False):
        return None
    cfg = dict(value) if isinstance(value, dict) else {}
    if value is not True and not isinstance(value, dict):
        raise TypeError("hessian_stages must be False, True, or a dict")
    cfg.setdefault("stage1_maxiter", 40)
    cfg.setdefault("min_iterations", 10)
    cfg.setdefault("switch_rule", "feasible_stall")
    cfg.setdefault("feas_tol", 1e-3)
    cfg.setdefault("patience", 5)
    cfg.setdefault("min_delta_rel", 1e-3)
    cfg.setdefault("theta_tol", 1e-4)
    cfg.setdefault("probe_interval", 5)
    cfg.setdefault("cost_ratio", 3.0)
    cfg.setdefault("exact_phase_iterations", 6.0)
    cfg.setdefault("warm_start_duals", True)
    if int(cfg["stage1_maxiter"]) < 1:
        raise ValueError("hessian_stages.stage1_maxiter must be positive")
    if int(cfg["min_iterations"]) < 0:
        raise ValueError("hessian_stages.min_iterations cannot be negative")
    if str(cfg["switch_rule"]).lower() not in (
        "feasible_stall",
        "cost_aware",
    ):
        raise ValueError(
            "hessian_stages.switch_rule must be 'feasible_stall' or " "'cost_aware'"
        )
    return cfg


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

    Returns a per-(producer, sensor) target count and target mean.  Multiple
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


def solve_transcription(estimator, method: tuple, options: Dict) -> SimpleNamespace:
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
        (``gauss_newton``, ``early_stopping``, ``pin_initial_state``,
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
    import twin4build.core as core

    # IPOPT/CasADi stays on the CPU (small numpy vectors); the torch rollouts
    # and Jacobians run on the model's device.  Inbound z vectors are placed
    # on `dev`, outbound values go through .cpu().numpy().
    dev = self.simulator.model.device

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
        state0_p = _warmstart_segment_states(self, layout, bounds_p, s_p, e_p, step_p)
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
    lb = np.concatenate(
        [
            np.asarray(self._lb_norm, dtype=np.float64),
            np.full(n_seg * D, -6.0, dtype=np.float64),  # generous box on states
        ]
    )
    ub = np.concatenate(
        [
            np.asarray(self._ub_norm, dtype=np.float64),
            np.full(n_seg * D, 6.0, dtype=np.float64),
        ]
    )

    self._eval_count = 0
    LOGGER.config(
        "Decision variables: %d (theta=%d, states=%d)", len(z0), n_theta, n_seg * D
    )

    # Sparse, hard-constraint collocation: the dynamics are hard equality
    # *defect constraints* with an explicit block-bidiagonal Jacobian; IPOPT's
    # sparse linear solver exploits the structure (each defect row touches
    # only s_i, s_{i+1}, theta).  The objective is data-fit only.
    return _solve_sparse_collocation(
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
    )


def _warmstart_segment_states(
    self, layout, bounds_idx, start_time, end_time, step_size
):
    """Capture the state trajectory at each segment boundary from one rollout.

    Runs a per-step simulation over the full window, snapshotting each stateful
    component's state at the segment-boundary steps.  Returns ``(K, D)``.
    """
    import twin4build.core as core

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
    hessian_stages = _normalize_hessian_stages(options.pop("hessian_stages", False))
    # ``gauss_newton``: supply IPOPT with a Gauss-Newton Hessian of the
    # least-squares objective (J^T W J from the measurement Jacobians) instead
    # of the default limited-memory BFGS approximation.  Second-order
    # information turns the >1000-iteration L-BFGS crawl into a
    # Newton-type solve; constraint curvature is ignored (classic GN).
    gauss_newton = bool(options.pop("gauss_newton", True))
    # ``exact_hessian``: add the constraint-curvature term sum(lam_g * d2g) that
    # plain Gauss-Newton drops.  GN is exact only for least squares with SMALL
    # residuals and LINEAR constraints; here the constraints are the nonlinear
    # dynamics, so the dropped term is significant -- it is why the dual
    # infeasibility plateaus, IPOPT's real convergence test is unreachable, and
    # the acceptable_* heuristics below are needed to stop the solve at all.
    # Costs ~3x the constraint Jacobian per iteration (Da+n_theta forward
    # tangents over one reverse pass, vs Da cotangents) in exchange for a
    # reachable KKT test and Newton-rate convergence.
    exact_hessian = bool(options.pop("exact_hessian", False))
    if hessian_stages is not None:
        gauss_newton = True
        exact_hessian = True
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
        early_stopping = gauss_newton

    # -- Stage-3 fast Jacobian via the functorch composer --------------------
    # Try to build a pure one-step map F(states, theta_phys, captured) from the
    # components' forward() methods.  If it works, the constraint Jacobian comes
    # from a single vmap(jacrev(F)) call instead of D reverse passes + n_theta
    # finite-difference re-simulations.  Shared parameters compose via the
    # indexed theta spec.  On any incompatibility (missing forward, n_c>1
    # states or multi-branch parameters), fall back to the exact-but-slow FD
    # path.
    composer = None
    try:
        # Indexed theta spec: shared parameters route several (comp, attr)
        # entries to one theta slot; raises on multi-branch (n_c > 1)
        # parameters.  Structural checks (stateful present, n_c == 1 states,
        # uniform step size, state-width match) live in Simulator.compose;
        # any incompatibility raises and lands in the except below.
        theta_spec, unique_parameters = self._composer_theta_spec()
        _, comp = self.simulator.compose(
            theta_spec=theta_spec,
            measurements=[md for md, _ in self._measurements],
            step_size=seg_steps[0],
        )
        LOGGER.config(
            "Composer captured (frozen exogenous) inputs: %s | "
            "cut-feedback edges: %s",
            comp._captured_keys,
            comp._feedback_keys,
        )
        # Plain (functorch-safe) denormalization from the parameters'
        # physical bounds + scaling (tps.Parameter.denormalize is a
        # Tensor-subclass method and breaks under functorch) -- one
        # representative parameter per unique theta entry.
        lb_t, ub_t, log_mask = theta_bound_tensors(unique_parameters, device=dev)
        composer = comp
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning(
            "Composer unavailable (%s) -- using finite-difference Jacobian.", exc
        )
        composer = None

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
        z0t = torch.tensor(z0, dtype=tps.float_dtype(), device=dev)
        with torch.no_grad():
            _simulate(z0t[:n_theta], z0t[n_theta:].reshape(n_seg, D))
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
            (n_seg, len(composer._captured_keys)), dtype=tps.float_dtype(), device=dev
        )
        fb0 = torch.zeros((n_seg, n_fb), dtype=tps.float_dtype(), device=dev)
        R = self.simulator.capture_rollout(
            composer,
            [seg_starts[chain[0]] for chain in _chains],
            [seg_ends[chain[-1]] for chain in _chains],
            [seg_steps[chain[0]] for chain in _chains],
        )
        for p, chain in enumerate(_chains):
            idx = torch.tensor(chain[: R.n_t[p]], dtype=torch.long, device=dev)
            CAP[idx] = R.CAP[p][: len(idx)]
            fb0[idx] = R.FB[p][: len(idx)]
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

        # One-step sensor lag -- the SAME correction FastSingleShooting applies
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
                _, _M_ws = vmap(lambda yi, ci: composer.F_aug(yi, theta0_phys, ci))(
                    y0_phys, CAP
                )
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
            Jm = jacrev(lambda y: composer.F_aug(y, theta0_phys, CAP[0])[1])(
                y0_phys[0].clone()
            )
            # Measurement predicted AT the warm start, needed for the correction
            # below (one cheap vmap over the segments).
            with torch.no_grad():
                _, M0 = vmap(lambda yi, ci: composer.F_aug(yi, theta0_phys, ci))(
                    y0_phys, CAP
                )
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
        z0_a = np.concatenate(
            [
                np.asarray(z0[:n_theta], dtype=np.float64),
                y0_norm.reshape(-1).detach().cpu().numpy(),
            ]
        )
        # Generous box on the boundary variables; feedback lag variables get a
        # wider box than states since their robust scale can under-shoot.
        seg_lb = np.concatenate([np.full(D, -6.0), np.full(n_fb, -30.0)])
        seg_ub = np.concatenate([np.full(D, 6.0), np.full(n_fb, 30.0)])
        lb_a = np.concatenate(
            [np.asarray(lb[:n_theta], dtype=np.float64), np.tile(seg_lb, n_seg)]
        )
        ub_a = np.concatenate(
            [np.asarray(ub[:n_theta], dtype=np.float64), np.tile(seg_ub, n_seg)]
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

        # Augmented block-bidiagonal sparsity pattern (D -> Da).
        jr, jcc = [], []
        for l, (i, j) in enumerate(cp):
            for r in range(Da):
                row = l * Da + r
                for c in range(n_theta):
                    jr.append(row)
                    jcc.append(c)
                for c in range(Da):
                    jr.append(row)
                    jcc.append(n_theta + i * Da + c)
                jr.append(row)
                jcc.append(n_theta + j * Da + r)
        jac_rows_a = np.asarray(jr, dtype=np.int64)
        jac_cols_a = np.asarray(jcc, dtype=np.int64)

    def _fwd_all(theta_norm, y_norm):
        """vmap ``F_aug`` over all segments -> (Y_next (n_seg,Da), Meas (n_seg,n_meas))."""
        theta_phys = _denorm(theta_norm)
        y_phys = y_from_norm(y_norm)
        Yn, Meas = vmap(lambda yi, ci: composer.F_aug(yi, theta_phys, ci))(y_phys, CAP)
        # Lag applies to the MEASUREMENTS only -- never to Yn, which carries the
        # continuity defects.
        return y_to_norm(Yn), _apply_meas_lag(Meas)

    def _mse_of_z(zt):
        _, Meas = _fwd_all(zt[:n_theta], zt[n_theta:].reshape(n_seg, Da))
        return (((ACT - Meas) / SD_meas) ** 2)[_incl].mean()

    def _obj_compute_fast(z):
        key = z.tobytes()
        if _c["key"] == key:
            return _c["f"], _c["gf"]
        self._eval_count += 1
        zt = torch.tensor(z, dtype=tps.float_dtype(), device=dev)
        with torch.no_grad():
            _, Meas = _fwd_all(zt[:n_theta], zt[n_theta:].reshape(n_seg, Da))
            mse = float((((ACT - Meas) / SD_meas) ** 2)[_incl].mean())
            self._last_rmse = float(((ACT - Meas) ** 2)[_incl].mean()) ** 0.5
        gf = torch.func.grad(_mse_of_z)(zt).cpu().numpy()
        _c.update(key=key, f=mse, gf=gf)
        if self._eval_count % 10 == 1:
            LOGGER.iter(
                "eval=%d | obj=%.6f | rmse=%.4f", self._eval_count, mse, self._last_rmse
            )
        return mse, gf

    def g_fun_fast(z):
        zt = torch.tensor(
            np.asarray(z, dtype=np.float64), dtype=tps.float_dtype(), device=dev
        )
        y_norm = zt[n_theta:].reshape(n_seg, Da)
        with torch.no_grad():
            Y_next, _ = _fwd_all(zt[:n_theta], y_norm)
            defect = Y_next[cp_i] - y_norm[cp_j]
        return defect.reshape(-1).cpu().numpy()

    def _end_norm_fn(y_norm_i, theta_norm, captured_i):
        Yn, _ = composer.F_aug(y_from_norm(y_norm_i), _denorm(theta_norm), captured_i)
        return y_to_norm(Yn)

    # One vmap(jacrev) evaluates d(end_norm, meas/SD)/d(y, theta) for EVERY
    # segment; the constraint Jacobian (rows :Da at the linked segments) and the
    # Gauss-Newton Hessian (rows Da: at the scored segments) are slices of it.
    # IPOPT asks for jac_g and hess_lag at the same iterate, so the cache makes
    # the Hessian nearly free on top of the Jacobian we already pay for.
    _dcache = {"key": None}

    # The batched reverse pass materializes gradient buffers of roughly
    # (segments x cotangents x Da^2) per matrix_exp intermediate -- growth is
    # ~quartic in model size, and evaluating all segments at once OOM-kills
    # the process (no traceback) already at ~50 states on a 12 GB host.
    # Chunk the segment dimension of the vmap (exact, only trades speed) so
    # the peak stays within a budget (bytes; TWIN4BUILD_DERIV_BYTES overrides).
    _n_cot = Da + len(md_list)
    _itemsize = torch.empty(0, dtype=tps.float_dtype()).element_size()
    _seg_bytes = float(_n_cot * Da * Da * _itemsize * 24)  # ~24 saved intermediates
    _budget = float(_os.environ.get("TWIN4BUILD_DERIV_BYTES", 2e9))
    _deriv_chunk = max(1, min(n_seg, int(_budget / max(1.0, _seg_bytes))))
    if _deriv_chunk < n_seg:
        LOGGER.config(
            "Derivative evaluation chunked: %d segments per chunk "
            "(%d segments, %d cotangents, Da=%d).",
            _deriv_chunk,
            n_seg,
            _n_cot,
            Da,
        )

    def _derivs(z):
        key = z.tobytes()
        if _dcache["key"] == key:
            return _dcache
        zt = torch.tensor(z, dtype=tps.float_dtype(), device=dev)
        theta_norm = zt[:n_theta]
        y_norm = zt[n_theta:].reshape(n_seg, Da)

        def _both(y_i, th, cap_i):
            Yn, meas = composer.F_aug(y_from_norm(y_i), _denorm(th), cap_i)
            return torch.cat([y_to_norm(Yn), meas / SD_meas])

        # One traced pass for BOTH Jacobians (the vjp sweeps over the output
        # rows are shared), instead of two separate vmap(jacrev) evaluations.
        Jx, Jt = vmap(
            lambda yi, ci: jacrev(_both, argnums=(0, 1))(yi, theta_norm, ci),
            chunk_size=None if _deriv_chunk >= n_seg else _deriv_chunk,
        )(y_norm, CAP)
        _dcache.update(key=key, Jx=Jx.detach(), Jt=Jt.detach())
        return _dcache

    def g_jac_vals_fast(z):
        """Sparse constraint Jacobian from the shared per-segment derivatives."""
        d = _derivs(np.asarray(z, dtype=np.float64))
        Jt = d["Jt"][cp_i, :Da, :]  # (n_links, Da, n_theta)
        Jx = d["Jx"][cp_i, :Da, :]  # (n_links, Da, Da)
        # Value ordering matches (jac_rows_a, jac_cols_a): per link l, per defect
        # row r: [d/d theta (n_theta), d/d y_i (Da), -1 (the y_j identity)].
        neg1 = -torch.ones((n_links, Da, 1), dtype=tps.float_dtype(), device=dev)
        return (
            torch.cat([Jt, Jx, neg1], dim=2)
            .reshape(-1)
            .cpu()
            .numpy()
            .astype(np.float64)
        )

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
                import traceback

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
        import os as _os

        if not _os.environ.get("TWIN4BUILD_JAC_CHECK"):
            return
        z0_use = np.asarray(z0_use, dtype=np.float64)
        g0 = gfun(z0_use)
        vals = gjac(z0_use)
        from collections import defaultdict

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

    from twin4build.estimator._casadi_ipopt import solve_ipopt_constrained

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
        import time as _time

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
        g_vals = np.asarray(g_fun_fast(z), dtype=np.float64)
        max_defect = float(np.abs(g_vals).max()) if g_vals.size else 0.0
        next_of = dict(cp)
        with torch.no_grad():
            _, Meas_nlp = _fwd_all(theta_norm, y_norm)
            theta_phys = _denorm(theta_norm)
            y_phys = y_from_norm(y_norm)
            Meas_roll = torch.zeros_like(Meas_nlp)
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
        result.transcription_audit = audit
        return audit

    # Augmented feedback-as-state collocation: the cut feedback signals are
    # decision variables (extra state) tied to their producer outputs by ordinary
    # continuity, so there is no frozen carry and no outer re-capture -- one solve.
    if composer is not None and CAP is not None:
        obj_fun = lambda z: _obj_compute_fast(np.asarray(z, dtype=np.float64))[0]
        obj_grad = lambda z: _obj_compute_fast(np.asarray(z, dtype=np.float64))[1]

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
        exact_hessian_provider = None
        gn_hessian_provider = None
        hessian_stats = {
            "gauss_newton": {"calls": 0, "seconds": 0.0},
            "exact": {"calls": 0, "seconds": 0.0},
        }
        if gauss_newton and exact_hessian:
            # -- EXACT Hessian of the Lagrangian ------------------------------
            # sigma * d2f + sum_l lam_l . d2 g_l.  The dropped term in plain GN
            # is the constraint curvature, and because the constraints ARE the
            # (nonlinear) dynamics it is not small: without it the dual
            # infeasibility plateaus, IPOPT's tol=1e-8 is unreachable, and the
            # solve has to fall back on the acceptable_* heuristics below --
            # which stop it while it is still descending.
            #
            # Each defect is g_l = end_norm(y_i, theta) - y_j.  The -y_j term is
            # LINEAR, so d2 g_l touches only (y_i, theta): contracting with the
            # multipliers gives one scalar per link whose Hessian is exactly the
            # arrowhead block structure the GN term already uses.  So the
            # sparsity pattern is unchanged -- only widened from the *scored*
            # segments to every segment carrying an outgoing link.
            options = dict(options or {})
            iu_t = np.triu_indices(n_theta)
            iu_y = np.triu_indices(Da)
            cross_r, cross_c = np.meshgrid(
                np.arange(n_theta), np.arange(Da), indexing="ij"
            )
            rows_h, cols_h = [iu_t[0]], [iu_t[1]]
            for g in range(n_seg):
                base = n_theta + int(g) * Da
                rows_h.append(cross_r.ravel())
                cols_h.append(base + cross_c.ravel())
                rows_h.append(base + iu_y[0])
                cols_h.append(base + iu_y[1])
            hess_rows = np.concatenate(rows_h).astype(np.int64)
            hess_cols = np.concatenate(cols_h).astype(np.int64)
            incl_np = np.nonzero(_incl.cpu().numpy())[0]
            n_i = len(incl_np)
            incl_t = torch.tensor(incl_np, dtype=torch.long, device=dev)
            gn_scale = 2.0 / float(n_i * len(md_list))

            # Objective curvature is indexed by the segment that PRODUCES each
            # scored measurement.  For a lagged sensor, scored segment g reads
            # the output of _prev_of[g].  Build the inverse attribution by
            # scattering SCORED rows onto those producers.
            #
            # This direction matters at period boundaries: mapping producers
            # forward with _next_of incorrectly assigns the final producer to
            # itself, even though its output is never read by a later segment.
            # It can also collapse two targets onto the first producer when
            # warmup is disabled.  Count/sum aggregation handles both cases and
            # gives the exact same objective derivatives; replacing multiple
            # targets by their mean changes only an additive constant.
            _lag_t = torch.tensor(meas_lag, dtype=torch.bool, device=dev)
            _obj_mask, _ACT_eff = _aggregate_objective_targets(
                ACT, _incl, _prev_of, _lag_t
            )

            def _lam_dot_end(y_i, th, cap_i, lam_i):
                """lam . end_norm(y_i, theta) -- scalar, so its Hessian is the
                per-link constraint-curvature block."""
                return (lam_i * _end_norm_fn(y_i, th, cap_i)).sum()

            def _res_dot_meas(y_i, th, cap_i, w_i):
                """w . (meas(y_i, theta)/SD) -- scalar.  With w = the weighted
                residual this gives the RESIDUAL-curvature term that plain
                Gauss-Newton drops from the objective Hessian:
                    d2 f = (2/N) * sum_m [ grad r_m grad r_m^T + r_m d2 r_m ]
                                            ^^^ GN keeps ^^^   ^^^ this ^^^
                """
                _, meas = composer.F_aug(y_from_norm(y_i), _denorm(th), cap_i)
                return (w_i * (meas / SD_meas)).sum()

            def _exact_lagrangian_segment(
                y_i, th, cap_i, lam_i, obj_mask_i, act_i, s_gn
            ):
                """One segment's exact nonlinear Lagrangian contribution.

                Differentiating the squared residual directly produces both its
                Gauss-Newton and residual-curvature terms.  Adding the contracted
                continuity output produces constraint curvature in that same
                transform.  The factor is ``s_gn / 2`` because ``s_gn`` already
                contains the MSE Hessian's leading ``2 / N``.
                """
                Yn, meas = composer.F_aug(y_from_norm(y_i), _denorm(th), cap_i)
                residual = (meas - act_i) / SD_meas
                return (lam_i * y_to_norm(Yn)).sum() + 0.5 * s_gn * (
                    obj_mask_i * residual.square()
                ).sum()

            # jacfwd(jacrev(scalar)) over (y_i, theta): (Da + n_theta) tangents
            # per segment, against the constraint Jacobian's (Da + n_meas)
            # cotangents.  Size the Hessian's chunk from ITS OWN budget rather
            # than reusing the Jacobian's with a fudge factor.
            #
            # This replaces `_hchunk = _deriv_chunk // 2`, an unconditional 2x
            # margin that bit even when memory was not the constraint: on the
            # full-workflow example the whole Hessian needs 0.30 GB against a
            # 2 GB budget, so the Jacobian ran as ONE vmap while the Hessian
            # still ran as TWO -- and no TWIN4BUILD_DERIV_BYTES could lift it,
            # because _deriv_chunk is capped by segment count, not by budget.
            # Measured cost of that split: 1.95x on the Hessian call and 1.64x
            # on the GPU solve; on CPU end to end 1307 s -> 1098 s, with the fit
            # slightly BETTER (pooled 30.78 -> 30.22).
            #
            # The scaling below is deliberately conservative -- it predicts
            # 0.67 GB where 0.30 GB is actually used -- so it errs toward
            # chunking rather than toward an OOM.  TWIN4BUILD_HESS_CHUNK_DIV
            # divides further if a model still does not fit (default 1: no
            # extra margin beyond the budget itself).
            _h_tangents = Da + n_theta
            _hseg_bytes = _seg_bytes * (_h_tangents / max(1.0, _n_cot))
            _hchunk_budget = max(1, min(n_seg, int(_budget / max(1.0, _hseg_bytes))))
            _hdiv = float(_os.environ.get("TWIN4BUILD_HESS_CHUNK_DIV", 1))
            _hchunk = max(1, int(_hchunk_budget / max(1e-9, _hdiv)))
            _use_combined_hessian = _os.environ.get(
                "TWIN4BUILD_COMBINED_HESSIAN", "1"
            ) not in ("0", "false", "False")
            if _hchunk < n_seg:
                LOGGER.config(
                    "Hessian evaluation chunked: %d segments per chunk of %d "
                    "(%.2f GB estimated for all segments, budget %.2f GB). Each "
                    "chunk is a separate sequential vmap pass -- on a GPU that "
                    "is parallelism given away, so raise TWIN4BUILD_DERIV_BYTES "
                    "if the device has the memory.",
                    _hchunk,
                    n_seg,
                    _hseg_bytes * n_seg / 1e9,
                    _budget / 1e9,
                )

            def _curvature(theta_norm, y_norm, lam_mat):
                """Per-link d2(lam.end_norm)/d(y_i,theta)^2, vmapped."""
                hess_fn = torch.func.hessian(_lam_dot_end, argnums=(0, 1))
                return vmap(
                    lambda yi, ci, li: hess_fn(yi, theta_norm, ci, li),
                    chunk_size=None if _hchunk >= n_links else _hchunk,
                )(y_norm[cp_i], CAP[cp_i], lam_mat)

            def _obj_curvature(theta_norm, y_norm, w_mat):
                """Per-segment d2(w.meas/SD)/d(y_s,theta)^2, vmapped over ALL
                segments (w is zero where the segment serves nothing scored)."""
                hess_fn = torch.func.hessian(_res_dot_meas, argnums=(0, 1))
                return vmap(
                    lambda yi, ci, wi: hess_fn(yi, theta_norm, ci, wi),
                    chunk_size=None if _hchunk >= n_seg else _hchunk,
                )(y_norm, CAP, w_mat)

            def _combined_curvature(theta_norm, y_norm, lam_by_seg, s_gn):
                """Full exact Lagrangian Hessian from one transform per segment."""
                hess_fn = torch.func.hessian(_exact_lagrangian_segment, argnums=(0, 1))
                return vmap(
                    lambda yi, ci, li, mi, ai: hess_fn(
                        yi, theta_norm, ci, li, mi, ai, s_gn
                    ),
                    chunk_size=None if _hchunk >= n_seg else _hchunk,
                )(y_norm, CAP, lam_by_seg, _obj_mask, _ACT_eff)

            # Device-side copies of the emission pattern, so the packing that
            # used to be a 360-iteration numpy loop happens in the graph.
            _iu_t0 = torch.as_tensor(iu_t[0], dtype=torch.long, device=dev)
            _iu_t1 = torch.as_tensor(iu_t[1], dtype=torch.long, device=dev)
            _iu_y0 = torch.as_tensor(iu_y[0], dtype=torch.long, device=dev)
            _iu_y1 = torch.as_tensor(iu_y[1], dtype=torch.long, device=dev)

            def _hess_core_combined(theta_norm, y_norm, lam_mat, s_gn):
                """The Hessian's whole tensor region: inputs and output are
                tensors, there are no host branches, and shapes are static --
                the three things CUDA-graph capture requires.

                ``s_gn`` is a 0-dim TENSOR, not a float, deliberately: the
                original code branched on ``s_gn != 0.0``, and a host branch on
                a changing value cannot be captured (a graph records one path).
                Multiplying unconditionally is the same arithmetic, because
                every objective term below is linear in ``s_gn`` -- when it is
                zero the terms vanish exactly rather than approximately.  The
                The exact objective is differentiated directly, so no cached
                first-derivative tensors are inputs to this callback.
                """
                Btt = torch.zeros(
                    (n_theta, n_theta), dtype=tps.float_dtype(), device=dev
                )
                Bty = torch.zeros(
                    (n_seg, n_theta, Da), dtype=tps.float_dtype(), device=dev
                )
                Byy = torch.zeros((n_seg, Da, Da), dtype=tps.float_dtype(), device=dev)

                # One Hessian supplies all three exact terms:
                #   sigma * [Gauss-Newton + residual curvature]
                #   + sum_l lam_l * constraint curvature.
                # Objective values are indexed by their PRODUCING segment via
                # _obj_mask/_ACT_eff.
                with torch.no_grad():
                    lam_by_seg = torch.zeros(
                        (n_seg, Da),
                        dtype=tps.float_dtype(),
                        device=dev,
                    ).index_add(0, cp_i, lam_mat)
                    (Cyy, _Cyt), (Cty, Ctt) = _combined_curvature(
                        theta_norm, y_norm, lam_by_seg, s_gn
                    )
                Btt = Btt + Ctt.sum(0)
                Bty = Bty + Cty
                Byy = Byy + Cyy

                # Emit in pattern order (upper triangle of the diagonal blocks),
                # matching the hess_rows/hess_cols loop above exactly:
                #   Btt[iu_t], then per segment Bty[g].ravel() and Byy[g][iu_y].
                return torch.cat(
                    [
                        Btt[_iu_t0, _iu_t1],
                        torch.cat(
                            [
                                Bty.reshape(n_seg, n_theta * Da),
                                Byy[:, _iu_y0, _iu_y1],
                            ],
                            dim=1,
                        ).reshape(-1),
                    ]
                )

            def _hess_core_legacy(theta_norm, y_norm, lam_mat, s_gn, Jt_, Jy):
                """Pre-combination implementation retained for same-run A/B."""
                Btt = torch.zeros(
                    (n_theta, n_theta), dtype=tps.float_dtype(), device=dev
                )
                Bty = torch.zeros(
                    (n_seg, n_theta, Da), dtype=tps.float_dtype(), device=dev
                )
                Byy = torch.zeros((n_seg, Da, Da), dtype=tps.float_dtype(), device=dev)

                Jt_m = Jt_ * _obj_mask.unsqueeze(-1)
                Jy_m = Jy * _obj_mask.unsqueeze(-1)
                Btt = Btt + torch.einsum("gmi,gmj->gij", Jt_m, Jt_).sum(0) * s_gn
                Bty = Bty + torch.einsum("gmi,gmj->gij", Jt_m, Jy) * s_gn
                Byy = Byy + torch.einsum("gmi,gmj->gij", Jy_m, Jy) * s_gn

                with torch.no_grad():
                    _, meas_raw = vmap(
                        lambda yi, ci: composer.F_aug(
                            y_from_norm(yi), _denorm(theta_norm), ci
                        )
                    )(y_norm, CAP)
                    w_mat = _obj_mask * (meas_raw - _ACT_eff) / SD_meas * s_gn
                    (Oyy, _Oyt), (Oty, Ott) = _obj_curvature(theta_norm, y_norm, w_mat)
                    (Hyy, _Hyt), (Hty, Htt) = _curvature(theta_norm, y_norm, lam_mat)
                Btt = Btt + Ott.sum(0) + Htt.sum(0)
                Bty = (Bty + Oty).index_add(0, cp_i, Hty)
                Byy = (Byy + Oyy).index_add(0, cp_i, Hyy)

                return torch.cat(
                    [
                        Btt[_iu_t0, _iu_t1],
                        torch.cat(
                            [
                                Bty.reshape(n_seg, n_theta * Da),
                                Byy[:, _iu_y0, _iu_y1],
                            ],
                            dim=1,
                        ).reshape(-1),
                    ]
                )

            _hess_core = (
                _hess_core_combined if _use_combined_hessian else _hess_core_legacy
            )
            _hess_graph = _CudaGraphRunner(_hess_core, name="collocation exact Hessian")
            if _os.environ.get("TWIN4BUILD_GRAPH_DEBUG"):
                # Capture failure reports cudaErrorStreamCaptureInvalidated,
                # which names the symptom rather than the offending operation.
                # Expose the pieces so a diagnostic can capture each in
                # isolation and bisect to the one that breaks the stream.
                from twin4build.estimator import _cuda_graph as _cg_dbg

                _cg_dbg.DEBUG_PARTS.update(
                    {
                        "runner": _hess_graph,
                        "core": _hess_core,
                        "combined_core": _hess_core_combined,
                        "legacy_core": _hess_core_legacy,
                        "curvature": _curvature,
                        "obj_curvature": _obj_curvature,
                        "combined_curvature": _combined_curvature,
                        "shapes": {
                            "n_seg": n_seg,
                            "Da": Da,
                            "n_theta": n_theta,
                            "n_links": n_links,
                        },
                    }
                )

            def _exact_hess_vals(z, sigma, lam_g):
                started = time.perf_counter()
                z_np = np.asarray(z, dtype=np.float64)
                zt = torch.tensor(z_np, dtype=tps.float_dtype(), device=dev)
                theta_norm = zt[:n_theta]
                y_norm = zt[n_theta:].reshape(n_seg, Da)
                lam_mat = torch.tensor(
                    np.asarray(lam_g, dtype=np.float64).reshape(n_links, Da),
                    dtype=tps.float_dtype(),
                    device=dev,
                )
                s_gn = float(sigma) * gn_scale
                if not _use_combined_hessian and s_gn != 0.0:
                    d = _derivs(z_np)
                    Jt_ = d["Jt"][:, Da:, :]
                    Jy = d["Jx"][:, Da:, :]
                else:
                    Jt_ = Jy = None
                if _os.environ.get("TWIN4BUILD_GRAPH_DEBUG"):
                    # Stash here rather than in the runner: when the graph is
                    # disabled the runner returns before it binds anything, and
                    # a clean capture diagnostic needs the graph OFF during the
                    # solve so the first capture attempt in the process is its
                    # own.
                    from twin4build.estimator import _cuda_graph as _cg_dbg

                    _cg_dbg.DEBUG_PARTS.setdefault(
                        "inputs",
                        {
                            "theta_norm": theta_norm.detach().clone(),
                            "y_norm": y_norm.detach().clone(),
                            "lam_mat": lam_mat.detach().clone(),
                            "s_gn": torch.as_tensor(
                                s_gn, dtype=tps.float_dtype(), device=dev
                            ),
                        },
                    )
                graph_inputs = {
                    "theta_norm": theta_norm,
                    "y_norm": y_norm,
                    "lam_mat": lam_mat,
                    "s_gn": torch.as_tensor(s_gn, dtype=tps.float_dtype(), device=dev),
                }
                if not _use_combined_hessian:
                    if Jt_ is None:
                        Jt_ = torch.zeros(
                            (n_seg, len(md_list), n_theta),
                            dtype=tps.float_dtype(),
                            device=dev,
                        )
                        Jy = torch.zeros(
                            (n_seg, len(md_list), Da),
                            dtype=tps.float_dtype(),
                            device=dev,
                        )
                    graph_inputs.update(Jt_=Jt_, Jy=Jy)
                vals = _hess_graph(**graph_inputs)
                hessian_stats["exact"]["calls"] += 1
                hessian_stats["exact"]["seconds"] += time.perf_counter() - started
                # Consume the runner's static output buffer immediately -- the
                # next replay overwrites it.
                return vals.detach().cpu().numpy()

            exact_hessian_provider = (
                _exact_hess_vals,
                hess_rows.copy(),
                hess_cols.copy(),
            )

            LOGGER.config(
                "EXACT Hessian of the Lagrangian enabled: %d nonzeros (upper "
                "triangle), %d segments, %d links, %d scored. Objective term = "
                "Gauss-Newton + residual curvature; constraint term = "
                "sum(lam*d2g); both via vmap(jacfwd(jacrev)). Verified against "
                "finite differences to 6e-5 (3.6e-5 for the constraint term "
                "alone).",
                len(hess_rows),
                n_seg,
                n_links,
                n_i,
            )
        if gauss_newton and (not exact_hessian or hessian_stages is not None):
            # Termination: GN drops the constraint curvature (lam_g * d2g), so
            # the dual infeasibility plateaus (oscillating ~1e-3..1e-1) and
            # IPOPT's default tol=1e-8 is unreachable -- it then burns hundreds
            # of iterations polishing duals with ZERO objective progress before
            # dying in restoration.  Stop at the plateau instead: accept when
            # the iterate is feasible (defects ~1e-9 here) and the objective
            # has stagnated (<1e-5 relative change) for 10 consecutive
            # iterations, ignoring the (unreachable) dual/complementarity
            # criteria.
            #
            # WHICH OF THESE ARE MEANINGFUL.  ``acceptable_tol`` (1e3) and
            # ``acceptable_dual_inf_tol`` (1e10) are deliberately vacuous and
            # MUST stay that way under GN -- they are the criteria the missing
            # curvature makes unreachable, and tightening either one stops this
            # exit from firing at all, which is the restoration-failure mode
            # described above.  What actually decides when the solve stops is
            # the stagnation pair below: ``acceptable_iter`` (how many
            # consecutive stagnant iterations) and ``acceptable_obj_change_tol``
            # (how small a relative objective change counts as stagnant).
            # Those are the two to tighten if "Solved_To_Acceptable_Level" is
            # firing too early -- raise the first, shrink the second, and give
            # ``maxiter`` room to absorb the extra iterations.
            #
            # Note this makes the status weak by construction: under GN,
            # "Solved_To_Acceptable_Level" means feasible-and-stagnant, NOT
            # converged.  With ``exact_hessian=True`` the curvature is exact,
            # dual infeasibility IS reachable, none of this block applies, and
            # IPOPT's ordinary tol=1e-8 convergence test governs.
            gn_options = dict(options or {})
            gn_options.setdefault("acceptable_tol", 1e3)
            # 5 iterations at a 1e-4 relative objective change.  (An earlier
            # version of this comment claimed 1e-5 over 10 iterations; the code
            # never did that -- the values below are what runs.)
            gn_options.setdefault("acceptable_iter", 5)
            # Loosened to 1e-2 (normalized defects): the full-horizon problem
            # plateaus with max|defect| ~ 1e-3 that the audit shows is benign
            # (NLP-internal fit == sequential rollout), and a tighter gate kept
            # this exit from ever firing there.
            gn_options.setdefault("acceptable_constr_viol_tol", 1e-2)
            gn_options.setdefault("acceptable_dual_inf_tol", 1e10)
            gn_options.setdefault("acceptable_compl_inf_tol", 1e3)
            gn_options.setdefault("acceptable_obj_change_tol", 1e-4)
            incl_np = np.nonzero(_incl.cpu().numpy())[0]
            n_i = len(incl_np)
            iu_t = np.triu_indices(n_theta)
            iu_y = np.triu_indices(Da)
            cross_r, cross_c = np.meshgrid(
                np.arange(n_theta), np.arange(Da), indexing="ij"
            )
            rows_h, cols_h = [iu_t[0]], [iu_t[1]]
            for g in incl_np:
                base = n_theta + int(g) * Da
                rows_h.append(cross_r.ravel())
                cols_h.append(base + cross_c.ravel())
                rows_h.append(base + iu_y[0])
                cols_h.append(base + iu_y[1])
            hess_rows = np.concatenate(rows_h).astype(np.int64)
            hess_cols = np.concatenate(cols_h).astype(np.int64)
            incl_t = torch.tensor(incl_np, dtype=torch.long, device=dev)
            gn_scale = 2.0 / float(n_i * len(md_list))

            def _gn_hess_vals(z, sigma):
                started = time.perf_counter()
                d = _derivs(np.asarray(z, dtype=np.float64))
                Jt_ = d["Jt"][incl_t, Da:, :]  # (n_i, n_meas, n_theta)
                Jy = d["Jx"][incl_t, Da:, :]  # (n_i, n_meas, Da)
                Js = torch.cat([Jt_, Jy], dim=2)  # (n_i, n_meas, nt+Da)
                B = (
                    (torch.einsum("gmi,gmj->gij", Js, Js) * (float(sigma) * gn_scale))
                    .cpu()
                    .numpy()
                )
                vals = [B[:, :n_theta, :n_theta].sum(axis=0)[iu_t]]
                Bty = B[:, :n_theta, n_theta:]
                Byy = B[:, n_theta:, n_theta:]
                for k in range(n_i):
                    vals.append(Bty[k].ravel())
                    vals.append(Byy[k][iu_y])
                packed = np.concatenate(vals)
                hessian_stats["gauss_newton"]["calls"] += 1
                hessian_stats["gauss_newton"]["seconds"] += (
                    time.perf_counter() - started
                )
                return packed

            gn_hessian_provider = (
                _gn_hess_vals,
                hess_rows.copy(),
                hess_cols.copy(),
            )

            LOGGER.config(
                "Gauss-Newton Hessian enabled: %d nonzeros (upper triangle), "
                "%d scored segments.",
                len(hess_rows),
                n_i,
            )
            if _os.environ.get("TWIN4BUILD_HESS_CHECK"):
                # The gradient identity grad f = (2/N) J^T r is EXACT (GN only
                # truncates the Hessian), so matching the autograd gradient
                # validates the measurement Jacobians, scaling and assembly.
                zt0 = torch.tensor(
                    np.asarray(z0_a, dtype=np.float64),
                    dtype=tps.float_dtype(),
                    device=dev,
                )
                th0_ = zt0[:n_theta]
                with torch.no_grad():
                    _, Meas0_raw = _fwd_all(th0_, zt0[n_theta:].reshape(n_seg, Da))
                r0 = (Meas0_raw[incl_t] - ACT[incl_t]) / SD_meas
                d0 = _derivs(np.asarray(z0_a, dtype=np.float64))
                Jt0 = d0["Jt"][incl_t, Da:, :]
                Jy0 = d0["Jx"][incl_t, Da:, :]
                g_gn = np.zeros(len(z0_a))
                g_gn[:n_theta] = (
                    (gn_scale * torch.einsum("gmt,gm->t", Jt0, r0)).cpu().numpy()
                )
                gy = (gn_scale * torch.einsum("gmd,gm->gd", Jy0, r0)).cpu().numpy()
                for k, g in enumerate(incl_np):
                    a = n_theta + int(g) * Da
                    g_gn[a : a + Da] = gy[k]
                g_auto = np.asarray(obj_grad(z0_a), dtype=np.float64)
                denom = max(1.0, float(np.abs(g_auto).max()))
                LOGGER.config(
                    "HESS-CHECK: max|grad_GN - grad_autograd| / scale = %.3e "
                    "(|grad|max=%.3e)",
                    float(np.abs(g_gn - g_auto).max()) / denom,
                    float(np.abs(g_auto).max()),
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
        if hessian_stages is not None:
            es_cfg = {
                key: hessian_stages[key]
                for key in (
                    "min_iterations",
                    "switch_rule",
                    "feas_tol",
                    "patience",
                    "min_delta_rel",
                    "theta_tol",
                    "probe_interval",
                    "cost_ratio",
                    "exact_phase_iterations",
                )
            }
            es_cfg["n_theta"] = n_theta
        elif early_stopping:
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

        def _stage_summary(stage_result):
            history = stage_result.iteration_history

            def _last(name):
                values = history.get(name, [])
                return float(values[-1]) if values else None

            return {
                "iterations": stage_result.nit,
                "seconds": stage_result.elapsed,
                "objective": stage_result.fun,
                "status": stage_result.status,
                "message": stage_result.message,
                "stop_reason": stage_result.stop_reason,
                "best_violation": stage_result.best_violation,
                "inf_pr": _last("inf_pr"),
                "inf_du": _last("inf_du"),
                "mu": _last("mu"),
                "regularization": _last("regularization_size"),
                "warm_start_duals": stage_result.warm_start_duals,
                "restored_best": stage_result.restored_best,
                "kkt_probes": stage_result.kkt_probes,
            }

        if hessian_stages is not None:
            if gn_hessian_provider is None or exact_hessian_provider is None:
                raise RuntimeError(
                    "hessian_stages requires both composable Gauss-Newton and "
                    "exact Hessian providers"
                )
            gn_fn, gn_rows, gn_cols = gn_hessian_provider
            stage1_options = dict(gn_options)
            stage1_options["maxiter"] = int(hessian_stages["stage1_maxiter"])
            # The callback owns the transition. IPOPT's independent acceptable
            # stop could otherwise bypass min_iterations and the chosen rule.
            stage1_options["acceptable_iter"] = 0
            stage1 = solve_ipopt_constrained(
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
                options=stage1_options,
                hess_vals=gn_fn,
                hess_rows=gn_rows,
                hess_cols=gn_cols,
                early_stopping=es_cfg,
            )
            fatal_markers = (
                "Infeasible",
                "Restoration_Failed",
                "Invalid_Number",
                "Error",
            )
            fatal_stage1 = any(marker in stage1.status for marker in fatal_markers)
            genuine_stage1_convergence = (
                stage1.status == "Solve_Succeeded" and stage1.stop_reason is None
            )
            stage_records = {"stage1": _stage_summary(stage1)}
            if fatal_stage1 or genuine_stage1_convergence:
                result = stage1
                stage_records["switched"] = False
                stage_records["switch_reason"] = (
                    "stage1 converged"
                    if genuine_stage1_convergence
                    else f"fatal stage1 exit: {stage1.status}"
                )
            else:
                exact_fn, exact_rows, exact_cols = exact_hessian_provider
                stage2_options = dict(options)
                for key in (
                    "acceptable_tol",
                    "acceptable_constr_viol_tol",
                    "acceptable_dual_inf_tol",
                    "acceptable_compl_inf_tol",
                    "acceptable_obj_change_tol",
                ):
                    stage2_options.pop(key, None)
                stage2_options["acceptable_iter"] = 0
                use_duals = bool(hessian_stages["warm_start_duals"])
                stage2 = solve_ipopt_constrained(
                    stage1.x,
                    lb_a,
                    ub_a,
                    obj_fun,
                    obj_grad,
                    n_g_a,
                    g_fun_fast,
                    g_jac_vals_fast,
                    jac_rows_a,
                    jac_cols_a,
                    options=stage2_options,
                    hess_vals=exact_fn,
                    hess_rows=exact_rows,
                    hess_cols=exact_cols,
                    early_stopping=None,
                    lam_x0=stage1.lam_x if use_duals else None,
                    lam_g0=stage1.lam_g if use_duals else None,
                    mu_init=(
                        None if stage1.restored_best else stage1.mu_final
                    ),
                )
                stage_records["stage2"] = _stage_summary(stage2)
                result = stage2
                result.nit = (stage1.nit or 0) + (result.nit or 0)
                result.elapsed += stage1.elapsed
                result.message = (
                    f"GN stage: {stage1.message}; exact stage: " f"{result.message}"
                )
                stage_records["switched"] = True
                stage_records["switch_reason"] = (
                    stage1.stop_reason or f"stage1 limit/status: {stage1.status}"
                )
            gn_calls = hessian_stats["gauss_newton"]["calls"]
            exact_calls = hessian_stats["exact"]["calls"]
            gn_avg = (
                hessian_stats["gauss_newton"]["seconds"] / gn_calls
                if gn_calls
                else None
            )
            exact_avg = (
                hessian_stats["exact"]["seconds"] / exact_calls if exact_calls else None
            )
            stage_records["hessian_callbacks"] = hessian_stats
            stage_records["rho_exact_over_gn"] = (
                exact_avg / gn_avg if gn_avg and exact_avg else None
            )
            stage_records["exact_hessian_evaluations_avoided"] = gn_calls
        else:
            if exact_hessian_provider is not None:
                hess_vals_fn, hess_rows, hess_cols = exact_hessian_provider
                solve_options = options
            elif gn_hessian_provider is not None:
                hess_vals_fn, hess_rows, hess_cols = gn_hessian_provider
                solve_options = gn_options
            else:
                solve_options = options
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
                options=solve_options,
                hess_vals=hess_vals_fn,
                hess_rows=hess_rows,
                hess_cols=hess_cols,
                early_stopping=es_cfg,
            )
        _attach_x0(result, result.x, Da, y_from_norm)
        audit = _audit_fast(result)
        audit["ipopt_diagnostics"] = _stage_summary(result)
        audit["hessian_callbacks"] = hessian_stats
        if hessian_stages is not None:
            audit["hessian_stages"] = stage_records
        result.x = np.asarray(result.x, dtype=np.float64)[:n_theta]
        result.nfev = self._eval_count
        return result

    if hessian_stages is not None:
        raise RuntimeError(
            "hessian_stages requires the composable torch.func collocation "
            "path; this model fell back to object-graph derivatives"
        )
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
    _attach_x0(result, result.x, D, s_from_norm)
    result.x = np.asarray(result.x, dtype=np.float64)[:n_theta]
    result.nfev = self._eval_count
    return result
