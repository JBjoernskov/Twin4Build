"""Collocation (simultaneous) transcription for :class:`Estimator`.

Single-shooting estimation evaluates the objective by simulating the *entire*
horizon from one fixed initial condition, then backpropagating through the full
unrolled trajectory.  Over long horizons this composition-of-``f``-with-itself
is badly conditioned (the exploding/vanishing-gradient problem of backprop
through time) and every iterate is a full rollout.

**Collocation** promotes the state at every timestep boundary ``s_i`` to a
decision variable, stacked alongside the physical parameters ``theta``.  The
dynamics are enforced as hard equality *continuity defects*

    d_i = x(t_{i+1}; s_i, theta) - s_{i+1}      (i = 0 .. K-2)

where the first term is one step of the model from ``s_i`` and ``s_{i+1}`` is
read directly off the decision vector.  Gradients only ever flow through a
single step, so the conditioning improves and the basins of attraction widen.
The boundary states are *nuisance* variables: they are discarded after the fit
(``EstimationResult`` reports only ``theta``; the per-period initial states are
additionally returned as ``estimated_initial_state``).

The solve (:func:`_solve_sparse_collocation`) hands IPOPT the defects as sparse
equality constraints with an explicit block-bidiagonal Jacobian, a Gauss-Newton
Hessian of the least-squares objective, and patience-based early stopping.
When the model is composable, the objective/constraints/derivatives all come
from the pure one-step map built by :class:`OneStepComposer` -- no per-eval
object-graph simulate; otherwise an exact-but-slow finite-difference fallback
runs the object graph.  Requires the CasADi/IPOPT backend.

A soft-penalty multiple-shooting form (``MSE + lambda * ||d||^2`` on a coarse
segmentation, first-order solver) used to live here; it was removed after
benchmarking showed it wanders without converging on exactly the problems the
hard-constraint solve handles -- see COLLOCATION_SESSION_SUMMARY.md.
"""

from __future__ import annotations

import datetime
import os as _os
from types import SimpleNamespace
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.func import jacrev, vmap

from twin4build.utils.print_progress import LOGGER
from twin4build.utils.types import denormalize_unit, theta_bound_tensors


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


class _StateLayout:
    """Flat <-> per-component packing of the stateful-component states.

    Enumerates the model's stateful components (those owning a ``tps.State``, in
    execution order) and lays their states out contiguously into a single vector
    of width ``D = sum_c (n_c * state_size_c)`` per segment boundary.  Also holds
    the per-dimension normalization (center ``c``, scale ``sigma``) so the
    boundary decision variables are O(1) regardless of physical units
    (temperatures in K vs. CO2 in ppm), which the mixed-scale warning requires.
    """

    def __init__(self, components: List):
        self.components = components
        self.slices: List[Tuple[int, int]] = []  # (start, stop) into flat vector
        self.shapes: List[Tuple[int, int]] = []  # (n_c, state_size) per component
        offset = 0
        for comp in components:
            state = comp.get_state()  # (n_s, n_c, state_size)
            n_c, ss = state.shape[1], state.shape[2]
            width = n_c * ss
            self.slices.append((offset, offset + width))
            self.shapes.append((n_c, ss))
            offset += width
        self.width = offset  # D

    def gather(self, n_s_index: int = 0) -> torch.Tensor:
        """Flatten current component states at sim-batch index into ``(D,)``."""
        parts = []
        for comp, (n_c, ss) in zip(self.components, self.shapes):
            s = comp.get_state()[n_s_index]  # (n_c, state_size)
            parts.append(s.reshape(-1))
        return torch.cat(parts) if parts else torch.zeros(0, dtype=torch.float64)

    def scatter(self, seg_states: torch.Tensor) -> None:
        """Write per-segment states into every component via ``set_state``.

        ``seg_states`` has shape ``(K, D)`` (one flat state per segment); it is
        unpacked and each component's ``set_state`` receives ``(K, n_c,
        state_size)`` -- i.e. K segments live on the simulator's n_s axis.
        """
        K = seg_states.shape[0]
        for comp, (start, stop), (n_c, ss) in zip(
            self.components, self.slices, self.shapes
        ):
            block = seg_states[:, start:stop].reshape(K, n_c, ss)
            comp.set_state(block)

    def end_states(self, K: int) -> torch.Tensor:
        """Collect each segment's *final* state into ``(K, D)`` after a sim."""
        parts = []
        for comp, (n_c, ss) in zip(self.components, self.shapes):
            s = comp.get_state()  # (K, n_c, state_size)
            parts.append(s.reshape(K, -1))
        return torch.cat(parts, dim=1) if parts else torch.zeros((K, 0), dtype=torch.float64)


def _collect_stateful(model) -> List:
    """Stateful components in execution order (composites only, not their subs).

    A composite like ``BuildingSpaceTorchSystem`` owns state (thermal|mass) and
    its submodels are not separate nodes in the execution order, so iterating
    ``_flat_execution_order`` and taking ``System.is_stateful()`` (which walks the
    owned ``tps.State``) yields each state exactly once.
    """
    order = getattr(model, "_flat_execution_order", None) or list(
        model.components.values()
    )
    return [c for c in order if c.is_stateful()]


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
        (``gauss_newton``, ``early_stopping``, ``pin_initial_state``, plus raw
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
        torch.tensor(self._x0_norm, dtype=torch.float64)
    )
    self.simulator.model.set_parameters(
        x0_param_values, self._flat_components, self._parameter_names,
        normalized=True, overwrite=True,
    )
    self.simulator.simulate(
        start_time=start_times, end_time=end_times, step_size=step_sizes,
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
    seg_actual: Dict[str, List[torch.Tensor]] = {md.id: [] for md, _ in self._measurements}
    seg_is_warmup: List[bool] = []  # first n_warmup segments of each period -> excluded from the data fit

    g = 0
    for p, (s_p, e_p, step_p) in enumerate(zip(start_times, end_times, step_sizes)):
        _, _, n_t_p, _ = core.Simulator.get_simulation_timesteps(s_p, e_p, step_p)
        bounds_p = _segment_boundaries(n_t_p, n_t_p)
        Kp = len(bounds_p) - 1
        state0_p = _warmstart_segment_states(self, layout, bounds_p, s_p, e_p, step_p)
        actual_p = {
            md.id: np.asarray(self.actual_readings[md.id][p].to_numpy(), dtype=np.float64).flatten()
            for md, _ in self._measurements
        }
        for i in range(Kp):
            seg_starts.append(s_p + datetime.timedelta(seconds=int(bounds_p[i] * step_p)))
            seg_ends.append(s_p + datetime.timedelta(seconds=int(bounds_p[i + 1] * step_p)))
            seg_steps.append(step_p)
            seg_len.append(bounds_p[i + 1] - bounds_p[i])
            warm_states.append(state0_p[i])
            seg_is_warmup.append(i < self._n_warmup)
            for md, _ in self._measurements:
                seg_actual[md.id].append(
                    torch.tensor(actual_p[md.id][bounds_p[i]:bounds_p[i + 1]], dtype=torch.float64)
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
        self._transcription, len(start_times), n_seg, len(continuity_pairs),
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
    z0 = np.concatenate([
        np.asarray(self._x0_norm, dtype=np.float64),
        seg_state0_norm.reshape(-1).detach().numpy(),
    ])
    lb = np.concatenate([
        np.asarray(self._lb_norm, dtype=np.float64),
        np.full(n_seg * D, -6.0, dtype=np.float64),  # generous box on states
    ])
    ub = np.concatenate([
        np.asarray(self._ub_norm, dtype=np.float64),
        np.full(n_seg * D, 6.0, dtype=np.float64),
    ])

    self._eval_count = 0
    LOGGER.config("Decision variables: %d (theta=%d, states=%d)", len(z0), n_theta, n_seg * D)

    # Sparse, hard-constraint collocation: the dynamics are hard equality
    # *defect constraints* with an explicit block-bidiagonal Jacobian; IPOPT's
    # sparse linear solver exploits the structure (each defect row touches
    # only s_i, s_{i+1}, theta).  The objective is data-fit only.
    return _solve_sparse_collocation(
        self, method, options, n_theta, D, n_seg, layout, seg_starts, seg_ends,
        seg_steps, seg_len, seg_actual, continuity_pairs, s_to_norm, s_from_norm,
        z0, lb, ub, seg_is_warmup,
    )


def _warmstart_segment_states(self, layout, bounds_idx, start_time, end_time, step_size):
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
    self, method, options, n_theta, D, n_seg, layout, seg_starts, seg_ends,
    seg_steps, seg_len, seg_actual, continuity_pairs, s_to_norm, s_from_norm,
    z0, lb, ub, seg_is_warmup=None,
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
    # ``gauss_newton``: supply IPOPT with a Gauss-Newton Hessian of the
    # least-squares objective (J^T W J from the measurement Jacobians) instead
    # of the default limited-memory BFGS approximation.  Second-order
    # information turns the >1000-iteration L-BFGS crawl into a
    # Newton-type solve; constraint curvature is ignored (classic GN).
    gauss_newton = bool(options.pop("gauss_newton", True))
    # ``early_stopping``: patience-based stagnation stop + best-feasible-iterate
    # checkpoint (see solve_ipopt_constrained).  False disables; a dict
    # overrides the patience/tolerance defaults.  Default: on whenever the GN
    # Hessian is on (that is the configuration whose dual criteria plateau).
    early_stopping = options.pop("early_stopping", None)
    if early_stopping is None:
        early_stopping = gauss_newton

    # -- Stage-3 fast Jacobian via the functorch composer --------------------
    # Try to build a pure one-step map F(states, theta_phys, captured) from the
    # components' forward() methods.  If it works, the constraint Jacobian comes
    # from a single vmap(jacrev(F)) call instead of D reverse passes + n_theta
    # finite-difference re-simulations.  On any incompatibility (missing forward,
    # n_c>1, shared/expanded theta), fall back to the exact-but-slow FD path.
    composer = None
    try:
        from twin4build.estimator._composer import OneStepComposer

        simple_theta = n_theta == len(self._flat_components)
        all_nc1 = all(getattr(c, "n_c", 1) == 1 for c in layout.components)
        if simple_theta and all_nc1:
            theta_spec = list(zip(self._flat_components, self._parameter_names))
            comp = OneStepComposer(
                self.simulator.model, layout.components, theta_spec, seg_steps[0],
                measurements=[md for md, _ in self._measurements],
            )
            if comp.D == D:
                LOGGER.config(
                    "Composer captured (frozen exogenous) inputs: %s | "
                    "cut-feedback edges: %s",
                    comp._captured_keys, comp._feedback_keys,
                )
                # Plain (functorch-safe) denormalization from the parameters'
                # physical bounds + scaling (tps.Parameter.denormalize is a
                # Tensor-subclass method and breaks under functorch).
                lb_t, ub_t, log_mask = theta_bound_tensors(self._flat_parameters)
                composer = comp
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Composer unavailable (%s) -- using finite-difference Jacobian.", exc)
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
            param_values, self._flat_components, self._parameter_names,
            normalized=True, overwrite=True,
        )
        self.simulator.simulate(
            start_time=seg_starts, end_time=seg_ends, step_size=seg_steps,
            show_progress_bar=False, after_initialize=lambda: layout.scatter(s_phys),
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
        zt = torch.tensor(z, dtype=torch.float64, requires_grad=True)
        mse, raw_mse, _ = _simulate(zt[:n_theta], zt[n_theta:].reshape(n_seg, D))
        (gf,) = torch.autograd.grad(mse, zt)
        self._last_rmse = float(raw_mse.detach()) ** 0.5
        _c.update(key=key, f=float(mse.detach()), gf=gf.detach().numpy())
        if self._eval_count % 10 == 1:
            LOGGER.iter("eval=%d | obj=%.6f | rmse=%.4f", self._eval_count, _c["f"], self._last_rmse)
        return _c["f"], _c["gf"]

    def obj_fun(z):
        return _obj_compute(np.asarray(z, dtype=np.float64))[0]

    def obj_grad(z):
        return _obj_compute(np.asarray(z, dtype=np.float64))[1]

    def _defect_from_end(end_norm, s_norm):
        rows = [end_norm[i] - s_norm[j] for (i, j) in cp]  # each (D,)
        return torch.stack(rows, dim=0)  # (n_links, D)

    def g_fun(z):
        zt = torch.tensor(z, dtype=torch.float64)
        with torch.no_grad():
            _, _, end_norm = _simulate(zt[:n_theta], zt[n_theta:].reshape(n_seg, D))
        s_norm = zt[n_theta:].reshape(n_seg, D)
        return _defect_from_end(end_norm, s_norm).reshape(-1).numpy()

    # Fixed sparsity pattern (rows, cols), in the exact order g_jac_vals fills.
    jac_rows, jac_cols = [], []
    for l, (i, j) in enumerate(cp):
        for r in range(D):
            row = l * D + r
            for c in range(n_theta):  # d/dtheta block
                jac_rows.append(row); jac_cols.append(c)
            for c in range(D):  # d/ds[i] block
                jac_rows.append(row); jac_cols.append(n_theta + i * D + c)
            jac_rows.append(row); jac_cols.append(n_theta + j * D + r)  # -I on s[j]
    jac_rows = np.asarray(jac_rows, dtype=np.int64)
    jac_cols = np.asarray(jac_cols, dtype=np.int64)

    def _assemble_vals(J_theta, J_s):
        """Pack per-segment blocks into the (jac_rows, jac_cols) order.

        ``J_theta[c][i, r]`` = d end_norm[i, r]/d theta[c];
        ``J_s[r][i, c]``     = d end_norm[i, r]/d s_norm[i, c].
        """
        vals = []
        for (i, j) in cp:
            for r in range(D):
                for c in range(n_theta):
                    vals.append(float(J_theta[c][i, r]))
                for c in range(D):
                    vals.append(float(J_s[r][i, c]))
                vals.append(-1.0)  # d defect_r / d s_norm[j, r]
        return np.asarray(vals, dtype=np.float64)

    cp_i = torch.tensor([i for i, _ in cp], dtype=torch.long)
    cp_j = torch.tensor([j for _, j in cp], dtype=torch.long)

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
        z0t = torch.tensor(z0, dtype=torch.float64)
        with torch.no_grad():
            _simulate(z0t[:n_theta], z0t[n_theta:].reshape(n_seg, D))
        theta0_phys = _denorm(torch.tensor(z0[:n_theta], dtype=torch.float64))
        s0_phys = s_from_norm(torch.tensor(z0[n_theta:], dtype=torch.float64).reshape(n_seg, D))
        # Continuity chains: segment -> next segment within the same period.
        _next_of = dict(cp)
        _chains = []
        for _s0 in period_starts:
            chain = [_s0]
            while chain[-1] in _next_of:
                chain.append(_next_of[chain[-1]])
            _chains.append(chain)
        # Capture the exogenous inputs AND the delayed-feedback warm start from
        # one CONTINUOUS do_step rollout per period at theta0 (the shared
        # capture_reference_rollout; see its docstring for why this must be a
        # continuous run: stateful exogenous drivers like OccupancySystem, and
        # Gauss-Seidel consumption semantics for the feedback warm start).
        # Segments are one step each, so the rollout's per-timestep rows map
        # 1:1 onto the chain's segment indices.
        from twin4build.estimator._composer import capture_reference_rollout

        CAP = torch.zeros((n_seg, len(composer._captured_keys)), dtype=torch.float64)
        fb0 = torch.zeros((n_seg, n_fb), dtype=torch.float64)
        for chain in _chains:
            R = capture_reference_rollout(
                self.simulator, composer,
                seg_starts[chain[0]], seg_ends[chain[-1]], seg_steps[chain[0]],
            )
            idx = torch.tensor(chain[: R.n_t], dtype=torch.long)
            CAP[idx] = R.CAP[: len(idx)]
            fb0[idx] = R.FB[: len(idx)]
        fb_center = fb0.mean(dim=0)
        # Robust scale: a ~constant feedback has tiny std; scaling by it would blow
        # the (bounded) decision variable up.  Floor by a fraction of the magnitude.
        fb_scale = torch.maximum(fb0.std(dim=0), 0.1 * fb_center.abs() + 1e-3)

        md_list = [md for md, _ in self._measurements]
        SD_meas = torch.tensor([float(sd) for _, sd in self._measurements], dtype=torch.float64)
        ACT = torch.zeros((n_seg, len(md_list)), dtype=torch.float64)
        for m, md in enumerate(md_list):
            for gi in range(n_seg):
                ACT[gi, m] = float(torch.as_tensor(seg_actual[md.id][gi]).reshape(-1)[0])
        # Warmup mask: exclude each period's first n_warmup segments from the data
        # fit, exactly as single-shooting does -- otherwise the collocation scores
        # the initial transient (e.g. CO2 settling from the default init, ~300 ppm)
        # that single-shooting throws away, which dominates the (all-sensor)
        # objective and drags the optimum off the good (temperature) solution.
        _incl = torch.tensor([not w for w in (seg_is_warmup or [False] * n_seg)], dtype=torch.bool)
        if not bool(_incl.any()):
            _incl = torch.ones(n_seg, dtype=torch.bool)
        LOGGER.config("Collocation objective: scoring %d/%d segments (%d warmup excluded)",
                      int(_incl.sum()), n_seg, n_seg - int(_incl.sum()))

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
        if _os.environ.get("TWIN4BUILD_NO_DATA_WARMSTART") is None:
            Jm = jacrev(lambda y: composer.F_aug(y, theta0_phys, CAP[0])[1])(y0_phys[0].clone())
            seeded, allmap = [], []
            for m in range(len(md_list)):
                j = int(Jm[m].abs().argmax())
                coeff = float(Jm[m, j])
                allmap.append((md_list[m].id, j, round(coeff, 3)))
                if abs(coeff) > 0.2:  # a state readout (unit-gain, or attenuated by a clamp)
                    y0_phys[:, j] = ACT[:, m] / coeff
                    seeded.append((md_list[m].id, j, round(coeff, 3)))
            LOGGER.config("Data-informed warm start: readouts %s | seeded %s", allmap, seeded)
        y0_norm = y_to_norm(y0_phys)
        z0_a = np.concatenate([np.asarray(z0[:n_theta], dtype=np.float64),
                               y0_norm.reshape(-1).detach().numpy()])
        # Generous box on the boundary variables; feedback lag variables get a
        # wider box than states since their robust scale can under-shoot.
        seg_lb = np.concatenate([np.full(D, -6.0), np.full(n_fb, -30.0)])
        seg_ub = np.concatenate([np.full(D, 6.0), np.full(n_fb, 30.0)])
        lb_a = np.concatenate([np.asarray(lb[:n_theta], dtype=np.float64), np.tile(seg_lb, n_seg)])
        ub_a = np.concatenate([np.asarray(ub[:n_theta], dtype=np.float64), np.tile(seg_ub, n_seg)])
        if pin_initial_state:
            for s0 in period_starts:
                a = n_theta + s0 * Da
                lb_a[a:a + Da] = z0_a[a:a + Da]
                ub_a[a:a + Da] = z0_a[a:a + Da]
            LOGGER.config(
                "Pinned the initial augmented state of %d period(s) at the "
                "warm-start value (bound equality).", len(period_starts),
            )

        # Augmented block-bidiagonal sparsity pattern (D -> Da).
        jr, jcc = [], []
        for l, (i, j) in enumerate(cp):
            for r in range(Da):
                row = l * Da + r
                for c in range(n_theta):
                    jr.append(row); jcc.append(c)
                for c in range(Da):
                    jr.append(row); jcc.append(n_theta + i * Da + c)
                jr.append(row); jcc.append(n_theta + j * Da + r)
        jac_rows_a = np.asarray(jr, dtype=np.int64)
        jac_cols_a = np.asarray(jcc, dtype=np.int64)

    def _fwd_all(theta_norm, y_norm):
        """vmap ``F_aug`` over all segments -> (Y_next (n_seg,Da), Meas (n_seg,n_meas))."""
        theta_phys = _denorm(theta_norm)
        y_phys = y_from_norm(y_norm)
        Yn, Meas = vmap(lambda yi, ci: composer.F_aug(yi, theta_phys, ci))(y_phys, CAP)
        return y_to_norm(Yn), Meas

    def _mse_of_z(zt):
        _, Meas = _fwd_all(zt[:n_theta], zt[n_theta:].reshape(n_seg, Da))
        return (((ACT - Meas) / SD_meas) ** 2)[_incl].mean()

    def _obj_compute_fast(z):
        key = z.tobytes()
        if _c["key"] == key:
            return _c["f"], _c["gf"]
        self._eval_count += 1
        zt = torch.tensor(z, dtype=torch.float64)
        with torch.no_grad():
            _, Meas = _fwd_all(zt[:n_theta], zt[n_theta:].reshape(n_seg, Da))
            mse = float((((ACT - Meas) / SD_meas) ** 2)[_incl].mean())
            self._last_rmse = float(((ACT - Meas) ** 2)[_incl].mean()) ** 0.5
        gf = torch.func.grad(_mse_of_z)(zt).numpy()
        _c.update(key=key, f=mse, gf=gf)
        if self._eval_count % 10 == 1:
            LOGGER.iter("eval=%d | obj=%.6f | rmse=%.4f", self._eval_count, mse, self._last_rmse)
        return mse, gf

    def g_fun_fast(z):
        zt = torch.tensor(np.asarray(z, dtype=np.float64), dtype=torch.float64)
        y_norm = zt[n_theta:].reshape(n_seg, Da)
        with torch.no_grad():
            Y_next, _ = _fwd_all(zt[:n_theta], y_norm)
            defect = Y_next[cp_i] - y_norm[cp_j]
        return defect.reshape(-1).numpy()

    def _end_norm_fn(y_norm_i, theta_norm, captured_i):
        Yn, _ = composer.F_aug(y_from_norm(y_norm_i), _denorm(theta_norm), captured_i)
        return y_to_norm(Yn)

    # One vmap(jacrev) evaluates d(end_norm, meas/SD)/d(y, theta) for EVERY
    # segment; the constraint Jacobian (rows :Da at the linked segments) and the
    # Gauss-Newton Hessian (rows Da: at the scored segments) are slices of it.
    # IPOPT asks for jac_g and hess_lag at the same iterate, so the cache makes
    # the Hessian nearly free on top of the Jacobian we already pay for.
    _dcache = {"key": None}

    def _derivs(z):
        key = z.tobytes()
        if _dcache["key"] == key:
            return _dcache
        zt = torch.tensor(z, dtype=torch.float64)
        theta_norm = zt[:n_theta]
        y_norm = zt[n_theta:].reshape(n_seg, Da)

        def _both(y_i, th, cap_i):
            Yn, meas = composer.F_aug(y_from_norm(y_i), _denorm(th), cap_i)
            return torch.cat([y_to_norm(Yn), meas / SD_meas])

        # One traced pass for BOTH Jacobians (the vjp sweeps over the output
        # rows are shared), instead of two separate vmap(jacrev) evaluations.
        Jx, Jt = vmap(
            lambda yi, ci: jacrev(_both, argnums=(0, 1))(yi, theta_norm, ci)
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
        neg1 = -torch.ones((n_links, Da, 1), dtype=torch.float64)
        return torch.cat([Jt, Jx, neg1], dim=2).reshape(-1).numpy().astype(np.float64)

    def g_jac_vals_fd(z):
        z = np.asarray(z, dtype=np.float64)
        zt = torch.tensor(z, dtype=torch.float64, requires_grad=True)
        _, _, end_norm = _simulate(zt[:n_theta], zt[n_theta:].reshape(n_seg, D))
        J_s = []
        for r in range(D):
            go = torch.zeros_like(end_norm)
            go[:, r] = 1.0
            (gz,) = torch.autograd.grad(end_norm, zt, grad_outputs=go, retain_graph=True)
            J_s.append(gz[n_theta:].reshape(n_seg, D).detach())
        end0 = end_norm.detach()
        eps = 1e-6
        J_theta = []
        for c in range(n_theta):
            zp = z.copy(); zp[c] += eps
            ztp = torch.tensor(zp, dtype=torch.float64)
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
                    exc, traceback.format_exc(),
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
            zp = z0_use.copy(); zp[c] += eps
            fd = (gfun(zp) - g0) / eps
            ana = np.zeros(ng)
            for r, v in colmap[c].items():
                ana[r] = v
            denom = max(1.0, float(np.abs(fd).max()))
            d = float(np.abs(ana - fd).max()) / denom
            worst = max(worst, d)
            LOGGER.config(
                "JAC-CHECK col %d (%s): max|ana-fd|/scale = %.3e  (|fd|max=%.3e)",
                c, "theta" if c < n_theta else "state", d, float(np.abs(fd).max()),
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
        y = from_norm(torch.tensor(z_full[n_theta:], dtype=torch.float64).reshape(n_seg, dim))
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
        "composer(augmented)" if composer is not None else "finite-diff", n_fb,
    )

    # Timestepping micro-benchmark (env TWIN4BUILD_BENCH_TIMESTEP): compute every
    # segment's one-step transition both ways -- the new vmap(F_aug) map vs the old
    # object-graph batched simulate (do_step traversal) -- and report wall time and
    # agreement.  This isolates *why* collocation is fast: the per-evaluation
    # forward cost that both the objective and the defects pay each iteration.
    if _os.environ.get("TWIN4BUILD_BENCH_TIMESTEP") and composer is not None and CAP is not None:
        import time as _time

        th0 = torch.tensor(z0_a[:n_theta], dtype=torch.float64)
        y0 = torch.tensor(z0_a[n_theta:], dtype=torch.float64).reshape(n_seg, Da)
        s0 = torch.tensor(z0[n_theta:], dtype=torch.float64).reshape(n_seg, D)
        with torch.no_grad():
            Yn, _ = _fwd_all(th0, y0)                 # warm up vmap path
            _, _, end_old = _simulate(th0, s0)        # warm up object-graph path
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
            n_seg, t_new * 1e3, t_old * 1e3, (t_old / t_new if t_new else float("nan")), agree,
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
        zt = torch.tensor(z, dtype=torch.float64)
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
            # Real object-graph (do_step) rollout from the same estimated
            # initial component states and theta.  Divergence from the F_aug
            # rollout is *model mismatch* between the composed map and do_step
            # (e.g. control loops cut by frozen "captured" inputs).
            Meas_step = torch.zeros_like(Meas_nlp)
            param_values = self._theta_to_param_values(zt[:n_theta])
            self.simulator.model.set_parameters(
                param_values, self._flat_components, self._parameter_names,
                normalized=True, overwrite=True,
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
                    start_time=[seg_starts[s0]], end_time=[seg_ends[chain[-1]]],
                    step_size=[seg_steps[s0]], show_progress_bar=False,
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
            "per_sensor": {},
        }
        for m, md in enumerate(md_list):
            e_nlp = float(torch.sqrt((((ACT[:, m] - Meas_nlp[:, m]) ** 2)[_incl]).mean()))
            e_roll = float(torch.sqrt((((ACT[:, m] - Meas_roll[:, m]) ** 2)[_incl]).mean()))
            e_step = float(torch.sqrt((((ACT[:, m] - Meas_step[:, m]) ** 2)[_incl]).mean()))
            audit["per_sensor"][md.id] = {
                "nlp_rmse": e_nlp, "rollout_rmse": e_roll, "do_step_rmse": e_step,
            }
        # Active box bounds on the (non-pinned) boundary variables.
        lb_m = torch.tensor(lb_a[n_theta:], dtype=torch.float64).reshape(n_seg, Da)
        ub_m = torch.tensor(ub_a[n_theta:], dtype=torch.float64).reshape(n_seg, Da)
        free = (ub_m - lb_m) > 1e-12
        at_bound = ((y_norm - lb_m).abs() < 1e-6) | ((ub_m - y_norm).abs() < 1e-6)
        audit["n_active_state_bounds"] = int(at_bound[free].sum())
        audit["n_free_state_vars"] = int(free.sum())
        th_lb = torch.tensor(lb_a[:n_theta], dtype=torch.float64)
        th_ub = torch.tensor(ub_a[:n_theta], dtype=torch.float64)
        audit["n_theta_at_bounds"] = int(
            (((theta_norm - th_lb).abs() < 1e-6) | ((th_ub - theta_norm).abs() < 1e-6)).sum()
        )
        LOGGER.result(
            "AUDIT: status=%s | max|defect|=%.3e (normalized units) | boundary "
            "vars at box bound: %d/%d | theta at bound: %d/%d",
            audit["return_status"], max_defect,
            audit["n_active_state_bounds"], audit["n_free_state_vars"],
            audit["n_theta_at_bounds"], n_theta,
        )
        for mid, e in audit["per_sensor"].items():
            LOGGER.result(
                "AUDIT %-32s NLP-internal RMSE=%.4f | sequential F_aug rollout "
                "RMSE=%.4f | do_step rollout RMSE=%.4f (raw units)",
                mid, e["nlp_rmse"], e["rollout_rmse"], e["do_step_rmse"],
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
        if gauss_newton:
            # Termination: GN drops the constraint curvature (lam_g * d2g), so
            # the dual infeasibility plateaus (oscillating ~1e-3..1e-1) and
            # IPOPT's default tol=1e-8 is unreachable -- it then burns hundreds
            # of iterations polishing duals with ZERO objective progress before
            # dying in restoration.  Stop at the plateau instead: accept when
            # the iterate is feasible (defects ~1e-9 here) and the objective
            # has stagnated (<1e-5 relative change) for 10 consecutive
            # iterations, ignoring the (unreachable) dual/complementarity
            # criteria.  Callers can still override any of these via options.
            options = dict(options or {})
            options.setdefault("acceptable_tol", 1e3)
            options.setdefault("acceptable_iter", 5)
            # Loosened to 1e-2 (normalized defects): the full-horizon problem
            # plateaus with max|defect| ~ 1e-3 that the audit shows is benign
            # (NLP-internal fit == sequential rollout), and a tighter gate kept
            # this exit from ever firing there.
            options.setdefault("acceptable_constr_viol_tol", 1e-2)
            options.setdefault("acceptable_dual_inf_tol", 1e10)
            options.setdefault("acceptable_compl_inf_tol", 1e3)
            options.setdefault("acceptable_obj_change_tol", 1e-4)
            incl_np = np.nonzero(_incl.numpy())[0]
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
            incl_t = torch.tensor(incl_np, dtype=torch.long)
            gn_scale = 2.0 / float(n_i * len(md_list))

            def hess_vals_fn(z, sigma):
                d = _derivs(np.asarray(z, dtype=np.float64))
                Jt_ = d["Jt"][incl_t, Da:, :]  # (n_i, n_meas, n_theta)
                Jy = d["Jx"][incl_t, Da:, :]   # (n_i, n_meas, Da)
                Js = torch.cat([Jt_, Jy], dim=2)  # (n_i, n_meas, nt+Da)
                B = (torch.einsum("gmi,gmj->gij", Js, Js) * (float(sigma) * gn_scale)).numpy()
                vals = [B[:, :n_theta, :n_theta].sum(axis=0)[iu_t]]
                Bty = B[:, :n_theta, n_theta:]
                Byy = B[:, n_theta:, n_theta:]
                for k in range(n_i):
                    vals.append(Bty[k].ravel())
                    vals.append(Byy[k][iu_y])
                return np.concatenate(vals)

            LOGGER.config(
                "Gauss-Newton Hessian enabled: %d nonzeros (upper triangle), "
                "%d scored segments.", len(hess_rows), n_i,
            )
            if _os.environ.get("TWIN4BUILD_HESS_CHECK"):
                # The gradient identity grad f = (2/N) J^T r is EXACT (GN only
                # truncates the Hessian), so matching the autograd gradient
                # validates the measurement Jacobians, scaling and assembly.
                zt0 = torch.tensor(np.asarray(z0_a, dtype=np.float64), dtype=torch.float64)
                th0_ = zt0[:n_theta]
                with torch.no_grad():
                    _, Meas0_raw = _fwd_all(th0_, zt0[n_theta:].reshape(n_seg, Da))
                r0 = (Meas0_raw[incl_t] - ACT[incl_t]) / SD_meas
                d0 = _derivs(np.asarray(z0_a, dtype=np.float64))
                Jt0 = d0["Jt"][incl_t, Da:, :]
                Jy0 = d0["Jx"][incl_t, Da:, :]
                g_gn = np.zeros(len(z0_a))
                g_gn[:n_theta] = (gn_scale * torch.einsum("gmt,gm->t", Jt0, r0)).numpy()
                gy = (gn_scale * torch.einsum("gmd,gm->gd", Jy0, r0)).numpy()
                for k, g in enumerate(incl_np):
                    a = n_theta + int(g) * Da
                    g_gn[a:a + Da] = gy[k]
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
            float(np.abs(g0).max()) if g0.size else 0.0, obj_fun(z0_a),
        )
        if g0.size:
            G0 = np.abs(g0).reshape(n_links, Da)
            worst = G0.max(axis=0)
            dim_labels = []
            for _comp, (_a, _b) in zip(layout.components, layout.slices):
                dim_labels += [f"{_comp.id}[{k}]" for k in range(_b - _a)]
            dim_labels += ["fb:" + ".".join(map(str, k)) for k in composer._feedback_keys]
            top = np.argsort(-worst)[:5]
            LOGGER.config(
                "Warm-start worst defect dims (normalized): %s",
                [
                    (dim_labels[i], round(float(worst[i]), 3),
                     f"link={int(G0[:, i].argmax())}")
                    for i in top
                ],
            )
        _jac_selfcheck(z0_a, jac_rows_a, jac_cols_a, n_g_a, g_fun_fast, g_jac_vals_fast, Da)
        es_cfg = None
        if early_stopping:
            es_cfg = dict(early_stopping) if isinstance(early_stopping, dict) else {}
            es_cfg.setdefault("n_theta", n_theta)
            LOGGER.config(
                "Early stopping enabled: feas_tol=%s patience=%s "
                "min_delta_rel=%s theta_tol=%s",
                es_cfg.get("feas_tol", 1e-2), es_cfg.get("patience", 10),
                es_cfg.get("min_delta_rel", 1e-3), es_cfg.get("theta_tol", 1e-4),
            )
        result = solve_ipopt_constrained(
            z0_a, lb_a, ub_a, obj_fun, obj_grad, n_g_a, g_fun_fast, g_jac_vals_fast,
            jac_rows_a, jac_cols_a, options=options,
            hess_vals=hess_vals_fn, hess_rows=hess_rows, hess_cols=hess_cols,
            early_stopping=es_cfg,
        )
        _attach_x0(result, result.x, Da, y_from_norm)
        _audit_fast(result)
        result.x = np.asarray(result.x, dtype=np.float64)[:n_theta]
        result.nfev = self._eval_count
        return result

    if pin_initial_state and (composer is None or CAP is None):
        for s0 in period_starts:
            a = n_theta + s0 * D
            lb[a:a + D] = z0[a:a + D]
            ub[a:a + D] = z0[a:a + D]
        LOGGER.config(
            "Pinned the initial state of %d period(s) at the warm-start value "
            "(bound equality).", len(period_starts),
        )
    _jac_selfcheck(z0, jac_rows, jac_cols, n_g, g_fun, g_jac_vals, D)
    result = solve_ipopt_constrained(
        z0, lb, ub, obj_fun, obj_grad, n_g, g_fun, g_jac_vals,
        jac_rows, jac_cols, options=options,
    )
    _attach_x0(result, result.x, D, s_from_norm)
    result.x = np.asarray(result.x, dtype=np.float64)[:n_theta]
    result.nfev = self._eval_count
    return result
