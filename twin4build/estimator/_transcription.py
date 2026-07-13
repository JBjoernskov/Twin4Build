"""Multiple-shooting / collocation transcription for :class:`Estimator`.

Single-shooting estimation evaluates the objective by simulating the *entire*
horizon from one fixed initial condition, then backpropagating through the full
unrolled trajectory.  Over long horizons this composition-of-``f``-with-itself
is badly conditioned (the exploding/vanishing-gradient problem of backprop
through time) and every iterate is a full rollout.

**Multiple-shooting** breaks the horizon into ``K`` segments and promotes each
segment's *initial state* ``s_i`` to a decision variable, stacked alongside the
physical parameters ``theta``.  The dynamics are enforced by *continuity
defects*

    d_i = x(t_{i+1}; s_i, theta) - s_{i+1}      (i = 0 .. K-2)

where the first term is what the simulator produces running segment ``i``
forward from ``s_i``, and ``s_{i+1}`` is read directly off the decision vector.
Gradients only ever flow through one segment's (short) rollout, so the
conditioning improves and the basins of attraction widen.  Taking ``K`` to one
segment per timestep recovers full simultaneous transcription (**collocation**).

This module implements the **soft** (penalty) form -- the defects enter the
objective as ``lambda * sum ||d_i||^2`` -- which needs only an unconstrained
scalar objective and therefore works with every backend, including the CasADi
IPOPT wrapper (:mod:`twin4build.estimator._casadi_ipopt`) and SciPy.  The
segment states are *nuisance* variables: they are discarded after the fit
(``EstimationResult`` reports only ``theta``).

Segments are mapped onto the simulator's existing period-batch dimension
(``n_s``): ``K`` segments become ``K`` parallel "periods", each initialized to
its own ``s_i`` via the ``after_initialize`` hook on
:meth:`Simulator.simulate`.  The per-segment initial states are seeded from a
single forward simulation at ``theta0`` (a warm start), so the optimizer begins
from a dynamically-consistent trajectory.
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
    """Run a multiple-shooting / collocation estimation solve.

    Parameters
    ----------
    estimator : Estimator
        The calling estimator, fully set up by ``estimate`` (parameters,
        measurements, normalized bounds ``_x0_norm``/``_lb_norm``/``_ub_norm``,
        ``actual_readings``, and a single time period in ``_start_time`` etc.).
    method : tuple
        ``(library, optimizer, mode)`` -- the optimizer backend for the NLP.
    options : dict
        Solver options; recognises ``n_segments`` (default 10; ``collocation``
        transcription overrides it to one-per-timestep) and
        ``continuity_lambda`` (default 100.0, the soft-defect weight).

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

    start_times = list(self._start_time)
    end_times = list(self._end_time)
    step_sizes = list(self._stepSize)
    continuity_lambda = float(options.pop("continuity_lambda", 100.0))
    n_segments_opt = (
        None if self._transcription == "collocation"
        else int(options.pop("n_segments", 10))
    )

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
    # Each period is split into contiguous segments; the segments of *all*
    # periods live together on the simulator's n_s axis.  Continuity defects
    # link only *consecutive segments within the same period* -- disjoint
    # training windows are independent experiments (no cross-window stitching).
    seg_starts: List[datetime.datetime] = []
    seg_ends: List[datetime.datetime] = []
    seg_steps: List[int] = []
    seg_len: List[int] = []
    warm_states: List[torch.Tensor] = []
    continuity_pairs: List[Tuple[int, int]] = []
    seg_actual: Dict[str, List[torch.Tensor]] = {md.id: [] for md, _ in self._measurements}

    g = 0
    for p, (s_p, e_p, step_p) in enumerate(zip(start_times, end_times, step_sizes)):
        _, _, n_t_p, _ = core.Simulator.get_simulation_timesteps(s_p, e_p, step_p)
        K_p = n_t_p if n_segments_opt is None else n_segments_opt
        bounds_p = _segment_boundaries(n_t_p, K_p)
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
            for md, _ in self._measurements:
                seg_actual[md.id].append(
                    torch.tensor(actual_p[md.id][bounds_p[i]:bounds_p[i + 1]], dtype=torch.float64)
                )
            if i < Kp - 1:
                continuity_pairs.append((g, g + 1))
            g += 1
    n_seg = g

    LOGGER.config(
        "Transcription: %s | %d periods | %d segments total | %d continuity links | lambda=%.3g",
        self._transcription, len(start_times), n_seg, len(continuity_pairs), continuity_lambda,
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

    # Continuity index tensors (end-of-segment i vs. start-var of segment i+1).
    if continuity_pairs:
        cp_i = torch.tensor([a for a, _ in continuity_pairs], dtype=torch.long)
        cp_j = torch.tensor([b for _, b in continuity_pairs], dtype=torch.long)
    else:
        cp_i = cp_j = None

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

    def _loss(zt: torch.Tensor) -> torch.Tensor:
        """Scalar transcription loss (data-fit MSE + lambda * continuity).

        ``zt`` is a torch tensor of the full decision vector ``[theta_norm |
        s_norm]``; the returned scalar keeps the autograd graph back to ``zt``.
        """
        theta = zt[:n_theta]
        s_norm = zt[n_theta:].reshape(n_seg, D)
        s_phys = s_from_norm(s_norm)  # (n_seg, D)

        # Physical parameters from theta.
        param_values = self._theta_to_param_values(theta)
        self.simulator.model.set_parameters(
            param_values, self._flat_components, self._parameter_names,
            normalized=True, overwrite=True,
        )

        # Simulate every segment in parallel; inject per-segment initial states
        # after (re)initialization via the simulator hook.
        self.simulator.simulate(
            start_time=seg_starts, end_time=seg_ends, step_size=seg_steps,
            show_progress_bar=False, after_initialize=lambda: layout.scatter(s_phys),
        )

        # Data-fit residuals over every (sensor, segment).  Track both the
        # sd-scaled residual (drives the objective) and the raw residual (for
        # an RMSE diagnostic comparable to the single-shooting path, which
        # reports raw RMSE in measurement units).
        res_terms = []
        res_raw_terms = []
        for md, sd in self._measurements:
            for gi in range(n_seg):
                L = seg_len[gi]
                y_model = md.input["measuredValue"].history(i_t=slice(0, L), i_s=gi, i_c=0)
                raw = seg_actual[md.id][gi] - y_model
                res_raw_terms.append(raw)
                res_terms.append(raw / sd)
        mse = torch.mean(torch.cat(res_terms) ** 2)
        raw_mse = torch.mean(torch.cat(res_raw_terms) ** 2)

        # Continuity: each segment's final state vs. the next segment's start var
        # within the same period (both normalized so the penalty is well-scaled).
        if cp_i is not None:
            end_norm = s_to_norm(layout.end_states(n_seg))  # (n_seg, D)
            defect = end_norm[cp_i] - s_norm[cp_j]  # (n_links, D)
            continuity = torch.sum(defect ** 2) / max(1, len(continuity_pairs) * D)
        else:
            continuity = torch.zeros((), dtype=torch.float64)

        self._last_mse = float(raw_mse.detach())
        self._last_rmse = float(raw_mse.detach()) ** 0.5  # raw units, comparable
        self._last_continuity = float(continuity.detach())
        return mse + continuity_lambda * continuity

    # Value+gradient are computed together and cached, so IPOPT's back-to-back
    # objective/gradient calls at the same z trigger only one simulation.
    _cache = {"key": None, "val": None, "grad": None}

    def _compute(z: np.ndarray):
        key = z.tobytes()
        if _cache["key"] == key:
            return _cache["val"], _cache["grad"]
        self._eval_count += 1
        zt = torch.tensor(z, dtype=torch.float64, requires_grad=True)
        val_t = _loss(zt)
        (grad_t,) = torch.autograd.grad(val_t, zt)
        val = float(val_t.detach())
        grad = grad_t.detach().numpy()
        _cache.update(key=key, val=val, grad=grad)
        if self._eval_count % 10 == 1:
            LOGGER.iter(
                "eval=%d | obj=%.6f | rmse=%.4f | continuity=%.3g",
                self._eval_count, val, self._last_rmse, self._last_continuity,
            )
        return val, grad

    def fun(z: np.ndarray) -> float:
        return _compute(np.asarray(z, dtype=np.float64))[0]

    def jac(z: np.ndarray) -> np.ndarray:
        return _compute(np.asarray(z, dtype=np.float64))[1]

    LOGGER.config("Decision variables: %d (theta=%d, states=%d)", len(z0), n_theta, n_seg * D)

    # ---- Stage 2: sparse, hard-constraint collocation ----------------------
    # For "collocation" on the CasADi/IPOPT backend, enforce the dynamics as
    # hard equality *defect constraints* with an explicit block-bidiagonal
    # Jacobian, rather than as a soft penalty.  IPOPT's sparse linear solver
    # then exploits the structure (each defect row touches only s_i, s_{i+1},
    # theta).  The objective becomes data-fit only; continuity moves into g.
    if self._transcription == "collocation" and method[0] == "casadi" and continuity_pairs:
        return _solve_sparse_collocation(
            self, method, options, n_theta, D, n_seg, layout, seg_starts, seg_ends,
            seg_steps, seg_len, seg_actual, continuity_pairs, s_to_norm, s_from_norm,
            z0, lb, ub,
        )

    # ---- Dispatch to the optimizer backend (soft-penalty path) -------------
    if method[0] == "casadi":
        from twin4build.estimator._casadi_ipopt import solve_ipopt

        result = solve_ipopt(z0, lb, ub, fun, jac, options=options)
    else:
        from scipy.optimize import minimize, Bounds

        opt = minimize(
            fun, z0, jac=jac, bounds=Bounds(lb, ub),
            method=method[1] if method[1] in ("L-BFGS-B", "SLSQP", "TNC") else "L-BFGS-B",
            options=options or None,
        )
        result = SimpleNamespace(
            x=opt.x, fun=float(opt.fun), success=bool(opt.success),
            nit=getattr(opt, "nit", None), message=str(getattr(opt, "message", "")),
        )

    # Return only theta to the Estimator's result tail; states are nuisance.
    result.x = np.asarray(result.x, dtype=np.float64)[:n_theta]
    result.nfev = self._eval_count
    return result


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
    z0, lb, ub,
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
                # Plain (functorch-safe) denormalization from the parameters'
                # physical bounds + scaling (tps.Parameter.denormalize is a
                # Tensor-subclass method and breaks under functorch).
                tp = self._flat_parameters
                lb_t = torch.tensor([float(np.asarray(p.min_value.detach()).flatten()[0]) for p in tp])
                ub_t = torch.tensor([float(np.asarray(p.max_value.detach()).flatten()[0]) for p in tp])
                log_mask = torch.tensor([getattr(p, "scaling", "linear") == "log" for p in tp])
                composer = comp
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Composer unavailable (%s) -- using finite-difference Jacobian.", exc)
        composer = None

    def _denorm(th_norm):
        lin = lb_t + th_norm * (ub_t - lb_t)
        safe_lb = lb_t.clamp(min=1e-30)
        safe_ub = ub_t.clamp(min=1e-30)
        logv = torch.exp(torch.log(safe_lb) + th_norm * (torch.log(safe_ub) - torch.log(safe_lb)))
        return torch.where(log_mask, logv, lin)

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

    def _capture_inputs():
        """Sample the captured (exogenous + carried) input values per segment from
        the current model state (post-``_simulate``).

        Collocation segments are one step long, so each cone component's input
        port ``.get()`` holds exactly the value used at that segment's step.
        """
        comps = self.simulator.model.components
        cap = torch.zeros((n_seg, len(composer._captured_keys)), dtype=torch.float64)
        for k, (cid, port) in enumerate(composer._captured_keys):
            v = comps[cid].input[port].get()  # (n_s=n_seg, n_c)
            cap[:, k] = v.reshape(n_seg, -1)[:, 0].detach()
        return cap

    cp_i = torch.tensor([i for i, _ in cp], dtype=torch.long)
    cp_j = torch.tensor([j for _, j in cp], dtype=torch.long)

    def _capture_ports(keys):
        """Sample per-segment input-port values for ``keys`` (comp_id, port) from
        the current model state.  Segments are one step, so ``.get()`` holds the
        value used at that segment's step."""
        comps = self.simulator.model.components
        out = torch.zeros((n_seg, len(keys)), dtype=torch.float64)
        for k, (cid, port) in enumerate(keys):
            out[:, k] = comps[cid].input[port].get().reshape(n_seg, -1)[:, 0].detach()
        return out

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
        CAP = _capture_ports(composer._captured_keys)      # (n_seg, n_exo), frozen
        fb0_raw = _capture_ports(composer._feedback_keys)  # fresh-init guess (inconsistent)
        # Consistent (delayed) feedback warm start.  Capturing from the batched
        # 1-step segments gives each segment's *fresh-init* feedback, which
        # violates the delayed continuity w_{t+1} = producer_output(y_t) by ~1e5
        # -> IPOPT stalls at an infeasible point.  Instead run F on the warm-start
        # states to get the producer outputs and shift by one segment.
        theta0_phys = _denorm(torch.tensor(z0[:n_theta], dtype=torch.float64))
        s0_phys = s_from_norm(torch.tensor(z0[n_theta:], dtype=torch.float64).reshape(n_seg, D))
        with torch.no_grad():
            _, _, FBout = vmap(
                lambda si, ci, wi: composer.F(si, theta0_phys, ci, wi)
            )(s0_phys, CAP, fb0_raw)
        fb0 = fb0_raw.clone()
        if n_fb:
            fb0[1:] = FBout[:-1]   # feedback at t+1 == producer output at t
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
        return torch.mean(((ACT - Meas) / SD_meas) ** 2)

    def _obj_compute_fast(z):
        key = z.tobytes()
        if _c["key"] == key:
            return _c["f"], _c["gf"]
        self._eval_count += 1
        zt = torch.tensor(z, dtype=torch.float64)
        with torch.no_grad():
            _, Meas = _fwd_all(zt[:n_theta], zt[n_theta:].reshape(n_seg, Da))
            mse = float(torch.mean(((ACT - Meas) / SD_meas) ** 2))
            self._last_rmse = float(torch.mean((ACT - Meas) ** 2)) ** 0.5
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

    def g_jac_vals_fast(z):
        """Sparse constraint Jacobian via one vmap(jacrev(F_aug)) over the composer."""
        zt = torch.tensor(np.asarray(z, dtype=np.float64), dtype=torch.float64)
        theta_norm = zt[:n_theta]
        y_norm = zt[n_theta:].reshape(n_seg, Da)
        Y_i = y_norm[cp_i]          # (n_links, Da)
        CAP_i = CAP[cp_i]           # (n_links, n_exo)
        Jx = vmap(lambda yi, ci: jacrev(_end_norm_fn, argnums=0)(yi, theta_norm, ci))(Y_i, CAP_i)
        Jt = vmap(lambda yi, ci: jacrev(_end_norm_fn, argnums=1)(yi, theta_norm, ci))(Y_i, CAP_i)
        J_s = [Jx[:, r, :].detach() for r in range(Da)]           # J_s[r][l, c]
        J_theta = [Jt[:, :, c].detach() for c in range(n_theta)]  # J_theta[c][l, r]
        vals = []
        for l, (i, j) in enumerate(cp):
            for r in range(Da):
                for c in range(n_theta):
                    vals.append(float(J_theta[c][l, r]))
                for c in range(Da):
                    vals.append(float(J_s[r][l, c]))
                vals.append(-1.0)
        return np.asarray(vals, dtype=np.float64)

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
        """Record the optimised *initial state* (first segment's boundary state,
        physical units) per stateful component.  ``z_full`` may be augmented with
        feedback lag variables (width ``dim`` per segment); the component states
        are the leading ``D`` entries."""
        z_full = np.asarray(z_full, dtype=np.float64)
        y0 = from_norm(torch.tensor(z_full[n_theta:], dtype=torch.float64).reshape(n_seg, dim))[0]
        x0 = y0[:D]  # component-state part (feedback lag vars, if any, trail)
        d = {}
        for comp, (start, stop), (n_c, ss) in zip(
            layout.components, layout.slices, layout.shapes
        ):
            d[comp.id] = x0[start:stop].reshape(n_c, ss).detach().clone()
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

    # Augmented feedback-as-state collocation: the cut feedback signals are
    # decision variables (extra state) tied to their producer outputs by ordinary
    # continuity, so there is no frozen carry and no outer re-capture -- one solve.
    if composer is not None and CAP is not None:
        obj_fun = lambda z: _obj_compute_fast(np.asarray(z, dtype=np.float64))[0]
        obj_grad = lambda z: _obj_compute_fast(np.asarray(z, dtype=np.float64))[1]
        _jac_selfcheck(z0_a, jac_rows_a, jac_cols_a, n_g_a, g_fun_fast, g_jac_vals_fast, Da)
        result = solve_ipopt_constrained(
            z0_a, lb_a, ub_a, obj_fun, obj_grad, n_g_a, g_fun_fast, g_jac_vals_fast,
            jac_rows_a, jac_cols_a, options=options,
        )
        _attach_x0(result, result.x, Da, y_from_norm)
        result.x = np.asarray(result.x, dtype=np.float64)[:n_theta]
        result.nfev = self._eval_count
        return result

    _jac_selfcheck(z0, jac_rows, jac_cols, n_g, g_fun, g_jac_vals, D)
    result = solve_ipopt_constrained(
        z0, lb, ub, obj_fun, obj_grad, n_g, g_fun, g_jac_vals,
        jac_rows, jac_cols, options=options,
    )
    _attach_x0(result, result.x, D, s_from_norm)
    result.x = np.asarray(result.x, dtype=np.float64)[:n_theta]
    result.nfev = self._eval_count
    return result
