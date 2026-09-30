"""Functional single-shooting estimation objective.

Single-shooting's per-iteration cost is one full object-graph simulation plus
autograd back through it.  Profiling shows most of that cost is Python
dispatch -- the per-step Gauss-Seidel traversal over components, ``tps``
wrapper bookkeeping, history logging -- and ``model.initialize`` re-reading
every CSV input on EVERY objective evaluation.  None of it is tensor math.

This module builds on the Simulator's functional API
(:meth:`Simulator.build_functional_model` /
:meth:`Simulator.record_exogenous_inputs` /
:meth:`Simulator.rollout_functional`, implemented in
:mod:`twin4build.simulator._functional`): the model's stateful cone is composed
into a pure one-step map ``F_aug(y, theta, cap)``, the truly-exogenous inputs
are captured ONCE from a reference ``do_step`` rollout, and the objective
becomes a plain sequential torch rollout

    y_{t+1}, meas_t = F_aug(y_t, theta, CAP[t])

per training period.  Autograd back through this rollout is the same reverse
pass single-shooting already paid for -- minus the object-graph overhead.

Exactness: **by construction**.  Every composable component's ``do_step`` is a
thin port-I/O wrapper that DELEGATES its math to the same ``forward`` the
composer threads (single source of truth -- the two cannot drift apart), and
cut feedback edges are carried as one-step lag state inside ``y`` (exactly
``do_step``'s delayed Gauss-Seidel semantics).  Only truly exogenous inputs
(weather, schedules, measured data series) are frozen -- and those are
theta-independent by definition; ``OneStepComposer._validate_theta_influence``
refuses to compose if a theta path would leak into a frozen signal.  The same single-source-of-truth rule holds
outside the components: theta denormalization is
:func:`twin4build.utils.types.denormalize_unit` (the function
``tps.Parameter.denormalize`` itself routes through) and everything downstream
of the raw residuals (sd weighting, MSE normalization, rescale-to-100,
diagnostics) is ``Estimator._loglike_from_residuals`` -- the method the
object-graph ``_obj`` ends in as well.  Shared parameters are supported: the
indexed theta spec routes every member of a shared group to the same theta
slot.  Construction performs the structural checks and the estimator silently
falls back to the exact path for un-composable models (components without
``forward`` or a measurement the composed map cannot produce).
``tests/estimator/test_functional_single_shooting.py`` regression-checks the end-to-end
value + gradient parity (guards the delegation contract and the composer's
wiring/capture logic).

Enable with ``Simulator(model, execution_mode="functional")``.
"""

from __future__ import annotations

import numpy as np
import os

import torch

import twin4build.utils.types as tps
from twin4build.utils.logger import LOGGER
from twin4build.utils.types import denormalize_unit, theta_bound_tensors


#: Roll equal-length periods out as one batch of windows (one wide step
#: per time index) instead of one after another.  ``T4B_WINDOW_BATCHING=0``
#: keeps the sequential rollout.
WINDOW_BATCHING = os.environ.get("T4B_WINDOW_BATCHING", "1").strip().lower() not in {"0", "false", "no"}


def _period_mask(mask, index, n_t: int) -> np.ndarray:
    """The device's ``scoring_mask`` for one period, ``n_t`` bools (True =
    scored).  A time-indexed ``pandas.Series`` is selected by the period's
    timestamps (a mask over a span of several periods; a time it does not
    cover is scored), a plain array is taken as the period's own steps
    (padded with True when short)."""
    if hasattr(mask, "reindex") and hasattr(mask, "index"):
        ns = _to_ns(mask.index)
        want = _to_ns(index)[:n_t]
        pos = np.searchsorted(ns, want)
        pos = np.clip(pos, 0, max(len(ns) - 1, 0))
        hit = (len(ns) > 0) & (ns[pos] == want)
        keep = np.ones(n_t, dtype=bool)
        keep[hit] = np.asarray(mask.to_numpy(), dtype=bool)[pos[hit]]
        if len(want) < n_t:
            keep = np.concatenate([keep[: len(want)], np.ones(n_t - len(want), dtype=bool)])
        return keep
    keep = np.asarray(mask, dtype=bool)[:n_t]
    if keep.shape[0] < n_t:
        keep = np.concatenate([keep, np.ones(n_t - keep.shape[0], dtype=bool)])
    return keep


def _to_ns(index) -> np.ndarray:
    """A DatetimeIndex (naive or tz-aware) as int64 nanoseconds since the epoch, UTC."""
    import pandas as pd

    idx = pd.DatetimeIndex(index)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    return idx.asi8


class FunctionalEstimationObjective:
    """Functional-model drop-in for :meth:`Estimator._obj` (scalar mode).

    Built AFTER the estimator's ``estimate`` setup (parameters, measurements,
    normalized bounds, ``actual_readings`` all populated, model initialized at
    ``x0``).  Construction raises on any model feature the composer cannot
    express; callers treat that as "use the object-graph objective".
    """

    def __init__(self, estimator):
        self.est = estimator

        if getattr(estimator, "_regularization_lambda", 0) > 0:
            raise RuntimeError("regularization penalty not supported")
        n_theta = len(estimator._x0_norm)

        # Indexed theta spec: shared parameters route several (comp, attr)
        # entries to one theta selector; compiled private parameters use a
        # per-branch slice.
        theta_spec, unique_parameters = estimator._composer_theta_spec()

        # Structural checks live in Simulator.build_functional_model.
        layout, composer = estimator.simulator.build_functional_model(
            theta_spec=theta_spec,
            measurements=[md for md, _ in estimator._measurements],
            step_size=estimator._stepSize,
        )
        if not composer.meas_sources:
            raise RuntimeError("no measurement sources")
        if any(s[0] != "fresh" for s in composer.meas_sources):
            # A frozen measurement would make its residual theta-independent:
            # the functional objective would silently ignore that sensor.
            raise RuntimeError(
                "a measurement is not producible by the functional model"
            )
        if any(s.stop - s.start != 1 for s in composer.meas_slices):
            raise RuntimeError(
                "each estimator measurement must resolve to exactly one branch"
            )

        self.layout = layout
        self.composer = composer
        self.n_theta = n_theta

        # Plain (functorch-safe) denormalization from the physical bounds --
        # one representative parameter per unique theta entry.
        self._lb_t, self._ub_t, self._log_mask = theta_bound_tensors(
            unique_parameters, device=estimator._device
        )

        # Sensor lag: a pass-through sensor that executes BEFORE its producer
        # in the Gauss-Seidel order reads the producer's PREVIOUS-step output
        # (e.g. office_co2_sensor runs before office).  The object-graph
        # objective therefore scores a one-step-lagged signal for that sensor;
        # F_aug returns the current step's.  Shift those columns to match.
        self.meas_lag = []
        for (md, _sd), spec in zip(estimator._measurements, composer.meas_sources):
            lag = spec[0] == "fresh" and composer.pos[md.id] < composer.pos[spec[1]]
            self.meas_lag.append(bool(lag))
        if any(self.meas_lag):
            LOGGER.config(
                "Functional single-shooting: one-step sensor lag on %s",
                [
                    md.id
                    for (md, _), l in zip(estimator._measurements, self.meas_lag)
                    if l
                ],
            )
        self._record_exogenous_inputs()
        self._sd = torch.tensor(
            [float(sd) for _md, sd in estimator._measurements],
            dtype=tps.float_dtype(),
            device=estimator._device,
        )
        self._denom = float(estimator._n_timesteps * len(estimator._measurements))
        # Multiple shooting: the windows' initial states as decision variables
        # with continuity defects between consecutive windows as residual
        # columns (``estimator._multiple_shooting``); ``n_init == 0`` without.
        self.n_init = 0
        self._setup_multiple_shooting(getattr(estimator, "_multiple_shooting", None))
        # A single canonical scale makes every start and every custom solver
        # optimize exactly the same function.  Per-start scaling would make
        # objective values and stopping tolerances incomparable.
        with torch.no_grad():
            x0 = torch.as_tensor(
                estimator._x0_norm,
                dtype=tps.float_dtype(),
                device=estimator._device,
            )
            weighted = self.raw_residuals(x0) / self._sd
            mse0 = torch.sum(weighted.square()) / self._denom
            self.loss_scale = torch.clamp(
                mse0 / 100.0,
                min=torch.finfo(weighted.dtype).tiny,
            ).detach()

    def _denorm(self, th_norm: torch.Tensor) -> torch.Tensor:
        # Single source of truth for the normalized->physical map
        # (tps.Parameter.denormalize routes through the same function).
        return denormalize_unit(th_norm, self._lb_t, self._ub_t, self._log_mask)

    # -- multiple shooting: initial-state variables and continuity defects ---
    def _setup_multiple_shooting(self, config) -> None:
        """Turn the windows' initial states into decision variables.

        ``config`` (a dict, ``None`` = off):

        * ``states``: ``"all"`` (every state of the flat state vector, the
          default) or a list of ``(component class name, [state index, ...]
          or "all")`` selecting the slow states only.  A class names an
          executing component or a member of a fused state-space block; the
          indices count within that class's own state vector.
        * ``sd_ref``, ``sd_rel`` / ``sd_abs``: the continuity tolerance per
          state, ``max(sd_abs, sd_rel * reference)``.  ``sd_ref="range"``
          (the default) takes as the reference how much the state varies
          over the windows' starts and ends in a rollout at the start point:
          each state is held in its own units and to its own scale, so a
          slow state that barely moves (a building's shared interior) is
          held tight and a fast one (a radiator's water) loosely; default
          ``sd_rel`` 0.01 there.  ``sd_ref="value"`` takes the state's
          magnitude (default ``sd_rel`` 0.0025: 0.05 K on a 20 C wall).
          The tolerance is a penalty weight, not a hard limit: a fit may
          still buy a better fit of a window with a jump at its start.
        * ``shift``: the augmented-Lagrangian shift of the defects, in the
          states' own units: ``{component id: (P - 1, state_size)}`` by the
          ids of the components the model was built from (a result's
          ``continuity_shift_instances``).  The defect of a state becomes
          ``(Y_end[p] - Y0[p + 1] + shift) / sd``.  Refitting with the shift
          the last fit returned (the method of multipliers: Hestenes 1969,
          Powell 1969) drives the jumps to zero at a fixed tolerance.
        * ``continuity``: ``True`` (multiple shooting) scores the defects;
          ``False`` (``Estimator.estimate(initial_state=...)``) estimates the
          periods' initial states with no tie between them.
        * ``bound_rel`` / ``bound_abs``: the box around the recorded initial
          state, ``max(bound_abs, bound_rel * |state|)`` (default 25 %).
        * ``first_window``: the first window's initial state is a variable
          too (default ``True``; it has no defect column, its own first
          measurements pin it).  ``False`` keeps it at the recorded state.

        Every window (every window but the first with ``first_window=False``)
        gets one variable per selected state,
        normalised linearly on its box; the defect ``Y_end[p] - Y0[p + 1]``
        over the selected states, divided by the tolerance, joins the
        residual columns (one column per selected state).  The variables
        and the columns attach to the block of the state's component, so
        the block solvers see them as any other parameter and residual.
        Needs the batched window rollout (equal-length periods).
        """
        if not config:
            return
        windows = self._windows()
        if windows is None:
            LOGGER.warning(
                "multiple shooting needs several equal-length periods rolled out as windows; "
                "the initial states stay fixed"
            )
            return
        config = dict(config)
        continuity = bool(config.get("continuity", True))
        Y0, _tape = windows  # (P, D_aug)
        P = int(Y0.shape[0])
        if P < 2 and continuity:
            return
        layout = self.layout
        D = int(layout.width)
        selection = config.get("states", "all")
        if selection == "all" or selection is None:
            index = np.arange(D, dtype=np.int64)
        else:
            picked = []
            wanted = {}
            for name, idx in selection:
                wanted[str(name)] = idx
            # a class names an executing component or a member of a fused
            # state-space block (a room fused with its radiator and walls):
            # the member's states are a span of the block's state vector
            model = getattr(self.est.simulator.model, "simulation_model", self.est.simulator.model)
            members_of = {}
            for comp, owner, offset, width in model._stateful_leaves():
                if owner is not comp:
                    members_of.setdefault(comp.id, []).append((type(owner).__name__, offset, width))
            for comp, (start, stop), (n_c, ss) in zip(layout.components, layout.slices, layout.shapes):
                spans = []
                if type(comp).__name__ in wanted:
                    spans.append((0, ss, wanted[type(comp).__name__]))
                for class_name, offset, width in members_of.get(comp.id, []):
                    if class_name in wanted:
                        spans.append((offset, width, wanted[class_name]))
                for offset, width, idx in spans:
                    ks = range(width) if idx == "all" else [int(k) for k in idx if 0 <= int(k) < width]
                    for i_c in range(n_c):
                        picked.extend(start + i_c * ss + offset + int(k) for k in ks)
            index = np.asarray(sorted(set(picked)), dtype=np.int64)
            if index.size == 0:
                LOGGER.warning(
                    "multiple shooting: the states selection %s matches no state of the model; "
                    "the initial states stay fixed", sorted(wanted),
                )
        blocks = self.composer.state_blocks(layout)
        attached = blocks[index] >= 0
        if not attached.all():
            LOGGER.warning(
                "multiple shooting: %d of %d selected states belong to no estimated block and stay fixed",
                int((~attached).sum()), int(index.size),
            )
            index = index[attached]
        if index.size == 0:
            return
        dev, dtype = Y0.device, Y0.dtype
        index_t = torch.as_tensor(index, dtype=torch.long, device=dev)
        # Every window's initial state is a variable, the first window's
        # too (boxed around its recorded value and pinned by its own first
        # measurements; no defect column, there is no window before it), so
        # no window needs a warm-up.  ``first_window=False`` keeps the first
        # window at its recorded state (windows 1..P-1 carry the variables).
        p_first = 0 if (bool(config.get("first_window", True)) or not continuity) else 1
        x0 = Y0[p_first:, index_t].detach()  # (P - p_first, n_slow) physical
        # The scale of every selected state: the largest magnitude it takes
        # at the windows' starts and ends in a rollout at the start point,
        # floored at one unit -- a state recorded at zero (a controller's
        # integral, a flow before the fan starts) must not get a vanishing
        # tolerance or a box it cannot leave.
        with torch.no_grad():
            theta0 = self._denorm(torch.as_tensor(np.asarray(self.est._x0_norm, dtype=np.float64), dtype=dtype, device=dev)[: self.n_theta])
            _out, end0 = self.est.simulator.rollout_functional_windows(self.composer, Y0, theta0, _tape, return_end=True)
        magnitude = torch.maximum(Y0[:, index_t].abs().amax(dim=0), end0[:, index_t].abs().amax(dim=0)).clamp(min=1.0)  # (n_slow,)
        sd_ref = str(config.get("sd_ref", "range"))
        if sd_ref == "range":
            # how much each state moves over the windows' starts and ends,
            # floored at a thousandth of its magnitude (a state that does not
            # move at the start point must not get a vanishing tolerance)
            both = torch.cat([Y0[:, index_t], end0[:, index_t]], dim=0)
            reference = torch.maximum(both.amax(dim=0) - both.amin(dim=0), 1e-3 * magnitude)
            sd_rel_default = 0.01
        elif sd_ref == "value":
            reference = magnitude
            sd_rel_default = 0.0025
        else:
            raise ValueError(f"multiple_shooting sd_ref must be 'range' or 'value'; got {sd_ref!r}")
        sd_rel, sd_abs = float(config.get("sd_rel", sd_rel_default)), float(config.get("sd_abs", 0.0))
        b_rel, b_abs = float(config.get("bound_rel", 0.25)), float(config.get("bound_abs", 0.0))
        sd = torch.clamp(torch.maximum(torch.full_like(reference, sd_abs), sd_rel * reference), min=1e-9)
        half = torch.maximum(torch.full_like(x0, b_abs), b_rel * magnitude.unsqueeze(0).expand_as(x0)).clamp(min=1e-6)
        low, high = x0.clone(), x0.clone()
        if continuity:
            # the box of a window's start also holds the end of the window
            # before it at the start point: the continuous trajectory must be
            # feasible, or a jump the box forces (a controller's integral
            # recorded at zero and ending far from it) is indistinguishable
            # from one the fit chose, and multiplier updates cannot close it
            first_tied = 1 - p_first  # the row of window 1 in x0
            previous_end = end0[:-1, index_t]  # the ends of windows 0 .. P-2
            low[first_tied:] = torch.minimum(low[first_tied:], previous_end)
            high[first_tied:] = torch.maximum(high[first_tied:], previous_end)
        lb, ub = low - half, high + half
        n_slow = int(index.size)
        self.init_first = int(p_first)
        self.n_init = int((P - p_first) * n_slow)
        self.init_index = index_t
        self.init_n_slow = n_slow
        self.init_blocks = np.asarray(blocks[index], dtype=np.int64)  # (n_slow,)
        self.init_sd = sd
        self.init_continuity = continuity
        # the augmented-Lagrangian shift of the defects, physical (P - 1, n_slow)
        self.init_shift = self._selected_from_instances(config.get("shift"), P - 1, dev, dtype, index_t) if continuity else None
        self.init_sd_reference = reference  # what sd_rel multiplies (see sd_ref)
        self.init_magnitude = magnitude
        self.init_x0 = x0
        self.init_lb, self.init_ub = lb, ub
        self.init_x0_norm = ((x0 - lb) / (ub - lb)).reshape(-1)
        rows = torch.arange(p_first, P, device=dev).repeat_interleave(n_slow)
        cols = index_t.repeat(P - p_first)
        self._init_put = (rows, cols)
        LOGGER.config(
            "multiple shooting: %d windows, %d states per window as variables (%d total, from window %d), "
            "continuity sd %.3g of the state's %s / %.3g absolute, box %.3g relative / %.3g absolute",
            P, n_slow, self.n_init, p_first, sd_rel, sd_ref, sd_abs, b_rel, b_abs,
        )
        if not continuity:
            LOGGER.config("initial states: estimated per period, not tied between periods")
        elif self.init_shift is not None:
            LOGGER.config(
                "multiple shooting: defects shifted by the last fit's multipliers (largest shift %.3g)",
                float(self.init_shift.abs().max()),
            )

    def _selected_from_instances(self, shift, rows: int, dev, dtype, index):
        """``{component id: (rows, state_size)}`` by the ids of the components
        the model was built from, as ``(rows, n_slow)`` over the selected
        states ``index`` (zero where the mapping has no value), or ``None``."""
        if not shift:
            return None
        model = self.est.simulator.model
        model = getattr(model, "simulation_model", model)
        flat = torch.zeros((rows, int(self.layout.width)), dtype=dtype, device=dev)
        start_of = {comp.id: start for comp, (start, _stop) in zip(self.layout.components, self.layout.slices)}
        shape_of = {comp.id: shape for comp, shape in zip(self.layout.components, self.layout.shapes)}
        found = 0
        for comp, owner, offset, width in model._stateful_leaves():
            if comp.id not in start_of:
                continue
            _n_c, ss = shape_of[comp.id]
            for i_c, cid in enumerate(model._instance_ids(owner)):
                value = shift.get(cid)
                if value is None:
                    continue
                value = torch.as_tensor(np.asarray(value, dtype=np.float64), dtype=dtype, device=dev).reshape(rows, -1)
                if value.shape[1] != width:
                    continue
                col = start_of[comp.id] + i_c * ss + offset
                flat[:, col : col + width] = torch.nan_to_num(value, nan=0.0)
                found += 1
        if found == 0:
            return None
        return flat[:, index]

    @property
    def n_theta_ext(self) -> int:
        """Length of the solver's vector: theta plus the initial-state variables."""
        return int(self.n_theta + self.n_init)

    def _split(self, x: torch.Tensor):
        """``(theta_norm, init_norm or None)`` from a solver vector: the plain
        theta vector (the recorded initial states then stand in for the
        variables) or the extended one."""
        if self.n_init == 0:
            return x, None
        if x.shape[-1] == self.n_theta:
            init = self.init_x0_norm.to(x.dtype)
            if x.dim() == 2:
                init = init.unsqueeze(0).expand(x.shape[0], -1)
            return x, init
        return x[..., : self.n_theta], x[..., self.n_theta :]

    def _denorm_init(self, init_norm: torch.Tensor) -> torch.Tensor:
        """Physical initial states ``(..., P-1, n_slow)`` from the normalised variables."""
        lb, ub = self.init_lb.reshape(-1), self.init_ub.reshape(-1)
        phys = lb + init_norm * (ub - lb)
        return phys.reshape(*init_norm.shape[:-1], self.init_lb.shape[0], self.init_n_slow)

    def _Y0_with(self, Y0: torch.Tensor, init_phys):
        """The windows' initial states with the variables written into rows 1..P-1."""
        if init_phys is None:
            return Y0
        rows, cols = self._init_put
        if init_phys.dim() == 3:  # a batch: (B, P-1, n_slow) -> (B, P, D_aug)
            B = init_phys.shape[0]
            base = Y0.unsqueeze(0).expand(B, -1, -1)
            b_idx = torch.arange(B, device=Y0.device).repeat_interleave(rows.numel())
            return base.index_put((b_idx, rows.repeat(B), cols.repeat(B)), init_phys.reshape(-1))
        return Y0.index_put((rows, cols), init_phys.reshape(-1))

    def _defects(self, end: torch.Tensor, Y0: torch.Tensor):
        """``(Y_end[p] - Y0[p + 1] + shift) / sd`` over the selected states:
        ``(..., P-1, n_slow)``; ``None`` when the periods are not tied."""
        if not getattr(self, "init_continuity", True):
            return None
        idx = self.init_index
        gap = end[..., :-1, :][..., idx] - Y0[..., 1:, :][..., idx]
        if getattr(self, "init_shift", None) is not None:
            gap = gap + self.init_shift
        return gap / self.init_sd

    def init_entry_names(self):
        """One label per initial-state variable, ``init[p]:<component>[i_c].x<k>``."""
        names = []
        if self.n_init == 0:
            return names
        layout = self.layout
        owner = {}
        for comp, (start, stop), (n_c, ss) in zip(layout.components, layout.slices, layout.shapes):
            for i_c in range(n_c):
                for k in range(ss):
                    owner[start + i_c * ss + k] = f"{comp.id}[{i_c}].x{k}"
        first = int(getattr(self, "init_first", 1))
        for p in range(first, first + self.init_lb.shape[0]):
            for j in self.init_index.tolist():
                names.append(f"init[{p}]:{owner.get(int(j), f'state{j}')}")
        return names

    def initial_state(self, x) -> dict:
        """The windows' initial states at a solver vector, per executing
        component: ``{component id: (P, n_c, state_size)}`` in physical
        units (the recorded states with the estimated variables written in),
        the format of the collocation's ``estimated_initial_state``; empty
        without multiple shooting."""
        x = torch.as_tensor(np.asarray(x, dtype=np.float64), dtype=self.init_lb.dtype if self.n_init else torch.float64, device=self.est._device) if not torch.is_tensor(x) else x
        windows = self._windows()
        if self.n_init == 0 or windows is None or x.shape[-1] != self.n_theta_ext:
            return {}
        _theta, init_phys = self._physical(x)
        Y = self._Y0_with(windows[0], init_phys)  # (P, D_aug); the component states lead
        out = {}
        layout = self.layout
        for comp, (start, stop), (n_c, ss) in zip(layout.components, layout.slices, layout.shapes):
            out[comp.id] = Y[:, start:stop].reshape(Y.shape[0], n_c, ss).detach().cpu().clone()
        return out

    def continuity_jumps(self, x) -> dict:
        """The jumps at the window boundaries at a solver vector, per
        executing component: ``{component id: (P-1, n_c, state_size)}``, the
        start of window ``p + 1`` minus the end of window ``p`` in physical
        units (``NaN`` for a state that is not a variable).  A jump is the
        part of a state the fit sets anew at a boundary instead of carrying
        it over; the defect the objective scores is its negative over the
        tolerance.  Empty without multiple shooting."""
        x = torch.as_tensor(np.asarray(x, dtype=np.float64), dtype=self.init_lb.dtype if self.n_init else torch.float64, device=self.est._device) if not torch.is_tensor(x) else x
        windows = self._windows()
        if self.n_init == 0 or windows is None or x.shape[-1] != self.n_theta_ext or not self.init_continuity:
            return {}
        theta_phys, init_phys = self._physical(x)
        Y0, tape = windows
        Y = self._Y0_with(Y0, init_phys)
        with torch.no_grad():
            _out, end = self.est.simulator.rollout_functional_windows(self.composer, Y, theta_phys, tape, return_end=True)
        jumps = torch.full_like(Y[1:], float("nan"))
        idx = self.init_index
        jumps[:, idx] = Y[1:, idx] - end[:-1, idx]
        return self._per_component(jumps)

    def continuity_shift_next(self, x) -> dict:
        """The shift for the next fit (the method of multipliers): this
        fit's shift plus the gap it left, ``Y_end[p] - Y0[p + 1] + shift``,
        per executing component ``{component id: (P-1, n_c, state_size)}``
        in physical units (``NaN`` for a state that is not a variable).
        Empty without multiple shooting."""
        x = torch.as_tensor(np.asarray(x, dtype=np.float64), dtype=self.init_lb.dtype if self.n_init else torch.float64, device=self.est._device) if not torch.is_tensor(x) else x
        windows = self._windows()
        if self.n_init == 0 or windows is None or x.shape[-1] != self.n_theta_ext or not self.init_continuity:
            return {}
        theta_phys, init_phys = self._physical(x)
        Y0, tape = windows
        Y = self._Y0_with(Y0, init_phys)
        with torch.no_grad():
            _out, end = self.est.simulator.rollout_functional_windows(self.composer, Y, theta_phys, tape, return_end=True)
        flat = torch.full_like(Y[1:], float("nan"))
        flat[:, self.init_index] = self._defects(end, Y) * self.init_sd
        return self._per_component(flat)

    def continuity_tolerance(self) -> dict:
        """The continuity tolerance of every state that is a variable, per
        executing component: ``{component id: (1, n_c, state_size)}``
        (``NaN`` elsewhere); a jump over its tolerance is the defect column
        the objective scores.  Empty without multiple shooting."""
        windows = self._windows()
        if self.n_init == 0 or windows is None or not self.init_continuity:
            return {}
        Y0 = windows[0]
        sd = torch.full_like(Y0[:1], float("nan"))
        sd[:, self.init_index] = self.init_sd.to(sd.dtype)
        return self._per_component(sd)

    def _per_component(self, flat: torch.Tensor) -> dict:
        """``(rows, D_aug)`` split into ``{component id: (rows, n_c, state_size)}``."""
        out = {}
        for comp, (start, stop), (n_c, ss) in zip(self.layout.components, self.layout.slices, self.layout.shapes):
            out[comp.id] = flat[:, start:stop].reshape(flat.shape[0], n_c, ss).detach().cpu().clone()
        return out

    def init_values(self, x) -> dict:
        """The optimised initial states per window and state label from a
        solver vector (``{window p: {label: value}}``), empty without them."""
        x = torch.as_tensor(np.asarray(x, dtype=np.float64), dtype=self.init_lb.dtype if self.n_init else torch.float64, device=self.est._device) if not torch.is_tensor(x) else x
        if self.n_init == 0 or x.shape[-1] != self.n_theta_ext:
            return {}
        phys = self._denorm_init(x[self.n_theta :]).detach().cpu().numpy()
        names = self.init_entry_names()
        out = {}
        k = 0
        first = int(getattr(self, "init_first", 1))
        for p in range(first, first + phys.shape[0]):
            out[p] = {}
            for j in range(phys.shape[1]):
                out[p][names[k].split(":", 1)[1]] = float(phys[p - first, j])
                k += 1
        return out

    # -- one-time capture of exogenous inputs + initial state ----------------
    def _record_exogenous_inputs(self):
        """One batched reference ``do_step`` rollout over all training periods
        (at the model's current parameters, i.e. x0) via
        :meth:`Simulator.record_exogenous_inputs`: per-step exogenous
        inputs, the augmented initial states ``Y0``, the lagged sensors'
        step-0 readings (``MEAS[0]``).  Also stacks the measured data per
        period."""
        est = self.est
        md_list = [md for md, _ in est._measurements]

        R = est.simulator.record_exogenous_inputs(
            self.composer,
            est._start_time,
            est._end_time,
            est._stepSize,
            layout=self.layout,
            meas_ids=[md.id for md in md_list],
        )
        self.CAP = R.exogenous_tape
        self.Y0 = R.Y0
        self.n_t = R.n_timesteps
        self.M0 = [values[0] for values in R.measurement_tape]
        self.ACT = []
        dev = est._device
        for p, n_t in enumerate(self.n_t):
            act = torch.zeros((n_t, len(md_list)), dtype=tps.float_dtype(), device=dev)
            for m, md in enumerate(md_list):
                vals = np.asarray(
                    est.actual_readings[md.id][p].to_numpy(), dtype=np.float64
                ).flatten()
                act[:, m] = torch.tensor(
                    vals[:n_t], dtype=tps.float_dtype(), device=dev
                )
                # ``scoring_mask``: a bool array per step on the measuring
                # device, False where the sample must not be scored (a duct
                # sensor while its fan is off).  Carried on the device, not
                # in the data: the loaders interpolate gaps away.  A masked
                # sample becomes NaN here and a zero residual below.
                mask = getattr(md, "scoring_mask", None)
                if mask is not None:
                    keep = torch.as_tensor(
                        _period_mask(mask, est.actual_readings[md.id][p].index, n_t), device=dev
                    )
                    act[~keep, m] = float("nan")
            self.ACT.append(act)

        LOGGER.config(
            "Functional single-shooting: %d period(s) x %s steps | captured inputs=%d | "
            "feedback lags=%d",
            len(self.n_t),
            self.n_t,
            len(self.composer._exogenous_keys),
            self.composer.n_feedback,
        )

    # -- the rollout ----------------------------------------------------------
    def _windows(self):
        """``(Y0 (P, D_aug), tape (n_t, P, n_exogenous))`` when the periods
        are several of equal length (one batched rollout advances them all),
        else ``None``; built once."""
        cached = self.__dict__.get("_windows_cache", ...)
        if cached is ...:
            cached = None
            if WINDOW_BATCHING and len(self.n_t) > 1 and len(set(int(n) for n in self.n_t)) == 1:
                cached = (torch.stack(list(self.Y0)), torch.stack(list(self.CAP), dim=1))
            self.__dict__["_windows_cache"] = cached
        return cached

    def _rollout(self, theta_phys: torch.Tensor, init_phys=None, *, transform_mode: bool = False):
        """Modelled measurements per period, ``[(n_t, n_meas), ...]``, and the
        multiple-shooting defects ``(P-1, n_slow)`` or ``None``.  Equal-length
        periods in transform mode roll out as one batch of windows
        (:meth:`Simulator.rollout_functional_windows`), else one after another
        (:meth:`Simulator.rollout_functional`)."""
        sim = self.est.simulator
        windows = self._windows() if (transform_mode or self.n_init) else None
        if windows is not None:
            Y0, tape = windows
            Y0 = self._Y0_with(Y0, init_phys)
            if self.n_init:
                out, end = sim.rollout_functional_windows(self.composer, Y0, theta_phys, tape, return_end=True)
                return [out[p] for p in range(out.shape[0])], self._defects(end, Y0)
            out = sim.rollout_functional_windows(self.composer, Y0, theta_phys, tape)
            return [out[p] for p in range(out.shape[0])], None
        return [
            sim.rollout_functional(
                self.composer,
                self.Y0[p],
                theta_phys,
                self.CAP[p],
                transform_mode=transform_mode,
            )
            for p in range(len(self.n_t))
        ], None

    def _rollout_meas(self, theta_phys: torch.Tensor, *, transform_mode: bool = False):
        """Modelled measurements per period (no defects); kept for callers of the plain rollout."""
        return self._rollout(theta_phys, None, transform_mode=transform_mode)[0]

    def _rollout_batched(self, theta_phys_batch: torch.Tensor, init_phys_batch=None):
        """Batched counterpart of :meth:`_rollout`: ``[(B, n_t, n_meas), ...]``
        and the defects ``(B, P-1, n_slow)`` or ``None``."""
        sim = self.est.simulator
        B = theta_phys_batch.shape[0]
        windows = self._windows()
        if windows is not None:
            Y0, tape = windows
            Y0 = self._Y0_with(Y0, init_phys_batch)
            if self.n_init:
                out, end = sim.rollout_functional_batched_windows(self.composer, Y0, theta_phys_batch, tape, return_end=True)
                Y0_rows = Y0 if Y0.dim() == 3 else Y0.unsqueeze(0).expand(B, -1, -1)
                return [out[:, p] for p in range(out.shape[1])], self._defects(end, Y0_rows)
            out = sim.rollout_functional_batched_windows(self.composer, Y0, theta_phys_batch, tape)
            return [out[:, p] for p in range(out.shape[1])], None
        return [
            sim.rollout_functional_batched(
                self.composer,
                self.Y0[p].unsqueeze(0).expand(B, -1),
                theta_phys_batch,
                self.CAP[p],
            )
            for p in range(len(self.n_t))
        ], None

    def _rollout_meas_batched(self, theta_phys_batch: torch.Tensor):
        """Batched modelled measurements per period (no defects)."""
        return self._rollout_batched(theta_phys_batch, None)[0]

    def _physical(self, x: torch.Tensor):
        """``(theta_phys, init_phys or None)`` from a solver vector."""
        theta_norm, init_norm = self._split(x)
        return self._denorm(theta_norm), (None if init_norm is None else self._denorm_init(init_norm))

    def raw_residuals(
        self, theta: torch.Tensor, *, transform_mode: bool = False
    ) -> torch.Tensor:
        """Pure scored residual matrix ``actual - model`` (the measured
        columns; the multiple-shooting defects are not part of it).

        This method has no logging or estimator-state mutation and is therefore
        safe under ``torch.func`` transforms and CUDA Graph capture.
        """
        theta_phys, init_phys = self._physical(theta)
        Ms, _defects = self._rollout(theta_phys, init_phys, transform_mode=transform_mode)
        return self._raw_residuals_from_meas(Ms)

    def _raw_residuals_from_meas(self, Ms) -> torch.Tensor:
        """Pure post-processing of rolled-out measurements (one sample; ``vmap``-able)."""
        nw = self.est._n_warmup
        raw_terms = []
        for p, M in enumerate(Ms):
            if any(self.meas_lag):
                cols = []
                for m, lag in enumerate(self.meas_lag):
                    if lag:
                        cols.append(torch.cat([self.M0[p][m : m + 1], M[:-1, m]]))
                    else:
                        cols.append(M[:, m])
                M = torch.stack(cols, dim=1)
            act = self.ACT[p][nw:]
            diff = act - M[nw:]
            # A NaN reading is an unscored sample (a sensor's ``allow_missing``
            # gap): zero residual and zero gradient there.
            raw_terms.append(torch.where(torch.isnan(act), torch.zeros_like(diff), diff))
        return torch.cat(raw_terms, dim=0)

    def _weighted(self, raw: torch.Tensor) -> torch.Tensor:
        """The measured residual matrix on the loss scale."""
        return raw / self._sd / torch.sqrt(self.loss_scale * self._denom)

    def _weighted_defects(self, defects):
        """The defects on the loss scale (already divided by their tolerance), or ``None``."""
        if defects is None:
            return None
        return defects / torch.sqrt(self.loss_scale * self._denom)

    def residual_vector(
        self, theta: torch.Tensor, *, transform_mode: bool = False
    ) -> torch.Tensor:
        """Weighted residual whose squared norm equals :meth:`loss`: the
        measured residuals, then the continuity defects."""
        theta_phys, init_phys = self._physical(theta)
        Ms, defects = self._rollout(theta_phys, init_phys, transform_mode=transform_mode)
        parts = [self._weighted(self._raw_residuals_from_meas(Ms)).reshape(-1)]
        wd = self._weighted_defects(defects)
        if wd is not None:
            parts.append(wd.reshape(-1))
        return torch.cat(parts)

    def loss(
        self, theta: torch.Tensor, *, transform_mode: bool = False
    ) -> torch.Tensor:
        residual = self.residual_vector(theta, transform_mode=transform_mode)
        return torch.sum(residual.square())

    def _loss_from_rollout(self, Ms, defects=None) -> torch.Tensor:
        value = torch.sum(self._weighted(self._raw_residuals_from_meas(Ms)).square())
        wd = self._weighted_defects(defects)
        if wd is not None:
            value = value + torch.sum(wd.square())
        return value

    def _loss_from_meas(self, Ms) -> torch.Tensor:
        return self._loss_from_rollout(Ms, None)

    # -- per-column (per residual signal) losses: sum over columns == loss ------
    def _column_loss_from_rollout(self, Ms, defects=None) -> torch.Tensor:
        """``(n_meas [+ n_slow],)``: the measured columns, then one column per
        selected state summing its defects over the window boundaries."""
        cols = self._weighted(self._raw_residuals_from_meas(Ms)).square().sum(dim=0)
        wd = self._weighted_defects(defects)
        if wd is not None:
            cols = torch.cat([cols, wd.square().sum(dim=0)])
        return cols

    def _column_loss_from_meas(self, Ms) -> torch.Tensor:
        return self._column_loss_from_rollout(Ms, None)

    def column_loss(self, theta: torch.Tensor, *, transform_mode: bool = False) -> torch.Tensor:
        """Squared-residual sum per residual column; ``column_loss(theta).sum() == loss(theta)``."""
        theta_phys, init_phys = self._physical(theta)
        Ms, defects = self._rollout(theta_phys, init_phys, transform_mode=transform_mode)
        return self._column_loss_from_rollout(Ms, defects)

    def batched_column_loss(self, theta_batch: torch.Tensor) -> torch.Tensor:
        """``(B, n_meas [+ n_slow])`` per-column losses for a batch of solver vectors."""
        if theta_batch.device.type == "cpu":
            return torch.stack([self.column_loss(th) for th in theta_batch])
        if theta_batch.shape[0] == 1:
            return self.column_loss(theta_batch[0], transform_mode=True).unsqueeze(0)
        theta_phys, init_phys = self._physical(theta_batch)
        Ms, defects = self._rollout_batched(theta_phys, init_phys)
        if defects is None:
            return torch.func.vmap(self._column_loss_from_meas)(Ms)
        return torch.func.vmap(self._column_loss_from_rollout)(Ms, defects)

    def batched_column_loss_and_grad(
        self, theta_batch: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-column losses ``(B, n_cols)`` and the gradient of their sum ``(B, n_theta_ext)``.

        For block-separable problems (:meth:`parameter_structure`) block ``k``
        of the gradient is the gradient of block ``k``'s own column sum, so one
        backward pass serves every block.
        """
        z = theta_batch.detach().clone().requires_grad_(True)
        cols = self.batched_column_loss(z)
        (grad,) = torch.autograd.grad(cols.sum(), z)
        return cols.detach(), grad.detach()

    def batched_column_gradients(
        self, theta_batch: torch.Tensor, selector: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-column losses ``(B, n_cols)`` and, per row, the gradient of
        ``selector[b] . cols[b]`` ``(B, n_theta_ext)``.

        With ``theta_batch`` the same parameters on every row and
        ``selector`` one-hot rows, this is one chunk of the Jacobian of the
        column losses, ``d c_j / d theta`` for ``B`` columns, from a single
        batched backward pass: the residual-side curvature a Gauss-Newton
        model needs, computed exactly rather than from gradient differences.
        """
        z = theta_batch.detach().clone().requires_grad_(True)
        cols = self.batched_column_loss(z)
        (grad,) = torch.autograd.grad((cols * selector).sum(), z)
        return cols.detach(), grad.detach()

    def parameter_structure(self):
        """``(theta_block, column_block, n_blocks)`` from the composer's wiring
        (:meth:`FunctionalModel.index_coupling`), extended by the
        multiple-shooting variables and defect columns (each in the block of
        its state's component); cached."""
        cached = self.__dict__.get("_parameter_structure")
        if cached is None:
            theta_block, column_block, n_blocks = self.composer.index_coupling()
            if self.n_init:
                P1 = self.init_lb.shape[0]
                theta_block = np.concatenate([np.asarray(theta_block), np.tile(self.init_blocks, P1)])
                if self.init_continuity:
                    column_block = np.concatenate([np.asarray(column_block), self.init_blocks])
            cached = (theta_block, column_block, n_blocks)
            self.__dict__["_parameter_structure"] = cached
        return cached

    def batched_loss(self, theta_batch: torch.Tensor) -> torch.Tensor:
        if theta_batch.device.type == "cpu":
            return torch.stack([self.loss(th) for th in theta_batch])
        if theta_batch.shape[0] == 1:
            return self.loss(theta_batch[0], transform_mode=True).unsqueeze(0)
        # Batched rollout (compiled batched step where available), then the
        # pure residual post-processing vmap-ed over the batch dimension.
        theta_phys, init_phys = self._physical(theta_batch)
        Ms, defects = self._rollout_batched(theta_phys, init_phys)
        if defects is None:
            return torch.func.vmap(self._loss_from_meas)(Ms)
        return torch.func.vmap(self._loss_from_rollout)(Ms, defects)

    def batched_value_and_grad(
        self, theta_batch: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Reverse mode is plain ``torch.autograd.grad`` on every device.
        # functorch's ``grad_and_value`` transform records a CUDA graph that
        # is valid for exactly one launch on some builds (torch 2.11+cu128 on
        # an A100: the second replay raises an illegal memory access even
        # back-to-back with nothing in between, while the same graph replays
        # fine on torch 2.13+cu130/Windows).  ``vmap`` of the *forward*
        # rollout alone is capture-safe (validated on the same A100 with an
        # eager cross-check), so batches keep one wide rollout; a single
        # start skips vmap, which captures faster.  The transform-mode
        # rollout (no parameter cache) keeps the captured graph free of
        # Python-side state.
        z = theta_batch.detach().clone().requires_grad_(True)
        value = self.batched_loss(z)
        (grad,) = torch.autograd.grad(value.sum(), z)
        return value.detach(), grad.detach()

    def batched_residual_and_jacobian(
        self, theta_batch: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        fn = torch.func.jacfwd(
            lambda th: self.residual_vector(th, transform_mode=True),
            argnums=0,
            has_aux=False,
        )
        residual = torch.func.vmap(
            lambda th: self.residual_vector(th, transform_mode=True)
        )(theta_batch)
        jacobian = torch.func.vmap(fn)(theta_batch)
        return residual, jacobian

    # -- Estimator._obj drop-in (scalar mode) ---------------------------------
    def loglike(self, theta: torch.Tensor, output: str = "scalar") -> torch.Tensor:
        """Same contract (and side-effect diagnostics) as ``Estimator._obj``
        in scalar mode; differentiable w.r.t. ``theta`` (normalized)."""
        if output != "scalar":
            raise ValueError("functional single-shooting objective is scalar-only")
        est = self.est
        raw = self.raw_residuals(theta)
        # Everything downstream of the raw residuals (sd weighting, padded-
        # horizon normalization, rescale-to-100, diagnostics) is THE shared
        # objective -- the same method the object-graph _obj ends in, so the
        # two paths cannot diverge there by construction.  ``raw`` holds only
        # the scored rows; the object-graph path passes the padded horizon
        # with zero rows -- identical sums either way.
        loss = est._loglike_from_residuals(raw, output)
        if not torch.isfinite(loss.detach()).all():
            # Mirror the object-graph recovery for diverging iterates.  The
            # do_step path VALIDATES port values and raises on NaN, which
            # ``_obj_ad`` converts to a large penalty + zero gradient; the
            # composed rollout has no such validation, so a physically
            # unstable theta would otherwise hand the solver a silent nan
            # (SLSQP then stalls and can terminate AT the nan iterate).
            LOGGER.warning(
                "functional objective non-finite at this theta -- returning penalty"
            )
            try:
                for line in est._format_theta_dump(theta.detach().cpu().numpy()):
                    LOGGER.warning("%s", line)
            except Exception:  # noqa: BLE001
                pass
            est._last_rmse = float("nan")
            est._last_rmse_per_sensor = {}
            # ``0 * theta.sum()`` keeps the result attached to theta so the
            # autograd path yields a well-defined ZERO gradient (same
            # backtracking behaviour as the object-graph recovery).
            penalty = 0.0 * theta.sum() + 1e10
            est._loglike = penalty
            return penalty
        return loss
