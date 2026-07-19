"""Fast single-shooting objective via the composed one-step map ``F_aug``.

Single-shooting's per-iteration cost is one full object-graph simulation plus
autograd back through it.  Profiling shows most of that cost is Python
dispatch -- the per-step Gauss-Seidel traversal over components, ``tps``
wrapper bookkeeping, history logging -- and ``model.initialize`` re-reading
every CSV input on EVERY objective evaluation.  None of it is tensor math.

This module reuses the collocation composer (:class:`OneStepComposer`): the
model's stateful cone is composed into a pure one-step map
``F_aug(y, theta, cap)``, the truly-exogenous inputs are captured ONCE from a
reference ``do_step`` rollout, and the objective becomes a plain sequential
torch rollout

    y_{t+1}, meas_t = F_aug(y_t, theta, CAP[t])

vmapped across the training periods.  Autograd back through this rollout is
the same reverse pass single-shooting already paid for -- minus the
object-graph overhead.

Exactness: **by construction**.  Every composable component's ``do_step`` is a
thin port-I/O wrapper that DELEGATES its math to the same ``forward`` the
composer threads (single source of truth -- the two cannot drift apart), and
cut feedback edges are carried as one-step lag state inside ``y`` (exactly
``do_step``'s delayed Gauss-Seidel semantics).  Only truly exogenous inputs
(weather, schedules, data-driven occupancy) are frozen -- and those are
theta-independent by definition.  The same single-source-of-truth rule holds
outside the components: theta denormalization is
:func:`twin4build.utils.types.denormalize_unit` (the function
``tps.Parameter.denormalize`` itself routes through) and everything downstream
of the raw residuals (sd weighting, MSE normalization, rescale-to-100,
diagnostics) is ``Estimator._loglike_from_residuals`` -- the method the
object-graph ``_obj`` ends in as well.  Construction performs the structural
checks and the estimator silently falls back to the exact path for
un-composable models (components without ``forward``, ``n_c > 1``,
shared/expanded theta, a measurement the composed map cannot produce).
``tests/estimator/test_fast_shooting.py`` regression-checks the end-to-end
value + gradient parity (guards the delegation contract and the composer's
wiring/capture logic); ``options={"fast_validate": True}`` re-enables the
runtime cross-check as a debugging aid.

Enable with ``estimator.estimate(..., options={"fast": True})``.
"""

from __future__ import annotations

import numpy as np
import torch

from twin4build.estimator._composer import OneStepComposer, capture_reference_rollout
from twin4build.estimator._transcription import _collect_stateful, _StateLayout
from twin4build.utils.print_progress import LOGGER
from twin4build.utils.types import denormalize_unit, theta_bound_tensors


class FastSingleShooting:
    """Composer-based drop-in for :meth:`Estimator._obj` (scalar mode).

    Built AFTER the estimator's ``estimate`` setup (parameters, measurements,
    normalized bounds, ``actual_readings`` all populated, model initialized at
    ``x0``).  Construction raises on any model feature the composer cannot
    express; callers treat that as "use the object-graph objective".
    """

    def __init__(self, estimator):
        self.est = estimator
        model = estimator.simulator.model

        if getattr(estimator, "_regularization_lambda", 0) > 0:
            raise RuntimeError("regularization penalty not supported")
        n_theta = len(estimator._x0_norm)
        if n_theta != len(estimator._flat_components):
            raise RuntimeError("shared/expanded theta (n_c>1 parameters)")

        stateful = _collect_stateful(model)
        if not stateful:
            raise RuntimeError("no stateful components")
        layout = _StateLayout(stateful)
        if any(getattr(c, "n_c", 1) != 1 for c in layout.components):
            raise RuntimeError("n_c > 1 states")

        step_sizes = [int(s) for s in estimator._stepSize]
        if len(set(step_sizes)) != 1:
            raise RuntimeError("mixed step sizes across periods")

        theta_spec = list(zip(estimator._flat_components, estimator._parameter_names))
        composer = OneStepComposer(
            model, layout.components, theta_spec, step_sizes[0],
            measurements=[md for md, _ in estimator._measurements],
        )
        if composer.D != layout.width:
            raise RuntimeError("composer state width mismatch")
        if not composer.meas_sources:
            raise RuntimeError("no measurement sources")
        if any(s[0] != "fresh" for s in composer.meas_sources):
            # A frozen measurement would make its residual theta-independent:
            # the fast objective would silently ignore that sensor.
            raise RuntimeError("a measurement is not producible by the composed map")

        self.layout = layout
        self.composer = composer
        self.n_theta = n_theta

        # Plain (functorch-safe) denormalization from the physical bounds.
        self._lb_t, self._ub_t, self._log_mask = theta_bound_tensors(
            estimator._flat_parameters
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
                "Fast single-shooting: one-step sensor lag on %s",
                [md.id for (md, _), l in zip(estimator._measurements, self.meas_lag) if l],
            )
        self._capture()

    def _denorm(self, th_norm: torch.Tensor) -> torch.Tensor:
        # Single source of truth for the normalized->physical map
        # (tps.Parameter.denormalize routes through the same function).
        return denormalize_unit(th_norm, self._lb_t, self._ub_t, self._log_mask)

    # -- one-time capture of exogenous inputs + initial state ----------------
    def _capture(self):
        """One reference ``do_step`` rollout per training period (at the
        model's current parameters, i.e. x0) via the shared
        :func:`capture_reference_rollout`: per-step captured (exogenous)
        inputs, the initial component states after ``initialize``, the
        feedback-lag warm values consumed at step 0 (``FB[0]``) and the lagged
        sensors' step-0 readings (``MEAS[0]``).  Also stacks the measured
        data per period."""
        est = self.est
        n_fb = self.composer.n_feedback
        md_list = [md for md, _ in est._measurements]

        self.CAP, self.Y0, self.ACT, self.n_t, self.M0 = [], [], [], [], []
        for p, (s, e, sz) in enumerate(
            zip(est._start_time, est._end_time, est._stepSize)
        ):
            R = capture_reference_rollout(
                est.simulator, self.composer, s, e, sz,
                layout=self.layout, meas_ids=[md.id for md in md_list],
            )
            self.CAP.append(R.CAP)
            self.Y0.append(torch.cat([R.state0, R.FB[0]]) if n_fb else R.state0)
            self.n_t.append(R.n_t)
            self.M0.append(R.MEAS[0])
            act = torch.zeros((R.n_t, len(md_list)), dtype=torch.float64)
            for m, md in enumerate(md_list):
                vals = np.asarray(
                    est.actual_readings[md.id][p].to_numpy(), dtype=np.float64
                ).flatten()
                act[:, m] = torch.tensor(vals[: R.n_t], dtype=torch.float64)
            self.ACT.append(act)

        LOGGER.config(
            "Fast single-shooting: %d period(s) x %s steps | captured inputs=%d | "
            "feedback lags=%d",
            len(self.n_t), self.n_t, len(self.composer._captured_keys), n_fb,
        )

    # -- the rollout ----------------------------------------------------------
    def _rollout_meas(self, theta_phys: torch.Tensor):
        """Modelled measurements per period: list of ``(n_t, n_meas)``.

        Periods are rolled out in a plain Python loop, NOT under ``vmap``:
        with a handful of periods the vmap dispatch overhead exceeds the
        batching gain, and staying in ordinary eager mode lets the state-space
        components use the fused ``torch.matrix_exp`` (which has no vmap rule)
        instead of the unrolled scaling-and-squaring fallback.
        """
        F_aug = self.composer.F_aug
        out = []
        for p in range(len(self.n_t)):
            y = self.Y0[p]
            meas_steps = []
            for t in range(self.n_t[p]):
                y, meas = F_aug(y, theta_phys, self.CAP[p][t])
                meas_steps.append(meas)
            out.append(torch.stack(meas_steps))
        return out

    # -- Estimator._obj drop-in (scalar mode) ---------------------------------
    def loglike(self, theta: torch.Tensor, output: str = "scalar") -> torch.Tensor:
        """Same contract (and side-effect diagnostics) as ``Estimator._obj``
        in scalar mode; differentiable w.r.t. ``theta`` (normalized)."""
        if output != "scalar":
            raise ValueError("fast single-shooting objective is scalar-only")
        est = self.est
        theta_phys = self._denorm(theta)
        Ms = self._rollout_meas(theta_phys)
        nw = est._n_warmup
        raw_terms = []
        for p, M in enumerate(Ms):
            if any(self.meas_lag):
                # Lagged sensors report the previous step's producer output
                # (see __init__); step 0 reads the initialized value.
                cols = []
                for m, lag in enumerate(self.meas_lag):
                    if lag:
                        cols.append(torch.cat([self.M0[p][m:m + 1], M[:-1, m]]))
                    else:
                        cols.append(M[:, m])
                M = torch.stack(cols, dim=1)
            raw_terms.append(self.ACT[p][nw:] - M[nw:])
        raw = torch.cat(raw_terms, dim=0)  # (N, n_meas)
        # Everything downstream of the raw residuals (sd weighting, padded-
        # horizon normalization, rescale-to-100, diagnostics) is THE shared
        # objective -- the same method the object-graph _obj ends in, so the
        # two paths cannot diverge there by construction.  ``raw`` holds only
        # the scored rows; the object-graph path passes the padded horizon
        # with zero rows -- identical sums either way.
        return est._loglike_from_residuals(raw, output)
