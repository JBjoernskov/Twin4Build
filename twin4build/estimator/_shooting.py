"""Fast single-shooting objective via the composed one-step map ``F_aug``.

Single-shooting's per-iteration cost is one full object-graph simulation plus
autograd back through it.  Profiling shows most of that cost is Python
dispatch -- the per-step Gauss-Seidel traversal over components, ``tps``
wrapper bookkeeping, history logging -- and ``model.initialize`` re-reading
every CSV input on EVERY objective evaluation.  None of it is tensor math.

This module builds on the Simulator's composed-map API
(:meth:`Simulator.compose` / :meth:`Simulator.capture_rollout` /
:meth:`Simulator.rollout_composed`, implemented in
:mod:`twin4build.simulator._composed`): the model's stateful cone is composed
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
``forward``, ``n_c > 1`` states or multi-branch parameters, a measurement the
composed map cannot produce).
``tests/estimator/test_fast_shooting.py`` regression-checks the end-to-end
value + gradient parity (guards the delegation contract and the composer's
wiring/capture logic).

Enable with ``Simulator(model, execution_mode="composed")``.
"""

from __future__ import annotations

import numpy as np
import torch

import twin4build.utils.types as tps
from twin4build.utils.logger import LOGGER
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

        if getattr(estimator, "_regularization_lambda", 0) > 0:
            raise RuntimeError("regularization penalty not supported")
        n_theta = len(estimator._x0_norm)

        # Indexed theta spec: shared parameters route several (comp, attr)
        # entries to one theta slot.  Raises on multi-branch (n_c > 1)
        # parameters, which the composed map cannot express.
        theta_spec, unique_parameters = estimator._composer_theta_spec()

        # Structural checks (stateful components exist, n_c == 1, uniform
        # step size, state-width match) live in Simulator.compose.
        layout, composer = estimator.simulator.compose(
            theta_spec=theta_spec,
            measurements=[md for md, _ in estimator._measurements],
            step_size=estimator._stepSize,
        )
        if not composer.meas_sources:
            raise RuntimeError("no measurement sources")
        if any(s[0] != "fresh" for s in composer.meas_sources):
            # A frozen measurement would make its residual theta-independent:
            # the fast objective would silently ignore that sensor.
            raise RuntimeError("a measurement is not producible by the composed map")

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
                "Fast single-shooting: one-step sensor lag on %s",
                [md.id for (md, _), l in zip(estimator._measurements, self.meas_lag) if l],
            )
        self._capture()
        self._sd = torch.tensor(
            [float(sd) for _md, sd in estimator._measurements],
            dtype=tps.float_dtype(),
            device=estimator._device,
        )
        self._denom = float(estimator._n_timesteps * len(estimator._measurements))
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

    # -- one-time capture of exogenous inputs + initial state ----------------
    def _capture(self):
        """One batched reference ``do_step`` rollout over all training periods
        (at the model's current parameters, i.e. x0) via
        :meth:`Simulator.capture_rollout`: per-step captured (exogenous)
        inputs, the augmented initial states ``Y0``, the lagged sensors'
        step-0 readings (``MEAS[0]``).  Also stacks the measured data per
        period."""
        est = self.est
        md_list = [md for md, _ in est._measurements]

        R = est.simulator.capture_rollout(
            self.composer, est._start_time, est._end_time, est._stepSize,
            layout=self.layout, meas_ids=[md.id for md in md_list],
        )
        self.CAP, self.Y0, self.n_t = R.CAP, R.Y0, R.n_t
        self.M0 = [M[0] for M in R.MEAS]
        self.ACT = []
        dev = est._device
        for p, n_t in enumerate(self.n_t):
            act = torch.zeros((n_t, len(md_list)), dtype=tps.float_dtype(), device=dev)
            for m, md in enumerate(md_list):
                vals = np.asarray(
                    est.actual_readings[md.id][p].to_numpy(), dtype=np.float64
                ).flatten()
                act[:, m] = torch.tensor(vals[:n_t], dtype=tps.float_dtype(), device=dev)
            self.ACT.append(act)

        LOGGER.config(
            "Fast single-shooting: %d period(s) x %s steps | captured inputs=%d | "
            "feedback lags=%d",
            len(self.n_t), self.n_t, len(self.composer._captured_keys),
            self.composer.n_feedback,
        )

    # -- the rollout ----------------------------------------------------------
    def _rollout_meas(
        self, theta_phys: torch.Tensor, *, transform_mode: bool = False
    ):
        """Modelled measurements per period: list of ``(n_t, n_meas)``
        (the shared sequential rollout, :meth:`Simulator.rollout_composed`)."""
        sim = self.est.simulator
        return [
            sim.rollout_composed(
                self.composer,
                self.Y0[p],
                theta_phys,
                self.CAP[p],
                transform_mode=transform_mode,
            )
            for p in range(len(self.n_t))
        ]

    def raw_residuals(
        self, theta: torch.Tensor, *, transform_mode: bool = False
    ) -> torch.Tensor:
        """Pure scored residual matrix ``actual - model``.

        This method has no logging or estimator-state mutation and is therefore
        safe under ``torch.func`` transforms and CUDA Graph capture.
        """
        theta_phys = self._denorm(theta)
        Ms = self._rollout_meas(theta_phys, transform_mode=transform_mode)
        nw = self.est._n_warmup
        raw_terms = []
        for p, M in enumerate(Ms):
            if any(self.meas_lag):
                cols = []
                for m, lag in enumerate(self.meas_lag):
                    if lag:
                        cols.append(torch.cat([self.M0[p][m:m + 1], M[:-1, m]]))
                    else:
                        cols.append(M[:, m])
                M = torch.stack(cols, dim=1)
            raw_terms.append(self.ACT[p][nw:] - M[nw:])
        return torch.cat(raw_terms, dim=0)

    def residual_vector(
        self, theta: torch.Tensor, *, transform_mode: bool = False
    ) -> torch.Tensor:
        """Weighted residual whose squared norm equals :meth:`loss`."""
        raw = self.raw_residuals(theta, transform_mode=transform_mode)
        return (
            raw / self._sd / torch.sqrt(self.loss_scale * self._denom)
        ).reshape(-1)

    def loss(
        self, theta: torch.Tensor, *, transform_mode: bool = False
    ) -> torch.Tensor:
        residual = self.residual_vector(theta, transform_mode=transform_mode)
        return torch.sum(residual.square())

    def batched_loss(self, theta_batch: torch.Tensor) -> torch.Tensor:
        if theta_batch.device.type == "cpu":
            return torch.stack([self.loss(th) for th in theta_batch])
        return torch.func.vmap(
            lambda th: self.loss(th, transform_mode=True)
        )(theta_batch)

    def batched_value_and_grad(
        self, theta_batch: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if theta_batch.device.type == "cpu":
            z = theta_batch.detach().clone().requires_grad_(True)
            value = torch.stack([self.loss(th) for th in z])
            (grad,) = torch.autograd.grad(value.sum(), z)
            return value.detach(), grad.detach()
        fn = torch.func.grad_and_value(
            lambda th: self.loss(th, transform_mode=True)
        )
        grad, value = torch.func.vmap(fn)(theta_batch)
        return value, grad

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
            raise ValueError("fast single-shooting objective is scalar-only")
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
                "fast objective non-finite at this theta -- returning penalty"
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
