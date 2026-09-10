r"""Batched block trust-region Pareto sweep.

The SLSQP and IPOPT routes solve the AUGMECON epsilon-subproblems one at a
time on the host, warm-started from a projected-Adam prepass.  This route
solves *all* of them at once with the block trust-region step of
:mod:`twin4build.estimator._batched_solvers` (issue #142): one batch row per
epsilon value, exact AD gradients from one backward pass over the batched
rollout, a per-block damped-BFGS model and a scaled trust box clipped to the
variable bounds.  There is no host solver and no separate prepass -- the
second-order batched solve *is* the sweep.

Row ``i`` minimises

.. math::

    L_i(\theta) = a_i f_1(\theta) + b_i f_2(\theta) + P(\theta)
                  + \mu_i \max(0,\, f_{2n}(\theta) - \varepsilon_i)^2

where :math:`P` is the optimizer's constraint penalty, :math:`f_{2n}` the
normalized second objective, and :math:`(a_i, b_i, \mu_i, \varepsilon_i)` the
per-row coefficients.  That one form covers both stages of the sweep: the
payoff-table anchors (:math:`a=1, b=\delta, \mu=0` and its mirror, matching
:func:`_pareto._anchor_solve`'s augmented scalarization) and the epsilon
subproblems (:math:`a=1`, :math:`b=\delta/\mathrm{range}_2`, quadratic
penalty on the epsilon constraint).

**On block structure.** The blocks come from the model wiring, as in
estimation.  When the decision variables are a single schedule broadcast to
every zone -- the canonical benchmark's shared valve position -- they form one
block and this reduces to a box-constrained damped-BFGS trust region; the
per-block machinery only earns its name when the controls are per-zone.  What
the batching buys either way is the sweep: N epsilon-subproblems advance per
gradient evaluation instead of one.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

import twin4build.utils.types as tps
from twin4build.estimator import _batched_solvers as bs
from twin4build.utils.logger import LOGGER

BATCHED_TR_METHOD = ("custom", "batched-tr", "ad")

#: Options accepted per call; everything else is rejected so a typo cannot
#: silently pick a default.  ``maxiter`` is shared with the host routes.
_SWEEP_OPTION_KEYS = frozenset(
    {
        "maxiter",
        "gtol",
        "ftol",
        "patience",
        "tr_blocks",
        "tr_radius",
        "tr_max_radius",
        "tr_min_radius",
        "tr_accept",
        "tr_expand",
        "tr_shrink",
        "tr_scale_floor",
        "tr_retries",
        "tr_validate",
        "max_nfev",
    }
)


class _BatchedScalarizedObjective:
    """Presents the N scalarized subproblems to the batched solvers.

    Exposes the bundle interface the batched solvers call
    (:class:`bs.BatchedObjectiveEvaluator`): per-row losses, per-row losses
    with the gradient of their sum (the rows are independent, so one backward
    pass yields every row's gradient), and the parameter structure the block
    step decomposes over.
    """

    def __init__(
        self,
        opt,
        coefficients: torch.Tensor,
        ideal2: float,
        range2: float,
        n_theta: int,
    ):
        self._opt = opt
        self._n_theta = int(n_theta)
        self._fast = opt._functional_objective
        if self._fast is None:
            raise ValueError(
                "The batched trust-region Pareto route needs the composed "
                'objective; construct Simulator(model, execution_mode="functional").'
            )
        self._coefficients = coefficients  # (N, 4): a, b, mu, eps
        self._ideal2 = float(ideal2)
        self._range2 = float(range2) if abs(float(range2)) > 1e-12 else 1.0
        # What the evaluator reads off an objective: device, backend, dtype.
        # The backend is reported as eager on purpose.  These bundles build
        # their forward graph under ``torch.func.vmap`` and differentiate it
        # afterwards; recording that in a CUDA graph fails during
        # ``capture:record``, and a failed capture sets a process-global flag
        # that makes every later capture and eager fallback unsafe -- one
        # arm's capture failure would take down every later arm in the
        # process.  The rollout still runs batched on the device; only the
        # graph replay is given up.
        self.est = SimpleNamespace(
            _device=opt._device,
            simulator=SimpleNamespace(execution_backend="eager"),
        )
        self._sd = torch.ones(1, dtype=tps.float_dtype(), device=opt._device)

    # -- the per-row loss ---------------------------------------------------
    def _row_loss(self, theta_row: torch.Tensor, coefficients: torch.Tensor):
        parts = self._fast.parts(theta_row)
        penalty = self._fast.penalty(parts)
        f2n = (parts.objs[1] - self._ideal2) / self._range2
        a, b, mu, eps = (
            coefficients[0],
            coefficients[1],
            coefficients[2],
            coefficients[3],
        )
        return (
            a * parts.objs[0]
            + b * parts.objs[1]
            + penalty
            + mu * torch.relu(f2n - eps) ** 2
        )

    def _losses(self, theta_batch: torch.Tensor) -> torch.Tensor:
        try:
            return torch.func.vmap(self._row_loss)(theta_batch, self._coefficients)
        except Exception:  # noqa: BLE001 - vmap may reject an op in the model
            return torch.stack(
                [
                    self._row_loss(theta_batch[i], self._coefficients[i])
                    for i in range(theta_batch.shape[0])
                ]
            )

    # -- bundles the batched solvers call -----------------------------------
    def batched_loss(self, theta_batch: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self._losses(theta_batch)

    def batched_value_and_grad(self, theta_batch: torch.Tensor):
        z = theta_batch.detach().clone().requires_grad_(True)
        losses = self._losses(z)
        (gradient,) = torch.autograd.grad(losses.sum(), z)
        return losses.detach(), gradient.detach()

    def batched_column_loss(self, theta_batch: torch.Tensor) -> torch.Tensor:
        # One column: the scalarized row loss is not a sum of independently
        # attributable residuals, so there is nothing to split it into.
        return self.batched_loss(theta_batch).unsqueeze(1)

    def batched_column_loss_and_grad(self, theta_batch: torch.Tensor):
        values, gradient = self.batched_value_and_grad(theta_batch)
        return values.unsqueeze(1), gradient

    def loss(self, theta: torch.Tensor, transform_mode: bool = False):
        return self._row_loss(theta, self._coefficients[0])

    def parameter_structure(self):
        """Blocks of the decision vector: from the model wiring when the
        composed objective reports a structure covering the decision vector,
        one block otherwise (the shared-schedule case)."""
        structure = getattr(self._fast, "parameter_structure", None)
        if structure is not None:
            try:
                theta_block, _column_block, n_blocks = structure()
                theta_block = np.asarray(theta_block)
                if theta_block.size == self._n_theta:
                    return theta_block, np.zeros(1, dtype=np.int64), int(n_blocks)
            except Exception:  # noqa: BLE001 - fall back to one block
                pass
        return (
            np.zeros(self._n_theta, dtype=np.int64),
            np.zeros(1, dtype=np.int64),
            1,
        )


def _solve_rows(
    opt,
    starts: np.ndarray,
    coefficients: np.ndarray,
    ideal2: float,
    range2: float,
    bounds_obj,
    options: dict,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Minimise every row of ``starts`` under its own coefficients.

    Returns ``(X, audit)``: the per-row solutions and the solver's per-row
    audit entries (status, iterations, objective).
    """
    device, dtype = opt._device, tps.float_dtype()
    starts = np.asarray(starts, dtype=np.float64)
    n_rows, n_theta = starts.shape
    coefficient_tensor = torch.as_tensor(
        np.asarray(coefficients, dtype=np.float64), dtype=dtype, device=device
    )
    objective = _BatchedScalarizedObjective(
        opt, coefficient_tensor, ideal2, range2, n_theta
    )

    if bounds_obj is not None:
        lb = np.asarray(bounds_obj.lb, dtype=np.float64)
        ub = np.asarray(bounds_obj.ub, dtype=np.float64)
    else:
        lb = np.full(n_theta, -np.inf)
        ub = np.full(n_theta, np.inf)

    unknown = set(options) - _SWEEP_OPTION_KEYS
    if unknown:
        raise TypeError(
            "Unsupported batched trust-region Pareto option(s): "
            f"{', '.join(sorted(unknown))}"
        )
    # One chunk: each row carries its own coefficients, so the row order the
    # solver sees must be the row order of `coefficients`.
    solver_options = {"batch_size": n_rows, **options}
    result = bs.solve_batched_multistart(
        objective, "batched-tr", starts, lb, ub, solver_options
    )
    audit = list(getattr(result, "multistart_audit", []) or [])
    if len(audit) == n_rows:
        rows = np.stack(
            [np.asarray(entry["x_norm"], dtype=np.float64) for entry in audit]
        )
    else:  # pragma: no cover - the solver always audits every start
        rows = np.repeat(np.asarray(result.x, dtype=np.float64)[None, :], n_rows, 0)
        audit = [
            {"status": result.status, "success": result.success, "nit": result.nit}
            for _ in range(n_rows)
        ]
    return np.clip(rows, lb, ub), audit


def anchors(opt, x0: np.ndarray, bounds_obj, delta: float, options: dict):
    """Both payoff-table anchors as one two-row batched solve.

    Row ``k`` minimises ``f_k + delta * f_other`` -- the same augmented
    scalarization :func:`_pareto._anchor_solve` uses, so the payoff table is
    comparable with the host routes'.
    """
    x0 = np.asarray(x0, dtype=np.float64)
    starts = np.stack([x0, x0])
    coefficients = np.array(
        [[1.0, float(delta), 0.0, 0.0], [float(delta), 1.0, 0.0, 0.0]]
    )
    LOGGER.task("Pareto sweep: batched trust-region anchors (2 rows)")
    X, audit = _solve_rows(opt, starts, coefficients, 0.0, 1.0, bounds_obj, options)
    return X, audit


def sweep(
    opt,
    eps_grid: np.ndarray,
    x_a1: np.ndarray,
    x_a2: np.ndarray,
    ideal2: float,
    range2: float,
    delta: float,
    bounds_obj,
    mu: float | None = None,
    options: dict | None = None,
):
    """Every epsilon-subproblem as one batched trust-region solve.

    Starts are spread along the anchor-to-anchor segment, so each row begins
    near where its epsilon slice will land (the same initialization the
    projected-Adam prepass uses).
    """
    eps_grid = np.asarray(eps_grid, dtype=np.float64)
    n_rows = len(eps_grid)
    a1 = np.asarray(x_a1, dtype=np.float64)
    a2 = np.asarray(x_a2, dtype=np.float64)
    t = np.linspace(0.0, 1.0, n_rows)[:, None]
    starts = (1.0 - t) * a1[None, :] + t * a2[None, :]
    penalty_weight = float(opt._constraint_penalty if mu is None else mu)
    scaled_delta = float(delta) / (float(range2) if abs(float(range2)) > 1e-12 else 1.0)
    coefficients = np.stack(
        [
            np.ones(n_rows),
            np.full(n_rows, scaled_delta),
            np.full(n_rows, penalty_weight),
            eps_grid,
        ],
        axis=1,
    )
    LOGGER.task("Pareto sweep: batched trust-region sweep (%d epsilon rows)" % n_rows)
    return _solve_rows(
        opt, starts, coefficients, ideal2, range2, bounds_obj, options or {}
    )
