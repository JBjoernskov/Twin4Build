"""Experimental batched bound-constrained single-shooting solvers.

The expensive model evaluations stay in torch and share a leading start-slot
dimension.  Solver bookkeeping is intentionally eager: for the small parameter
vectors used by building calibration, dense batched linear algebra is cheap
compared with the sequential differentiable rollout.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np
import torch

from twin4build.utils import _cuda_graph
from twin4build.utils._cuda_graph import CudaGraphCallable


def _projected_gradient(x, g, lb, ub, eps=1e-12):
    pg = g.clone()
    pg = torch.where((x <= lb + eps) & (g > 0), torch.zeros_like(pg), pg)
    pg = torch.where((x >= ub - eps) & (g < 0), torch.zeros_like(pg), pg)
    return pg


def _fixed_basis_hessian(loss_fn, x):
    """Dense exact Hessian using a static VJP basis."""
    grad_fn = torch.func.grad(loss_fn)
    _, pullback = torch.func.vjp(grad_fn, x)
    basis = torch.eye(x.numel(), dtype=x.dtype, device=x.device)
    return torch.func.vmap(lambda row: pullback(row)[0])(basis)


class BatchedObjectiveEvaluator:
    """Derivative bundles for ``FunctionalEstimationObjective``."""

    # Direct replay is validated for primal values and first-order reverse-mode
    # gradients (plain autograd, see ``batched_value_and_grad``).  functorch
    # transforms under capture -- jacfwd, nested higher-order transforms, and
    # on some builds even ``vmap(grad_and_value)`` -- record graphs that raise
    # cudaErrorIllegalAddress on a later replay although their eager CUDA
    # evaluations are valid.  Keep those bundles eager; objective-only line
    # searches remain captured for every method.
    _CAPTURE_SAFE_BUNDLES = frozenset({"values", "value_grad"})
    _NONFINITE_PENALTY = 1e20

    def __init__(self, objective):
        self.objective = objective
        self.capture = objective.est.simulator.execution_backend == "cuda_graph"
        self._graphs = {}
        self.stats = {}
        self.last_nonfinite = {}

    def values(self, x):
        fn = self.objective.batched_loss
        return self._call("values", fn, x)

    def value_grad(self, x):
        fn = self.objective.batched_value_and_grad
        return self._call("value_grad", fn, x)

    def residual_jacobian(self, x):
        fn = self.objective.batched_residual_and_jacobian
        return self._call("residual_jacobian", fn, x)

    def value_grad_hessian(self, x):
        loss = lambda th: self.objective.loss(th, transform_mode=True)
        grad_value = torch.func.grad_and_value(loss)

        def fn(batch):
            grad, value = torch.func.vmap(grad_value)(batch)
            hess = torch.func.vmap(lambda th: _fixed_basis_hessian(loss, th))(batch)
            return value, grad, hess

        return self._call("value_grad_hessian", fn, x)

    def _call(self, name, fn, x):
        try:
            return self._call_inner(name, fn, x)
        except Exception as exc:
            if x.device.type == "cuda" and hasattr(exc, "add_note"):
                exc.add_note(_cuda_graph.phase_note())
            raise

    def _call_inner(self, name, fn, x):
        if x.device.type == "cuda":
            torch.cuda.synchronize(x.device)
        started = time.perf_counter()
        capture_bundle = (
            self.capture
            and x.device.type == "cuda"
            and name in self._CAPTURE_SAFE_BUNDLES
        )
        if not capture_bundle:
            if x.device.type == "cuda":
                _cuda_graph.mark_phase(f"{name}:eager")
            output = fn(x)
            graph_event = None
        else:
            graph_key = (name, tuple(x.shape))
            graph = self._graphs.get(graph_key)
            if graph is None:
                graph = CudaGraphCallable(fn)
                self._graphs[graph_key] = graph
                graph_event = "capture"
            else:
                graph_event = "replay"
            output = graph(x)
            # CUDAGraph reuses static output buffers. Solver iterates retain
            # previous values across later replays, so return owned snapshots.
            _cuda_graph.mark_phase(f"{name}:{graph_event}:clone-outputs")
            if isinstance(output, tuple):
                output = tuple(value.clone() for value in output)
            else:
                output = output.clone()
        if x.device.type == "cuda":
            _cuda_graph.mark_phase(f"{name}:finite-bundle")
        output = self._finite_bundle(name, output)
        if x.device.type == "cuda":
            _cuda_graph.mark_phase(f"{name}:final-sync")
            torch.cuda.synchronize(x.device)
            _cuda_graph.mark_phase(f"{name}:done")
        seconds = time.perf_counter() - started
        stat = self.stats.setdefault(
            name,
            {
                "calls": 0,
                "seconds": 0.0,
                "first_seconds": None,
                "captures": 0,
                "replays": 0,
            },
        )
        stat["calls"] += 1
        stat["seconds"] += seconds
        if graph_event == "capture":
            stat["captures"] += 1
        elif graph_event == "replay":
            stat["replays"] += 1
        if stat["first_seconds"] is None:
            stat["first_seconds"] = seconds
        return output

    def _finite_bundle(self, name, output):
        """Return finite solver inputs while preserving a rejection mask."""
        if name == "values":
            invalid = ~torch.isfinite(output)
            output = torch.where(
                invalid, torch.full_like(output, self._NONFINITE_PENALTY), output
            )
        elif name == "value_grad":
            value, grad = output
            invalid = ~torch.isfinite(value) | ~torch.isfinite(grad).all(dim=1)
            value = torch.where(
                invalid, torch.full_like(value, self._NONFINITE_PENALTY), value
            )
            grad = torch.where(invalid[:, None], torch.zeros_like(grad), grad)
            output = value, grad
        else:
            invalid = torch.zeros(
                output[0].shape[0],
                dtype=torch.bool,
                device=output[0].device,
            )
        self.last_nonfinite[name] = invalid
        return output


def _armijo(evaluator, x, f, g, direction, active, lb, ub, max_backtracks):
    """Independent masked Armijo searches evaluated as one fixed batch."""
    alpha = torch.ones(x.shape[0], dtype=x.dtype, device=x.device)
    accepted = ~active
    x_best = x.clone()
    f_best = f.clone()
    slope = torch.sum(g * direction, dim=1)
    # Any non-descent model falls back to projected steepest descent.
    bad = (slope >= 0) & active
    direction = torch.where(bad[:, None], -g, direction)
    slope = torch.sum(g * direction, dim=1)
    calls = 0
    for _ in range(max_backtracks):
        calls += 1
        trial = torch.clamp(x + alpha[:, None] * direction, lb, ub)
        ft = evaluator.values(trial)
        projected_slope = torch.sum(g * (trial - x), dim=1)
        ok = (
            active & ~accepted & torch.isfinite(ft) & (ft <= f + 1e-4 * projected_slope)
        )
        x_best = torch.where(ok[:, None], trial, x_best)
        f_best = torch.where(ok, ft, f_best)
        accepted |= ok
        alpha = torch.where(accepted, alpha, alpha * 0.5)
        if bool((accepted | ~active).all()):
            break
    return x_best, f_best, accepted & active, calls


def _batched_armijo(evaluator, x, f, g, direction, active, lb, ub, candidates):
    """Evaluate a geometric line-search schedule in one batched rollout."""
    alphas = torch.pow(
        torch.as_tensor(0.5, dtype=x.dtype, device=x.device),
        torch.arange(candidates, dtype=x.dtype, device=x.device),
    )
    trial = torch.clamp(
        x[:, None, :] + alphas[None, :, None] * direction[:, None, :],
        lb,
        ub,
    )
    batch, _, n_theta = trial.shape
    ft = evaluator.values(trial.reshape(batch * candidates, n_theta)).reshape(
        batch, candidates
    )
    step = trial - x[:, None, :]
    projected_slope = torch.sum(g[:, None, :] * step, dim=2)
    acceptable = (
        active[:, None]
        & torch.isfinite(ft)
        & (ft <= f[:, None] + 1e-4 * projected_slope)
    )
    accepted = acceptable.any(dim=1)
    first = torch.argmax(acceptable.to(torch.int64), dim=1)
    row = torch.arange(batch, device=x.device)
    selected_x = trial[row, first]
    selected_f = ft[row, first]
    return (
        torch.where(accepted[:, None], selected_x, x),
        torch.where(accepted, selected_f, f),
        accepted,
        candidates,
    )


def _adaptive_armijo(
    evaluator,
    x,
    f,
    g,
    direction,
    active,
    lb,
    ub,
    max_backtracks,
    *,
    initial_alpha=1.0,
):
    """Backtrack unresolved slots independently and retain finite diagnostics."""
    alpha = torch.full(
        (x.shape[0],), float(initial_alpha), dtype=x.dtype, device=x.device
    )
    accepted = ~active
    selected_alpha = torch.zeros_like(alpha)
    x_best = x.clone()
    f_best = f.clone()
    finite_candidates = torch.zeros(
        x.shape[0], dtype=torch.int64, device=x.device
    )
    evaluations = torch.zeros_like(finite_candidates)
    for _ in range(int(max_backtracks)):
        unresolved = active & ~accepted
        if not bool(unresolved.any()):
            break
        trial = torch.clamp(x + alpha[:, None] * direction, lb, ub)
        ft = evaluator.values(trial)
        invalid = getattr(evaluator, "last_nonfinite", {}).get(
            "values", ~torch.isfinite(ft)
        )
        finite = unresolved & ~invalid
        finite_candidates += finite.to(torch.int64)
        evaluations += unresolved.to(torch.int64)
        projected_slope = torch.sum(g * (trial - x), dim=1)
        ok = finite & (ft <= f + 1e-4 * projected_slope)
        x_best = torch.where(ok[:, None], trial, x_best)
        f_best = torch.where(ok, ft, f_best)
        selected_alpha = torch.where(ok, alpha, selected_alpha)
        accepted |= ok
        alpha = torch.where(accepted, alpha, alpha * 0.5)
    return (
        x_best,
        f_best,
        accepted & active,
        evaluations,
        selected_alpha,
        finite_candidates,
    )


def _curvature_layout(spec, n_theta, device):
    """Normalize an integer, block-size list, or explicit index layout."""
    if spec is None:
        raw = [list(range(n_theta))]
    elif isinstance(spec, (int, np.integer)):
        size = int(spec)
        if size <= 0 or n_theta % size:
            raise ValueError(
                "sqp_curvature_blocks integer must evenly divide n_theta"
            )
        raw = [list(range(first, first + size)) for first in range(0, n_theta, size)]
    else:
        supplied = list(spec)
        if supplied and all(isinstance(value, (int, np.integer)) for value in supplied):
            if sum(int(value) for value in supplied) != n_theta:
                raise ValueError("sqp_curvature_blocks sizes must sum to n_theta")
            raw = []
            first = 0
            for value in supplied:
                size = int(value)
                if size <= 0:
                    raise ValueError("sqp_curvature_blocks sizes must be positive")
                raw.append(list(range(first, first + size)))
                first += size
        else:
            raw = [list(block) for block in supplied]
    flat = [int(index) for block in raw for index in block]
    if not raw or any(not block for block in raw) or sorted(flat) != list(range(n_theta)):
        raise ValueError(
            "sqp_curvature_blocks must partition every theta index exactly once"
        )
    return [torch.as_tensor(block, dtype=torch.long, device=device) for block in raw]


def _identity_curvature(layout, batch, dtype, device):
    return [
        torch.eye(len(index), dtype=dtype, device=device)
        .expand(batch, len(index), len(index))
        .clone()
        for index in layout
    ]


def _block_sqp_direction(models, layout, grad, x, lb, ub, active):
    direction = torch.zeros_like(grad)
    resets = torch.zeros_like(active)
    fallbacks = torch.zeros_like(active)
    updated_models = []
    for model, index in zip(models, layout):
        block_eye = torch.eye(
            len(index), dtype=x.dtype, device=x.device
        ).expand(x.shape[0], len(index), len(index))
        block_direction, block_model, reset, fallback = _safeguard_sqp_direction(
            model,
            grad.index_select(1, index),
            x.index_select(1, index),
            lb.index_select(0, index),
            ub.index_select(0, index),
            active,
            block_eye,
        )
        direction[:, index] = block_direction
        updated_models.append(block_model)
        resets |= reset
        fallbacks |= fallback
    return direction, updated_models, resets, fallbacks


def _solve_box_qp(hess, grad, x, lb, ub, active, tol=1e-10):
    """Solve small positive-definite, box-constrained QPs by active sets."""
    batch, n_theta = grad.shape
    step_lb = lb - x
    step_ub = ub - x
    step = torch.zeros_like(grad)
    at_lower = torch.zeros_like(grad, dtype=torch.bool)
    at_upper = torch.zeros_like(grad, dtype=torch.bool)
    eye = torch.eye(n_theta, dtype=x.dtype, device=x.device).expand(
        batch, n_theta, n_theta
    )

    # At most n variables can enter and n can leave the working set. Fixed
    # iteration count keeps all tensor shapes stable for batched GPU solves.
    for _ in range(2 * n_theta + 1):
        fixed = at_lower | at_upper | ~active[:, None]
        fixed_step = torch.where(fixed, step, torch.zeros_like(step))
        rhs = -grad - torch.bmm(hess, fixed_step[:, :, None]).squeeze(-1)
        free = ~fixed
        matrix = torch.where(
            free[:, :, None] & free[:, None, :],
            hess,
            torch.zeros_like(hess),
        )
        matrix = matrix + torch.diag_embed(fixed.to(x.dtype))
        rhs = torch.where(free, rhs, torch.zeros_like(rhs))
        solved, info = torch.linalg.solve_ex(matrix, rhs[:, :, None])
        candidate = fixed_step + solved.squeeze(-1)
        fallback = -_projected_gradient(x, grad, lb, ub)
        candidate = torch.where((info == 0)[:, None], candidate, fallback)

        below = free & (candidate < step_lb - tol)
        above = free & (candidate > step_ub + tol)
        primal_violation = below | above
        if bool(primal_violation.any()):
            step = torch.where(below, step_lb, step)
            step = torch.where(above, step_ub, step)
            at_lower |= below
            at_upper |= above
            continue

        step = torch.clamp(candidate, step_lb, step_ub)
        model_grad = grad + torch.bmm(hess, step[:, :, None]).squeeze(-1)
        lower_bad = at_lower & (model_grad < -tol)
        upper_bad = at_upper & (model_grad > tol)
        dual_violation = lower_bad | upper_bad
        if not bool(dual_violation.any()):
            break

        # Release only the worst multiplier per slot to avoid cycling.
        violation = torch.where(
            lower_bad,
            -model_grad,
            torch.where(upper_bad, model_grad, torch.zeros_like(model_grad)),
        )
        release = torch.argmax(violation, dim=1)
        release_mask = (
            torch.nn.functional.one_hot(release, num_classes=n_theta).to(torch.bool)
            & dual_violation
        )
        at_lower &= ~release_mask
        at_upper &= ~release_mask
        step = torch.where(release_mask, torch.zeros_like(step), step)

    return torch.where(active[:, None], step, torch.zeros_like(step))


def _safeguard_sqp_direction(hess, grad, x, lb, ub, active, eye):
    """Reset a bad BFGS model and guarantee a feasible descent direction."""
    direction = _solve_box_qp(hess, grad, x, lb, ub, active)
    slope = torch.sum(grad * direction, dim=1)
    invalid = active & (
        ~torch.isfinite(direction).all(dim=1) | ~torch.isfinite(slope) | (slope >= 0)
    )

    # SLSQP resets its BFGS factorization when the QP produces a positive
    # directional derivative. Do the same independently for each start slot,
    # then resolve the box QP with an identity Hessian.
    reset_hess = torch.where(invalid[:, None, None], eye, hess)
    if bool(invalid.any()):
        retry = _solve_box_qp(reset_hess, grad, x, lb, ub, invalid)
        direction = torch.where(invalid[:, None], retry, direction)

    retry_slope = torch.sum(grad * direction, dim=1)
    needs_gradient_fallback = active & (
        ~torch.isfinite(direction).all(dim=1)
        | ~torch.isfinite(retry_slope)
        | (retry_slope >= 0)
    )
    projected_gradient = _projected_gradient(x, grad, lb, ub)
    direction = torch.where(
        needs_gradient_fallback[:, None], -projected_gradient, direction
    )
    return direction, reset_hess, invalid, needs_gradient_fallback


def _solve_chunk(
    evaluator,
    x0,
    method,
    lb,
    ub,
    *,
    maxiter=200,
    max_nfev=None,
    gtol=1e-4,
    ftol=1e-8,
    patience=4,
    max_backtracks=25,
    max_step=0.25,
    sqp_line_search_candidates=25,
    sqp_max_step=1.0,
    sqp_curvature_blocks=None,
):
    if max_nfev is None:
        evaluations_per_iteration = (
            2 * sqp_line_search_candidates + 1
            if method == "batched-sqp"
            else max_backtracks + 1
        )
        max_nfev = 1 + int(maxiter) * evaluations_per_iteration
    x = torch.clamp(x0.clone(), lb, ub)
    batch, n_theta = x.shape
    eye = (
        torch.eye(n_theta, dtype=x.dtype, device=x.device)
        .expand(batch, n_theta, n_theta)
        .clone()
    )
    inv_h = eye.clone()
    curvature_layout = _curvature_layout(
        sqp_curvature_blocks if method == "batched-sqp" else None,
        n_theta,
        x.device,
    )
    bfgs_models = _identity_curvature(
        curvature_layout, batch, x.dtype, x.device
    )
    damping = torch.full((batch,), 1e-3, dtype=x.dtype, device=x.device)
    active = torch.ones(batch, dtype=torch.bool, device=x.device)
    converged_all = torch.zeros_like(active)
    nonfinite_failed = torch.zeros_like(active)
    line_search_failed = torch.zeros_like(active)
    stalled_slots = torch.zeros_like(active)
    hessian_resets = torch.zeros(batch, dtype=torch.int64, device=x.device)
    descent_fallbacks = torch.zeros(batch, dtype=torch.int64, device=x.device)
    stagnant = torch.zeros(batch, dtype=torch.int64, device=x.device)
    nit = torch.zeros(batch, dtype=torch.int64, device=x.device)
    nfev = torch.zeros(batch, dtype=torch.int64, device=x.device)
    njev = torch.zeros(batch, dtype=torch.int64, device=x.device)
    history = []
    last_alpha = torch.zeros(batch, dtype=x.dtype, device=x.device)
    finite_candidate_total = torch.zeros(
        batch, dtype=torch.int64, device=x.device
    )
    restoration_uses = torch.zeros(
        batch, dtype=torch.int64, device=x.device
    )

    if method == "batched-lm":
        residual, jac = evaluator.residual_jacobian(x)
        f = torch.sum(residual.square(), dim=1)
        g = 2.0 * torch.bmm(jac.transpose(1, 2), residual[:, :, None]).squeeze(-1)
        objective_nonfinite = ~torch.isfinite(f)
    elif method == "batched-newton":
        f, g, hess = evaluator.value_grad_hessian(x)
        objective_nonfinite = ~torch.isfinite(f)
    else:
        f, g = evaluator.value_grad(x)
        objective_nonfinite = evaluator.last_nonfinite.get(
            "value_grad", ~torch.isfinite(f)
        ).clone()
    nfev += 1
    njev += 1

    for iteration in range(int(maxiter)):
        active &= nfev < int(max_nfev)
        pg = _projected_gradient(x, g, lb, ub)
        pg_norm = torch.amax(torch.abs(pg), dim=1)
        derivatives_finite = torch.isfinite(g).all(dim=1)
        if method == "batched-lm":
            derivatives_finite &= torch.isfinite(jac).all(dim=(1, 2))
        elif method == "batched-newton":
            derivatives_finite &= torch.isfinite(hess).all(dim=(1, 2))
        failed_now = active & (
            objective_nonfinite | ~torch.isfinite(f) | ~derivatives_finite
        )
        nonfinite_failed |= failed_now
        active &= ~failed_now
        converged = active & (pg_norm <= gtol * (1.0 + torch.abs(f)))
        converged_all |= converged
        active &= ~converged
        history.append(
            {
                "iteration": iteration,
                "best_objective": float(torch.min(f).detach().cpu()),
                "active": int(active.sum().detach().cpu()),
                "projected_gradient_max": float(
                    torch.max(
                        torch.where(
                            torch.isfinite(pg_norm),
                            pg_norm,
                            torch.full_like(pg_norm, float("inf")),
                        )
                    )
                    .detach()
                    .cpu()
                ),
            }
        )
        if not bool(active.any()):
            break
        attempted = active.clone()
        last_alpha = torch.where(attempted, torch.zeros_like(last_alpha), last_alpha)
        finite_before = finite_candidate_total.clone()
        resets_before = hessian_resets.clone()
        fallbacks_before = descent_fallbacks.clone()
        restorations_before = restoration_uses.clone()

        if method == "batched-bfgs":
            direction = -torch.bmm(inv_h, pg[:, :, None]).squeeze(-1)
        elif method == "batched-sqp":
            direction, bfgs_models, reset, fallback = _block_sqp_direction(
                bfgs_models, curvature_layout, g, x, lb, ub, active
            )
            hessian_resets += reset.to(torch.int64)
            descent_fallbacks += fallback.to(torch.int64)
        elif method == "batched-lm":
            safe_jac = torch.where(active[:, None, None], jac, torch.zeros_like(jac))
            safe_g = torch.where(active[:, None], g, torch.zeros_like(g))
            normal = 2.0 * torch.bmm(safe_jac.transpose(1, 2), safe_jac)
            diagonal_scale = torch.clamp(
                torch.diagonal(normal, dim1=1, dim2=2), min=1.0
            )
            matrix = normal + torch.diag_embed(damping[:, None] * diagonal_scale)
            matrix = torch.where(active[:, None, None], matrix, eye)
            direction, solve_info = torch.linalg.solve_ex(matrix, (-safe_g)[:, :, None])
            direction = direction.squeeze(-1)
            solve_failed = active & (solve_info != 0)
            nonfinite_failed |= solve_failed
            active &= ~solve_failed
            direction = torch.where(
                active[:, None], direction, torch.zeros_like(direction)
            )
        else:
            safe_hess = torch.where(active[:, None, None], hess, torch.zeros_like(hess))
            safe_g = torch.where(active[:, None], g, torch.zeros_like(g))
            sym = 0.5 * (safe_hess + safe_hess.transpose(1, 2))
            # Increase a per-slot shift until all active systems are positive
            # definite; inactive slots receive identity systems.
            shift = damping.clone()
            direction = -pg
            for _ in range(12):
                matrix = sym + shift[:, None, None] * eye
                chol, info = torch.linalg.cholesky_ex(matrix)
                ok = (info == 0) | ~active
                solved = torch.cholesky_solve((-safe_g)[:, :, None], chol).squeeze(-1)
                direction = torch.where(ok[:, None], solved, direction)
                if bool(ok.all()):
                    break
                shift = torch.where(ok, shift, shift * 10.0)
            damping = shift

        direction_norm = torch.amax(torch.abs(direction), dim=1)
        step_limit = sqp_max_step if method == "batched-sqp" else max_step
        direction = (
            direction
            * torch.clamp(
                step_limit / torch.clamp(direction_norm, min=1e-30),
                max=1.0,
            )[:, None]
        )

        old_x, old_f, old_g = x, f, g
        if method == "batched-sqp":
            (
                x_trial,
                _f_trial,
                accepted,
                trial_evaluations,
                accepted_alpha,
                finite_candidates,
            ) = _adaptive_armijo(
                evaluator,
                x,
                f,
                g,
                direction,
                active,
                lb,
                ub,
                sqp_line_search_candidates,
            )
            nfev += trial_evaluations
            finite_candidate_total += finite_candidates

            # Restore rejected slots with fresh block curvature and projected
            # steepest descent. The second adaptive search starts cautiously
            # and can recover after nonfinite large trials within its budget.
            retry_active = attempted & ~accepted
            if bool(retry_active.any()):
                fresh_models = _identity_curvature(
                    curvature_layout, batch, x.dtype, x.device
                )
                bfgs_models = [
                    torch.where(retry_active[:, None, None], fresh, current)
                    for current, fresh in zip(bfgs_models, fresh_models)
                ]
                hessian_resets += retry_active.to(torch.int64)
                restoration_uses += retry_active.to(torch.int64)
                retry_direction = -pg
                retry_norm = torch.amax(torch.abs(retry_direction), dim=1)
                retry_direction = (
                    retry_direction
                    * torch.clamp(
                        sqp_max_step / torch.clamp(retry_norm, min=1e-30),
                        max=1.0,
                    )[:, None]
                )
                (
                    retry_x,
                    retry_f,
                    retry_accepted,
                    retry_evaluations,
                    retry_alpha,
                    retry_finite,
                ) = _adaptive_armijo(
                    evaluator,
                    x,
                    f,
                    g,
                    retry_direction,
                    retry_active,
                    lb,
                    ub,
                    sqp_line_search_candidates,
                    initial_alpha=0.5,
                )
                nfev += retry_evaluations
                finite_candidate_total += retry_finite
                x_trial = torch.where(retry_accepted[:, None], retry_x, x_trial)
                _f_trial = torch.where(retry_accepted, retry_f, _f_trial)
                accepted_alpha = torch.where(
                    retry_accepted, retry_alpha, accepted_alpha
                )
                accepted |= retry_accepted
            last_alpha = torch.where(accepted, accepted_alpha, last_alpha)
        else:
            x_trial, _f_trial, accepted, trial_calls = _armijo(
                evaluator, x, f, pg, direction, active, lb, ub, max_backtracks
            )
            nfev += active.to(torch.int64) * trial_calls
        failed_search_now = attempted & ~accepted
        line_search_failed |= failed_search_now
        x = torch.where(accepted[:, None], x_trial, x)

        if method == "batched-lm":
            residual_new, jac_new = evaluator.residual_jacobian(x)
            f_new = torch.sum(residual_new.square(), dim=1)
            g_new = 2.0 * torch.bmm(
                jac_new.transpose(1, 2), residual_new[:, :, None]
            ).squeeze(-1)
            jac = torch.where(accepted[:, None, None], jac_new, jac)
            residual = torch.where(accepted[:, None], residual_new, residual)
            damping = torch.where(accepted, damping * 0.5, damping * 4.0)
            candidate_nonfinite = ~torch.isfinite(f_new)
        elif method == "batched-newton":
            f_new, g_new, hess_new = evaluator.value_grad_hessian(x)
            hess = torch.where(accepted[:, None, None], hess_new, hess)
            damping = torch.where(accepted, damping * 0.5, damping * 4.0)
            candidate_nonfinite = ~torch.isfinite(f_new)
        else:
            f_new, g_new = evaluator.value_grad(x)
            candidate_nonfinite = evaluator.last_nonfinite.get(
                "value_grad", ~torch.isfinite(f_new)
            )
        njev += active.to(torch.int64)
        nfev += active.to(torch.int64)
        f = torch.where(accepted, f_new, old_f)
        g = torch.where(accepted[:, None], g_new, old_g)
        objective_nonfinite = torch.where(
            accepted, candidate_nonfinite, objective_nonfinite
        )

        if method == "batched-bfgs":
            s = x - old_x
            y = g - old_g
            ys = torch.sum(y * s, dim=1)
            ss = torch.sum(s * s, dim=1)
            # Powell-style curvature damping: modify y only when the observed
            # secant curvature is too small to keep the inverse approximation
            # positive definite.
            target_curvature = 1e-4 * ss
            correction = torch.clamp(
                (target_curvature - ys) / torch.clamp(ss, min=1e-30),
                min=0.0,
            )
            y = y + correction[:, None] * s
            ys = torch.sum(y * s, dim=1)
            valid = accepted & (ss > 1e-20) & (ys > 1e-12)
            rho = torch.where(
                valid,
                1.0 / torch.clamp(ys, min=1e-12),
                torch.zeros_like(ys),
            )
            sy = s[:, :, None] * y[:, None, :]
            ident_minus = eye - rho[:, None, None] * sy
            updated = (
                torch.bmm(
                    torch.bmm(ident_minus, inv_h),
                    ident_minus.transpose(1, 2),
                )
                + rho[:, None, None] * s[:, :, None] * s[:, None, :]
            )
            inv_h = torch.where(valid[:, None, None], updated, eye)
        elif method == "batched-sqp":
            s = x - old_x
            y = g - old_g
            updated_models = []
            invalid_any = torch.zeros_like(active)
            for model, index in zip(bfgs_models, curvature_layout):
                block_s = s.index_select(1, index)
                block_y = y.index_select(1, index)
                bs = torch.bmm(model, block_s[:, :, None]).squeeze(-1)
                sbs = torch.sum(block_s * bs, dim=1)
                sy = torch.sum(block_s * block_y, dim=1)
                use_damping = sy < 0.2 * sbs
                theta = torch.where(
                    use_damping,
                    0.8 * sbs / torch.clamp(sbs - sy, min=1e-30),
                    torch.ones_like(sy),
                )
                r = theta[:, None] * block_y + (1.0 - theta)[:, None] * bs
                sr = torch.sum(block_s * r, dim=1)
                valid = (
                    accepted
                    & torch.isfinite(r).all(dim=1)
                    & (sbs > 1e-14)
                    & (sr > 1e-14)
                )
                updated = (
                    model
                    - bs[:, :, None]
                    * bs[:, None, :]
                    / torch.clamp(sbs, min=1e-14)[:, None, None]
                    + r[:, :, None]
                    * r[:, None, :]
                    / torch.clamp(sr, min=1e-14)[:, None, None]
                )
                updated = 0.5 * (updated + updated.transpose(1, 2))
                invalid = attempted & accepted & ~valid
                block_eye = torch.eye(
                    len(index), dtype=x.dtype, device=x.device
                ).expand(batch, len(index), len(index))
                updated_models.append(
                    torch.where(
                        valid[:, None, None],
                        updated,
                        torch.where(invalid[:, None, None], block_eye, model),
                    )
                )
                invalid_any |= invalid
            bfgs_models = updated_models
            hessian_resets += invalid_any.to(torch.int64)

        rel = torch.abs(old_f - f) / torch.clamp(torch.abs(old_f), min=1.0)
        stagnant = torch.where(
            accepted & (rel <= ftol), stagnant + 1, torch.zeros_like(stagnant)
        )
        # Small objective changes alone do not satisfy first-order KKT
        # conditions. Stop stalled slots without labelling them converged.
        stalled_now = attempted & accepted & (stagnant >= patience)
        stalled_slots |= stalled_now
        active &= ~stalled_now
        active &= accepted
        nit += attempted.to(torch.int64)
        history[-1].update(
            {
                "alpha": last_alpha.detach().cpu().tolist(),
                "projected_gradient": pg_norm.detach().cpu().tolist(),
                "finite_candidates": (
                    finite_candidate_total - finite_before
                ).detach().cpu().tolist(),
                "hessian_resets": (
                    hessian_resets - resets_before
                ).detach().cpu().tolist(),
                "descent_fallbacks": (
                    descent_fallbacks - fallbacks_before
                ).detach().cpu().tolist(),
                "restoration_uses": (
                    restoration_uses - restorations_before
                ).detach().cpu().tolist(),
            }
        )

    success = converged_all & ~nonfinite_failed & ~line_search_failed
    return {
        "x": x.detach(),
        "fun": f.detach(),
        "success": success.detach(),
        "nonfinite_failed": nonfinite_failed.detach(),
        "line_search_failed": line_search_failed.detach(),
        "stalled": stalled_slots.detach(),
        "nit": nit.detach(),
        "nfev": nfev.detach(),
        "njev": njev.detach(),
        "hessian_resets": hessian_resets.detach(),
        "descent_fallbacks": descent_fallbacks.detach(),
        "restoration_uses": restoration_uses.detach(),
        "finite_candidates": finite_candidate_total.detach(),
        "last_alpha": last_alpha.detach(),
        "curvature_block_sizes": [len(index) for index in curvature_layout],
        "history": history,
    }


def solve_batched_multistart(
    objective,
    method,
    x0,
    lb,
    ub,
    options=None,
):
    """Solve one or more normalized starts and return a SciPy-like result."""
    options = dict(options or {})
    batch_size = int(options.pop("batch_size", len(x0)))
    if "sqp_max_backtracks" in options:
        if "sqp_line_search_candidates" in options:
            raise TypeError(
                "Specify only one of sqp_max_backtracks and "
                "sqp_line_search_candidates"
            )
        options["sqp_line_search_candidates"] = options.pop(
            "sqp_max_backtracks"
        )
    options.setdefault(
        "sqp_curvature_blocks",
        getattr(objective, "sqp_curvature_blocks", None),
    )
    device = objective.est._device
    for legacy in ("capture", "capture_derivatives", "capture_hessian"):
        if legacy in options:
            raise TypeError(
                f"Custom solver option {legacy!r} has been removed; CUDA graph "
                "capture is selected by Simulator.execution_backend."
            )
    dtype = objective._sd.dtype
    starts = torch.as_tensor(x0, dtype=dtype, device=device)
    if starts.ndim == 1:
        starts = starts[None, :]
    lb_t = torch.as_tensor(lb, dtype=dtype, device=device)
    ub_t = torch.as_tensor(ub, dtype=dtype, device=device)
    evaluator = BatchedObjectiveEvaluator(objective)
    chunks = []
    chunk_lengths = []
    started = time.perf_counter()
    for first in range(0, starts.shape[0], batch_size):
        start_chunk = starts[first : first + batch_size]
        chunk_lengths.append(start_chunk.shape[0])
        if start_chunk.shape[0] < batch_size:
            padding = start_chunk[-1:].expand(batch_size - start_chunk.shape[0], -1)
            start_chunk = torch.cat([start_chunk, padding], dim=0)
        solved = _solve_chunk(
            evaluator,
            start_chunk,
            method,
            lb_t,
            ub_t,
            **options,
        )
        keep = chunk_lengths[-1]
        for key in (
            "x",
            "fun",
            "success",
            "nonfinite_failed",
            "line_search_failed",
            "stalled",
            "nit",
            "nfev",
            "njev",
            "hessian_resets",
            "descent_fallbacks",
            "restoration_uses",
            "finite_candidates",
            "last_alpha",
        ):
            solved[key] = solved[key][:keep]
        chunks.append(solved)
    elapsed = time.perf_counter() - started
    x_all = torch.cat([c["x"] for c in chunks])
    f_all = torch.cat([c["fun"] for c in chunks])
    success_all = torch.cat([c["success"] for c in chunks])
    nonfinite_failed_all = torch.cat([c["nonfinite_failed"] for c in chunks])
    line_search_failed_all = torch.cat(
        [c["line_search_failed"] for c in chunks]
    )
    stalled_all = torch.cat([c["stalled"] for c in chunks])
    nit_all = torch.cat([c["nit"] for c in chunks])
    nfev_all = torch.cat([c["nfev"] for c in chunks])
    njev_all = torch.cat([c["njev"] for c in chunks])
    hessian_resets_all = torch.cat([c["hessian_resets"] for c in chunks])
    descent_fallbacks_all = torch.cat([c["descent_fallbacks"] for c in chunks])
    restoration_uses_all = torch.cat([c["restoration_uses"] for c in chunks])
    finite_candidates_all = torch.cat([c["finite_candidates"] for c in chunks])
    last_alpha_all = torch.cat([c["last_alpha"] for c in chunks])
    finite_objectives = torch.where(
        torch.isfinite(f_all), f_all, torch.full_like(f_all, float("inf"))
    )
    eligible = torch.where(
        success_all, finite_objectives, torch.full_like(f_all, float("inf"))
    )
    best = int(
        torch.argmin(
            eligible
            if torch.isfinite(eligible).any()
            else finite_objectives
        )
    )
    audit = []
    for i in range(starts.shape[0]):
        audit.append(
            {
                "start_index": i,
                "success": bool(success_all[i].cpu()),
                "status": (
                    "converged"
                    if bool(success_all[i].cpu())
                    else (
                        "failed_nonfinite"
                        if bool(nonfinite_failed_all[i].cpu())
                        else (
                            "failed_line_search"
                            if bool(line_search_failed_all[i].cpu())
                            else (
                                "stalled"
                                if bool(stalled_all[i].cpu())
                                else "iteration_or_evaluation_limit"
                            )
                        )
                    )
                ),
                "objective": float(f_all[i].cpu()),
                "nit": int(nit_all[i].cpu()),
                "nfev": int(nfev_all[i].cpu()),
                "njev": int(njev_all[i].cpu()),
                "hessian_resets": int(hessian_resets_all[i].cpu()),
                "descent_fallbacks": int(descent_fallbacks_all[i].cpu()),
                "restoration_uses": int(restoration_uses_all[i].cpu()),
                "finite_candidates": int(finite_candidates_all[i].cpu()),
                "last_alpha": float(last_alpha_all[i].cpu()),
                "x_norm": x_all[i].cpu().numpy(),
            }
        )
    selected_status = audit[best]["status"]
    messages = {
        "converged": "Projected-KKT conditions satisfied",
        "failed_nonfinite": "Stopped after a nonfinite objective or derivative",
        "failed_line_search": "Adaptive line search and restoration failed",
        "stalled": "Stopped after objective stagnation without KKT convergence",
        "iteration_or_evaluation_limit": "Iteration/evaluation limit reached",
    }
    return SimpleNamespace(
        x=x_all[best].cpu().numpy(),
        fun=float(f_all[best].cpu()),
        success=bool(success_all[best].cpu()),
        status=selected_status,
        message=messages[selected_status],
        nit=int(nit_all[best].cpu()),
        nfev=int(nfev_all[best].cpu()),
        njev=int(njev_all[best].cpu()),
        aggregate_nfev=int(nfev_all.sum().cpu()),
        aggregate_njev=int(njev_all.sum().cpu()),
        elapsed=elapsed,
        multistart_audit=audit,
        derivative_stats=evaluator.stats,
        iteration_history=[chunk["history"] for chunk in chunks],
        curvature_block_sizes=chunks[0]["curvature_block_sizes"],
    )
