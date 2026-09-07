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

from twin4build.estimator._cuda_graph import CudaGraphCallable


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


class BatchedShootingEvaluator:
    """Derivative bundles for :class:`FastSingleShooting`."""

    # Direct replay is validated for primal values and first-order reverse-mode
    # gradients. PyTorch's jacfwd and nested higher-order transforms currently
    # trigger cudaErrorIllegalAddress when their captured graphs are replayed
    # on the full shooting horizon, even though their eager CUDA evaluations
    # are valid. Keep those bundles eager; objective-only line searches remain
    # captured for every method.
    _CAPTURE_SAFE_BUNDLES = frozenset({"values", "value_grad"})

    def __init__(self, objective, capture=False):
        self.objective = objective
        self.capture = bool(capture)
        self._graphs = {}
        self.stats = {}

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
        if x.device.type == "cuda":
            torch.cuda.synchronize(x.device)
        started = time.perf_counter()
        capture_bundle = (
            self.capture
            and x.device.type == "cuda"
            and name in self._CAPTURE_SAFE_BUNDLES
        )
        if not capture_bundle:
            output = fn(x)
        else:
            graph_key = (name, tuple(x.shape))
            graph = self._graphs.get(graph_key)
            if graph is None:
                graph = CudaGraphCallable(fn)
                self._graphs[graph_key] = graph
            output = graph(x)
            # CUDAGraph reuses static output buffers. Solver iterates retain
            # previous values across later replays, so return owned snapshots.
            if isinstance(output, tuple):
                output = tuple(value.clone() for value in output)
            else:
                output = output.clone()
        if x.device.type == "cuda":
            torch.cuda.synchronize(x.device)
        seconds = time.perf_counter() - started
        stat = self.stats.setdefault(
            name, {"calls": 0, "seconds": 0.0, "first_seconds": None}
        )
        stat["calls"] += 1
        stat["seconds"] += seconds
        if stat["first_seconds"] is None:
            stat["first_seconds"] = seconds
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
        candidate = torch.where((info == 0)[:, None], candidate, -grad)

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
    sqp_line_search_candidates=12,
    sqp_max_step=1.0,
):
    if max_nfev is None:
        evaluations_per_iteration = (
            sqp_line_search_candidates + 1
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
    bfgs_h = eye.clone()
    damping = torch.full((batch,), 1e-3, dtype=x.dtype, device=x.device)
    active = torch.ones(batch, dtype=torch.bool, device=x.device)
    converged_all = torch.zeros_like(active)
    failed = torch.zeros_like(active)
    stalled_slots = torch.zeros_like(active)
    stagnant = torch.zeros(batch, dtype=torch.int64, device=x.device)
    nit = torch.zeros(batch, dtype=torch.int64, device=x.device)
    nfev = torch.zeros(batch, dtype=torch.int64, device=x.device)
    njev = torch.zeros(batch, dtype=torch.int64, device=x.device)
    history = []

    if method == "batched-lm":
        residual, jac = evaluator.residual_jacobian(x)
        f = torch.sum(residual.square(), dim=1)
        g = 2.0 * torch.bmm(jac.transpose(1, 2), residual[:, :, None]).squeeze(-1)
    elif method == "batched-newton":
        f, g, hess = evaluator.value_grad_hessian(x)
    else:
        f, g = evaluator.value_grad(x)
    nfev += 1
    njev += 1

    for iteration in range(int(maxiter)):
        active &= nfev < int(max_nfev)
        pg = _projected_gradient(x, g, lb, ub)
        pg_norm = torch.amax(torch.abs(pg), dim=1)
        converged = active & (pg_norm <= gtol * (1.0 + torch.abs(f)))
        converged_all |= converged
        active &= ~converged
        derivatives_finite = torch.isfinite(g).all(dim=1)
        if method == "batched-lm":
            derivatives_finite &= torch.isfinite(jac).all(dim=(1, 2))
        elif method == "batched-newton":
            derivatives_finite &= torch.isfinite(hess).all(dim=(1, 2))
        failed_now = active & (~torch.isfinite(f) | ~derivatives_finite)
        failed |= failed_now
        active &= ~failed_now
        history.append(
            {
                "iteration": iteration,
                "best_objective": float(torch.min(f).detach().cpu()),
                "active": int(active.sum().detach().cpu()),
            }
        )
        if not bool(active.any()):
            break
        attempted = active.clone()

        if method == "batched-bfgs":
            direction = -torch.bmm(inv_h, pg[:, :, None]).squeeze(-1)
        elif method == "batched-sqp":
            direction = _solve_box_qp(bfgs_h, g, x, lb, ub, active)
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
            failed |= solve_failed
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
            x_trial, _f_trial, accepted, trial_calls = _batched_armijo(
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
        else:
            x_trial, _f_trial, accepted, trial_calls = _armijo(
                evaluator, x, f, pg, direction, active, lb, ub, max_backtracks
            )
        line_search_failed = attempted & ~accepted
        failed |= line_search_failed
        nfev += active.to(torch.int64) * trial_calls
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
        elif method == "batched-newton":
            f_new, g_new, hess_new = evaluator.value_grad_hessian(x)
            hess = torch.where(accepted[:, None, None], hess_new, hess)
            damping = torch.where(accepted, damping * 0.5, damping * 4.0)
        else:
            f_new, g_new = evaluator.value_grad(x)
        njev += active.to(torch.int64)
        nfev += active.to(torch.int64)
        f = torch.where(accepted, f_new, old_f)
        g = torch.where(accepted[:, None], g_new, old_g)

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
            bs = torch.bmm(bfgs_h, s[:, :, None]).squeeze(-1)
            sbs = torch.sum(s * bs, dim=1)
            sy = torch.sum(s * y, dim=1)
            use_damping = sy < 0.2 * sbs
            theta = torch.where(
                use_damping,
                0.8 * sbs / torch.clamp(sbs - sy, min=1e-30),
                torch.ones_like(sy),
            )
            r = theta[:, None] * y + (1.0 - theta)[:, None] * bs
            sr = torch.sum(s * r, dim=1)
            valid = (
                accepted & torch.isfinite(r).all(dim=1) & (sbs > 1e-14) & (sr > 1e-14)
            )
            updated = (
                bfgs_h
                - bs[:, :, None]
                * bs[:, None, :]
                / torch.clamp(sbs, min=1e-14)[:, None, None]
                + r[:, :, None]
                * r[:, None, :]
                / torch.clamp(sr, min=1e-14)[:, None, None]
            )
            updated = 0.5 * (updated + updated.transpose(1, 2))
            bfgs_h = torch.where(valid[:, None, None], updated, bfgs_h)

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

    success = converged_all & ~failed
    return {
        "x": x.detach(),
        "fun": f.detach(),
        "success": success.detach(),
        "failed": failed.detach(),
        "stalled": stalled_slots.detach(),
        "nit": nit.detach(),
        "nfev": nfev.detach(),
        "njev": njev.detach(),
        "history": history,
    }


def solve_batched_shooting(
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
    device = objective.est._device
    capture = bool(options.pop("capture", device.type == "cuda"))
    dtype = objective._sd.dtype
    starts = torch.as_tensor(x0, dtype=dtype, device=device)
    if starts.ndim == 1:
        starts = starts[None, :]
    lb_t = torch.as_tensor(lb, dtype=dtype, device=device)
    ub_t = torch.as_tensor(ub, dtype=dtype, device=device)
    evaluator = BatchedShootingEvaluator(objective, capture=capture)
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
        for key in ("x", "fun", "success", "failed", "stalled", "nit", "nfev", "njev"):
            solved[key] = solved[key][:keep]
        chunks.append(solved)
    elapsed = time.perf_counter() - started
    x_all = torch.cat([c["x"] for c in chunks])
    f_all = torch.cat([c["fun"] for c in chunks])
    success_all = torch.cat([c["success"] for c in chunks])
    failed_all = torch.cat([c["failed"] for c in chunks])
    stalled_all = torch.cat([c["stalled"] for c in chunks])
    nit_all = torch.cat([c["nit"] for c in chunks])
    nfev_all = torch.cat([c["nfev"] for c in chunks])
    njev_all = torch.cat([c["njev"] for c in chunks])
    eligible = torch.where(success_all, f_all, torch.full_like(f_all, float("inf")))
    best = int(torch.argmin(eligible if torch.isfinite(eligible).any() else f_all))
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
                        "failed_nonfinite_or_line_search"
                        if bool(failed_all[i].cpu())
                        else (
                            "stalled"
                            if bool(stalled_all[i].cpu())
                            else "iteration_or_evaluation_limit"
                        )
                    )
                ),
                "objective": float(f_all[i].cpu()),
                "nit": int(nit_all[i].cpu()),
                "nfev": int(nfev_all[i].cpu()),
                "njev": int(njev_all[i].cpu()),
                "x_norm": x_all[i].cpu().numpy(),
            }
        )
    return SimpleNamespace(
        x=x_all[best].cpu().numpy(),
        fun=float(f_all[best].cpu()),
        success=bool(success_all[best].cpu()),
        message=(
            "Converged"
            if bool(success_all[best].cpu())
            else "Iteration/evaluation limit reached"
        ),
        nit=int(nit_all[best].cpu()),
        nfev=int(nfev_all.sum().cpu()),
        njev=int(njev_all.sum().cpu()),
        elapsed=elapsed,
        multistart_audit=audit,
        derivative_stats=evaluator.stats,
        iteration_history=[chunk["history"] for chunk in chunks],
    )
