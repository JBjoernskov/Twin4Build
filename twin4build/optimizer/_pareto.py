"""Bi-objective Pareto front generation via the augmented epsilon-constraint
method (AUGMECON) for :class:`~twin4build.optimizer.optimizer.Optimizer`.

Building-energy multi-objective optimization is dominated by evolutionary
methods (NSGA-II and friends) that ignore gradients and handle dynamics
constraints poorly.  Twin4Build's simulator is differentiable, so each Pareto
point can instead be an exact gradient-based NLP solve:

1. Solve the two **anchor** problems (each objective alone) -> ideal/nadir
   estimates of the second objective.
2. Normalize f2 with them and lay an **epsilon grid** between the anchors.
3. Per grid point solve ``min f1 + delta*f2_norm  s.t.  f2_norm <= eps``
   (SLSQP with the epsilon constraint as a HARD scipy constraint, warm-started
   from the neighbouring solution).  The small ``delta*f2_norm`` term is the
   AUGMECON augmentation: it guarantees *properly* Pareto-optimal points
   instead of weakly optimal ones, and -- unlike a weighted sum -- the
   epsilon-constraint scheme recovers non-convex front regions.
4. Filter dominated points; report the finite-difference front slope
   ``-d f1 / d eps`` (the marginal price of the second objective; exact
   multipliers arrive with the IPOPT backend).

**GPU batching**: before the sequential exact sweep, an optional *batched
prepass* stacks all N epsilon-subproblems into one tensor ``(N, n_theta)``
and minimizes the sum of per-copy penalty losses with projected Adam -- the
subproblems are independent, so ONE backward pass per iteration yields every
copy's gradient, and the batched rollout is exactly the workload shape where
a GPU pays off.  The prepass solutions then warm-start the exact SLSQP
polish.

Known limits (by construction):

- Bi-objective only; epsilon grids scale poorly beyond ~3 objectives
  (use NBI-style methods there).
- A uniform epsilon grid gives non-uniform point spacing along steep front
  segments; refine adaptively if needed.
- Each point is only locally optimal (inherited NLP non-convexity).
- Prepass solutions are approximate (quadratic penalty) until polished.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from scipy.optimize import minimize

import twin4build.utils.types as tps
from twin4build.utils.print_progress import LOGGER


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class ParetoResult:
    """Pareto sweep output.

    Rows are ordered from the f1-anchor (``eps=1``, second objective
    unconstrained) to the f2-anchor (``eps=0``).  ``f1``/``f2`` are the
    PHYSICAL objective means (natural units, unsigned); ``f1_min``/``f2_min``
    are the internal min-oriented normalized values ("min" objectives keep
    their sign, "max" objectives are negated) used for dominance and slope.
    """

    eps: np.ndarray
    f1: np.ndarray
    f2: np.ndarray
    f1_min: np.ndarray
    f2_min: np.ndarray
    theta: np.ndarray  # (n_points, n_theta) normalized decision vectors
    success: np.ndarray
    nit: np.ndarray
    pareto_mask: np.ndarray
    slope: np.ndarray  # -d f1_min / d eps (finite difference)
    ideal: tuple  # (min f1_min, min f2_min) from the payoff table
    nadir: tuple  # (f1_min at f2-anchor, f2_min at f1-anchor)
    labels: tuple  # ("comp.port (min)", "comp.port (max)")
    _optimizer: object = field(default=None, repr=False)

    def apply(self, i: int) -> None:
        """Write Pareto point ``i``'s decision trajectories into the model and
        re-simulate, so component histories hold that solution."""
        self._optimizer.apply_solution(self.theta[i])

    def plot(self, ax=None):
        """Scatter the front in physical objective space (Pareto-optimal
        points connected, dominated points crossed out)."""
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(6, 5))
        m = self.pareto_mask
        order = np.argsort(self.f2[m])
        ax.plot(
            self.f2[m][order], self.f1[m][order], "o-", label="Pareto front"
        )
        if (~m).any():
            ax.plot(self.f2[~m], self.f1[~m], "x", label="dominated")
        ax.set_xlabel(self.labels[1])
        ax.set_ylabel(self.labels[0])
        ax.grid(True, alpha=0.3)
        ax.legend()
        return ax


def _pareto_mask(f1: np.ndarray, f2: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    """Non-dominated mask for a min-min bi-objective point set."""
    n = len(f1)
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if j == i:
                continue
            if (
                f1[j] <= f1[i] + tol
                and f2[j] <= f2[i] + tol
                and (f1[j] < f1[i] - tol or f2[j] < f2[i] - tol)
            ):
                mask[i] = False
                break
    return mask


# ---------------------------------------------------------------------------
# Subproblem evaluation (one rollout serves objective AND constraint)
# ---------------------------------------------------------------------------
class _EpsSubproblem:
    """Callback provider for one scalarized subproblem.

    Serves scipy's objective ``fun``/``jac`` and the epsilon-constraint
    ``c(x) = eps - f2_norm(x) >= 0`` with its jacobian from a SINGLE forward
    pass per iterate (cached on the theta bytes; the fast composed objective
    adds one extra backward for the constraint gradient, the object-graph
    fallback evaluates a 2-row ``jacrev``).

    ``obj_index`` picks which objective is scalarized (anchor solves) and
    ``delta`` adds the AUGMECON term; f2 normalization is
    ``f2_norm = (f2_min - ideal2) / range2`` (identity until the anchors set
    it via :meth:`set_normalization`).
    """

    def __init__(self, opt, obj_index: int = 0, delta: float = 0.0):
        self.opt = opt
        self.obj_index = obj_index
        self.delta = float(delta)
        self.ideal2 = 0.0
        self.range2 = 1.0
        self._key = None
        self._c = {}

    def set_normalization(self, ideal2: float, nadir2: float) -> None:
        self.ideal2 = float(ideal2)
        self.range2 = float(nadir2 - ideal2)

    # -- evaluation ---------------------------------------------------------
    def _fast_compute(self, z: torch.Tensor) -> dict:
        fast = self.opt._fast_obj
        z = z.detach().clone().requires_grad_(True)
        p = fast.parts(z)
        pen = fast.penalty(p)
        f2n = (p.objs[1] - self.ideal2) / self.range2
        f = p.objs[self.obj_index] + pen
        if self.delta:
            f = f + self.delta * f2n
        (gf,) = torch.autograd.grad(f, z, retain_graph=True)
        (gc,) = torch.autograd.grad(f2n, z)
        return {
            "f": float(f),
            "gf": gf.detach().cpu().numpy().astype(np.float64),
            "f2n": float(f2n),
            "gc": gc.detach().cpu().numpy().astype(np.float64),
            "objs": [float(o) for o in p.objs],
            "phys": [float(o) for o in p.phys],
        }

    def _graph_compute(self, z: torch.Tensor) -> dict:
        opt = self.opt

        def fn(zz):
            p = opt._graph_parts(zz)
            pen = torch.tensor(
                0.0, dtype=tps.float_dtype(), device=opt._device
            )
            for e in p.eq:
                pen = pen + e
            if p.ineq is not None:
                pen = pen + p.ineq
            f2n = (p.objs[1] - self.ideal2) / self.range2
            f = p.objs[self.obj_index] + pen
            if self.delta:
                f = f + self.delta * f2n
            out = torch.stack([f, f2n])
            aux = (
                out.detach(),
                [float(o) for o in p.objs],
                [float(o) for o in p.phys],
            )
            return out, aux

        J, aux = torch.func.jacrev(fn, has_aux=True)(z)
        vals, objs, phys = aux
        J = J.detach().cpu().numpy().astype(np.float64)
        return {
            "f": float(vals[0]),
            "gf": J[0],
            "f2n": float(vals[1]),
            "gc": J[1],
            "objs": objs,
            "phys": phys,
        }

    def _compute(self, x: np.ndarray) -> dict:
        key = x.tobytes()
        if key == self._key:
            return self._c
        z = torch.tensor(
            np.asarray(x, dtype=np.float64),
            dtype=tps.float_dtype(),
            device=self.opt._device,
        )
        if self.opt._fast_obj is not None:
            self._c = self._fast_compute(z)
        else:
            self._c = self._graph_compute(z)
        self._key = key
        return self._c

    # -- scipy-facing callbacks ----------------------------------------------
    def fun(self, x):
        return self._compute(x)["f"]

    def jac(self, x):
        return self._compute(x)["gf"]

    def f2_norm(self, x):
        return self._compute(x)["f2n"]

    def f2_norm_jac(self, x):
        return self._compute(x)["gc"]

    def values(self, x) -> dict:
        return self._compute(x)


def _solve_subproblem(
    sub: _EpsSubproblem,
    x_init: np.ndarray,
    bounds_obj,
    solver_options: dict,
    method_name: str,
    eps: Optional[float] = None,
):
    """One SLSQP/trust-constr solve; ``eps`` adds the hard epsilon constraint."""
    cons = ()
    if eps is not None:
        cons = [
            {
                "type": "ineq",
                "fun": lambda x, e=eps: e - sub.f2_norm(x),
                "jac": lambda x: -sub.f2_norm_jac(x),
            }
        ]
    return minimize(
        sub.fun,
        np.asarray(x_init, dtype=np.float64),
        jac=sub.jac,
        method=method_name,
        bounds=bounds_obj,
        constraints=cons,
        options=dict(solver_options),
    )


# ---------------------------------------------------------------------------
# Batched torch prepass
# ---------------------------------------------------------------------------
def batched_prepass(
    opt,
    eps_grid: np.ndarray,
    x_a1: np.ndarray,
    x_a2: np.ndarray,
    ideal2: float,
    range2: float,
    delta: float,
    bounds_obj,
    mu: float = None,
    lr: float = 0.05,
    max_iter: int = 150,
    patience: int = 15,
    rel_tol: float = 1e-5,
) -> np.ndarray:
    """Approximate ALL epsilon-subproblems at once with a batched penalty loss.

    Copies are stacked into ``theta (N, n)``; per-copy loss is the scalarized
    objective plus a quadratic penalty ``mu * relu(f2_norm - eps_i)^2``.  The
    copies are independent, so ``sum_i loss_i`` yields every per-copy gradient
    from ONE backward pass, and the batched forward (``torch.func.vmap`` over
    the composed rollout, with a loop fallback if vmap rejects an op) is the
    batched-kernel workload shape where a GPU pays off.  Projected Adam
    (clamp to the box bounds after each step) with plateau stopping.

    Returns the ``(N, n)`` prepass solutions (float64 numpy), used as warm
    starts for the exact SLSQP polish.
    """
    fast = opt._fast_obj
    dev, dt = opt._device, tps.float_dtype()
    N, n = len(eps_grid), len(x_a1)
    mu = float(opt._constraint_penalty if mu is None else mu)

    # Spread the initial copies along the anchor-to-anchor segment: each copy
    # starts near where its epsilon slice will land.
    t = torch.linspace(0.0, 1.0, N, dtype=dt, device=dev).unsqueeze(1)
    a1 = torch.tensor(x_a1, dtype=dt, device=dev)
    a2 = torch.tensor(x_a2, dtype=dt, device=dev)
    theta = ((1.0 - t) * a1 + t * a2).detach().requires_grad_(True)
    eps_t = torch.tensor(
        np.asarray(eps_grid, dtype=np.float64), dtype=dt, device=dev
    )

    if bounds_obj is not None:
        lb = torch.tensor(np.asarray(bounds_obj.lb), dtype=dt, device=dev)
        ub = torch.tensor(np.asarray(bounds_obj.ub), dtype=dt, device=dev)
    else:
        lb = ub = None

    def per_copy(th_i, eps_i):
        p = fast.parts(th_i)
        pen = fast.penalty(p)
        f2n = (p.objs[1] - ideal2) / range2
        return (
            p.objs[0]
            + pen
            + delta * f2n
            + mu * torch.relu(f2n - eps_i) ** 2
        )

    use_vmap = True
    optimz = torch.optim.Adam([theta], lr=lr)
    best, stall = float("inf"), 0
    for it in range(max_iter):
        optimz.zero_grad()
        try:
            if use_vmap:
                losses = torch.func.vmap(per_copy)(theta, eps_t)
            else:
                losses = torch.stack(
                    [per_copy(theta[i], eps_t[i]) for i in range(N)]
                )
            total = losses.sum()
            total.backward()
        except Exception as exc:
            if not use_vmap:
                raise
            # vmap rejected an op somewhere in the composed rollout: fall back
            # to a per-copy loop (still one graph and one backward per iter).
            use_vmap = False
            LOGGER.config(
                "Pareto prepass: vmap unavailable (%s); looping copies", exc
            )
            optimz.zero_grad()
            losses = torch.stack(
                [per_copy(theta[i], eps_t[i]) for i in range(N)]
            )
            total = losses.sum()
            total.backward()
        optimz.step()
        if lb is not None:
            with torch.no_grad():
                theta.clamp_(min=lb, max=ub)

        val = float(total)
        if best - val > rel_tol * max(1.0, abs(best)):
            best, stall = val, 0
        else:
            stall += 1
            if stall >= patience:
                break

    LOGGER.config(
        "Pareto prepass: %d iteration(s), batched=%s, total loss %.6f",
        it + 1, use_vmap, float(total),
    )
    return theta.detach().cpu().numpy().astype(np.float64)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def pareto_front(
    opt,
    n_points: int = 11,
    delta: float = 1e-3,
    method: tuple = ("scipy", "SLSQP", "ad"),
    use_prepass: bool = True,
    prepass_options: dict = None,
    options: dict = None,
) -> ParetoResult:
    """Run the augmented epsilon-constraint sweep.

    Assumes the optimizer task attributes are already set up (variables,
    ``_objectives = [objective1, objective2]``, constraints, periods) --
    :meth:`Optimizer.pareto_front` does that before delegating here.
    """
    options = dict(options or {})
    x0, bounds_obj = opt._prepare_scipy_problem(method, options)
    solver_options = dict(options)  # objective-related keys were consumed
    method_name = method[1]

    labels = tuple(
        f"{c.id}.{p} ({t})" for c, p, t in opt._objectives
    )

    # -- anchors (payoff table) ---------------------------------------------
    LOGGER.task("Pareto sweep: anchor solves")
    sub1 = _EpsSubproblem(opt, obj_index=0)
    res_a1 = _solve_subproblem(sub1, x0, bounds_obj, solver_options, method_name)
    v_a1 = sub1.values(res_a1.x)

    sub2 = _EpsSubproblem(opt, obj_index=1)
    res_a2 = _solve_subproblem(
        sub2, res_a1.x, bounds_obj, solver_options, method_name
    )
    v_a2 = sub2.values(res_a2.x)

    ideal = (v_a1["objs"][0], v_a2["objs"][1])
    nadir = (v_a2["objs"][0], v_a1["objs"][1])
    ideal2, nadir2 = ideal[1], nadir[1]
    range2 = nadir2 - ideal2
    LOGGER.config(
        "Payoff table: f1 in [%.6f, %.6f], f2 in [%.6f, %.6f] (min-oriented)",
        ideal[0], nadir[0], ideal2, nadir2,
    )

    rows = []  # (eps, values dict, theta, success, nit)
    rows.append((1.0, v_a1, res_a1.x, bool(res_a1.success), int(res_a1.nit)))

    if abs(range2) < 1e-9:
        # Non-conflicting objectives: the front is a single point.
        LOGGER.warning(
            "Objectives are non-conflicting (f2 range ~ 0); returning the "
            "anchor solutions only."
        )
        rows.append(
            (0.0, v_a2, res_a2.x, bool(res_a2.success), int(res_a2.nit))
        )
    else:
        # -- interior epsilon grid ------------------------------------------
        n_interior = max(0, n_points - 2)
        eps_grid = np.linspace(1.0, 0.0, n_interior + 2)[1:-1]

        sub = _EpsSubproblem(opt, obj_index=0, delta=delta)
        sub.set_normalization(ideal2, nadir2)

        warm = None
        if use_prepass and n_interior > 0 and opt._fast_obj is not None:
            LOGGER.task("Pareto sweep: batched prepass (%d copies)" % n_interior)
            try:
                warm = batched_prepass(
                    opt, eps_grid, res_a1.x, res_a2.x, ideal2, range2, delta,
                    bounds_obj, **(prepass_options or {}),
                )
            except Exception as exc:
                LOGGER.warning(
                    "Pareto prepass failed (%s); falling back to sequential "
                    "warm starts.", exc,
                )

        LOGGER.task("Pareto sweep: %d epsilon solves" % n_interior)
        x_prev = res_a1.x
        for i, eps in enumerate(eps_grid):
            x_init = warm[i] if warm is not None else x_prev
            res = _solve_subproblem(
                sub, x_init, bounds_obj, solver_options, method_name, eps=eps
            )
            x_prev = res.x
            v = sub.values(res.x)
            rows.append((float(eps), v, res.x, bool(res.success), int(res.nit)))
            LOGGER.iter(
                "eps=%.3f | f1=%.6f f2=%.6f | feas=%.2e | success=%s",
                eps, v["objs"][0], v["objs"][1],
                max(0.0, v["f2n"] - eps), res.success,
            )

        rows.append(
            (0.0, v_a2, res_a2.x, bool(res_a2.success), int(res_a2.nit))
        )

    # -- assemble -------------------------------------------------------------
    eps_arr = np.array([r[0] for r in rows])
    f1_min = np.array([r[1]["objs"][0] for r in rows])
    f2_min = np.array([r[1]["objs"][1] for r in rows])
    f1_phys = np.array([r[1]["phys"][0] for r in rows])
    f2_phys = np.array([r[1]["phys"][1] for r in rows])
    theta = np.stack([np.asarray(r[2], dtype=np.float64) for r in rows])
    success = np.array([r[3] for r in rows])
    nit = np.array([r[4] for r in rows])

    mask = _pareto_mask(f1_min, f2_min)
    # -d f1 / d eps: the local exchange rate between the objectives (a
    # finite-difference stand-in for the epsilon-constraint multiplier).
    if len(eps_arr) >= 2 and abs(range2) >= 1e-9:
        slope = -np.gradient(f1_min, eps_arr)
    else:
        slope = np.zeros_like(eps_arr)

    LOGGER.ok(
        "Pareto sweep complete: %d point(s), %d non-dominated",
        len(rows), int(mask.sum()),
    )
    return ParetoResult(
        eps=eps_arr,
        f1=f1_phys,
        f2=f2_phys,
        f1_min=f1_min,
        f2_min=f2_min,
        theta=theta,
        success=success,
        nit=nit,
        pareto_mask=mask,
        slope=slope,
        ideal=ideal,
        nadir=nadir,
        labels=labels,
        _optimizer=opt,
    )
