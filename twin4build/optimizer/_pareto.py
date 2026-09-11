"""Bi-objective Pareto front generation via the augmented epsilon-constraint
method (AUGMECON) for :class:`~twin4build.optimizer.optimizer.Optimizer`.

Building-energy multi-objective optimization is dominated by evolutionary
methods (NSGA-II and friends) that ignore gradients and handle dynamics
constraints poorly.  Twin4Build's simulator is differentiable, so each Pareto
point can instead be an exact gradient-based NLP solve:

1. Solve the two **anchor** problems with an AUGMECON-augmented objective
   ``f_i + delta * f_other``.  The anchors only build the payoff table
   (ideal/nadir estimates for normalization); they are NOT reported as front
   points, because an anchor whose objective has a flat optimum (e.g. a relu
   discomfort residual, which is exactly zero for every sufficiently-heated
   trajectory) stops at an arbitrary point of the flat region and is only
   *weakly* Pareto optimal.
2. Normalize f2 with them and lay an **epsilon grid** spanning the anchors,
   endpoints included.
3. Per grid point solve ``min f1 + delta*f2_norm  s.t.  f2_norm <= eps``
   (SLSQP or IPOPT with one HARD epsilon constraint, warm-started from the
   neighbouring solution).  The small ``delta*f2_norm`` term is the
   AUGMECON augmentation: it guarantees *properly* Pareto-optimal points
   instead of weakly optimal ones, and -- unlike a weighted sum -- the
   epsilon-constraint scheme recovers non-convex front regions.  Solving the
   eps=0 endpoint this way (instead of reporting the raw f2 anchor)
   approaches the f2 optimum from the infeasible side, where the constraint
   gradient is informative, and finds the CHEAPEST f2-optimal solution.
4. Filter dominated points; report the finite-difference front slope
   ``-d f1 / d eps`` (the marginal price of the second objective; exact
   multipliers arrive with the IPOPT backend).

SLSQP remains a direct-shooting solve driven by the captured fixed-shape
first-order bundle. IPOPT Pareto uses the separate sparse one-step collocation
transcription in :mod:`twin4build.optimizer._pareto_collocation`: augmented
boundary states are decision variables, continuity defects are hard
equalities, and replica-colored JVP/VJP derivatives expose an exact
control/global arrowhead with replica-local state blocks (directly captured on
CUDA). Directional work per segment is independent of the compiled replica
count. Both numerical solvers remain host-side callback consumers.

**GPU batching**: before the sequential exact sweep, an optional *batched
prepass* stacks all N epsilon-subproblems into one tensor ``(N, n_theta)``
and minimizes the sum of per-copy penalty losses with projected Adam -- the
subproblems are independent, so ONE backward pass per iteration yields every
copy's gradient, and the batched rollout is exactly the workload shape where
a GPU pays off. The prepass solutions then warm-start the exact host-solver
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

from typing import Optional

import numpy as np
import torch
from scipy.optimize import minimize

import twin4build.utils.types as tps
from twin4build.solvers.ipopt import solve_ipopt_constrained
from twin4build.optimizer._pareto_collocation import pareto_front_collocation
from twin4build.optimizer._pareto_common import (
    ParetoResult,
    _pareto_mask,
    batched_prepass,
)
from twin4build.utils._cuda_graph import CudaGraphCallable
from twin4build.utils.logger import LOGGER


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Subproblem evaluation (one rollout serves objective AND constraint)
# ---------------------------------------------------------------------------
class _EpsSubproblem:
    """Callback provider for one scalarized subproblem.

    Serves the host solver's objective ``fun``/``jac`` and epsilon-constraint
    ``c(x) = eps - f2_norm(x) >= 0`` with its jacobian from a SINGLE forward
    pass per iterate (cached on the theta bytes; the composed objective
    adds one extra backward for the constraint gradient, the object-graph
    fallback evaluates a 2-row ``jacrev``).

    ``obj_index`` picks which objective is scalarized (anchor solves),
    ``con_index`` which one the epsilon constraint bounds (the second
    objective in the sweep; the FIRST during the lexicographic anchor
    polish), and ``delta`` adds the AUGMECON term.  The constrained
    objective is normalized as ``(f - ideal) / range`` (identity until
    :meth:`set_normalization`).
    """

    def __init__(
        self,
        opt,
        obj_index: int = 0,
        con_index: int = 1,
        delta: float = 0.0,
    ):
        self.opt = opt
        self.obj_index = obj_index
        self.con_index = con_index
        self.delta = float(delta)
        self.ideal2 = 0.0
        self.range2 = 1.0
        self._key = None
        self._c = {}
        self._bundle_graph = None
        self._hessian_graph = None
        self.capture_derivatives = (
            opt.simulator.execution_backend == "cuda_graph"
            and opt._device.type == "cuda"
        )
        # The reachable direct-shooting Pareto route is SciPy SLSQP, which
        # never asks for a Hessian. IPOPT is collocation-only.
        self.capture_hessian = False
        self.stats = {
            "bundle_requested": opt.simulator.execution_backend == "cuda_graph",
            "bundle_enabled": self.capture_derivatives,
            "bundle_captured": False,
            "bundle_calls": 0,
            "bundle_replays": 0,
            "hessian_requested": self.capture_hessian,
            "hessian_enabled": self.capture_hessian,
            "hessian_captured": False,
            "hessian_calls": 0,
            "hessian_replays": 0,
        }

    def set_normalization(self, ideal2: float, nadir2: float) -> None:
        self.ideal2 = float(ideal2)
        self.range2 = float(nadir2 - ideal2)
        # Captured graphs close over these constants.  A changed payoff table
        # must never replay an old normalization.
        self._key = None
        self._c = {}
        self.close()

    def close(self):
        """Release callback graphs; safe for repeated lifecycle cleanup."""
        for name in ("_bundle_graph", "_hessian_graph"):
            graph = getattr(self, name, None)
            if graph is not None:
                graph.close()
                setattr(self, name, None)

    # -- evaluation ---------------------------------------------------------
    def _parts(self, z: torch.Tensor):
        if self.opt._functional_objective is not None:
            return self.opt._functional_objective.parts(
                z,
                transform_mode=(self.capture_derivatives or self.capture_hessian),
            )
        return self.opt._graph_parts(z)

    def _value_vector(self, z: torch.Tensor) -> torch.Tensor:
        p = self._parts(z)
        pen = torch.zeros((), dtype=z.dtype, device=z.device)
        for e in p.eq:
            pen = pen + e
        if p.ineq is not None:
            pen = pen + p.ineq
        f2n = (p.objs[self.con_index] - self.ideal2) / self.range2
        f = p.objs[self.obj_index] + pen + self.delta * f2n
        return torch.cat(
            (torch.stack((f, f2n)), torch.stack(p.objs), torch.stack(p.phys))
        )

    def _bundle_tensor(self, z: torch.Tensor) -> torch.Tensor:
        # One primal rollout builds every reported value. Two reverse sweeps
        # share that graph and produce the exact objective and epsilon
        # gradients; this explicit pattern is CUDA-graph safe on the full
        # composed model, unlike capturing a jacrev transform.
        zz = z.detach().clone().requires_grad_(True)
        values = self._value_vector(zz)
        (gf,) = torch.autograd.grad(values[0], zz, retain_graph=True)
        (gc,) = torch.autograd.grad(values[1], zz)
        return torch.cat((values.detach(), gf, gc))

    def _compute_tensor(self, z: torch.Tensor) -> torch.Tensor:
        self.stats["bundle_calls"] += 1
        if not self.capture_derivatives:
            return self._bundle_tensor(z)
        if self._bundle_graph is None:
            self._bundle_graph = CudaGraphCallable(self._bundle_tensor)
            output = self._bundle_graph(z)
            self.stats["bundle_captured"] = True
        else:
            output = self._bundle_graph(z)
            self.stats["bundle_replays"] += 1
        # CUDAGraph outputs alias static storage and the host solver keeps
        # values from earlier callbacks.  Always return an owned snapshot.
        return output.clone()

    def _unpack(self, bundle: torch.Tensor, n: int) -> dict:
        n_values = 2 + len(self.opt._objectives) * 2
        values = bundle[:n_values].detach().cpu().numpy().astype(np.float64)
        jac = bundle[n_values:].reshape(2, n).detach().cpu().numpy().astype(np.float64)
        return {
            "f": float(values[0]),
            "gf": jac[0],
            "f2n": float(values[1]),
            "gc": jac[1],
            "objs": values[2:4].tolist(),
            "phys": values[4:6].tolist(),
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
        self._c = self._unpack(self._compute_tensor(z), z.numel())
        self._key = key
        return self._c

    def _hessian_tensor(
        self, z: torch.Tensor, sigma: torch.Tensor, lam: torch.Tensor
    ) -> torch.Tensor:
        def lagrangian(zz):
            values = self._value_vector(zz)
            return sigma * values[0] + lam * values[1]

        # For scalar objectives, forward-over-reverse forms the full Hessian
        # more efficiently than differentiating every gradient component with
        # a reverse-over-reverse VJP basis.  The fixed-shape transform is
        # directly captured and replayed by CudaGraphCallable below.
        return torch.func.hessian(lagrangian)(z)

    def hessian(self, x, sigma: float, lam_g) -> np.ndarray:
        z = torch.as_tensor(
            np.asarray(x, dtype=np.float64),
            dtype=tps.float_dtype(),
            device=self.opt._device,
        )
        sigma_t = torch.as_tensor(sigma, dtype=z.dtype, device=z.device)
        lam_t = torch.as_tensor(
            np.asarray(lam_g, dtype=np.float64).reshape(-1)[0],
            dtype=z.dtype,
            device=z.device,
        )
        self.stats["hessian_calls"] += 1
        if self.capture_hessian:
            if self._hessian_graph is None:
                self._hessian_graph = CudaGraphCallable(self._hessian_tensor)
                dense = self._hessian_graph(z, sigma_t, lam_t)
                self.stats["hessian_captured"] = True
            else:
                dense = self._hessian_graph(z, sigma_t, lam_t)
                self.stats["hessian_replays"] += 1
            dense = dense.clone()
        else:
            dense = self._hessian_tensor(z, sigma_t, lam_t)
        iu = np.triu_indices(z.numel())
        return dense.detach().cpu().numpy().astype(np.float64)[iu]

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


def _anchor_solve(
    opt,
    obj_index: int,
    x_init: np.ndarray,
    bounds_obj,
    solver_options: dict,
    method_name: str,
    delta: float,
):
    """Augmented anchor solve (payoff-table entry).

    Minimizes ``f_obj + delta * f_other`` instead of ``f_obj`` alone
    (approximate lexicographic payoff table).  Without the delta term, an
    anchor whose objective has a flat optimum (e.g. a relu discomfort
    residual, which is exactly 0 for EVERY sufficiently-heated trajectory)
    stops at an arbitrary point of the flat region and is only *weakly*
    Pareto optimal: the gradient vanishes and the solver declares
    convergence while the other objective is far from its best attainable
    value.  The delta term keeps a descent direction alive across the flat
    region and drives the solve to the proper Pareto anchor.

    Returns ``(x, values_dict, success, nit, callback_stats)``.
    """
    sub = _EpsSubproblem(
        opt,
        obj_index=obj_index,
        con_index=1 - obj_index,
        delta=delta,
    )
    res = _solve_subproblem(sub, x_init, bounds_obj, solver_options, method_name)
    v = sub.values(res.x)
    output = (
        res.x,
        v,
        bool(res.success),
        int(res.nit) if res.nit is not None else 0,
        dict(sub.stats),
    )
    sub.close()
    return output


def _solve_subproblem(
    sub: _EpsSubproblem,
    x_init: np.ndarray,
    bounds_obj,
    solver_options: dict,
    method_name: str,
    eps: Optional[float] = None,
):
    """Solve one SLSQP or IPOPT epsilon subproblem."""
    if method_name.lower() == "ipopt":
        x_init = np.asarray(x_init, dtype=np.float64)
        n = x_init.size
        if bounds_obj is None:
            lb = np.full(n, -np.inf)
            ub = np.full(n, np.inf)
        else:
            lb = np.asarray(bounds_obj.lb, dtype=np.float64)
            ub = np.asarray(bounds_obj.ub, dtype=np.float64)
        rows = np.zeros(n, dtype=np.int64)
        cols = np.arange(n, dtype=np.int64)
        hr, hc = np.triu_indices(n)
        return solve_ipopt_constrained(
            x_init,
            lb,
            ub,
            sub.fun,
            sub.jac,
            1,
            lambda x: np.asarray([sub.f2_norm(x)], dtype=np.float64),
            sub.f2_norm_jac,
            rows,
            cols,
            options=solver_options,
            hess_vals=sub.hessian,
            hess_rows=hr,
            hess_cols=hc,
            lbg=np.asarray([-np.inf]),
            ubg=np.asarray([np.inf if eps is None else eps]),
        )
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
    if tuple(method) == ("casadi", "ipopt", "ad", "collocation"):
        return pareto_front_collocation(
            opt,
            n_points=n_points,
            delta=delta,
            method=tuple(method[:3]),
            use_prepass=use_prepass,
            prepass_options=prepass_options,
            options=options,
        )

    options = dict(options or {})
    removed_capture = {
        "capture",
        "capture_derivatives",
        "capture_hessian",
    }.intersection(options)
    if removed_capture:
        raise TypeError(
            f"Removed Pareto option(s): {', '.join(sorted(removed_capture))}; "
            "capture is selected by Simulator.execution_backend."
        )
    x0, bounds_obj = opt._prepare_scipy_problem(method, options)
    opt._pareto_prepass_stats = {
        "requested": False,
        "enabled": False,
        "captured": False,
        "replays": 0,
        "fallback_reason": "batched prepass not run",
    }
    solver_options = dict(options)  # objective-related keys were consumed
    method_name = method[1]
    capture_requested = opt.simulator.execution_backend == "cuda_graph"
    capture_derivatives = capture_requested and opt._device.type == "cuda"
    capture_hessian = capture_derivatives and method_name.lower() == "ipopt"

    labels = tuple(f"{c.id}.{p} ({t})" for c, p, t in opt._objectives)

    # -- anchors (lexicographic payoff table) ---------------------------------
    LOGGER.task("Pareto sweep: anchor solves")
    x_a1, v_a1, ok_a1, nit_a1, stats_a1 = _anchor_solve(
        opt,
        0,
        x0,
        bounds_obj,
        solver_options,
        method_name,
        delta,
    )
    x_a2, v_a2, ok_a2, nit_a2, stats_a2 = _anchor_solve(
        opt,
        1,
        x_a1,
        bounds_obj,
        solver_options,
        method_name,
        delta,
    )

    ideal = (v_a1["objs"][0], v_a2["objs"][1])
    nadir = (v_a2["objs"][0], v_a1["objs"][1])
    ideal2, nadir2 = ideal[1], nadir[1]
    range2 = nadir2 - ideal2
    LOGGER.config(
        "Payoff table: f1 in [%.6f, %.6f], f2 in [%.6f, %.6f] (min-oriented)",
        ideal[0],
        nadir[0],
        ideal2,
        nadir2,
    )

    rows = []  # (eps, values dict, theta, success, nit)

    if abs(range2) < 1e-9:
        # Non-conflicting objectives: the front is a single point.
        LOGGER.warning(
            "Objectives are non-conflicting (f2 range ~ 0); returning the "
            "anchor solutions only."
        )
        rows.append((1.0, v_a1, x_a1, ok_a1, nit_a1))
        rows.append((0.0, v_a2, x_a2, ok_a2, nit_a2))
    else:
        # -- epsilon grid (endpoints included) --------------------------------
        # Every front point -- endpoints too -- is solved as an AUGMECON
        # epsilon-subproblem (min f1 + delta*f2n s.t. f2n <= eps) with
        # warm-started sequential solves.  The anchors above only build the
        # payoff table: a raw anchor with a flat optimum (e.g. relu
        # discomfort at exactly 0) is only *weakly* Pareto optimal, whereas
        # the eps=0 subproblem approaches the same f2 level from the
        # infeasible side, where the constraint gradient is informative, and
        # finds the CHEAPEST f2-optimal solution.
        eps_grid = np.linspace(1.0, 0.0, n_points)

        sub = _EpsSubproblem(
            opt,
            obj_index=0,
            delta=delta,
        )
        sub.set_normalization(ideal2, nadir2)

        warm = None
        # The composed objective exists only when execution was selected on
        # Simulator via execution_mode="functional".
        if use_prepass and opt._functional_objective is not None:
            LOGGER.task("Pareto sweep: batched prepass (%d copies)" % n_points)
            try:
                warm = batched_prepass(
                    opt,
                    eps_grid,
                    x_a1,
                    x_a2,
                    ideal2,
                    range2,
                    delta,
                    bounds_obj,
                    **(prepass_options or {}),
                )
            except Exception as exc:
                LOGGER.warning(
                    "Pareto prepass failed (%s); falling back to sequential "
                    "warm starts.",
                    exc,
                )

        LOGGER.task("Pareto sweep: %d epsilon solves" % n_points)
        x_prev = x_a1
        for i, eps in enumerate(eps_grid):
            x_init = warm[i] if warm is not None else x_prev
            res = _solve_subproblem(
                sub, x_init, bounds_obj, solver_options, method_name, eps=eps
            )
            x_prev = res.x
            v = sub.values(res.x)
            rows.append(
                (
                    float(eps),
                    v,
                    res.x,
                    bool(res.success),
                    int(res.nit) if res.nit is not None else 0,
                )
            )
            LOGGER.iter(
                "eps=%.3f | f1=%.6f f2=%.6f | feas=%.2e | success=%s",
                eps,
                v["objs"][0],
                v["objs"][1],
                max(0.0, v["f2n"] - eps),
                res.success,
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
        len(rows),
        int(mask.sum()),
    )
    stats = sub.stats if abs(range2) >= 1e-9 else {}
    anchor_bundle_captured = [
        bool(stats_a1.get("bundle_captured", False)),
        bool(stats_a2.get("bundle_captured", False)),
    ]
    anchor_bundle_replays = [
        int(stats_a1.get("bundle_replays", 0)),
        int(stats_a2.get("bundle_replays", 0)),
    ]
    anchor_hessian_captured = [
        bool(stats_a1.get("hessian_captured", False)),
        bool(stats_a2.get("hessian_captured", False)),
    ]
    anchor_hessian_replays = [
        int(stats_a1.get("hessian_replays", 0)),
        int(stats_a2.get("hessian_replays", 0)),
    ]
    capture_meta = {
        "requested": capture_requested,
        "enabled": capture_derivatives,
        "captured_bundle": bool(
            stats.get("bundle_captured", False) or any(anchor_bundle_captured)
        ),
        "bundle_calls": int(
            stats.get("bundle_calls", 0)
            + stats_a1.get("bundle_calls", 0)
            + stats_a2.get("bundle_calls", 0)
        ),
        "bundle_replays": int(
            stats.get("bundle_replays", 0) + sum(anchor_bundle_replays)
        ),
        "sweep_captured_bundle": bool(stats.get("bundle_captured", False)),
        "sweep_bundle_calls": int(stats.get("bundle_calls", 0)),
        "sweep_bundle_replays": int(stats.get("bundle_replays", 0)),
        "anchor_bundle_captured": anchor_bundle_captured,
        "anchor_bundle_replays": anchor_bundle_replays,
        "scope": "device first-order bundle; host solver callbacks",
    }
    hessian_meta = {
        "kind": (
            "exact_dense_lagrangian" if method_name.lower() == "ipopt" else "not_used"
        ),
        "requested": capture_hessian,
        "enabled": capture_hessian,
        "direct_cuda_graph": bool(
            stats.get("hessian_captured", False) or any(anchor_hessian_captured)
        ),
        "calls": int(
            stats.get("hessian_calls", 0)
            + stats_a1.get("hessian_calls", 0)
            + stats_a2.get("hessian_calls", 0)
        ),
        "replays": int(stats.get("hessian_replays", 0) + sum(anchor_hessian_replays)),
        "sweep_direct_cuda_graph": bool(stats.get("hessian_captured", False)),
        "sweep_calls": int(stats.get("hessian_calls", 0)),
        "sweep_replays": int(stats.get("hessian_replays", 0)),
        "anchor_direct_cuda_graph": anchor_hessian_captured,
        "anchor_replays": anchor_hessian_replays,
    }
    result = ParetoResult(
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
        method=tuple(method),
        capture={
            **capture_meta,
            "batched_prepass": dict(opt._pareto_prepass_stats),
        },
        hessian=hessian_meta,
        _optimizer=opt,
    )
    if abs(range2) >= 1e-9:
        sub.close()
    return result
