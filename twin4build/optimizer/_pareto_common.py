"""Shared Pareto result and prepass utilities."""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import matplotlib.pyplot as plt

import twin4build.utils.types as tps
from twin4build.utils._cuda_graph import CudaGraphCallable
from twin4build.utils.logger import LOGGER


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
    method: tuple = ()
    capture: dict = field(default_factory=dict)
    hessian: dict = field(default_factory=dict)
    boundary_states: Optional[np.ndarray] = None
    max_defect: Optional[np.ndarray] = None
    rollout_parity: list = field(default_factory=list)
    callback_shapes: dict = field(default_factory=dict)
    _optimizer: object = field(default=None, repr=False)

    def apply(self, i: int) -> None:
        """Write Pareto point ``i``'s decision trajectories into the model and
        re-simulate, so component histories hold that solution."""
        self._optimizer.apply_solution(self.theta[i])

    def plot(self, ax=None):
        """Scatter the front in physical objective space (Pareto-optimal
        points connected, dominated points crossed out)."""
        if ax is None:
            _, ax = plt.subplots(figsize=(6, 5))
        m = self.pareto_mask
        order = np.argsort(self.f2[m])
        ax.plot(self.f2[m][order], self.f1[m][order], "o-", label="Pareto front")
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
    starts for the exact SLSQP or IPOPT polish.
    """
    fast = opt._functional_objective
    dev, dt = opt._device, tps.float_dtype()
    N, n = len(eps_grid), len(x_a1)
    mu = float(opt._constraint_penalty if mu is None else mu)

    # Spread the initial copies along the anchor-to-anchor segment: each copy
    # starts near where its epsilon slice will land.
    t = torch.linspace(0.0, 1.0, N, dtype=dt, device=dev).unsqueeze(1)
    a1 = torch.tensor(x_a1, dtype=dt, device=dev)
    a2 = torch.tensor(x_a2, dtype=dt, device=dev)
    theta = ((1.0 - t) * a1 + t * a2).detach().requires_grad_(True)
    eps_t = torch.tensor(np.asarray(eps_grid, dtype=np.float64), dtype=dt, device=dev)

    if bounds_obj is not None:
        lb = torch.tensor(np.asarray(bounds_obj.lb), dtype=dt, device=dev)
        ub = torch.tensor(np.asarray(bounds_obj.ub), dtype=dt, device=dev)
    else:
        lb = ub = None

    def per_copy(th_i, eps_i):
        p = fast.parts(th_i)
        pen = fast.penalty(p)
        f2n = (p.objs[1] - ideal2) / range2
        return p.objs[0] + pen + delta * f2n + mu * torch.relu(f2n - eps_i) ** 2

    use_vmap = True
    graph = None
    capture_requested = (
        opt.simulator.execution_backend == "cuda_graph" and dev.type == "cuda"
    )
    prepass_stats = {
        "requested": capture_requested,
        "enabled": False,
        "captured": False,
        "replays": 0,
        "fixed_batch_shape": (N, n),
        "fallback_reason": None,
        "scope": "fixed-batch losses and gradients; Adam and stopping remain eager",
    }
    opt._pareto_prepass_stats = prepass_stats

    def captured_bundle(theta_batch, epsilon_batch):
        z = theta_batch.detach().clone().requires_grad_(True)
        losses_ = torch.func.vmap(per_copy)(z, epsilon_batch)
        (gradient_,) = torch.autograd.grad(losses_.sum(), z)
        return torch.cat((losses_.detach(), gradient_.detach().reshape(-1)))

    if capture_requested:
        # Prove the full fixed-batch vmap works eagerly before entering stream
        # capture. If it does not, retain the safe loop implementation and
        # record why capture was intentionally not attempted.
        try:
            captured_bundle(theta, eps_t)
            graph = CudaGraphCallable(captured_bundle)
            prepass_stats["enabled"] = True
        except Exception as exc:
            prepass_stats["fallback_reason"] = f"{type(exc).__name__}: {exc}"
            use_vmap = False
            if graph is not None:
                graph.close()
                graph = None

    optimz = torch.optim.Adam([theta], lr=lr)
    best, stall = float("inf"), 0
    try:
        for it in range(max_iter):
            optimz.zero_grad()
            if graph is not None:
                packed = graph(theta, eps_t).clone()
                if prepass_stats["captured"]:
                    prepass_stats["replays"] += 1
                else:
                    prepass_stats["captured"] = True
                losses = packed[:N]
                theta.grad = packed[N:].reshape_as(theta)
                total = losses.sum()
            else:
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
                    use_vmap = False
                    prepass_stats["fallback_reason"] = (
                        f"vmap unavailable: {type(exc).__name__}: {exc}"
                    )
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
    finally:
        if graph is not None:
            graph.close()

    LOGGER.config(
        "Pareto prepass: %d iteration(s), batched=%s, total loss %.6f",
        it + 1,
        use_vmap,
        float(total),
    )
    return theta.detach().cpu().numpy().astype(np.float64)
