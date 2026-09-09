"""Block trust-region step of the batched shooting solver (issue #142).

A synthetic separable least-squares objective stands in for the shooting
objective: it exposes the same bundles (``batched_column_loss``,
``batched_column_loss_and_grad``, ``batched_value_and_grad``, ``batched_loss``)
and a ``parameter_structure`` with known blocks, so the solver's structure
handling, per-block acceptance and robustness to one rough block are checked
without a model or a GPU.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from twin4build.estimator import _batched_solvers as bs


class SeparableQuadratic:
    """Blocks of parameters, each with its own residual columns.

    Block ``k`` owns parameters ``k*w:(k+1)*w`` and columns ``k*c:(k+1)*c``;
    column ``j`` of block ``k`` is ``(A_k x_k - b_k)[j]`` so the loss is a sum
    of independent quadratics.  ``rough`` adds a deterministic high-frequency
    ripple to one block's columns (a rough loss surface whose gradient is
    exact but unreliable at the step scale).
    """

    def __init__(self, n_blocks=4, width=3, cols=5, rough_block=None, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.n_blocks, self.width, self.cols = n_blocks, width, cols
        self.A = [torch.randn(cols, width, generator=g, dtype=torch.float64) for _ in range(n_blocks)]
        self.b = [torch.randn(cols, generator=g, dtype=torch.float64) for _ in range(n_blocks)]
        self.rough_block = rough_block
        # what the solver reads from a real objective: device, execution backend, dtype
        self._sd = torch.ones(cols * n_blocks, dtype=torch.float64)
        self.est = SimpleNamespace(
            _device=torch.device("cpu"), simulator=SimpleNamespace(execution_backend="eager")
        )

    def _columns(self, theta):  # theta (n_theta,) -> (n_cols_total,)
        out = []
        for k in range(self.n_blocks):
            xk = theta[k * self.width : (k + 1) * self.width]
            r = self.A[k] @ xk - self.b[k]
            if k == self.rough_block:
                r = r + 0.05 * torch.sin(1e7 * xk.sum())  # ripple with slope ~5e5 (cf. 1e9 on the real problem)
            out.append(r.square())
        return torch.cat(out)

    def batched_column_loss(self, theta_batch):
        return torch.stack([self._columns(th) for th in theta_batch])

    def batched_column_loss_and_grad(self, theta_batch):
        z = theta_batch.detach().clone().requires_grad_(True)
        cols = self.batched_column_loss(z)
        (grad,) = torch.autograd.grad(cols.sum(), z)
        return cols.detach(), grad.detach()

    def batched_loss(self, theta_batch):
        return self.batched_column_loss(theta_batch).sum(dim=1)

    def batched_value_and_grad(self, theta_batch):
        cols, grad = self.batched_column_loss_and_grad(theta_batch)
        return cols.sum(dim=1), grad

    def parameter_structure(self):
        theta_block = np.repeat(np.arange(self.n_blocks), self.width)
        column_block = np.repeat(np.arange(self.n_blocks), self.cols)
        return theta_block, column_block, self.n_blocks

    @property
    def n_theta(self):
        return self.n_blocks * self.width


def _solve(obj, method, maxiter=60, **opts):
    n = obj.n_theta
    x0 = np.full(n, 0.5)
    lb, ub = np.full(n, -3.0), np.full(n, 3.0)
    res = bs.solve_batched_multistart(obj, method, x0[None, :], lb, ub, {"maxiter": maxiter, "batch_size": 1, "gtol": 1e-9, "ftol": 1e-15, "patience": 20, **opts})
    return res


def test_structure_is_used_and_all_blocks_converge():
    obj = SeparableQuadratic()
    res = _solve(obj, "batched-tr")
    assert res.curvature_block_sizes == [3, 3, 3, 3]
    # each block reaches its own least-squares optimum
    x = torch.as_tensor(res.x, dtype=torch.float64)
    for k in range(obj.n_blocks):
        xk = x[k * 3 : (k + 1) * 3]
        opt = torch.linalg.lstsq(obj.A[k], obj.b[k]).solution
        torch.testing.assert_close(xk, opt, rtol=1e-5, atol=1e-5)


def test_single_block_reduces_to_unstructured_solve():
    obj = SeparableQuadratic()
    res = _solve(obj, "batched-tr", tr_blocks=None)
    assert res.curvature_block_sizes == [12]
    x = torch.as_tensor(res.x, dtype=torch.float64)
    for k in range(obj.n_blocks):
        opt = torch.linalg.lstsq(obj.A[k], obj.b[k]).solution
        torch.testing.assert_close(x[k * 3 : (k + 1) * 3], opt, rtol=1e-5, atol=1e-5)


def test_rough_block_does_not_stall_the_others():
    obj = SeparableQuadratic(rough_block=1)
    res = _solve(obj, "batched-tr", maxiter=80)
    x = torch.as_tensor(res.x, dtype=torch.float64)
    for k in (0, 2, 3):
        xk = x[k * 3 : (k + 1) * 3]
        opt = torch.linalg.lstsq(obj.A[k], obj.b[k]).solution
        torch.testing.assert_close(xk, opt, rtol=1e-5, atol=1e-5)
    # the rough block made progress from the start point even if it did not converge
    cols = obj.batched_column_loss(x[None, :])[0]
    start = obj.batched_column_loss(torch.full((1, obj.n_theta), 0.5, dtype=torch.float64))[0]
    assert cols[5:10].sum() < start[5:10].sum()


def test_joint_line_search_sqp_is_hurt_by_the_rough_block():
    """Reference behaviour the block step is meant to fix: with one joint
    step the healthy blocks end far from their optimum."""
    obj = SeparableQuadratic(rough_block=1)
    tr = _solve(obj, "batched-tr", maxiter=40)
    sqp = _solve(obj, "batched-sqp", maxiter=40)
    x_tr = torch.as_tensor(tr.x, dtype=torch.float64)
    x_sqp = torch.as_tensor(sqp.x, dtype=torch.float64)
    err = lambda x: sum(
        float((x[k * 3 : (k + 1) * 3] - torch.linalg.lstsq(obj.A[k], obj.b[k]).solution).norm())
        for k in (0, 2, 3)
    )
    assert err(x_tr) < 1e-4
    assert err(x_tr) < err(x_sqp)


def test_explicit_blocks_and_validation_fallback():
    obj = SeparableQuadratic()
    # explicit blocks that cut across the true structure are merged into unions
    # of structure components (here: everything), never accepted per block
    wrong = [[0, 3, 6, 9], [1, 4, 7, 10], [2, 5, 8, 11]]
    res = _solve(obj, "batched-tr", tr_blocks=wrong)
    assert res.curvature_block_sizes == [12]
    x = torch.as_tensor(res.x, dtype=torch.float64)
    for k in range(obj.n_blocks):
        opt = torch.linalg.lstsq(obj.A[k], obj.b[k]).solution
        torch.testing.assert_close(x[k * 3 : (k + 1) * 3], opt, rtol=1e-5, atol=1e-5)


def test_unknown_option_is_rejected_by_the_solver_entry():
    obj = SeparableQuadratic()
    with pytest.raises(TypeError):
        bs._solve_block_trust_region(
            bs.BatchedObjectiveEvaluator(obj),
            torch.zeros(1, obj.n_theta, dtype=torch.float64),
            torch.full((obj.n_theta,), -3.0, dtype=torch.float64),
            torch.full((obj.n_theta,), 3.0, dtype=torch.float64),
            not_an_option=1,
        )
