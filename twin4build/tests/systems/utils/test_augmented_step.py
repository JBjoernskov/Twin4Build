"""The augmented one-step map (``zoh_expm="aug"``) and large exponentials under ``torch.compile``.

``x_next = exp(A T) x + phi1(A T) B T u`` read off ONE exponential of the
``(n + 1)`` matrix ``[[A T, B T u], [0, 0]]`` is the same exact ZOH step as the
``phi1`` pair (twice the products, then an ``n x m`` product), to rounding,
with the same gradients, under ``vmap`` and on a fused block's own matrices.
Matrices larger than the ``small_matmul_max`` setting keep ``bmm`` products
while ``torch.compile`` traces (a broadcast multiply and sum is ``n`` times the
memory traffic).
"""

# Standard library imports
import unittest
from unittest import mock

# Third party imports
import torch

# Local application imports
import twin4build as tb
from twin4build.systems.utils import discrete_statespace_system as dss
from twin4build.systems.utils.discrete_statespace_system import bilinear_onestep
from twin4build.tests.simulator.test_sparse_bilinear import fused_model, random_sparse, simulate

tb._IS_TESTING = True


def stable_system(n_c, n, m, scale, gen, dtype=torch.float64):
    """A diagonally dominant (stable) A, inputs of a physical size, bilinear terms on the diagonal (flows) and off it."""
    A = torch.rand(n_c, n, n, generator=gen, dtype=dtype) * scale
    A = A - torch.diag_embed(A.sum(-1) + scale * torch.rand(n_c, n, generator=gen, dtype=dtype))
    B = torch.rand(n_c, n, m, generator=gen, dtype=dtype) * scale
    E = -torch.rand(n_c, m, n, n, generator=gen, dtype=dtype) * scale * torch.eye(n, dtype=dtype)  # flow x own state
    F = torch.rand(n_c, m, n, m, generator=gen, dtype=dtype) * scale * 1e-2
    C = torch.randn(n_c, 2, n, generator=gen, dtype=dtype)
    D = torch.randn(n_c, 2, m, generator=gen, dtype=dtype)
    return A, B, C, D, E, F


def step(mode, *args, **kwargs):
    with tb.settings.override(zoh_expm=mode):
        return bilinear_onestep(*args, **kwargs, transform_mode=True)


def block_form(A, B, C, D, E, F, x, u, T):
    """The reference: ``torch.matrix_exp`` of the (n + m) block."""
    return bilinear_onestep(A, B, C, D, E, F, x, u, T, transform_mode=False)


class TestAugmentedStep(unittest.TestCase):
    T = 600.0

    def _cases(self):
        gen = torch.Generator().manual_seed(11)
        # (n_c, n, m, scale of A entries in 1/s): a zone, a cluster, a stiff storey-sized block, a radiator-like stiff one
        for n_c, n, m, scale in ((3, 6, 11, 1e-4), (2, 17, 9, 1e-3), (1, 40, 25, 2e-2), (2, 3, 4, 50.0)):
            A, B, C, D, E, F = stable_system(n_c, n, m, scale, gen)
            for lead in ((), (4,)):  # one window, four windows
                x = torch.randn(*lead, n_c, n, generator=gen, dtype=torch.float64) * 5 + 20
                u = torch.rand(*lead, n_c, m, generator=gen, dtype=torch.float64) * 3
                yield (A, B, C, D, E, F, x, u)

    def test_the_setting_selects_the_augmented_step(self):
        args = next(self._cases())
        with mock.patch.object(dss, "_onestep_augmented", wraps=dss._onestep_augmented) as recorded:
            step("phi1", *args, self.T)
            self.assertFalse(recorded.called)
            step("aug", *args, self.T)
            self.assertTrue(recorded.called)
            with tb.settings.override(zoh_expm="aug"):  # outside transform mode: torch.matrix_exp
                bilinear_onestep(*args, self.T, transform_mode=False)
            self.assertEqual(recorded.call_count, 1)

    def test_the_step_equals_the_block_form(self):
        # Against torch.matrix_exp of the (n + m) block: the augmented step to ~1e-15 of the state's scale on every
        # case.  The phi1 pair agrees as closely until the matrix is stiff (|A T| ~ 1e5: ~1e-6, its recurrence
        # rounding through the squarings once exp(A T / 2^k) vanishes), so it is checked loosely.
        for args in self._cases():
            x_a, y_a = step("aug", *args, self.T)
            x_p, y_p = step("phi1", *args, self.T)
            x_b, y_b = block_form(*args, self.T)
            for a, ref in ((x_a, x_b), (y_a, y_b)):
                self.assertLessEqual(float((a - ref).abs().max()), 1e-13 * float(ref.abs().max()))
            for p, ref in ((x_p, x_b), (y_p, y_b)):
                self.assertLessEqual(float((p - ref).abs().max()), 1e-5 * float(ref.abs().max()))

    def test_sparse_bilinear_terms(self):
        gen = torch.Generator().manual_seed(3)
        n_c, n, m = 2, 9, 7
        A, B, C, D, _, _ = stable_system(n_c, n, m, 1e-3, gen)
        Es, Ed = random_sparse(n_c, m, n, n, 12)
        Fs, Fd = random_sparse(n_c, m, n, m, 15)
        x = torch.randn(n_c, n, dtype=torch.float64)
        u = torch.rand(n_c, m, dtype=torch.float64)
        x_s, y_s = step("aug", A, B, C, D, Es, Fs, x, u, self.T)
        x_d, y_d = step("phi1", A, B, C, D, Ed, Fd, x, u, self.T)
        torch.testing.assert_close(x_s, x_d, rtol=1e-11, atol=1e-11)
        torch.testing.assert_close(y_s, y_d, rtol=1e-11, atol=1e-11)

    def test_gradients_equal_the_block_forms(self):
        # Against torch.matrix_exp's own gradient.  (The phi1 pair is off by ~1e-7 of the gradient's scale on the
        # 40-state case, its recurrence rounding through the squarings; the augmented form stays at ~1e-15.)
        for args in list(self._cases())[::3]:
            grads = {}
            for mode in ("aug", "ref"):
                leaves = [a.detach().clone().requires_grad_(True) for a in args]
                x1, y1 = block_form(*leaves, self.T) if mode == "ref" else step(mode, *leaves, self.T)
                loss = x1.square().sum() + (y1 * torch.linspace(-1, 1, y1.shape[-1], dtype=y1.dtype)).sum()
                grads[mode] = torch.autograd.grad(loss, leaves, allow_unused=True)
            for g_a, g_r in zip(grads["aug"], grads["ref"]):
                scale = float(g_r.abs().max())
                self.assertLessEqual(float((g_a - g_r).abs().max()), 1e-11 * max(scale, 1.0))

    def test_under_vmap(self):
        A, B, C, D, E, F, x, u = next(c for c in self._cases() if c[6].dim() == 3)  # four windows
        f = torch.func.vmap(lambda xx, uu: step("aug", A, B, C, D, E, F, xx, uu, self.T))
        x_v, y_v = f(x, u)
        x_l, y_l = step("aug", A, B, C, D, E, F, x, u, self.T)
        torch.testing.assert_close(x_v, x_l, rtol=1e-13, atol=1e-12)
        torch.testing.assert_close(y_v, y_l, rtol=1e-13, atol=1e-12)

    def test_float32(self):
        A, B, C, D, E, F, x, u = [a.float() for a in next(self._cases())]
        x_a, _ = step("aug", A, B, C, D, E, F, x, u, self.T)
        x_p, _ = step("phi1", A, B, C, D, E, F, x, u, self.T)
        torch.testing.assert_close(x_a, x_p, rtol=1e-5, atol=1e-4)

    def test_a_fused_blocks_own_matrices(self):
        model, _ = fused_model("augmented_step_fused", pairs=2)
        simulate(model, "object")
        blocks = list(model.simulation_model._fused_components.values())
        self.assertTrue(blocks)
        gen = torch.Generator().manual_seed(5)
        for blk in blocks:
            A, B, C, D, E, F = blk._assemble(blk._forward_params())
            n, m = A.shape[-1], B.shape[-1]
            x = 20 + torch.randn(A.shape[0], n, generator=gen, dtype=A.dtype)
            u = torch.rand(A.shape[0], m, generator=gen, dtype=A.dtype) * 2
            x_a, y_a = step("aug", A, B, C, D, E, F, x, u, 600.0)
            x_p, y_p = step("phi1", A, B, C, D, E, F, x, u, 600.0)
            torch.testing.assert_close(x_a, x_p, rtol=1e-11, atol=1e-9)
            torch.testing.assert_close(y_a, y_p, rtol=1e-11, atol=1e-9)


def _traced_products(n, device="cpu", backend="inductor"):
    """``_expm_ss`` of ``n x n`` matrices compiled: (compiled, eager, sizes of the products written for fusion)."""
    calls = []
    real = dss._small_matmul

    def counting(a, b):
        calls.append(a.shape[-1])
        return real(a, b)

    gen = torch.Generator(device=device).manual_seed(2)
    M = -torch.rand(3, n, n, generator=gen, device=device, dtype=torch.float32) * 0.3
    M = M - torch.diag_embed(M.abs().sum(-1))
    with mock.patch.object(dss, "_small_matmul", counting):
        try:
            torch._dynamo.reset()
            out = torch.compile(dss._expm_ss, fullgraph=True, dynamic=False, backend=backend)(M)
        finally:
            torch._dynamo.reset()
    return out, dss._expm_ss(M), calls


class TestSmallMatmulMax(unittest.TestCase):
    """While ``torch.compile`` traces, a matrix up to ``small_matmul_max`` is
    exponentiated with broadcast products (``_small_matmul``), a larger one
    with ``bmm``; the setting holds from its next trace."""

    def test_the_setting_moves_the_threshold(self):
        for limit, n, fused in ((16, 6, True), (16, 24, False), (32, 24, True), (4, 6, False)):
            with self.subTest(limit=limit, n=n), tb.settings.override(small_matmul_max=limit):
                out, eager, calls = _traced_products(n, backend="eager")
                self.assertEqual(bool(calls), fused)
                torch.testing.assert_close(out, eager, rtol=1e-5, atol=1e-6)


@unittest.skipUnless(torch.cuda.is_available(), "Inductor's compiled path is checked on the GPU")
class TestLargeExponentialUnderCompile(unittest.TestCase):
    def test_small_matrices_fuse_and_large_ones_keep_bmm(self):
        for n, fused in ((6, True), (tb.settings.value("small_matmul_max") + 8, False)):
            out, eager, calls = _traced_products(n, device="cuda")
            self.assertEqual(bool(calls), fused, f"n={n}")
            torch.testing.assert_close(out, eager, rtol=2e-5, atol=2e-6)


if __name__ == "__main__":
    unittest.main()
