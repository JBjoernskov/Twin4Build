"""The matrix exponential's schedule (``_expm_ss``): balanced, a degree-8
series of four products, doublings from the ``expm_squarings`` setting, NaN
beyond the series' range.

A balance equation's matrix is badly scaled rather than stiff: its 1-norm is
far above its spectral radius.  Balanced first, it needs fewer doublings for
the same accuracy; a matrix the doublings do not bring into the series' range
is refused (NaN) instead of exponentiated wrongly.
"""
import unittest

import torch

import twin4build as tb
from twin4build.systems.utils import discrete_statespace_system as dss

tb._IS_TESTING = True


def stable(n, batch, gen, scale=1.0):
    """Diagonally dominant matrices with a negative diagonal, ``(batch, n, n)`` in float64."""
    M = torch.rand(batch, n, n, generator=gen, dtype=torch.float64) * scale
    return M - torch.diag_embed(M.sum(-1) * 1.5)


def badly_scaled(M, decades, gen):
    """``D M D^-1`` for a diagonal ``D`` spread over ``decades`` powers of ten: the same eigenvalues, a far larger norm."""
    d = 10.0 ** ((torch.rand(M.shape[:-1], generator=gen, dtype=M.dtype) - 0.5) * decades)
    return M * d.unsqueeze(-1) / d.unsqueeze(-2)


def norm1(M):
    return M.abs().sum(dim=-2).amax(dim=-1)


def phi1_reference(X):
    """``phi1(X)`` read off ``exp([[X, I], [0, 0]])``."""
    n = X.shape[-1]
    top = torch.cat([X, torch.eye(n, dtype=X.dtype).expand_as(X)], dim=-1)
    return torch.matrix_exp(torch.cat([top, torch.zeros_like(top)], dim=-2))[..., :n, n:]


class TestSeries(unittest.TestCase):
    def test_the_exponential_and_the_pair_are_matrix_exp(self):
        gen = torch.Generator().manual_seed(11)
        for n in (1, 3, 12, 40):
            with self.subTest(n=n):
                M = stable(n, 4, gen, scale=3.0)
                torch.testing.assert_close(dss._expm_ss(M), torch.matrix_exp(M), rtol=1e-11, atol=1e-12)
                expM, phi = dss._expm_phi1_ss(M)
                torch.testing.assert_close(expM, torch.matrix_exp(M), rtol=1e-11, atol=1e-12)
                torch.testing.assert_close(phi, phi1_reference(M), rtol=1e-11, atol=1e-12)
                torch.testing.assert_close(dss._expm_ss_fused(M), dss._expm_ss(M), rtol=1e-12, atol=1e-13)

    def test_the_gradient_is_matrix_exps(self):
        gen = torch.Generator().manual_seed(12)
        M = badly_scaled(stable(4, 1, gen), 3, gen)[0]
        weight = torch.rand(4, 4, generator=gen, dtype=torch.float64)

        def gradient(exp):
            return torch.func.grad(lambda m: (exp(m) * weight).sum())(M)

        # the balancing carries no gradient: the exponential is the same for any diagonal
        torch.testing.assert_close(gradient(dss._expm_ss), gradient(torch.matrix_exp), rtol=1e-9, atol=1e-11)


class TestBalancing(unittest.TestCase):
    def setUp(self):
        gen = torch.Generator().manual_seed(13)
        self.well = stable(12, 3, gen, scale=2.0)
        self.bad = badly_scaled(self.well, 5, gen)

    def test_the_balanced_norm_is_near_the_well_scaled_one(self):
        d = dss._balance(self.bad)
        balanced = self.bad * d.unsqueeze(-2) / d.unsqueeze(-1)
        self.assertTrue(bool((norm1(self.bad) > 100 * norm1(self.well)).all()), "the test's matrices are badly scaled")
        self.assertTrue(bool((norm1(balanced) < 4 * norm1(self.well)).all()))
        # powers of two: scaling by them is exact
        torch.testing.assert_close(torch.log2(d), torch.round(torch.log2(d)), rtol=0, atol=0)

    def test_a_balanced_matrix_is_left_alone(self):
        symmetric = self.well + self.well.transpose(-1, -2)
        torch.testing.assert_close(dss._balance(symmetric), torch.ones(3, 12, dtype=torch.float64), rtol=0, atol=0)

    def test_doublings_sized_for_the_balanced_norm_are_enough(self):
        # doublings that bring the balanced matrix into the series' range and leave the raw one far outside it
        doublings = int(torch.ceil(torch.log2(norm1(self.well).max() * 4 / 0.5)))
        self.assertTrue(bool((norm1(self.bad) / 2.0**doublings > dss._EXPM_REFUSE_NORM).all()))
        with tb.settings.override(expm_squarings=doublings):
            out = dss._expm_ss(self.bad)
        self.assertTrue(bool(torch.isfinite(out).all()))
        reference = torch.matrix_exp(self.bad)
        self.assertLess(float(((out - reference).abs() / reference.abs().amax(dim=(-2, -1), keepdim=True)).max()), 1e-9)


class TestDoublings(unittest.TestCase):
    def test_the_setting_is_read_at_the_call(self):
        gen = torch.Generator().manual_seed(14)
        M = stable(6, 2, gen, scale=0.01)  # a norm of about 0.1: the series alone is exact to rounding
        self.assertEqual(tb.settings.value("expm_squarings"), 18)
        for doublings in (0, 4, 18):
            with self.subTest(doublings=doublings), tb.settings.override(expm_squarings=doublings):
                torch.testing.assert_close(dss._expm_ss(M), torch.matrix_exp(M), rtol=1e-10, atol=1e-12)
        torch.testing.assert_close(dss._expm_ss(M, 3), torch.matrix_exp(M), rtol=1e-10, atol=1e-12)

    def test_a_matrix_beyond_the_series_range_is_nan_and_only_that_one(self):
        gen = torch.Generator().manual_seed(15)
        small = stable(5, 1, gen, scale=0.05)
        large = stable(5, 1, gen, scale=40.0)
        both = torch.cat([small, large])
        with tb.settings.override(expm_squarings=2):
            expM, phi = dss._expm_phi1_ss(both)
        torch.testing.assert_close(expM[0], torch.matrix_exp(small[0]), rtol=1e-10, atol=1e-12)
        self.assertTrue(bool(torch.isfinite(phi[0]).all()))
        self.assertTrue(bool(torch.isnan(expM[1]).all()))
        self.assertTrue(bool(torch.isnan(phi[1]).all()))
        # with the default doublings the same matrix is in range
        torch.testing.assert_close(dss._expm_ss(large), torch.matrix_exp(large), rtol=1e-9, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
