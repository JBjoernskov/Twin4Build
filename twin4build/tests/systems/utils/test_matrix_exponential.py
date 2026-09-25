import unittest

import torch

from twin4build.systems.utils.discrete_statespace_system import _expm_ss


class TestScalingAndSquaringMatrixExponential(unittest.TestCase):
    @staticmethod
    def rc_block(scale):
        return torch.tensor(
            [
                [-scale, 0.2 * scale, 0.1],
                [0.0, -0.5 * scale, 0.3],
                [0.0, 0.0, 0.0],
            ],
            dtype=torch.float64,
        )

    def test_matches_native_across_fixed_scaling_range(self):
        for scale in (0.02, 0.27, 2.4, 27.0, 3e3, 2.6e5):
            matrix = self.rc_block(scale)
            actual = _expm_ss(matrix)
            expected = torch.matrix_exp(matrix)
            relative_error = (
                (actual - expected).abs().max()
                / expected.abs().max().clamp_min(torch.finfo(expected.dtype).tiny)
            )
            self.assertLess(
                float(relative_error),
                2e-12,
                f"relative error at scale {scale:g}",
            )

    def test_first_and_second_derivatives_match_native(self):
        matrix = self.rc_block(0.27)
        direction = torch.tensor(
            [
                [0.3, -0.2, 0.1],
                [0.0, -0.4, 0.2],
                [0.0, 0.0, 0.0],
            ],
            dtype=torch.float64,
        )

        _, actual_jvp = torch.func.jvp(_expm_ss, (matrix,), (direction,))
        _, expected_jvp = torch.func.jvp(
            torch.matrix_exp, (matrix,), (direction,)
        )
        torch.testing.assert_close(
            actual_jvp, expected_jvp, rtol=2e-10, atol=2e-12
        )

        actual_hessian = torch.func.hessian(
            lambda value: _expm_ss(value).square().sum()
        )(matrix)
        expected_hessian = torch.func.hessian(
            lambda value: torch.matrix_exp(value).square().sum()
        )(matrix)
        torch.testing.assert_close(
            actual_hessian, expected_hessian, rtol=2e-9, atol=2e-11
        )


if __name__ == "__main__":
    unittest.main()


class TestPhi1Pair(unittest.TestCase):
    """``_expm_phi1_ss`` against the block form and ``torch.matrix_exp``."""

    def _cases(self):
        gen = torch.Generator().manual_seed(7)
        for n, m, scale in ((6, 11, 1.0), (3, 2, 50.0), (17, 4, 1e3), (1, 1, 1e-3)):
            A = -torch.rand(2, n, n, generator=gen, dtype=torch.float64) * scale
            A = A - torch.diag_embed(A.abs().sum(-1))  # diagonally dominant, stable
            B = torch.randn(2, n, m, generator=gen, dtype=torch.float64) * scale
            yield A, B

    def test_matches_block_form(self):
        from twin4build.systems.utils.discrete_statespace_system import _expm_phi1_ss

        for A, B in self._cases():
            n, m = A.shape[-1], B.shape[-1]
            top = torch.cat([A, B], dim=-1)
            M = torch.nn.functional.pad(top, (0, 0, 0, m))
            expM = torch.matrix_exp(M)
            expA, phi = _expm_phi1_ss(A)
            torch.testing.assert_close(expA, expM[..., :n, :n], rtol=1e-9, atol=1e-11)
            torch.testing.assert_close(phi @ B, expM[..., :n, n:], rtol=1e-9, atol=1e-11)

    def test_gradients_match_block_form(self):
        from twin4build.systems.utils.discrete_statespace_system import _expm_phi1_ss

        A, B = next(self._cases())
        A = A.clone().requires_grad_(True)
        n, m = A.shape[-1], B.shape[-1]
        expA, phi = _expm_phi1_ss(A)
        loss = expA.square().sum() + (phi @ B).square().sum()
        (gA,) = torch.autograd.grad(loss, A)
        A2 = A.detach().clone().requires_grad_(True)
        M = torch.nn.functional.pad(torch.cat([A2, B], dim=-1), (0, 0, 0, m))
        expM = torch.matrix_exp(M)
        loss2 = expM[..., :n, :n].square().sum() + expM[..., :n, n:].square().sum()
        (gA2,) = torch.autograd.grad(loss2, A2)
        torch.testing.assert_close(gA, gA2, rtol=1e-7, atol=1e-9)

    def test_vmap_batches(self):
        from twin4build.systems.utils.discrete_statespace_system import _expm_phi1_ss

        A, _ = next(self._cases())
        e1, p1 = torch.func.vmap(_expm_phi1_ss)(A)
        e2, p2 = _expm_phi1_ss(A)
        torch.testing.assert_close(e1, e2)
        torch.testing.assert_close(p1, p2)
