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
