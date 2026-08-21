"""Derivative parity for the collocation matrix-exponential backends."""

from contextlib import contextmanager
import os
import unittest

import torch
from torch.func import hessian, jacrev, vmap

from twin4build.systems.utils.discrete_statespace_system import (
    _discretize_onestep,
    _expm_ss,
    _matrix_exp_dispatch,
)


@contextmanager
def _matrix_exp_mode(mode):
    previous = os.environ.get("TWIN4BUILD_MATRIX_EXP")
    os.environ["TWIN4BUILD_MATRIX_EXP"] = mode
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("TWIN4BUILD_MATRIX_EXP", None)
        else:
            os.environ["TWIN4BUILD_MATRIX_EXP"] = previous


def _matrices():
    A = torch.tensor(
        [[[-1.2e-3, 4.0e-4], [2.0e-4, -8.0e-4]]],
        dtype=torch.float64,
    )
    B = torch.tensor(
        [[[2.0e-4, 1.0e-5], [3.0e-5, 1.5e-4]]],
        dtype=torch.float64,
    )
    E = torch.zeros((1, 2, 2, 2), dtype=torch.float64)
    E[0, 0, 0, 0] = -2.0e-4
    E[0, 0, 1, 0] = 1.0e-4
    E[0, 1, 1, 1] = -1.0e-4
    return A, B, E


def _response(u):
    A, B, E = _matrices()
    Ad, Bd = _discretize_onestep(A, B, E, None, u, 1200.0)
    return torch.cat([Ad.flatten(), Bd.flatten()])


class TestMatrixExpDispatch(unittest.TestCase):
    def test_vmap_value_parity(self):
        inputs = torch.tensor(
            [[[0.0, 0.0]], [[0.3, -0.2]], [[1.0, 0.5]]],
            dtype=torch.float64,
        )
        with _matrix_exp_mode("ss"):
            expected = vmap(_response)(inputs)
        with _matrix_exp_mode("native"):
            actual = vmap(_response)(inputs)
        torch.testing.assert_close(actual, expected, rtol=2e-9, atol=2e-10)

    def test_jacobian_and_hessian_parity(self):
        u = torch.tensor([[0.35, -0.1]], dtype=torch.float64)

        def scalar(value):
            response = _response(value)
            return (response.square()).sum()

        with _matrix_exp_mode("ss"):
            jac_expected = jacrev(_response)(u)
            hess_expected = hessian(scalar)(u)
        with _matrix_exp_mode("native"):
            jac_actual = jacrev(_response)(u)
            hess_actual = hessian(scalar)(u)

        torch.testing.assert_close(
            jac_actual, jac_expected, rtol=2e-8, atol=2e-9
        )
        torch.testing.assert_close(
            hess_actual, hess_expected, rtol=2e-7, atol=2e-8
        )

    def test_vmap_of_hessian_parity(self):
        inputs = torch.tensor(
            [[[0.1, -0.2]], [[0.7, 0.4]]],
            dtype=torch.float64,
        )
        weights = torch.tensor(
            [[1.0] * 8, [0.5] * 8],
            dtype=torch.float64,
        )

        def contracted(u, weight):
            return (_response(u) * weight).sum()

        hess_fn = hessian(contracted, argnums=0)

        def evaluate(mode):
            with _matrix_exp_mode(mode):
                return vmap(lambda u, w: hess_fn(u, w))(inputs, weights)

        expected = evaluate("ss")
        actual = evaluate("native")
        torch.testing.assert_close(
            actual, expected, rtol=2e-7, atol=2e-8
        )

    def test_stiff_matrix_stays_finite_and_matches_native(self):
        M = torch.tensor(
            [
                [-3.0e5, 0.0, 2.0e5],
                [0.0, -1.5e5, 7.5e4],
                [0.0, 0.0, 0.0],
            ],
            dtype=torch.float64,
        )
        expected = torch.linalg.matrix_exp(M)
        actual = _expm_ss(M)
        self.assertTrue(torch.isfinite(expected).all())
        self.assertTrue(torch.isfinite(actual).all())
        torch.testing.assert_close(actual, expected, rtol=2e-8, atol=2e-9)

    def test_invalid_mode_fails_inside_transform(self):
        M = torch.eye(2, dtype=torch.float64)
        with _matrix_exp_mode("invalid"):
            with self.assertRaisesRegex(ValueError, "must be 'ss' or 'native'"):
                jacrev(lambda scale: _matrix_exp_dispatch(M * scale))(
                    torch.tensor(1.0, dtype=torch.float64)
                )


if __name__ == "__main__":
    unittest.main()
