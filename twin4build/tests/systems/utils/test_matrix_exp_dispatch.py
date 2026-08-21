"""Derivative parity for the collocation matrix-exponential backends."""

from contextlib import contextmanager
import os
import unittest
import warnings

import torch
from torch.func import hessian, jacfwd, jacrev, vmap
from torch.profiler import ProfilerActivity, profile

from twin4build.systems.utils.discrete_statespace_system import (
    _discretize_onestep,
    _expm_ss,
    _matrix_exp_dispatch,
    _matrix_exp_native_vmap,
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

        torch.testing.assert_close(jac_actual, jac_expected, rtol=2e-8, atol=2e-9)
        torch.testing.assert_close(hess_actual, hess_expected, rtol=2e-7, atol=2e-8)

    def test_native_vmap_gradcheck_and_gradgradcheck(self):
        A = torch.tensor(
            [[-0.4, 0.2], [-0.1, -0.3]],
            dtype=torch.float64,
            requires_grad=True,
        )
        self.assertTrue(
            torch.autograd.gradcheck(
                _matrix_exp_native_vmap,
                (A,),
                check_forward_ad=True,
                check_backward_ad=True,
            )
        )
        self.assertTrue(torch.autograd.gradgradcheck(_matrix_exp_native_vmap, (A,)))

    def test_native_vmap_exact_nested_transform_parity(self):
        matrices = torch.tensor(
            [
                [[-0.4, 0.2], [-0.1, -0.3]],
                [[-0.7, 0.1], [0.3, -0.5]],
            ],
            dtype=torch.float64,
        )

        def scalar(A):
            return _matrix_exp_native_vmap(A).square().sum()

        def native_scalar(A):
            return torch.linalg.matrix_exp(A).square().sum()

        # This is the transform ordering used by the collocation Hessian:
        # segment vmap outside a forward-over-reverse Hessian.
        actual = vmap(jacfwd(jacrev(scalar)))(matrices)
        expected = torch.stack([jacfwd(jacrev(native_scalar))(A) for A in matrices])
        torch.testing.assert_close(actual, expected, rtol=2e-9, atol=2e-10)

    def test_native_vmap_handles_nonzero_and_nested_batch_dims(self):
        nonzero_dim_input = torch.randn(2, 3, 2, dtype=torch.float64) * 0.1
        actual_nonzero = vmap(_matrix_exp_native_vmap, in_dims=1, out_dims=1)(
            nonzero_dim_input
        )
        expected_nonzero = torch.stack(
            [
                torch.linalg.matrix_exp(nonzero_dim_input[:, i, :])
                for i in range(nonzero_dim_input.shape[1])
            ],
            dim=1,
        )
        torch.testing.assert_close(actual_nonzero, expected_nonzero)

        nested_input = torch.randn(2, 3, 2, 2, dtype=torch.float64) * 0.1
        actual_nested = vmap(vmap(_matrix_exp_native_vmap))(nested_input)
        expected_nested = torch.linalg.matrix_exp(nested_input)
        torch.testing.assert_close(actual_nested, expected_nested)

    def test_native_vmap_has_no_batching_fallback_warning(self):
        matrices = torch.randn(2, 2, 2, dtype=torch.float64) * 0.1

        def scalar(A):
            return _matrix_exp_native_vmap(A).square().sum()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            vmap(jacfwd(jacrev(scalar)))(matrices)

        fallback_messages = [
            str(item.message)
            for item in caught
            if "batching rule" in str(item.message)
            or "BatchedFallback" in str(item.message)
        ]
        self.assertEqual(fallback_messages, [])

    def test_native_vmap_uses_one_physical_matrix_exp_dispatch(self):
        matrices = torch.randn(7, 2, 2, dtype=torch.float64) * 0.1
        with profile(activities=[ProfilerActivity.CPU]) as prof:
            vmap(_matrix_exp_native_vmap)(matrices)
        matrix_exp_events = sum(
            event.count
            for event in prof.key_averages()
            if event.key == "aten::linalg_matrix_exp"
        )
        self.assertEqual(matrix_exp_events, 1)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_native_vmap_cuda_nested_transform_parity(self):
        matrices = torch.randn(2, 2, 2, device="cuda", dtype=torch.float64) * 0.1

        def scalar(A):
            return _matrix_exp_native_vmap(A).square().sum()

        def ss_scalar(A):
            return _expm_ss(A).square().sum()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            actual = vmap(jacfwd(jacrev(scalar)))(matrices)
        expected = vmap(jacfwd(jacrev(ss_scalar)))(matrices)
        torch.testing.assert_close(actual, expected, rtol=2e-7, atol=2e-8)
        self.assertFalse(any("batching rule" in str(item.message) for item in caught))

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
        for mode in ("native", "native_vmap"):
            actual = evaluate(mode)
            torch.testing.assert_close(actual, expected, rtol=2e-7, atol=2e-8)

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
        torch.testing.assert_close(
            _matrix_exp_native_vmap(M), expected, rtol=2e-12, atol=2e-13
        )

    def test_invalid_mode_fails_inside_transform(self):
        M = torch.eye(2, dtype=torch.float64)
        with _matrix_exp_mode("invalid"):
            with self.assertRaisesRegex(ValueError, "native_vmap"):
                jacrev(lambda scale: _matrix_exp_dispatch(M * scale))(
                    torch.tensor(1.0, dtype=torch.float64)
                )


if __name__ == "__main__":
    unittest.main()
