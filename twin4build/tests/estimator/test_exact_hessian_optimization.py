"""Regression tests for collocation exact-Hessian optimizations."""

# Standard library imports
import unittest

# Third party imports
import numpy as np
import torch

# Local application imports
import twin4build

twin4build._IS_TESTING = True

# Local application imports
from twin4build.estimator._transcription import (  # noqa: E402
    _aggregate_objective_targets,
)


def _pack_reference(Btt, Bty, Byy, iu_t, iu_y, n_seg):
    """Reference the original per-segment NumPy packing loop."""
    vals = [Btt.cpu().numpy()[iu_t]]
    for g in range(n_seg):
        vals.append(Bty[g].cpu().numpy().ravel())
        vals.append(Byy[g].cpu().numpy()[iu_y])
    return np.concatenate(vals)


def _pack_vectorized(Btt, Bty, Byy, iu_t, iu_y, n_seg, n_theta, Da):
    """Pack blocks in the order declared by IPOPT's sparsity pattern."""
    dev = Btt.device
    t0 = torch.as_tensor(iu_t[0], dtype=torch.long, device=dev)
    t1 = torch.as_tensor(iu_t[1], dtype=torch.long, device=dev)
    y0 = torch.as_tensor(iu_y[0], dtype=torch.long, device=dev)
    y1 = torch.as_tensor(iu_y[1], dtype=torch.long, device=dev)
    return torch.cat(
        [
            Btt[t0, t1],
            torch.cat(
                [
                    Bty.reshape(n_seg, n_theta * Da),
                    Byy[:, y0, y1],
                ],
                dim=1,
            ).reshape(-1),
        ]
    )


class TestHessianPackingOrder(unittest.TestCase):
    """Vectorized packing must reproduce the original loop exactly."""

    def test_packing_matches_reference(self):
        for n_theta, Da, n_seg in ((4, 3, 5), (28, 15, 7), (1, 2, 3)):
            with self.subTest(n_theta=n_theta, Da=Da, n_seg=n_seg):
                gen = torch.Generator().manual_seed(n_theta * 100 + Da * 10 + n_seg)
                Btt = torch.randn(n_theta, n_theta, dtype=torch.float64, generator=gen)
                Bty = torch.randn(
                    n_seg, n_theta, Da, dtype=torch.float64, generator=gen
                )
                Byy = torch.randn(n_seg, Da, Da, dtype=torch.float64, generator=gen)
                iu_t = np.triu_indices(n_theta)
                iu_y = np.triu_indices(Da)

                ref = _pack_reference(Btt, Bty, Byy, iu_t, iu_y, n_seg)
                got = _pack_vectorized(
                    Btt, Bty, Byy, iu_t, iu_y, n_seg, n_theta, Da
                ).numpy()

                self.assertEqual(got.shape, ref.shape)
                np.testing.assert_array_equal(
                    got,
                    ref,
                    "vectorized packing does not match the sparsity pattern",
                )

    def test_length_matches_the_sparsity_pattern(self):
        n_theta, Da, n_seg = 6, 4, 9
        iu_t, iu_y = np.triu_indices(n_theta), np.triu_indices(Da)
        expected = len(iu_t[0]) + n_seg * (n_theta * Da + len(iu_y[0]))
        Btt = torch.zeros(n_theta, n_theta, dtype=torch.float64)
        Bty = torch.zeros(n_seg, n_theta, Da, dtype=torch.float64)
        Byy = torch.zeros(n_seg, Da, Da, dtype=torch.float64)
        got = _pack_vectorized(Btt, Bty, Byy, iu_t, iu_y, n_seg, n_theta, Da)
        self.assertEqual(got.numel(), expected)


class TestCombinedExactHessian(unittest.TestCase):
    """One scalar transform must equal the three separate curvature terms."""

    def test_combined_matches_gauss_newton_residual_and_constraint_curvature(self):
        dtype = torch.float64
        n_seg, n_theta, Da, n_meas = 4, 2, 2, 2
        theta = torch.tensor([0.25, -0.35], dtype=dtype)
        states = torch.tensor(
            [[0.1, -0.2], [0.3, 0.15], [-0.25, 0.4], [0.05, -0.1]],
            dtype=dtype,
        )
        caps = torch.tensor([0.2, -0.1, 0.35, 0.05], dtype=dtype)
        actual = torch.tensor(
            [[0.4, -0.3], [0.1, 0.2], [-0.2, 0.5], [0.3, -0.1]],
            dtype=dtype,
        )
        included = torch.ones(n_seg, dtype=torch.bool)
        previous = torch.tensor([0, 0, 1, 2])
        lagged = torch.tensor([True, False])
        obj_mask, targets = _aggregate_objective_targets(
            actual, included, previous, lagged
        )
        sd = torch.tensor([0.7, 1.3], dtype=dtype)
        lam_by_seg = torch.zeros((n_seg, Da), dtype=dtype)
        lam_by_seg[:3] = torch.tensor(
            [[0.4, -0.2], [-0.1, 0.3], [0.25, 0.15]], dtype=dtype
        )
        scale = torch.tensor(0.8, dtype=dtype)

        def outputs(q, cap):
            y, th = q[:Da], q[Da:]
            end = torch.stack(
                [
                    torch.tanh(y[0] + th[0] * y[1] + cap),
                    y[1].square() + th[1] * y[0] + 0.2 * th[0].square(),
                ]
            )
            meas = torch.stack(
                [
                    torch.sin(y[0] + th[1]) + 0.1 * th[0] * y[1],
                    y[1] * th[0] + torch.exp(0.2 * th[1] + 0.1 * cap),
                ]
            )
            return end, meas

        Btt_combined = torch.zeros((n_theta, n_theta), dtype=dtype)
        Bty_combined = torch.zeros((n_seg, n_theta, Da), dtype=dtype)
        Byy_combined = torch.zeros((n_seg, Da, Da), dtype=dtype)
        Btt_separate = torch.zeros_like(Btt_combined)
        Bty_separate = torch.zeros_like(Bty_combined)
        Byy_separate = torch.zeros_like(Byy_combined)

        for g in range(n_seg):
            q = torch.cat([states[g], theta])

            def residuals(q_):
                return (outputs(q_, caps[g])[1] - targets[g]) / sd

            def combined_scalar(q_):
                end, _ = outputs(q_, caps[g])
                residual = residuals(q_)
                return (lam_by_seg[g] * end).sum() + 0.5 * scale * (
                    obj_mask[g] * residual.square()
                ).sum()

            combined = torch.func.hessian(combined_scalar)(q)

            jac_residual = torch.func.jacrev(residuals)(q)
            residual = residuals(q)
            gn = scale * torch.einsum(
                "m,mi,mj->ij", obj_mask[g], jac_residual, jac_residual
            )
            residual_curvature = torch.zeros_like(gn)
            for m in range(n_meas):
                residual_curvature += (
                    scale
                    * obj_mask[g, m]
                    * residual[m]
                    * torch.func.hessian(lambda q_: residuals(q_)[m])(q)
                )
            constraint_curvature = torch.func.hessian(
                lambda q_: (lam_by_seg[g] * outputs(q_, caps[g])[0]).sum()
            )(q)
            separate = gn + residual_curvature + constraint_curvature

            Btt_combined += combined[Da:, Da:]
            Bty_combined[g] = combined[Da:, :Da]
            Byy_combined[g] = combined[:Da, :Da]
            Btt_separate += separate[Da:, Da:]
            Bty_separate[g] = separate[Da:, :Da]
            Byy_separate[g] = separate[:Da, :Da]

        iu_t, iu_y = np.triu_indices(n_theta), np.triu_indices(Da)
        packed_combined = _pack_vectorized(
            Btt_combined,
            Bty_combined,
            Byy_combined,
            iu_t,
            iu_y,
            n_seg,
            n_theta,
            Da,
        )
        packed_separate = _pack_vectorized(
            Btt_separate,
            Bty_separate,
            Byy_separate,
            iu_t,
            iu_y,
            n_seg,
            n_theta,
            Da,
        )
        torch.testing.assert_close(
            packed_combined, packed_separate, rtol=1e-11, atol=1e-12
        )


class TestObjectiveCurvatureAttribution(unittest.TestCase):
    """Lagged objective terms must be assigned to their true producer."""

    def test_lagged_period_boundaries(self):
        actual = torch.tensor(
            [[10.0, 100.0], [20.0, 200.0], [30.0, 300.0], [40.0, 400.0]],
            dtype=torch.float64,
        )
        included = torch.tensor([True, True, True, True])
        previous = torch.tensor([0, 0, 1, 2])
        lagged = torch.tensor([True, False])

        count, target = _aggregate_objective_targets(actual, included, previous, lagged)

        torch.testing.assert_close(count[:, 0], actual.new_tensor([2.0, 1.0, 1.0, 0.0]))
        torch.testing.assert_close(
            target[:, 0], actual.new_tensor([15.0, 30.0, 40.0, 0.0])
        )
        torch.testing.assert_close(count[:, 1], actual.new_ones(4))
        torch.testing.assert_close(target[:, 1], actual[:, 1])

    def test_warmup_exclusion_removes_first_duplicate(self):
        actual = torch.tensor([[10.0], [20.0], [30.0], [40.0]])
        included = torch.tensor([False, True, True, True])
        previous = torch.tensor([0, 0, 1, 2])

        count, target = _aggregate_objective_targets(
            actual, included, previous, torch.tensor([True])
        )

        torch.testing.assert_close(count[:, 0], torch.tensor([1.0, 1.0, 1.0, 0.0]))
        torch.testing.assert_close(target[:, 0], torch.tensor([20.0, 30.0, 40.0, 0.0]))


if __name__ == "__main__":
    unittest.main()
