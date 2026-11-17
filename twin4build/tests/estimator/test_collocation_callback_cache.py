"""Tests for lazy, shared collocation callback evaluation."""

# Standard library imports
import unittest

# Third party imports
import numpy as np
import torch

# Local application imports
import twin4build

twin4build._IS_TESTING = True

# Local application imports
from twin4build.estimator._collocation import (  # noqa: E402
    _aggregate_objective_targets,
    _assemble_objective_gradient,
    _IterateCache,
    _pack_colored_jacobian_vals,
)


class TestIterateCache(unittest.TestCase):
    """Callback order must not change physical evaluation counts."""

    def test_forward_and_derivatives_are_lazy_and_shared(self):
        calls = {"forward": 0, "derivative": 0}
        cache = _IterateCache()
        z = np.array([0.25, -0.5, 1.5])

        def forward(value):
            calls["forward"] += 1
            return value**2

        def derivatives(value):
            calls["derivative"] += 1
            return 2.0 * value

        # IPOPT may request callbacks in any order. Repeated constraints and
        # objectives need only the forward; a gradient/Jacobian triggers the
        # derivative transform exactly once.
        np.testing.assert_array_equal(cache.forward(z, forward), z**2)
        np.testing.assert_array_equal(cache.forward(z.copy(), forward), z**2)
        np.testing.assert_array_equal(cache.derivatives(z, derivatives), 2.0 * z)
        np.testing.assert_array_equal(cache.forward(z, forward), z**2)
        np.testing.assert_array_equal(cache.derivatives(z.copy(), derivatives), 2.0 * z)
        self.assertEqual(calls, {"forward": 1, "derivative": 1})

        # A changed value invalidates both entries, even when the caller reuses
        # and mutates the same ndarray object.
        z[0] += 1.0
        cache.derivatives(z, derivatives)
        cache.forward(z, forward)
        self.assertEqual(calls, {"forward": 2, "derivative": 2})
        self.assertEqual(
            cache.stats,
            {
                "forward_evaluations": 2,
                "forward_cache_hits": 2,
                "derivative_evaluations": 2,
                "derivative_cache_hits": 1,
            },
        )


class TestSharedObjectiveGradient(unittest.TestCase):
    """The shared measurement Jacobian must reproduce autograd exactly."""

    def test_assembled_gradient_matches_lagged_objective_autograd(self):
        dtype = torch.float64
        n_seg, n_theta, Da = 4, 2, 2
        theta = torch.tensor([0.3, -0.2], dtype=dtype)
        states = torch.tensor(
            [[0.1, -0.4], [0.25, 0.2], [-0.3, 0.5], [0.4, -0.1]],
            dtype=dtype,
        )
        z = torch.cat([theta, states.reshape(-1)])
        actual = torch.tensor(
            [[0.2, 1.1], [0.4, 0.9], [-0.1, 1.3], [0.5, 0.8]],
            dtype=dtype,
        )
        sd = torch.tensor([0.7, 1.4], dtype=dtype)
        # Includes both a duplicated lagged producer (segments 0 and 1 map to
        # producer 0) and an excluded segment, covering lag and warmup masking.
        included = torch.tensor([True, True, False, True])
        previous = torch.tensor([0, 0, 1, 2])
        lagged_sensor = torch.tensor([True, False])
        target_count, target_mean = _aggregate_objective_targets(
            actual, included, previous, lagged_sensor
        )

        def measurement(y, th):
            return torch.stack(
                [
                    torch.sin(y[0] + th[1]) + 0.2 * y[1] * th[0],
                    torch.exp(0.1 * y[1]) + th[0] * y[0] + th[1].square(),
                ]
            )

        def raw_measurements(z_):
            th = z_[:n_theta]
            y = z_[n_theta:].reshape(n_seg, Da)
            return torch.vmap(lambda yi: measurement(yi, th))(y)

        def objective(z_):
            raw = raw_measurements(z_)
            scored = torch.where(
                lagged_sensor.unsqueeze(0),
                raw[previous],
                raw,
            )
            return (((actual - scored) / sd).square())[included].mean()

        raw = raw_measurements(z)
        Jx, Jt = torch.vmap(
            lambda yi: torch.func.jacrev(measurement, argnums=(0, 1))(yi, theta)
        )(states)
        assembled = _assemble_objective_gradient(
            Jt / sd.reshape(1, -1, 1),
            Jx / sd.reshape(1, -1, 1),
            raw,
            target_count,
            target_mean,
            sd,
            int(included.sum()) * actual.shape[1],
        )
        expected = torch.func.grad(objective)(z)
        torch.testing.assert_close(assembled, expected, rtol=1e-11, atol=1e-12)


class TestPackColoredJacobianVals(unittest.TestCase):
    """COO packing must match the old per-entry host loop exactly."""

    def test_vectorized_pack_matches_per_entry_loop(self):
        n_seg, n_links, n_rows = 4, 3, 5
        n_tg, n_tl, n_xg, n_xl = 2, 3, 1, 4
        device = torch.device("cpu")
        generator = torch.Generator().manual_seed(0)
        Jtg = torch.randn(n_seg, n_rows, n_tg, generator=generator)
        Jtl = torch.randn(n_seg, n_rows, n_tl, generator=generator)
        Jxg = torch.randn(n_seg, n_rows, n_xg, generator=generator)
        Jxl = torch.randn(n_seg, n_rows, n_xl, generator=generator)
        cp_i = torch.tensor([0, 2, 3], dtype=torch.long)
        row_includes_local = torch.tensor(
            [False, True, True, False, True], dtype=torch.bool
        )

        expected = []
        for i in cp_i.tolist():
            for row, include_local in enumerate(row_includes_local.tolist()):
                expected.extend(Jtg[i, row].tolist())
                if include_local:
                    expected.extend(Jtl[i, row].tolist())
                expected.extend(Jxg[i, row].tolist())
                if include_local:
                    expected.extend(Jxl[i, row].tolist())
                expected.append(-1.0)

        packed = _pack_colored_jacobian_vals(
            Jtg, Jtl, Jxg, Jxl, cp_i, row_includes_local
        )
        torch.testing.assert_close(
            packed, torch.tensor(expected, dtype=packed.dtype, device=device)
        )

    def test_empty_local_colors_keep_global_blocks_and_minus_one(self):
        Jtg = torch.arange(8, dtype=torch.float64).reshape(2, 2, 2)
        Jtl = torch.zeros(2, 2, 0, dtype=torch.float64)
        Jxg = torch.arange(4, dtype=torch.float64).reshape(2, 2, 1)
        Jxl = torch.zeros(2, 2, 0, dtype=torch.float64)
        cp_i = torch.tensor([1], dtype=torch.long)
        packed = _pack_colored_jacobian_vals(
            Jtg,
            Jtl,
            Jxg,
            Jxl,
            cp_i,
            torch.tensor([False, True]),
        )
        expected = torch.tensor(
            [4.0, 5.0, 2.0, -1.0, 6.0, 7.0, 3.0, -1.0], dtype=torch.float64
        )
        torch.testing.assert_close(packed, expected)


if __name__ == "__main__":
    unittest.main()

