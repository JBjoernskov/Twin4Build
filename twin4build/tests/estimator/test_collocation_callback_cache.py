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
from twin4build.estimator._transcription import (  # noqa: E402
    _aggregate_objective_targets,
    _assemble_objective_gradient,
    _IterateCache,
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


if __name__ == "__main__":
    unittest.main()
