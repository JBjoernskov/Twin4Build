"""``clamp(mode="smooth")`` must stay a clamp for every range width.

Regression: the smooth clamp used a fixed transition width (``curve_start``,
0.1 by default) on each side.  With a range narrower than twice that width
the two transition points crossed, and the map overshot the upper bound by
up to ``curve_start``, undershot the lower one, and stopped being monotone.
A PI controller whose output range had been identified as ``[0, 0.036]``
therefore produced a "clamped" command swinging between -0.026 and +0.100
-- inside a closed loop, which then oscillated every step.
"""

# Standard library imports
import unittest

# Third party imports
import torch

# Local application imports
from twin4build.systems.utils.smooth_saturation import clamp


class TestSmoothClampStaysWithinBounds(unittest.TestCase):
    def _check(self, lower, upper, curve_start=0.1):
        u = torch.linspace(lower - 1.0, upper + 1.0, 4001, dtype=torch.float64)
        y = clamp(u, lower=lower, upper=upper, curve_start=curve_start, mode="smooth")
        self.assertGreaterEqual(float(y.min()), lower - 1e-12, (lower, upper))
        self.assertLessEqual(float(y.max()), upper + 1e-12, (lower, upper))
        # Monotone non-decreasing.
        self.assertGreaterEqual(float((y[1:] - y[:-1]).min()), -1e-12, (lower, upper))
        # Identity well inside the range.
        mid = 0.5 * (lower + upper)
        self.assertAlmostEqual(
            float(clamp(torch.tensor(mid, dtype=torch.float64), lower, upper, mode="smooth")), mid, places=12
        )

    def test_wide_range_unchanged(self):
        self._check(0.0, 1.0)

    def test_tight_range(self):
        """The case that failed: a range narrower than 2 * curve_start."""
        self._check(0.0, 0.0359)

    def test_very_tight_and_shifted_ranges(self):
        for lower, upper in ((0.0, 0.01), (0.5, 0.52), (-1.0, -0.9), (0.0, 0.2)):
            self._check(lower, upper)

    def test_tensor_bounds(self):
        """Bounds arrive as tensors from ``params`` in a component's forward."""
        lower = torch.tensor(0.0, dtype=torch.float64)
        upper = torch.tensor(0.0359, dtype=torch.float64)
        u = torch.linspace(-0.5, 0.5, 2001, dtype=torch.float64)
        y = clamp(u, lower=lower, upper=upper, mode="smooth")
        self.assertGreaterEqual(float(y.min()), 0.0)
        self.assertLessEqual(float(y.max()), 0.0359)

    def test_hard_mode_untouched(self):
        u = torch.tensor([-1.0, 0.5, 2.0])
        self.assertTrue(torch.equal(clamp(u, 0.0, 1.0, mode="hard"), torch.tensor([0.0, 0.5, 1.0])))


if __name__ == "__main__":
    unittest.main()
