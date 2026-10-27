"""Resampling never invents data: nothing beyond the readings, no gap longer than ``dt_limit`` bridged.

The loader used to hold the first and last readings over the whole window (``ffill().bfill()``) and interpolate
across gaps of any length: a window outside a series came back as its last value repeated, a month without a
reading as a straight line, both silently.  The three methods (``linear``: the value at the step, ``constant``: the
last reading at or before it, ``mean``: the time-weighted mean over the step) now give NaN wherever the readings do
not reach.
"""

# Standard library imports
import datetime
import unittest

# Third party imports
import numpy as np
import pandas as pd

# Local application imports
from twin4build.utils.data_loaders.load import DEFAULT_DT_LIMIT, sample_from_df, sample_on_grid

UTC = datetime.timezone.utc
T0 = datetime.datetime(2024, 1, 1, tzinfo=UTC)


def readings(minutes, values):
    """Time-indexed readings at ``minutes`` after T0."""
    return pd.DataFrame({"value": np.asarray(values, dtype=float)},
                        index=pd.DatetimeIndex([T0 + datetime.timedelta(minutes=float(m)) for m in minutes]))


def grid(start_min, end_min, frame, step=600, method="linear", dt_limit=DEFAULT_DT_LIMIT):
    out = sample_on_grid(frame, T0 + datetime.timedelta(minutes=start_min), T0 + datetime.timedelta(minutes=end_min),
                         step, method, dt_limit)
    return out["value"].to_numpy(float)


class TestEdges(unittest.TestCase):
    def test_nothing_before_the_first_or_after_the_last_reading(self):
        frame = readings([30, 40, 50, 60], [1.0, 2.0, 3.0, 4.0])
        for method in ("linear", "constant", "mean"):
            v = grid(0, 100, frame, method=method)  # steps at 0, 10, ..., 100
            self.assertTrue(np.isnan(v[:3]).all(), method)  # 0, 10, 20 before the first reading
            self.assertTrue(np.isnan(v[7:]).all(), method)  # 70 and later after the last one

    def test_a_window_outside_the_series_is_all_nan(self):
        """The 12 Oct case: a 2024 series sampled for a February 2025 window gave its last value 1008 times."""
        frame = readings(np.arange(0, 6 * 24 * 60, 10), np.linspace(15.0, 20.0, 6 * 24 * 6))
        start = 400 * 24 * 60
        for method in ("linear", "constant", "mean"):
            v = grid(start, start + 7 * 24 * 60, frame, method=method)
            self.assertTrue(np.isnan(v).all(), method)

    def test_sample_from_df_keeps_the_nan(self):
        df = pd.DataFrame({"time": [T0 + datetime.timedelta(minutes=m) for m in (30, 40, 50)], "value": [1.0, 2.0, 3.0]})
        out = sample_from_df(df, date_column=0, value_column=1, step_size=600, start_time=T0,
                             end_time=T0 + datetime.timedelta(minutes=90), tz=UTC)
        v = out["value"].to_numpy(float)
        self.assertEqual(len(v), 9)  # the whole window, [start, end)
        np.testing.assert_allclose(v[3:6], [1.0, 2.0, 3.0])
        self.assertTrue(np.isnan(v[:3]).all() and np.isnan(v[6:]).all())


class TestGaps(unittest.TestCase):
    def setUp(self):
        # readings every 10 minutes for 2 hours, none for 8 hours, then 2 hours again
        minutes = list(range(0, 121, 10)) + list(range(600, 721, 10))
        self.frame = readings(minutes, np.arange(len(minutes), dtype=float))

    def test_a_gap_longer_than_dt_limit_is_nan(self):
        v = grid(0, 720, self.frame, dt_limit=6 * 3600)
        self.assertFalse(np.isnan(v[:13]).any())  # 0 .. 120 read
        self.assertTrue(np.isnan(v[13:60]).all())  # 130 .. 590: 8 hours without a reading
        self.assertFalse(np.isnan(v[60:72]).any())

    def test_a_gap_within_dt_limit_is_interpolated(self):
        v = grid(0, 720, self.frame, dt_limit=9 * 3600)
        self.assertFalse(np.isnan(v[:72]).any())
        self.assertAlmostEqual(v[36], 12.0 + (360 - 120) / 480)  # halfway between 12 at 120 min and 13 at 600 min

    def test_none_bridges_inner_gaps_but_not_the_edges(self):
        v = grid(-60, 780, self.frame, dt_limit=None)
        self.assertTrue(np.isnan(v[:6]).all() and np.isnan(v[-5:]).all())
        self.assertFalse(np.isnan(v[6:-6]).any())

    def test_constant_holds_at_most_dt_limit(self):
        v = grid(0, 720, self.frame, method="constant", dt_limit=3 * 3600)
        np.testing.assert_allclose(v[12:31], 12.0)  # 120 .. 300 min: the reading at 120 held for 3 hours
        self.assertTrue(np.isnan(v[31:60]).all())  # longer than 3 hours since the last reading


class TestMean(unittest.TestCase):
    def test_the_mean_of_a_ramp_is_its_value_mid_step(self):
        minutes = np.arange(0, 121, 1)
        frame = readings(minutes, minutes.astype(float))  # y = t in minutes
        v = grid(0, 110, frame, method="mean")
        np.testing.assert_allclose(v[:11], np.arange(0, 110, 10) + 5.0)

    def test_the_mean_over_a_step_weighs_time_not_readings(self):
        # 1 for the first 9 minutes, 10 at minute 9 and after (readings bunched at the end)
        minutes = [0, 9, 9.5, 9.9, 10, 20]
        frame = readings(minutes, [1, 1, 10, 10, 10, 10])
        v = grid(0, 10, frame, method="mean")
        expected = (9 * 1 + 0.5 * (1 + 10) * 0.5 + 0.5 * 10) / 10  # trapezoids of the interpolant
        self.assertAlmostEqual(v[0], expected)

    def test_a_step_reaching_beyond_the_readings_is_nan(self):
        frame = readings([0, 10, 20, 25], [1.0, 1.0, 1.0, 1.0])
        v = grid(0, 30, frame, method="mean")  # steps [0, 10), [10, 20), [20, 30), [30, 40)
        np.testing.assert_allclose(v[:2], 1.0)
        self.assertTrue(np.isnan(v[2:]).all())  # [20, 30) ends after the last reading at 25

    def test_a_step_overlapping_a_long_gap_is_nan(self):
        frame = readings([0, 10, 20, 500, 510, 520], np.ones(6))
        v = grid(0, 520, frame, method="mean", dt_limit=3600)
        np.testing.assert_allclose(v[:2], 1.0)
        self.assertTrue(np.isnan(v[2:50]).all())  # [20, 30) .. [490, 500) inside the 480-minute gap
        np.testing.assert_allclose(v[50:52], 1.0)


if __name__ == "__main__":
    unittest.main()
