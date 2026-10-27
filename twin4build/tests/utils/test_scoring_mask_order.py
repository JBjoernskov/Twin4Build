"""period_mask selects a time-indexed scoring mask by the period's timestamps, whatever the order of the periods the
mask was built over."""
import datetime
import unittest

import numpy as np
import pandas as pd
from dateutil import tz

from twin4build.utils.scoring_mask import period_mask

CPH = tz.gettz("Europe/Copenhagen")


def grid(year, month, day, n=100):
    return pd.date_range(datetime.datetime(year, month, day, tzinfo=CPH), periods=n, freq="600s")


class TestPeriodMask(unittest.TestCase):
    def test_periods_in_time_order(self):
        winter, spring = grid(2024, 1, 1), grid(2024, 3, 1)
        mask = pd.Series(True, index=winter.append(spring))
        mask.iloc[10:30] = False
        mask.iloc[150:155] = False
        self.assertEqual(np.flatnonzero(~period_mask(mask, winter, 100)).tolist(), list(range(10, 30)))
        self.assertEqual(np.flatnonzero(~period_mask(mask, spring, 100)).tolist(), list(range(50, 55)))

    def test_periods_out_of_time_order(self):
        """The controllers stage fits a winter span and the summer before it: the mask is built in that order."""
        winter, summer = grid(2023, 12, 24), grid(2023, 6, 4)
        mask = pd.Series(True, index=winter.append(summer))
        mask.iloc[10:30] = False
        mask.iloc[160:170] = False
        self.assertEqual(np.flatnonzero(~period_mask(mask, winter, 100)).tolist(), list(range(10, 30)))
        self.assertEqual(np.flatnonzero(~period_mask(mask, summer, 100)).tolist(), list(range(60, 70)))

    def test_a_time_the_mask_does_not_cover_is_scored(self):
        mask = pd.Series(False, index=grid(2024, 1, 1, 10))
        self.assertTrue(period_mask(mask, grid(2024, 2, 1, 10), 10).all())

    def test_a_plain_array_is_the_periods_own_steps(self):
        keep = period_mask(np.array([True, False, True]), grid(2024, 1, 1, 5), 5)
        self.assertEqual(keep.tolist(), [True, False, True, True, True])


if __name__ == "__main__":
    unittest.main()
