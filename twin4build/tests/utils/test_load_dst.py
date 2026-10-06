"""Periods across a daylight-saving change load and simulate (#243).

psycopg2 returns ``timestamptz`` in the session's time zone: a session in
local time gives one column two UTC offsets across a DST change, which
``pd.to_datetime`` refused, so every window over a DST change failed.  The
loader then returns the real duration (23 or 25 hours a day), and the
simulator's steps must count the same: they were wall-clock arithmetic (an
hour more in spring, padded with the last value and every step label an hour
behind; an hour less in autumn, where the input could not take the data).
"""

# Standard library imports
import datetime
import unittest

# Third party imports
import numpy as np
import pandas as pd
from dateutil import tz

# Local application imports
import twin4build as tb
from twin4build.utils.data_loaders.load import sample_from_df
from twin4build.utils.simulation_time import get_simulation_timesteps

CPH = tz.gettz("Europe/Copenhagen")


class TestSampleAcrossDst(unittest.TestCase):
    def _rows(self, start_utc, hours):
        """Ten-minute rows as the database returns them: local wall time with
        a fixed offset each, the value the minutes since ``start_utc``."""
        times = [start_utc + datetime.timedelta(minutes=10 * i) for i in range(6 * hours + 1)]
        local = [t.astimezone(CPH) for t in times]
        fixed = [t.replace(tzinfo=datetime.timezone(t.utcoffset())) for t in local]
        values = [(t - start_utc).total_seconds() / 60 for t in times]
        return pd.DataFrame({"time": fixed, "value": values})

    def test_mixed_offsets_load_as_one_utc_timeline(self):
        start = datetime.datetime(2023, 3, 25, 12, tzinfo=datetime.timezone.utc)  # DST on 26 Mar 01:00 UTC
        df = self._rows(start, 24)
        self.assertEqual({str(t.utcoffset()) for t in df["time"]}, {"1:00:00", "2:00:00"})
        out = sample_from_df(
            df, date_column=0, value_column=1, step_size=600,
            start_time=start.astimezone(CPH), end_time=(start + datetime.timedelta(hours=24)).astimezone(CPH), tz=CPH,
        )
        # the window [start, end) in ten-minute steps, continuous over the change
        np.testing.assert_allclose(out.iloc[:, -1].to_numpy(float), np.arange(0, 24 * 60, 10, dtype=float))

    def test_one_offset_is_unchanged(self):
        start = datetime.datetime(2023, 3, 10, 12, tzinfo=datetime.timezone.utc)
        df = self._rows(start, 6)
        out = sample_from_df(
            df, date_column=0, value_column=1, step_size=600,
            start_time=start.astimezone(CPH), end_time=(start + datetime.timedelta(hours=6)).astimezone(CPH), tz=CPH,
        )
        np.testing.assert_allclose(out.iloc[:, -1].to_numpy(float), np.arange(0, 6 * 60, 10, dtype=float))



class TestStepsAcrossDst(unittest.TestCase):
    """Two local days over each change, ten-minute steps."""

    SPRING = (datetime.datetime(2023, 3, 25, tzinfo=CPH), datetime.datetime(2023, 3, 27, tzinfo=CPH), 47)
    AUTUMN = (datetime.datetime(2023, 10, 28, tzinfo=CPH), datetime.datetime(2023, 10, 30, tzinfo=CPH), 49)

    def test_the_steps_count_the_real_duration_in_local_time(self):
        for start, end, hours in (self.SPRING, self.AUTUMN):
            with self.subTest(start=start):
                _, dts, n_t, _ = get_simulation_timesteps(start, end, 600)
                self.assertEqual(n_t, 6 * hours)
                steps = list(dts[0])
                gaps = {(b.astimezone(datetime.timezone.utc) - a.astimezone(datetime.timezone.utc)).total_seconds() for a, b in zip(steps, steps[1:])}
                self.assertEqual(gaps, {600.0})
                self.assertEqual(steps[-1] + datetime.timedelta(minutes=10), end)  # the last step ends the local day

    def test_spring_skips_the_missing_hour_autumn_repeats_one(self):
        _, dts, _, _ = get_simulation_timesteps(*self.SPRING[:2], 600)
        labels = [f"{d:%H:%M}" for d in dts[0] if d.day == 26 and d.hour < 4]
        self.assertNotIn("02:00", labels)
        self.assertEqual(labels[labels.index("01:50") + 1], "03:00")
        _, dts, _, _ = get_simulation_timesteps(*self.AUTUMN[:2], 600)
        self.assertEqual(sum(1 for d in dts[0] if d.day == 29 and f"{d:%H:%M}" == "02:30"), 2)

    def test_an_input_carries_the_real_series_without_padding(self):
        for start, end, hours in (self.SPRING, self.AUTUMN):
            with self.subTest(start=start):
                utc = datetime.timezone.utc
                times = pd.date_range(start.astimezone(utc), end.astimezone(utc), freq="10min", inclusive="left")
                df = pd.DataFrame({"value": (times - times[0]).total_seconds() / 60}, index=times.tz_convert(CPH))
                source = tb.TimeSeriesInputSystem(id=f"dst_{start:%m}", df=df)
                source.initialize(start_time=[start], end_time=[end], step_size=[600])
                values = np.asarray(source.values, dtype=float).reshape(-1)
                np.testing.assert_allclose(values, np.arange(0, hours * 60, 10, dtype=float))


if __name__ == "__main__":
    unittest.main()
