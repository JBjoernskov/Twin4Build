"""Database rows across a daylight-saving change load (#243).

psycopg2 returns ``timestamptz`` in the session's time zone: a session in
local time gives one column two UTC offsets across a DST change, which
``pd.to_datetime`` refused, so every window over a DST change failed.
"""

# Standard library imports
import datetime
import unittest

# Third party imports
import numpy as np
import pandas as pd
from dateutil import tz

# Local application imports
from twin4build.utils.data_loaders.load import sample_from_df

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


if __name__ == "__main__":
    unittest.main()
