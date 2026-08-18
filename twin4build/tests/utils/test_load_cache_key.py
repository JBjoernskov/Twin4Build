"""The spreadsheet cache must not confuse two files that share a basename.

``load_from_spreadsheet`` caches parsed/resampled data to disk.  The key was
built from ``os.path.basename`` alone, so two different CSVs with the same
name -- which the repo actually ships, e.g. ``estimator_example/`` and
``full_workflow_example/`` both contain ``damper_position_sensor.csv`` --
mapped to the same cache entry and the second silently returned the first
one's data.

The failure mode is the dangerous kind: no exception, no warning, just wrong
values.  In practice it made a calibration example load its damper
measurements as all-zeros, and the estimator faithfully switched the
ventilation branch off to fit them.
"""

# Standard library imports
import datetime
import unittest

# Third party imports
import pandas as pd
from dateutil import tz

# Local application imports
from twin4build.utils.data_loaders.load import load_from_spreadsheet


class TestSpreadsheetCacheKey(unittest.TestCase):
    """Same basename, different directories, different contents."""

    START = datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=tz.UTC)
    END = datetime.datetime(2023, 1, 1, 6, 0, 0, tzinfo=tz.UTC)
    STEP = 600

    def _write(self, directory, value):
        """A minimal two-column CSV of a constant `value`."""
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "reading.csv"
        stamps = pd.date_range(self.START, self.END, freq="10min", tz=tz.UTC)
        pd.DataFrame(
            {
                "time": stamps.tz_localize(None),
                "value": [value] * len(stamps),
            }
        ).to_csv(path, index=False)
        return str(path)

    def _load(self, path):
        df = load_from_spreadsheet(
            path, 0, 1, step_size=self.STEP,
            start_time=self.START, end_time=self.END, cache=True,
        )
        return float(pd.DataFrame(df).to_numpy().astype(float).mean())

    def test_same_basename_different_dirs_do_not_share_cache(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = self._write(root / "example_a", 1.0)
            b = self._write(root / "example_b", 9.0)

            got_a = self._load(a)          # populates the cache
            got_b = self._load(b)          # must NOT reuse a's entry
            got_a_again = self._load(a)    # and a must still be a

            self.assertAlmostEqual(got_a, 1.0, places=6)
            self.assertAlmostEqual(
                got_b, 9.0, places=6,
                msg="second file returned the first file's cached values -- "
                    "the cache key does not distinguish directories",
            )
            self.assertAlmostEqual(got_a_again, 1.0, places=6)

    def test_editing_a_file_in_place_invalidates_its_cache(self):
        """Rewriting a CSV at the same path must not serve stale values."""
        import os
        import tempfile
        import time
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp) / "example"
            path = self._write(d, 2.0)
            self.assertAlmostEqual(self._load(path), 2.0, places=6)

            time.sleep(1.1)  # ensure a distinct mtime at 1 s resolution
            self._write(d, 7.0)
            os.utime(path, None)
            self.assertAlmostEqual(
                self._load(path), 7.0, places=6,
                msg="edited file still served its previous cached values",
            )


if __name__ == "__main__":
    unittest.main()
