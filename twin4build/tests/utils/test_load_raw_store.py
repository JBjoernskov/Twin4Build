"""``load_from_database``'s raw store: one parquet file of raw rows per point,
with the intervals it covers.

The disk cache holds the processed series of one exact window and step, so
every new split of a period into windows queried the database once per point
and window (about 4,800 queries for the HTR ring's 601 measured points over
eight windows).  A window missing from the store now fetches the whole span of
windows around it, and a later split of the same span is cut from the store.
"""

# Standard library imports
import datetime
import os
import tempfile
import unittest
from unittest import mock

# Third party imports
import pandas as pd

# Local application imports
import twin4build
import twin4build.utils.data_loaders.load as load_mod
from twin4build.tests.utils.test_load_database_reuse import FakeServer

twin4build._IS_TESTING = True

UTC = datetime.timezone.utc
CET = datetime.timezone(datetime.timedelta(hours=1))


def day(d, h=0):
    return datetime.datetime(2024, 3, d, h, tzinfo=CET)


class TestRawStore(unittest.TestCase):
    def setUp(self):
        index = pd.date_range("2024-02-20", "2024-04-10", freq="300s", tz=UTC, inclusive="left")
        self.server = FakeServer({"P1": pd.Series(range(len(index)), index=index, dtype=float)})
        self.tmp = tempfile.TemporaryDirectory()
        for state in (load_mod._CONNECTIONS, load_mod._TABLES, load_mod._RAW_MEMO):
            state.clear()
            self.addCleanup(state.clear)
        patch = mock.patch.object(load_mod.psycopg2, "connect", side_effect=self.server.connect)
        patch.start()
        self.addCleanup(patch.stop)
        self.addCleanup(self.tmp.cleanup)

    def load(self, start, end, cache=True, fetch_span=None, point="P1", step=600):
        return load_mod.load_from_database(
            start_time=start,
            end_time=end,
            step_size=step,
            cache=cache,
            cache_root=self.tmp.name,
            table_name="measurements",
            sensor_id=point,
            id_column="name",
            db_host="localhost",
            db_port=5432,
            db_name="db",
            db_user="u",
            db_password="p",
            fetch_span=fetch_span,
        )

    def data_queries(self):
        return [q for q in self.server.queries if "information_schema" not in q]

    def windows(self, first, count, days):
        starts = [first + datetime.timedelta(days=days * k) for k in range(count)]
        return starts, [s + datetime.timedelta(days=days) for s in starts]

    def test_a_span_of_windows_is_one_query_and_matches_the_database(self):
        starts, ends = self.windows(day(4), 4, 1.5)
        span = load_mod.contiguous_spans(starts, ends)[0]
        stored = [self.load(a, b, fetch_span=span) for a, b in zip(starts, ends)]
        self.assertEqual(len(self.data_queries()), 1)
        direct = [self.load(a, b, cache=False) for a, b in zip(starts, ends)]
        for got, want in zip(stored, direct):
            pd.testing.assert_series_equal(got, want)

    def test_a_new_split_of_the_span_is_cut_from_the_store(self):
        starts, ends = self.windows(day(4), 4, 1.5)
        span = load_mod.contiguous_spans(starts, ends)[0]
        for a, b in zip(starts, ends):
            self.load(a, b, fetch_span=span)
        queries = len(self.data_queries())
        starts, ends = self.windows(day(4), 3, 2.0)  # the same six days as three windows, another step
        stored = [self.load(a, b, fetch_span=span, step=300) for a, b in zip(starts, ends)]
        self.assertEqual(len(self.data_queries()), queries)
        direct = [self.load(a, b, cache=False, step=300) for a, b in zip(starts, ends)]
        for got, want in zip(stored, direct):
            pd.testing.assert_series_equal(got, want)

    def test_only_the_missing_part_is_fetched(self):
        self.load(day(4), day(6))
        self.load(day(5), day(8))  # overlaps: fetches [6 Mar, 8 Mar + buffer) only
        params = self.server.fetched
        self.assertEqual(len(params), 2)
        self.assertEqual(pd.Timestamp(params[1][1]), pd.Timestamp(day(6)) + pd.Timedelta(seconds=1800))

    def test_a_window_outside_the_span_is_fetched_alone(self):
        # A window shifted back (gap filling) under its original span's name.
        span = (day(20), day(27))
        self.load(day(6), day(7), fetch_span=span)
        a, b = self.server.fetched[-1][1:]
        self.assertLessEqual(pd.Timestamp(b) - pd.Timestamp(a), pd.Timedelta(days=1, hours=1))

    def test_separate_periods_are_separate_spans(self):
        starts = [day(1), day(2), day(20)]
        ends = [day(2), day(3), day(21)]
        self.assertEqual(load_mod.contiguous_spans(starts, ends), [(day(1), day(3)), (day(20), day(21))])

    def test_recent_rows_are_fetched_again(self):
        now = datetime.datetime.now(UTC).replace(microsecond=0)
        index = pd.date_range(now - datetime.timedelta(days=4), now, freq="300s")
        self.server.data["NEW"] = pd.Series(1.0, index=index)
        start, end = now - datetime.timedelta(days=3), now - datetime.timedelta(hours=1)
        self.load(start, end, point="NEW")
        os.remove(self.processed_cache_file(start, end, "NEW"))  # the processed window: fall through to the store
        self.load(start, end, point="NEW")
        self.assertEqual(len(self.data_queries()), 2)
        a, b = self.server.fetched[-1][1:]
        self.assertGreaterEqual(pd.Timestamp(a), pd.Timestamp(now - load_mod.EMPTY_WINDOW_CACHE_AGE) - pd.Timedelta(seconds=1))

    def processed_cache_file(self, start, end, point):
        folder = os.path.join(self.tmp.name, "generated_files", "cached_data")
        matches = [f for f in os.listdir(folder) if f.startswith(f"db_measurements_sensor_{point}_")]
        self.assertEqual(len(matches), 1)
        return os.path.join(folder, matches[0])


if __name__ == "__main__":
    unittest.main()
