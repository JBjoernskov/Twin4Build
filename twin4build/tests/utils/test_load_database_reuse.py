"""``load_from_database`` against a fake server: one connection per process,
the table checked once, past empty windows cached, and a connection the
server dropped while idle opened again.

Every load used to open its own connection and check the table (about
0.45 s per load on the HTR server), and a window without rows was never
cached: a model with points that log nothing in a period queried them again
at every simulate.
"""

# Standard library imports
import datetime
import tempfile
import unittest
from unittest import mock

# Third party imports
import pandas as pd
import psycopg2

# Local application imports
import twin4build
import twin4build.utils.data_loaders.load as load_mod

twin4build._IS_TESTING = True

UTC = datetime.timezone.utc


class FakeServer:
    def __init__(self, data):
        self.data = data  # point -> pd.Series
        self.connections = []
        self.queries = []
        self.down = False

    def connect(self, conn_string):
        conn = FakeConnection(self)
        self.connections.append(conn)
        return conn

    def rows(self, point, start, end):
        s = self.data.get(point)
        if s is None:
            return []
        s = s[(s.index >= start) & (s.index < end)]
        return [{"time": t.to_pydatetime(), "value": float(v)} for t, v in s.items()]


class FakeConnection:
    def __init__(self, server):
        self.server = server
        self.closed = 0
        self.dropped = False  # the server closed it while idle
        self.autocommit = False

    def cursor(self, cursor_factory=None):
        return FakeCursor(self)

    def close(self):
        self.closed = 1


class FakeCursor:
    def __init__(self, conn):
        self.conn = conn

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, query, params=None):
        if self.conn.dropped or self.conn.server.down:
            raise psycopg2.OperationalError("server closed the connection unexpectedly")
        self.conn.server.queries.append(query)
        if "information_schema" in query:
            self.one = {"exists": True}
        else:
            self.all = self.conn.server.rows(*params)

    def fetchone(self):
        return self.one

    def fetchall(self):
        return self.all


class TestLoadDatabaseReuse(unittest.TestCase):
    def setUp(self):
        index = pd.date_range("2024-03-01", "2024-04-01", freq="600s", tz=UTC, inclusive="left")
        self.server = FakeServer({"P1": pd.Series(range(len(index)), index=index, dtype=float)})
        self.tmp = tempfile.TemporaryDirectory()
        load_mod._CONNECTIONS.clear()
        load_mod._TABLES.clear()
        patch = mock.patch.object(load_mod.psycopg2, "connect", side_effect=self.server.connect)
        patch.start()
        self.addCleanup(patch.stop)
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(load_mod._TABLES.clear)
        self.addCleanup(load_mod._CONNECTIONS.clear)

    def load(self, point, start, end, cache=True):
        return load_mod.load_from_database(
            start_time=start,
            end_time=end,
            step_size=600,
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
        )

    def data_queries(self):
        return [q for q in self.server.queries if "information_schema" not in q]

    def table_checks(self):
        return [q for q in self.server.queries if "information_schema" in q]

    def test_one_connection_and_one_table_check(self):
        a = self.load("P1", datetime.datetime(2024, 3, 4, tzinfo=UTC), datetime.datetime(2024, 3, 5, tzinfo=UTC), cache=False)
        b = self.load("P1", datetime.datetime(2024, 3, 6, tzinfo=UTC), datetime.datetime(2024, 3, 7, tzinfo=UTC), cache=False)
        self.assertEqual(len(self.server.connections), 1)
        self.assertEqual(len(self.table_checks()), 1)
        self.assertEqual(len(self.data_queries()), 2)
        self.assertGreater(len(a), 0)
        self.assertGreater(len(b), 0)

    def test_a_past_empty_window_is_cached(self):
        start, end = datetime.datetime(2024, 3, 4, tzinfo=UTC), datetime.datetime(2024, 3, 5, tzinfo=UTC)
        first = self.load("DEAD", start, end)
        second = self.load("DEAD", start, end)
        self.assertEqual(len(self.data_queries()), 1)
        self.assertEqual(len(first), 0)
        self.assertEqual(len(second), 0)

    def test_a_recent_empty_window_is_not_cached(self):
        # Its rows may still arrive.
        end = datetime.datetime.now(UTC) - datetime.timedelta(hours=12)
        start = end - datetime.timedelta(days=1)
        self.load("DEAD", start, end)
        self.load("DEAD", start, end)
        self.assertEqual(len(self.data_queries()), 2)

    def test_a_dropped_connection_is_opened_again(self):
        self.load("P1", datetime.datetime(2024, 3, 4, tzinfo=UTC), datetime.datetime(2024, 3, 5, tzinfo=UTC), cache=False)
        self.server.connections[0].dropped = True
        again = self.load("P1", datetime.datetime(2024, 3, 6, tzinfo=UTC), datetime.datetime(2024, 3, 7, tzinfo=UTC), cache=False)
        self.assertEqual(len(self.server.connections), 2)
        self.assertEqual(len(self.table_checks()), 2)  # checked again on the new connection
        self.assertGreater(len(again), 0)

    def test_a_new_connection_that_fails_raises(self):
        self.server.down = True
        with self.assertRaises(psycopg2.OperationalError):
            self.load("P1", datetime.datetime(2024, 3, 4, tzinfo=UTC), datetime.datetime(2024, 3, 5, tzinfo=UTC), cache=False)
        self.assertEqual(len(self.server.connections), 1)  # no retry on a connection that was just opened


if __name__ == "__main__":
    unittest.main()
