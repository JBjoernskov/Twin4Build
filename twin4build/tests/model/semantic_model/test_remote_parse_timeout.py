"""A remote ontology fetch fails after a timeout instead of waiting for a host
that accepts the connection and never answers (26 Oct 2026: VoID redirects to
``http://vocab.deri.ie/void``, whose host hung every reasoning step and every
test that reasons)."""
import socket
import threading
import time
import unittest
from unittest import mock

from rdflib import Graph

import twin4build as tb
from twin4build.model.semantic_model import semantic_model

tb._IS_TESTING = True


class _SilentServer:
    """A TCP server that accepts connections and never sends a byte."""

    def __enter__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(4)
        self.held = []
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()
        return f"http://127.0.0.1:{self.sock.getsockname()[1]}/ontology.ttl"

    def _serve(self):
        self.sock.settimeout(0.1)
        while not self.stop.is_set():
            try:
                conn, _ = self.sock.accept()
                self.held.append(conn)
            except OSError:
                continue

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join(timeout=2)
        for conn in self.held:
            conn.close()
        self.sock.close()


class TestRemoteParseTimeout(unittest.TestCase):
    def test_a_silent_host_fails_after_the_timeout(self):
        with _SilentServer() as url, mock.patch.object(semantic_model, "REMOTE_PARSE_TIMEOUT", 0.5):
            started = time.monotonic()
            with self.assertRaises(OSError):  # socket.timeout / URLError
                semantic_model.parse_wrapper(Graph(), source=url, format="turtle")
            self.assertLess(time.monotonic() - started, 10.0)
        # the process-wide default is left as it was
        self.assertIsNone(socket.getdefaulttimeout())


if __name__ == "__main__":
    unittest.main()
