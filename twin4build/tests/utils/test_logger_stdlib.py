"""The logger as a guest in somebody else's application (#138).

Three things made ``twin4build`` hard to embed:

- ``logfile=None`` did not mean "no file": the logger fell back to a
  relative ``progress.log``, so every process wrote a log file into whatever
  its working directory happened to be, and the stdout output was dead code.
- ``is_interactive()`` looked at ``__main__``, which is not the module the
  author had in mind under ``multiprocessing`` spawn, in an embedded
  interpreter or in a WSGI/ASGI server.
- Nothing reached the standard ``logging`` module, so an application could
  not filter, route or format the library's output.

``logfile=None`` writes no file and prints on stdout, ``is_interactive()``
asks the stream (or an explicit flag), and ``use_stdlib_logging()`` turns
every line into a record on ``logging.getLogger("twin4build")`` without
touching the root logger or ``logging.disable``.
"""

# Standard library imports
import contextlib
import io
import logging
import os
import shutil
import sys
import tempfile
import unittest
import warnings
from unittest import mock

# Local application imports
import twin4build as tb
from twin4build.utils.logger import Logger

tb._IS_TESTING = True


class _Terminal(io.StringIO):
    """A text stream that says it is a terminal."""

    def isatty(self):
        return True


class _ListHandler(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.NOTSET)
        self.records = []

    def emit(self, record):
        self.records.append(record)


class _LoggerTestCase(unittest.TestCase):
    """A fresh logger in an empty working directory."""

    def setUp(self):
        self.cwd = os.getcwd()
        self.tmp = tempfile.mkdtemp()
        os.chdir(self.tmp)

    def tearDown(self):
        os.chdir(self.cwd)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _logger(self):
        logger = Logger()
        logger.verbose = 10
        logger._allow_in_tests = True
        logger._enabled = True
        logger._show_location = False
        return logger

    @staticmethod
    def _tree(logger):
        logger.task("outer")
        logger.add_level()
        logger.iter("child")
        logger.remove_level()
        logger.ok("outer", change_status=True)


class TestLogfileNone(_LoggerTestCase):
    def test_no_file_is_written(self):
        logger = self._logger()
        self.assertIsNone(logger.logfile)
        with contextlib.redirect_stdout(io.StringIO()):
            self._tree(logger)
            logger.reset()
        self.assertIsNone(logger._get_logfile_path())
        self.assertEqual(os.listdir(self.tmp), [])

    def test_stdout_is_the_default_and_prints_a_line_once(self):
        logger = self._logger()
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self._tree(logger)
        self.assertEqual(
            out.getvalue().splitlines(), ["[TASK] outer", "|______[ITER] child"]
        )

    def test_a_reset_starts_the_next_tree(self):
        logger = self._logger()
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self._tree(logger)
            logger.reset()  # what a wrapped component call does at depth zero
            logger.task("next")
        self.assertEqual(
            out.getvalue().splitlines(),
            ["[TASK] outer", "|______[ITER] child", "[TASK] next"],
        )

    def test_a_logfile_that_was_asked_for_is_written_instead(self):
        logger = self._logger()
        logger.logfile = os.path.join(self.tmp, "run.log")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self._tree(logger)
        self.assertEqual(out.getvalue(), "")
        self.assertEqual(os.listdir(self.tmp), ["run.log"])
        with open(logger.logfile, encoding="utf-8") as f:
            self.assertEqual(
                f.read().splitlines(), ["[TASK] outer", "|______[ITER] child"]
            )

    def test_leaving_the_logfile_continues_on_stdout(self):
        logger = self._logger()
        logger.logfile = os.path.join(self.tmp, "run.log")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            logger.task("to the file")
            logger.logfile = None
            logger.task("to stdout")
        self.assertEqual(out.getvalue().splitlines(), ["[TASK] to stdout"])

    def test_an_unwritable_logfile_warns_once(self):
        logger = self._logger()
        logger.logfile = os.path.join(self.tmp, "no_such_directory", "run.log")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            logger.task("first")
            logger.task("second")
        caught = [w for w in caught if issubclass(w.category, RuntimeWarning)]
        self.assertEqual(len(caught), 1)
        self.assertIn("no_such_directory", str(caught[0].message))
        self.assertEqual(os.listdir(self.tmp), [])


class TestIsInteractive(_LoggerTestCase):
    def test_it_asks_stdout(self):
        logger = self._logger()
        with contextlib.redirect_stdout(_Terminal()):
            self.assertTrue(logger.is_interactive())
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertFalse(logger.is_interactive())

    def test_it_does_not_look_at_main(self):
        logger = self._logger()
        main = sys.modules["__main__"]
        with contextlib.redirect_stdout(io.StringIO()):
            with mock.patch.object(main, "__file__", "script.py", create=True):
                self.assertFalse(logger.is_interactive())
                del main.__file__  # an interpreter session, an embedded one
                self.assertFalse(logger.is_interactive())
                main.__file__ = "script.py"  # for the patch to restore

    def test_no_stdout_is_not_interactive(self):
        logger = self._logger()
        with mock.patch.object(sys, "stdout", None):
            self.assertFalse(logger.is_interactive())
            logger.task("nowhere to print, and no error")

    def test_the_explicit_flag_decides(self):
        logger = self._logger()
        self.assertIsNone(logger.interactive)
        with contextlib.redirect_stdout(io.StringIO()):
            logger.interactive = True
            self.assertTrue(logger.is_interactive())
        with contextlib.redirect_stdout(_Terminal()):
            logger.interactive = False
            self.assertFalse(logger.is_interactive())
            logger.interactive = None
            self.assertTrue(logger.is_interactive())

    def test_colors_only_when_interactive(self):
        logger = self._logger()
        plain = io.StringIO()
        with contextlib.redirect_stdout(plain):
            logger.task("plain")
        self.assertEqual(plain.getvalue(), "[TASK] plain\n")

        colored = _Terminal()
        with contextlib.redirect_stdout(colored):
            logger.task("colored")
        self.assertIn("\033[", colored.getvalue())
        self.assertIn("colored", colored.getvalue())


class TestStdlibLogging(_LoggerTestCase):
    NAME = "twin4build"

    def setUp(self):
        super().setUp()
        self.target = logging.getLogger(self.NAME)
        self.handler = _ListHandler()
        self.saved = (self.target.level, self.target.propagate)
        self.target.addHandler(self.handler)
        self.target.setLevel(logging.DEBUG)
        self.target.propagate = False

    def tearDown(self):
        self.target.removeHandler(self.handler)
        self.target.setLevel(self.saved[0])
        self.target.propagate = self.saved[1]
        super().tearDown()

    def _forwarding_logger(self):
        logger = self._logger()
        logger.use_stdlib_logging()
        return logger

    def _messages(self):
        return [r.getMessage() for r in self.handler.records]

    def test_it_is_off_until_asked_for(self):
        logger = self._logger()
        self.assertIsNone(logger.stdlib_logger)
        with contextlib.redirect_stdout(io.StringIO()):
            self._tree(logger)
        self.assertEqual(self.handler.records, [])

    def test_lines_are_records_on_the_twin4build_logger(self):
        logger = self._forwarding_logger()
        self.assertIs(logger.stdlib_logger, logging.getLogger("twin4build"))
        logger.task("outer")
        logger.add_level()
        logger.config("Method: %s", "SLSQP")
        logger.iter("eval=%d | obj=%.2f", 3, 0.5)
        logger.warning("Slow.")
        logger.error("Failed.")
        logger.remove_level()
        logger.info("100% done")

        self.assertEqual(
            [
                (r.name, r.levelname, r.getMessage(), r.t4b_badge, r.t4b_depth)
                for r in self.handler.records
            ],
            [
                ("twin4build", "INFO", "outer", "TASK", 0),
                ("twin4build", "INFO", "Method: SLSQP", "CONFIG", 1),
                ("twin4build", "INFO", "eval=3 | obj=0.50", "ITER", 1),
                ("twin4build", "WARNING", "Slow.", "OUTCOME", 1),
                ("twin4build", "ERROR", "Failed.", "OUTCOME", 1),
                ("twin4build", "INFO", "100% done", "INFO", 0),
            ],
        )
        self.assertTrue(all(r.t4b_updates is None for r in self.handler.records))

    def test_nothing_is_printed_and_no_file_is_written(self):
        logger = self._forwarding_logger()
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self._tree(logger)
        self.assertEqual(out.getvalue(), "")
        self.assertEqual(os.listdir(self.tmp), [])
        self.assertEqual(self._messages(), ["outer", "child", "outer - OK"])

    def test_a_logfile_that_was_asked_for_is_still_written(self):
        logger = self._forwarding_logger()
        logger.logfile = os.path.join(self.tmp, "run.log")
        self._tree(logger)
        with open(logger.logfile, encoding="utf-8") as f:
            self.assertEqual(
                f.read().splitlines(), ["[TASK] outer", "|______[ITER] child"]
            )
        self.assertEqual(self._messages(), ["outer", "child", "outer - OK"])

    def test_a_status_change_is_a_record_that_names_its_line(self):
        logger = self._forwarding_logger()
        logger.task("first")
        logger.task("second")
        logger.task("third")
        logger.ok("first", change_status=True)
        logger.warning("second", change_status=True)
        logger.error("third", change_status=True)
        logger.ok("never logged", change_status=True)

        lines, updates = self.handler.records[:3], self.handler.records[3:]
        self.assertEqual([r.t4b_line for r in lines], [0, 1, 2])
        self.assertEqual(
            [
                (r.levelname, r.getMessage(), r.t4b_badge, r.t4b_updates)
                for r in updates
            ],
            [
                ("INFO", "first - OK", "TASK", 0),
                ("WARNING", "second - WARNING", "TASK", 1),
                ("ERROR", "third - ERROR", "TASK", 2),
            ],
        )

    def test_a_record_names_the_caller(self):
        logger = self._forwarding_logger()
        logger.task("outer")
        logger.ok("outer", change_status=True)
        for record in self.handler.records:
            self.assertEqual(
                os.path.basename(record.pathname), os.path.basename(__file__)
            )
            self.assertEqual(record.funcName, "test_a_record_names_the_caller")
            self.assertGreater(record.lineno, 0)

    def test_the_level_of_the_stdlib_logger_filters(self):
        logger = self._forwarding_logger()
        self.target.setLevel(logging.WARNING)
        logger.task("outer")
        logger.warning("Slow.")
        logger.ok("outer", change_status=True)
        self.assertEqual(self._messages(), ["Slow."])

    def test_status_filters_and_verbose_still_apply(self):
        logger = self._forwarding_logger()
        logger.debug("hidden by default")
        logger.show_status("debug")
        logger.debug("shown")
        logger.hide_status("iter")
        logger.iter("hidden")
        self.assertEqual(
            [(r.levelname, r.getMessage()) for r in self.handler.records],
            [("DEBUG", "shown")],
        )

        logger.verbose = 1
        logger.task("outer")
        logger.add_level()
        logger.info("too deep for verbose=1")
        logger.remove_level()
        self.assertEqual(self._messages(), ["shown", "outer"])

        logger.verbose = 0
        logger.task("silent")
        self.assertEqual(self._messages(), ["shown", "outer"])

    def test_the_application_logging_is_left_alone(self):
        root = logging.getLogger()
        before = (
            list(root.handlers),
            root.level,
            logging.root.manager.disable,
            list(self.target.handlers),
            self.target.level,
            self.target.propagate,
        )
        app = logging.getLogger("test_logger_stdlib.app")
        app.setLevel(logging.INFO)

        logger = self._forwarding_logger()
        self._tree(logger)
        logger.reset()
        logger.use_stdlib_logging(False)

        self.assertEqual(
            before,
            (
                list(root.handlers),
                root.level,
                logging.root.manager.disable,
                list(self.target.handlers),
                self.target.level,
                self.target.propagate,
            ),
        )
        self.assertTrue(app.isEnabledFor(logging.INFO))

    def test_another_logger_by_name_or_instance(self):
        other = logging.getLogger("test_logger_stdlib.other")
        handler = _ListHandler()
        other.addHandler(handler)
        other.setLevel(logging.INFO)
        other.propagate = False
        try:
            logger = self._logger()
            logger.use_stdlib_logging(logger="test_logger_stdlib.other")
            self.assertIs(logger.stdlib_logger, other)
            logger.task("by name")
            logger.use_stdlib_logging(logger=other)
            self.assertIs(logger.stdlib_logger, other)
            logger.task("by instance")
        finally:
            other.removeHandler(handler)
        self.assertEqual(
            [(r.name, r.getMessage()) for r in handler.records],
            [
                ("test_logger_stdlib.other", "by name"),
                ("test_logger_stdlib.other", "by instance"),
            ],
        )
        self.assertEqual(self.handler.records, [])

    def test_turning_it_off_returns_to_stdout(self):
        logger = self._forwarding_logger()
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            logger.task("forwarded")
            logger.use_stdlib_logging(False)
            self.assertIsNone(logger.stdlib_logger)
            logger.task("printed")
        self.assertEqual(self._messages(), ["forwarded"])
        self.assertEqual(out.getvalue().splitlines(), ["[TASK] printed"])

    def test_the_setting_survives_a_reset(self):
        logger = self._forwarding_logger()
        logger.interactive = False
        logger.reset()
        self.assertIs(logger.stdlib_logger, self.target)
        self.assertIs(logger.interactive, False)
        logger.task("after the reset")
        self.assertEqual(self._messages(), ["after the reset"])
        self.assertEqual(self.handler.records[0].t4b_line, 0)


if __name__ == "__main__":
    unittest.main()
