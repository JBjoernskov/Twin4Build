"""The logger's level stack survives an unbalanced remove.

``@reset_print`` (applied to every component method) resets the logger when
its call depth returns to zero.  A task that holds open levels around such a
call -- ``Estimator.estimate`` around ``model.initialize`` -- then closes them
against a stack that was already reset to its base.  The base entry is never
popped and the closing removes do not raise (``estimator_example.ipynb`` died
with ``IndexError: list index out of range`` in ``remove_level`` once the
identifiability report printed its lines after the reset).
"""
import unittest

import twin4build as tb
from twin4build.utils.logger import Logger

tb._IS_TESTING = True


class TestLevelStackUnderflow(unittest.TestCase):
    def _logger(self):
        logger = Logger()
        logger.verbose = 10
        logger._allow_in_tests = True
        logger._enabled = True
        return logger

    def test_removes_after_a_reset_never_pop_the_base(self):
        logger = self._logger()
        logger.task("outer")
        logger.add_level()
        logger.task("inner")
        logger.add_level()
        self.assertEqual(len(logger.level_stack), 3)
        logger.reset()  # what a wrapped component call does at depth zero
        self.assertEqual(logger.level_stack, [0])
        logger.task("report")
        logger.add_level()
        logger.iter("a line at the deeper level")
        logger.remove_level()  # the report's own close
        logger.remove_level()  # the inner task's close: already gone
        logger.remove_level()  # the outer task's close: already gone
        self.assertEqual(logger.level_stack, [0])
        self.assertEqual(logger._current_level_indent, 0)
        logger.ok("outer", change_status=True, ignore_no_match=True)
        logger.remove_level()  # one more never raises
        self.assertEqual(logger.level_stack, [0])

    def test_balanced_levels_still_pop(self):
        logger = self._logger()
        logger.task("outer")
        logger.add_level()
        logger.iter("child")
        logger.add_level()
        logger.iter("grandchild")
        logger.remove_level()
        self.assertEqual(len(logger.level_stack), 2)
        logger.remove_level()
        self.assertEqual(logger.level_stack, [0])


if __name__ == "__main__":
    unittest.main()
