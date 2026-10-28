"""A path rule takes its predicate as a URI, a tuple of URIs or a ``Predicate``,
without a warning: a bare URI is the form the docstrings and the example
patterns use (it warned "deprecated" on every one of them)."""
import unittest
import warnings

import twin4build as tb
import twin4build.core as core
from twin4build.translator.translator import AnyPathRule, Node, PathRule, Predicate, SetAnyPathRule

tb._IS_TESTING = True

BRICK = core.namespace.BRICK
FSO = core.namespace.FSO


class TestPathRulePredicates(unittest.TestCase):
    def test_every_form_builds_without_a_warning(self):
        forms = (BRICK.feeds, (BRICK.feeds, FSO.feedsFluidTo), Predicate(BRICK.feeds))
        for rule_cls in (PathRule, AnyPathRule, SetAnyPathRule):
            for predicate in forms:
                with self.subTest(rule=rule_cls.__name__, predicate=predicate), warnings.catch_warnings():
                    warnings.simplefilter("error")
                    rule = rule_cls(subject=Node(cls=BRICK.AHU), object=Node(cls=BRICK.VAV), predicate=predicate)
                self.assertIsNotNone(rule)


if __name__ == "__main__":
    unittest.main()
