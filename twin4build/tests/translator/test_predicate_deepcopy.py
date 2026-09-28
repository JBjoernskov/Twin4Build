"""A signature pattern survives ``copy.deepcopy``.

``Predicate`` compares and hashes by the predicate URIs it names (#211).
Rebuilt through the copy protocol inside a cyclic structure (a pattern's
rules reference their predicates, which reference the pattern's nodes), a
half-built instance was hashed as a dict key before its state was set and
failed on the missing ``preds``: ``copy.deepcopy`` of a translated
template (``benchmarks/common.py``) broke.  A Predicate is an immutable
value, so a deep copy shares it, and an uninitialised one hashes by
identity.
"""

# Standard library imports
import copy
import unittest

# Local application imports
import twin4build.core as core
from twin4build.systems.sensor.sensor_system import SensorSystem
from twin4build.translator.translator import (
    Node,
    Predicate,
    SignaturePattern,
    StepRule,
)


class TestPredicateDeepcopy(unittest.TestCase):
    def test_pattern_deepcopy_shares_its_predicates(self):
        BRICK = core.namespace.BRICK
        room = Node(cls=BRICK.Room)
        sensor = Node(cls=BRICK.Zone_Air_Temperature_Sensor)
        sp = SignaturePattern(id="deepcopy_pattern", system=SensorSystem)
        rule = StepRule(subject=room, object=sensor, predicate=BRICK.hasPoint)
        sp.add_rule(rule)
        sp.add_modeled_node(sensor)
        predicate = rule.predicate
        self.assertIsInstance(predicate, Predicate)
        # a cyclic, predicate-keyed structure like the ones a translated model holds
        holder = {"pattern": sp, "by_predicate": {predicate: rule}}
        rule.back = holder

        copied = copy.deepcopy(holder)

        predicates = list(copied["by_predicate"])
        self.assertEqual(len(predicates), 1)
        self.assertEqual(predicates[0], predicate)
        self.assertIs(predicates[0], predicate)  # shared value object
        self.assertIsNot(copied["pattern"], sp)

    def test_uninitialised_predicate_hashes_by_identity(self):
        bare = Predicate.__new__(Predicate)
        self.assertEqual(hash(bare), hash(("id", id(bare))))
        self.assertNotEqual(bare, Predicate.__new__(Predicate))


if __name__ == "__main__":
    unittest.main()
