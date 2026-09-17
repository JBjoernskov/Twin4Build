"""Patterns are an explicit translator input (#200).

- ``twin4build.patterns.default_patterns()`` is bound and non-empty.
- Translating with the explicit example set produces the same components as
  the deprecated implicit path.
- An unbound pattern is rejected with a clear message; ``systems=`` is an
  allow-list; ``bind`` is chainable; no ``System`` carries a pattern list.
"""

import unittest
import warnings

import twin4build as tb
from twin4build.examples import patterns as example_patterns
import twin4build.core as core
from twin4build.tests.translator.test_brick14_bms_patterns import build_graph
from twin4build.translator.translator import SignaturePattern, Translator


class TestExplicitPatterns(unittest.TestCase):
    def test_example_set_is_bound_and_grouped(self):
        ps = example_patterns.default_patterns()
        self.assertGreater(len(ps), 30)
        self.assertTrue(all(sp.system is not None for sp in ps))
        groups = Translator._group_patterns(ps)
        self.assertIn(tb.SensorSystem, groups)
        self.assertIn(tb.BuildingSpaceSystem, groups)
        self.assertEqual(sum(len(v) for v in groups.values()), len(ps))

    def test_no_system_carries_patterns(self):
        self.assertFalse(hasattr(tb.System, "sp"))
        self.assertFalse(hasattr(tb.System, "add_signature_pattern"))
        for cls in (tb.SensorSystem, tb.BuildingSpaceSystem, tb.SpaceHeaterSystem, tb.ScheduleSystem):
            self.assertFalse(hasattr(cls, "sp"), cls)

    def test_unbound_pattern_is_rejected(self):
        sp = SignaturePattern(id="unbound")
        with self.assertRaisesRegex(ValueError, "not bound"):
            Translator._group_patterns([sp])
        self.assertIs(sp.bind(tb.SensorSystem), sp)
        self.assertIs(Translator._group_patterns([sp])[tb.SensorSystem][0], sp)
        sp2 = SignaturePattern(id="bound_at_construction", system=tb.DamperSystem)
        self.assertIs(sp2.system, tb.DamperSystem)

    def test_systems_is_an_allow_list(self):
        ps = example_patterns.default_patterns()
        groups = Translator._group_patterns(ps, systems=[tb.SensorSystem])
        self.assertEqual(set(groups), {tb.SensorSystem})

    def test_explicit_equals_implicit_translation(self):
        sm1 = core.SemanticModel(id="explicit_patterns_a")
        build_graph(sm1)
        explicit = Translator().translate(
            sm1, patterns=example_patterns.default_patterns(), id="explicit_patterns_a"
        )
        sm2 = core.SemanticModel(id="explicit_patterns_b")
        build_graph(sm2)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            implicit = Translator().translate(sm2, id="explicit_patterns_b")
        self.assertTrue(any("patterns=None" in str(w.message) for w in caught), "deprecation warning expected")
        # Blank-node ids differ between translations (#186): mask them.
        import re
        ids = lambda m: sorted(
            re.sub(r"\[N[0-9a-f]{32}\]", "[bnode]", type(c).__name__ + ":" + str(c.id))
            for c in m.components.values()
        )
        self.assertEqual(ids(explicit), ids(implicit))
        self.assertGreater(len(explicit.components), 5)


if __name__ == "__main__":
    unittest.main()
