"""Two set rules on one set node are two claims about the same bundle.

"A feeds all of these terminals, and C feeds all of these too" binds the
terminal bundle once; the bundle is the terminals every claim holds for
(what both units feed), and a unit that feeds none of them prunes the match.  Before, the second
rule restarted the bundle, re-broadcast the downstream rules against the
bindings of the first, and the bundle shrank to whatever survived.
"""

import unittest

from rdflib import RDF, Namespace

import twin4build as tb
import twin4build.core as core
from twin4build.translator.translator import (
    ModeledNode,
    Node,
    SetAnyPathRule,
    SetStepRule,
    SignaturePattern,
    StepRule,
    Translator,
)

BRICK = core.namespace.BRICK
EX = Namespace("http://example.org/unify#")


def pair_pattern(rule_cls):
    """Two units feeding one and the same bundle of terminals."""
    unit_a, unit_b = Node(cls=BRICK.AHU), Node(cls=BRICK.AHU)
    vavs, rooms = Node(cls=BRICK.VAV), Node(cls=BRICK.Room)
    sp = SignaturePattern(id=f"pair_{rule_cls.__name__}", system=tb.AirHandlingUnitSystem)
    sp.add_rule(rule_cls(subject=unit_a, object=vavs, predicate=BRICK.feeds))
    sp.add_rule(rule_cls(subject=unit_b, object=vavs, predicate=BRICK.feeds))
    sp.add_rule(StepRule(subject=vavs, object=rooms, predicate=BRICK.feeds))
    ModeledNode([unit_a, unit_b, vavs])
    return sp, unit_a, unit_b, vavs, rooms


def graph(sm, feeds):
    """``feeds``: {unit name: [terminal names]}; every terminal feeds its room."""
    g = sm.instance_graph
    terminals = sorted({t for ts in feeds.values() for t in ts})
    for t in terminals:
        g.add((EX[t], RDF.type, BRICK.VAV))
        g.add((EX[f"R_{t}"], RDF.type, BRICK.Room))
        g.add((EX[t], BRICK.feeds, EX[f"R_{t}"]))
    for unit, ts in feeds.items():
        g.add((EX[unit], RDF.type, BRICK.AHU))
        for t in ts:
            g.add((EX[unit], BRICK.feeds, EX[t]))


def complete_groups(sp, sm):
    complete, _ = Translator._match_patterns(
        pattern_groups=Translator._group_patterns([sp]), semantic_model=sm
    )
    return [g for groups in complete.get(tb.AirHandlingUnitSystem, {}).values() for g in groups]


def names(binding):
    return sorted(str(x).split("#")[-1] for x in binding)


class TestSetNodeUnification(unittest.TestCase):
    def test_same_bundle_from_two_units_binds_the_whole_bundle(self):
        for rule_cls in (SetStepRule, SetAnyPathRule):
            sp, unit_a, unit_b, vavs, rooms = pair_pattern(rule_cls)
            sm = core.SemanticModel(id=f"unify_same_{rule_cls.__name__}", namespaces={"ex": str(EX)})
            graph(sm, {"AHU0": ["VAV0", "VAV1", "VAV2"], "AHU1": ["VAV0", "VAV1", "VAV2"]})
            groups = complete_groups(sp, sm)
            self.assertTrue(groups, rule_cls.__name__)
            for group in groups:
                self.assertEqual(names(group[vavs]), ["VAV0", "VAV1", "VAV2"], rule_cls.__name__)
                self.assertEqual(names(group[rooms]), ["R_VAV0", "R_VAV1", "R_VAV2"], rule_cls.__name__)
                self.assertNotEqual(group[unit_a], group[unit_b], rule_cls.__name__)

    def test_the_bundle_is_what_both_units_feed(self):
        """AHU0 feeds three terminals, AHU1 two of them: the bundle both
        claims hold for is those two.  A bundle's elements are the ones
        every claim about the bundle holds for, so the third terminal drops
        out instead of the match failing.  The two readings (which unit is
        first) are the same match and collapse to one component."""
        sp, unit_a, unit_b, vavs, rooms = pair_pattern(SetStepRule)
        sm = core.SemanticModel(id="unify_differ", namespaces={"ex": str(EX)})
        graph(sm, {"AHU0": ["VAV0", "VAV1", "VAV2"], "AHU1": ["VAV0", "VAV1"]})
        groups = complete_groups(sp, sm)
        self.assertTrue(groups)
        for group in groups:
            self.assertEqual(names(group[vavs]), ["VAV0", "VAV1"])
            self.assertEqual(names(group[rooms]), ["R_VAV0", "R_VAV1"])
            self.assertEqual(names([group[unit_a], group[unit_b]]), ["AHU0", "AHU1"])

    def test_a_single_unit_is_not_a_pair(self):
        sp, *_ = pair_pattern(SetStepRule)
        sm = core.SemanticModel(id="unify_single", namespaces={"ex": str(EX)})
        graph(sm, {"AHU0": ["VAV0", "VAV1"]})
        self.assertEqual(complete_groups(sp, sm), [])


if __name__ == "__main__":
    unittest.main()
