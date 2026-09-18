"""``NoStepRule`` with an unbound node means "no node OTHER than the ones
this match binds".

A pattern node is one node of the graph; an unbound node of a negation
therefore ranges over nodes the match has not already bound.  "An air
handler feeds this terminal, and no other air handler does" is the shape
that needs it: the unit bound by the positive rule must not trip its own
veto.  Before this semantics the negation vetoed the bound unit too and the
pattern never matched at all.
"""

import unittest

from rdflib import RDF, Namespace

import twin4build as tb
import twin4build.core as core
from twin4build.translator.translator import (
    NoStepRule,
    Node,
    SetStepRule,
    SignaturePattern,
    StepRule,
    Translator,
)

BRICK = core.namespace.BRICK
EX = Namespace("http://example.org/no_other#")


def single_tree_pattern(set_bound_terminals: bool):
    """An AHU feeding terminals that no other AHU feeds."""
    ahu = Node(cls=BRICK.AHU)
    other = Node(cls=BRICK.AHU)
    vav = Node(cls=BRICK.VAV)
    room = Node(cls=BRICK.Room)
    sp = SignaturePattern(id=f"single_tree_{set_bound_terminals}", system=tb.AirHandlingUnitSystem)
    if set_bound_terminals:
        sp.add_rule(SetStepRule(subject=ahu, object=vav, predicate=BRICK.feeds))
    else:
        sp.add_rule(StepRule(subject=ahu, object=vav, predicate=BRICK.feeds))
    sp.add_rule(StepRule(subject=vav, object=room, predicate=BRICK.feeds))
    sp.add_rule(NoStepRule(subject=other, object=vav, predicate=BRICK.feeds))
    sp.add_modeled_node(ahu)
    return sp


def graph(sm, n_units: int, n_terminals: int = 2):
    g = sm.instance_graph
    for i in range(n_units):
        g.add((EX[f"AHU{i}"], RDF.type, BRICK.AHU))
    for j in range(n_terminals):
        g.add((EX[f"VAV{j}"], RDF.type, BRICK.VAV))
        g.add((EX[f"VAV{j}"], BRICK.feeds, EX[f"R{j}"]))
        g.add((EX[f"R{j}"], RDF.type, BRICK.Room))
        for i in range(n_units):
            g.add((EX[f"AHU{i}"], BRICK.feeds, EX[f"VAV{j}"]))


def matches(sp, sm):
    complete, _ = Translator._match_patterns(
        pattern_groups=Translator._group_patterns([sp]), semantic_model=sm
    )
    return sum(len(v) for v in complete.get(tb.AirHandlingUnitSystem, {}).values())


class TestNoOther(unittest.TestCase):
    def test_one_unit_matches(self):
        for set_bound in (False, True):
            sm = core.SemanticModel(id=f"no_other_one_{set_bound}", namespaces={"ex": str(EX)})
            graph(sm, n_units=1)
            self.assertGreater(matches(single_tree_pattern(set_bound), sm), 0, set_bound)

    def test_two_units_on_the_same_terminals_do_not_match(self):
        for set_bound in (False, True):
            sm = core.SemanticModel(id=f"no_other_two_{set_bound}", namespaces={"ex": str(EX)})
            graph(sm, n_units=2)
            self.assertEqual(matches(single_tree_pattern(set_bound), sm), 0, set_bound)

    def test_plain_negation_is_unchanged(self):
        """A negation whose forbidden class has no positive counterpart on
        the same edge still vetoes any such neighbour (a VAV with a reheat
        coil is not a no-reheat VAV)."""
        vav = Node(cls=BRICK.VAV)
        room = Node(cls=BRICK.Room)
        sp = SignaturePattern(id="no_reheat", system=tb.DamperSystem)
        sp.add_rule(StepRule(subject=vav, object=room, predicate=BRICK.feeds))
        sp.add_rule(NoStepRule(subject=vav, object=Node(cls=BRICK.Heating_Coil), predicate=BRICK.hasPart))
        sp.add_modeled_node(vav)
        for with_coil, expected in ((False, 1), (True, 0)):
            sm = core.SemanticModel(id=f"no_reheat_{with_coil}", namespaces={"ex": str(EX)})
            g = sm.instance_graph
            g.add((EX.VAV, RDF.type, BRICK.VAV))
            g.add((EX.VAV, BRICK.feeds, EX.R))
            g.add((EX.R, RDF.type, BRICK.Room))
            if with_coil:
                g.add((EX.VAV, BRICK.hasPart, EX.COIL))
                g.add((EX.COIL, RDF.type, BRICK.Heating_Coil))
            complete, _ = Translator._match_patterns(
                pattern_groups=Translator._group_patterns([sp]), semantic_model=sm
            )
            n = sum(len(v) for v in complete.get(tb.DamperSystem, {}).values())
            self.assertEqual(n, expected, with_coil)


if __name__ == "__main__":
    unittest.main()
