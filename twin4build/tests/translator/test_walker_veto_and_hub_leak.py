"""Two matcher regressions found on the Hoeje-Taastrup Raadhus graph.

1. A stand-alone :class:`NoStepRule` (not ``&``-composed with a positive
   rule) pruned every branch: with the predicate absent the walker hit the
   "missing predicate" prune, with the predicate present but no forbidden
   neighbour it hit the "no pair matched" prune.  The veto only ever worked
   inside ``AnyPathRule & NoStepRule``.

2. Bindings leaked between sibling matches through a shared hub: seeded at
   room R01 the walker crossed ``R01 -> VAV -> AHU -> VAV -> R02`` and
   carried R02's downstream literals (``brick:volume`` value) back into
   R01's map, producing one component per room *per* literal.
"""

# Standard library imports
import unittest

# Third party imports
from rdflib import RDF, BNode, Literal, Namespace

# Local application imports
import twin4build
import twin4build.core as core
from twin4build.model.semantic_model.semantic_model import SemanticModel
from twin4build.systems.building_space.building_space_system import (
    BuildingSpaceSystem,
)
from twin4build.translator.translator import (
    Node,
    NoStepRule,
    Predicate,
    SignaturePattern,
    StepRule,
    Translator,
)

twin4build._IS_TESTING = True

BRICK = core.namespace.BRICK
REC = core.namespace.REC
EX = Namespace("http://example.org/walker#")


def _graph(sm, reheat_room=None):
    g = sm.instance_graph
    g.add((EX.AHU01, RDF.type, BRICK.AHU))
    for i in (1, 2):
        room, vav = EX[f"R0{i}"], EX[f"R0{i}_VAV"]
        g.add((room, RDF.type, REC.Room))
        g.add((vav, RDF.type, BRICK.VAV))
        g.add((EX.AHU01, BRICK.feeds, vav))
        g.add((vav, BRICK.feeds, room))
        g.add((vav, BRICK.hasPoint, EX[f"R0{i}_CMD"]))
        g.add((EX[f"R0{i}_CMD"], RDF.type, BRICK.Damper_Position_Command))
        vol = BNode()
        g.add((room, BRICK.volume, vol))
        g.add((vol, BRICK.value, Literal(20.0 + i)))
        if reheat_room == i:
            g.add((vav, BRICK.hasPoint, EX[f"R0{i}_REHEAT"]))
            g.add((EX[f"R0{i}_REHEAT"], RDF.type, BRICK.Reheat_Command))


def _pattern(veto_part=False, veto_point=False, with_volume=False):
    ahu = Node(cls=BRICK.AHU)
    vav = Node(cls=BRICK.VAV)
    space = Node(cls=(REC.Room,))
    feeds = Predicate((BRICK.feeds,))
    sp = SignaturePattern(id=f"walker_probe_{veto_part}_{veto_point}_{with_volume}")
    sp.add_rule(StepRule(subject=ahu, object=vav, predicate=feeds))
    sp.add_rule(StepRule(subject=vav, object=space, predicate=feeds))
    if veto_part:
        sp.add_rule(
            NoStepRule(
                subject=vav,
                object=Node(cls=(BRICK.Heating_Coil, BRICK.Cooling_Coil)),
                predicate=BRICK.hasPart,
            )
        )
    if veto_point:
        sp.add_rule(
            NoStepRule(
                subject=vav,
                object=Node(cls=(BRICK.Reheat_Command, BRICK.Valve_Command)),
                predicate=BRICK.hasPoint,
            )
        )
    if with_volume:
        vol = Node(cls=(core.BlankNode,))
        val = Node(cls=(core.namespace.XSD.float, core.namespace.XSD.double, core.namespace.XSD.decimal))
        sp.add_rule(StepRule(subject=space, object=vol, predicate=BRICK.volume))
        sp.add_rule(StepRule(subject=vol, object=val, predicate=BRICK.value))
        sp.add_parameter("mass.V", val)
        sp.add_modeled_node(vol)
    sp.add_connection(ahu, "supplyAirTemperature", "supplyAirTemperature")
    sp.add_modeled_node(space)
    return sp, space


class TestStandaloneNoStepRule(unittest.TestCase):
    def _complete(self, sp, sm):
        complete, _ = Translator._match_patterns([BuildingSpaceSystem], sm)
        return complete[BuildingSpaceSystem].get(sp, [])

    def setUp(self):
        self._saved_sp = list(BuildingSpaceSystem.sp)

    def tearDown(self):
        BuildingSpaceSystem.sp = self._saved_sp

    def test_veto_passes_when_predicate_absent(self):
        sm = SemanticModel(id="veto_absent", namespaces={"ex": str(EX)})
        _graph(sm)
        sp, _ = _pattern(veto_part=True)  # no VAV has hasPart at all
        BuildingSpaceSystem.sp = [sp]
        self.assertEqual(len(self._complete(sp, sm)), 2)

    def test_veto_passes_when_only_allowed_neighbours(self):
        sm = SemanticModel(id="veto_allowed", namespaces={"ex": str(EX)})
        _graph(sm)
        sp, _ = _pattern(veto_point=True)  # hasPoint exists, but no reheat cmd
        BuildingSpaceSystem.sp = [sp]
        self.assertEqual(len(self._complete(sp, sm)), 2)

    def test_veto_prunes_forbidden_neighbour(self):
        sm = SemanticModel(id="veto_fires", namespaces={"ex": str(EX)})
        _graph(sm, reheat_room=2)
        sp, space = _pattern(veto_point=True)
        BuildingSpaceSystem.sp = [sp]
        groups = self._complete(sp, sm)
        self.assertEqual([str(g[space].uri).split("#")[-1] for g in groups], ["R01"])


class TestHubLeak(unittest.TestCase):
    def setUp(self):
        self._saved_sp = list(BuildingSpaceSystem.sp)

    def tearDown(self):
        BuildingSpaceSystem.sp = self._saved_sp

    def test_literal_stays_with_its_room(self):
        sm = SemanticModel(id="hub_leak", namespaces={"ex": str(EX)})
        _graph(sm)
        sp, space = _pattern(with_volume=True)
        BuildingSpaceSystem.sp = [sp]
        complete, _ = Translator._match_patterns([BuildingSpaceSystem], sm)
        groups = complete[BuildingSpaceSystem][sp]
        # One match per room, each carrying its own volume literal.
        self.assertEqual(len(groups), 2)
        val_node = sp.parameters["mass.V"]
        bound = {str(g[space].uri).split("#")[-1]: float(g[val_node].uri) for g in groups}
        self.assertEqual(bound, {"R01": 21.0, "R02": 22.0})


if __name__ == "__main__":
    unittest.main()
