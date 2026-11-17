"""A list sender whose slots come from another list of the same match.

An air handler's pattern binds every VAV it feeds (``vavs``) and, by
``StepRule(subject=vavs, object=spaces)``, the room of each VAV
(``spaces``, one entry per VAV, in the same order).  Indexing the rooms'
temperatures by ``vavs`` puts each room on the slots of its own VAVs, so the
temperature input lines up slot by slot with the damper input, one slot per
branch.  The translator used to give every room every VAV's slot; it now
reads the pairing, and raises for two lists the pattern does not pair.
"""

# Standard library imports
import unittest

# Third party imports
from rdflib import RDF, Namespace

# Local application imports
import twin4build
import twin4build.core as core
import twin4build.utils.types as tps
from twin4build.model.semantic_model.semantic_model import SemanticModel
from twin4build.translator.translator import (
    Node,
    SetAnyPathRule,
    SetStepRule,
    SignaturePattern,
    StepRule,
    Translator,
)

twin4build._IS_TESTING = True

BRICK = core.namespace.BRICK
REC = core.namespace.REC
EX = Namespace("http://example.org/paired_slot_lists#")

# R1 and R3 are served by two VAVs each, R2 by one.
VAVS_OF = {"R1": ("V1a", "V1b"), "R2": ("V2",), "R3": ("V3a", "V3b")}


class _Leaf(core.System):
    def __init__(self, inputs=None, outputs=None, **kwargs):
        super().__init__(**kwargs)
        self.input = inputs or {}
        self.output = outputs or {}
        self._config = {"parameters": []}

    @property
    def config(self):
        return self._config


class Unit(_Leaf):
    def __init__(self, **kwargs):
        super().__init__(
            inputs={"damperPosition": tps.Vector(), "exhaustTemperature": tps.Vector()}, **kwargs
        )


class Terminal(_Leaf):
    def __init__(self, **kwargs):
        super().__init__(outputs={"position": tps.Scalar()}, **kwargs)


class Room(_Leaf):
    def __init__(self, **kwargs):
        super().__init__(outputs={"temperature": tps.Scalar()}, **kwargs)


def _unit_pattern(paired=True):
    ahu = Node(cls=BRICK.AHU)
    vavs = Node(cls=BRICK.VAV)
    spaces = Node(cls=REC.Room)
    sp = SignaturePattern(id=f"paired_slot_unit_{paired}", system=Unit)
    sp.add_rule(SetStepRule(subject=ahu, object=vavs, predicate=BRICK.feeds))
    if paired:
        # the room of each VAV
        sp.add_rule(StepRule(subject=vavs, object=spaces, predicate=BRICK.feeds))
    else:
        # every room the unit reaches, a list of its own
        sp.add_rule(SetAnyPathRule(subject=ahu, object=spaces, predicate=BRICK.feeds))
    sp.add_connection(vavs, "position", "damperPosition", input_port_index=vavs)
    sp.add_connection(spaces, "temperature", "exhaustTemperature", input_port_index=vavs)
    sp.add_modeled_node(ahu)
    return sp, vavs, spaces


def _modeled(cls, system, id_):
    node = Node(cls=cls)
    sp = SignaturePattern(id=id_, system=system)
    sp.add_modeled_node(node)
    return sp


def _graph(model_id):
    sm = SemanticModel(id=model_id, namespaces={"ex": str(EX)})
    g = sm.instance_graph
    g.add((EX.AHU, RDF.type, BRICK.AHU))
    for room, vavs in VAVS_OF.items():
        g.add((EX[room], RDF.type, REC.Room))
        for vav in vavs:
            g.add((EX[vav], RDF.type, BRICK.VAV))
            g.add((EX.AHU, BRICK.feeds, EX[vav]))
            g.add((EX[vav], BRICK.feeds, EX[room]))
    return sm


def _translate(model_id, paired=True):
    patterns = [
        _unit_pattern(paired)[0],
        _modeled(BRICK.VAV, Terminal, "paired_slot_terminal"),
        _modeled(REC.Room, Room, "paired_slot_room"),
    ]
    return Translator().translate(_graph(model_id), patterns=patterns, id=model_id)


def _slots(component, port):
    """``{sender id: [input slots]}``."""
    out = {}
    for cp in component.connects_at:
        if cp.input_port != port:
            continue
        for conn in cp.connects_system_through:
            v = cp.input_port_index[conn]
            slots = v.reshape(-1).tolist() if hasattr(v, "reshape") else [int(v)]
            out.setdefault(conn.connects_system.id, []).extend(slots)
    return {k: sorted(v) for k, v in out.items()}


class TestPairedSlotLists(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = _translate("paired_slot_lists")
        (cls.unit,) = [c for c in cls.model.components.values() if isinstance(c, Unit)]

    def test_each_room_sits_on_the_slots_of_its_own_vavs(self):
        dampers = _slots(self.unit, "damperPosition")  # one slot per VAV
        temperatures = _slots(self.unit, "exhaustTemperature")
        self.assertEqual(sorted(s for v in dampers.values() for s in v), list(range(5)))
        slot_of_vav = {vav: slots[0] for vav, slots in dampers.items()}
        expected = {room: sorted(slot_of_vav[v] for v in vavs) for room, vavs in VAVS_OF.items()}
        self.assertEqual(temperatures, expected)

    def test_a_room_on_several_slots_is_one_connection(self):
        (cp,) = [cp for cp in self.unit.connects_at if cp.input_port == "exhaustTemperature"]
        per_sender = {conn.connects_system.id: conn for conn in cp.connects_system_through}
        self.assertEqual(sorted(per_sender), ["R1", "R2", "R3"])
        self.assertEqual(len(cp.input_port_index[per_sender["R1"]].reshape(-1)), 2)
        self.assertIsNone(cp.output_port_index[per_sender["R1"]])
        self.assertEqual(self.unit.get_n_v_from_connections("exhaustTemperature"), 5)

    def test_the_pattern_pairs_lists_filled_from_one_set_rule(self):
        sp, vavs, spaces = _unit_pattern(paired=True)
        self.assertTrue(sp._paired(vavs, spaces))
        sp, vavs, spaces = _unit_pattern(paired=False)
        self.assertFalse(sp._paired(vavs, spaces))

    def test_lists_the_pattern_does_not_pair_raise(self):
        with self.assertRaisesRegex(ValueError, "do not pair"):
            _translate("paired_slot_lists_unpaired", paired=False)


if __name__ == "__main__":
    unittest.main()
