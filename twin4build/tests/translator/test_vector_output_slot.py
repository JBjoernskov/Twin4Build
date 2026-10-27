"""A sender's Vector output, indexed by another node of the receiver's pattern.

``sp.add_connection(boxes, "sourceFlow", "exhaust", output_port_index=space,
input_port_index=boxes)`` in a room's pattern: ``boxes`` (a set, the
senders) indexes the room's input, ``space`` (the room the pattern models)
indexes the box's output, whose slots are the box's rooms in the box's own
pattern.  The output slot was resolved by looking up the sender in its own
groups, so every room read slot 0: each room of a box with several rooms
drew the first room's share.
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
    SetStepRule,
    SignaturePattern,
    StepRule,
    Translator,
)

twin4build._IS_TESTING = True

BRICK = core.namespace.BRICK
REC = core.namespace.REC
EX = Namespace("http://example.org/vector_slot#")


class Box(core.System):
    """An extract box: a temperature per room it serves in, a flow per room out."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input = {"sourceTemperature": tps.Vector()}
        self.output = {"sourceFlow": tps.Vector()}
        self._config = {"parameters": []}

    @property
    def config(self):
        return self._config


class Room(core.System):
    """A room: an exhaust slot per box drawing on it."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input = {"exhaust": tps.Vector()}
        self.output = {"temperature": tps.Scalar()}
        self._config = {"parameters": []}

    @property
    def config(self):
        return self._config


def _box_pattern():
    box = Node(cls=BRICK.VAV)
    room = Node(cls=REC.Room)
    sp = SignaturePattern(id="vector_slot_box", system=Box)
    sp.add_rule(StepRule(subject=room, object=box, predicate=BRICK.feeds))
    sp.add_connection(room, "temperature", "sourceTemperature", input_port_index=room)
    sp.add_modeled_node(box)
    return sp


def _room_pattern():
    space = Node(cls=REC.Room)
    boxes = Node(cls=BRICK.VAV)
    sp = SignaturePattern(id="vector_slot_room", system=Room)
    sp.add_rule(SetStepRule(subject=space, object=boxes, predicate=BRICK.feeds))
    sp.add_connection(boxes, "sourceFlow", "exhaust", output_port_index=space, input_port_index=boxes)
    sp.add_modeled_node(space)
    return sp


def _senders(component, port):
    """``{input slot: (sender id, output slot)}``."""
    out = {}
    for cp in component.connects_at:
        if cp.input_port != port:
            continue
        for conn in cp.connects_system_through:
            in_v, out_v = cp.input_port_index.get(conn), cp.output_port_index.get(conn)
            out[0 if in_v is None else int(in_v)] = (conn.connects_system.id, None if out_v is None else int(out_v))
    return out


class TestVectorOutputSlot(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # FA extracts three rooms, FB two of them: each room has a slot per box, each box a slot per room
        sm = SemanticModel(id="vector_output_slot", namespaces={"ex": str(EX)})
        g = sm.instance_graph
        for box in ("FA", "FB"):
            g.add((EX[box], RDF.type, BRICK.VAV))
        for room, boxes in (("R1", ("FA", "FB")), ("R2", ("FA",)), ("R3", ("FA", "FB"))):
            g.add((EX[room], RDF.type, REC.Room))
            for box in boxes:
                g.add((EX[room], BRICK.feeds, EX[box]))
        cls.model = Translator().translate(sm, patterns=[_box_pattern(), _room_pattern()], id="vector_output_slot")
        cls.by_id = {c.id: c for c in cls.model.components.values()}

    def test_each_room_reads_its_own_slot_of_each_box(self):
        checked = 0
        for box_id in ("FA", "FB"):
            slot_of = {room: k for k, (room, _) in _senders(self.by_id[box_id], "sourceTemperature").items()}
            self.assertEqual(len(slot_of), 3 if box_id == "FA" else 2, slot_of)
            for room_id, slot in slot_of.items():
                draws = [v for v in _senders(self.by_id[room_id], "exhaust").values() if v[0] == box_id]
                self.assertEqual(draws, [(box_id, slot)], f"{room_id} on {box_id}")
                checked += 1
        self.assertEqual(checked, 5)

    def test_each_box_has_its_own_slot_in_the_room(self):
        self.assertEqual(sorted(_senders(self.by_id["R1"], "exhaust")), [0, 1])
        self.assertEqual(sorted(_senders(self.by_id["R2"], "exhaust")), [0])


if __name__ == "__main__":
    unittest.main()
