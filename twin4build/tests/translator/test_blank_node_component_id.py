"""A component modeled on a group with a blank node keeps its id across parses.

rdflib labels a blank node afresh on every parse, and the translator put
that label into the component's id (``[n831375d0...][R08_01]``) and into
the group's relational fingerprint.  An estimation result saved from one
translation could then not be loaded onto the next (#186).  The blank
node carries no identity of its own: the member it hangs off names the
component, and the ``(subject, predicate)`` that reaches it is what the
fingerprint keeps.
"""

# Standard library imports
import os
import re
import shutil
import unittest

# Third party imports
from rdflib import RDF, BNode, Literal

# Local application imports
import twin4build
import twin4build.core as core
from twin4build.model.semantic_model.semantic_model import SemanticModel
from twin4build.systems.sensor.sensor_system import SensorSystem
from twin4build.translator.translator import (
    ModeledNode,
    Node,
    SignaturePattern,
    StepRule,
    Translator,
)

twin4build._IS_TESTING = True

BRICK = core.namespace.BRICK
REF = core.namespace.BRICKREF
EX = core.namespace.T4B
BLANK_LABEL = re.compile(r"\[[Nn][0-9a-f]{20,}")


def sensor_with_reference_pattern():
    """A sensor and the blank node holding its timeseries id, modeled
    together (the Brick 1.4 ``[ brick:value x ]`` idiom binds a blank node
    the same way)."""
    sensor = Node(cls=BRICK.Point)
    externalref = Node(cls=(REF.ExternalReference, core.BlankNode))
    timeseries_id = Node(cls=core.namespace.XSD.string)
    sp = SignaturePattern(id="test_sensor_with_reference", system=SensorSystem)
    sp.add_rule(StepRule(subject=sensor, object=externalref, predicate=REF.hasExternalReference))
    sp.add_rule(StepRule(subject=externalref, object=timeseries_id, predicate=REF.hasTimeseriesId))
    sp.add_parameter("uuid", timeseries_id)
    ModeledNode([sensor, externalref])
    return sp


def graph(model_id):
    """The same three sensors, fresh blank nodes each time (as a re-parse gives)."""
    sm = SemanticModel(id=model_id)
    g = sm.instance_graph
    for i in range(3):
        sensor, ref = EX[f"zone_temp_{i}"], BNode()
        g.add((sensor, RDF.type, BRICK.Zone_Air_Temperature_Sensor))
        g.add((sensor, REF.hasExternalReference, ref))
        g.add((ref, RDF.type, REF.ExternalReference))
        g.add((ref, REF.hasTimeseriesId, Literal(f"uuid-{i}")))
    return sm


class TestBlankNodeComponentId(unittest.TestCase):
    MODEL_IDS = ("test_blank_node_id_a", "test_blank_node_id_b")

    def tearDown(self):
        for model_id in self.MODEL_IDS:
            path = os.path.join("generated_files", "models", model_id)
            if os.path.exists(path):
                shutil.rmtree(path)

    def test_ids_survive_a_reparse(self):
        ids = []
        for model_id in self.MODEL_IDS:
            model = Translator().translate(
                graph(model_id), patterns=[sensor_with_reference_pattern()], id=model_id
            )
            self.assertEqual(
                sorted(c.uuid for c in model.components.values()),
                ["uuid-0", "uuid-1", "uuid-2"],
            )
            ids.append(sorted(model.components))
        self.assertEqual(ids[0], ids[1])
        for id_ in ids[0]:
            self.assertIsNone(BLANK_LABEL.search(id_), id_)
            self.assertIn("zone_temp_", id_)
        self.assertEqual(len(set(ids[0])), 3)


if __name__ == "__main__":
    unittest.main()
