"""Two nodes whose names sanitize to the same component id keep two components.

A singleton component's id is its node's sanitized name, and ``R00.18_VAV02``
(a box) and ``R00_18_VAV02`` (one of its points) both sanitize to
``R00_18_VAV02``.  Within one class the second node used to be folded into
the first node's component; across classes the model refused the second
component.  The first node keeps the id; the other gets a short hash of its
URI appended, the same on every translation of the graph.
"""
import os
import shutil
import unittest

from rdflib import RDF, Literal

import twin4build
import twin4build.core as core
from twin4build.model.semantic_model.semantic_model import SemanticModel
from twin4build.systems.sensor.sensor_system import SensorSystem
from twin4build.translator.translator import Node, SignaturePattern, StepRule, Translator

twin4build._IS_TESTING = True

BRICK = core.namespace.BRICK
REF = core.namespace.BRICKREF
EX = core.namespace.T4B


def leaf_sensor_pattern():
    sensor = Node(cls=BRICK.Point)
    externalref = Node(cls=(REF.ExternalReference, core.BlankNode))
    timeseries_id = Node(cls=core.namespace.XSD.string)
    sp = SignaturePattern(id="test_leaf_sensor", system=SensorSystem)
    sp.add_rule(StepRule(subject=sensor, object=externalref, predicate=REF.hasExternalReference))
    sp.add_rule(StepRule(subject=externalref, object=timeseries_id, predicate=REF.hasTimeseriesId))
    sp.add_parameter("uuid", timeseries_id)
    sp.add_modeled_node(sensor)
    return sp


def graph(model_id):
    sm = SemanticModel(id=model_id)
    g = sm.instance_graph
    for name, uuid in (("R00.18_VAV02", "uuid-box"), ("R00_18_VAV02", "uuid-point")):
        node, ref = EX[name], EX[name + "_ref"]
        g.add((node, RDF.type, BRICK.Return_Air_Flow_Sensor))
        g.add((node, REF.hasExternalReference, ref))
        g.add((ref, RDF.type, REF.ExternalReference))
        g.add((ref, REF.hasTimeseriesId, Literal(uuid)))
    return sm


class TestSanitizedIdCollision(unittest.TestCase):
    MODEL_IDS = ("test_id_collision_a", "test_id_collision_b")

    def tearDown(self):
        for model_id in self.MODEL_IDS:
            path = os.path.join("generated_files", "models", model_id)
            if os.path.exists(path):
                shutil.rmtree(path)

    def test_two_components_with_stable_ids(self):
        ids = []
        for model_id in self.MODEL_IDS:
            model = Translator().translate(graph(model_id), patterns=[leaf_sensor_pattern()], id=model_id)
            self.assertEqual(sorted(c.uuid for c in model.components.values()), ["uuid-box", "uuid-point"])
            ids.append(sorted(model.components))
        self.assertEqual(ids[0], ids[1])  # the same on every translation
        first, second = sorted(ids[0], key=len)
        self.assertTrue(first.endswith("R00_18_VAV02"), first)
        self.assertRegex(second, r"^" + first + r"_[0-9a-f]{8}$")


if __name__ == "__main__":
    unittest.main()
