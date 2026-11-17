"""A model knows which semantic nodes its components model, also after
``Model.serialize()`` -> ``Model.load(filename=...)``.

The translation links the two halves of a model in the translator's maps
(``sim2sem_map``, ``sem2sim_map``).  They lived on the translator only, so a
model rebuilt from its serialized file had lost them: nothing could say which
semantic node a restored component stood for.  ``Model.sim2sem_map`` and
``Model.sem2sim_map`` answer on both, and ``serialize`` writes the link with
the simulation model (``t4b:modelsSemanticNode``).
"""

# Standard library imports
import os
import shutil
import unittest

# Third party imports
from rdflib import RDF, BNode, Graph, Literal

# Local application imports
import twin4build
import twin4build as tb
import twin4build.core as core
from twin4build.model.model import MODELS_SEMANTIC_NODE
from twin4build.model.semantic_model.semantic_model import SemanticModel
from twin4build.systems.schedule.schedule_system import ScheduleSystem
from twin4build.systems.sensor.sensor_system import SensorSystem
from twin4build.translator.translator import (
    Node,
    SignaturePattern,
    StepRule,
    Translator,
)

twin4build._IS_TESTING = True

BRICK = core.namespace.BRICK
REF = core.namespace.BRICKREF
EX = core.namespace.T4B


def leaf_sensor_pattern():
    """Any Brick point with a timeseries reference becomes a SensorSystem
    reading that id; it models the point and the (blank) reference node."""
    sensor = Node(cls=BRICK.Point)
    externalref = Node(cls=(REF.ExternalReference, core.BlankNode))
    timeseries_id = Node(cls=core.namespace.XSD.string)
    sp = SignaturePattern(id="test_leaf_sensor", system=SensorSystem)
    sp.add_rule(
        StepRule(subject=sensor, object=externalref, predicate=REF.hasExternalReference)
    )
    sp.add_rule(
        StepRule(
            subject=externalref, object=timeseries_id, predicate=REF.hasTimeseriesId
        )
    )
    sp.add_parameter("uuid", timeseries_id)
    sp.add_modeled_node(sensor)
    sp.add_modeled_node(externalref)
    return sp


def building_graph(graph: Graph) -> None:
    """Zone temperature sensors and their setpoints, each with a timeseries reference."""
    for name, cls in (
        ("zone_temp_0", BRICK.Zone_Air_Temperature_Sensor),
        ("zone_temp_1", BRICK.Zone_Air_Temperature_Sensor),
        ("setpoint_0", BRICK.Zone_Air_Temperature_Setpoint),
        ("setpoint_1", BRICK.Zone_Air_Temperature_Setpoint),
    ):
        ref = BNode()
        graph.add((EX[name], RDF.type, cls))
        graph.add((EX[name], REF.hasExternalReference, ref))
        graph.add((ref, RDF.type, REF.ExternalReference))
        graph.add((ref, REF.hasTimeseriesId, Literal(f"uuid-{name}")))


def _schedule(id_, value):
    return ScheduleSystem(
        id=id_,
        weekday_ruleset={
            "ruleset_start_minute": [],
            "ruleset_end_minute": [],
            "ruleset_start_hour": [],
            "ruleset_end_hour": [],
            "ruleset_value": [],
            "ruleset_default_value": value,
        },
    )


def named(mapping) -> dict:
    """``{component id: the URIs of its named (not blank) semantic nodes}``."""
    return {
        component.id: {
            str(node.uri) for node in nodes if not isinstance(node.uri, BNode)
        }
        for component, nodes in mapping.items()
    }


class TestModelKeepsTranslationMaps(unittest.TestCase):
    MODEL_ID = "test_translation_maps"
    RESTORED_ID = "test_translation_maps_restored"

    def tearDown(self):
        for model_id in (self.MODEL_ID, self.RESTORED_ID):
            path = os.path.join("generated_files", "models", model_id)
            if os.path.exists(path):
                shutil.rmtree(path)

    def linked(self):
        """Two setpoints, each read by a sensor, as a translation leaves them: the translator's
        maps say which semantic nodes each component models (a setpoint its point, a sensor its
        point and the blank reference node that names its series)."""
        model = tb.Model(id=self.MODEL_ID)
        building_graph(model.semantic_model.instance_graph)
        node = model.semantic_model.get_instance
        translator = Translator()
        for i in range(2):
            setpoint, sensor = _schedule(f"setpoint_{i}", 21.0 + i), SensorSystem(
                id=f"sensor_{i}"
            )
            model.add_connection(setpoint, sensor, "scheduleValue", "measuredValue")
            translator._sim2sem_map[setpoint] = {node(EX[f"setpoint_{i}"])}
            translator._sim2sem_map[sensor] = {
                node(EX[f"zone_temp_{i}"]),
                node(BNode()),
            }
            for semantic in (
                translator._sim2sem_map[setpoint] | translator._sim2sem_map[sensor]
            ):
                translator._sem2sim_map.setdefault(semantic, set()).add(
                    setpoint
                    if semantic in translator._sim2sem_map[setpoint]
                    else sensor
                )
        model._translator = translator
        model.load(draw_semantic_model=False, draw_simulation_model=False)
        return model, translator

    def restored(self, path: str) -> tb.Model:
        model = tb.Model(id=self.RESTORED_ID)
        model.load(
            filename=path, draw_semantic_model=False, draw_simulation_model=False
        )
        return model

    def test_a_translated_model_answers_with_the_translators_maps(self):
        semantic = SemanticModel(id=self.MODEL_ID)
        building_graph(semantic.instance_graph)
        translator = Translator()
        model = translator.translate(
            semantic, patterns=[leaf_sensor_pattern()], id=self.MODEL_ID
        )
        self.assertEqual(len(model.sim2sem_map), 4)
        for component, nodes in model.sim2sem_map.items():
            self.assertEqual(nodes, set(translator.sim2sem_map[component]))
            # the point and its blank reference node
            self.assertEqual(len(nodes), 2)
        self.assertEqual(
            {uri for uris in named(model.sim2sem_map).values() for uri in uris},
            {
                str(EX[name])
                for name in ("zone_temp_0", "zone_temp_1", "setpoint_0", "setpoint_1")
            },
        )
        # the other way round: every semantic node -> the components that model it
        for node, components in model.sem2sim_map.items():
            self.assertEqual(components, set(translator.sem2sim_map[node]))
        self.assertEqual(sum(len(c) for c in model.sem2sim_map.values()), 8)

    def test_only_components_still_in_the_model_are_answered_for(self):
        model, translator = self.linked()
        model.remove_component(model.components["sensor_1"])
        self.assertEqual(len(translator.sim2sem_map), 4)
        self.assertEqual(
            sorted(c.id for c in model.sim2sem_map),
            ["sensor_0", "setpoint_0", "setpoint_1"],
        )

    def test_the_link_survives_serialize_and_load(self):
        model, _ = self.linked()
        before = named(model.sim2sem_map)
        self.assertEqual(before["sensor_0"], {str(EX["zone_temp_0"])})
        restored = self.restored(model.serialize())
        self.assertIsNone(restored._translator)
        self.assertEqual(named(restored.sim2sem_map), before)
        # a blank node has no identity outside its graph: the named nodes are kept
        self.assertTrue(all(len(nodes) == 1 for nodes in restored.sim2sem_map.values()))
        by_node = {
            str(node.uri): {c.id for c in components}
            for node, components in restored.sem2sim_map.items()
        }
        self.assertEqual(
            by_node,
            {
                str(EX["zone_temp_0"]): {"sensor_0"},
                str(EX["zone_temp_1"]): {"sensor_1"},
                str(EX["setpoint_0"]): {"setpoint_0"},
                str(EX["setpoint_1"]): {"setpoint_1"},
            },
        )
        # the restored components are the model's own
        for component in restored.sim2sem_map:
            self.assertIs(restored.components[component.id], component)

    def test_restored_nodes_answer_for_their_classes_with_the_semantic_graph(self):
        model, _ = self.linked()
        restored = self.restored(model.serialize())
        node = next(iter(restored.sim2sem_map[restored.components["sensor_0"]]))
        # by URI alone: an instance of the model's semantic model
        self.assertIs(
            node, restored.semantic_model.get_instance(str(EX["zone_temp_0"]))
        )
        self.assertFalse(node.isinstance(BRICK.Zone_Air_Temperature_Sensor))
        # with the graph the model was translated from, it knows what it is
        building_graph(restored.semantic_model.instance_graph)
        restored.semantic_model._instances.clear()
        node = next(iter(restored.sim2sem_map[restored.components["sensor_0"]]))
        self.assertTrue(node.isinstance(BRICK.Zone_Air_Temperature_Sensor))

    def test_a_restored_model_serializes_the_link_again(self):
        model, _ = self.linked()
        first = self.restored(model.serialize())
        first.remove_component(first.components["sensor_1"])
        self.assertEqual(
            sorted(c.id for c in first.sim2sem_map),
            ["sensor_0", "setpoint_0", "setpoint_1"],
        )
        again = tb.Model(id=self.MODEL_ID)
        again.load(
            filename=first.serialize(),
            draw_semantic_model=False,
            draw_simulation_model=False,
        )
        self.assertEqual(
            sorted(named(again.sim2sem_map)), ["sensor_0", "setpoint_0", "setpoint_1"]
        )

    def test_the_link_is_one_triple_per_component_and_named_node(self):
        model, _ = self.linked()
        graph = Graph()
        graph.parse(model.serialize(), format="turtle")
        links = list(graph.triples((None, MODELS_SEMANTIC_NODE, None)))
        self.assertEqual(len(links), 4)
        self.assertEqual(
            {str(o) for _, _, o in links},
            {
                str(EX[name])
                for name in ("zone_temp_0", "zone_temp_1", "setpoint_0", "setpoint_1")
            },
        )
        # written once, however often the model is serialized
        model.serialize()
        graph = Graph()
        graph.parse(model.instance_graph_path, format="turtle")
        self.assertEqual(
            len(list(graph.triples((None, MODELS_SEMANTIC_NODE, None)))), 4
        )

    def test_a_model_built_by_hand_has_no_link(self):
        model = tb.Model(id=self.MODEL_ID)
        model.add_connection(
            _schedule("setpoint", 21.0),
            SensorSystem(id="by_hand"),
            "scheduleValue",
            "measuredValue",
        )
        model.load(draw_semantic_model=False, draw_simulation_model=False)
        self.assertEqual(model.sim2sem_map, {})
        self.assertEqual(model.sem2sim_map, {})
        graph = Graph()
        graph.parse(model.serialize(), format="turtle")
        self.assertEqual(list(graph.triples((None, MODELS_SEMANTIC_NODE, None))), [])
        self.assertEqual(self.restored(model.instance_graph_path).sim2sem_map, {})


if __name__ == "__main__":
    unittest.main()
