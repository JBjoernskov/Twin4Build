"""Regression test: MILP-active components without any connection must
still be part of the translated model.

``Translator._connect_components`` used to derive the set of "used"
components from the active *connections* only.  A solution made of
components that do not connect to anything -- e.g. Brick leaf sensors
bound to a timeseries id through ``ref:hasExternalReference`` -- was
therefore silently discarded: ``translate`` returned a ``Model`` with
zero components and empty ``sim2sem`` / ``sem2sim`` maps.
"""

# Standard library imports
import os
import shutil
import unittest

# Third party imports
from rdflib import RDF, BNode, Literal

# Local application imports
import twin4build
import twin4build.core as core
from twin4build.model.semantic_model.semantic_model import SemanticModel
from twin4build.systems.sensor.sensor_system import SensorSystem
from twin4build.translator.translator import Translator

twin4build._IS_TESTING = True


class TestUnconnectedComponentsAreKept(unittest.TestCase):
    MODEL_ID = "test_unconnected_components"

    def tearDown(self):
        path = os.path.join("generated_files", "models", self.MODEL_ID)
        if os.path.exists(path):
            shutil.rmtree(path)

    def test_leaf_sensors_survive_translation(self):
        BRICK = core.namespace.BRICK
        REF = core.namespace.BRICKREF
        EX = core.namespace.T4B

        sm = SemanticModel(id=self.MODEL_ID)
        g = sm.instance_graph
        expected_ids = []
        for i in range(3):
            sensor = EX[f"zone_temp_{i}"]
            ref = BNode()
            g.add((sensor, RDF.type, BRICK.Zone_Air_Temperature_Sensor))
            g.add((sensor, REF.hasExternalReference, ref))
            # Typed directly as ExternalReference so the test does not depend
            # on the Brick ``ref`` ontology being available for subclass reasoning.
            g.add((ref, RDF.type, REF.ExternalReference))
            g.add((ref, REF.hasTimeseriesId, Literal(f"uuid-{i}")))
            expected_ids.append(f"zone_temp_{i}")
        # A sensor without any data source or upstream cannot be simulated
        # and must still be dropped, as before.
        g.add((EX["dangling_virtual"], RDF.type, BRICK.Zone_Air_Temperature_Sensor))

        translator = Translator()
        model = translator.translate(sm, systems=[SensorSystem], id=self.MODEL_ID)

        components = model.components
        self.assertEqual(len(components), 3, components)
        self.assertNotIn("dangling_virtual", components)
        for comp_id, comp in components.items():
            self.assertIsInstance(comp, SensorSystem)
        self.assertEqual(
            sorted(c.uuid for c in components.values()),
            ["uuid-0", "uuid-1", "uuid-2"],
        )
        # The translator-side maps must cover the unconnected components too.
        self.assertEqual(len(translator.sim2sem_map), 3)
        self.assertEqual(
            set(translator.sim2sem_map.keys()), set(components.values())
        )
        for comp in components.values():
            self.assertIn(comp, translator._sim2group_map)

    def test_standalone_rule_requires_a_data_source(self):
        """An input-free component is kept only if it can emit something.

        A ScheduleSystem instantiated from a semantic node alone has none of
        its source flags set and fails at simulation, so keeping it turns a
        translatable model into one that cannot run (the full-workflow
        example's consumer-less cooling setpoint).
        """
        import twin4build as tb
        from twin4build.translator.translator import Translator

        is_standalone = Translator._is_standalone_component
        with_data = tb.ScheduleSystem(
            weekday_ruleset={"ruleset_default_value": 21.0}, id="with_data"
        )
        self.assertTrue(is_standalone(with_data))
        without_data = tb.ScheduleSystem(id="without_data")
        self.assertFalse(is_standalone(without_data))
        # Data-bound leaves stay standalone through their source binding.
        self.assertTrue(is_standalone(SensorSystem(uuid="uuid-x", id="leaf")))
        self.assertFalse(is_standalone(SensorSystem(id="virtual")))


if __name__ == "__main__":
    unittest.main()
