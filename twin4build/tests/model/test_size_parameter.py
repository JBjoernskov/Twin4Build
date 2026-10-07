"""Size quantities take bounds relative to their sized value (#245).

A class declares ``"relative": (lo, hi)`` next to the absolute ``lb`` /
``ub`` of a size quantity (a nominal flow, a capacity, a volume).
:meth:`System.size_parameter` makes the bounds relative to the value it
sizes; a later value (a warm start, a fit's result) moves the parameter, not
the bounds.  Unsized, the absolute bounds stay and a warning names the
attribute.  On the HTR ring the heat recovery's flows were declared 0.5-10
kg/s while the two merged units supply 11-23 kg/s.
"""

# Standard library imports
import os
import shutil
import unittest
from unittest import mock

# Third party imports
from rdflib import RDF, BNode, Literal

# Local application imports
import twin4build as tb
import twin4build.core as core
import twin4build.systems.saref4syst.system as system_module
from twin4build.model.semantic_model.semantic_model import SemanticModel
from twin4build.tests.model.test_batching_vector_slots import Hub, Leaf, Sink
from twin4build.tests.translator.test_unconnected_components import leaf_sensor_pattern
from twin4build.translator.translator import Node, SignaturePattern, StepRule, Translator


def _bounds(component, path):
    (entry,) = [e for e in component.get_estimable_parameters() if e[1] == path]
    return entry[3], entry[4]


class SizedLeaf(Leaf):
    """A leaf whose ``p`` is a size quantity: absolute fallback 0-100,
    relative factors 0.5-2."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.parameter = {"p": {"lb": 0.0, "ub": 100.0, "relative": (0.5, 2.0)}}


class TestSizeParameter(unittest.TestCase):
    def setUp(self):
        system_module._UNSIZED_WARNED.clear()

    def test_sizing_makes_the_bounds_relative(self):
        unit = tb.AirToAirHeatRecoverySystem(id="recovery")
        self.assertEqual(_bounds(unit, "primaryAirFlowRateMax"), (0.5, 10.0))  # the absolute fallback
        self.assertTrue(unit.size_parameter("primaryAirFlowRateMax", 11.0))
        self.assertAlmostEqual(float(unit.primaryAirFlowRateMax.get()), 11.0)
        self.assertEqual(_bounds(unit, "primaryAirFlowRateMax"), (0.25 * 11.0, 4.0 * 11.0))
        # the anchor stays: a warm start's value moves the parameter only
        unit.primaryAirFlowRateMax.set(30.0, normalized=False)
        self.assertEqual(_bounds(unit, "primaryAirFlowRateMax"), (0.25 * 11.0, 4.0 * 11.0))
        # the other instance's spec is untouched
        self.assertEqual(_bounds(tb.AirToAirHeatRecoverySystem(id="other"), "primaryAirFlowRateMax"), (0.5, 10.0))

    def test_a_sub_system_path_is_sized(self):
        unit = tb.AirHandlingUnitCoreSystem(id="unit")
        self.assertTrue(unit.size_parameter("heat_recovery.secondaryAirFlowRateMax", 20.0))
        self.assertEqual(_bounds(unit, "heat_recovery.secondaryAirFlowRateMax"), (5.0, 80.0))

    def test_an_unsized_size_quantity_keeps_its_bounds_and_warns_once(self):
        with mock.patch.object(system_module.LOGGER, "warning") as warning:
            self.assertEqual(_bounds(tb.FanSystem(id="fan1"), "nominalAirFlowRate"), (0.5, 10.0))
            _bounds(tb.FanSystem(id="fan2"), "nominalAirFlowRate")
        named = [c for c in warning.call_args_list if c.args[1:3] == ("FanSystem", "nominalAirFlowRate")]
        self.assertEqual(len(named), 1)

    def test_an_intensive_quantity_is_not_made_relative(self):
        fan = tb.FanSystem(id="fan")
        self.assertFalse(fan.size_parameter("f_total", 0.7))
        self.assertEqual(_bounds(fan, "f_total"), (0.3, 0.9))

    def test_a_value_that_is_not_positive_keeps_the_absolute_bounds(self):
        fan = tb.FanSystem(id="fan")
        self.assertFalse(fan.size_parameter("nominalAirFlowRate", 0.0))
        self.assertEqual(_bounds(fan, "nominalAirFlowRate"), (0.5, 10.0))


class TestBatchedSizing(unittest.TestCase):
    N = 3

    def _model(self, sized):
        """Leaves 0..N-1 feed hub slots 0..N-1, sink k reads slot k; leaf i
        is sized at 10 (i + 1) when ``sized`` holds i."""
        model = tb.Model(id="test_size_parameter_batched")
        hub = Hub(id="hub")
        leaves = [SizedLeaf(p=float(i + 1), id=f"leaf{i}") for i in range(self.N)]
        for i, leaf in enumerate(leaves):
            if i in sized:
                leaf.size_parameter("p", 10.0 * (i + 1))
            model.add_connection(leaf, hub, "v", "x", input_port_index=i)
            model.add_connection(hub, Sink(id=f"sink{i}"), "y", "u", output_port_index=i, input_port_index=0)
        model.load(draw_semantic_model=False, draw_simulation_model=False)
        batched = model.batch_components()
        batched.load(draw_semantic_model=False, draw_simulation_model=False)
        return model._component_to_meta["leaf0"][0]

    def test_the_meta_reports_each_instances_sized_bounds(self):
        meta = self._model(sized={0, 1, 2})
        self.assertIsNot(meta, None)
        lb, ub = _bounds(meta, "p")
        self.assertEqual(list(lb), [5.0, 10.0, 15.0])
        self.assertEqual(list(ub), [20.0, 40.0, 60.0])
        self.assertTrue(meta.parameter["p"]["sized"])

    def test_a_meta_with_an_unsized_instance_is_not_sized(self):
        meta = self._model(sized={0, 1})
        lb, ub = _bounds(meta, "p")
        self.assertEqual(list(lb), [5.0, 10.0, 0.0])
        self.assertEqual(list(ub), [20.0, 40.0, 100.0])
        self.assertFalse(meta.parameter["p"]["sized"])


class TestTranslatorSizing(unittest.TestCase):
    MODEL_ID = "test_size_parameter_translation"

    def tearDown(self):
        path = os.path.join("generated_files", "models", self.MODEL_ID)
        if os.path.exists(path):
            shutil.rmtree(path)

    def test_a_value_from_the_semantic_model_sizes_the_parameter(self):
        """Two rooms with their volumes (``brick:volume [ brick:value x ]``)
        and a data-bound point each that feeds the room (so the translator
        keeps it); the volume mapped onto ``mass.V`` sizes it."""
        BRICK, REF, EX = core.namespace.BRICK, core.namespace.BRICKREF, core.namespace.T4B
        sm = SemanticModel(id=self.MODEL_ID)
        g = sm.instance_graph
        for i, volume in ((1, 21.0), (2, 22.0)):
            room, point, ref, holder = EX[f"R0{i}"], EX[f"R0{i}_T"], BNode(), BNode()
            g.add((room, RDF.type, core.namespace.REC.Room))
            g.add((room, BRICK.volume, holder))
            g.add((holder, BRICK.value, Literal(volume)))
            g.add((room, BRICK.hasPoint, point))
            g.add((point, RDF.type, BRICK.Point))
            g.add((point, REF.hasExternalReference, ref))
            g.add((ref, RDF.type, REF.ExternalReference))
            g.add((ref, REF.hasTimeseriesId, Literal(f"uuid-{i}")))
        space = Node(cls=(core.namespace.REC.Room,))
        point = Node(cls=BRICK.Point)
        holder = Node(cls=(core.BlankNode,))
        value = Node(cls=(core.namespace.XSD.float, core.namespace.XSD.double, core.namespace.XSD.decimal))
        sp = SignaturePattern(id="test_size_parameter_room", system=tb.BuildingSpaceSystem)
        sp.add_rule(StepRule(subject=space, object=point, predicate=BRICK.hasPoint))
        sp.add_rule(StepRule(subject=space, object=holder, predicate=BRICK.volume))
        sp.add_rule(StepRule(subject=holder, object=value, predicate=BRICK.value))
        sp.add_parameter("mass.V", value)
        sp.add_connection(sender_node=point, output_port="measuredValue", input_port="outdoorTemperature")
        sp.add_modeled_node(holder)
        sp.add_modeled_node(space)
        model = Translator().translate(sm, patterns=[leaf_sensor_pattern(), sp], id=self.MODEL_ID)
        spaces = sorted(model.get_components_by_class(tb.BuildingSpaceSystem), key=lambda c: c.id)
        self.assertEqual(len(spaces), 2)
        for space_component, volume in zip(spaces, (21.0, 22.0)):
            self.assertAlmostEqual(float(space_component.mass.V.get()), volume)
            self.assertEqual(_bounds(space_component, "mass.V"), (0.5 * volume, 2.0 * volume))

if __name__ == "__main__":
    unittest.main()
