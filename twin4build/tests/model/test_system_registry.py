"""External ``System`` classes through a system registry (#132).

The class below stands for a component that lives in another package: it is
defined here, not in the library, and it is never set as an attribute of
``twin4build.systems``.

Covered:

* the registry itself: registration, duplicate type ids, a second type id
  for a class, malformed ids, built-in classes, ``replace`` and
  ``unregister``;
* ``Model.serialize()`` records the type id, the provider and the provider
  version of a registered class and leaves built-in components as they were;
* ``Model.load(filename=...)`` gives the same class, id, parameter, ports,
  connections and simulation back -- with a custom registry, with the
  default registry, and in a fresh process;
* a type that is not registered fails with an error that names the
  component, the type id, the provider and the version;
* a model that records only class names still loads (built-in classes, a
  registered class, and a class set on ``twin4build.systems`` by hand);
* translation: patterns bound to a registered class translate next to
  built-in ones, a registry given to the translator is handed to the model
  and rejects a class it does not know, ``systems=`` stays an allow-list.
"""

# Standard library imports
import datetime
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

# Third party imports
import torch
from dateutil import tz
from rdflib import RDF, BNode, Graph, Literal

# Local application imports
import twin4build as tb
import twin4build.core as core
import twin4build.systems as systems_module
import twin4build.systems.registry as registry_module
import twin4build.utils.types as tps
from twin4build.model.semantic_model.semantic_model import SemanticModel
from twin4build.systems.registry import (
    PROVIDER_KEY,
    PROVIDER_VERSION_KEY,
    TYPE_ID_KEY,
    SystemRegistry,
    UnknownSystemTypeError,
)
from twin4build.translator.translator import (
    Node,
    SignaturePattern,
    StepRule,
    Translator,
)

tb._IS_TESTING = True

TYPE_ID = "acme:GainSystem@1"
PROVIDER = "acme-t4b-components"
VERSION = "1.4.2"
START = datetime.datetime(2023, 1, 2, tzinfo=tz.UTC)


class AcmeGainSystem(core.System):
    """``y = gain * x`` with an estimable gain: the external component."""

    def __init__(self, gain: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.gain = tps.Parameter(
            torch.tensor(gain, dtype=tps.float_dtype()),
            min_value=0.0,
            max_value=10.0,
            requires_grad=False,
        )
        self.input = {"x": tps.Scalar()}
        self.output = {"y": tps.Scalar()}
        self.parameter = {"gain": {"lb": 0.0, "ub": 10.0}}
        self._config = {"parameters": ["gain"]}

    @property
    def config(self):
        return self._config

    def initialize(self, start_time, end_time, step_size):
        _, _, n_t, _ = core.Simulator.get_simulation_timesteps(
            start_time, end_time, step_size
        )
        for port in (*self.input.values(), *self.output.values()):
            port.initialize(n_t=n_t, n_s=len(start_time), n_c=self.n_c)

    def forward(self, x, inputs, params, sample_time):
        return x, {"y": inputs["x"] * self.gain.get()}

    def do_step(self, second_time, date_time, step_size, step_index):
        _, outputs = self.forward(None, {"x": self.input["x"].get()}, {}, step_size)
        self.output["y"]._set(outputs["y"], i_t=step_index)


class AcmeOtherSystem(AcmeGainSystem):
    """A second external class (a subclass is a class of its own)."""


def register(registry):
    """What the providing package calls on the registry of the application."""
    return registry.register(
        AcmeGainSystem, type_id=TYPE_ID, provider=PROVIDER, version=VERSION
    )


def _registry():
    registry = SystemRegistry()
    register(registry)
    return registry


def _remove_models(*model_ids):
    for model_id in model_ids:
        shutil.rmtree(
            os.path.join("generated_files", "models", model_id), ignore_errors=True
        )


def _build(model_id, registry=None):
    """schedule -> AcmeGainSystem -> ScalarProductSystem (both inputs)."""
    if registry is None:
        model = tb.Model(id=model_id)
    else:
        model = tb.Model(id=model_id, system_registry=registry)
    signal = tb.ScheduleSystem(
        id="signal",
        weekday_ruleset={
            "ruleset_start_minute": [0],
            "ruleset_end_minute": [0],
            "ruleset_start_hour": [1],
            "ruleset_end_hour": [2],
            "ruleset_value": [3.0],
            "ruleset_default_value": 1.0,
        },
    )
    gain = AcmeGainSystem(id="gain", gain=2.5)
    square = tb.ScalarProductSystem(id="square", scale_factor=0.5)
    model.add_connection(signal, gain, "scheduleValue", "x")
    model.add_connection(gain, square, "y", "input_1")
    model.add_connection(gain, square, "y", "input_2")
    model.load(draw_semantic_model=False, draw_simulation_model=False)
    return model


def _simulate(model):
    simulator = tb.Simulator(model, execution_mode="object")
    simulator.simulate(
        start_time=START,
        end_time=START + datetime.timedelta(hours=3),
        step_size=600,
        show_progress_bar=False,
    )
    return {
        key: model.components[component].output[port]._history.detach().clone()
        for key, component, port in (("y", "gain", "y"), ("square", "square", "output"))
    }


def _serialize(model):
    model.serialize()
    path, _ = model.simulation_model._semantic_model.get_dir(
        filename="instance_graph.ttl"
    )
    return path


def _literals(path, component_id):
    """The registry literals a serialized component carries, by key."""
    graph = Graph()
    graph.parse(path, format="turtle")
    subject = core.namespace.T4B[component_id]
    return {
        key: [str(o) for o in graph.objects(subject, core.namespace.T4B[key])]
        for key in (TYPE_ID_KEY, PROVIDER_KEY, PROVIDER_VERSION_KEY)
    }


def _types(path, component_id):
    """The class names a serialized component is typed with."""
    graph = Graph()
    graph.parse(path, format="turtle")
    subject = core.namespace.T4B[component_id]
    return sorted(
        str(o).replace(str(core.namespace.T4B), "")
        for o in graph.objects(subject, RDF.type)
        if str(o).startswith(str(core.namespace.T4B))
    )


def _connections(model):
    return sorted(
        (component.id, connection.output_port, point.connection_point_of.id, point.input_port)
        for component in model.components.values()
        for connection in component.connected_through
        for point in connection.connects_system_at
    )


class TestSystemRegistry(unittest.TestCase):
    def test_public_names(self):
        self.assertIs(tb.SystemRegistry, SystemRegistry)
        self.assertIs(tb.system_registry, registry_module.system_registry)
        self.assertIsInstance(tb.system_registry, SystemRegistry)

    def test_register_and_look_up(self):
        registry = SystemRegistry()
        registration = register(registry)
        self.assertIs(registration.system, AcmeGainSystem)
        self.assertEqual(
            (registration.type_id, registration.provider, registration.version),
            (TYPE_ID, PROVIDER, VERSION),
        )
        self.assertIs(registry.get(TYPE_ID), registration)
        self.assertIs(registry.registration_of(AcmeGainSystem), registration)
        self.assertEqual(registry.registrations(), (registration,))
        self.assertIs(registry.resolve(TYPE_ID), AcmeGainSystem)
        self.assertIsNone(registry.get("acme:Missing@1"))

    def test_known_classes(self):
        registry = _registry()
        self.assertTrue(registry.is_known(AcmeGainSystem))
        self.assertTrue(registry.is_known(tb.SensorSystem))
        # A subclass of a registered class is a class of its own.
        self.assertFalse(registry.is_known(AcmeOtherSystem))
        self.assertIsNone(registry.registration_of(AcmeOtherSystem))
        self.assertFalse(SystemRegistry().is_known(AcmeGainSystem))

    def test_registering_again_is_a_no_op(self):
        registry = SystemRegistry()
        first = register(registry)
        self.assertIs(register(registry), first)
        self.assertEqual(len(registry.registrations()), 1)

    def test_version_is_stored_as_text(self):
        registry = SystemRegistry()
        registration = registry.register(AcmeGainSystem, type_id=TYPE_ID, version=2)
        self.assertEqual(registration.version, "2")
        self.assertIsNone(registration.provider)

    def test_duplicate_type_id_is_rejected(self):
        registry = _registry()
        with self.assertRaisesRegex(ValueError, "already registered") as caught:
            registry.register(AcmeOtherSystem, type_id=TYPE_ID, provider="other")
        self.assertIn(TYPE_ID, str(caught.exception))
        self.assertIn(PROVIDER, str(caught.exception))
        self.assertIs(registry.resolve(TYPE_ID), AcmeGainSystem)

    def test_second_type_id_for_a_class_is_rejected(self):
        registry = _registry()
        with self.assertRaisesRegex(ValueError, "AcmeGainSystem is already registered"):
            registry.register(AcmeGainSystem, type_id="acme:GainSystem@2")
        self.assertIsNone(registry.get("acme:GainSystem@2"))

    def test_same_type_id_with_other_metadata_is_rejected(self):
        registry = _registry()
        with self.assertRaisesRegex(ValueError, "already registered"):
            registry.register(
                AcmeGainSystem, type_id=TYPE_ID, provider=PROVIDER, version="2.0.0"
            )

    def test_replace(self):
        registry = _registry()
        registry.register(AcmeOtherSystem, type_id=TYPE_ID, replace=True)
        self.assertIs(registry.resolve(TYPE_ID), AcmeOtherSystem)
        self.assertIsNone(registry.registration_of(AcmeGainSystem))
        registry.register(AcmeOtherSystem, type_id="acme:Other@1", replace=True)
        self.assertIsNone(registry.get(TYPE_ID))
        self.assertEqual([r.type_id for r in registry.registrations()], ["acme:Other@1"])

    def test_unregister(self):
        registry = _registry()
        registry.unregister(TYPE_ID)
        registry.unregister(TYPE_ID)
        self.assertEqual(registry.registrations(), ())
        self.assertFalse(registry.is_known(AcmeGainSystem))
        register(registry)

    def test_malformed_type_id_is_rejected(self):
        registry = SystemRegistry()
        for type_id in ("GainSystem", "", ":GainSystem", "acme:", "acme:Gain System", None, 3):
            with self.subTest(type_id=type_id):
                with self.assertRaisesRegex(ValueError, "<namespace>:<name>"):
                    registry.register(AcmeGainSystem, type_id=type_id)
        self.assertEqual(registry.registrations(), ())

    def test_only_system_classes_are_registered(self):
        registry = SystemRegistry()
        for candidate in (object, AcmeGainSystem(id="instance"), "AcmeGainSystem"):
            with self.subTest(candidate=candidate):
                with self.assertRaisesRegex(TypeError, "System subclass"):
                    registry.register(candidate, type_id=TYPE_ID)

    def test_builtin_classes_are_not_registered(self):
        registry = SystemRegistry()
        with self.assertRaisesRegex(ValueError, "built-in"):
            registry.register(tb.SensorSystem, type_id="acme:SensorSystem@1")
        self.assertIs(registry.resolve(class_name="SensorSystem"), tb.SensorSystem)
        # The deprecated aliases of the built-in classes stay resolvable.
        self.assertIs(registry.resolve(class_name="fmuSystem"), tb.FmuSystem)

    def test_names_that_are_not_systems_do_not_resolve(self):
        registry = SystemRegistry()
        for class_name in ("registry", "importlib", "score_pair", "NoSuchSystem"):
            with self.subTest(class_name=class_name):
                with self.assertRaises(UnknownSystemTypeError):
                    registry.resolve(class_name=class_name, component_id="c")

    def test_ambiguous_class_name(self):
        registry = _registry()
        clash = type("AcmeGainSystem", (AcmeGainSystem,), {})
        registry.register(clash, type_id="other:GainSystem@1", provider="other")
        with self.assertRaisesRegex(UnknownSystemTypeError, "ambiguous") as caught:
            registry.resolve(class_name="AcmeGainSystem", component_id="gain")
        self.assertIn(TYPE_ID, str(caught.exception))
        self.assertIn("other:GainSystem@1", str(caught.exception))
        # The type id is what tells them apart.
        self.assertIs(registry.resolve(TYPE_ID), AcmeGainSystem)
        self.assertIs(registry.resolve("other:GainSystem@1"), clash)


class TestExternalSystemRoundTrip(unittest.TestCase):
    MODEL_ID = "test_system_registry"

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def setUp(self):
        self.model_ids = []
        self.addCleanup(lambda: _remove_models(*self.model_ids))
        # Nothing here may depend on the class being reachable by name.
        self.assertFalse(hasattr(systems_module, "AcmeGainSystem"))
        self.assertFalse(hasattr(tb, "AcmeGainSystem"))

    def _id(self, suffix):
        model_id = f"{self.MODEL_ID}_{suffix}"
        self.model_ids.append(model_id)
        return model_id

    def _load(self, suffix, path, registry=None):
        if registry is None:
            model = tb.Model(id=self._id(suffix))
        else:
            model = tb.Model(id=self._id(suffix), system_registry=registry)
        model.load(filename=path, draw_semantic_model=False, draw_simulation_model=False)
        return model

    def _assert_same(self, reloaded, model, reference):
        self.assertEqual(set(reloaded.components), set(model.components))
        gain = reloaded.components["gain"]
        self.assertIs(type(gain), AcmeGainSystem)
        self.assertEqual(gain.id, "gain")
        self.assertAlmostEqual(float(gain.gain.get().reshape(-1)[0]), 2.5, places=6)
        self.assertEqual(list(gain.input), ["x"])
        self.assertEqual(list(gain.output), ["y"])
        self.assertIs(type(reloaded.components["signal"]), tb.ScheduleSystem)
        self.assertIs(type(reloaded.components["square"]), tb.ScalarProductSystem)
        self.assertEqual(_connections(reloaded), _connections(model))
        self.assertEqual(len(_connections(reloaded)), 3)
        result = _simulate(reloaded)
        for key, expected in reference.items():
            torch.testing.assert_close(result[key], expected)

    def test_round_trip_with_a_custom_registry(self):
        model = _build(self._id("custom"), _registry())
        self.assertIsInstance(model.system_registry, SystemRegistry)
        self.assertIsNot(model.system_registry, tb.system_registry)
        reference = _simulate(model)
        # The schedule steps and the gain acts on it: not a trivial signal.
        self.assertGreater(float(reference["y"].max()), float(reference["y"].min()))
        self.assertAlmostEqual(float(reference["y"].max()), 7.5, places=6)
        path = _serialize(model)

        self.assertEqual(
            _literals(path, "gain"),
            {TYPE_ID_KEY: [TYPE_ID], PROVIDER_KEY: [PROVIDER], PROVIDER_VERSION_KEY: [VERSION]},
        )
        # Built-in components are serialized as before.
        for component_id in ("signal", "square"):
            self.assertEqual(
                _literals(path, component_id),
                {TYPE_ID_KEY: [], PROVIDER_KEY: [], PROVIDER_VERSION_KEY: []},
            )

        # The loading side has a registry of its own.
        reloaded = self._load("custom_reloaded", path, _registry())
        self.assertFalse(hasattr(systems_module, "AcmeGainSystem"))
        self._assert_same(reloaded, model, reference)
        # A reloaded model serializes the type again.
        self.assertEqual(_literals(_serialize(reloaded), "gain")[TYPE_ID_KEY], [TYPE_ID])

    def test_round_trip_with_the_default_registry(self):
        self.addCleanup(tb.system_registry.unregister, TYPE_ID)
        register(tb.system_registry)
        model = _build(self._id("default"))
        self.assertIs(model.system_registry, tb.system_registry)
        reference = _simulate(model)
        path = _serialize(model)
        self.assertEqual(_literals(path, "gain")[TYPE_ID_KEY], [TYPE_ID])
        reloaded = self._load("default_reloaded", path)
        self._assert_same(reloaded, model, reference)

    def test_round_trip_in_a_fresh_process(self):
        if __name__ == "__main__":
            self.skipTest("the provider module must be importable by name")
        model = _build(self._id("process"), _registry())
        reference = _simulate(model)
        path = _serialize(model)
        result_file = os.path.join(self.tmp, "fresh_process.json")
        environment = dict(os.environ)
        environment["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                "import importlib, sys; "
                "importlib.import_module(sys.argv[1]).fresh_process_main(*sys.argv[2:])",
                __name__,
                self._id("process_reloaded"),
                path,
                result_file,
            ],
            env=environment,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr[-4000:])
        with open(result_file) as f:
            result = json.load(f)
        self.assertEqual(result["class"], "AcmeGainSystem")
        self.assertFalse(result["attribute_of_systems"])
        self.assertAlmostEqual(result["gain"], 2.5, places=6)
        for key, expected in reference.items():
            torch.testing.assert_close(
                torch.tensor(result[key], dtype=expected.dtype).reshape(expected.shape),
                expected,
            )

    def test_missing_registration_names_type_and_provider(self):
        model = _build(self._id("missing"), _registry())
        path = _serialize(model)
        with self.assertRaises(UnknownSystemTypeError) as caught:
            self._load("missing_reloaded", path, SystemRegistry())
        error = caught.exception
        self.assertEqual(
            (error.component_id, error.type_id, error.class_name, error.provider, error.version),
            ("gain", TYPE_ID, "AcmeGainSystem", PROVIDER, VERSION),
        )
        for text in ("'gain'", TYPE_ID, "AcmeGainSystem", PROVIDER, VERSION, "not registered"):
            self.assertIn(text, str(error))
        # The default registry does not know the type either.
        with self.assertRaises(UnknownSystemTypeError):
            self._load("missing_reloaded_default", path)

    def test_other_revision_of_the_type_does_not_load_the_model(self):
        model = _build(self._id("revision"), _registry())
        path = _serialize(model)
        registry = SystemRegistry()
        registry.register(
            AcmeGainSystem, type_id="acme:GainSystem@2", provider=PROVIDER, version="2.0.0"
        )
        with self.assertRaises(UnknownSystemTypeError) as caught:
            self._load("revision_reloaded", path, registry)
        for text in (TYPE_ID, "acme:GainSystem@2", "2.0.0", "another revision"):
            self.assertIn(text, str(caught.exception))

    def test_renamed_class_keeps_its_type_id(self):
        model = _build(self._id("renamed"), _registry())
        reference = _simulate(model)
        path = _serialize(model)
        # A later release of the provider: another class name, same type id.
        registry = SystemRegistry()
        registry.register(
            AcmeOtherSystem, type_id=TYPE_ID, provider=PROVIDER, version="1.5.0"
        )
        for suffix in ("renamed_reloaded", "renamed_reloaded_again"):
            with self.subTest(suffix=suffix):
                reloaded = self._load(suffix, path, registry)
                gain = reloaded.components["gain"]
                self.assertIs(type(gain), AcmeOtherSystem)
                self.assertAlmostEqual(float(gain.gain.get().reshape(-1)[0]), 2.5, places=6)
                self.assertEqual(_connections(reloaded), _connections(model))
                result = _simulate(reloaded)
                for key, expected in reference.items():
                    torch.testing.assert_close(result[key], expected)
                path = _serialize(reloaded)
                self.assertEqual(
                    _literals(path, "gain"),
                    {
                        TYPE_ID_KEY: [TYPE_ID],
                        PROVIDER_KEY: [PROVIDER],
                        PROVIDER_VERSION_KEY: ["1.5.0"],
                    },
                )
                self.assertEqual(_types(path, "gain"), ["AcmeOtherSystem"])
                with open(path) as f:
                    self.assertNotIn("AcmeGainSystem", f.read())

    def test_model_that_records_class_names_only(self):
        # A class that is not registered is serialized by its class name, as
        # every class was before type ids were recorded.
        model = _build(self._id("legacy"), SystemRegistry())
        reference = _simulate(model)
        path = _serialize(model)
        self.assertEqual(
            _literals(path, "gain"),
            {TYPE_ID_KEY: [], PROVIDER_KEY: [], PROVIDER_VERSION_KEY: []},
        )

        with self.assertRaises(UnknownSystemTypeError) as caught:
            self._load("legacy_unknown", path, SystemRegistry())
        self.assertEqual(caught.exception.class_name, "AcmeGainSystem")
        self.assertIsNone(caught.exception.type_id)
        for text in ("'gain'", "AcmeGainSystem", "not a built-in"):
            self.assertIn(text, str(caught.exception))

        # The registered class of that name loads it, and the model records
        # the type id from then on.
        reloaded = self._load("legacy_registered", path, _registry())
        self._assert_same(reloaded, model, reference)
        self.assertEqual(_literals(_serialize(reloaded), "gain")[TYPE_ID_KEY], [TYPE_ID])

    def test_class_set_on_the_systems_module_still_loads(self):
        model = _build(self._id("setattr"), SystemRegistry())
        reference = _simulate(model)
        path = _serialize(model)
        self.addCleanup(delattr, systems_module, "AcmeGainSystem")
        setattr(systems_module, "AcmeGainSystem", AcmeGainSystem)
        reloaded = self._load("setattr_reloaded", path, SystemRegistry())
        self._assert_same(reloaded, model, reference)

    def test_batched_model_keeps_the_registry(self):
        registry = _registry()
        model = _build(self._id("batched"), registry)
        self.model_ids.append(f"{model.id}_batched")
        batched = model.batch_components()
        self.assertIs(batched.system_registry, registry)


def fresh_process_main(model_id, path, result_file):
    """Entry point of the process ``test_round_trip_in_a_fresh_process`` starts."""
    attribute_of_systems = hasattr(systems_module, "AcmeGainSystem") or hasattr(
        tb, "AcmeGainSystem"
    )
    model = tb.Model(id=model_id, system_registry=_registry())
    model.load(filename=path, draw_semantic_model=False, draw_simulation_model=False)
    result = {key: value.tolist() for key, value in _simulate(model).items()}
    gain = model.components["gain"]
    result["class"] = type(gain).__name__
    result["gain"] = float(gain.gain.get().reshape(-1)[0])
    result["attribute_of_systems"] = attribute_of_systems
    with open(result_file, "w") as f:
        json.dump(result, f)


def leaf_sensor_pattern():
    """A Brick point with a timeseries reference is a ``SensorSystem``."""
    sensor = Node(cls=core.namespace.BRICK.Point)
    reference = Node(cls=(core.namespace.BRICKREF.ExternalReference, core.BlankNode))
    timeseries_id = Node(cls=core.namespace.XSD.string)
    sp = SignaturePattern(id="test_registry_leaf_sensor", system=tb.SensorSystem)
    sp.add_rule(
        StepRule(
            subject=sensor,
            object=reference,
            predicate=core.namespace.BRICKREF.hasExternalReference,
        )
    )
    sp.add_rule(
        StepRule(
            subject=reference,
            object=timeseries_id,
            predicate=core.namespace.BRICKREF.hasTimeseriesId,
        )
    )
    sp.add_parameter("uuid", timeseries_id)
    sp.add_modeled_node(sensor)
    return sp


def gain_pattern():
    """A Brick fan with a point is an ``AcmeGainSystem`` fed by that point."""
    fan = Node(cls=core.namespace.BRICK.Fan)
    point = Node(cls=core.namespace.BRICK.Point)
    sp = SignaturePattern(id="test_registry_gain", system=AcmeGainSystem)
    sp.add_rule(
        StepRule(subject=fan, object=point, predicate=core.namespace.BRICK.hasPoint)
    )
    sp.add_connection(sender_node=point, output_port="measuredValue", input_port="x")
    sp.add_modeled_node(fan)
    return sp


class TestExternalSystemTranslation(unittest.TestCase):
    MODEL_ID = "test_system_registry_translation"

    def setUp(self):
        self.addCleanup(_remove_models, self.MODEL_ID, self.MODEL_ID + "_reloaded")

    def _semantic_model(self):
        BRICK = core.namespace.BRICK
        REF = core.namespace.BRICKREF
        EX = core.namespace.T4B
        sm = SemanticModel(id=self.MODEL_ID, namespaces={"T4B": EX})
        g = sm.instance_graph
        reference = BNode()
        g.add((EX["flow_sensor"], RDF.type, BRICK.Point))
        g.add((EX["flow_sensor"], REF.hasExternalReference, reference))
        g.add((reference, RDF.type, REF.ExternalReference))
        g.add((reference, REF.hasTimeseriesId, Literal("uuid-flow")))
        g.add((EX["fan"], RDF.type, BRICK.Fan))
        g.add((EX["fan"], BRICK.hasPoint, EX["flow_sensor"]))
        return sm

    def test_external_class_translates_next_to_builtin_classes(self):
        registry = _registry()
        model = Translator(system_registry=registry).translate(
            self._semantic_model(),
            patterns=[leaf_sensor_pattern(), gain_pattern()],
            id=self.MODEL_ID,
        )
        self.assertEqual(
            sorted((c.id, type(c)) for c in model.components.values()),
            [("fan", AcmeGainSystem), ("flow_sensor", tb.SensorSystem)],
        )
        self.assertEqual(
            _connections(model), [("flow_sensor", "measuredValue", "fan", "x")]
        )
        # The model carries the registry of the translator ...
        self.assertIs(model.system_registry, registry)
        self.assertIs(model.simulation_model.system_registry, registry)
        # ... so the translated component is serialized under its type id.
        path = _serialize(model)
        self.assertEqual(
            _literals(path, "fan"),
            {TYPE_ID_KEY: [TYPE_ID], PROVIDER_KEY: [PROVIDER], PROVIDER_VERSION_KEY: [VERSION]},
        )
        self.assertEqual(_literals(path, "flow_sensor")[TYPE_ID_KEY], [])

        reloaded = tb.Model(id=self.MODEL_ID + "_reloaded", system_registry=_registry())
        reloaded.load(
            filename=path,
            draw_semantic_model=False,
            draw_simulation_model=False,
            validate_model=False,
        )
        self.assertIs(type(reloaded.components["fan"]), AcmeGainSystem)
        self.assertIs(type(reloaded.components["flow_sensor"]), tb.SensorSystem)
        self.assertEqual(reloaded.components["flow_sensor"].uuid, "uuid-flow")
        self.assertEqual(_connections(reloaded), _connections(model))

    def test_translator_without_a_registry_uses_the_default(self):
        model = Translator().translate(
            self._semantic_model(),
            patterns=[leaf_sensor_pattern(), gain_pattern()],
            id=self.MODEL_ID,
        )
        self.assertIs(type(model.components["fan"]), AcmeGainSystem)
        self.assertIs(model.system_registry, tb.system_registry)

    def test_registry_of_the_translator_is_a_whitelist(self):
        patterns = [leaf_sensor_pattern(), gain_pattern()]
        with self.assertRaisesRegex(ValueError, "AcmeGainSystem") as caught:
            Translator(system_registry=SystemRegistry()).translate(
                self._semantic_model(), patterns=patterns, id=self.MODEL_ID
            )
        self.assertIn("test_registry_gain", str(caught.exception))
        self.assertIn("not registered", str(caught.exception))

    def test_systems_stays_an_allow_list(self):
        patterns = [leaf_sensor_pattern(), gain_pattern()]
        for registry in (None, _registry(), SystemRegistry()):
            with self.subTest(registry=registry):
                groups = Translator._group_patterns(
                    patterns, systems=[tb.SensorSystem], registry=registry
                )
                self.assertEqual(set(groups), {tb.SensorSystem})
        groups = Translator._group_patterns(patterns, registry=_registry())
        self.assertEqual(set(groups), {tb.SensorSystem, AcmeGainSystem})
        model = Translator(system_registry=_registry()).translate(
            self._semantic_model(),
            patterns=patterns,
            systems=[tb.SensorSystem],
            id=self.MODEL_ID,
        )
        self.assertEqual(
            [(c.id, type(c)) for c in model.components.values()],
            [("flow_sensor", tb.SensorSystem)],
        )


if __name__ == "__main__":
    unittest.main()
