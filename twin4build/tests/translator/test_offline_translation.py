"""Regression tests for issue #114: translation must not depend on the
network.

Ontology class hierarchies used to be downloaded on every model load, and a
failed download was silently swallowed - signature-pattern matching then
degraded and the translator dropped components from the model (the flaky
``KeyError: 'office_temperature_heating_setpoint'`` in
``test_translator_example``).  The ontologies are now vendored as package
data and parsed from disk; these tests pin that behavior.
"""

# Standard library imports
import os
import shutil
import unittest
import warnings

# Third party imports
import rdflib

# Local application imports
import twin4build
import twin4build.core as core
import twin4build.examples.utils as utils
import twin4build.model.semantic_model.semantic_model as sm_mod
from twin4build.model.semantic_model.semantic_model import SemanticModel
from twin4build.translator.translator import Translator

twin4build._IS_TESTING = True

EXPECTED_COMPONENTS = [
    "office",
    "office_co2_controller",
    "office_co2_sensor",
    "office_co2_setpoint",
    "office_damper_position_sensor",
    "office_exhaust_damper",
    "office_occupancy_profile",
    "office_space_heater",
    "office_space_heater_valve",
    "office_supply_damper",
    "office_temperature_heating_controller",
    "office_temperature_heating_setpoint",
    "office_temperature_sensor",
    "office_valve_position_sensor",
    "outdoor_environment",
    "supply_air_temperature_sensor",
]


class NetworkBlocked(Exception):
    pass


def _no_network_parse_wrapper(graph, source=None, **kwargs):
    src = str(source)
    if src.startswith(("http://", "https://")):
        raise NetworkBlocked(f"remote ontology fetch attempted: {src}")
    return _ORIG_PARSE_WRAPPER(graph, source=source, **kwargs)


_ORIG_PARSE_WRAPPER = sm_mod.parse_wrapper


class TestVendoredOntologies(unittest.TestCase):
    """The vendored ontology snapshots exist and parse."""

    VENDORED = [
        "FSO", "SAREF", "S4BLDG", "S4SYST", "BRICK", "BOT",
        "RDF", "RDFS", "OWL", "REC",
    ]

    def test_files_exist_and_parse(self):
        for name in self.VENDORED:
            path = getattr(core.ontology, name)
            self.assertTrue(
                os.path.isfile(path), f"vendored ontology missing: {path}"
            )
            g = rdflib.Graph()
            g.parse(path)
            self.assertGreater(
                len(g), 0, f"vendored ontology parsed empty: {path}"
            )

    def test_local_source_lookup(self):
        # Namespace-URI-keyed lookup used by parse_namespaces.
        self.assertEqual(
            core.ontology.local_source(core.namespace.SAREF),
            core.ontology.SAREF,
        )
        self.assertIsNone(core.ontology.local_source("http://unknown.org/ns#"))


class TestOfflineTranslation(unittest.TestCase):
    """Full example-model translation with ALL remote ontology fetches blocked
    must still produce the complete component set."""

    MODEL_ID = "test_offline_translation_model"

    @classmethod
    def setUpClass(cls):
        sm_mod.parse_wrapper = _no_network_parse_wrapper

    @classmethod
    def tearDownClass(cls):
        sm_mod.parse_wrapper = _ORIG_PARSE_WRAPPER
        path = os.path.join("generated_files", "models", cls.MODEL_ID)
        if os.path.exists(path):
            shutil.rmtree(path)

    def test_offline_translation_is_complete(self):
        filename = utils.get_path(
            ["estimator_example", "one_room_example_model.xlsm"]
        )
        sm = SemanticModel(rdf_file=filename, id=self.MODEL_ID)
        translated = Translator().translate(sm)

        self.assertEqual(sorted(translated.components.keys()), EXPECTED_COMPONENTS)
        self.assertEqual(
            sm.error_namespaces,
            set(),
            "some namespaces failed to parse offline - vendored ontologies "
            f"are incomplete: {sm.error_namespaces}",
        )


class TestFailedNamespaceWarns(unittest.TestCase):
    """A namespace that fails every source must raise a visible UserWarning
    instead of silently degrading pattern matching."""

    def test_total_parse_failure_warns(self):
        sm = SemanticModel()

        def _always_fail(graph, source=None, **kwargs):
            raise Exception("simulated parse failure")

        sm_mod.parse_wrapper = _always_fail
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                sm.parse_namespaces(
                    namespaces={"fso": core.namespace.FSO}
                )
        finally:
            sm_mod.parse_wrapper = _ORIG_PARSE_WRAPPER

        messages = [
            str(w.message) for w in caught if issubclass(w.category, UserWarning)
        ]
        self.assertTrue(
            any("Failed to parse ontology namespace" in m for m in messages),
            f"expected a loud UserWarning, got: {messages}",
        )
        self.assertIn(str(core.namespace.FSO), sm.error_namespaces)


if __name__ == "__main__":
    unittest.main()
