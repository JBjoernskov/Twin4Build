"""A component rejects a keyword argument its constructor does not take.

``System.__init__`` used to drop every keyword it did not know, so a misspelled
or removed keyword built a different component without a word
(``PIDControllerSystem(isReverse=True)``: a direct-acting loop;
``BuildingSpaceSystem(C_air=...)``: the default capacitance).  The translator
and the model loader build components from a node's literals, which also name
properties a class does not take: they pass only
:meth:`System.constructor_keywords`.

Covered:

1. an unknown keyword raises ``TypeError`` naming it and the class;
2. the constructor keywords of a class: its own, its parents' that it hands on,
   and :class:`System`'s;
3. a deprecated keyword a class translates itself still works, with a warning;
4. the translator builds the example graph and leaves out the literals a class
   does not take;
5. a saved model holding a literal no constructor takes still loads.
"""

# Standard library imports
import os
import re
import shutil
import tempfile
import unittest
import warnings
from unittest import mock

# Local application imports
import twin4build as tb
import twin4build.core as core
import twin4build.examples.utils as example_utils
import twin4build.translator.translator as translator_module
from twin4build.examples import patterns as example_patterns
from twin4build.utils.logger import LOGGER

tb._IS_TESTING = True


def _schedule(value, sid):
    return tb.ScheduleSystem(
        weekday_ruleset={
            "ruleset_default_value": value,
            "ruleset_start_minute": [],
            "ruleset_end_minute": [],
            "ruleset_start_hour": [],
            "ruleset_end_hour": [],
            "ruleset_value": [],
        },
        id=sid,
    )


class TestUnknownKeywords(unittest.TestCase):
    def test_an_unknown_keyword_raises_naming_it(self):
        with self.assertRaisesRegex(TypeError, r"PIDControllerSystem.*'isReverse'"):
            tb.PIDControllerSystem(id="pid", isReverse=True)
        # the composite room takes its submodels' values through thermal_kwargs / mass_kwargs only
        with self.assertRaisesRegex(TypeError, r"BuildingSpaceSystem.*'C_air'"):
            tb.BuildingSpaceSystem(id="room", C_air=1e5)
        room = tb.BuildingSpaceSystem(id="room", thermal_kwargs={"C_air": 1e5})
        self.assertAlmostEqual(float(room.thermal.C_air.get()), 1e5, delta=1e-6)

    def test_constructor_keywords_follow_the_chain(self):
        pid = tb.PIDControllerSystem.constructor_keywords()
        self.assertTrue({"kp", "Ti", "Td", "is_reverse", "id"} <= pid)
        self.assertNotIn("isReverse", pid)
        self.assertEqual(
            tb.BuildingSpaceSystem.constructor_keywords() & {"thermal_kwargs", "mass_kwargs", "id"},
            {"thermal_kwargs", "mass_kwargs", "id"},
        )

    def test_a_deprecated_keyword_a_class_translates_still_works(self):
        ruleset = {
            "ruleset_default_value": 1.0,
            "ruleset_start_minute": [],
            "ruleset_end_minute": [],
            "ruleset_start_hour": [],
            "ruleset_end_hour": [],
            "ruleset_value": [],
        }
        with self.assertWarns(DeprecationWarning):
            schedule = tb.ScheduleSystem(weekDayRulesetDict=ruleset, id="legacy")
        self.assertEqual(schedule.weekday_ruleset["ruleset_default_value"], 1.0)
        # the loader and the translator pass it on too (a model saved with the old name loads)
        self.assertIn("weekDayRulesetDict", tb.ScheduleSystem.constructor_keywords())


class TestTranslatorPassesOnlyConstructorKeywords(unittest.TestCase):
    def test_example_graph_translates_without_the_literals_a_class_does_not_take(self):
        filename = example_utils.get_path(["estimator_example", "one_room_example_model.xlsm"])
        messages = []

        def debug(message, *args, **kwargs):
            if isinstance(message, str) and "takes no" in message:
                messages.append(message % args)

        with mock.patch.object(translator_module.LOGGER, "debug", side_effect=debug):
            model = tb.Translator().translate(
                core.SemanticModel(rdf_file=filename, id="constructor_keywords_translate"),
                patterns=example_patterns.default_patterns(),
                id="constructor_keywords_translate",
            )
        self.assertTrue(model.components)
        self.assertTrue(messages, "the example graph's nodes carry literals no constructor takes")


class TestSavedModelWithAStaleLiteral(unittest.TestCase):
    MODEL_ID = "constructor_keywords_saved"

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)
        for suffix in ("", "_old"):
            shutil.rmtree(os.path.join("generated_files", "models", self.MODEL_ID + suffix), ignore_errors=True)

    def test_a_literal_no_constructor_takes_is_left_out_on_load(self):
        model = tb.Model(id=self.MODEL_ID)
        controller = tb.PIDControllerSystem(id="pid", kp=0.5)
        model.add_connection(_schedule(20.0, "setpoint"), controller, "scheduleValue", "setpointValue")
        model.add_connection(_schedule(19.0, "measured"), controller, "scheduleValue", "actualValue")
        model.load(draw_semantic_model=False, draw_simulation_model=False)
        model.serialize()
        path, _ = model._simulation_model._semantic_model.get_dir(filename="instance_graph.ttl")
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
        # the PID controller's node, as an older model might have recorded it: one more literal
        block = re.search(r"(?m)^(\S*pid\S* a [^\n]*\n)", text)
        self.assertIsNotNone(block)
        old_text = text.replace(block.group(1), block.group(1) + '    t4b:isReverse "true" ;\n', 1)
        old_path = os.path.join(self.tmp, "instance_graph.ttl")
        with open(old_path, "w", encoding="utf-8") as handle:
            handle.write(old_text)
        loaded = tb.Model(id=self.MODEL_ID + "_old")
        verbose = LOGGER.verbose
        LOGGER.verbose = 0
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                loaded.load(filename=old_path, draw_semantic_model=False, draw_simulation_model=False)
        finally:
            LOGGER.verbose = verbose
        self.assertAlmostEqual(float(loaded.components["pid"].kp.get()), 0.5)


if __name__ == "__main__":
    unittest.main()
