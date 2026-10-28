"""The rooms' outdoor CO2 port is ``outdoorCo2Concentration``, the name the
occupancy inversion and the outdoor environment use; ``outdoorCO2`` is its
deprecated name (#195).

Covered:

1. both ports read 400 ppmv while unwired (a 0 ppmv default made the
   inversion book the whole indoor concentration as people);
2. the deprecated ``outdoorCO2`` connects, reads and removes, warns, and the
   model holds ``outdoorCo2Concentration`` (the mass balance and the
   composite room);
3. ``fill_missing_inputs`` given both names wires each room once;
4. a signature pattern written with ``outdoorCO2`` translates into the same
   connections;
5. a model saved with ``outdoorCO2`` loads, simulates as before and saves
   ``outdoorCo2Concentration``.
"""

# Standard library imports
import datetime
import os
import re
import shutil
import tempfile
import unittest
import warnings

# Third party imports
import torch
from dateutil import tz

# Local application imports
import twin4build as tb
import twin4build.core as core
import twin4build.examples.utils as example_utils
from twin4build.examples import patterns as example_patterns
from twin4build.utils.logger import LOGGER

tb._IS_TESTING = True

START = datetime.datetime(2024, 1, 4, tzinfo=tz.UTC)
OLD, NEW = "outdoorCO2", "outdoorCo2Concentration"


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


def _room(sid="room"):
    return tb.BuildingSpaceSystem(
        id=sid,
        thermal_kwargs=dict(C_wall=1e6, C_air=1e5, R_out=0.01, R_in=0.02),
        mass_kwargs=dict(V=100.0),
    )


def _series(port):
    hist = port.history()
    return hist.reshape(hist.shape[0], -1)[:, 0].detach().cpu().clone()


class TestDefaults(unittest.TestCase):
    def test_unwired_ports_read_400_ppmv(self):
        for component in (
            tb.BuildingSpaceMassSystem(V=100.0, G_occ=5e-6, m_inf=0.004, id="mass"),
            _room(),
            tb.OccupancySystem(id="occupancy"),
        ):
            port = component.input[NEW]
            port.initialize(n_t=2)
            torch.testing.assert_close(port.get(), torch.full_like(port.get(), 400.0), msg=component.id)


class TestDeprecatedName(unittest.TestCase):
    def test_connection_through_the_old_name_holds_the_new_name(self):
        for component in (tb.BuildingSpaceMassSystem(V=100.0, G_occ=5e-6, m_inf=0.004, id="mass"), _room()):
            model = tb.Model(id=f"outdoor_co2_alias_{component.id}")
            source = _schedule(400.0, "c_out")
            name = type(component).__name__
            with self.assertWarnsRegex(DeprecationWarning, f"{name}.{NEW}") as caught:
                model.add_connection(source, component, "scheduleValue", OLD)
            self.assertEqual(caught.filename, __file__)  # points at the caller
            self.assertEqual([p.input_port for p in component.connects_at], [NEW])
            with self.assertWarns(DeprecationWarning):
                model.remove_connection(source, component, "scheduleValue", OLD)
            self.assertEqual(component.connects_at, [])

    def test_input_mapping_answers_to_the_old_name(self):
        room = _room()
        with self.assertWarnsRegex(DeprecationWarning, NEW) as caught:
            port = room.input[OLD]
        self.assertEqual(caught.filename, __file__)
        self.assertIs(port, room.input[NEW])
        self.assertIs(port, room.mass.input[NEW])  # one port object for the room and its mass balance
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self.assertIn(OLD, room.input)  # no warning
            self.assertNotIn(OLD, list(room.input))
            self.assertNotIn(OLD, list(room.mass.input))
            # The occupancy never had the old name.
            occupancy = tb.OccupancySystem(id="occupancy")
            self.assertEqual(occupancy.resolve_input_port(OLD), OLD)
            self.assertNotIn(OLD, occupancy.input)

    def test_fill_missing_inputs_given_both_names_wires_the_room_once(self):
        model = tb.Model(id="outdoor_co2_fill")
        room = _room()
        occupancy = tb.OccupancySystem(id="occupancy")
        model.add_component(room)
        model.add_component(occupancy)
        with self.assertWarnsRegex(DeprecationWarning, NEW):
            model.simulation_model.fill_missing_inputs({OLD: 400.0, NEW: 410.0})
        sources = {
            c.id: [(p.input_port, [s.connects_system.id for s in p.connects_system_through]) for p in c.connects_at]
            for c in (room, occupancy)
        }
        self.assertEqual(sources["room"], [(NEW, [f"_sched_{OLD}"])])
        self.assertEqual(sources["occupancy"], [(NEW, [f"_sched_{NEW}"])])


class TestPatternWithOldName(unittest.TestCase):
    def test_pattern_written_with_the_old_name_translates_the_same(self):
        """A signature pattern that fills the room's ``outdoorCO2`` (an add-on
        package's pattern) still translates, into the same connections."""
        filename = example_utils.get_path(["estimator_example", "one_room_example_model.xlsm"])

        def connections(model):
            ids = lambda c: re.sub(r"\[N[0-9a-f]{32}\]", "[bnode]", c.id)
            return sorted(
                (ids(c), connection.output_port, ids(point.connection_point_of), point.input_port)
                for c in model.components.values()
                for connection in c.connected_through
                for point in connection.connects_system_at
            )

        current = tb.Translator().translate(
            core.SemanticModel(rdf_file=filename, id="outdoor_co2_pattern_new"),
            patterns=example_patterns.default_patterns(),
            id="outdoor_co2_pattern_new",
        )
        self.assertIn(NEW, {c[3] for c in connections(current)})
        old_patterns = example_patterns.default_patterns()
        renamed = 0
        for sp in old_patterns:
            if sp.system is tb.BuildingSpaceSystem and NEW in sp.inputs:
                sp.inputs[OLD] = sp.inputs.pop(NEW)
                renamed += 1
        self.assertGreater(renamed, 0)
        with self.assertWarnsRegex(DeprecationWarning, f"BuildingSpaceSystem.{NEW}"):
            old = tb.Translator().translate(
                core.SemanticModel(rdf_file=filename, id="outdoor_co2_pattern_old"),
                patterns=old_patterns,
                id="outdoor_co2_pattern_old",
            )
        self.assertEqual(connections(old), connections(current))


class TestModelSavedWithOldName(unittest.TestCase):
    MODEL_ID = "outdoor_co2_saved"

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)
        for suffix in ("", "_old"):
            shutil.rmtree(os.path.join("generated_files", "models", self.MODEL_ID + suffix), ignore_errors=True)

    def _build(self):
        model = tb.Model(id=self.MODEL_ID)
        room = _room()
        for port, value in (
            ("outdoorTemperature", 5.0), ("numberOfPeople", 2.0), ("globalIrradiation", 0.0),
            ("supplyAirFlowRate", 0.2), ("supplyAirTemperature", 18.0), ("heatGain", 0.0), (NEW, 420.0),
        ):
            model.add_connection(_schedule(value, f"s_{port}"), room, "scheduleValue", port)
        model.load(draw_semantic_model=False, draw_simulation_model=False)
        return model

    @staticmethod
    def _run(model):
        tb.Simulator(model, execution_mode="object").simulate(
            start_time=START,
            end_time=START + datetime.timedelta(hours=3),
            step_size=600,
            show_progress_bar=False,
        )
        return _series(model.components["room"].output["indoorCO2"])

    @staticmethod
    def _input_ports(path):
        with open(path, encoding="utf-8") as handle:
            return re.findall(r't4b:input_port "([^"]+)"', handle.read())

    def test_old_file_loads_simulates_and_resaves_the_new_name(self):
        model = self._build()
        reference = self._run(model)
        model.serialize()
        path, _ = model._simulation_model._semantic_model.get_dir(filename="instance_graph.ttl")
        self.assertIn(NEW, self._input_ports(path))

        # A file written before the rename records the room's ``outdoorCO2``.
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
        old_text = text.replace(f't4b:input_port "{NEW}"', f't4b:input_port "{OLD}"')
        self.assertNotEqual(old_text, text)
        old_path = os.path.join(self.tmp, "instance_graph.ttl")
        with open(old_path, "w", encoding="utf-8") as handle:
            handle.write(old_text)

        loaded = tb.Model(id=self.MODEL_ID + "_old")
        # a quiet load (LOGGER.verbose 0, as an example run elsewhere in the
        # suite leaves it) silences every warning: load verbosely here
        verbose = LOGGER.verbose
        LOGGER.verbose = 3
        try:
            with self.assertWarnsRegex(DeprecationWarning, f"BuildingSpaceSystem.{NEW}"):
                loaded.load(filename=old_path, draw_semantic_model=False, draw_simulation_model=False)
        finally:
            LOGGER.verbose = verbose
        room = loaded.components["room"]
        self.assertIn(NEW, [p.input_port for p in room.connects_at])
        torch.testing.assert_close(self._run(loaded), reference)

        loaded.serialize()
        resaved, _ = loaded._simulation_model._semantic_model.get_dir(filename="instance_graph.ttl")
        ports = self._input_ports(resaved)
        self.assertIn(NEW, ports)
        self.assertNotIn(OLD, ports)


if __name__ == "__main__":
    unittest.main()
