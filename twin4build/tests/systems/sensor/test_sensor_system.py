"""The sensor as a declared System (ADR 0001, decision 6).

Covered:

1. a sensor with nothing wired reads its recorded series: ``measuredData``
   and ``measuredValue`` are the series, in object mode and in the
   functional engine;
2. a wired sensor reads what is wired to it; ``measuredData`` stays the
   series;
3. a sensor without data and with nothing wired raises at the run start;
4. the reader: ``source``, or the database point ``uuid`` names;
   ``set_dbconfig`` gives a uuid-only sensor a database reader;
5. the 1.x keywords and attributes raise and name their replacement; a
   saved ``transformation_ref`` loads as ``transformation``, with a warning;
6. the gap policy: a sensor that measures a model output may not fill, one
   that drives the model may not leave gaps unscored.
"""

# Standard library imports
import datetime
import math
import os
import shutil
import unittest

# Third party imports
import pandas as pd
from dateutil import tz

# Local application imports
import twin4build as tb
from twin4build.systems.sensor.sensor_system import SensorSystem
from twin4build.tests.model.test_batching_vector_slots import Sink
from twin4build.utils.data_loaders.sources import DatabaseSource, DataFrameSource

tb._IS_TESTING = True

START = datetime.datetime(2023, 1, 1, tzinfo=tz.UTC)
STEP = 600


def _frame(values):
    index = pd.date_range(start=START, periods=len(values), freq="10min")
    return pd.DataFrame({"value": [float(v) for v in values]}, index=index)


def _constant(value, sid):
    return tb.ScheduleSystem(
        id=sid,
        weekday_ruleset={
            "ruleset_default_value": value,
            "ruleset_start_minute": [],
            "ruleset_end_minute": [],
            "ruleset_start_hour": [],
            "ruleset_end_hour": [],
            "ruleset_value": [],
        },
    )


def _simulate(model, mode="object", steps=6):
    tb.Simulator(model, execution_mode=mode).simulate(
        start_time=START,
        end_time=START + datetime.timedelta(seconds=steps * STEP),
        step_size=STEP,
        show_progress_bar=False,
    )


def _values(port):
    return port.history().reshape(-1).detach().cpu().tolist()


def _remove(*model_ids):
    for model_id in model_ids:
        for suffix in ("", "_simulation_model"):
            shutil.rmtree(os.path.join("generated_files", "models", model_id + suffix), ignore_errors=True)


class TestReading(unittest.TestCase):
    def tearDown(self):
        _remove("sensor_leaf", "sensor_wired", "sensor_no_data")

    def test_nothing_wired_reads_the_recorded_series_in_both_engines(self):
        series = [20.0, 21.0, 22.5, 23.0, 22.0, 21.5]
        for mode in ("object", "functional"):
            with self.subTest(mode=mode):
                model = tb.Model(id="sensor_leaf")
                sensor = SensorSystem(id="sensor", source=DataFrameSource(_frame(series)))
                sink = Sink(id="sink")  # a state, so the functional engine has one to roll
                model.add_connection(sensor, sink, "measuredValue", "u", input_port_index=0)
                model.load(draw_semantic_model=False, draw_simulation_model=False)
                _simulate(model, mode)
                self.assertEqual(_values(sensor.output["measuredData"]), series)
                self.assertAlmostEqual(_values(sink.output["w"])[-1], 3.0 * sum(series))
                self.assertTrue(sensor.is_leaf)
                self.assertIsInstance(sensor.time_series_input, tb.TimeSeriesInputSystem)
        _simulate(model, "object")
        self.assertEqual(_values(sensor.output["measuredValue"]), series)

    def test_a_wired_sensor_reads_what_is_wired(self):
        model = tb.Model(id="sensor_wired")
        sensor = SensorSystem(id="sensor", source=DataFrameSource(_frame([5.0] * 6)))
        model.add_connection(_constant(2.0, "signal"), sensor, "scheduleValue", "measuredValue")
        model.load(draw_semantic_model=False, draw_simulation_model=False)
        _simulate(model)
        self.assertEqual(_values(sensor.output["measuredValue"]), [2.0] * 6)
        self.assertEqual(_values(sensor.output["measuredData"]), [5.0] * 6)
        self.assertFalse(sensor.is_leaf)

    def test_no_data_and_nothing_wired_raises(self):
        model = tb.Model(id="sensor_no_data")
        model.add_component(SensorSystem(id="sensor"))
        model.load(draw_semantic_model=False, draw_simulation_model=False)
        with self.assertRaisesRegex(ValueError, "nothing is wired to measuredValue"):
            _simulate(model)


class TestReader(unittest.TestCase):
    def test_the_reader_is_the_source_or_the_uuids_point(self):
        frame_reader = DataFrameSource(_frame([1.0] * 6))
        self.assertIs(SensorSystem(id="s", source=frame_reader, uuid="X").reader, frame_reader)
        self.assertIsNone(SensorSystem(id="s").reader)
        self.assertFalse(SensorSystem(id="s").has_data)
        point = SensorSystem(id="s", uuid="X")
        self.assertIsInstance(point.reader, DatabaseSource)
        self.assertEqual(point.reader.uuid, "X")
        self.assertTrue(point.has_data)
        point.set_dbconfig({"host": "historian"})
        self.assertIsInstance(point.source, DatabaseSource)
        self.assertEqual(point.source.dbconfig, {"host": "historian"})
        # a string is a database point's id, as a translator pattern gives it
        self.assertEqual(SensorSystem(id="s", source="Y").source.uuid, "Y")


class TestRemovedKeywords(unittest.TestCase):
    def test_the_1x_keywords_and_attributes_name_their_replacement(self):
        with self.assertRaisesRegex(TypeError, "DataFrameSource"):
            SensorSystem(id="s", df=_frame([1.0] * 6))
        with self.assertRaisesRegex(TypeError, "CsvSource"):
            SensorSystem(id="s", filename="room.csv")
        with self.assertRaisesRegex(TypeError, "reading options"):
            SensorSystem(id="s", fill="constant")
        sensor = SensorSystem(id="s")
        with self.assertRaisesRegex(AttributeError, "CsvSource"):
            sensor.filename = "room.csv"
        with self.assertRaisesRegex(AttributeError, "source="):
            sensor.use_df
        with self.assertRaises(TypeError):
            SensorSystem(id="s", usedf=True)

    def test_a_saved_transformation_ref_loads_as_the_transformation(self):
        with self.assertWarns(DeprecationWarning):
            sensor = SensorSystem(id="s", transformation_ref="math:sqrt")
        self.assertIs(sensor.transformation, math.sqrt)


class TestGaps(unittest.TestCase):
    def tearDown(self):
        _remove("sensor_scored_fill", "sensor_input_gaps")

    def test_a_scored_sensor_does_not_fill(self):
        model = tb.Model(id="sensor_scored_fill")
        sensor = SensorSystem(id="sensor", source=DataFrameSource(_frame([5.0] * 6), fill="constant"))
        model.add_connection(_constant(2.0, "signal"), sensor, "scheduleValue", "measuredValue")
        model.load(draw_semantic_model=False, draw_simulation_model=False)
        with self.assertRaisesRegex(ValueError, "a fill would invent measurements"):
            _simulate(model)

    def test_a_sensor_that_drives_the_model_leaves_no_gap_unscored(self):
        model = tb.Model(id="sensor_input_gaps")
        sensor = SensorSystem(id="sensor", source=DataFrameSource(_frame([5.0] * 6)), allow_missing=True)
        model.add_connection(sensor, Sink(id="sink"), "measuredValue", "u", input_port_index=0)
        model.load(draw_semantic_model=False, draw_simulation_model=False)
        with self.assertRaisesRegex(ValueError, "drives the model"):
            _simulate(model)


if __name__ == "__main__":
    unittest.main()
