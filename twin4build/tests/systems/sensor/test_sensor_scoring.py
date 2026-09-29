"""What a sensor is scored with, and a sensor on an in-memory series.

``scoring_mask`` (the samples an estimation scores) and ``measurement_sd``
(the standard deviation it is scored with) are attributes of the sensor:
they travel with it into a batched model and the estimator reads them.
``set_series`` puts a sensor on an in-memory series and gives it an
identity (``uuid``) without switching it to database mode, and the sensor
stays on its series when ``Model.load`` restores its saved configuration.
"""
import datetime
import os
import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch
from dateutil import tz

import twin4build as tb
from twin4build.systems.sensor.sensor_system import SensorSystem
from twin4build.utils.get_main_dir import get_main_dir, set_main_dir

tb._IS_TESTING = True

from twin4build.tests.simulator.test_fusion_batched import START, STEP, build, history, simulate

HOURS = 12
END = START + datetime.timedelta(hours=HOURS)


def setUpModule():
    """The models of this module keep their files in a temporary folder."""
    global _MAIN_DIR, _FILES
    _MAIN_DIR = get_main_dir()
    _FILES = tempfile.mkdtemp()
    set_main_dir(_FILES)


def tearDownModule():
    set_main_dir(_MAIN_DIR)
    shutil.rmtree(_FILES, ignore_errors=True)


def index(n, start=START):
    return pd.DatetimeIndex([start + datetime.timedelta(seconds=STEP * k) for k in range(n)], name="time")


def frame(values, start=START):
    values = np.asarray(values, dtype=float)
    return pd.DataFrame({"value": values}, index=index(len(values), start))


def readings(sensor):
    """The values the sensor reads over the first six steps."""
    sensor.initialize([START], [START + datetime.timedelta(seconds=6 * STEP)], [STEP])
    return sensor.time_series_input.values.reshape(-1).tolist()


def room_model(n, model_id, data=None, **sensor_kwargs):
    """``n`` heated zones, each with a sensor on its temperature."""
    model = build(n_pairs=n, model_id=model_id)
    for k in range(n):
        values = np.full(HOURS * 6, 20.0) if data is None else data[k]
        sensor = SensorSystem(id=f"T{k}", df=frame(values), uuid=f"ROOM{k}_T", **sensor_kwargs)
        model.add_connection(model.components[f"Zone{k}"], sensor, "indoorTemperature", "measuredValue")
    model.load(draw_semantic_model=False, draw_simulation_model=False)
    return model


class TestScoringAttributes(unittest.TestCase):
    def test_defaults(self):
        sensor = SensorSystem(id="s", df=frame([20.0] * 6))
        self.assertIsNone(sensor.scoring_mask)
        self.assertIsNone(sensor.measurement_sd)

    def test_set_at_construction_and_afterwards(self):
        mask = pd.Series(True, index=index(6))
        sensor = SensorSystem(id="s", df=frame([20.0] * 6), scoring_mask=mask, measurement_sd=0.3)
        self.assertIs(sensor.scoring_mask, mask)
        self.assertEqual(sensor.measurement_sd, 0.3)
        other = pd.Series(False, index=index(6))
        sensor.scoring_mask = other
        sensor.measurement_sd = 2
        self.assertIs(sensor.scoring_mask, other)
        self.assertEqual(sensor.measurement_sd, 2.0)
        self.assertIsInstance(sensor.measurement_sd, float)
        sensor.scoring_mask = None
        sensor.measurement_sd = None
        self.assertIsNone(sensor.scoring_mask)
        self.assertIsNone(sensor.measurement_sd)

    def test_deleting_the_mask_clears_it(self):
        """``del sensor.scoring_mask`` worked while the mask was a plain
        attribute; it still does."""
        sensor = SensorSystem(id="s", df=frame([20.0] * 6))
        sensor.scoring_mask = np.ones(6, dtype=bool)
        del sensor.scoring_mask
        self.assertIsNone(sensor.scoring_mask)

    def test_a_standard_deviation_is_positive(self):
        sensor = SensorSystem(id="s", df=frame([20.0] * 6))
        for value in (0.0, -0.1):
            with self.assertRaises(ValueError):
                sensor.measurement_sd = value
        with self.assertRaises(ValueError):
            SensorSystem(id="s2", df=frame([20.0] * 6), measurement_sd=0.0)


class TestInMemorySeries(unittest.TestCase):
    def test_a_frame_and_a_uuid_give_an_in_memory_sensor(self):
        sensor = SensorSystem(id="s", df=frame([20.0, 21.0, 22.0, 23.0, 24.0, 25.0]), uuid="X")
        self.assertEqual(sensor.uuid, "X")
        self.assertTrue(sensor.use_df)
        self.assertFalse(sensor.use_database)
        self.assertFalse(sensor.use_spreadsheet)
        self.assertEqual(readings(sensor), [20.0, 21.0, 22.0, 23.0, 24.0, 25.0])

    def test_a_uuid_alone_is_still_a_database_source(self):
        sensor = SensorSystem(id="s", uuid="X")
        self.assertTrue(sensor.use_database)
        self.assertFalse(sensor.use_df)
        sensor = SensorSystem(id="s", df=frame([20.0] * 6))
        sensor.uuid = "X"  # the setter switches to the database, as before
        self.assertTrue(sensor.use_database)
        self.assertFalse(sensor.use_df)

    def test_a_frame_and_a_database_are_still_ambiguous(self):
        with self.assertRaises(AssertionError):
            SensorSystem(id="s", df=frame([20.0] * 6), uuid="X", dbconfig={"host": "h"})

    def test_set_series_takes_a_series(self):
        sensor = SensorSystem(id="s", uuid="DB_POINT", dbconfig={"host": "h"})
        self.assertTrue(sensor.use_database)
        series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], index=index(6))
        sensor.set_series(series)
        self.assertTrue(sensor.use_df)
        self.assertFalse(sensor.use_database)
        self.assertFalse(sensor.use_spreadsheet)
        self.assertEqual(sensor.uuid, "DB_POINT")  # kept
        self.assertTrue(sensor.has_data)
        self.assertEqual(readings(sensor), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    def test_set_series_names_the_sensor_without_a_database(self):
        sensor = SensorSystem(id="s", df=frame([20.0] * 6))
        self.assertIsNone(sensor.uuid)
        sensor.set_series(frame([7.0] * 6), uuid="UNIT_supplyAirTemperature")
        self.assertEqual(sensor.uuid, "UNIT_supplyAirTemperature")
        self.assertTrue(sensor.use_df)
        self.assertFalse(sensor.use_database)
        self.assertEqual(readings(sensor), [7.0] * 6)

    def test_set_series_replaces_the_series_of_an_initialized_sensor(self):
        sensor = SensorSystem(id="s", df=frame([20.0] * 6))
        self.assertEqual(readings(sensor), [20.0] * 6)
        sensor.set_series(pd.Series([3.0] * 6, index=index(6), name="flow"))
        self.assertEqual(readings(sensor), [3.0] * 6)

    def test_set_series_refuses_what_is_not_a_time_series(self):
        sensor = SensorSystem(id="s", df=frame([20.0] * 6))
        with self.assertRaises(TypeError):
            sensor.set_series([1.0, 2.0])
        with self.assertRaises(TypeError):
            sensor.set_series(pd.Series([1.0, 2.0]))  # a RangeIndex
        with self.assertRaises(ValueError):
            sensor.set_series(pd.DataFrame({"a": [1.0] * 6, "b": [2.0] * 6}, index=index(6)))
        self.assertEqual(readings(sensor), [20.0] * 6)  # unchanged


class TestSavedConfiguration(unittest.TestCase):
    """``Model.load`` writes every component's configuration to the model's
    folder and, when the folder holds one already (a second run, a second
    load), restores it by assignment.  Restoring the ``uuid``, ``dbconfig``
    or ``filename`` a sensor already has does not switch its source."""

    def setUp(self):
        self.main_dir = get_main_dir()
        self.tmp = tempfile.mkdtemp()
        set_main_dir(self.tmp)

    def tearDown(self):
        set_main_dir(self.main_dir)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _model(self, sensor, model_id):
        """A heated zone whose radiator reads its water temperature from
        ``sensor``."""
        model = build(n_pairs=1, model_id=model_id)
        radiator = model.components["Radiator0"]
        model.remove_connection(model.components["WaterTemp"], radiator, "scheduleValue", "supplyWaterTemperature")
        model.add_connection(sensor, radiator, "measuredValue", "supplyWaterTemperature")
        model.load(draw_semantic_model=False, draw_simulation_model=False)
        return model

    def _assert_in_memory(self, model, value):
        sensor = model.components["water"]
        self.assertTrue(sensor.use_df)
        self.assertFalse(sensor.use_database)
        self.assertFalse(sensor.use_spreadsheet)
        simulate(model, hours=2)
        read = sensor.output["measuredValue"].history().reshape(-1)
        self.assertTrue(bool((read == value).all()))

    def test_a_named_in_memory_sensor_stays_in_memory(self):
        for run in range(2):  # the second run finds the configuration the first one saved
            sensor = SensorSystem(id="water", df=frame([55.0] * 20), uuid="WATER_T")
            model = self._model(sensor, "saved_named")
            saved = os.path.join(
                self.tmp, "generated_files", "models", "saved_named", "simulation_model",
                "model_parameters", "SensorSystem", "water.json",
            )
            self.assertTrue(os.path.isfile(saved))
            model.load(draw_semantic_model=False, draw_simulation_model=False)
            self.assertEqual(model.components["water"].uuid, "WATER_T")
            self._assert_in_memory(model, 55.0)

    def test_a_database_sensor_put_on_a_series_stays_on_it(self):
        sensor = SensorSystem(id="water", uuid="WATER_T", dbconfig={"host": "historian"})
        self.assertTrue(sensor.use_database)
        sensor.set_series(frame([57.0] * 20))
        model = self._model(sensor, "saved_database")
        model.load(draw_semantic_model=False, draw_simulation_model=False)
        self.assertEqual(sensor.uuid, "WATER_T")
        self.assertEqual(sensor.dbconfig, {"host": "historian"})
        self._assert_in_memory(model, 57.0)
        # a new uuid or database configuration is a new source, as before
        sensor.dbconfig = {"host": "another"}
        self.assertTrue(sensor.use_database)
        self.assertFalse(sensor.use_df)

    def test_a_spreadsheet_sensor_put_on_a_series_stays_on_it(self):
        filename = os.path.join(self.tmp, "water.csv")
        frame([50.0] * 20).to_csv(filename)
        sensor = SensorSystem(id="water", filename=filename)
        self.assertTrue(sensor.use_spreadsheet)
        sensor.set_series(frame([58.0] * 20))
        model = self._model(sensor, "saved_spreadsheet")
        model.load(draw_semantic_model=False, draw_simulation_model=False)
        self.assertEqual(sensor.filename, filename)
        self._assert_in_memory(model, 58.0)
        other = os.path.join(self.tmp, "other.csv")
        frame([51.0] * 20).to_csv(other)
        sensor.filename = other
        self.assertTrue(sensor.use_spreadsheet)
        self.assertFalse(sensor.use_df)


class TestBatchingCarriesTheScoring(unittest.TestCase):
    def test_the_batched_sensor_has_the_mask_and_the_sd(self):
        model = room_model(2, "scoring_batched")
        masks = []
        for k in range(2):
            sensor = model.components[f"T{k}"]
            mask = pd.Series(True, index=index(HOURS * 6))
            mask.iloc[k : k + 3] = False
            sensor.scoring_mask = mask
            sensor.measurement_sd = 0.1 * (k + 1)
            masks.append(mask)
        batched = model.batch_components()
        for k in range(2):
            stand_in, i_c = model.get_batched_component_info(f"T{k}")
            self.assertIs(stand_in, batched.components[f"T{k}"])
            self.assertIsNot(stand_in, model.components[f"T{k}"])
            self.assertIs(stand_in.scoring_mask, masks[k])
            self.assertAlmostEqual(stand_in.measurement_sd, 0.1 * (k + 1))
            self.assertEqual(stand_in.uuid, f"ROOM{k}_T")
            self.assertTrue(stand_in.use_df)


class TestEstimatorReadsTheSensor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.main_dir = get_main_dir()
        cls.tmp = tempfile.mkdtemp()
        set_main_dir(cls.tmp)
        truth = build(n_pairs=3, model_id="scoring_truth")
        simulate(truth, hours=HOURS)
        cls.data = [history(truth, f"Zone{k}", "indoorTemperature").numpy() for k in range(3)]

    @classmethod
    def tearDownClass(cls):
        set_main_dir(cls.main_dir)
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _estimate(self, model, measurements):
        heater = model.get_components_by_class(tb.SpaceHeaterSystem)[0]
        estimator = tb.Estimator(
            tb.Simulator(model, execution_mode="functional", execution_backend="eager", compile_step=False)
        )
        estimator.estimate(
            parameters=[(heater, "UA", None, 1.0, 500.0)],
            measurements=measurements,
            start_time=[START],
            end_time=[END],
            step_size=STEP,
            n_warmup=2,
            method=("scipy", "SLSQP", "ad"),
            options={"maxiter": 1},
        )
        return estimator

    def test_the_sd_comes_from_the_sensor_when_the_entry_gives_none(self):
        model = room_model(3, "scoring_sd", self.data, measurement_sd=0.1)
        batched = model.batch_components()
        batched.load(draw_semantic_model=False, draw_simulation_model=False)
        sensors = [batched.components[f"T{k}"] for k in range(3)]
        mask = pd.Series(True, index=index(HOURS * 6))
        mask.iloc[10:20] = False
        sensors[0].scoring_mask = mask
        # the sensor alone, the sensor without a standard deviation, and one given in the entry
        estimator = self._estimate(batched, [sensors[0], (sensors[1], None), (sensors[2], 0.25)])
        self.assertEqual([s.id for s, _ in estimator._measurements], ["T0", "T1", "T2"])
        self.assertEqual([sd for _, sd in estimator._measurements], [0.1, 0.1, 0.25])
        # the masked samples of the first sensor are not scored, every other one is
        measured = estimator._functional_objective.ACT[0]
        self.assertTrue(torch.isnan(measured[10:20, 0]).all())
        self.assertFalse(torch.isnan(measured[:10, 0]).any())
        self.assertFalse(torch.isnan(measured[20:, 0]).any())
        self.assertFalse(torch.isnan(measured[:, 1:]).any())

    def test_the_pairs_of_before_are_taken_as_they_are(self):
        model = room_model(2, "scoring_pairs", self.data, measurement_sd=0.1)
        sensors = [model.components[f"T{k}"] for k in range(2)]
        estimator = self._estimate(model, [(sensors[0], 0.5), (sensors[1], 0.7)])
        self.assertEqual([sd for _, sd in estimator._measurements], [0.5, 0.7])

    def test_a_sensor_without_any_sd_is_named(self):
        model = room_model(2, "scoring_missing", self.data)
        sensors = [model.components[f"T{k}"] for k in range(2)]
        sensors[0].measurement_sd = 0.1
        with self.assertRaisesRegex(ValueError, "T1"):
            self._estimate(model, [sensors[0], sensors[1]])
        with self.assertRaisesRegex(ValueError, "T1"):
            self._estimate(model, [(sensors[0], None), (sensors[1], None)])


if __name__ == "__main__":
    unittest.main()
