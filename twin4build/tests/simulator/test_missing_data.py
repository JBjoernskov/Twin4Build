"""Missing data are never invented: a scored sensor's gap is an unscored sample, an input's gap an error unless the
input declares a fill, and what a fill reaches is not scored.

The loaders used to repeat the first and last readings over a whole window and bridge gaps of any length, and the
time series input repeated its last value past the data: a model was fitted to values nobody measured, silently.
Now (``TimeSeriesInputSystem``, ``SensorSystem``):

* a scored sensor (one measuring a model output) carries NaN where it has no reading; the sample is not scored;
* an input sensor (one whose readings drive the model) with a step its readings do not give is an error, naming the
  sensor and the step, unless it declares ``fill`` (``"linear"``, ``"constant"``, ``"copy_week"``); the steps a
  fill gave a value are logged and kept, and every scored sensor downstream leaves them unscored, in the estimator
  and in the simulator's measurement frames;
* ``allow_missing`` on an input and ``fill`` on a scored sensor are refused: one would drive the model with zeros,
  the other invent measurements to score.
"""

# Standard library imports
import datetime
import shutil
import tempfile
import unittest

# Third party imports
import numpy as np
import pandas as pd

# Local application imports
import twin4build as tb
from twin4build.systems.utils.time_series_input_system import TimeSeriesInputSystem
from twin4build.utils.get_main_dir import get_main_dir, set_main_dir
from twin4build.utils.scoring_mask import filled_downstream

tb._IS_TESTING = True

from twin4build.tests.simulator.test_fusion_batched import START, STEP, build  # noqa: E402
from twin4build.tests.simulator.test_measurement_errors import HOURS, N, OFFSETS, frame, simulate, truth  # noqa: E402

GAP = slice(10, 30)  # 200 minutes without a supply-water reading


def setUpModule():
    global _MAIN_DIR, _FILES
    _MAIN_DIR = get_main_dir()
    _FILES = tempfile.mkdtemp()
    set_main_dir(_FILES)


def tearDownModule():
    set_main_dir(_MAIN_DIR)
    shutil.rmtree(_FILES, ignore_errors=True)


def water(n, gap=None):
    """The supply-water readings (60 C), ``gap`` rows left out."""
    f = frame([60.0] * n, START)
    return f.drop(f.index[gap]) if gap is not None else f


def model_with_input(model_id, data, water_frame, **sensor_kwargs):
    """Three heated zones, a scored sensor on each temperature; radiator 2 reads its supply water from a data leaf."""
    model = build(n_pairs=3, model_id=model_id)
    for k in range(3):
        sensor = tb.SensorSystem(id=f"T{k}", df=frame(data[k] - OFFSETS[k], START), uuid=f"ROOM{k}_T")
        model.add_connection(model.components[f"Zone{k}"], sensor, "indoorTemperature", "measuredValue")
    leaf = tb.SensorSystem(id="water_sensor", df=water_frame, dt_limit=3600, **sensor_kwargs)
    radiator = model.components["Radiator2"]
    model.remove_connection(model.components["WaterTemp"], radiator, "scheduleValue", "supplyWaterTemperature")
    model.add_connection(leaf, radiator, "measuredValue", "supplyWaterTemperature")
    model.load(draw_semantic_model=False, draw_simulation_model=False)
    return model


class TestTimeSeriesInput(unittest.TestCase):
    def series(self, minutes, values):
        return pd.DataFrame({"value": np.asarray(values, dtype=float)},
                            index=pd.DatetimeIndex([START + datetime.timedelta(minutes=float(m)) for m in minutes]))

    def initialize(self, tsi, hours=2):
        tsi.initialize(start_time=[START], end_time=[START + datetime.timedelta(hours=hours)], step_size=[STEP])
        return tsi.values[:, 0, 0].numpy()

    def test_no_value_is_repeated_past_the_data(self):
        tsi = TimeSeriesInputSystem(id="t", df=self.series(range(0, 61, 10), np.arange(7)))
        tsi.allow_missing = True
        self.initialize(tsi)
        self.assertEqual(int(np.isnan(tsi.df[0].to_numpy(float)).sum()), 5)  # 70 .. 110 min: after the last reading
        np.testing.assert_allclose(tsi.df[0].to_numpy(float)[:7], np.arange(7))

    def test_an_input_with_a_gap_is_an_error(self):
        tsi = TimeSeriesInputSystem(id="t", df=self.series(range(0, 121, 10), np.arange(13)), dt_limit=None)
        tsi.initialize(start_time=[START], end_time=[START + datetime.timedelta(hours=2)], step_size=[STEP])  # no gap
        tsi = TimeSeriesInputSystem(id="t", df=self.series([0, 10, 100, 110, 120], [0, 1, 10, 11, 12]), dt_limit=1800)
        with self.assertRaisesRegex(AssertionError, "declare a fill"):
            self.initialize(tsi)

    def test_linear_fill_bridges_the_gap_and_says_so(self):
        tsi = TimeSeriesInputSystem(id="t", df=self.series([0, 10, 100, 110, 120], [0, 1, 10, 11, 12]),
                                    dt_limit=1800, fill="linear")
        v = self.initialize(tsi)
        np.testing.assert_allclose(v, np.arange(12))
        np.testing.assert_array_equal(np.flatnonzero(tsi.filled[0]), np.arange(2, 10))

    def test_linear_fill_never_reaches_beyond_the_readings(self):
        tsi = TimeSeriesInputSystem(id="t", df=self.series([30, 40, 50], [3, 4, 5]), fill="linear")
        with self.assertRaisesRegex(AssertionError, "left these"):
            self.initialize(tsi)

    def test_interpolate_fill_bridges_gaps_and_holds_beyond_the_readings(self):
        """Straight lines inside, the nearest reading held at the edges; a period without a reading as ``constant``
        (23 Oct 2026: room temperatures a cut replays into exhaust boxes, some dead for a week)."""
        tsi = TimeSeriesInputSystem(id="t", df=self.series([30, 40, 70, 80], [3, 4, 7, 8]), dt_limit=1200, fill="interpolate")
        v = self.initialize(tsi)
        np.testing.assert_allclose(v, [3, 3, 3, 3, 4, 5, 6, 7, 8, 8, 8, 8])
        np.testing.assert_array_equal(np.flatnonzero(tsi.filled[0]), [0, 1, 2, 5, 6, 9, 10, 11])
        tsi = TimeSeriesInputSystem(id="t", df=self.series([-600, -590, 300], [7, 8, 9]), dt_limit=1800, fill="interpolate")
        np.testing.assert_allclose(self.initialize(tsi), np.full(12, 8.0))

    def test_constant_fill_holds_both_ways(self):
        tsi = TimeSeriesInputSystem(id="t", df=self.series([30, 40, 50], [3, 4, 5]), fill="constant")
        v = self.initialize(tsi)
        np.testing.assert_allclose(v, [3, 3, 3, 3, 4, 5, 5, 5, 5, 5, 5, 5])
        self.assertEqual(int(tsi.filled.sum()), 9)

    def test_constant_fill_holds_the_nearest_reading_over_a_period_without_one(self):
        """No reading in the period: the last one before it; none before, the first after it."""
        tsi = TimeSeriesInputSystem(id="t", df=self.series([-600, -590, 300], [7, 8, 9]), dt_limit=1800, fill="constant")
        v = self.initialize(tsi)
        np.testing.assert_allclose(v, np.full(12, 8.0))
        self.assertTrue(tsi.filled[0].all())
        tsi = TimeSeriesInputSystem(id="t", df=self.series([300, 310], [9, 10]), dt_limit=1800, fill="constant")
        np.testing.assert_allclose(self.initialize(tsi), np.full(12, 9.0))

    def test_copy_week_takes_the_same_steps_a_week_earlier(self):
        week = 7 * 24 * 60
        minutes = list(range(-week, -week + 121, 10)) + [0, 10, 100, 110, 120]
        values = list(np.arange(13) + 100.0) + [0, 1, 10, 11, 12]
        tsi = TimeSeriesInputSystem(id="t", df=self.series(minutes, values), dt_limit=1800, fill="copy_week")
        v = self.initialize(tsi)
        np.testing.assert_allclose(v[[0, 1, 10, 11]], [0, 1, 10, 11])
        np.testing.assert_allclose(v[2:10], np.arange(2, 10) + 100.0)  # the week before
        np.testing.assert_array_equal(np.flatnonzero(tsi.filled[0]), np.arange(2, 10))


class TestSensorsAndScoring(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = truth()

    def test_an_input_without_a_fill_names_the_gap(self):
        model = model_with_input("missing_nofill", self.data, water(N + 6, GAP))
        with self.assertRaisesRegex(AssertionError, "declare a fill"):
            simulate(model)

    def test_allow_missing_on_an_input_is_refused(self):
        model = model_with_input("missing_allow", self.data, water(N + 6, GAP), allow_missing=True)
        with self.assertRaisesRegex(ValueError, "declare a fill"):
            simulate(model)

    def test_a_fill_on_a_scored_sensor_is_refused(self):
        model = model_with_input("missing_scoredfill", self.data, water(N + 6))
        model.components["T0"].fill = "linear"
        with self.assertRaisesRegex(ValueError, "invent measurements"):
            simulate(model)

    def test_a_scored_sensors_gap_is_unscored_by_default(self):
        data = [d.copy() for d in self.data]
        model = build(n_pairs=3, model_id="missing_scored")
        f = frame(data[0] - OFFSETS[0], START)
        sensor = tb.SensorSystem(id="T0", df=f.drop(f.index[GAP]), uuid="ROOM0_T", dt_limit=3600)
        model.add_connection(model.components["Zone0"], sensor, "indoorTemperature", "measuredValue")
        model.load(draw_semantic_model=False, draw_simulation_model=False)
        errors = simulate(model).measurement_errors()
        self.assertEqual(int(errors["n"][0]), N - (GAP.stop - GAP.start))  # the steps of the readings left out
        np.testing.assert_allclose(errors["bias"][0], OFFSETS[0], atol=1e-6)

    def test_what_a_fill_reaches_is_not_scored(self):
        model = model_with_input("missing_fill", self.data, water(N + 6, GAP), fill="linear")
        simulator = simulate(model)
        filled = model.components["water_sensor"].time_series_input.filled[0]
        self.assertEqual(np.flatnonzero(filled).tolist(), list(range(GAP.start, GAP.stop)))
        frames = simulator.measurement_frames()
        np.testing.assert_array_equal(frames["T2"]["scored"].to_numpy(), ~filled[:N])  # downstream of radiator 2
        self.assertTrue(frames["T0"]["scored"].all() and frames["T1"]["scored"].all())  # not reached
        self.assertEqual(sorted(filled_downstream(model)), ["T2"])
        errors = simulator.measurement_errors()
        self.assertEqual(list(errors["n"]), [N, N, N - int(filled.sum())])

    def test_a_fill_reach_limits_what_is_unscored(self):
        """A filled input with a ``fill_reach`` unscores only those of the
        sensors it reaches; an empty reach unscores nothing."""
        model = model_with_input("missing_reach", self.data, water(N + 6, GAP), fill="linear")
        model.components["water_sensor"].fill_reach = set()
        simulator = simulate(model)
        self.assertEqual(filled_downstream(model), {})
        self.assertTrue(simulator.measurement_frames()["T2"]["scored"].all())
        model.components["water_sensor"].fill_reach = {"T2", "T0"}  # T0 is not reached: still scored
        self.assertEqual(sorted(filled_downstream(model)), ["T2"])

    def test_a_fill_recovery_unscores_the_steps_after_the_gap(self):
        model = model_with_input("missing_recovery", self.data, water(N + 6, GAP), fill="linear")
        model.components["water_sensor"].fill_recovery_steps = 4
        simulate(model)
        unscored = filled_downstream(model)["T2"][0]
        self.assertEqual(np.flatnonzero(unscored).tolist(), list(range(GAP.start, GAP.stop + 4)))

    def test_a_batched_model_finds_the_same(self):
        source = model_with_input("missing_batched", self.data, water(N + 6, GAP), fill="linear")
        batched = source.batch_components()
        batched.load(draw_semantic_model=False, draw_simulation_model=False)
        simulator = simulate(batched)
        reached = sorted(filled_downstream(batched))
        self.assertEqual(len(reached), 1)
        sensor = batched.components[reached[0]]
        self.assertEqual(getattr(sensor, "_source_component_ids", (sensor.id,))[0], "T2")
        frames = simulator.measurement_frames()
        self.assertFalse(frames[reached[0]]["scored"].to_numpy()[GAP.start:GAP.stop].any())


if __name__ == "__main__":
    unittest.main()
