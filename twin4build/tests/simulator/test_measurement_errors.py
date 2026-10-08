"""The simulation against the measurements: ``Simulator.measurement_frames``
and ``Simulator.measurement_errors``.

Every sensor that reads a computed port and holds data gives a frame
(measured, simulated, scored) and a row of the error table, computed over
the samples its ``scoring_mask`` scores after the warm-up.  The same on the
unbatched model and on a batched one, in object mode and in the functional
engine.
"""
import datetime
import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd

import twin4build as tb
from twin4build.utils.get_main_dir import get_main_dir, set_main_dir
from twin4build.utils.scoring_mask import period_mask

tb._IS_TESTING = True

from twin4build.tests.simulator.test_fusion_batched import START, STEP, build, history

HOURS = 6
END = START + datetime.timedelta(hours=HOURS)
N = HOURS * 3600 // STEP
OFFSETS = (0.5, -0.25, 1.0)
PORT = "BuildingSpaceThermalSystem.indoorTemperature"


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


def plausible(x):
    """A reading above 100 degrees is no reading."""
    return x if x < 100.0 else float("nan")


def simulate(model, start=START, end=END, **kwargs):
    simulator = tb.Simulator(model, **kwargs)
    simulator.simulate(start_time=start, end_time=end, step_size=STEP, show_progress_bar=False)
    return simulator


def truth(hours=HOURS, start=START):
    """The zones' temperatures of the model itself."""
    model = build(n_pairs=3, model_id="errors_truth")
    simulate(model, start=start, end=start + datetime.timedelta(hours=hours))
    return [history(model, f"Zone{k}", "indoorTemperature").numpy() for k in range(3)]


def room_model(model_id, data, start=START):
    """Three heated zones; a sensor on each temperature holding the
    simulation's own values moved by an offset, the first two with a uuid."""
    model = build(n_pairs=3, model_id=model_id)
    for k in range(3):
        sensor = tb.SensorSystem(id=f"T{k}", df=frame(data[k] - OFFSETS[k], start), uuid=f"ROOM{k}_T" if k < 2 else None)
        model.add_connection(model.components[f"Zone{k}"], sensor, "indoorTemperature", "measuredValue")
    # sensors that read a port and hold no data, and a data leaf: no measurements
    for k in range(3):
        model.add_connection(model.components[f"Radiator{k}"], tb.SensorSystem(id=f"P{k}"), "Power", "measuredValue")
    leaf = tb.SensorSystem(id="water_sensor", df=frame([60.0] * (len(data[0]) + 6), start))
    radiator = model.components["Radiator2"]
    model.remove_connection(model.components["WaterTemp"], radiator, "scheduleValue", "supplyWaterTemperature")
    model.add_connection(leaf, radiator, "measuredValue", "supplyWaterTemperature")
    model.load(draw_semantic_model=False, draw_simulation_model=False)
    return model


class TestMeasurementErrors(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = truth()

    def test_frames_hold_the_measured_and_the_simulated_series(self):
        model = room_model("errors_frames", self.data)
        simulator = simulate(model)
        frames = simulator.measurement_frames()
        self.assertEqual(sorted(frames), ["T0", "T1", "T2"])
        for k in range(3):
            f = frames[f"T{k}"]
            self.assertEqual(list(f.columns), ["measured", "simulated", "scored"])
            self.assertEqual(len(f), N)
            self.assertEqual(f.index[0].to_pydatetime(), START)
            self.assertEqual(f.index[-1].to_pydatetime(), END - datetime.timedelta(seconds=STEP))
            np.testing.assert_allclose(f["measured"].to_numpy(), self.data[k][:N] - OFFSETS[k], rtol=1e-12)
            np.testing.assert_allclose(
                f["simulated"].to_numpy(), history(model, f"Zone{k}", "indoorTemperature").numpy(), rtol=1e-12
            )
            self.assertTrue(f["scored"].all())
            self.assertEqual(f["scored"].dtype, bool)

    def test_errors_per_sensor(self):
        model = room_model("errors_table", self.data)
        errors = simulate(model).measurement_errors()
        self.assertEqual(list(errors.columns), ["sensor", "port", "n", "mae", "rmse", "bias"])
        # the uuid names the sensor, its id where it has none
        self.assertEqual(list(errors["sensor"]), ["ROOM0_T", "ROOM1_T", "T2"])
        self.assertEqual(list(errors["port"]), [PORT] * 3)
        self.assertEqual(list(errors["n"]), [N] * 3)
        # the model simulates its own data: what is left is the offset
        np.testing.assert_allclose(errors["bias"], OFFSETS, atol=1e-6)
        np.testing.assert_allclose(errors["mae"], np.abs(OFFSETS), atol=1e-6)
        np.testing.assert_allclose(errors["rmse"], np.abs(OFFSETS), atol=1e-6)

    def test_the_warm_up_is_skipped(self):
        data = [d.copy() for d in self.data]
        data[0][:4] += 10.0  # the first steps of the first sensor are off by ten
        model = room_model("errors_skip", data)
        simulator = simulate(model)
        errors = simulator.measurement_errors()
        self.assertGreater(float(errors["rmse"][0]), 1.0)
        errors = simulator.measurement_errors(skip=4)
        self.assertEqual(list(errors["n"]), [N - 4] * 3)
        np.testing.assert_allclose(errors["rmse"], np.abs(OFFSETS), atol=1e-6)

    def test_unscored_samples_are_left_out(self):
        data = [d.copy() for d in self.data]
        data[1][10:20] += 10.0
        model = room_model("errors_mask", data)
        mask = pd.Series(True, index=index(3 * N, START - datetime.timedelta(hours=HOURS)))  # a longer span
        mask.iloc[N + 10 : N + 20] = False
        model.components["T1"].scoring_mask = mask
        simulator = simulate(model)
        scored = simulator.measurement_frames()["T1"]["scored"].to_numpy()
        self.assertFalse(scored[10:20].any())
        self.assertTrue(scored[:10].all() and scored[20:].all())
        errors = simulator.measurement_errors(skip=2)
        self.assertEqual(list(errors["n"]), [N - 2, N - 12, N - 2])
        np.testing.assert_allclose(errors["bias"], OFFSETS, atol=1e-6)
        # a mask over the steps of the period, as an array
        model.components["T1"].scoring_mask = np.arange(N) >= 20
        errors = simulate(model).measurement_errors()
        self.assertEqual(list(errors["n"]), [N, N - 20, N])
        np.testing.assert_allclose(errors["bias"], OFFSETS, atol=1e-6)

    def test_samples_without_a_measurement_are_left_out(self):
        data = [d.copy() for d in self.data]
        data[2][5:8] = 6551.0 + OFFSETS[2]  # a glitch of the sensor
        model = room_model("errors_gaps", data)
        model.components["T2"].transformation = plausible
        model.components["T2"].allow_missing = True  # gaps are unscored samples, not an error
        simulator = simulate(model)
        measured = simulator.measurement_frames()["T2"]["measured"].to_numpy()
        self.assertTrue(np.isnan(measured[5:8]).all())
        self.assertFalse(np.isnan(np.delete(measured, [5, 6, 7])).any())
        errors = simulator.measurement_errors()
        self.assertEqual(list(errors["n"]), [N, N, N - 3])
        np.testing.assert_allclose(errors["bias"], OFFSETS, atol=1e-6)

    def test_a_sensor_without_scored_samples(self):
        model = room_model("errors_none", self.data)
        model.components["T0"].scoring_mask = pd.Series(False, index=index(N))
        errors = simulate(model).measurement_errors()
        self.assertEqual(int(errors["n"][0]), 0)
        self.assertTrue(np.isnan(errors.loc[0, ["mae", "rmse", "bias"]].to_numpy(dtype=float)).all())
        np.testing.assert_allclose(errors["bias"][1:], OFFSETS[1:], atol=1e-6)

    def test_a_batched_model_gives_the_same(self):
        model = room_model("errors_unbatched", self.data)
        model.components["T1"].scoring_mask = np.arange(N) >= 20
        expected = simulate(model).measurement_errors(skip=2)
        expected_frames = simulate(model).measurement_frames()

        source = room_model("errors_batched", self.data)
        source.components["T1"].scoring_mask = np.arange(N) >= 20
        batched = source.batch_components()
        batched.load(draw_semantic_model=False, draw_simulation_model=False)
        self.assertEqual(getattr(source.get_batched_component_info("Zone0")[0], "n_c", 1), 2)
        for kwargs in ({}, {"execution_mode": "functional", "execution_backend": "eager", "compile_step": False}):
            simulator = simulate(batched, **kwargs)
            errors = simulator.measurement_errors(skip=2)
            self.assertEqual(list(errors["sensor"]), list(expected["sensor"]))
            self.assertEqual(list(errors["port"]), [PORT] * 3)
            self.assertEqual(list(errors["n"]), list(expected["n"]))
            for column in ("mae", "rmse", "bias"):
                np.testing.assert_allclose(errors[column], expected[column], atol=1e-9, err_msg=str(kwargs))
            frames = simulator.measurement_frames()
            for key, f in expected_frames.items():
                pd.testing.assert_frame_equal(frames[key], f, check_exact=False, atol=1e-9)

    def test_several_periods_follow_one_another(self):
        second = START + datetime.timedelta(days=1)
        model = build(n_pairs=3, model_id="errors_periods")
        for k in range(3):
            values = pd.concat(
                [frame(self.data[k] - OFFSETS[k], START), frame(self.data[k] - 2 * OFFSETS[k], second)]
            )
            sensor = tb.SensorSystem(id=f"T{k}", df=values)
            model.add_connection(model.components[f"Zone{k}"], sensor, "indoorTemperature", "measuredValue")
        model.load(draw_semantic_model=False, draw_simulation_model=False)
        simulator = tb.Simulator(model)
        simulator.simulate(
            start_time=[START, second],
            end_time=[END, second + datetime.timedelta(hours=HOURS)],
            step_size=STEP,
            show_progress_bar=False,
        )
        f = simulator.measurement_frames()["T0"]
        self.assertEqual(len(f), 2 * N)
        self.assertEqual(f.index[N].to_pydatetime(), second)
        np.testing.assert_allclose(f["simulated"].to_numpy()[:N], self.data[0][:N], rtol=1e-9)
        np.testing.assert_allclose(f["simulated"].to_numpy()[N:], self.data[0][:N], rtol=1e-9)
        errors = simulator.measurement_errors(skip=3)  # of each period
        self.assertEqual(list(errors["n"]), [2 * (N - 3)] * 3)
        np.testing.assert_allclose(errors["bias"], 1.5 * np.asarray(OFFSETS), atol=1e-6)

    def test_before_a_simulation_there_is_nothing_to_read(self):
        model = room_model("errors_before", self.data)
        simulator = tb.Simulator(model)
        with self.assertRaises(RuntimeError):
            simulator.measurement_frames()
        with self.assertRaises(RuntimeError):
            simulator.measurement_errors()


class TestPeriodMask(unittest.TestCase):
    def test_a_series_is_selected_by_time(self):
        mask = pd.Series([True, False, True, False], index=index(4, START + datetime.timedelta(seconds=STEP)))
        # the period starts one step before the mask and ends after it: what it does not cover is scored
        keep = period_mask(mask, index(6), 6)
        self.assertEqual(keep.tolist(), [True, True, False, True, False, True])

    def test_an_index_shorter_than_the_period(self):
        mask = pd.Series([False, False, True], index=index(3))
        keep = period_mask(mask, index(3), 5)
        self.assertEqual(keep.tolist(), [False, False, True, True, True])

    def test_an_array_is_the_steps_of_the_period(self):
        self.assertEqual(period_mask(np.array([False, True]), index(4), 4).tolist(), [False, True, True, True])
        self.assertEqual(period_mask(np.array([False, True, True, False, False]), index(3), 3).tolist(), [False, True, True])


if __name__ == "__main__":
    unittest.main()
