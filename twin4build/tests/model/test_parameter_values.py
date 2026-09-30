"""Parameters by component id, across batchings.

A batched model keeps the parameters of its components stacked on metas
whose ids depend on the batching.  ``Model.get_parameter_values`` reads them
by the ids of the components the model was built from and
``Model.set_parameter_values`` writes them to any batching of the model, or
to the unbatched model.  A saved fit carries its parameters that way
(``parameter_instances``, ``parameter_instances_x0``,
``parameter_instance_bounds``), so ``load_estimation_result`` loads it onto
another batching than the one it was fitted on.
"""
import datetime
import pickle
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd
import torch

import twin4build as tb
import twin4build.core as core
import twin4build.utils.types as tps
from twin4build.estimator.estimator import EstimationResult
from twin4build.utils.get_main_dir import get_main_dir, set_main_dir
from twin4build.utils.logger import LOGGER

tb._IS_TESTING = True

from twin4build.tests.simulator.test_fusion_batched import START, STEP, build, history, simulate

HOURS = 12
END = START + datetime.timedelta(hours=HOURS)
N_STEPS = HOURS * 3600 // STEP


def setUpModule():
    """The models of this module keep their files in a temporary folder."""
    global _MAIN_DIR, _FILES
    _MAIN_DIR = get_main_dir()
    _FILES = tempfile.mkdtemp()
    set_main_dir(_FILES)


def tearDownModule():
    set_main_dir(_MAIN_DIR)
    shutil.rmtree(_FILES, ignore_errors=True)


def frame(values):
    values = np.asarray(values, dtype=float)
    index = pd.DatetimeIndex(
        [START + datetime.timedelta(seconds=STEP * k) for k in range(len(values))], name="time"
    )
    return pd.DataFrame({"value": values}, index=index)


def batch(model):
    batched = model.batch_components()
    batched.load(draw_semantic_model=False, draw_simulation_model=False)
    return batched


def split(model):
    """Feed the last radiator's water temperature from a sensor holding the
    schedule's value: the same model, but its radiators and zones batch in
    two parts (the first ones, and the last one on its own)."""
    last = max(int(cid[len("Radiator"):]) for cid in model.components if cid.startswith("Radiator"))
    radiator = model.components[f"Radiator{last}"]
    sensor = tb.SensorSystem(id="water_sensor", df=frame([60.0] * (N_STEPS + 6)))
    model.remove_connection(model.components["WaterTemp"], radiator, "scheduleValue", "supplyWaterTemperature")
    model.add_connection(sensor, radiator, "measuredValue", "supplyWaterTemperature")
    model.load(draw_semantic_model=False, draw_simulation_model=False)
    return model


def heater_parameters(model):
    heaters = model.get_components_by_class(tb.SpaceHeaterSystem)
    return [(h, attr) for h in heaters for attr in ("UA", "thermalMassHeatCapacity")]


def temperatures(model, batched=None, n=3):
    return [history(model, f"Zone{k}", "indoorTemperature", batched) for k in range(n)]


VALUES = {
    ("Radiator0", "UA"): np.asarray(55.0),
    ("Radiator1", "UA"): np.asarray(75.0),
    ("Radiator2", "UA"): np.asarray(95.0),
    ("Radiator0", "thermalMassHeatCapacity"): np.asarray(4e4),
    ("Radiator1", "thermalMassHeatCapacity"): np.asarray(6e4),
    ("Radiator2", "thermalMassHeatCapacity"): np.asarray(8e4),
}


class Rows(core.System):
    """A component with a vector parameter ``w`` and a scalar parameter ``g``."""

    def __init__(self, w, g, **kwargs):
        super().__init__(**kwargs)
        self.input = {}
        self.output = {"y": tps.Scalar()}
        self.w = tps.TensorParameter(
            torch.tensor(w, dtype=tps.float_dtype()), min_value=0.0, max_value=100.0, normalized=False
        )
        self.g = tps.Parameter(torch.tensor(float(g), dtype=tps.float_dtype()), min_value=0.0, max_value=10.0)
        self._config = {"parameters": []}

    @property
    def config(self):
        return self._config


class TestSourceComponentIds(unittest.TestCase):
    def test_a_meta_stands_for_the_components_it_was_built_from(self):
        model = build(n_pairs=3, model_id="pv_ids")
        batched = batch(model)
        meta, _ = model.get_batched_component_info("Radiator1")
        self.assertEqual(model.get_source_component_ids(meta), ("Radiator0", "Radiator1", "Radiator2"))
        self.assertEqual(tb.Model.get_source_component_ids(meta), ("Radiator0", "Radiator1", "Radiator2"))
        # a component that was not lumped with others stands for itself, batched or not
        self.assertEqual(batched.get_source_component_ids(batched.components["Outdoor"]), ("Outdoor",))
        self.assertEqual(model.get_source_component_ids(model.components["Radiator1"]), ("Radiator1",))


class TestGetParameterValues(unittest.TestCase):
    def test_a_batched_meta_gives_one_key_per_component(self):
        model = build(n_pairs=3, model_id="pv_get")
        batched = batch(model)
        values = batched.get_parameter_values(heater_parameters(batched))
        self.assertEqual(
            set(values),
            {(f"Radiator{k}", attr) for k in range(3) for attr in ("UA", "thermalMassHeatCapacity")},
        )
        for k in range(3):
            self.assertIsInstance(values[(f"Radiator{k}", "UA")], np.ndarray)
            self.assertEqual(values[(f"Radiator{k}", "UA")].shape, ())
            self.assertAlmostEqual(float(values[(f"Radiator{k}", "UA")]), 40.0 + 5 * k)
            self.assertAlmostEqual(
                float(values[(f"Radiator{k}", "thermalMassHeatCapacity")]) / (5e4 * (1 + 0.1 * k)), 1.0
            )
        # the same values as the unbatched model's, under the same keys
        unbatched = model.get_parameter_values(heater_parameters(model))
        self.assertEqual(set(unbatched), set(values))
        for key in values:
            np.testing.assert_allclose(unbatched[key], values[key], rtol=1e-12)

    def test_entries_are_the_estimators(self):
        """The entries may carry what the estimator's do (start and
        bounds) and a list of components."""
        model = build(n_pairs=2, model_id="pv_entries")
        zones = [model.components[f"Zone{k}"] for k in range(2)]
        heater = model.components["Radiator0"]
        values = model.get_parameter_values(
            [(zones, "C_air", 2e6, 1e5, 1e7, "shared"), (heater, "UA", None, 1.0, 500.0)]
        )
        self.assertEqual(set(values), {("Zone0", "C_air"), ("Zone1", "C_air"), ("Radiator0", "UA")})
        self.assertAlmostEqual(float(values[("Zone1", "C_air")]), 1e6)

    def test_what_is_not_a_parameter_is_refused(self):
        model = build(n_pairs=1, model_id="pv_refused")
        heater = model.components["Radiator0"]
        with self.assertRaises(TypeError):
            model.get_parameter_values([(heater, "nelements")])
        with self.assertRaises(TypeError):
            model.get_parameter_values([(heater, "no_such_parameter")])

    def test_rows_and_shared_values_of_a_meta(self):
        """A meta whose parameter holds several values per instance gives
        each component its row; one value is the value of every instance."""
        model = tb.Model(id="pv_rows")
        meta = Rows([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 0.5, id="meta")
        meta._source_component_ids = ("a", "b")
        single = Rows([7.0, 8.0], 0.25, id="single")
        model.add_component(meta)
        model.add_component(single)
        values = model.get_parameter_values([(meta, "w"), (meta, "g"), (single, "w"), (single, "g")])
        np.testing.assert_allclose(values[("a", "w")], [1.0, 2.0, 3.0])
        np.testing.assert_allclose(values[("b", "w")], [4.0, 5.0, 6.0])
        self.assertAlmostEqual(float(values[("a", "g")]), 0.5)
        self.assertAlmostEqual(float(values[("b", "g")]), 0.5)
        np.testing.assert_allclose(values[("single", "w")], [7.0, 8.0])
        self.assertEqual(values[("single", "g")].shape, ())

        counts = model.set_parameter_values(
            {
                ("b", "w"): [40.0, 50.0, 60.0],  # one instance's row
                ("a", "g"): 0.75,  # the value the instances share
                ("single", "w"): [70.0, 80.0],
                ("single", "g"): np.asarray(0.125),
                ("a", "w"): [1.0, 2.0],  # two values for a row of three
            }
        )
        self.assertEqual(counts, {"applied": 4, "missing": 1})
        np.testing.assert_allclose(meta.w.get().numpy(), [1.0, 2.0, 3.0, 40.0, 50.0, 60.0])
        self.assertAlmostEqual(float(meta.g.get()), 0.75)
        np.testing.assert_allclose(single.w.get().numpy(), [70.0, 80.0])
        self.assertAlmostEqual(float(single.g.get()), 0.125)
        # one value for a vector parameter of a component fills it
        self.assertEqual(model.set_parameter_values({("single", "w"): 9.0}), {"applied": 1, "missing": 0})
        np.testing.assert_allclose(single.w.get().numpy(), [9.0, 9.0])


class TestSetParameterValues(unittest.TestCase):
    def test_an_instance_is_written_and_the_others_are_left_alone(self):
        model = build(n_pairs=3, model_id="pv_set")
        batched = batch(model)
        meta, i_c = model.get_batched_component_info("Radiator1")
        self.assertEqual(i_c, 1)
        counts = batched.set_parameter_values({("Radiator1", "UA"): 123.0})
        self.assertEqual(counts, {"applied": 1, "missing": 0})
        np.testing.assert_allclose(meta.UA.get().detach().numpy(), [40.0, 123.0, 50.0])
        np.testing.assert_allclose(
            meta.thermalMassHeatCapacity.get().detach().numpy(), [5e4, 5.5e4, 6e4], rtol=1e-12
        )

    def test_a_meta_is_named_by_its_own_id_too(self):
        model = build(n_pairs=3, model_id="pv_meta_id")
        batched = batch(model)
        meta, _ = model.get_batched_component_info("Radiator0")
        counts = batched.set_parameter_values({(meta.id, "UA"): [10.0, 20.0, 30.0]})
        self.assertEqual(counts, {"applied": 1, "missing": 0})
        np.testing.assert_allclose(meta.UA.get().detach().numpy(), [10.0, 20.0, 30.0])

    def test_what_the_model_does_not_have_is_counted(self):
        model = build(n_pairs=2, model_id="pv_missing")
        batched = batch(model)
        counts = batched.set_parameter_values(
            {
                ("Radiator0", "UA"): 60.0,
                ("Radiator7", "UA"): 60.0,  # no such component
                ("Radiator1", "no_such_parameter"): 1.0,  # no such parameter
                ("Radiator1", "nelements"): 5,  # not a parameter
            }
        )
        self.assertEqual(counts, {"applied": 1, "missing": 3})
        meta, _ = model.get_batched_component_info("Radiator0")
        np.testing.assert_allclose(meta.UA.get().detach().numpy(), [60.0, 45.0])
        self.assertEqual(meta.nelements, 3)

    def test_strict_raises_and_writes_nothing(self):
        model = build(n_pairs=2, model_id="pv_strict")
        batched = batch(model)
        meta, _ = model.get_batched_component_info("Radiator0")
        with self.assertRaises(KeyError):
            batched.set_parameter_values({("Radiator0", "UA"): 60.0, ("Radiator7", "UA"): 60.0}, strict=True)
        np.testing.assert_allclose(meta.UA.get().detach().numpy(), [40.0, 45.0])
        self.assertEqual(
            batched.set_parameter_values({("Radiator0", "UA"): 60.0}, strict=True), {"applied": 1, "missing": 0}
        )

    def test_a_sub_object_with_an_id_of_its_own_and_a_dotted_attribute(self):
        """The ids ``load_estimation_result`` resolves inside a component
        (an occupancy's dampers) resolve here as well, and so does the
        parameter of a sub-model (``thermal.C_air``)."""
        from twin4build.tests.estimator.example_fixture import load_model

        main_dir = get_main_dir()
        try:
            model = load_model()
        finally:
            set_main_dir(main_dir)
        damper = model.components["office_occupancy"].supply_damper
        self.assertNotIn(damper.id, model.components)
        counts = model.set_parameter_values({(damper.id, "nominalAirFlowRate"): 0.25})
        self.assertEqual(counts, {"applied": 1, "missing": 0})
        self.assertAlmostEqual(float(damper.nominalAirFlowRate.get().reshape(-1)[0]), 0.25)
        office = model.components["office"]
        self.assertEqual(model.set_parameter_values({("office", "thermal.C_air"): 3e6}), {"applied": 1, "missing": 0})
        self.assertAlmostEqual(float(office.thermal.C_air.get().reshape(-1)[0]), 3e6)
        values = model.get_parameter_values([(office, "thermal.C_air"), (damper, "nominalAirFlowRate")])
        self.assertAlmostEqual(float(values[("office", "thermal.C_air")]), 3e6)
        self.assertAlmostEqual(float(values[(damper.id, "nominalAirFlowRate")]), 0.25)

    def test_values_go_round_the_batchings(self):
        """Set on a batched model, read, and applied to the unbatched model
        and to another batching: the three simulate alike."""
        source = build(n_pairs=3, model_id="pv_round_source")
        batched = batch(source)
        self.assertEqual(batched.set_parameter_values(VALUES), {"applied": 6, "missing": 0})
        values = batched.get_parameter_values(heater_parameters(batched))
        self.assertEqual(set(values), set(VALUES))
        for key in VALUES:
            np.testing.assert_allclose(values[key], VALUES[key], rtol=1e-12)
        simulate(batched, hours=HOURS)
        reference = temperatures(source, batched)

        # the parameters change the simulation
        untouched = build(n_pairs=3, model_id="pv_round_untouched")
        simulate(untouched, hours=HOURS)
        self.assertGreater(float((temperatures(untouched)[2] - reference[2]).abs().max()), 1e-2)

        unbatched = build(n_pairs=3, model_id="pv_round_unbatched")
        self.assertEqual(unbatched.set_parameter_values(values), {"applied": 6, "missing": 0})
        simulate(unbatched, hours=HOURS)
        for k, t in enumerate(temperatures(unbatched)):
            torch.testing.assert_close(t, reference[k], rtol=1e-9, atol=1e-9, msg=f"Zone{k}")

        # the same model batched in two parts: a meta of two and a radiator on its own
        other = split(build(n_pairs=3, model_id="pv_round_split"))
        other_batched = batch(other)
        meta, _ = other.get_batched_component_info("Radiator0")
        self.assertEqual(other_batched.get_source_component_ids(meta), ("Radiator0", "Radiator1"))
        self.assertEqual(other_batched.set_parameter_values(values), {"applied": 6, "missing": 0})
        simulate(other_batched, hours=HOURS)
        for k, t in enumerate(temperatures(other, other_batched)):
            torch.testing.assert_close(t, reference[k], rtol=1e-9, atol=1e-9, msg=f"Zone{k}")

        # a smaller model takes what it has
        small = build(n_pairs=2, model_id="pv_round_small")
        small_batched = batch(small)
        self.assertEqual(small_batched.set_parameter_values(values), {"applied": 4, "missing": 2})
        simulate(small_batched, hours=HOURS)
        for k, t in enumerate(temperatures(small, small_batched, n=2)):
            torch.testing.assert_close(t, reference[k], rtol=1e-9, atol=1e-9, msg=f"Zone{k}")


class TestFitAcrossBatchings(unittest.TestCase):
    """A fit on a batched model, saved, and loaded onto other batchings."""

    @classmethod
    def setUpClass(cls):
        cls.main_dir = get_main_dir()
        cls.tmp = tempfile.mkdtemp()
        set_main_dir(cls.tmp)
        truth = build(n_pairs=3, model_id="pv_fit_truth")
        truth.set_parameter_values(VALUES)
        simulate(truth, hours=HOURS)
        cls.data = [t.numpy() for t in temperatures(truth)]

        cls.source = cls.with_sensors(build(n_pairs=3, model_id="pv_fit_source"))
        cls.batched = batch(cls.source)
        meta, _ = cls.source.get_batched_component_info("Radiator0")
        cls.meta = meta
        cls.parameters = [
            (meta, "UA", [60.0, 60.0, 60.0], 1.0, [400.0, 500.0, 600.0]),
            (meta, "thermalMassHeatCapacity", None, 1e4, 1e7),
        ]
        estimator = tb.Estimator(
            tb.Simulator(cls.batched, execution_mode="functional", execution_backend="eager", compile_step=False)
        )
        cls.result = estimator.estimate(
            parameters=cls.parameters,
            measurements=[cls.batched.components[f"T{k}"] for k in range(3)],
            start_time=[START],
            end_time=[END],
            step_size=STEP,
            n_warmup=2,
            method=("scipy", "SLSQP", "ad"),
            options={"maxiter": 3},
        )
        cls.filename = estimator.result_savedir_pickle
        cls.batched.set_save_simulation_result(True)  # an estimation switches the histories off
        simulate(cls.batched, hours=HOURS)
        cls.reference = temperatures(cls.source, cls.batched)

    @classmethod
    def tearDownClass(cls):
        set_main_dir(cls.main_dir)
        shutil.rmtree(cls.tmp, ignore_errors=True)

    @classmethod
    def with_sensors(cls, model):
        for k in range(3):
            sensor = tb.SensorSystem(id=f"T{k}", df=frame(cls.data[k]), uuid=f"ROOM{k}_T", measurement_sd=0.1)
            model.add_connection(model.components[f"Zone{k}"], sensor, "indoorTemperature", "measuredValue")
        model.load(draw_semantic_model=False, draw_simulation_model=False)
        return model

    def test_the_result_carries_the_parameters_by_component(self):
        with open(self.filename, "rb") as handle:
            saved = pickle.load(handle)
        keys = {(f"Radiator{k}", attr) for k in range(3) for attr in ("UA", "thermalMassHeatCapacity")}
        for result in (self.result, saved):
            fitted = result["parameter_instances"]
            start = result["parameter_instances_x0"]
            bounds = result["parameter_instance_bounds"]
            self.assertEqual(set(fitted), keys)
            self.assertEqual(set(start), keys)
            self.assertEqual(set(bounds), keys)
            self.assertEqual(result["component_source_ids"], [["Radiator0", "Radiator1", "Radiator2"]] * 2)
            # the fitted values are the model's and the result's
            now = self.batched.get_parameter_values(self.parameters)
            for k in range(3):
                np.testing.assert_allclose(fitted[(f"Radiator{k}", "UA")], now[(f"Radiator{k}", "UA")], rtol=1e-9)
                np.testing.assert_allclose(fitted[(f"Radiator{k}", "UA")], result["result_x"][k], rtol=1e-9)
                np.testing.assert_allclose(
                    fitted[(f"Radiator{k}", "thermalMassHeatCapacity")], result["result_x"][3 + k], rtol=1e-9
                )
                # the fit moved away from where it started
                self.assertNotAlmostEqual(float(fitted[(f"Radiator{k}", "UA")]), 60.0, places=3)
                # the start: the entry's, or the model's value where the entry gave none
                self.assertEqual(float(start[(f"Radiator{k}", "UA")]), 60.0)
                np.testing.assert_allclose(
                    start[(f"Radiator{k}", "thermalMassHeatCapacity")], 5e4 * (1 + 0.1 * k), rtol=1e-12
                )
                # the bounds: one per instance, or one for all
                low, high = bounds[(f"Radiator{k}", "UA")]
                self.assertEqual((float(low), float(high)), (1.0, 400.0 + 100.0 * k))
                low, high = bounds[(f"Radiator{k}", "thermalMassHeatCapacity")]
                self.assertEqual((float(low), float(high)), (1e4, 1e7))

    def _loaded(self, model, run=None):
        """Simulate ``run`` (``model`` or its batching) with the result
        loaded; the temperatures and what the loading logged."""
        run = model if run is None else run
        with mock.patch.object(LOGGER, "info") as info:
            run.load_estimation_result(filename=self.filename)
        logged = [call.args[0] % call.args[1:] for call in info.call_args_list]
        simulate(run, hours=HOURS)
        return temperatures(model, None if run is model else run), logged

    def test_the_result_loads_onto_the_unbatched_model(self):
        model = self.with_sensors(build(n_pairs=3, model_id="pv_fit_unbatched"))
        self.assertNotIn(self.meta.id, model.components)
        loaded, logged = self._loaded(model)
        self.assertTrue(any("6 parameter values applied by component id, 0 missing" in m for m in logged), logged)
        for k in range(3):
            torch.testing.assert_close(loaded[k], self.reference[k], rtol=1e-9, atol=1e-9, msg=f"Zone{k}")
            np.testing.assert_allclose(
                model.components[f"Radiator{k}"].UA.get().detach().numpy().reshape(-1)[0],
                self.result["parameter_instances"][(f"Radiator{k}", "UA")],
                rtol=1e-12,
            )

    def test_the_result_loads_onto_another_batching(self):
        """The other batching has a meta under the fitted meta's id that
        stands for two of the three radiators."""
        model = split(self.with_sensors(build(n_pairs=3, model_id="pv_fit_split")))
        batched = batch(model)
        meta, _ = model.get_batched_component_info("Radiator0")
        self.assertEqual(batched.get_source_component_ids(meta), ("Radiator0", "Radiator1"))
        self.assertEqual(meta.id, self.meta.id)
        loaded, logged = self._loaded(model, batched)
        self.assertTrue(any("6 parameter values applied by component id, 0 missing" in m for m in logged), logged)
        for k in range(3):
            torch.testing.assert_close(loaded[k], self.reference[k], rtol=1e-9, atol=1e-9, msg=f"Zone{k}")

    def test_the_same_batching_loads_as_before(self):
        """Every id of the result names the component it named in the fit:
        the values are set by those ids, with the bounds of the fit."""
        model = self.with_sensors(build(n_pairs=3, model_id="pv_fit_same"))
        batched = batch(model)
        meta, _ = model.get_batched_component_info("Radiator0")
        self.assertEqual(meta.id, self.meta.id)
        loaded, logged = self._loaded(model, batched)
        self.assertFalse(any("applied by component id" in m for m in logged), logged)
        for k in range(3):
            torch.testing.assert_close(loaded[k], self.reference[k], rtol=1e-9, atol=1e-9, msg=f"Zone{k}")
        np.testing.assert_allclose(meta.UA.max_value.numpy(), [400.0, 500.0, 600.0])

    def test_the_result_carries_what_the_fit_held_fixed(self):
        """Every estimable parameter the fit did not estimate is in the
        result at the value the fit ran with; loading sets it back on a
        model whose own setup gave it another value (``fixed=False``
        leaves it)."""
        held = self.result["parameter_instances_fixed"]
        estimated = set(self.result["parameter_instances"])
        self.assertTrue(held)
        self.assertFalse(set(held) & estimated)
        key = ("Zone0", "Q_occ_gain")  # a pinned occupant gain, as a site pins it
        self.assertIn(key, held)
        fitted_value = float(np.asarray(held[key]).reshape(-1)[0])

        def fresh(model_id):
            model = self.with_sensors(build(n_pairs=3, model_id=model_id))
            model.set_parameter_values({key: 3.0 * fitted_value})  # a setup of its own
            return model

        model = fresh("pv_fit_held")
        model.load_estimation_result(filename=self.filename)
        np.testing.assert_allclose(model.get_parameter_values([(model.components["Zone0"], "Q_occ_gain")])[key], fitted_value, rtol=1e-12)
        model = fresh("pv_fit_held_off")
        model.load_estimation_result(filename=self.filename, fixed=False)
        np.testing.assert_allclose(model.get_parameter_values([(model.components["Zone0"], "Q_occ_gain")])[key], 3.0 * fitted_value, rtol=1e-12)

    def test_a_result_without_the_values_by_component_loads_as_before(self):
        """A result saved before the values by component existed is set by
        its ids, and refused by a model that does not have them."""
        old = EstimationResult(
            **{
                k: v
                for k, v in self.result.items()
                if k not in ("parameter_instances", "parameter_instances_x0", "parameter_instance_bounds", "component_source_ids")
            }
        )
        model = self.with_sensors(build(n_pairs=3, model_id="pv_fit_old"))
        batched = batch(model)
        batched.load_estimation_result(result=old)
        meta, _ = model.get_batched_component_info("Radiator0")
        np.testing.assert_allclose(meta.UA.get().detach().numpy(), self.result["result_x"][:3], rtol=1e-9)
        with self.assertRaises(KeyError):
            model.load_estimation_result(result=old)


if __name__ == "__main__":
    unittest.main()
