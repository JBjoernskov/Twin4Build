"""The model's state: ``Model.get_state`` / ``set_state`` / ``clear_state``.

A state set on the model is where the simulations that follow start: it is
applied at every ``initialize`` after the components have set their own
initial conditions, by the ids of the components the model was built from
(a fused block's members, a batched meta's instances), for the period whose
start time the row names.  ``load_estimation_result`` sets it from the
initial states an estimation result carries.
"""
import datetime
import unittest

import torch

import twin4build as tb

tb._IS_TESTING = True

from twin4build.tests.simulator.test_fusion_batched import START, STEP, build, history, simulate

END = START + datetime.timedelta(hours=6)


def _init(model, starts=None):
    starts = starts or [START]
    model.initialize(starts, [s + datetime.timedelta(hours=6) for s in starts], [STEP] * len(starts))


class TestModelState(unittest.TestCase):
    def test_get_state_names_the_components_the_model_was_built_from(self):
        model = build(n_pairs=2, model_id="state_ids")
        _init(model)
        state = model.get_state()
        # the zone and its radiator execute as one fused block; its members are reported
        self.assertEqual(set(state), {"Zone0", "Radiator0", "Zone1", "Radiator1"})
        self.assertEqual(tuple(state["Radiator0"].shape), (1, 3))  # one period, three elements
        self.assertEqual(state["Zone0"].shape[0], 1)

    def test_the_simulation_starts_from_the_set_state(self):
        reference = build(n_pairs=2, model_id="state_ref")
        simulate(reference)
        model = build(n_pairs=2, model_id="state_set")
        _init(model)
        warm = model.get_state()["Zone0"][0] + 4.0
        model.set_state({"Zone0": warm})
        _init(model)
        torch.testing.assert_close(model.get_state()["Zone0"][0], warm)
        simulate(model)
        t_set, t_ref = history(model, "Zone0", "indoorTemperature"), history(reference, "Zone0", "indoorTemperature")
        self.assertGreater(float(t_set[0] - t_ref[0]), 1.0)  # the first step starts 4 K warmer
        # the other zone keeps its own initial condition and its trajectory
        torch.testing.assert_close(history(model, "Zone1", "indoorTemperature"), history(reference, "Zone1", "indoorTemperature"))
        # a second simulation starts from the set state again, not from the end of the first
        simulate(model)
        torch.testing.assert_close(history(model, "Zone0", "indoorTemperature"), t_set)

    def test_clear_state_restores_the_components_defaults(self):
        reference = build(n_pairs=1, model_id="state_clear_ref")
        simulate(reference)
        model = build(n_pairs=1, model_id="state_clear")
        _init(model)
        model.set_state({"Zone0": model.get_state()["Zone0"][0] + 4.0})
        model.clear_state()
        simulate(model)
        torch.testing.assert_close(history(model, "Zone0", "indoorTemperature"), history(reference, "Zone0", "indoorTemperature"))

    def test_nan_entries_keep_the_components_own_value(self):
        model = build(n_pairs=1, model_id="state_nan")
        _init(model)
        default = model.get_state()["Radiator0"][0]
        partial = default.clone()
        partial[0] = 55.0
        partial[1:] = float("nan")
        model.set_state({"Radiator0": partial})
        _init(model)
        new = model.get_state()["Radiator0"][0]
        self.assertAlmostEqual(float(new[0]), 55.0)
        torch.testing.assert_close(new[1:], default[1:])

    def test_rows_go_to_the_periods_that_start_at_their_time(self):
        model = build(n_pairs=1, model_id="state_periods")
        second = START + datetime.timedelta(hours=6)
        _init(model)
        base = model.get_state()["Zone0"][0]
        rows = torch.stack([base + 1.0, base + 2.0])
        model.set_state({"Zone0": rows}, period_starts=[START, second])
        _init(model, [second])  # the second period alone takes the second row
        torch.testing.assert_close(model.get_state()["Zone0"][0], rows[1])
        _init(model, [START, second])
        torch.testing.assert_close(model.get_state()["Zone0"], rows)
        _init(model, [START + datetime.timedelta(days=3)])  # a time the state was not set for: the defaults
        torch.testing.assert_close(model.get_state()["Zone0"][0], base)

    def test_a_batched_model_sets_its_instances(self):
        reference = build(n_pairs=3, model_id="state_batched_ref")
        _init(reference)
        target = reference.get_state()["Zone1"][0] + 3.0
        reference.set_state({"Zone1": target})
        simulate(reference)
        model = build(n_pairs=3, model_id="state_batched")
        batched = model.batch_components()
        batched.load(draw_semantic_model=False, draw_simulation_model=False)
        batched.set_state({"Zone1": target})
        _init(batched)
        state = batched.get_state()
        self.assertEqual(set(state), {f"{kind}{k}" for kind in ("Zone", "Radiator") for k in range(3)})
        torch.testing.assert_close(state["Zone1"][0], target)
        simulate(batched)
        for k in range(3):
            torch.testing.assert_close(
                history(model, f"Zone{k}", "indoorTemperature", batched), history(reference, f"Zone{k}", "indoorTemperature")
            )
        simulate(batched, execution_mode="functional", execution_backend="eager")
        for k in range(3):
            torch.testing.assert_close(
                history(model, f"Zone{k}", "indoorTemperature", batched), history(reference, f"Zone{k}", "indoorTemperature")
            )

    def test_state_of_another_size_is_refused(self):
        model = build(n_pairs=1, model_id="state_size")
        model.set_state({"Radiator0": torch.zeros(2)})  # three elements
        with self.assertRaises(AssertionError):
            _init(model)


if __name__ == "__main__":
    unittest.main()
