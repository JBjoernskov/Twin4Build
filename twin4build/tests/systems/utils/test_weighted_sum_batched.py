"""A ``WeightedSumSystem`` fed by a batched meta: the meta's ONE connection
carries a tensor of slot indices, and the sum must size its vector from the
wired slots, not from one index per connection."""
import unittest

import torch

import twin4build as tb

tb._IS_TESTING = True

from twin4build.tests.simulator.test_fusion_batched import build, history, simulate


class TestWeightedSumBatched(unittest.TestCase):
    def test_batched_zones_into_one_area_weighted_sum(self):
        reference = build(n_pairs=3, model_id="wsum_ref")
        model = build(n_pairs=3, model_id="wsum_src")
        for m in (reference, model):
            zones = sorted(m.get_components_by_class(tb.BuildingSpaceThermalSystem), key=lambda z: z.id)
            wsum = tb.WeightedSumSystem(weights=[0.5, 0.3, 0.2], id="mean_temperature")
            m.add_component(wsum)
            for k, z in enumerate(zones):
                m.add_connection(z, wsum, "indoorTemperature", "inputs", input_port_index=k)
        reference.load(draw_semantic_model=False, draw_simulation_model=False)
        simulate(reference)
        model.load(draw_semantic_model=False, draw_simulation_model=False)  # the sum joined after the first load
        batched = model.batch_components()
        batched.load(draw_semantic_model=False, draw_simulation_model=False)
        simulate(batched)  # initialises the sum on the batched model: one connection, three slots
        wsum_b = batched.components["mean_temperature"]
        self.assertEqual(wsum_b.input["inputs"].n_v, 3)
        got = wsum_b.output["value"].history().reshape(-1)
        expected = sum(w * history(reference, f"Zone{k}", "indoorTemperature") for k, w in enumerate([0.5, 0.3, 0.2]))
        torch.testing.assert_close(got.detach().cpu(), expected, rtol=1e-6, atol=1e-6)

    def _two_sums(self, weights_a, weights_b, model_id):
        """A reference model and a batched one, each with two weighted sums over the same three zones."""
        reference = build(n_pairs=3, model_id=f"{model_id}_ref")
        model = build(n_pairs=3, model_id=f"{model_id}_src")
        for m in (reference, model):
            zones = sorted(m.get_components_by_class(tb.BuildingSpaceThermalSystem), key=lambda z: z.id)
            for sum_id, weights in (("sum_a", weights_a), ("sum_b", weights_b)):
                wsum = tb.WeightedSumSystem(weights=weights, id=sum_id)
                m.add_component(wsum)
                for k, z in enumerate(zones):
                    m.add_connection(z, wsum, "indoorTemperature", "inputs", input_port_index=k)
        reference.load(draw_semantic_model=False, draw_simulation_model=False)
        simulate(reference)
        model.load(draw_semantic_model=False, draw_simulation_model=False)
        batched = model.batch_components()
        batched.load(draw_semantic_model=False, draw_simulation_model=False)
        simulate(batched)
        return reference, model, batched

    def _value(self, model, batched, sum_id):
        meta, i_c = model._component_to_meta[sum_id]
        return batched.components[meta.id].output["value"].history(i_c=i_c).reshape(-1).detach().cpu()

    def test_differently_weighted_sums_keep_their_weights(self):
        # Two sums of one shape but different weights (an area-weighted overheating and underheating) must not
        # share a meta: a meta is built with its constructor's arguments, and without the weights it sums plainly.
        weights_a, weights_b = [0.5, 0.3, 0.2], [0.1, 0.1, 0.8]
        reference, model, batched = self._two_sums(weights_a, weights_b, "wsum_two")
        self.assertIsNot(model._component_to_meta["sum_a"][0], model._component_to_meta["sum_b"][0])
        for sum_id, weights in (("sum_a", weights_a), ("sum_b", weights_b)):
            expected = sum(w * history(reference, f"Zone{k}", "indoorTemperature") for k, w in enumerate(weights))
            torch.testing.assert_close(self._value(model, batched, sum_id), expected, rtol=1e-6, atol=1e-6)

    def test_identically_weighted_sums_share_a_meta(self):
        weights = [0.5, 0.3, 0.2]
        reference, model, batched = self._two_sums(weights, weights, "wsum_same")
        self.assertIs(model._component_to_meta["sum_a"][0], model._component_to_meta["sum_b"][0])
        expected = sum(w * history(reference, f"Zone{k}", "indoorTemperature") for k, w in enumerate(weights))
        for sum_id in ("sum_a", "sum_b"):
            torch.testing.assert_close(self._value(model, batched, sum_id), expected, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
