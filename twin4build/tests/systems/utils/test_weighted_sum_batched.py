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


if __name__ == "__main__":
    unittest.main()
