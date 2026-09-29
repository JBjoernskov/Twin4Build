"""A ``FunctionSystem`` batched into one meta: its ports must carry one
slot per member, so every member's routed input lands in bounds."""
import unittest

import torch

import twin4build as tb

tb._IS_TESTING = True

from twin4build.tests.simulator.test_fusion_batched import build, history, simulate


def _plus_one(inputs):
    return inputs["t"] + 1.0


class TestFunctionSystemBatched(unittest.TestCase):
    def test_batched_function_systems_keep_one_slot_per_member(self):
        reference = build(n_pairs=3, model_id="fn_ref")
        model = build(n_pairs=3, model_id="fn_src")
        for m in (reference, model):
            zones = sorted(m.get_components_by_class(tb.BuildingSpaceThermalSystem), key=lambda z: z.id)
            for k, z in enumerate(zones):
                # one shared callable: members batch on their constructor signature
                fn = tb.FunctionSystem(inputs=["t"], fn=_plus_one, id=f"plus_one_{k}")
                m.add_component(fn)
                m.add_connection(z, fn, "indoorTemperature", "t")
        reference.load(draw_semantic_model=False, draw_simulation_model=False)
        simulate(reference)
        model.load(draw_semantic_model=False, draw_simulation_model=False)
        batched = model.batch_components()
        batched.load(draw_semantic_model=False, draw_simulation_model=False)
        simulate(batched)
        metas = [c for c in batched.components.values() if isinstance(c, tb.FunctionSystem)]
        self.assertEqual(len(metas), 1, [c.id for c in metas])
        meta = metas[0]
        self.assertEqual(meta.input["t"].n_c, 3)
        self.assertEqual(meta.output["output"].n_c, 3)
        for k in range(3):
            got = history(model, f"plus_one_{k}", "output", batched=True)
            expected = history(reference, f"plus_one_{k}", "output")
            torch.testing.assert_close(got, expected, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
