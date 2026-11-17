"""A star of walls into one shared ThermalMassNodeSystem: the node moves
with the rooms, the zone-wall pairs fuse and batch while the node stays its
own block, and the batched, functional and object-mode simulations agree."""
import datetime
import unittest

import torch

import twin4build as tb
from twin4build.tests.simulator.test_fusion_batched import START, STEP, _schedule, _zone, history, simulate

tb._IS_TESTING = True


def build_star(n_zones=3, model_id="node_star"):
    model = tb.Model(id=model_id)
    outdoor = _schedule(5.0, "Outdoor")
    zero = _schedule(0.0, "Zero")
    supply_t = _schedule(20.0, "SupplyAirTemp")
    node = tb.ThermalMassNodeSystem(C=5e7, T_init=24.0, id="Core")
    model.add_component(node)
    for k in range(n_zones):
        z = _zone(f"Zone{k}")
        for port, src in (
            ("outdoorTemperature", outdoor), ("supplyAirFlowRate", zero), ("exhaustAirFlowRate", zero),
            ("supplyAirTemperature", supply_t), ("globalIrradiation", zero), ("numberOfPeople", zero),
            ("heatGain", zero),
        ):
            model.add_connection(src, z, "scheduleValue", port)
        w = tb.WallSystem(C=2e5, R_a=0.02 + 0.005 * k, R_b=0.02, T_init=22.0, id=f"Wall{k}")
        model.add_connection(z, w, "indoorTemperature", "temperatureA")
        model.add_connection(node, w, "temperature", "temperatureB")
        model.add_connection(w, z, "heatFlowRateA", "wallHeatGain", input_port_index=0)
        model.add_connection(w, node, "heatFlowRateB", "heatFlowRate", input_port_index=k)
    model.load(draw_semantic_model=False, draw_simulation_model=False)
    return model


class TestThermalMassNodeStar(unittest.TestCase):
    def test_node_is_fed_by_every_wall_and_cools_toward_the_rooms(self):
        model = build_star(3)
        simulate(model, hours=12)
        node = model.components["Core"]
        self.assertEqual(node.n_flows, 3)
        t = history(model, "Core", "temperature")
        self.assertTrue(torch.isfinite(t).all())
        self.assertLess(float(t[-1]), 24.0)  # the warm node feeds the cooler rooms
        self.assertGreater(float(t[-1]), 15.0)

    def test_node_edges_are_cut_by_declaration_not_by_cycle_search(self):
        """Every node -> wall edge is a declared one-step lag: it is in the
        loader's cut list, the node runs last, and the walls read the node's
        initial temperature at the first step."""
        model = build_star(3, model_id="star_lag")
        sm = model.simulation_model
        cut = {(a, b) for a, b in sm._removed_cycle_edges}
        self.assertEqual(len(cut), 3)
        self.assertTrue(all(a == "Core" for a, _ in cut))
        order = [c.id for grp in sm._execution_order for c in grp]
        self.assertEqual(order[-1], "Core")
        simulate(model, hours=1)
        h = model.components["Wall0"].input["temperatureB"].history()
        self.assertAlmostEqual(float(h.reshape(h.shape[0], -1)[0, 0]), 24.0, places=9)

    def test_zone_wall_pairs_fuse_and_the_node_stays_apart(self):
        model = build_star(3, model_id="star_fused")
        fused = list(model.simulation_model._fused_components.values())
        self.assertEqual(len(fused), 3)
        for f in fused:
            self.assertEqual({type(m).__name__ for m in f.members}, {"BuildingSpaceThermalSystem", "WallSystem"})

    def test_batched_and_functional_match_object_mode(self):
        reference = build_star(3, model_id="star_ref")
        simulate(reference, hours=12)
        model = build_star(3, model_id="star_src")
        batched = model.batch_components()
        batched.load(draw_semantic_model=False, draw_simulation_model=False)
        fused = list(batched.simulation_model._fused_components.values())
        self.assertEqual(len(fused), 1)
        self.assertEqual(fused[0].n_c, 3)
        for mode, backend in (("object", None), ("functional", "eager")):
            kwargs = {"execution_mode": mode, "execution_backend": backend} if backend else {}
            simulate(batched, hours=12, **kwargs)
            torch.testing.assert_close(history(model, "Core", "temperature", batched), history(reference, "Core", "temperature"), rtol=1e-6, atol=1e-6, msg=mode)
            for k in range(3):
                torch.testing.assert_close(history(model, f"Zone{k}", "indoorTemperature", batched), history(reference, f"Zone{k}", "indoorTemperature"), rtol=1e-6, atol=1e-6, msg=f"{mode} Zone{k}")


if __name__ == "__main__":
    unittest.main()
