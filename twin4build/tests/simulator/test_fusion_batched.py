"""Fusion of batched metas, and the radiator as a fusable member.

Several copies of one cluster (a zone with its radiator; a pair of zones
with their wall) batch as clusters: every meta of the cluster type orders its
instances by the same cluster order, the arcs between the metas pair
instance i with instance i, and the batched model fuses the metas into one
block with an instance axis.  The batched fused simulation equals the
unbatched fused one per instance, in object mode and in the functional
engine; clusters of different shapes batch apart; the estimator's structure
walk sees one block per instance.
"""
import datetime
import unittest

import torch
from dateutil import tz

import twin4build as tb
from twin4build.systems.utils.fused_statespace_system import FusedStateSpaceSystem

tb._IS_TESTING = True

START = datetime.datetime(2024, 1, 4, tzinfo=tz.UTC)
STEP = 600


def _zone(zone_id, heated=True):
    return tb.BuildingSpaceThermalSystem(
        C_air=1e6, C_wall=5e6, R_out=0.01, R_in=0.01, f_wall=0.0, f_air=0.0,
        Q_occ_gain=100.0, id=zone_id,
    )


def _schedule(value, sid):
    return tb.ScheduleSystem(weekday_ruleset={"ruleset_default_value": value}, id=sid)


def build(n_pairs=3, radiators=True, walls=False, plain_zones=0, model_id="fusion_batched"):
    """``n_pairs`` copies of a heated zone; each with a radiator when
    ``radiators``, each paired to a second zone through a wall when
    ``walls``; plus ``plain_zones`` zones with neither."""
    model = tb.Model(id=model_id)
    outdoor = _schedule(5.0, "Outdoor")
    zero = _schedule(0.0, "Zero")
    supply_t = _schedule(20.0, "SupplyAirTemp")
    water_t = _schedule(60.0, "WaterTemp")
    flow = _schedule(0.02, "WaterFlow")
    for k in range(n_pairs):
        z = _zone(f"Zone{k}")
        for port, src in (
            ("outdoorTemperature", outdoor), ("supplyAirFlowRate", zero), ("exhaustAirFlowRate", zero),
            ("supplyAirTemperature", supply_t), ("globalIrradiation", zero), ("numberOfPeople", zero),
        ):
            model.add_connection(src, z, "scheduleValue", port)
        if radiators:
            r = tb.SpaceHeaterSystem(
                thermalMassHeatCapacity=5e4 * (1 + 0.1 * k), UA=40.0 + 5 * k, nelements=3, id=f"Radiator{k}"
            )
            model.add_connection(water_t, r, "scheduleValue", "supplyWaterTemperature")
            model.add_connection(flow, r, "scheduleValue", "waterFlowRate")
            model.add_connection(z, r, "indoorTemperature", "indoorTemperature")
            model.add_connection(r, z, "Power", "heatGain")
        else:
            model.add_connection(zero, z, "scheduleValue", "heatGain")
        if walls:
            z2 = _zone(f"Zone{k}b")
            for port, src in (
                ("outdoorTemperature", outdoor), ("supplyAirFlowRate", zero), ("exhaustAirFlowRate", zero),
                ("supplyAirTemperature", supply_t), ("globalIrradiation", zero), ("numberOfPeople", zero),
                ("heatGain", zero),
            ):
                model.add_connection(src, z2, "scheduleValue", port)
            w = tb.WallSystem(C=2e5, R_a=0.02 + 0.005 * k, R_b=0.02, id=f"Wall{k}")
            model.add_connection(z, w, "indoorTemperature", "temperatureA")
            model.add_connection(z2, w, "indoorTemperature", "temperatureB")
            model.add_connection(w, z, "heatFlowRateA", "wallHeatGain", input_port_index=0)
            model.add_connection(w, z2, "heatFlowRateB", "wallHeatGain", input_port_index=0)
    for k in range(plain_zones):
        z = _zone(f"Plain{k}")
        for port, src in (
            ("outdoorTemperature", outdoor), ("supplyAirFlowRate", zero), ("exhaustAirFlowRate", zero),
            ("supplyAirTemperature", supply_t), ("globalIrradiation", zero), ("numberOfPeople", zero),
            ("heatGain", zero),
        ):
            model.add_connection(src, z, "scheduleValue", port)
    model.load(draw_semantic_model=False, draw_simulation_model=False)
    return model


def simulate(model, hours=6, **kwargs):
    sim = tb.Simulator(model, **kwargs)
    sim.simulate(start_time=START, end_time=START + datetime.timedelta(hours=hours), step_size=STEP, show_progress_bar=False)
    return sim


def history(model, cid, port, batched=None):
    if batched is None:
        hist = model.components[cid].output[port].history()
        return hist.reshape(hist.shape[0], -1)[:, 0].detach().cpu().clone()
    meta, i_c = model._component_to_meta[cid]
    hist = meta.output[port].history()
    return hist.reshape(hist.shape[0], -1)[:, i_c].detach().cpu().clone()


class TestBatchedFusion(unittest.TestCase):
    def test_radiator_fuses_with_its_zone(self):
        model = build(n_pairs=1)
        fused = list(model.simulation_model._fused_components.values())
        self.assertEqual(len(fused), 1)
        self.assertEqual({m.id for m in fused[0].members}, {"Zone0", "Radiator0"})
        simulate(model)
        power = history(model, "Radiator0", "Power")
        self.assertTrue(torch.isfinite(power).all())
        self.assertGreater(float(power[-1]), 0.0)

    def test_power_row_matches_the_direct_formula(self):
        r = tb.SpaceHeaterSystem(thermalMassHeatCapacity=5e4, UA=40.0, nelements=3, id="r")
        r.initialize(start_time=[START], end_time=[START + datetime.timedelta(hours=1)], step_size=STEP)
        x = torch.tensor([[50.0, 45.0, 40.0]], dtype=torch.float64)
        inputs = {
            "supplyWaterTemperature": torch.tensor([60.0], dtype=torch.float64),
            "waterFlowRate": torch.tensor([0.02], dtype=torch.float64),
            "indoorTemperature": torch.tensor([21.0], dtype=torch.float64),
        }
        x_next, outs = r.forward(x, inputs, r._forward_params(), STEP)
        expected = 40.0 / 3 * torch.sum(x_next - 21.0, dim=-1)
        torch.testing.assert_close(outs["Power"], expected)

    def test_batched_equals_unbatched_fused(self):
        reference = build(n_pairs=3, model_id="ref")
        simulate(reference)
        model = build(n_pairs=3, model_id="batched_src")
        batched = model.batch_components()
        batched.load(draw_semantic_model=False, draw_simulation_model=False)
        fused = list(batched.simulation_model._fused_components.values())
        self.assertEqual(len(fused), 1)
        self.assertEqual(fused[0].n_c, 3)
        self.assertEqual({type(m).__name__ for m in fused[0].members}, {"BuildingSpaceThermalSystem", "SpaceHeaterSystem"})
        simulate(batched)
        for k in range(3):
            torch.testing.assert_close(history(model, f"Zone{k}", "indoorTemperature", batched), history(reference, f"Zone{k}", "indoorTemperature"))
            torch.testing.assert_close(history(model, f"Radiator{k}", "Power", batched), history(reference, f"Radiator{k}", "Power"))
        simulate(batched, execution_mode="functional", execution_backend="eager")
        for k in range(3):
            torch.testing.assert_close(history(model, f"Zone{k}", "indoorTemperature", batched), history(reference, f"Zone{k}", "indoorTemperature"))
            torch.testing.assert_close(history(model, f"Radiator{k}", "Power", batched), history(reference, f"Radiator{k}", "Power"))

    def test_clusters_of_different_shape_batch_apart(self):
        reference = build(n_pairs=2, walls=True, plain_zones=2, model_id="ref_mixed")
        simulate(reference)
        model = build(n_pairs=2, walls=True, plain_zones=2, model_id="mixed")
        batched = model.batch_components()
        batched.load(draw_semantic_model=False, draw_simulation_model=False)
        fused = list(batched.simulation_model._fused_components.values())
        # one fused meta over the two (zone, radiator, wall, zone) clusters; the plain zones stay unfused
        self.assertEqual(len(fused), 1)
        self.assertEqual(fused[0].n_c, 2)
        self.assertEqual(len(fused[0].members), 4)
        plain_meta = model._component_to_meta["Plain0"][0]
        self.assertIs(plain_meta, model._component_to_meta["Plain1"][0])
        self.assertIsNot(plain_meta, model._component_to_meta["Zone0"][0])
        simulate(batched)
        for cid in ("Zone0", "Zone1", "Zone0b", "Zone1b", "Plain0", "Plain1"):
            torch.testing.assert_close(history(model, cid, "indoorTemperature", batched), history(reference, cid, "indoorTemperature"), msg=cid)
        for cid in ("Wall0", "Wall1"):
            torch.testing.assert_close(history(model, cid, "wallTemperature", batched), history(reference, cid, "wallTemperature"), msg=cid)

    def test_members_with_different_signatures_split_the_cluster_type(self):
        """Two radiators fed from a sensor instead of a schedule batch apart
        from the others; their zones must batch apart too, or the room meta
        would face two radiator metas with no aligned pairing."""
        model = build(n_pairs=4, model_id="split")
        sensor = tb.SensorSystem(id="water_sensor", df=__import__("pandas").DataFrame(
            {"value": [55.0] * 40},
            index=__import__("pandas").DatetimeIndex([START + datetime.timedelta(seconds=STEP * k) for k in range(40)], name="time"),
        ), use_df=True)
        for k in (2, 3):
            r = model.components[f"Radiator{k}"]
            model.remove_connection(model.components["WaterTemp"], r, "scheduleValue", "supplyWaterTemperature")
            model.add_connection(sensor, r, "measuredValue", "supplyWaterTemperature")
        model.load(draw_semantic_model=False, draw_simulation_model=False)
        reference = build(n_pairs=4, model_id="split_ref")
        s2 = tb.SensorSystem(id="water_sensor", df=sensor.df, use_df=True)
        for k in (2, 3):
            r = reference.components[f"Radiator{k}"]
            reference.remove_connection(reference.components["WaterTemp"], r, "scheduleValue", "supplyWaterTemperature")
            reference.add_connection(s2, r, "measuredValue", "supplyWaterTemperature")
        reference.load(draw_semantic_model=False, draw_simulation_model=False)
        simulate(reference)
        batched = model.batch_components()
        batched.load(draw_semantic_model=False, draw_simulation_model=False)
        fused = list(batched.simulation_model._fused_components.values())
        self.assertEqual(sorted(f.n_c for f in fused), [2, 2])
        self.assertIsNot(model._component_to_meta["Zone0"][0], model._component_to_meta["Zone2"][0])
        simulate(batched)
        for k in range(4):
            torch.testing.assert_close(history(model, f"Zone{k}", "indoorTemperature", batched), history(reference, f"Zone{k}", "indoorTemperature"), msg=f"Zone{k}")

    def test_structure_walk_sees_one_block_per_instance(self):
        model = build(n_pairs=3, model_id="blocks")
        for k in range(3):
            model.add_connection(model.components[f"Zone{k}"], tb.SensorSystem(id=f"T{k}"), "indoorTemperature", "measuredValue")
        model.load(draw_semantic_model=False, draw_simulation_model=False)
        batched = model.batch_components()
        batched.load(draw_semantic_model=False, draw_simulation_model=False)
        end = START + datetime.timedelta(hours=1)
        batched.initialize(start_time=[START], end_time=[end], step_size=STEP)
        zone_meta = model._component_to_meta["Zone0"][0]
        rad_meta = model._component_to_meta["Radiator0"][0]
        fused = batched.simulation_model._fusion_member_to_fused[zone_meta.id]
        theta_spec = [
            (fused, f"{fused._member_keys[zone_meta.id]}.C_air", slice(0, 3)),
            (fused, f"{fused._member_keys[rad_meta.id]}.UA", slice(3, 6)),
        ]
        sensors = [batched.components[f"T{k}"] for k in range(3)]
        simulator = tb.Simulator(batched, execution_mode="functional", execution_backend="eager")
        _, fm = simulator.build_functional_model(theta_spec=theta_spec, measurements=sensors, step_size=STEP)
        fm.prepare_routes(torch.device("cpu"))
        theta_block, column_block, n_blocks = fm.index_coupling()
        self.assertEqual(n_blocks, 3)
        self.assertEqual(list(theta_block[:3]), list(theta_block[3:6]))
        self.assertTrue((column_block >= 0).all())


if __name__ == "__main__":
    unittest.main()
