"""Sensors on different series (database uuids or in-memory frames) with
identical wiring must not be batched into one meta: a meta carries one data
source, so batching them would make every member read the first's series."""
import datetime
import unittest

import numpy as np
import pandas as pd
import torch

import twin4build as tb

tb._IS_TESTING = True

from twin4build.tests.simulator.test_fusion_batched import build, simulate


def _frame(start, hours, value):
    index = pd.date_range(start, periods=hours * 6, freq="600s")
    return pd.DataFrame({"value": np.full(len(index), float(value))}, index=pd.DatetimeIndex(index, name="time"))


class TestBatchSensorSources(unittest.TestCase):
    def test_sensors_on_their_own_frames_stay_apart(self):
        model = build(n_pairs=2, model_id="sensor_sources")
        zones = sorted(model.get_components_by_class(tb.BuildingSpaceThermalSystem), key=lambda z: z.id)
        start = datetime.datetime(2024, 1, 1, tzinfo=datetime.timezone.utc)
        sensors = []
        for k, z in enumerate(zones):
            s = tb.SensorSystem(id=f"T_meas_{k}", df=_frame(start, 12, 20.0 + k), use_df=True)
            model.add_component(s)
            model.add_connection(z, s, "indoorTemperature", "measuredValue")
            sensors.append(s)
        model.load(draw_semantic_model=False, draw_simulation_model=False)
        batched = model.batch_components()
        metas = [model._component_to_meta[s.id] for s in sensors]
        self.assertIsNot(metas[0][0], metas[1][0], "two sensors on different frames were batched into one meta")
        for meta, i_c in metas:
            self.assertEqual(int(getattr(meta, "n_c", 1) or 1), 1)
        batched.load(draw_semantic_model=False, draw_simulation_model=False)
        simulate(batched, hours=6)
        for k, s in enumerate(sensors):
            meta, _ = model._component_to_meta[s.id]
            ts = meta.time_series_input
            self.assertAlmostEqual(float(np.asarray(ts.values).reshape(-1)[0]), 20.0 + k)


if __name__ == "__main__":
    unittest.main()
