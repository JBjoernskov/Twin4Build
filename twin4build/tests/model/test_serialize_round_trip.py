"""``Model.serialize()`` -> ``Model.load(filename=...)`` must give the model back.

Regressions covered:

* nested parameters (``thermal.C_wall``, ``mass.V``) are serialized as
  dotted literals that no constructor accepts, and used to reload at their
  class defaults;
* ``SpaceHeaterSystem`` had no ``UA`` constructor argument, so a fitted UA
  reloaded as the placeholder (and ``initialize`` then re-solved it);
* a sensor's unit transformation is a callable, not a literal, and was
  dropped -- a saved calibrated model replayed in raw units (#190).  It now
  travels as an import path (``transformation_ref``).
"""

# Standard library imports
import datetime
import os
import shutil
import tempfile
import unittest

# Third party imports
import pandas as pd
import torch
from dateutil import tz

# Local application imports
import twin4build as tb
from twin4build.systems.building_space.building_space_system import BuildingSpaceSystem
from twin4build.systems.schedule.schedule_system import ScheduleSystem
from twin4build.systems.sensor.sensor_system import SensorSystem
from twin4build.systems.space_heater.space_heater_system import SpaceHeaterSystem

tb._IS_TESTING = True


def celsius_from_deci(x):
    """A module-level transformation: importable by name."""
    return x / 10.0


def _schedule(id_, value):
    return ScheduleSystem(
        id=id_,
        weekday_ruleset={
            "ruleset_start_minute": [], "ruleset_end_minute": [], "ruleset_start_hour": [],
            "ruleset_end_hour": [], "ruleset_value": [], "ruleset_default_value": value,
        },
    )


class TestSerializeRoundTrip(unittest.TestCase):
    MODEL_ID = "test_serialize_round_trip"
    START = datetime.datetime(2023, 1, 1, tzinfo=tz.UTC)

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        index = pd.date_range(cls.START, periods=13, freq="10min", tz="UTC")
        pd.DataFrame({"time": index, "value": [50.0 + i for i in range(13)]}).to_csv(
            os.path.join(cls.tmp, "outdoor.csv"), index=False
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)
        for mid in (cls.MODEL_ID, cls.MODEL_ID + "_reloaded"):
            path = os.path.join("generated_files", "models", mid)
            if os.path.exists(path):
                shutil.rmtree(path, ignore_errors=True)

    def _build(self):
        model = tb.Model(id=self.MODEL_ID)
        room = BuildingSpaceSystem(
            id="room", C_wall=1e6, C_air=1e4, C_boundary=5e5, R_out=0.01, R_in=0.02, R_boundary=0.03,
            f_wall=0.5, f_air=0.3, Q_occ_gain=100.0, CO2_occ_gain=0.004, CO2_start=400.0, airVolume=100.0,
            T_wall_start=20.0, T_air_start=20.0, T_int_start=20.0, T_boundary_start=18.0,
        )
        heater = SpaceHeaterSystem(id="heater")
        outdoor = SensorSystem(
            id="outdoor_sensor",
            filename=os.path.join(self.tmp, "outdoor.csv"),
            datecolumn=0, valuecolumn=1,
            use_spreadsheet=True,
            transformation=celsius_from_deci,
        )
        model.add_connection(outdoor, room, "measuredValue", "outdoorTemperature")
        model.add_connection(_schedule("people", 0.0), room, "scheduleValue", "numberOfPeople")
        model.add_connection(_schedule("sun", 0.0), room, "scheduleValue", "globalIrradiation")
        model.add_connection(_schedule("flow", 0.05), room, "scheduleValue", "supplyAirFlowRate")
        model.add_connection(_schedule("t_sup", 18.0), room, "scheduleValue", "supplyAirTemperature")
        model.add_connection(_schedule("water", 0.01), heater, "scheduleValue", "waterFlowRate")
        model.add_connection(_schedule("t_water", 60.0), heater, "scheduleValue", "supplyWaterTemperature")
        model.add_connection(room, heater, "indoorTemperature", "indoorTemperature")
        model.add_connection(heater, room, "Power", "heatGain")
        model.load()
        # "Fitted" values, nested and flat, written the way the estimator writes them.
        model.simulation_model.set_parameters(
            [4.2e6, 150.0, 37.5, 2.5e5],
            [room, room, heater, heater],
            ["thermal.C_wall", "mass.V", "UA", "thermalMassHeatCapacity"],
            overwrite=True,
        )
        heater.initialize_UA = False
        return model, room, heater, outdoor

    def _simulate(self, model):
        sim = tb.Simulator(model, execution_mode="object")
        sim.simulate(
            start_time=self.START, end_time=self.START + datetime.timedelta(hours=2),
            step_size=600, show_progress_bar=False,
        )
        return model.components["room"].output["indoorTemperature"]._history.detach().clone()

    def test_values_transformation_and_simulation_survive(self):
        model, room, heater, outdoor = self._build()
        reference = self._simulate(model)
        model.serialize()
        path, _ = model._simulation_model._semantic_model.get_dir(filename="instance_graph.ttl")

        reloaded = tb.Model(id=self.MODEL_ID + "_reloaded")
        reloaded.load(filename=path)
        self.assertEqual(set(reloaded.components), set(model.components))
        room2, heater2, outdoor2 = (reloaded.components[k] for k in ("room", "heater", "outdoor_sensor"))
        self.assertAlmostEqual(float(room2.thermal.C_wall.get().reshape(-1)[0]), 4.2e6, delta=1.0)
        self.assertAlmostEqual(float(room2.mass.V.get().reshape(-1)[0]), 150.0, places=6)
        self.assertAlmostEqual(float(heater2.UA.get().reshape(-1)[0]), 37.5, places=6)
        self.assertAlmostEqual(float(heater2.thermalMassHeatCapacity.get().reshape(-1)[0]), 2.5e5, delta=1e-3)
        self.assertFalse(heater2.initialize_UA)
        self.assertIs(outdoor2.transformation, celsius_from_deci)
        self.assertEqual(outdoor2.transformation_ref, f"{__name__}:celsius_from_deci")

        torch.testing.assert_close(self._simulate(reloaded), reference)


if __name__ == "__main__":
    unittest.main()
