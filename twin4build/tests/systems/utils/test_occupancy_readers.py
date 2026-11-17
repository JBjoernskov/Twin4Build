"""The occupancy reads its measurements through its own readers (ADR 0001,
decision 6, amended): nothing connects the sensors to it, so no loop runs
through the model and the damper position sensor reads its damper at the
same step."""

# Standard library imports
import datetime
import unittest

# Third party imports
import numpy as np
import pandas as pd
import torch

# Local application imports
import twin4build as tb
from benchmarks import common
from twin4build.utils.data_loaders.sources import DataFrameSource

tb._IS_TESTING = True

START = datetime.datetime(2024, 1, 1, tzinfo=datetime.timezone.utc)
STEP = 600
CO2 = [600.0, 640.0, 700.0, 690.0, 650.0, 620.0]
DAMPERS = ([0.5, 0.5, 0.8, 1.0, 0.6, 0.4], [0.2, 0.3, 0.3, 0.5, 0.5, 0.4])


def _reader(values):
    index = pd.date_range(START, periods=len(values) + 1, freq=f"{STEP}s")
    return DataFrameSource(pd.Series(list(values) + [values[-1]], index=index))


def _occupancy(**kwargs):
    return tb.OccupancySystem(
        id="occupancy",
        mass=dict(V=100.0),
        supply_damper=dict(nominalAirFlowRate=0.1),
        exhaust_damper=dict(nominalAirFlowRate=0.1),
        **kwargs,
    )


def _run(model, steps=len(CO2)):
    model.load(draw_semantic_model=False, draw_simulation_model=False)
    tb.Simulator(model, execution_mode="object").simulate(
        start_time=START,
        end_time=START + datetime.timedelta(seconds=steps * STEP),
        step_size=STEP,
        show_progress_bar=False,
    )


def _history(port):
    return port.history().detach().reshape(port.history().shape[0], -1).numpy()


class TestOccupancyReaders(unittest.TestCase):
    def test_the_series_are_the_readers_and_the_step_before(self):
        occupancy = _occupancy(co2=_reader(CO2), damper_position=[_reader(d) for d in DAMPERS])
        model = tb.Model(id="occupancy_readers")
        model.add_component(occupancy)
        _run(model)
        np.testing.assert_array_equal(_history(occupancy.output["indoorCo2Measured"])[:, 0], CO2)
        # a period's first step takes its own CO2: no change, the static balance
        np.testing.assert_array_equal(_history(occupancy.output["previousIndoorCo2Measured"])[:, 0], [CO2[0]] + CO2[:-1])
        np.testing.assert_array_equal(_history(occupancy.output["damperPositionMeasured"]), np.array(DAMPERS).T)
        # one damper element per reader, and the inputs hold what they read
        self.assertEqual(occupancy.supply_damper.nominalAirFlowRate.get().numel(), 2)
        np.testing.assert_array_equal(_history(occupancy.input["indoorCo2Measured"])[:, 0], CO2)
        self.assertTrue(bool(torch.isfinite(occupancy.output["scheduleValue"].history()).all()))

    def test_it_needs_a_reader_of_the_co2(self):
        model = tb.Model(id="occupancy_without_co2")
        model.add_component(_occupancy())
        with self.assertRaises(ValueError) as raised:
            _run(model)
        self.assertIn("no reader of the room's CO2", str(raised.exception))

    def test_its_measurements_are_not_wired(self):
        occupancy = _occupancy(co2=_reader(CO2))
        co2 = tb.ScheduleSystem(
            id="co2",
            weekday_ruleset={
                "ruleset_default_value": 600.0,
                "ruleset_start_minute": [],
                "ruleset_end_minute": [],
                "ruleset_start_hour": [],
                "ruleset_end_hour": [],
                "ruleset_value": [],
            },
        )
        model = tb.Model(id="occupancy_wired")
        model.add_connection(co2, occupancy, "scheduleValue", "indoorCo2Measured")
        with self.assertRaises(ValueError) as raised:
            _run(model)
        self.assertIn("indoorCo2Measured is wired", str(raised.exception))


class TestNoLoopThroughTheSensors(unittest.TestCase):
    def test_the_damper_sensor_reads_its_damper_at_the_same_step(self):
        # Wired from the sensors' measuredData, the occupancy closed the loop
        # occupancy -> room -> CO2 controller -> damper -> damper sensor ->
        # occupancy, the loader cut it into the damper sensor, and the
        # sensor's modelled reading lagged its damper by one step.
        model, parts = common.build_multizone_model(1, model_id="occupancy_no_loop")
        zone = parts["zone_parts"][0]
        occupancy = zone["office_occupancy"]
        self.assertEqual({point.input_port for point in occupancy.connects_at}, {"outdoorCo2Concentration"})
        self.assertIs(occupancy.co2, zone["office_co2_sensor"].source)
        self.assertEqual(occupancy.damper_position, (zone["office_damper_position_sensor"].source,))
        common._simulate(model, device="cpu", execution_mode="object", hours=6)
        sensor = zone["office_damper_position_sensor"].output["measuredValue"].history()
        damper = zone["office_damper_max"].output["value"].history()
        torch.testing.assert_close(sensor, damper, rtol=0, atol=0)
        self.assertGreater(float(damper.max() - damper.min()), 0.0)  # the damper moves


if __name__ == "__main__":
    unittest.main()
