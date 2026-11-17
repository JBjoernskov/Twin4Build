"""Histories only where they are used.

A port that does not log (``set_save_simulation_result``) and holds no data
keeps only its current value, scalar and vector ports alike, in object and
functional mode; what is logged is bit for bit what it was with every port
logging; a port that logs again has its history at the next simulation.
"""

# Standard library imports
import datetime
import unittest

# Third party imports
import torch
from dateutil import tz

# Local application imports
import twin4build as tb
from twin4build.tests.simulator.test_fusion import build_model

tb._IS_TESTING = True

START = datetime.datetime(2024, 1, 4, tzinfo=tz.gettz("Europe/Copenhagen"))
END = START + datetime.timedelta(hours=6)


def _simulate(model, mode):
    kwargs = {} if mode == "object" else {"execution_mode": "functional", "execution_backend": "eager"}
    tb.Simulator(model, **kwargs).simulate(start_time=START, end_time=END, step_size=600, show_progress_bar=False)


def _model(tag, log_all):
    model, zone_a, zone_b, wall = build_model(fuse=False, model_id=f"history_{tag}")
    if not log_all:
        model.set_save_simulation_result(False)
        model.set_save_simulation_result(True, c=[zone_a])
    return model, zone_a, zone_b, wall


class TestHistoryAllocation(unittest.TestCase):
    def _check(self, mode):
        model, zone_a, zone_b, wall = _model(f"lazy_{mode}", log_all=False)
        _simulate(model, mode)
        # the logged zone keeps its histories, outputs and inputs, the vector port too
        self.assertIsNotNone(zone_a.output["indoorTemperature"]._history)
        self.assertIsNotNone(zone_a.input["wallHeatGain"]._history)
        # the rest keep their current value only
        self.assertIsNone(zone_b.output["indoorTemperature"]._history)
        self.assertIsNone(zone_b.input["wallHeatGain"]._history)
        # the wall's side B feeds the unlogged zone only (its side A feeds the
        # logged zone's input, which a functional run fills from it)
        self.assertIsNone(wall.output["heatFlowRateB"]._history)
        self.assertIsNotNone(zone_b.output["indoorTemperature"]._tensor)
        # data publishers (schedules) keep their data
        self.assertIsNotNone(model.components["Outdoor"].output["scheduleValue"]._history)
        # what is logged is what it was with every port logging
        full, zone_a_full, _, _ = _model(f"full_{mode}", log_all=True)
        _simulate(full, mode)
        for port in ("indoorTemperature", "wallTemperature"):
            torch.testing.assert_close(
                zone_a.output[port].history(), zone_a_full.output[port].history(), rtol=0, atol=0
            )
        torch.testing.assert_close(
            zone_a.input["wallHeatGain"].history(), zone_a_full.input["wallHeatGain"].history(), rtol=0, atol=0
        )

    def test_object_mode(self):
        self._check("object")

    def test_functional_mode(self):
        self._check("functional")

    def test_logging_again_allocates_again(self):
        model, zone_a, zone_b, wall = _model("again", log_all=False)
        _simulate(model, "object")
        self.assertIsNone(zone_b.output["indoorTemperature"]._history)
        model.set_save_simulation_result(True)
        _simulate(model, "object")
        self.assertEqual(tuple(zone_b.output["indoorTemperature"].history().shape)[0], 36)
        self.assertIsNotNone(wall.output["heatFlowRateA"].history())


if __name__ == "__main__":
    unittest.main()
