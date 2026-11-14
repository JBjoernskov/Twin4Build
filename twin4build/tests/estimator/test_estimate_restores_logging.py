"""``Estimator.estimate`` puts every port's logging back when it returns or raises.

The fit's simulations log the measuring devices only (``set_save_simulation_result(False)``, then ``True`` for
them), and ``estimate`` left it so.  Since a port that does not log keeps no history and
``set_save_simulation_result`` sets vector ports too, a simulation after ``estimate`` kept no history of a vector
port such as a room's ``supplyAirFlowRate``: reading it raised "History is not populated".
"""

# Standard library imports
import datetime
import unittest
from unittest import mock

# Third party imports
import torch

# Local application imports
import twin4build as tb
import twin4build.utils.types as tps
from twin4build.tests.model.test_split_worlds import START, STEP, _room

tb._IS_TESTING = True

END = START + datetime.timedelta(hours=1)  # the sensors' series hold two hours


def _model(model_id):
    model = tb.Model(id=model_id)
    room, sensors = _room(model, "A")  # sensors: the room's temperature, its CO2
    model.load(draw_semantic_model=False, draw_simulation_model=False)
    return model, room, sensors


def _flags(model):
    return {
        (cid, side, name): port.log_history
        for cid, component in model.components.items()
        for side in ("input", "output")
        for name, port in getattr(component, side).items()
        if isinstance(port, (tps.Scalar, tps.Vector))
    }


def _estimate(simulator, room, sensors):
    tb.Estimator(simulator).estimate(
        start_time=START,
        end_time=END,
        step_size=STEP,
        parameters=[(room, "thermal.C_air", 2e6, 1e6, 1e7)],
        measurements=[(sensors[0], 0.1)],
        n_warmup=0,
        method=("scipy", "SLSQP", "ad"),
        options={"maxiter": 1},
    )


def _simulate(simulator):
    simulator.simulate(start_time=START, end_time=END, step_size=STEP, show_progress_bar=False)


class TestEstimateRestoresLogging(unittest.TestCase):
    def test_a_vector_port_s_history_is_readable_after_estimate(self):
        model, room, sensors = _model("estimate_restores_logging_vector")
        simulator = tb.Simulator(model)
        port = room.input["supplyAirFlowRate"]
        self.assertIsInstance(port, tps.Vector)
        _simulate(simulator)
        before = port.history().clone()
        _estimate(simulator, room, sensors)
        _simulate(simulator)
        self.assertEqual(tuple(port.history().shape)[0], 6)
        torch.testing.assert_close(port.history(), before, rtol=0, atol=0)  # a schedule's: C_air does not change it

    def test_every_flag_is_what_it_was(self):
        model, room, sensors = _model("estimate_restores_logging_flags")
        # some ports not logging before the fit, the measurement among them
        room.output["indoorCO2"].log_history = False
        model.set_save_simulation_result(False, c=[sensors[0]])
        before = _flags(model)
        _estimate(tb.Simulator(model), room, sensors)
        self.assertEqual(_flags(model), before)

    def test_every_flag_is_what_it_was_when_the_fit_raises(self):
        model, room, sensors = _model("estimate_restores_logging_raises")
        before = _flags(model)
        during = {}

        def fail(*args, **kwargs):
            during.update(_flags(model))
            raise RuntimeError("the solver failed")

        with mock.patch.object(tb.Estimator, "_run_schedule", side_effect=fail):
            with self.assertRaisesRegex(RuntimeError, "the solver failed"):
                _estimate(tb.Simulator(model), room, sensors)
        # the fit logged the measurement only
        self.assertFalse(during[("A", "input", "supplyAirFlowRate")])
        self.assertTrue(during[(sensors[0].id, "input", "measuredValue")])
        self.assertEqual(_flags(model), before)


if __name__ == "__main__":
    unittest.main()
