"""A functional run fills the histories of the inputs nothing is wired to.

The composer hands ``forward`` the input an unwired ``Input(default=otherInput)``
reads (a room's ``makeUpAirTemperature`` reads ``outdoorTemperature``) and
skips a ``"wired"``-size vector with no wires (no slots).  The run filled the
histories of the wired ports and of the unwired ones on the exogenous tape
only, so after a functional run these ports held no history (#204 added
``makeUpAirTemperature``).
"""

import datetime
import unittest

import torch
from dateutil import tz

import twin4build as tb

START = datetime.datetime(2024, 1, 1, tzinfo=tz.UTC)


def _sched(value, sid):
    return tb.ScheduleSystem(weekday_ruleset={"ruleset_default_value": value}, id=sid)


def _room_model(model_id):
    model = tb.Model(id=model_id)
    room = tb.BuildingSpaceSystem(
        thermal=dict(C_air=2e6, C_wall=5e6, R_out=0.01, R_in=0.01, f_wall=0.0, f_air=0.0, Q_occ_gain=100.0),
        mass=dict(V=100.0, G_occ=5e-6, m_inf=0.01),
        id="room",
    )
    for port, value in (("outdoorTemperature", 5.0), ("supplyAirTemperature", 20.0), ("globalIrradiation", 0.0),
                        ("numberOfPeople", 2.0), ("heatGain", 500.0), ("outdoorCo2Concentration", 420.0)):
        model.add_connection(_sched(value, f"room_{port}"), room, "scheduleValue", port)
    model.add_connection(_sched(0.1, "room_sup"), room, "scheduleValue", "supplyAirFlowRate", input_port_index=0)
    model.add_connection(_sched(0.12, "room_exh"), room, "scheduleValue", "exhaustAirFlowRate", input_port_index=0)
    model.load(draw_semantic_model=False, draw_simulation_model=False)
    return model, room


class TestFunctionalUnwiredInputHistories(unittest.TestCase):
    def test_every_logged_input_history_is_filled(self):
        model, room = _room_model("functional_unwired_input_histories")
        tb.Simulator(model, execution_mode="functional").simulate(
            start_time=START, end_time=START + datetime.timedelta(hours=1), step_size=600, show_progress_bar=False
        )
        for component in model.components.values():
            for name, port in component.input.items():
                if port.log_history and port._history is not None:
                    self.assertTrue(port._history_is_populated, f"{component.id}.{name}")
        # an input reading another holds the values it read
        for name, read in (("makeUpAirTemperature", "outdoorTemperature"),
                           ("makeUpAirCO2", "outdoorCo2Concentration"),
                           ("supplyAirCO2", "outdoorCo2Concentration")):
            torch.testing.assert_close(room.input[name]._history, room.input[read]._history, rtol=0, atol=0)
        # a wired-size vector with no wires has no slots
        self.assertEqual(room.input["exchangeCO2Gain"]._history.shape[-1], 0)


if __name__ == "__main__":
    unittest.main()
