"""Learned schedule gate: a loop whose regimes no measured gate-bus slot
separates gets a ``ScheduleSystem`` on the gate bus, derived from the
actuator's own weekly pattern, and the gate selects it."""

# Standard library imports
import datetime
import unittest

# Third party imports
import numpy as np
import pandas as pd
import torch

# Local application imports
import twin4build as tb
from twin4build.systems.controller.controller_identification.controller_identification_pi_system import (
    ControllerIdentificationPISystem,
)
from twin4build.systems.sensor.sensor_system import SensorSystem

tb._IS_TESTING = True


def _df(index, values):
    """A df-backed sensor wants a datetime index and one value column."""
    return pd.DataFrame({"value": np.asarray(values, dtype=float)}, index=index)


class TestScheduleGate(unittest.TestCase):
    def test_rewire_adds_and_selects_the_schedule_slot(self):
        start = datetime.datetime(2024, 3, 4, 0, 0, tzinfo=datetime.timezone.utc)  # Monday
        n = 7 * 144
        index = pd.date_range(start, periods=n, freq="10min")
        stamps = index.to_pydatetime()
        # Room warm by day, cool at night; heating setpoint 22; the valve
        # is parked open outside working hours and shut inside them
        # (with short ramps) -- a schedule, not the flow setpoint, which
        # here is uninformative (constant).
        t_room = 22.5 + 1.5 * np.sin(2 * np.pi * (np.arange(n) % 144) / 144 - np.pi / 2)
        cmd = np.ones(n)
        for k, ts in enumerate(stamps):
            minutes = ts.hour * 60 + ts.minute
            if ts.weekday() < 5 and 8 * 60 <= minutes < 18 * 60:
                cmd[k] = 0.0
            elif ts.weekday() < 5 and 18 * 60 <= minutes < 19 * 60:
                cmd[k] = (minutes - 18 * 60) / 60.0
        model = tb.Model(id="test_cits_schedule_gate")
        cits = ControllerIdentificationPISystem(id="cits", n_sensors=1, n_setpoints=1, n_on_off_signals=1)
        sensor = SensorSystem(id="zone_t", df=_df(index, t_room), use_df=True)
        setpoint = SensorSystem(id="sp_heat", df=_df(index, np.full(n, 22.0)), use_df=True)
        flat_gate = SensorSystem(id="flow_sp", df=_df(index, np.full(n, 300.0)), use_df=True)
        command = SensorSystem(id="cmd", df=_df(index, cmd), use_df=True)
        model.add_connection(sensor, cits, "measuredValue", "sensorValue", input_port_index=0)
        model.add_connection(setpoint, cits, "measuredValue", "setpointValue", input_port_index=0)
        model.add_connection(flat_gate, cits, "measuredValue", "onOffSignal", input_port_index=0)
        model.add_connection(cits, command, "inputSignal", "measuredValue", output_port_index=0)

        reports = model.rewire(start_time=[start], end_time=[start + datetime.timedelta(days=7)], step_size=600, mode="train")
        reports = reports if isinstance(reports, dict) else model.simulation_model.rewire_reports
        rep = reports["cits"]
        self.assertIsNotNone(rep.schedule_seeds, "no schedule gate was offered")
        wd = rep.schedule_seeds.rulesets["weekday_ruleset"]
        self.assertTrue(7 <= wd["ruleset_start_hour"][0] <= 9)
        self.assertTrue(17 <= wd["ruleset_end_hour"][0] <= 19)
        schedule = model.components.get("cits_schedule_gate")
        self.assertIsNotNone(schedule)
        self.assertEqual(cits.n_on_off_signals, 2)
        gamma = cits.gamma_gate_0.get().detach().reshape(-1)
        self.assertEqual(int(torch.argmax(gamma)), 1, f"gamma_gate {gamma.tolist()}")
        self.assertGreater(float(gamma[1]), 0.9)


if __name__ == "__main__":
    unittest.main()
