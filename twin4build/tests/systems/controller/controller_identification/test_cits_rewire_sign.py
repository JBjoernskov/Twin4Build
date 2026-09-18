"""The rewire's data-driven action sign must reach the PI the CITS runs:
``PIDControllerSystem.forward`` reads ``is_reverse``; writing only the
deprecated ``isReverse`` alias left every heating loop direct-acting."""

# Standard library imports
import datetime
import unittest

# Third party imports
import numpy as np
import pandas as pd

# Local application imports
import twin4build as tb
from twin4build.systems.controller.controller_identification.controller_identification_pi_system import (
    ControllerIdentificationPISystem,
)
from twin4build.systems.sensor.sensor_system import SensorSystem

tb._IS_TESTING = True


def _df(index, values):
    return pd.DataFrame({"value": np.asarray(values, dtype=float)}, index=index)


class TestRewireActionSign(unittest.TestCase):
    def test_reverse_acting_loop_gets_is_reverse(self):
        start = datetime.datetime(2024, 3, 4, 0, 0, tzinfo=datetime.timezone.utc)
        n = 3 * 144
        index = pd.date_range(start, periods=n, freq="10min")
        t = np.arange(n)
        # Room temperature swinging around the 22 C heating setpoint; a
        # reverse-acting P law opens the valve when the room is cold.
        fb = 22.0 + 1.5 * np.sin(2 * np.pi * t / 144) + 0.2 * np.sin(2 * np.pi * t / 23)
        sp = np.full(n, 22.0)
        u = np.clip(0.5 + 0.4 * (sp - fb), 0.0, 1.0)
        model = tb.Model(id="test_cits_rewire_sign")
        cits = ControllerIdentificationPISystem(id="cits", n_sensors=1, n_setpoints=1, n_on_off_signals=1)
        sensor = SensorSystem(id="zone_t", df=_df(index, fb), use_df=True)
        setpoint = SensorSystem(id="sp_heat", df=_df(index, sp), use_df=True)
        gate = SensorSystem(id="flow_sp", df=_df(index, np.full(n, 300.0)), use_df=True)
        command = SensorSystem(id="cmd", df=_df(index, u), use_df=True)
        model.add_connection(sensor, cits, "measuredValue", "sensorValue", input_port_index=0)
        model.add_connection(setpoint, cits, "measuredValue", "setpointValue", input_port_index=0)
        model.add_connection(gate, cits, "measuredValue", "onOffSignal", input_port_index=0)
        model.add_connection(cits, command, "inputSignal", "measuredValue", output_port_index=0)
        model.rewire(start_time=[start], end_time=[start + datetime.timedelta(days=3)], step_size=600, mode="train")
        rep = model.simulation_model.rewire_reports["cits"]
        self.assertTrue(rep.is_reverse, rep)
        self.assertTrue(cits.candidate_0_0.is_reverse)
        # A constant gate signal separates nothing: the gate is bypassed.
        self.assertAlmostEqual(float(cits._get_alpha_gate(0).reshape(-1)[0]), 0.0)

    def test_unexcited_loop_replays_its_command_in_simulate_mode(self):
        """A valve shut all window (command 0 with one short blip) cannot be
        identified; in simulate mode the loop is opened and replays the
        measured command, in train mode it is left closed."""
        start = datetime.datetime(2024, 3, 4, 0, 0, tzinfo=datetime.timezone.utc)
        n = 3 * 144
        index = pd.date_range(start, periods=n, freq="10min")
        fb = 22.5 + 1.0 * np.sin(2 * np.pi * np.arange(n) / 144)
        u = np.zeros(n)
        u[200:204] = 0.25
        for mode, expect_playback in (("simulate", True), ("train", False)):
            model = tb.Model(id=f"test_cits_unexcited_{mode}")
            cits = ControllerIdentificationPISystem(id="cits", n_sensors=1, n_setpoints=1, n_on_off_signals=1)
            sensor = SensorSystem(id="zone_t", df=_df(index, fb), use_df=True)
            setpoint = SensorSystem(id="sp_heat", df=_df(index, np.full(n, 22.0)), use_df=True)
            gate = SensorSystem(id="flow_sp", df=_df(index, np.full(n, 300.0)), use_df=True)
            command = SensorSystem(id="cmd", df=_df(index, u), use_df=True)
            model.add_connection(sensor, cits, "measuredValue", "sensorValue", input_port_index=0)
            model.add_connection(setpoint, cits, "measuredValue", "setpointValue", input_port_index=0)
            model.add_connection(gate, cits, "measuredValue", "onOffSignal", input_port_index=0)
            model.add_connection(cits, command, "inputSignal", "measuredValue", output_port_index=0)
            model.rewire(start_time=[start], end_time=[start + datetime.timedelta(days=3)], step_size=600, mode=mode)
            rep = model.simulation_model.rewire_reports["cits"]
            self.assertFalse(rep.excited, mode)
            self.assertEqual(bool(getattr(cits, "playback", False)), expect_playback, mode)


if __name__ == "__main__":
    unittest.main()
