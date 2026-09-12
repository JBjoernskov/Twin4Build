"""``rewire(mode="playback")``: the identified loop is opened and the
controller outputs its historised command, so the plant is driven by what
the BMS commanded while the physics is calibrated."""

# Standard library imports
import unittest

# Third party imports
import pandas as pd
import torch

# Local application imports
import twin4build as tb
from twin4build.systems.controller.controller_identification.controller_identification_pi_system import (
    ControllerIdentificationPISystem,
)
from twin4build.systems.controller.controller_identification.pi_loop_rewire import (
    _apply_playback,
)
from twin4build.systems.sensor.sensor_system import SensorSystem

tb._IS_TESTING = True


def _outgoing(component, port):
    return [
        (cp.connection_point_of, cp.input_port)
        for conn in component.connected_through
        if conn.output_port == port
        for cp in conn.connects_system_at
    ]


class TestPlayback(unittest.TestCase):
    def test_forward_returns_the_measured_command(self):
        cits = ControllerIdentificationPISystem(id="cits", n_sensors=1, n_setpoints=1, n_on_off_signals=1)
        cits.n_actuators = 1
        cits._build_components()
        cits.playback = True
        measured = torch.tensor([[0.37]], dtype=torch.float64)
        inputs = {
            "sensorValue": torch.tensor([[21.0]], dtype=torch.float64),
            "setpointValue": torch.tensor([[23.0]], dtype=torch.float64),
            "onOffSignal": torch.tensor([[1.0]], dtype=torch.float64),
            "actuatorMeasured": measured,
        }
        x = None
        x_next, out = cits.forward(x, inputs, {}, 600.0)
        self.assertIs(x_next, x)
        torch.testing.assert_close(out["inputSignal"], measured)

    def test_apply_playback_reverses_the_command_sensor(self):
        model = tb.Model(id="test_cits_playback")
        cits = ControllerIdentificationPISystem(id="cits", n_sensors=1, n_setpoints=1, n_on_off_signals=1)
        command = SensorSystem(
            id="cmd", df=pd.DataFrame({"time": pd.date_range("2024-01-01", periods=3, freq="10min"), "value": [0.0, 0.5, 1.0]}),
            use_df=True,
        )
        setpoint = SensorSystem(
            id="sp", df=pd.DataFrame({"time": pd.date_range("2024-01-01", periods=3, freq="10min"), "value": [22.0] * 3}),
            use_df=True,
        )
        model.add_connection(setpoint, cits, "measuredValue", "setpointValue", input_port_index=0)
        model.add_connection(cits, command, "inputSignal", "measuredValue", output_port_index=0)
        self.assertEqual(_outgoing(cits, "inputSignal"), [(command, "measuredValue")])

        _apply_playback(model.simulation_model, [cits])

        self.assertTrue(cits.playback)
        # controller -> command sensor is gone: the sensor is a data leaf again ...
        self.assertEqual(_outgoing(cits, "inputSignal"), [])
        self.assertEqual(command.connects_at, [])
        # ... and it now feeds the controller's actuatorMeasured slot 0.
        self.assertEqual(_outgoing(command, "measuredValue"), [(cits, "actuatorMeasured")])

        # A second rewire (a model reloaded from its serialized graph carries
        # the playback wiring but no flag) recognises the opened loop.
        cits.playback = False
        _apply_playback(model.simulation_model, [cits])
        self.assertTrue(cits.playback)
        self.assertEqual(_outgoing(command, "measuredValue"), [(cits, "actuatorMeasured")])
        self.assertEqual(_outgoing(cits, "inputSignal"), [])


if __name__ == "__main__":
    unittest.main()
