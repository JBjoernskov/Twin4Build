"""``rewire(mode="playback")``: the identified loop is opened and the
controller outputs its historised command, so the plant is driven by what
the BMS commanded while the physics is calibrated."""

# Standard library imports
import datetime
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
    _resolve_actuator_measurement,
)
from twin4build.systems.sensor.sensor_system import SensorSystem
from twin4build.tests.model.test_batching_vector_slots import Sink
from twin4build.utils.data_loaders.sources import DataFrameSource

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
            id="cmd",
            source=DataFrameSource(pd.DataFrame({"time": pd.date_range("2024-01-01", periods=3, freq="10min"), "value": [0.0, 0.5, 1.0]})),
        )
        setpoint = SensorSystem(
            id="sp",
            source=DataFrameSource(pd.DataFrame({"time": pd.date_range("2024-01-01", periods=3, freq="10min"), "value": [22.0] * 3})),
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
        # the playback wiring but no flag) recognises the opened loop, and
        # still finds the command sensor as the loop's actuator measurement
        # (the data-driven seeds -- action sign, gate -- need it).
        cits.playback = False
        _apply_playback(model.simulation_model, [cits])
        self.assertTrue(cits.playback)
        self.assertIs(_resolve_actuator_measurement(cits), command)
        self.assertEqual(_outgoing(command, "measuredValue"), [(cits, "actuatorMeasured")])
        self.assertEqual(_outgoing(cits, "inputSignal"), [])


_START = datetime.datetime(2024, 1, 1, tzinfo=datetime.timezone.utc)
_END = _START + datetime.timedelta(hours=1)
_TIMES = pd.date_range(_START - datetime.timedelta(hours=1), _END + datetime.timedelta(hours=1), freq="10min")


def _sensor(id, values):
    return SensorSystem(id=id, source=DataFrameSource(pd.DataFrame({"value": values}, index=_TIMES)))


def _commanded(n, batch):
    """``n`` dampers, each driven by a controller in playback that replays its
    command sensor's series; each damper's flow integrated by a stateful sink
    (the functional engine needs one to roll)."""
    model = tb.Model(id=f"test_cits_playback_commanded_{n}_{batch}")
    dampers = []
    for k in range(n):
        cits = ControllerIdentificationPISystem(id=f"cits{k}", n_sensors=1, n_setpoints=1, n_on_off_signals=1)
        command = _sensor(f"cmd{k}", [0.1 * (k + 1) + 0.05 * i for i in range(len(_TIMES))])
        damper = tb.DamperSystem(id=f"damper{k}", nominalAirFlowRate=0.1, a=1.0)
        model.add_connection(_sensor(f"sp{k}", [22.0] * len(_TIMES)), cits, "measuredValue", "setpointValue", input_port_index=0)
        model.add_connection(_sensor(f"room{k}", [21.0] * len(_TIMES)), cits, "measuredValue", "sensorValue", input_port_index=0)
        model.add_connection(_sensor(f"onoff{k}", [1.0] * len(_TIMES)), cits, "measuredValue", "onOffSignal", input_port_index=0)
        model.add_connection(cits, damper, "inputSignal", "damperPosition", output_port_index=0)
        model.add_connection(cits, command, "inputSignal", "measuredValue", output_port_index=0)
        model.add_connection(damper, Sink(id=f"sink{k}"), "airFlowRate", "u", input_port_index=0)
        dampers.append(damper.id)
    model.rewire(start_time=[_START], end_time=[_END], step_size=600, mode="playback")
    model.load(draw_semantic_model=False, draw_simulation_model=False)
    if batch:
        model = model.batch_components()
        model.load(draw_semantic_model=False, draw_simulation_model=False)
    return model, dampers


def _positions(model, dampers):
    """Each damper's position over the run, from its batched meta when batched."""
    out = []
    for damper in dampers:
        for comp in model.components.values():
            ids = list(model.get_source_component_ids(comp) or [comp.id])
            if damper in ids:
                h = comp.input["damperPosition"].history(i_c=ids.index(damper))
                out.append(h.reshape(h.shape[0], -1)[:, 0].detach().cpu().clone())
                break
    return torch.stack(out)


class TestPlaybackInTheFunctionalEngine(unittest.TestCase):
    """A controller in playback replays its command sensor's recorded series
    on the functional engine's tape.  A 2.0 sensor's ``measuredValue`` is a
    stepped output (its ``measuredData`` the series the run start fills), so
    the replay read an empty ``measuredValue`` history, fell back to stepping
    the controller without its inputs, and every damper stood closed."""

    def test_the_actuator_follows_the_command_series(self):
        for n, batch in ((1, False), (2, True)):
            with self.subTest(n=n, batch=batch):
                expected = torch.tensor(
                    [[0.1 * (k + 1) + 0.05 * (6 + i) for i in range(6)] for k in range(n)]
                )
                results = {}
                for mode in ("object", "functional"):
                    model, dampers = _commanded(n, batch)
                    kwargs = {"execution_backend": "eager"} if mode == "functional" else {}
                    tb.Simulator(model, execution_mode=mode, **kwargs).simulate(
                        start_time=_START, end_time=_END, step_size=600, show_progress_bar=False
                    )
                    results[mode] = _positions(model, dampers)
                for mode, got in results.items():
                    torch.testing.assert_close(got, expected.to(got.dtype), msg=lambda m, mode=mode: f"{mode}: {m}")


if __name__ == "__main__":
    unittest.main()
