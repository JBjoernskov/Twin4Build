"""Identified loops that the data decide, not the structure assumes.

* The rewire ranks a loop's (feedback, setpoint) pairs by the open-loop fit
  of the PI to the command, every sample scored: a radiator valve that stays
  shut all night while the room sits below a flat 22 C setpoint follows the
  20 C setback setpoint, and only the shut samples say so.
* A loop on another feedback joined by max (a CO2 loop over a VAV damper's
  temperature loop, ``max(temperature PI, CO2 P)``): the ``"max"``
  candidate, its constant setpoint and its gains are identified from the
  command, and the controller commands the larger demand.
"""

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
from twin4build.systems.controller.controller_identification.loop_fit import fit_max_loop, fit_pi, pi_outputs
from twin4build.systems.controller.setpoint_controller.pid_controller.pid_controller_system import (
    PIDControllerSystem,
)
from twin4build.systems.sensor.sensor_system import SensorSystem
from twin4build.systems.utils.smooth_saturation import saturation_mode

tb._IS_TESTING = True

START = datetime.datetime(2024, 3, 4, 0, 0, tzinfo=datetime.timezone.utc)
STEP = 600.0
DAYS = 6
N = DAYS * 144
T = np.arange(N)
HOUR = (T % 144) / 6.0
DAY = (HOUR >= 7) & (HOUR < 17)


def _df(values):
    index = pd.date_range(START, periods=len(values), freq="10min")
    return pd.DataFrame({"value": np.asarray(values, dtype=float)}, index=index)


def _sensor(sid, values):
    return SensorSystem(id=sid, df=_df(values), use_df=True)


#: The temperature loop's setpoint of the damper rooms.
COOLING = 21.6


def _room():
    """A heated room: its temperature crosses the setback schedule (21 C by
    day, 20 C at night) both by day and at night, and a flat 22 C setpoint
    lies above it all the time."""
    rng = np.random.default_rng(1)
    temperature = 20.4 + 0.6 * np.sin(2 * np.pi * T / 144) + 0.8 * DAY + 0.05 * rng.normal(size=N)
    setback = np.where(DAY, 21.0, 20.0)
    return temperature, setback, np.full(N, 22.0)


def _ventilated_room():
    """A ventilated room: a morning meeting raises the CO2 above 900 ppm,
    the afternoon sun the temperature above the cooling setpoint
    (:data:`COOLING`); the two loops act at different hours."""
    rng = np.random.default_rng(2)
    temperature = 21.2 + 0.9 * np.sin(np.pi * np.clip(HOUR - 12, 0, 5) / 5) + 0.03 * rng.normal(size=N)
    co2 = 430 + 520 * np.sin(np.pi * np.clip(HOUR - 7, 0, 5) / 5) + 5 * rng.normal(size=N)
    return temperature, co2


class TestLoopFit(unittest.TestCase):
    def test_the_setpoint_level_is_identified(self):
        """The valve follows the setback setpoint: shut at night, as the room
        is above 20 C.  A flat 22 C setpoint would hold it open all night."""
        temperature, setback, flat = _room()
        u = pi_outputs(setback - temperature, np.array([0.4]), np.array([3600.0]), STEP)[:, 0]
        self.assertGreater(float(u.std()), 0.1)  # the valve modulates
        right = fit_pi(u, setback, temperature, STEP)
        wrong = fit_pi(u, flat, temperature, STEP)
        self.assertLess(right.rmse, 0.02)
        self.assertGreater(wrong.rmse, 5 * right.rmse)
        self.assertTrue(right.is_reverse)  # heating: opens below the setpoint

    def test_a_loop_joined_by_max_is_recovered(self):
        """A damper opened by the room temperature (direct PI about 21 C)
        and, proportionally above 800 ppm, by the CO2 (fully at 1000 ppm):
        the CO2 loop's setpoint, direction and proportional action are
        found from the command and the temperature loop's command."""
        temperature, co2 = _ventilated_room()
        base = pi_outputs(temperature - COOLING, np.array([0.3]), np.array([7200.0]), STEP)[:, 0]
        u = np.maximum(base, np.clip((co2 - 800.0) / 200.0, 0.0, 1.0))
        self.assertGreater(float((u - base).max()), 0.3)  # the CO2 loop acts
        self.assertGreater(float(base.max()), 0.2)  # and so does the temperature loop
        fit = fit_max_loop(u, co2, base, STEP)
        self.assertLess(fit.rmse, 0.02)
        self.assertFalse(fit.is_reverse)  # opens above its setpoint
        self.assertAlmostEqual(fit.setpoint, 800.0, delta=40.0)
        # proportional: an integral time of days adds a few hundredths over a
        # morning's error, which the command cannot tell from none
        self.assertGreater(fit.Ti, 1e5)
        self.assertAlmostEqual(fit.kp * 200.0, 1.0, delta=0.25)


def _max_controller(k=0):
    """A damper's controller: a direct PI on the temperature about the
    setpoint, and a direct proportional loop on the CO2 about 800 ppm
    (``k`` shifts it), joined by max; no gate."""
    cits = ControllerIdentificationPISystem(
        id=f"cits{k}", n_sensors=2, n_setpoints=1, n_on_off_signals=1, max_controllers=[PIDControllerSystem]
    )
    self_set = lambda name, value: getattr(cits, name).set(torch.tensor(value, dtype=torch.float64), normalized=False)
    self_set("alpha_0", [1.0, 0.0])
    self_set("beta_0", [1.0, 0.0])
    self_set("gamma_0", [1.0])
    self_set("beta_0_1", [0.0, 1.0])
    self_set("setpoint_0_1", 800.0 + 50.0 * k)
    self_set("alpha_gate_0", 0.0)
    main, co2_loop = cits.candidate_0_0, cits.candidate_0_1
    main.kp.set(torch.tensor(0.3, dtype=torch.float64), normalized=False)
    main.Ti.set(torch.tensor(7200.0, dtype=torch.float64), normalized=False)
    main.is_reverse = False
    co2_loop.kp.max_value = torch.tensor(1.0, dtype=torch.float64)
    co2_loop.kp.min_value = torch.tensor(1e-4, dtype=torch.float64)
    co2_loop.kp.set(torch.tensor(1 / 200.0, dtype=torch.float64), normalized=False)
    co2_loop.Ti.max_value = torch.tensor(1e13, dtype=torch.float64)
    co2_loop.Ti.set(torch.tensor(1e12, dtype=torch.float64), normalized=False)
    co2_loop.is_reverse = False
    return cits


def _max_model(controllers):
    temperature, co2 = _ventilated_room()
    model = tb.Model(id="test_cits_max_loop")
    for k, cits in enumerate(controllers):
        model.add_connection(_sensor(f"t{k}", temperature), cits, "measuredValue", "sensorValue", input_port_index=0)
        model.add_connection(_sensor(f"co2_{k}", co2), cits, "measuredValue", "sensorValue", input_port_index=1)
        model.add_connection(_sensor(f"sp{k}", np.full(N, COOLING)), cits, "measuredValue", "setpointValue", input_port_index=0)
        model.add_connection(_sensor(f"gate{k}", np.ones(N)), cits, "measuredValue", "onOffSignal", input_port_index=0)
        model.add_connection(cits, _sensor(f"cmd{k}", np.zeros(N)), "inputSignal", "measuredValue", output_port_index=0)
    model.load(draw_semantic_model=False, draw_simulation_model=False)
    return model


def _simulate(model, hours=48):
    simulator = tb.Simulator(model)
    with saturation_mode("hard"):
        simulator.simulate(start_time=START, end_time=START + datetime.timedelta(hours=hours), step_size=int(STEP), show_progress_bar=False)
    return simulator


class TestMaxCandidate(unittest.TestCase):
    def test_the_controller_commands_the_larger_demand(self):
        model = _max_model([_max_controller()])
        _simulate(model)
        command = model.components["cits0"].output["inputSignal"].history()[:, 0, 0, 0].detach().cpu().numpy()
        temperature, co2 = _ventilated_room()
        n = command.size
        main = pi_outputs(temperature[:n] - COOLING, np.array([0.3]), np.array([7200.0]), STEP)[:, 0]
        co2_loop = np.clip((co2[:n] - 800.0) / 200.0, 0.0, 1.0)
        np.testing.assert_allclose(command, np.maximum(main, co2_loop), atol=1e-6)
        self.assertGreater(float((co2_loop - main).max()), 0.3)  # the CO2 loop sets the command somewhere

    def test_controllers_with_a_max_loop_batch(self):
        ids = ["cits0", "cits1"]
        reference = _max_model([_max_controller(k) for k in range(2)])
        _simulate(reference)
        expected = {cid: reference.components[cid].output["inputSignal"].history()[:, 0, 0].detach().clone() for cid in ids}
        model = _max_model([_max_controller(k) for k in range(2)])
        batched = model.batch_components()
        metas = [c for c in batched.components.values() if isinstance(c, ControllerIdentificationPISystem)]
        self.assertEqual(len(metas), 1)
        batched.load(draw_semantic_model=False, draw_simulation_model=False)
        _simulate(batched)
        for cid in ids:
            meta, i_c = model._component_to_meta[cid]
            got = meta.output["inputSignal"].history()[:, 0, i_c].detach().clone()
            torch.testing.assert_close(got, expected[cid], rtol=1e-7, atol=1e-9, msg=cid)
        self.assertGreater(float((expected["cits0"] - expected["cits1"]).abs().max()), 1e-2)  # their setpoints act


class TestRewire(unittest.TestCase):
    def _wired(self, model_id, u, temperature, co2, setpoints):
        """The command ``u``, the room's temperature and CO2 on the feedback
        bus, ``setpoints`` (``{id: series}``) on the setpoint bus."""
        model = tb.Model(id=model_id)
        cits = ControllerIdentificationPISystem(id="cits")
        model.add_connection(_sensor("zone_t", temperature), cits, "measuredValue", "sensorValue", input_port_index=0)
        model.add_connection(_sensor("zone_co2", co2), cits, "measuredValue", "sensorValue", input_port_index=1)
        for j, (sid, values) in enumerate(setpoints.items()):
            model.add_connection(_sensor(sid, values), cits, "measuredValue", "setpointValue", input_port_index=j)
        model.add_connection(_sensor("flow_sp", np.ones(N)), cits, "measuredValue", "onOffSignal", input_port_index=0)
        model.add_connection(cits, _sensor("cmd", u), "inputSignal", "measuredValue", output_port_index=0)
        return model, cits

    def _wired_sensors(self, model_id, u, sensors, setpoints):
        """The command ``u``; ``sensors`` and ``setpoints`` (``{id: series}``)
        on the feedback and the setpoint bus."""
        model = tb.Model(id=model_id)
        cits = ControllerIdentificationPISystem(id="cits")
        for j, (sid, values) in enumerate(sensors.items()):
            model.add_connection(_sensor(sid, values), cits, "measuredValue", "sensorValue", input_port_index=j)
        for j, (sid, values) in enumerate(setpoints.items()):
            model.add_connection(_sensor(sid, values), cits, "measuredValue", "setpointValue", input_port_index=j)
        model.add_connection(_sensor("flow_sp", np.ones(N)), cits, "measuredValue", "onOffSignal", input_port_index=0)
        model.add_connection(cits, _sensor("cmd", u), "inputSignal", "measuredValue", output_port_index=0)
        return model, cits

    def _rewire(self, model, mode="train"):
        model.rewire(start_time=[START], end_time=[START + datetime.timedelta(days=DAYS)], step_size=int(STEP), mode=mode)
        return model.simulation_model.rewire_reports["cits"]

    def test_the_valve_is_paired_with_the_setpoint_it_follows(self):
        temperature, setback, flat = _room()
        _, co2 = _ventilated_room()
        u = pi_outputs(setback - temperature, np.array([0.4]), np.array([3600.0]), STEP)[:, 0]
        model, cits = self._wired("test_cits_loops_valve", u, temperature, co2, {"sp_flat": flat, "sp_setback": setback})
        report = self._rewire(model)
        self.assertEqual(report.winner[1], "sp_setback")
        self.assertTrue(report.is_reverse)
        self.assertIsNone(report.max_loop)  # the CO2 explains nothing more
        self.assertEqual(cits.n_sensors, 1)
        self.assertNotIn("max", [entry["type"] for entry in cits.candidate_structure])

    def test_a_room_with_two_sensors_is_controlled_on_their_mix(self):
        """The valve acts on the mean of the room's two temperature sensors:
        the rewire keeps both, weighted half and half, and adds no loop on
        the second sensor."""
        temperature, setback, flat = _room()
        _, co2 = _ventilated_room()
        second = temperature + 0.6 + 0.3 * np.sin(2 * np.pi * T / 37)
        u = pi_outputs(setback - (temperature + second) / 2, np.array([0.4]), np.array([3600.0]), STEP)[:, 0]
        model, cits = self._wired_sensors(
            "test_cits_loops_mix", u, {"zone_t1": temperature, "zone_t2": second, "zone_co2": co2}, {"sp_flat": flat, "sp_setback": setback}
        )
        report = self._rewire(model)
        self.assertEqual(report.feedback, [("zone_t1", 0.5), ("zone_t2", 0.5)])
        self.assertEqual(report.winner[1], "sp_setback")
        self.assertIsNone(report.max_loop)
        self.assertEqual(cits.n_sensors, 2)
        torch.testing.assert_close(cits.beta_0.get().reshape(-1), torch.tensor([0.5, 0.5], dtype=torch.float64))

    def test_a_room_far_from_its_setpoint_keeps_its_pair(self):
        """A room 2.5 K below the heating setpoint its valve follows (the
        valve open in proportion): the pair is ranked by the fit, not
        removed for sitting more than 1.5 K from the setpoint, so a closer
        setpoint the valve does not follow does not win."""
        _, setback, _ = _room()
        _, co2 = _ventilated_room()
        temperature = setback - 2.5 + 0.8 * np.sin(2 * np.pi * T / 144) + 0.2 * np.sin(2 * np.pi * T / 23)
        self.assertGreater(float(np.min(np.abs(setback - temperature))), 1.5)  # on any samples
        u = pi_outputs(setback - temperature, np.array([0.25]), np.array([np.inf]), STEP)[:, 0]
        self.assertGreater(float(u.std()), 0.1)  # the valve modulates
        model, cits = self._wired_sensors(
            "test_cits_loops_far", u, {"zone_t": temperature, "zone_co2": co2}, {"sp_low": np.full(N, 19.0), "sp_setback": setback}
        )
        report = self._rewire(model)
        self.assertEqual(report.winner[1], "sp_setback")
        self.assertEqual(report.feedback, [("zone_t", 1.0)])

    def test_the_damper_gets_its_co2_loop(self):
        temperature, co2 = _ventilated_room()
        base = pi_outputs(temperature - COOLING, np.array([0.3]), np.array([7200.0]), STEP)[:, 0]
        u = np.maximum(base, np.clip((co2 - 800.0) / 200.0, 0.0, 1.0))
        model, cits = self._wired("test_cits_loops_damper", u, temperature, co2, {"sp_cool": np.full(N, COOLING), "sp_heat": np.full(N, 20.0)})
        report = self._rewire(model)
        self.assertIsNotNone(report.max_loop)
        self.assertEqual(report.max_loop["sensor"], "zone_co2")
        self.assertFalse(report.max_loop["is_reverse"])
        self.assertAlmostEqual(report.max_loop["setpoint"], 800.0, delta=40.0)
        self.assertEqual([entry["type"] for entry in cits.candidate_structure], ["setpoint", "max"])
        self.assertEqual(cits.n_sensors, 2)
        # the temperature loop reads the temperature, the max loop the CO2
        sensors = {conn.connects_system.id: cp.input_port_index[conn] for cp in cits.connects_at if cp.input_port == "sensorValue" for conn in cp.connects_system_through}
        self.assertEqual(sensors, {"zone_t": 0, "zone_co2": 1})
        torch.testing.assert_close(cits.beta_0.get().reshape(-1), torch.tensor([1.0, 0.0], dtype=torch.float64))
        torch.testing.assert_close(cits.beta_0_1.get().reshape(-1), torch.tensor([0.0, 1.0], dtype=torch.float64))
        self.assertAlmostEqual(float(cits.setpoint_0_1.get().reshape(-1)[0]), report.max_loop["setpoint"], places=6)
        # in the other modes the structure decided in train is kept
        self.assertEqual(self._rewire(model, mode="simulate").reason, "structure_decided_with_a_max_loop")
        self.assertEqual(cits.n_sensors, 2)


if __name__ == "__main__":
    unittest.main()
