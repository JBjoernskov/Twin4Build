"""Identified controllers batch (#233): ``Model.batch_components()`` builds one
meta controller per structure, with its candidates and gates, stacks the
instances' parameters and their per-loop ``onOffSignal`` normalisation
bounds onto it, and the batched model commands what the instances command.
Before, the meta was built without its candidates and gates and every
identified controller stayed a separate component."""

# Standard library imports
import datetime
import math
import unittest

# Third party imports
import pandas as pd
import torch
from dateutil import tz

# Local application imports
import twin4build as tb
from twin4build.systems.controller.controller_identification.controller_identification_pi_system import (
    ControllerIdentificationPISystem,
)
from twin4build.systems.sensor.sensor_system import SensorSystem

tb._IS_TESTING = True

START = datetime.datetime(2024, 1, 8, tzinfo=tz.UTC)
STEP = 600
HOURS = 6
N = HOURS * 3600 // STEP + 1


def _series(sid, values):
    """An in-memory data sensor: one value column on a time index."""
    times = pd.DatetimeIndex(pd.date_range(START, periods=N, freq=f"{STEP}s"), name="time")
    return SensorSystem(id=sid, df=pd.DataFrame({"value": [float(v) for v in values]}, index=times), use_df=True)


def _controller(k, **kwargs):
    """A built PI controller with gains, gate and normalisation bounds of its own."""
    cits = ControllerIdentificationPISystem(id=f"cits{k}", n_sensors=1, n_setpoints=1, n_on_off_signals=1, **kwargs)
    # the selection weights as the forward test sets them: the PI candidate reaches the command
    for name in ("alpha_0", "beta_0", "gamma_0", "gamma_gate_0"):
        p = getattr(cits, name)
        p.set(torch.full(p.get().shape, 0.6, dtype=torch.float64), normalized=False)
    cits.alpha_gate_0.set(torch.tensor(0.7, dtype=torch.float64), normalized=False)
    cits.default_output_0.set(torch.tensor(0.3, dtype=torch.float64), normalized=False)
    cand = cits.candidate_0_0
    cand.kp.set(torch.tensor(0.2 + 0.3 * k, dtype=torch.float64), normalized=False)
    cand.Ti.set(torch.tensor(900.0 + 600.0 * k, dtype=torch.float64), normalized=False)
    cits.gate_0.threshold.set(torch.tensor(0.2 + 0.1 * k, dtype=torch.float64), normalized=False)
    cits.on_off_signal_norm_min = [18.0 - k]
    cits.on_off_signal_norm_max = [24.0 + k]
    return cits


def _wire(model, cits, k):
    """Room temperature, setpoint and the gate's signal from data; the command to a sensor.  The temperature
    swings around the setpoint so the PI law acts unsaturated (direct action: above the setpoint opens) and the
    loops' different gains give different commands."""
    setpoint = [21.0 + 0.5 * k] * N
    temperature = [setpoint[i] + 0.3 * math.sin(2.0 * math.pi * i / 12.0) for i in range(N)]  # zero mean: no wind-up
    on_off = [19.0 + (6.0 if 6 <= i < 30 else 0.0) for i in range(N)]
    model.add_connection(_series(f"t{k}", temperature), cits, "measuredValue", "sensorValue", input_port_index=0)
    model.add_connection(_series(f"sp{k}", setpoint), cits, "measuredValue", "setpointValue", input_port_index=0)
    model.add_connection(_series(f"oo{k}", on_off), cits, "measuredValue", "onOffSignal", input_port_index=0)
    model.add_connection(cits, _series(f"cmd{k}", [0.0] * N), "inputSignal", "measuredValue", output_port_index=0)


def _model(controllers):
    model = tb.Model(id="test_cits_batching")
    for k, cits in enumerate(controllers):
        _wire(model, cits, k)
    model.load(draw_semantic_model=False, draw_simulation_model=False)
    return model


def _simulate(model, **kwargs):
    simulator = tb.Simulator(model, **kwargs)
    simulator.simulate(start_time=START, end_time=START + datetime.timedelta(hours=HOURS), step_size=STEP, show_progress_bar=False)
    return simulator


def _command(component, i_c=0):
    """One instance's command history, ``(n_t, n_actuators)`` (the port's history is ``(n_t, n_s, n_c, n_v)``)."""
    return component.output["inputSignal"].history()[:, 0, i_c].detach().clone()


def _commands(model, ids):
    """Each controller's command, read through the meta component it runs in."""
    return {cid: _command(*model._component_to_meta[cid]) for cid in ids}


class TestBatchingIdentifiedControllers(unittest.TestCase):
    def test_one_meta_steps_like_its_instances(self):
        ids = [f"cits{k}" for k in range(3)]
        reference = _model([_controller(k) for k in range(3)])
        _simulate(reference)
        expected = {cid: _command(reference.components[cid]) for cid in ids}

        model = _model([_controller(k) for k in range(3)])
        batched = model.batch_components()
        metas = [c for c in batched.components.values() if isinstance(c, ControllerIdentificationPISystem)]
        self.assertEqual(len(metas), 1)
        meta = metas[0]
        self.assertEqual(meta._n_c_batched, 3)
        # the per-loop bounds and gains, one row per instance
        torch.testing.assert_close(meta.on_off_signal_norm_min, torch.tensor([[18.0], [17.0], [16.0]], dtype=torch.float64))
        torch.testing.assert_close(meta.on_off_signal_norm_max, torch.tensor([[24.0], [25.0], [26.0]], dtype=torch.float64))
        torch.testing.assert_close(meta.candidate_0_0.kp.get().reshape(-1), torch.tensor([0.2, 0.5, 0.8], dtype=torch.float64))

        batched.load(draw_semantic_model=False, draw_simulation_model=False)
        _simulate(batched)
        got = _commands(model, ids)
        for cid in ids:
            torch.testing.assert_close(got[cid], expected[cid], rtol=1e-7, atol=1e-9, msg=cid)
        # the loops differ (their gains act), so the comparison is not trivially equal
        self.assertGreater(float((expected["cits0"] - expected["cits2"]).abs().max()), 1e-2)

    def test_the_functional_rollout_of_the_meta_steps_like_its_instances(self):
        """The functional rollout (what the estimator rolls out) of the batched
        controllers commands what the instances command step by step."""
        ids = [f"cits{k}" for k in range(3)]
        reference = _model([_controller(k) for k in range(3)])
        _simulate(reference)
        expected = {cid: _command(reference.components[cid]) for cid in ids}
        model = _model([_controller(k) for k in range(3)])
        batched = model.batch_components()
        batched.load(draw_semantic_model=False, draw_simulation_model=False)
        _simulate(batched, execution_mode="functional", execution_backend="eager")
        got = _commands(model, ids)
        for cid in ids:
            torch.testing.assert_close(got[cid], expected[cid], rtol=1e-7, atol=1e-9, msg=cid)

    def test_the_meta_offers_one_start_and_bounds_per_loop(self):
        """``parameters="auto"`` on the batched model: each estimable entry of
        the meta holds one start per loop and the rewire's per-loop bounds
        (before, ``.item()`` on the stacked value raised and the estimator
        skipped the meta, so nothing was estimated)."""
        from twin4build.systems.controller.controller_identification.pi_loop_rewire import _set_param

        controllers = [_controller(k) for k in range(3)]
        for k, cits in enumerate(controllers):
            _set_param(cits.candidate_0_0, "kp", 0.2 + 0.3 * k, 0.05 * (k + 1), 2.0 + k)
        alone = {attr: (x0, lb, ub) for _, attr, x0, lb, ub in controllers[0].get_estimable_parameters()}
        self.assertIsInstance(alone["candidate_0_0.kp"][0], float)
        self.assertAlmostEqual(alone["candidate_0_0.kp"][1], 0.05)

        batched = _model(controllers).batch_components()
        meta = next(c for c in batched.components.values() if isinstance(c, ControllerIdentificationPISystem))
        entries = {attr: (comp, x0, lb, ub) for comp, attr, x0, lb, ub in meta.get_estimable_parameters()}
        self.assertEqual(set(entries), set(alone))
        comp, x0, lb, ub = entries["candidate_0_0.kp"]
        self.assertIs(comp, meta)
        torch.testing.assert_close(torch.as_tensor(x0), torch.tensor([0.2, 0.5, 0.8], dtype=torch.float64))
        torch.testing.assert_close(torch.as_tensor(lb), torch.tensor([0.05, 0.10, 0.15], dtype=torch.float64))
        torch.testing.assert_close(torch.as_tensor(ub), torch.tensor([2.0, 3.0, 4.0], dtype=torch.float64))
        _, x0, _, _ = entries["gate_0.threshold"]
        torch.testing.assert_close(torch.as_tensor(x0), torch.tensor([0.2, 0.3, 0.4], dtype=torch.float64))
        _, x0, _, _ = entries["gamma_gate_0"]
        self.assertEqual(len(x0), 3)  # one slot per loop

    def test_another_structure_or_playback_batches_apart(self):
        controllers = [_controller(0), _controller(1), _controller(2, playback=True)]
        controllers.append(
            ControllerIdentificationPISystem(id="cits3", n_sensors=1, n_setpoints=1, n_on_off_signals=1, n_actuators=1,
                                             setpoint_controllers=[tb.PIDControllerSystem, tb.PIDControllerSystem],
                                             setpoint_controller_kwargs=[{"kp": 1.0, "Ti": 1800.0, "Td": 0.0}] * 2)
        )
        signatures = {c.id: c._batch_init_kwargs for c in controllers}
        self.assertEqual(signatures["cits0"], signatures["cits1"])
        self.assertNotEqual(signatures["cits0"], signatures["cits2"])  # playback
        self.assertNotEqual(signatures["cits0"]["candidate_structure"], signatures["cits3"]["candidate_structure"])
        model = tb.Model(id="test_cits_batching_apart")
        for k, cits in enumerate(controllers):
            _wire(model, cits, k)
        model.load(draw_semantic_model=False, draw_simulation_model=False)
        model.batch_components()
        groups = {}
        for cid in ("cits0", "cits1", "cits2", "cits3"):
            meta, _ = model._component_to_meta[cid]
            groups.setdefault(id(meta), []).append(cid)
        self.assertIn(["cits0", "cits1"], list(groups.values()))
        self.assertEqual(len(groups), 3)


if __name__ == "__main__":
    unittest.main()
