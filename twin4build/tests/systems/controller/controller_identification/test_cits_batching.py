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


def _controller(k, slots=1, **kwargs):
    """A built PI controller with gains, gate and normalisation bounds of its own; with several ``onOffSignal``
    slots the gate weights slot ``k % slots`` most."""
    cits = ControllerIdentificationPISystem(id=f"cits{k}", n_sensors=1, n_setpoints=1, n_on_off_signals=slots, **kwargs)
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
    cits.on_off_signal_norm_min = [18.0 - k - s for s in range(slots)]
    cits.on_off_signal_norm_max = [24.0 + k + s for s in range(slots)]
    if slots > 1:
        weights = [0.9 if s == k % slots else 0.1 for s in range(slots)]
        cits.gamma_gate_0.set(torch.tensor(weights, dtype=torch.float64), normalized=False)
    return cits


def _wire(model, cits, k, slots=1):
    """Room temperature, setpoint and the gate's signal from data; the command to a sensor.  The temperature
    swings around the setpoint so the PI law acts unsaturated (direct action: above the setpoint opens) and the
    loops' different gains give different commands."""
    setpoint = [21.0 + 0.5 * k] * N
    temperature = [setpoint[i] + 0.3 * math.sin(2.0 * math.pi * i / 12.0) for i in range(N)]  # zero mean: no wind-up
    model.add_connection(_series(f"t{k}", temperature), cits, "measuredValue", "sensorValue", input_port_index=0)
    model.add_connection(_series(f"sp{k}", setpoint), cits, "measuredValue", "setpointValue", input_port_index=0)
    for s in range(slots):  # each slot switches on at its own time
        on_off = [19.0 + (6.0 if 6 + 8 * s <= i < 30 + 4 * s else 0.0) for i in range(N)]
        model.add_connection(_series(f"oo{k}_{s}", on_off), cits, "measuredValue", "onOffSignal", input_port_index=s)
    model.add_connection(cits, _series(f"cmd{k}", [0.0] * N), "inputSignal", "measuredValue", output_port_index=0)


def _model(controllers, slots=1):
    model = tb.Model(id="test_cits_batching")
    for k, cits in enumerate(controllers):
        _wire(model, cits, k, slots)
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


class TestBlocks(unittest.TestCase):
    def test_flat_slot_weights_keep_one_block_per_loop(self):
        """A batched controller holds each loop's gate-slot weights flat, one
        row per loop: estimated, they belong to their own loop, so two loops
        stay two independent blocks of the estimation problem (they were one,
        and the HTR ring's 317 loops two dense blocks of 1120 and 1570
        parameters under one trust region)."""
        model = _model([_controller(k, slots=2) for k in range(2)], slots=2)
        batched = model.batch_components()
        batched.load(draw_semantic_model=False, draw_simulation_model=False)
        batched.initialize(start_time=[START], end_time=[START + datetime.timedelta(hours=HOURS)], step_size=STEP)
        meta = next(c for c in batched.components.values() if isinstance(c, ControllerIdentificationPISystem))
        self.assertEqual(tuple(meta.gamma_gate_0.get().shape), (4,))  # flat: two loops x two slots
        theta_spec = [(meta, "candidate_0_0.kp", slice(0, 2)), (meta, "gamma_gate_0", slice(2, 6))]
        simulator = tb.Simulator(batched, execution_mode="functional", execution_backend="eager")
        commands = [batched.components[f"cmd{k}"] for k in range(2)]
        _, fm = simulator.build_functional_model(theta_spec=theta_spec, measurements=commands, step_size=STEP)
        fm.prepare_routes(torch.device("cpu"))
        theta_block, column_block, n_blocks = fm.index_coupling()
        self.assertEqual(n_blocks, 2)
        kp0, kp1 = theta_block[0], theta_block[1]
        self.assertNotEqual(kp0, kp1)
        self.assertEqual(list(theta_block[2:]), [kp0, kp0, kp1, kp1])  # each loop's two slot weights with its kp
        self.assertTrue((column_block >= 0).all())


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

    def test_loops_acting_in_opposite_directions_share_a_meta(self):
        """The rewire sets each loop's direction of action
        (``candidate_0_0.is_reverse``): the meta holds one per loop, the sign of
        its error, and every loop commands what it commands alone (before, the
        meta's candidate acted in the default direction for all of them)."""
        ids = [f"cits{k}" for k in range(4)]

        def controllers():
            out = [_controller(k) for k in range(4)]
            for k, cits in enumerate(out):
                cits.candidate_0_0.is_reverse = k % 2 == 1
            return out

        reference = _model(controllers())
        _simulate(reference)
        expected = {cid: _command(reference.components[cid]) for cid in ids}
        model = _model(controllers())
        batched = model.batch_components()
        metas = [c for c in batched.components.values() if isinstance(c, ControllerIdentificationPISystem)]
        self.assertEqual([m._n_c_batched for m in metas], [4])
        batched.load(draw_semantic_model=False, draw_simulation_model=False)
        # a load writes the stacked directions through the config and back
        torch.testing.assert_close(metas[0].candidate_0_0.is_reverse, torch.tensor([False, True, False, True]))
        _simulate(batched)
        got = _commands(model, ids)
        for cid in ids:
            torch.testing.assert_close(got[cid], expected[cid], rtol=1e-7, atol=1e-9, msg=cid)
        # the directions act: a reverse loop commands otherwise than a direct one
        self.assertGreater(float((expected["cits0"] - expected["cits1"]).abs().max()), 1e-2)

    def test_loops_with_several_gate_slots_share_a_meta(self):
        """Loops whose gate reads several ``onOffSignal`` slots: the meta holds
        the slot weights flat, ``(n_c * k,)``, normalises each loop's over its
        own slots, and every loop commands what it commands alone, in the object
        and the functional rollout (before, the weights could not be stacked and
        every such loop stayed a component of its own)."""
        ids = [f"cits{k}" for k in range(3)]
        reference = _model([_controller(k, slots=3) for k in range(3)], slots=3)
        _simulate(reference)
        expected = {cid: _command(reference.components[cid]) for cid in ids}
        self.assertGreater(float((expected["cits0"] - expected["cits1"]).abs().max()), 1e-2)
        for mode in ({}, {"execution_mode": "functional", "execution_backend": "eager"}):
            with self.subTest(mode=mode):
                model = _model([_controller(k, slots=3) for k in range(3)], slots=3)
                batched = model.batch_components()
                metas = [c for c in batched.components.values() if isinstance(c, ControllerIdentificationPISystem)]
                self.assertEqual([m._n_c_batched for m in metas], [3])
                self.assertEqual(tuple(metas[0].gamma_gate_0.get().shape), (9,))
                batched.load(draw_semantic_model=False, draw_simulation_model=False)
                _simulate(batched, **mode)
                got = _commands(model, ids)
                for cid in ids:
                    torch.testing.assert_close(got[cid], expected[cid], rtol=1e-7, atol=1e-9, msg=cid)
        entries = {attr: x0 for _, attr, x0, _, _ in metas[0].get_estimable_parameters()}
        self.assertEqual(len(entries["gamma_gate_0"]), 9)

    def test_loops_in_playback_replay_their_own_commands(self):
        """A batched controller in playback hands over every loop's measured
        command (``replayed_output_history``), each at its instance; before,
        it handed over the first loop's alone, and the functional rollout
        could not take it."""
        ids = [f"cits{k}" for k in range(3)]
        measured = {k: [0.1 * (k + 1) + 0.05 * math.sin(2.0 * math.pi * i / (6.0 + k)) for i in range(N)] for k in range(3)}

        model = tb.Model(id="test_cits_batching_playback")
        for k in range(3):
            cits = _controller(k, playback=True)
            _wire(model, cits, k)
            model.add_connection(_series(f"u{k}", measured[k]), cits, "measuredValue", "actuatorMeasured", input_port_index=0)
        _wire(model, _controller(3), 3)  # a computing loop: the rollout has a state to step
        model.load(draw_semantic_model=False, draw_simulation_model=False)
        batched = model.batch_components()
        metas = [c for c in batched.components.values() if isinstance(c, ControllerIdentificationPISystem)]
        self.assertEqual(sorted(m._n_c_batched for m in metas), [1, 3])
        batched.load(draw_semantic_model=False, draw_simulation_model=False)
        _simulate(batched, execution_mode="functional", execution_backend="eager")
        got = _commands(model, ids)
        for k, cid in enumerate(ids):
            command = got[cid].reshape(-1)  # one value per step; the series also holds the end point
            torch.testing.assert_close(command, torch.tensor(measured[k][: command.numel()], dtype=torch.float64), msg=cid)

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
