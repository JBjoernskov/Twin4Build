"""``ControllerIdentificationSystem`` is composable: ``do_step`` delegates to a
pure ``forward`` (single source of truth), so a closed loop that runs through
a CITS controller can be threaded by the functional-map engine
(``Simulator(execution_mode="functional")``, fast single-shooting, batched GPU
shooting).

Two guards:

* ``do_step`` reproduces the pre-``forward`` algorithm exactly (weighted
  signals, candidate ``do_step``s, alpha blending, on/off gate) -- checked
  against a verbatim re-implementation of that algorithm on an identical
  twin instance, for the PI subclass and for the generic class with a
  cascade candidate;
* a closed-loop model (CITS -> plant -> CITS feedback) composes, and the fast
  single-shooting objective matches the object-graph objective in value and
  gradient with a candidate gain as theta.
"""

# Standard library imports
import datetime
import unittest
from zoneinfo import ZoneInfo

# Third party imports
import numpy as np
import pandas as pd
import torch

# Local application imports
import twin4build
import twin4build as tb
import twin4build.utils.types as tps
from twin4build.systems.controller.controller_identification.controller_identification_pi_system import (
    ControllerIdentificationPISystem,
)
from twin4build.systems.controller.controller_identification.controller_identification_system import (
    ControllerIdentificationSystem,
)

twin4build._IS_TESTING = True

TZ = ZoneInfo("Europe/Copenhagen")
START = datetime.datetime(2024, 3, 4, tzinfo=TZ)
STEP = 600


def _legacy_step(cits, step_index):
    """The pre-``forward`` ``do_step`` algorithm, verbatim, driving the
    candidates through their own ``do_step``."""
    n_s = cits.input["sensorValue"].n_s
    n_c = cits.input["sensorValue"].n_c
    actuator_outputs = torch.zeros(n_s, n_c, cits.n_actuators, dtype=tps.float_dtype())
    for a in range(cits.n_actuators):
        signals = cits._compute_weighted_signals(a)
        weighted_setpoint, weighted_feedback = signals[0], signals[1]
        weighted_feedback_b = signals[2] if len(signals) > 2 else None
        candidate_outputs = []
        for c in range(cits.n_candidates):
            ctrl = cits._get_candidate(a, c)
            if cits._candidate_types[c] == cits.CTRL_CASCADE:
                ctrl.input["setpointValue_a"].set(weighted_setpoint, step_index)
                ctrl.input["actualValue_a"].set(weighted_feedback, step_index)
                ctrl.input["actualValue_b"].set(weighted_feedback_b, step_index)
            else:
                ctrl.input["setpointValue"].set(weighted_setpoint, step_index)
                ctrl.input["actualValue"].set(weighted_feedback, step_index)
            ctrl.do_step(step_index * STEP, START, STEP, step_index)
            candidate_outputs.append(ctrl.output["inputSignal"].get())
        candidate_outputs = torch.stack(candidate_outputs, dim=0)
        shape = candidate_outputs.shape[1:]
        alpha = cits._get_alpha_vector(a)
        alpha_norm = alpha / (torch.sum(alpha) + 1e-8)
        combined = torch.einsum(
            "c,cb->b", alpha_norm, candidate_outputs.reshape(cits.n_candidates, -1)
        ).reshape(shape)
        oo = cits.input["onOffSignal"].get()
        oo_range = (cits.on_off_signal_norm_max - cits.on_off_signal_norm_min).clamp(min=1e-6)
        oo_norm = (oo - cits.on_off_signal_norm_min) / oo_range
        gamma_gate = cits._get_gamma_gate_vector(a)
        gamma_gate_norm = gamma_gate / (torch.sum(gamma_gate) + 1e-8)
        gate_signal = cits._get_gate(a).compute_gate(torch.sum(gamma_gate_norm * oo_norm, dim=-1))
        alpha_g = cits._get_alpha_gate(a)
        gate = (1 - alpha_g) + alpha_g * gate_signal
        actuator_outputs[..., a] = gate * combined + (1 - gate) * cits._get_default_output(a)
    cits.output["inputSignal"].set(actuator_outputs, step_index)


def _randomise_weights(cits, rng):
    """Non-trivial selection weights / gains so the blend is exercised."""
    for a in range(cits.n_actuators):
        for name in ("alpha", "beta", "gamma", "gamma_gate"):
            p = getattr(cits, f"{name}_{a}")
            p.set(torch.tensor(rng.uniform(0.2, 0.9, size=p.get().shape), dtype=tps.float_dtype()), normalized=False)
        getattr(cits, f"alpha_gate_{a}").set(torch.tensor(0.7, dtype=tps.float_dtype()), normalized=False)
        getattr(cits, f"default_output_{a}").set(torch.tensor(0.3, dtype=tps.float_dtype()), normalized=False)
        for c in range(cits.n_candidates):
            ctrl = cits._get_candidate(a, c)
            for attr, lo, hi in (("kp", 0.2, 2.0), ("Ti", 600.0, 3600.0), ("kp_a", 0.1, 1.0), ("kp_b", 0.1, 1.0)):
                p = getattr(ctrl, attr, None)
                if p is not None:
                    p.set(torch.tensor(rng.uniform(lo, hi), dtype=tps.float_dtype()), normalized=False)


class TestCitsForwardMatchesLegacyDoStep(unittest.TestCase):
    N_STEPS = 24

    def _run_pair(self, factory, seed=0):
        rng = np.random.default_rng(seed)
        new, legacy = factory(), factory()
        end = START + datetime.timedelta(seconds=STEP * self.N_STEPS)
        for cits in (new, legacy):
            cits.initialize(start_time=[START], end_time=[end], step_size=[STEP])
            _randomise_weights(cits, np.random.default_rng(seed))
        n_sens, n_sp, n_oo = new.n_sensors, new.n_setpoints, new.n_on_off_signals
        outs_new, outs_legacy = [], []
        for k in range(self.N_STEPS):
            sens = torch.tensor(rng.uniform(18, 26, size=(1, 1, n_sens)), dtype=tps.float_dtype())
            sp = torch.tensor(rng.uniform(20, 24, size=(1, 1, n_sp)), dtype=tps.float_dtype())
            oo = torch.tensor(rng.uniform(0, 1, size=(1, 1, n_oo)), dtype=tps.float_dtype())
            for cits, outs, stepper in ((new, outs_new, None), (legacy, outs_legacy, _legacy_step)):
                cits.input["sensorValue"].set(sens.clone(), k)
                cits.input["setpointValue"].set(sp.clone(), k)
                cits.input["onOffSignal"].set(oo.clone(), k)
                if stepper is None:
                    cits.do_step(k * STEP, START, STEP, k)
                else:
                    stepper(cits, k)
                outs.append(cits.output["inputSignal"].get().detach().clone())
        a = torch.stack(outs_new).flatten()
        b = torch.stack(outs_legacy).flatten()
        self.assertGreater(float(a.std()), 1e-6, "controller output should move")
        self.assertTrue(torch.allclose(a, b, atol=1e-12, rtol=1e-12), f"max |diff| = {float((a - b).abs().max()):.3e}")
        # The candidate memories advanced identically through both paths.
        self.assertTrue(torch.allclose(new.get_state(), legacy.get_state(), atol=1e-12))

    def test_pi_subclass(self):
        self._run_pair(
            lambda: ControllerIdentificationPISystem(
                n_sensors=2, n_setpoints=2, n_on_off_signals=1, n_actuators=1, id="cits_pi"
            )
        )

    def test_generic_with_cascade_candidate(self):
        """Default candidate set: two PIDs and a cascade -- exercises the
        cascade input routing and the multi-candidate state slicing."""
        self._run_pair(
            lambda: ControllerIdentificationSystem(
                n_sensors=2, n_setpoints=1, n_on_off_signals=2, n_actuators=1, id="cits_generic"
            )
        )

    def test_param_names_cover_every_tunable(self):
        cits = ControllerIdentificationPISystem(
            n_sensors=1, n_setpoints=1, n_on_off_signals=1, n_actuators=1, id="cits"
        )
        names = set(cits.PARAM_NAMES)
        for expected in ("alpha_0", "beta_0", "gamma_0", "gamma_gate_0", "alpha_gate_0",
                         "default_output_0", "gate_0.threshold", "candidate_0_0.kp", "candidate_0_0.Ti"):
            self.assertIn(expected, names)


def build_closed_loop_model():
    """CITS -> gate (the 'plant') -> CITS feedback; theta on the PI gain."""
    model = tb.Model(id="test_cits_closed_loop")
    cits = ControllerIdentificationPISystem(
        n_sensors=1, n_setpoints=1, n_on_off_signals=1, n_actuators=1, id="cits"
    )
    plant = tb.SigmoidGate(threshold=0.4, steepness=6.0, id="plant")
    setpoint = tb.ScheduleSystem(
        weekday_ruleset={
            "ruleset_default_value": 0.3,
            "ruleset_start_minute": [0],
            "ruleset_end_minute": [0],
            "ruleset_start_hour": [8],
            "ruleset_end_hour": [16],
            "ruleset_value": [0.7],
        },
        id="setpoint",
    )
    on_off = tb.ScheduleSystem(weekday_ruleset={"ruleset_default_value": 1.0}, id="on_off")
    model.add_connection(setpoint, cits, "scheduleValue", "setpointValue", input_port_index=0)
    model.add_connection(on_off, cits, "scheduleValue", "onOffSignal", input_port_index=0)
    model.add_connection(cits, plant, "inputSignal", "inputSignal", output_port_index=0)
    model.add_connection(plant, cits, "outputSignal", "sensorValue", input_port_index=0)
    sensor = tb.SensorSystem(id="plant_sensor")
    model.add_connection(plant, sensor, "outputSignal", "measuredValue")
    model.load(draw_semantic_model=False, draw_simulation_model=False)
    return model, cits, plant, sensor


class TestCitsClosedLoopComposes(unittest.TestCase):
    TOL_VALUE = 1e-6
    TOL_GRAD = 1e-5

    @classmethod
    def setUpClass(cls):
        cls.model, cls.cits, cls.plant, cls.sensor = build_closed_loop_model()
        cls.end = START + datetime.timedelta(hours=24)
        sim = tb.Simulator(cls.model, execution_mode="functional")
        sim.simulate(start_time=cls.start_list(), end_time=[cls.end], step_size=STEP, show_progress_bar=False)
        y = cls.plant.output["outputSignal"].history().detach().flatten().numpy()
        index = pd.date_range(start=START, periods=len(y), freq=f"{STEP}s")
        rng = np.random.default_rng(0)
        cls.sensor.df = pd.DataFrame({"value": y + 0.01 * rng.standard_normal(len(y))}, index=index)
        cls.estimator = tb.Estimator(tb.Simulator(cls.model, execution_mode="functional"))
        cls.estimator.estimate(
            parameters=[(cls.cits, "candidate_0_0.kp", 1.0, 0.1, 10.0), (cls.cits, "candidate_0_0.Ti", 1800.0, 300.0, 7200.0)],
            measurements=[(cls.sensor, 0.05)],
            start_time=cls.start_list(),
            end_time=[cls.end],
            step_size=STEP,
            n_warmup=3,
            method=("scipy", "SLSQP", "ad"),
            options={"maxiter": 1},
        )

    @staticmethod
    def start_list():
        return [START]

    def _eval(self, theta_np, use_fast):
        est = self.estimator
        fast = est._functional_objective
        est._functional_objective = fast if use_fast else None
        est._mse_scaled = 1.0
        try:
            z = torch.tensor(theta_np, dtype=torch.float64, requires_grad=True)
            f = est._obj(z, "scalar")
            (g,) = torch.autograd.grad(f, z)
            return float(f.detach()), g.numpy()
        finally:
            est._functional_objective = fast
            est._mse_scaled = None

    def test_fast_objective_built_and_controller_in_cone(self):
        fast = self.estimator._functional_objective
        self.assertIsNotNone(fast, "closed loop through the CITS did not compose")
        cone_ids = {c.id for c in fast.composer.cone}
        self.assertIn(self.cits.id, cone_ids)
        self.assertIn(self.plant.id, cone_ids)

    def test_value_and_gradient_parity(self):
        est = self.estimator
        x0 = np.asarray(est._x0_norm, dtype=np.float64)
        lbn = np.asarray(est._lb_norm, dtype=np.float64)
        ubn = np.asarray(est._ub_norm, dtype=np.float64)
        rng = np.random.default_rng(1)
        for i, theta in enumerate([x0, np.clip(x0 + 0.2 * (rng.random(x0.shape) - 0.5), lbn, ubn)]):
            f_slow, g_slow = self._eval(theta, use_fast=False)
            f_fast, g_fast = self._eval(theta, use_fast=True)
            self.assertLess(abs(f_fast - f_slow) / max(1e-12, abs(f_slow)), self.TOL_VALUE, f"theta[{i}] value")
            gscale = max(1e-12, float(np.abs(g_slow).max()))
            self.assertLess(float(np.abs(g_fast - g_slow).max()) / gscale, self.TOL_GRAD, f"theta[{i}] gradient")
            self.assertGreater(gscale, 1e-9, "theta must influence the objective")


if __name__ == "__main__":
    unittest.main()
