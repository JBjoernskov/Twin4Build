"""A stateless composable producer that executes AFTER its consumer (the cut
edge of a cycle) must be part of the functional map's influence cone.

Regression: ``OneStepComposer._influence_cone`` only followed producers that
execute *earlier* than the consumer, so a stateless component on the cut edge
-- e.g. an AHU ordered after the zones it feeds, driven by controllers that
read those zones -- was frozen into a captured constant, and the composer then
(correctly) refused the model because that constant depends on theta.  The
consumer reads the previous step's value there (Gauss-Seidel lag), which is
exactly the feedback lag variable the composer already threads for stateful
producers.
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

twin4build._IS_TESTING = True

START = datetime.datetime(2024, 3, 4, tzinfo=ZoneInfo("Europe/Copenhagen"))
STEP = 600


def build_loop():
    """PID (stateful) -> gate (stateless plant) -> PID feedback."""
    model = tb.Model(id="test_later_producer")
    pid = tb.PIDControllerSystem(kp=1.0, Ti=1800.0, Td=0.0, is_reverse=False, id="pid")
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
    model.add_connection(setpoint, pid, "scheduleValue", "setpointValue")
    model.add_connection(pid, plant, "inputSignal", "inputSignal")
    model.add_connection(plant, pid, "outputSignal", "actualValue")
    sensor = tb.SensorSystem(id="plant_sensor")
    model.add_connection(plant, sensor, "outputSignal", "measuredValue")
    model.load(draw_semantic_model=False, draw_simulation_model=False)
    return model, pid, plant, sensor


class TestLaterProducerInCone(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.pid, cls.plant, cls.sensor = build_loop()
        end = START + datetime.timedelta(hours=24)
        sim = tb.Simulator(cls.model, execution_mode="functional")
        sim.simulate(start_time=[START], end_time=[end], step_size=STEP, show_progress_bar=False)
        y = cls.plant.output["outputSignal"].history().detach().flatten().numpy()
        index = pd.date_range(start=START, periods=len(y), freq=f"{STEP}s")
        cls.sensor.df = pd.DataFrame({"value": y + 0.01 * np.random.default_rng(0).standard_normal(len(y))}, index=index)
        cls.estimator = tb.Estimator(tb.Simulator(cls.model, execution_mode="functional"))
        cls.estimator.estimate(
            parameters=[(cls.pid, "kp", 1.0, 0.1, 10.0)],
            measurements=[(cls.sensor, 0.05)],
            start_time=[START],
            end_time=[end],
            step_size=STEP,
            n_warmup=3,
            method=("scipy", "SLSQP", "ad"),
            options={"maxiter": 1},
        )

    def test_plant_is_composed_whatever_the_cut(self):
        fast = self.estimator._functional_objective
        self.assertIsNotNone(fast, "loop through a stateless later producer did not compose")
        composer = fast.composer
        self.assertIn(self.plant.id, {c.id for c in composer.cone})
        # Whichever edge the cycle cut removed, the plant's output (a
        # composable, theta-dependent signal) must not have been frozen:
        # captured keys are (consumer, port), so only the schedule-fed
        # setpoint may appear for the PID.
        self.assertNotIn((self.pid.id, "actualValue"), composer._exogenous_keys)
        self.assertNotIn((self.plant.id, "inputSignal"), composer._exogenous_keys)
        if composer.pos[self.plant.id] > composer.pos[self.pid.id]:
            # The cut fell on plant -> pid: the plant is a LATER producer and
            # must be threaded as a feedback lag variable.
            self.assertEqual(len(composer._feedback_keys), 1)

    def test_value_and_gradient_parity(self):
        est = self.estimator
        fast = est._functional_objective

        def ev(theta, use_fast):
            est._functional_objective = fast if use_fast else None
            est._mse_scaled = 1.0
            try:
                z = torch.tensor(theta, dtype=torch.float64, requires_grad=True)
                f = est._obj(z, "scalar")
                (g,) = torch.autograd.grad(f, z)
                return float(f.detach()), g.numpy()
            finally:
                est._functional_objective = fast
                est._mse_scaled = None

        x0 = np.asarray(est._x0_norm, dtype=np.float64)
        for theta in (x0, np.clip(x0 + 0.15, est._lb_norm, est._ub_norm)):
            f_slow, g_slow = ev(theta, False)
            f_fast, g_fast = ev(theta, True)
            self.assertLess(abs(f_fast - f_slow) / max(1e-12, abs(f_slow)), 1e-6)
            gscale = max(1e-12, float(np.abs(g_slow).max()))
            self.assertLess(float(np.abs(g_fast - g_slow).max()) / gscale, 1e-5)


if __name__ == "__main__":
    unittest.main()
