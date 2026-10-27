# Standard library imports
import datetime
import math
import unittest

# Third party imports
import numpy as np
import torch
from dateutil import tz

# Local application imports
# Set test flag
import twin4build
from twin4build.systems.fan.fan_system import FanSystem, bernstein_to_polynomial, polynomial_to_bernstein
from twin4build.systems.utils.smooth_saturation import saturation_mode

twin4build._IS_TESTING = True


def _polynomial_power(P_nom, m_nom, c, m):
    x = np.asarray(m, dtype=float) / m_nom
    return P_nom * (c[0] + c[1] * x + c[2] * x**2 + c[3] * x**3)


def _fan_power(fan, m, hard=False):
    """The fan's power at the flows ``m`` from its own parameters."""
    params = {name: getattr(fan, name).get().detach().reshape(-1).to(torch.float64) for name in FanSystem.PARAM_NAMES}
    inputs = {"airFlowRate": torch.as_tensor(m, dtype=torch.float64),
              "inletAirTemperature": torch.full((len(m),), 20.0, dtype=torch.float64)}
    if hard:
        with saturation_mode("hard"):
            _, out = fan.forward(None, inputs, params, 600.0)
    else:
        _, out = fan.forward(None, inputs, params, 600.0)
    return out["Power"].detach().numpy()


class TestFanTorchSystem(unittest.TestCase):
    def setUp(self):
        # the default curve: 1 kW at 1 kg/s, 0.8 linear + 0.2 square, over 0-1 kg/s
        self.fan = FanSystem(id="test_fan", f_total=0.9)

    def test_initialization(self):
        """Test fan system initialization."""
        start_time = [datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=tz.UTC)]
        end_time = [datetime.datetime(2023, 1, 1, 1, 40, 0, tzinfo=tz.UTC)]
        step_size = [600]
        self.fan.initialize(
            start_time=start_time, end_time=end_time, step_size=step_size
        )
        self.assertIsNotNone(self.fan)
        self.assertEqual(self.fan.id, "test_fan")

    def test_do_step(self):
        """Test fan system do_step method."""
        start_time = [datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=tz.UTC)]
        end_time = [datetime.datetime(2023, 1, 1, 1, 40, 0, tzinfo=tz.UTC)]
        step_size = [600]
        self.fan.initialize(
            start_time=start_time, end_time=end_time, step_size=step_size
        )

        # Set inputs
        self.fan.input["airFlowRate"].set(torch.tensor([1.0]), i_t=0)
        self.fan.input["inletAirTemperature"].set(torch.tensor([20.0]), i_t=0)

        # Set required inputs (may vary based on implementation)
        # Execute a time step
        datetime_val = datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=tz.UTC)
        self.fan.do_step(
            second_time=0, date_time=datetime_val, step_size=600, step_index=0
        )

        self.assertIsNotNone(self.fan.output["outletAirTemperature"].get())
        self.assertIsNotNone(self.fan.output["Power"].get())
        self.assertGreater(self.fan.output["outletAirTemperature"].get().item(), 20.0)
        # the top of the default range: 1 kW
        self.assertAlmostEqual(self.fan.output["Power"].get().item(), 1000.0, places=6)

    def test_zero_flow_keeps_outlet_finite(self):
        """Dampers closed / fan off (zero flow) is a normal state: the outlet
        temperature must equal the inlet temperature, not NaN from the
        ``P * f / (m * cp)`` temperature-rise term (regression: an AHU fed
        by closed VAV dampers at night produced a NaN supply temperature)."""
        start_time = [datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=tz.UTC)]
        end_time = [datetime.datetime(2023, 1, 1, 1, 40, 0, tzinfo=tz.UTC)]
        self.fan.initialize(start_time=start_time, end_time=end_time, step_size=[600])
        self.fan.input["airFlowRate"].set(torch.tensor([0.0]), i_t=0)
        self.fan.input["inletAirTemperature"].set(torch.tensor([20.0]), i_t=0)
        datetime_val = datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=tz.UTC)
        self.fan.do_step(second_time=0, date_time=datetime_val, step_size=600, step_index=0)
        outlet = self.fan.output["outletAirTemperature"].get().item()
        self.assertTrue(math.isfinite(outlet))
        self.assertAlmostEqual(outlet, 20.0)
        # zero power, floored: the smooth mode's curve value at zero, exactly zero in hard saturation
        self.assertAlmostEqual(self.fan.output["Power"].get().item(), FanSystem.POWER_CURVE_START / math.sqrt(3.0), places=6)
        with saturation_mode("hard"):
            self.fan.do_step(second_time=0, date_time=datetime_val, step_size=600, step_index=0)
        self.assertEqual(self.fan.output["Power"].get().item(), 0.0)

    def test_power_is_floored_at_zero(self):
        """A curve below zero (here betas a fit could only reach outside the
        class bounds) gives no negative power: zero in hard saturation, at most
        the curve width in the smooth mode, with a finite gradient there -- and
        no cooling of the air."""
        fan = FanSystem(id="dipping_fan", beta0=-500.0, beta1=-433.3, beta2=-366.7, beta3=-300.0, f_total=0.9)
        start_time = [datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=tz.UTC)]
        end_time = [datetime.datetime(2023, 1, 1, 1, 40, 0, tzinfo=tz.UTC)]
        fan.initialize(start_time=start_time, end_time=end_time, step_size=[600])
        params = {name: torch.tensor([float(v)], dtype=torch.float64, requires_grad=name == "beta3") for name, v in
                  (("beta0", -500.0), ("beta1", -433.3), ("beta2", -366.7), ("beta3", -300.0),
                   ("flowRangeLower", 0.0), ("flowRangeUpper", 1.0), ("f_total", 0.9))}
        inputs = {"airFlowRate": torch.tensor([1.0], dtype=torch.float64),
                  "inletAirTemperature": torch.tensor([20.0], dtype=torch.float64)}
        _, out = fan.forward(None, inputs, params, 600.0)
        self.assertGreaterEqual(float(out["Power"]), 0.0)
        self.assertLessEqual(float(out["Power"]), FanSystem.POWER_CURVE_START)
        self.assertGreaterEqual(float(out["outletAirTemperature"]), 20.0)
        (grad,) = torch.autograd.grad(out["Power"].sum(), params["beta3"])
        self.assertTrue(math.isfinite(float(grad)))
        with saturation_mode("hard"):
            _, out = fan.forward(None, inputs, params, 600.0)
        self.assertEqual(float(out["Power"]), 0.0)
        self.assertEqual(float(out["outletAirTemperature"]), 20.0)

    def test_trickle_flow_bounds_the_temperature_rise(self):
        """A fan running against closed dampers moves a trickle of air; the
        ``P * f / (m * cp)`` rise must stay physical (bounded by
        ``MAX_DELTA_T``), not thousands of K (regression: an AHU supply
        temperature of 1450 C at 1e-4 kg/s)."""
        start_time = [datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=tz.UTC)]
        end_time = [datetime.datetime(2023, 1, 1, 1, 40, 0, tzinfo=tz.UTC)]
        self.fan.initialize(start_time=start_time, end_time=end_time, step_size=[600])
        self.fan.input["airFlowRate"].set(torch.tensor([1e-4]), i_t=0)
        self.fan.input["inletAirTemperature"].set(torch.tensor([20.0]), i_t=0)
        datetime_val = datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=tz.UTC)
        self.fan.do_step(second_time=0, date_time=datetime_val, step_size=600, step_index=0)
        outlet = self.fan.output["outletAirTemperature"].get().item()
        self.assertTrue(math.isfinite(outlet))
        self.assertLessEqual(outlet, 20.0 + FanSystem.MAX_DELTA_T + 1e-9)
        self.assertGreater(outlet, 20.0)

    def test_do_step_batch(self):
        """Test fan system do_step method with batch size > 1."""
        fan_batch = FanSystem(id="test_fan_batch", f_total=0.9)

        batch_size = 2

        start_time = [datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=tz.UTC)]
        end_time = [datetime.datetime(2023, 1, 1, 1, 40, 0, tzinfo=tz.UTC)]
        step_size = [600]
        fan_batch.initialize(
            start_time=start_time, end_time=end_time, step_size=step_size
        )

        # Initialize inputs with batch size
        fan_batch.input["airFlowRate"].initialize(n_t=1, n_s=batch_size)
        fan_batch.input["inletAirTemperature"].initialize(n_t=1, n_s=batch_size)

        # Set inputs with batch size 2
        fan_batch.input["airFlowRate"].set(torch.tensor([1.0, 0.8]), i_t=0)
        fan_batch.input["inletAirTemperature"].set(torch.tensor([20.0, 22.0]), i_t=0)

        fan_batch.output["outletAirTemperature"].initialize(n_t=1, n_s=batch_size)
        fan_batch.output["Power"].initialize(n_t=1, n_s=batch_size)

        # Execute a time step
        datetime_val = datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=tz.UTC)
        fan_batch.do_step(
            second_time=0, date_time=datetime_val, step_size=600, step_index=0
        )

        # Check outputs - verify all outputs have consistent batch shape
        outlet_temp = fan_batch.output["outletAirTemperature"].get()
        power = fan_batch.output["Power"].get()

        self.assertIsNotNone(outlet_temp)
        self.assertIsNotNone(power)
        self.assertEqual(
            outlet_temp.shape[0], batch_size
        )  # Output batch matches input batch
        self.assertEqual(power.shape[0], batch_size)  # Output batch matches input batch


class TestBernsteinForm(unittest.TestCase):
    """The power curve in Bernstein form and its map to the polynomial form."""

    def test_round_trip(self):
        """Polynomial -> Bernstein -> polynomial gives the coefficients back,
        for any nominal values and range, and Bernstein -> polynomial ->
        Bernstein the betas."""
        rng = np.random.default_rng(0)
        for _ in range(50):
            P_nom, m_nom = rng.uniform(100, 5000), rng.uniform(0.3, 10)
            lo = rng.uniform(0, 5)
            hi = lo + rng.uniform(0.5, 20)
            c = rng.uniform(-2, 2, 4)
            beta = polynomial_to_bernstein(P_nom, m_nom, c, lo, hi)
            np.testing.assert_allclose(bernstein_to_polynomial(beta, lo, hi, P_nom, m_nom), c, rtol=1e-9, atol=1e-9)
            b = rng.uniform(-1000, 5000, 4)
            np.testing.assert_allclose(
                polynomial_to_bernstein(P_nom, m_nom, bernstein_to_polynomial(b, lo, hi, P_nom, m_nom), lo, hi),
                b, rtol=1e-9, atol=1e-6)

    def test_same_curve_inside_and_outside_the_range(self):
        """The fan built from mapped betas gives the polynomial's power at
        every flow, inside the range and in the extrapolation beyond it (where
        the curve stays positive here, so the floor does not act)."""
        P_nom, m_nom, c = 900.0, 1.03, (-0.333, 0.643, 0.189, -0.002)
        lo, hi = 4.81, 10.09
        beta = polynomial_to_bernstein(P_nom, m_nom, c, lo, hi)
        fan = FanSystem(id="mapped", **{f"beta{k}": float(b) for k, b in enumerate(beta)},
                        flowRangeLower=lo, flowRangeUpper=hi)
        m = np.linspace(2.0, 14.0, 25)
        np.testing.assert_allclose(_fan_power(fan, m, hard=True), _polynomial_power(P_nom, m_nom, c, m), rtol=1e-6)
        # beta0 / beta3 are the powers at the ends of the range
        np.testing.assert_allclose(beta[0], _polynomial_power(P_nom, m_nom, c, lo), rtol=1e-12)
        np.testing.assert_allclose(beta[3], _polynomial_power(P_nom, m_nom, c, hi), rtol=1e-12)

    def test_non_negative_betas_give_non_negative_power_on_the_range(self):
        """Every beta >= 0 (the class bounds) keeps the power >= 0 over the
        whole range, and between the smallest and largest beta."""
        rng = np.random.default_rng(1)
        m = np.linspace(3.0, 23.0, 201)
        for _ in range(50):
            beta = rng.uniform(0, 1, 4) * rng.choice([0.0, 1.0], 4) * 20000.0  # some at the bound
            fan = FanSystem(id="nonneg", **{f"beta{k}": float(b) for k, b in enumerate(beta)},
                            flowRangeLower=3.0, flowRangeUpper=23.0)
            P = _fan_power(fan, m, hard=True)
            self.assertGreaterEqual(P.min(), 0.0)
            self.assertGreaterEqual(P.min(), beta.min() - 1e-6)
            self.assertLessEqual(P.max(), beta.max() + 1e-6)

    def test_set_flow_range_keeps_the_curve(self):
        """Moving the range maps the betas: the same power at every flow."""
        fan = FanSystem(id="moved", f_total=0.9)
        m = np.linspace(0.0, 25.0, 51)
        before = _fan_power(fan, m, hard=True)
        fan.set_flow_range(3.0, 23.0)
        self.assertAlmostEqual(float(fan.flowRangeLower.get()), 3.0)
        self.assertAlmostEqual(float(fan.flowRangeUpper.get()), 23.0)
        np.testing.assert_allclose(_fan_power(fan, m, hard=True), before, rtol=1e-9, atol=1e-6)
        with self.assertRaises(ValueError):
            fan.set_flow_range(5.0, 5.0)

    def test_estimated_parameters_are_the_betas_and_f_total(self):
        """The solver sees the four betas (lower bound zero) and f_total; the
        flow range is fixed, since it gives the betas their meaning."""
        fan = FanSystem(id="estimable")
        self.assertEqual(fan.config["parameters"], ["beta0", "beta1", "beta2", "beta3", "f_total"])
        for k in range(4):
            self.assertEqual(fan.parameter[f"beta{k}"]["lb"], 0.0)
        self.assertIn("flowRangeLower", FanSystem.PARAM_NAMES)
        self.assertNotIn("flowRangeLower", fan.parameter)


if __name__ == "__main__":
    unittest.main()
