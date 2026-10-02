# Standard library imports
import datetime
import unittest

# Third party imports
import numpy as np
import torch
from dateutil import tz

# Local application imports
# Set test flag
import twin4build
from twin4build.systems.controller.setpoint_controller.pid_controller.pid_controller_system import (
    PIDControllerSystem,
)
from twin4build.systems.utils.smooth_saturation import saturation_mode

twin4build._IS_TESTING = True


class TestPIDControllerSystem(unittest.TestCase):
    def setUp(self):
        self.controller = PIDControllerSystem(id="test_pid", Kp=1.0, Ki=0.1, Kd=0.01)

    def test_initialization(self):
        """Test PID controller initialization."""
        self.assertIsNotNone(self.controller)
        self.assertEqual(self.controller.id, "test_pid")

    def test_do_step(self):
        """Test PID controller do_step method."""
        start_time = [datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=tz.UTC)]
        end_time = [datetime.datetime(2023, 1, 1, 1, 40, 0, tzinfo=tz.UTC)]
        step_size = [600]
        self.controller.initialize(
            start_time=start_time, end_time=end_time, step_size=step_size
        )

        # Set inputs
        self.controller.input["actualValue"].set(torch.tensor([20.0]), i_t=0)
        self.controller.input["setpointValue"].set(torch.tensor([22.0]), i_t=0)

        # Execute a time step
        datetime_val = datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=tz.UTC)
        self.controller.do_step(
            second_time=0, date_time=datetime_val, step_size=step_size, step_index=0
        )

        # Check output
        output = self.controller.output["inputSignal"].get()
        self.assertIsNotNone(output)

    def test_do_step_batch(self):
        """Test PID controller do_step method with batch size > 1."""
        controller_batch = PIDControllerSystem(
            id="test_pid_batch", Kp=1.0, Ki=0.1, Kd=0.01
        )

        batch_size = 3

        start_time = [
            datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=tz.UTC)
        ] * batch_size
        end_time = [datetime.datetime(2023, 1, 1, 1, 40, 0, tzinfo=tz.UTC)] * batch_size
        step_size = [600] * batch_size
        controller_batch.initialize(
            start_time=start_time, end_time=end_time, step_size=step_size
        )

        # Set inputs with batch size 3
        controller_batch.input["actualValue"].set(
            torch.tensor([20.0, 21.0, 19.0]), i_t=0
        )
        controller_batch.input["setpointValue"].set(
            torch.tensor([22.0, 22.0, 22.0]), i_t=0
        )

        # Execute a time step
        datetime_val = datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=tz.UTC)
        controller_batch.do_step(
            second_time=0, date_time=datetime_val, step_size=step_size, step_index=0
        )

        # Check output - verify batch shape consistency
        output = controller_batch.output["inputSignal"].get()
        self.assertIsNotNone(output)
        self.assertEqual(
            output.shape[0], batch_size
        )  # Output batch matches input batch

    def test_transform_path_bypasses_identity_cache_and_compiles(self):
        x = torch.tensor([[0.2, -0.1]], dtype=torch.float64)
        inputs = {
            "setpointValue": torch.tensor([22.0], dtype=torch.float64),
            "actualValue": torch.tensor([20.0], dtype=torch.float64),
        }
        params = {
            "kp": torch.tensor([0.5], dtype=torch.float64),
            "Ti": torch.tensor([30.0], dtype=torch.float64),
            "Td": torch.tensor([0.1], dtype=torch.float64),
            "output_min": torch.tensor([0.0], dtype=torch.float64),
            "output_max": torch.tensor([1.0], dtype=torch.float64),
        }

        self.controller._fwd_coef_cache = None

        def transformed(x_, kp):
            live_params = dict(params)
            live_params["kp"] = kp
            return self.controller.forward(
                x_,
                inputs,
                live_params,
                600.0,
                transform_mode=True,
            )[0]

        expected = transformed(x, params["kp"])
        self.assertIsNone(self.controller._fwd_coef_cache)
        if hasattr(torch, "compile"):
            compiled = torch.compile(
                transformed, backend="aot_eager", fullgraph=True, dynamic=False
            )
            torch.testing.assert_close(compiled(x, params["kp"]), expected)


def _run(controller, setpoint, actual, kp, Ti, Td=0.0, step=600.0):
    """The controller's output over the series, through ``forward`` from a
    zero state."""
    params = {
        "kp": torch.tensor(kp, dtype=torch.float64),
        "Ti": torch.tensor(Ti, dtype=torch.float64),
        "Td": torch.tensor(Td, dtype=torch.float64),
        "output_min": torch.tensor(0.0, dtype=torch.float64),
        "output_max": torch.tensor(1.0, dtype=torch.float64),
    }
    x = torch.zeros(2, dtype=torch.float64)
    out = []
    with saturation_mode("hard"):
        for sp, y in zip(setpoint, actual):
            x, o = controller.forward(
                x,
                {"setpointValue": torch.tensor(float(sp), dtype=torch.float64), "actualValue": torch.tensor(float(y), dtype=torch.float64)},
                params,
                step,
                transform_mode=True,
            )
            out.append(float(o["inputSignal"]))
    return np.array(out)


class TestPositionalForm(unittest.TestCase):
    """The positional PID: proportional about its setpoint when the integral
    time is infinite, the velocity form's output while unsaturated, and no
    windup."""

    def test_an_infinite_integral_time_is_proportional_about_the_setpoint(self):
        """A CO2 loop that opens a damper above 800 ppm, fully at 1000: one
        day of CO2 rising from 420 to 1000 ppm and back gives exactly
        ``clamp(kp * (CO2 - 800))``, closed below the setpoint."""
        t = np.arange(144)
        co2 = 420 + 580 * np.clip(np.minimum((t - 48) / 24, (110 - t) / 24), 0, 1)
        kp = 1 / 200
        direct = PIDControllerSystem(kp=kp, Ti=1e12, is_reverse=False, id="co2_loop")
        u = _run(direct, [800.0] * len(t), co2, kp, 1e12)
        np.testing.assert_allclose(u, np.clip(kp * (co2 - 800), 0, 1), atol=1e-6)
        self.assertEqual(float(u[co2 < 800].max()), 0.0)

    def test_unsaturated_it_is_the_velocity_form(self):
        # the room 0.2 - 0.6 K below its setpoint: the valve opens, never fully
        actual = 20.6 + 0.2 * np.sin(2 * np.pi * np.arange(60) / 20)
        setpoint = np.full(60, 21.0)
        kp, Ti, Td, step = 0.05, 1800.0, 60.0, 600.0
        u = _run(PIDControllerSystem(kp=kp, Ti=Ti, Td=Td, is_reverse=True, id="unsaturated"), setpoint, actual, kp, Ti, Td, step)
        self.assertTrue(((u > 0) & (u < 1)).all())  # the comparison holds while unsaturated
        e = setpoint - actual
        velocity, u_prev, e1, e2 = [], 0.0, 0.0, 0.0
        for k in range(len(e)):
            u_prev = u_prev + kp * ((1 + step / Ti + Td / step) * e[k] - (1 + 2 * Td / step) * e1 + Td / step * e2)
            e2, e1 = e1, e[k]
            velocity.append(u_prev)
        np.testing.assert_allclose(u, velocity, atol=1e-12)

    def test_the_integral_does_not_wind_up(self):
        """A room 2 K below its setpoint for a day saturates the valve open;
        the integral holds what saturates it (``1 - kp * 2``), not a day of
        error.  Once the room is 0.5 K above, the valve closes from there at
        the integral's rate, where a wound-up integral (``144 * ki * 2``)
        would hold it open for hundreds of steps."""
        actual = np.r_[np.full(144, 19.0), np.full(12, 21.5)]
        kp, Ti, step = 0.2, 1800.0, 600.0
        ki = kp * step / Ti
        u = _run(PIDControllerSystem(kp=kp, Ti=Ti, is_reverse=True, id="windup"), np.full(len(actual), 21.0), actual, kp, Ti, step=step)
        self.assertEqual(float(u[143]), 1.0)
        held = 1.0 - kp * 2.0
        np.testing.assert_allclose(u[144:], held - kp * 0.5 - ki * 0.5 * np.arange(1, 13), atol=1e-12)
        self.assertGreater(144 * ki * 2.0 - kp * 0.5 - 12 * ki * 0.5, 1.0)  # wound up: still saturated


if __name__ == "__main__":
    unittest.main()
