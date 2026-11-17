# Standard library imports
import datetime
import math
import unittest

# Third party imports
import torch
from dateutil import tz

# Local application imports
import twin4build

twin4build._IS_TESTING = True

from twin4build.systems.wall.wall_system import WallSystem


class TestWallTorchSystem(unittest.TestCase):
    """Unit checks for the 2R1C wall: exact discrete update, energy balance,
    steady state between the side temperatures, and side symmetry."""

    C = 2e5
    R_A = 0.05
    R_B = 0.02
    T_INIT = 20.0
    DT = 600

    def _make_wall(self, R_a, R_b, T_init=T_INIT, n_steps=6, batch_size=1):
        wall = WallSystem(
            C=self.C, R_a=R_a, R_b=R_b, T_init=T_init, id="test_wall"
        )
        start = [
            datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=tz.UTC)
        ] * batch_size
        end = [
            datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=tz.UTC)
            + datetime.timedelta(seconds=self.DT * n_steps)
        ] * batch_size
        wall.initialize(
            start_time=start, end_time=end, step_size=[self.DT] * batch_size
        )
        return wall

    def _step(self, wall, t_a, t_b, i_t):
        wall.input["temperatureA"].set(torch.tensor([t_a]), i_t=i_t)
        wall.input["temperatureB"].set(torch.tensor([t_b]), i_t=i_t)
        wall.do_step(
            second_time=i_t * self.DT,
            date_time=None,
            step_size=[self.DT],
            step_index=i_t,
        )
        return (
            float(wall.output["heatFlowRateA"].get()),
            float(wall.output["heatFlowRateB"].get()),
            float(wall.output["wallTemperature"].get()),
        )

    def test_exact_discrete_update_and_flux_formulas(self):
        """One ZOH step must follow the exact scalar exponential update, and
        the heat flows must equal (T_w - T_side)/R_side at the end-of-step
        wall state."""
        t_a, t_b = 25.0, 15.0
        wall = self._make_wall(self.R_A, self.R_B)

        alpha = 1 / self.R_A + 1 / self.R_B
        t_ss = (t_a / self.R_A + t_b / self.R_B) / alpha
        decay = math.exp(-alpha * self.DT / self.C)

        t_w = self.T_INIT
        for k in range(4):
            q_a, q_b, t_w_out = self._step(wall, t_a, t_b, i_t=k)
            t_w = t_ss + (t_w - t_ss) * decay  # exact expected update
            self.assertAlmostEqual(t_w_out, t_w, places=9)
            self.assertAlmostEqual(q_a, (t_w - t_a) / self.R_A, places=9)
            self.assertAlmostEqual(q_b, (t_w - t_b) / self.R_B, places=9)

    def test_energy_balance(self):
        """The wall stores exactly what the two sides exchange: the integral
        of -(Q_a + Q_b) must match C * (T_w(end) - T_w(0)) to the ZOH
        sampling error (small steps -> small error)."""
        t_a, t_b = 30.0, 10.0
        dt = 60  # small vs the wall time constant (~2860 s)
        wall = WallSystem(
            C=self.C, R_a=self.R_A, R_b=self.R_B, T_init=self.T_INIT, id="wall_eb"
        )
        n_steps = 60
        start = [datetime.datetime(2023, 1, 1, tzinfo=tz.UTC)]
        end = [start[0] + datetime.timedelta(seconds=dt * n_steps)]
        wall.initialize(start_time=start, end_time=end, step_size=[dt])

        q_sum = 0.0
        for k in range(n_steps):
            wall.input["temperatureA"].set(torch.tensor([t_a]), i_t=k)
            wall.input["temperatureB"].set(torch.tensor([t_b]), i_t=k)
            wall.do_step(
                second_time=k * dt, date_time=None, step_size=[dt], step_index=k
            )
            q_sum += (
                float(wall.output["heatFlowRateA"].get())
                + float(wall.output["heatFlowRateB"].get())
            ) * dt
        t_w_end = float(wall.output["wallTemperature"].get())

        stored = self.C * (t_w_end - self.T_INIT)
        # -(Q_a + Q_b) integrated ~= stored energy; tolerance is the
        # first-order sampling error of evaluating fluxes at step ends.
        self.assertAlmostEqual(
            -q_sum / abs(stored), stored / abs(stored), delta=0.05
        )

    def test_steady_state(self):
        """After many time constants T_w settles at the R-weighted mean of the
        side temperatures and the two heat flows cancel (pure through-flux)."""
        t_a, t_b = 25.0, 15.0
        wall = self._make_wall(self.R_A, self.R_B, n_steps=200)
        for k in range(200):
            q_a, q_b, t_w = self._step(wall, t_a, t_b, i_t=k)

        alpha = 1 / self.R_A + 1 / self.R_B
        t_ss = (t_a / self.R_A + t_b / self.R_B) / alpha
        self.assertAlmostEqual(t_w, t_ss, places=6)
        self.assertAlmostEqual(q_a + q_b, 0.0, places=6)
        # Heat flows from the hot side (A) to the cold side (B).
        self.assertLess(q_a, 0.0)
        self.assertGreater(q_b, 0.0)

    def test_side_symmetry(self):
        """Swapping the sides (temperatures and resistances) must mirror the
        heat flows and leave the wall temperature unchanged."""
        t_a, t_b = 28.0, 12.0
        wall_1 = self._make_wall(self.R_A, self.R_B)
        wall_2 = self._make_wall(self.R_B, self.R_A)
        for k in range(4):
            q1_a, q1_b, t1_w = self._step(wall_1, t_a, t_b, i_t=k)
            q2_a, q2_b, t2_w = self._step(wall_2, t_b, t_a, i_t=k)
        self.assertAlmostEqual(t1_w, t2_w, places=9)
        self.assertAlmostEqual(q1_a, q2_b, places=9)
        self.assertAlmostEqual(q1_b, q2_a, places=9)

    def test_batch(self):
        """Batched simulation (n_s = 2) produces per-batch outputs."""
        wall = self._make_wall(self.R_A, self.R_B, batch_size=2)
        wall.input["temperatureA"].set(torch.tensor([25.0, 18.0]), i_t=0)
        wall.input["temperatureB"].set(torch.tensor([15.0, 22.0]), i_t=0)
        wall.do_step(
            second_time=0, date_time=None, step_size=[self.DT] * 2, step_index=0
        )
        q_a = wall.output["heatFlowRateA"].get()
        self.assertEqual(q_a.shape[0], 2)
        # Batch 0: A is the hot side (Q_a < 0); batch 1: A is the cold side.
        self.assertLess(float(q_a[0]), 0.0)
        self.assertGreater(float(q_a[1]), 0.0)


if __name__ == "__main__":
    unittest.main()
