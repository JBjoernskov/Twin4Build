"""Device/dtype support tests for ``Model.to(device, dtype)``.

CPU-only guarantees (always run):

1. The default device is ``cpu`` / ``float64`` and ``Model.to("cpu")`` is a
   no-op: the simulation results are bit-identical.
2. ``Model.to("cpu", torch.float32)`` converts the whole model to single
   precision: outputs come back as float32 and agree with the float64
   reference within accumulated-roundoff tolerance.

CUDA parity (auto-skipped without a GPU):

3. A float64 simulation on ``cuda`` matches the CPU reference to solver
   precision.
4. The fast single-shooting estimation objective builds and passes its
   internal value+gradient validation against the object-graph objective
   with the model on ``cuda``.
"""

# Standard library imports
import datetime
import unittest

# Third party imports
import numpy as np
import pandas as pd
import torch
from dateutil import tz

# Local application imports
import twin4build as tb
import twin4build.utils.types as tps
from twin4build.tests.simulator.test_fusion import build_model, simulate

tb._IS_TESTING = True

START = datetime.datetime(2024, 1, 4, tzinfo=tz.gettz("Europe/Copenhagen"))
STEP_SIZE = 600
N_HOURS = 24


class TestModelToCPU(unittest.TestCase):
    """``Model.to`` semantics that must hold everywhere (no GPU required)."""

    def test_default_device_and_dtype(self):
        model, *_ = build_model(model_id="test_device_default")
        self.assertEqual(model.device, torch.device("cpu"))
        self.assertEqual(model.dtype, torch.float64)

    def test_to_cpu_is_noop(self):
        m1, *_ = build_model(model_id="test_device_ref")
        r1 = simulate(m1, N_HOURS, STEP_SIZE)

        m2, *_ = build_model(model_id="test_device_tocpu")
        self.assertIs(m2.to("cpu"), m2)  # returns self for chaining
        self.assertEqual(m2.device, torch.device("cpu"))
        r2 = simulate(m2, N_HOURS, STEP_SIZE)

        for key in ("t_a", "t_b", "t_w"):
            self.assertTrue(
                torch.equal(r1[key], r2[key]),
                f"Model.to('cpu') changed the {key} trajectory",
            )

    def test_float32_optin(self):
        # float64 reference first (the global dtype is still float64 here).
        m64, *_ = build_model(model_id="test_device_fp64")
        r64 = simulate(m64, N_HOURS, STEP_SIZE)

        # The dtype switch is process-wide -- restore it no matter what.
        self.addCleanup(tps.set_float_dtype, torch.float64)
        m32, *_ = build_model(model_id="test_device_fp32")
        m32.to("cpu", torch.float32)
        self.assertEqual(m32.dtype, torch.float32)
        r32 = simulate(m32, N_HOURS, STEP_SIZE)

        for key in ("t_a", "t_b", "t_w"):
            self.assertEqual(r32[key].dtype, torch.float32)
            err = float((r64[key] - r32[key].double()).abs().max())
            self.assertLess(
                err, 1e-3, f"fp32 {key} deviates {err:.2e} K from fp64"
            )

    def test_parameter_bounds_follow_dtype(self):
        """tps.Parameter bounds are tensors nn.Module.to() does not know
        about; Model.to must convert them alongside the parameter data."""
        self.addCleanup(tps.set_float_dtype, torch.float64)
        model, zone_a, *_ = build_model(model_id="test_device_bounds")
        p = zone_a.C_air
        p.min_value = 1e5
        p.max_value = 1e7
        model.to("cpu", torch.float32)
        self.assertEqual(p.dtype, torch.float32)
        self.assertEqual(p.min_value.dtype, torch.float32)
        self.assertEqual(p.max_value.dtype, torch.float32)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
class TestModelToCUDA(unittest.TestCase):
    """Float64 CPU/GPU parity for simulation and the fast estimation path."""

    def test_simulate_parity(self):
        m_cpu, *_ = build_model(model_id="test_device_cuda_ref")
        r_cpu = simulate(m_cpu, N_HOURS, STEP_SIZE)

        m_gpu, *_ = build_model(model_id="test_device_cuda")
        m_gpu.to("cuda")
        self.assertEqual(m_gpu.device.type, "cuda")
        r_gpu = simulate(m_gpu, N_HOURS, STEP_SIZE)

        for key in ("t_a", "t_b", "t_w"):
            self.assertEqual(r_gpu[key].device.type, "cuda")
            err = float((r_cpu[key] - r_gpu[key].cpu()).abs().max())
            self.assertLess(
                err, 1e-9, f"cuda {key} deviates {err:.2e} K from the CPU run"
            )

    def test_fast_estimation_objective_on_cuda(self):
        """The fast single-shooting objective must build AND pass its internal
        value+gradient cross-check against the object-graph objective (both
        running on the GPU); the returned optimum must be finite."""
        from twin4build.tests.estimator.test_two_zone_wall import (
            build_two_zone_model,
        )

        model, zone_a, zone_b, wall, sensor_a, sensor_b = build_two_zone_model()
        simulator = tb.Simulator(model, execution_mode="composed")
        start = START
        end = START + datetime.timedelta(hours=N_HOURS)

        # Synthetic measurements from a reference run (device-agnostic numpy).
        simulator.simulate(start_time=start, end_time=end, step_size=STEP_SIZE)
        t_a = zone_a.output["indoorTemperature"].history().detach().flatten()
        t_b = zone_b.output["indoorTemperature"].history().detach().flatten()
        index = pd.date_range(start=start, periods=len(t_a), freq=f"{STEP_SIZE}s")
        rng = np.random.default_rng(0)
        sensor_a.df = pd.DataFrame(
            {"value": t_a.cpu().numpy() + 0.05 * rng.standard_normal(len(t_a))},
            index=index,
        )
        sensor_b.df = pd.DataFrame(
            {"value": t_b.cpu().numpy() + 0.05 * rng.standard_normal(len(t_b))},
            index=index,
        )

        model.to("cuda")
        estimator = tb.Estimator(simulator)
        result = estimator.estimate(
            parameters=[
                (zone_a, "C_air", 1e6, 1e5, 1e7),
                (zone_b, "C_air", 1e6, 1e5, 1e7),
                (wall, "C", 2e5, 1e4, 1e7),
                (wall, "R_a", 0.02, 1e-3, 1.0),
                (wall, "R_b", 0.02, 1e-3, 1.0),
            ],
            measurements=[(sensor_a, 0.05), (sensor_b, 0.05)],
            start_time=[start],
            end_time=[end],
            step_size=STEP_SIZE,
            n_warmup=0,
            method=("scipy", "SLSQP", "ad"),
            options={"maxiter": 2},
        )
        # _setup_fast_objective only keeps the fast path when its value and
        # gradient agree with the object-graph objective -- on the GPU.
        self.assertIsNotNone(
            estimator._fast_obj,
            "fast single-shooting objective was not built/validated on cuda",
        )
        self.assertTrue(np.all(np.isfinite(result["result_x"])))


if __name__ == "__main__":
    unittest.main()
