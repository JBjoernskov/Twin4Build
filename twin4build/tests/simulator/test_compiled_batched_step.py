"""``compiled_batched_step`` (``torch.compile(vmap(F_aug))``) steps every row as ``F_aug`` steps it alone (#241).

torch 2.11 Inductor miscompiled ``F_aug``'s assembly of vector inputs fed by several producers (``index_put`` into
an unbatched buffer under ``vmap``): every batch row wrote the same storage.  On the example office (the damper's
max over the CO2 and temperature loops is such a port) the CO2 was 421.5 instead of 438.9 ppm and the damper
command 0.0267 instead of 0.0182 after ten steps on the GPU; on the CPU Inductor failed in code generation.
"""
import datetime
import unittest

import numpy as np
import torch

import twin4build as tb
from twin4build.tests.estimator.example_fixture import EXAMPLE_START, STEP_SIZE, example_measurements, example_parameters, load_model


def _has_triton() -> bool:
    try:
        import triton  # noqa: F401
    except ImportError:
        return False
    return True


class TestCompiledBatchedStep(unittest.TestCase):
    STEPS, ROWS = 4, 3

    def _rows_agree(self, device):
        model = load_model()
        model.to(device=device, dtype=torch.float64)
        est = tb.Estimator(tb.Simulator(model, execution_mode="functional", execution_backend="eager", compile_step=False))
        start = EXAMPLE_START[0]
        est.estimate(
            parameters=example_parameters(model),
            measurements=example_measurements(model),
            start_time=[start],
            end_time=[start + datetime.timedelta(hours=24)],
            step_size=STEP_SIZE,
            n_warmup=5,
            method=("scipy", "SLSQP", "ad"),
            options={"maxiter": 1},
        )
        obj = est._functional_objective
        fm = obj.composer
        x0 = torch.tensor(np.asarray(est._x0_norm, dtype=np.float64), dtype=torch.float64, device=device)
        theta0 = obj._physical(x0)[0] if hasattr(obj, "_physical") else obj._denorm(x0)
        Theta = torch.stack([theta0 * (1.0 + 0.02 * b) for b in range(self.ROWS)])  # distinct rows
        Y = obj.Y0[0].unsqueeze(0).expand(self.ROWS, -1).contiguous()
        rows = [Y[b] for b in range(self.ROWS)]
        with torch.no_grad():
            for t in range(self.STEPS):
                u = obj.CAP[0][t]
                Y, M = fm.compiled_batched_step(Y, Theta, u)
                for b in range(self.ROWS):
                    rows[b], m = fm.F_aug(rows[b], Theta[b], u, transform_mode=True)
                    torch.testing.assert_close(M[b], m, rtol=1e-9, atol=1e-9, msg=f"step {t}, row {b}: outputs")
                    torch.testing.assert_close(Y[b], rows[b], rtol=1e-9, atol=1e-9, msg=f"step {t}, row {b}: state")
        # the rows do differ, so a batch collapsed onto one row would show
        self.assertFalse(torch.allclose(rows[0], rows[-1]))

    def test_rows_agree_on_the_cpu(self):
        self._rows_agree("cpu")

    @unittest.skipUnless(torch.cuda.is_available() and _has_triton(), "needs CUDA and Triton")
    def test_rows_agree_on_the_gpu(self):
        self._rows_agree("cuda")


if __name__ == "__main__":
    unittest.main()
