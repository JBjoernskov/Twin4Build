"""Quick timing/accuracy check for Model.to(device, dtype).

Companion to gpu_verify_device_support.ipynb: simulates the same two-zone
model on cpu/fp64, cuda/fp64 and cuda/fp32 and reports wall time and the
maximum zone-temperature deviation from the cpu/fp64 reference.  Run as a
subprocess so it is independent of the notebook kernel's import state:

    python twin4build/examples/gpu_verify_timing.py
"""

# Standard library imports
import time

# Third party imports
import torch

# Local application imports
import twin4build.utils.types as tps
from twin4build.tests.simulator.test_fusion import build_model, simulate


def main():
    configs = [("cpu", torch.float64)]
    if torch.cuda.is_available():
        configs += [("cuda", torch.float64), ("cuda", torch.float32)]
    else:
        print("No CUDA -- running the CPU configuration only.")

    ref = None
    for device, dtype in configs:
        tps.set_float_dtype(torch.float64)  # fresh fp64 default per build
        name = str(dtype).replace("torch.", "")
        model, *_ = build_model(model_id=f"verify_{device}_{name}")
        model.to(device, dtype)
        t0 = time.perf_counter()
        r = simulate(model, 24, 600)
        wall = time.perf_counter() - t0
        if ref is None:
            ref = r
        err = float((ref["t_a"] - r["t_a"].double().cpu()).abs().max())
        print(
            f"{device}/{name}: {wall:6.2f} s | output on {r['t_a'].device} | "
            f"max |dT_a| vs cpu/fp64: {err:.2e} K"
        )

    tps.set_float_dtype(torch.float64)
    print("\ndone -- expect ~1e-12 K for cuda/fp64 and < 1e-3 K for cuda/fp32")


if __name__ == "__main__":
    main()
