"""CPU vs GPU scaling benchmark for the REAL estimation/optimization pipelines.

Companion module for ``gpu_benchmark_scaling.ipynb``.  Builds a parametric
N-zone building -- zones in a chain, each consecutive pair coupled by a
``WallTorchSystem``, every zone with its own heater schedule and temperature
sensor -- and times:

- ``run_estimation_case``:  ``Estimator.estimate`` (scipy SLSQP, AD, fast
  single-shooting objective) calibrating one ``C_air`` per zone and one wall
  ``C`` per wall against synthetic noisy measurements.  The reported number
  is seconds per objective+gradient evaluation, the cost every solver
  iteration pays.
- ``run_optimization_case``:  ``Optimizer.optimize`` (scipy SLSQP, AD, fast
  composed objective) choosing every zone's heater schedule to minimize
  energy subject to a comfort constraint.  Reported as seconds per SLSQP
  iteration.

Model size scales with N: each zone contributes 2 thermal states, each wall
1 more, and compile-time fusion turns the whole chain into ONE state-space
block -- so N directly controls the matrix sizes in the hot loop, which is
exactly the axis that decides whether a GPU pays off at batch size 1.

Run standalone for a quick CPU-only check::

    python twin4build/examples/gpu_benchmark_scaling.py
"""

# Standard library imports
import datetime
import threading
import time

# Third party imports
import numpy as np
import pandas as pd
import torch
from dateutil import tz

# Local application imports
import twin4build as tb
import twin4build.utils.types as tps

tb._IS_TESTING = True

START = datetime.datetime(2024, 1, 4, tzinfo=tz.gettz("Europe/Copenhagen"))
N_HOURS = 24
EST_STEP = 600  # 10-min steps -> 144 timesteps for estimation
OPT_STEP = 3600  # 1-h decision steps -> 24 decision vars per zone


class GpuUtilSampler:
    """Samples NVML GPU utilization while a timed section runs.

    NVML's ``utilization.gpu`` is the percentage of the past sample period
    during which at least one kernel was executing -- i.e. exactly the
    "time on CUDA vs wall-clock" fraction.  Sampled from a background
    thread so the timed section itself is undisturbed (unlike wrapping it
    in ``torch.profiler``, which inflates the wall time it measures).
    ``mean`` is NaN on CPU runs or when NVML is unavailable.
    """

    def __init__(self, enabled: bool, period_s: float = 0.1):
        self.enabled = enabled
        self.period_s = period_s
        self.samples = []
        self._stop = None
        self._thread = None

    def __enter__(self):
        if not self.enabled:
            return self
        try:  # torch.cuda.utilization() needs pynvml/nvidia-ml-py
            torch.cuda.utilization()
        except Exception:
            self.enabled = False
            return self
        self._stop = threading.Event()

        def _loop():
            while not self._stop.wait(self.period_s):
                try:
                    self.samples.append(torch.cuda.utilization())
                except Exception:
                    pass

        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        if self._thread is not None:
            self._stop.set()
            self._thread.join()
        return False

    @property
    def mean(self) -> float:
        """Average GPU-busy percentage over the section (NaN if no samples)."""
        return float(np.mean(self.samples)) if self.samples else float("nan")


def build_chain_model(n_zones: int, model_id: str):
    """N thermal zones in a chain, consecutive zones coupled by one wall.

    Every zone gets the shared outdoor/zero/supply-air schedules, its own
    heater schedule (decision variable in the optimization case) and its own
    temperature sensor (measurement in the estimation case).
    """
    model = tb.Model(id=model_id)

    zones = [
        tb.BuildingSpaceThermalTorchSystem(
            C_air=1e6, C_wall=5e6, R_out=0.01, R_in=0.01,
            f_wall=0.0, f_air=0.0, Q_occ_gain=100.0,
            id=f"Zone{i}",
        )
        for i in range(n_zones)
    ]
    walls = [
        tb.WallTorchSystem(C=2e5, R_a=0.02, R_b=0.02, id=f"Wall{i}")
        for i in range(n_zones - 1)
    ]

    outdoor = tb.ScheduleSystem(
        weekDayRulesetDict={"ruleset_default_value": 5.0}, id="Outdoor"
    )
    zero = tb.ScheduleSystem(
        weekDayRulesetDict={"ruleset_default_value": 0.0}, id="Zero"
    )
    supply_air = tb.ScheduleSystem(
        weekDayRulesetDict={"ruleset_default_value": 20.0}, id="SupplyAirTemp"
    )
    heaters = [
        tb.ScheduleSystem(
            weekDayRulesetDict={
                "ruleset_default_value": 0.0,
                "ruleset_start_minute": [0], "ruleset_end_minute": [0],
                "ruleset_start_hour": [6], "ruleset_end_hour": [20],
                "ruleset_value": [1500.0],
            },
            id=f"Heater{i}",
        )
        for i in range(n_zones)
    ]
    sensors = [tb.SensorSystem(id=f"TempSensor{i}") for i in range(n_zones)]

    for i, zone in enumerate(zones):
        model.add_connection(outdoor, zone, "scheduleValue", "outdoorTemperature")
        model.add_connection(zero, zone, "scheduleValue", "supplyAirFlowRate")
        model.add_connection(zero, zone, "scheduleValue", "exhaustAirFlowRate")
        model.add_connection(supply_air, zone, "scheduleValue", "supplyAirTemperature")
        model.add_connection(zero, zone, "scheduleValue", "globalIrradiation")
        model.add_connection(zero, zone, "scheduleValue", "numberOfPeople")
        model.add_connection(heaters[i], zone, "scheduleValue", "heatGain")
        model.add_connection(zone, sensors[i], "indoorTemperature", "measuredValue")

    # Middle zones receive wallHeatGain from BOTH neighbouring walls, so the
    # vector-port index is a per-zone counter, not a per-wall constant.
    port_idx = [0] * n_zones
    for i, wall in enumerate(walls):
        za, zb = zones[i], zones[i + 1]
        model.add_connection(za, wall, "indoorTemperature", "temperatureA")
        model.add_connection(zb, wall, "indoorTemperature", "temperatureB")
        model.add_connection(
            wall, za, "heatFlowRateA", "wallHeatGain", input_port_index=port_idx[i]
        )
        port_idx[i] += 1
        model.add_connection(
            wall, zb, "heatFlowRateB", "wallHeatGain", input_port_index=port_idx[i + 1]
        )
        port_idx[i + 1] += 1

    model.load(draw_semantic_model=False, draw_simulation_model=False)
    return model, zones, walls, sensors, heaters


def _attach_synthetic_measurements(model, zones, sensors, step_size):
    """Simulate the truth on the current device and attach noisy readings."""
    simulator = tb.Simulator(model)
    end = START + datetime.timedelta(hours=N_HOURS)
    simulator.simulate(
        start_time=START, end_time=end, step_size=step_size,
        show_progress_bar=False,
    )
    rng = np.random.default_rng(0)
    for zone, sensor in zip(zones, sensors):
        truth = (
            zone.output["indoorTemperature"].history()
            .detach().cpu().flatten().double().numpy()
        )
        index = pd.date_range(start=START, periods=len(truth), freq=f"{step_size}s")
        sensor.df = pd.DataFrame(
            {"value": truth + 0.05 * rng.standard_normal(len(truth))}, index=index
        )
    return simulator


def run_estimation_case(n_zones, device="cpu", dtype=torch.float64, maxiter=2):
    """Time the real estimation pipeline; returns a result row (dict)."""
    tps.set_float_dtype(torch.float64)  # fresh default before each build
    model, zones, walls, sensors, _ = build_chain_model(
        n_zones, f"bench_est_{n_zones}_{device}_{str(dtype).replace('torch.', '')}"
    )
    simulator = _attach_synthetic_measurements(model, zones, sensors, EST_STEP)
    model.to(device, dtype)

    parameters = [(z, "C_air", 1e6, 1e5, 1e7) for z in zones]
    parameters += [(w, "C", 2e5, 1e4, 1e7) for w in walls]
    measurements = [(s, 0.05) for s in sensors]
    end = START + datetime.timedelta(hours=N_HOURS)

    estimator = tb.Estimator(simulator)
    with GpuUtilSampler(enabled=device != "cpu") as util:
        t0 = time.perf_counter()
        result = estimator.estimate(
            parameters=parameters,
            measurements=measurements,
            start_time=[START],
            end_time=[end],
            step_size=EST_STEP,
            n_warmup=0,
            method=("scipy", "SLSQP", "ad"),
            options={"maxiter": maxiter, "fast": True},
        )
        wall_s = time.perf_counter() - t0
    n_eval = result["nfev"] if "nfev" in result else None

    tps.set_float_dtype(torch.float64)
    return {
        "case": "estimation",
        "n_zones": n_zones,
        "n_states": 3 * n_zones - 1,
        "n_theta": len(parameters),
        "device": device,
        "dtype": str(dtype).replace("torch.", ""),
        "fast": estimator._fast_obj is not None,
        "wall_s": wall_s,
        "n_eval": n_eval,
        "s_per_eval": wall_s / n_eval if n_eval else float("nan"),
        "gpu_util_pct": util.mean,
    }


def run_optimization_case(n_zones, device="cpu", dtype=torch.float64, maxiter=5):
    """Time the real optimization pipeline; returns a result row (dict)."""
    tps.set_float_dtype(torch.float64)
    model, zones, walls, sensors, heaters = build_chain_model(
        n_zones, f"bench_opt_{n_zones}_{device}_{str(dtype).replace('torch.', '')}"
    )
    model.to(device, dtype)

    heating_setpoint = tb.ScheduleSystem(
        weekDayRulesetDict={
            "ruleset_default_value": 18.0,
            "ruleset_start_minute": [0], "ruleset_end_minute": [0],
            "ruleset_start_hour": [8], "ruleset_end_hour": [17],
            "ruleset_value": [21.0],
        },
        id="HeatingSetpoint",
    )
    end = START + datetime.timedelta(hours=N_HOURS)

    simulator = tb.Simulator(model)
    optimizer = tb.Optimizer(simulator)
    with GpuUtilSampler(enabled=device != "cpu") as util:
        t0 = time.perf_counter()
        optimizer.optimize(
            start_time=START,
            end_time=end,
            step_size=OPT_STEP,
            variables=[(h, "scheduleValue", 0.0, 3000.0) for h in heaters],
            objectives=[(h, "scheduleValue", "min") for h in heaters],
            ineq_cons=[
                (z, "indoorTemperature", "lower", heating_setpoint) for z in zones
            ],
            method=("scipy", "SLSQP", "ad"),
            options={"maxiter": maxiter},
        )
        wall_s = time.perf_counter() - t0

    n_steps = int(N_HOURS * 3600 / OPT_STEP)
    tps.set_float_dtype(torch.float64)
    return {
        "case": "optimization",
        "n_zones": n_zones,
        "n_states": 3 * n_zones - 1,
        "n_vars": n_zones * n_steps,
        "device": device,
        "dtype": str(dtype).replace("torch.", ""),
        "fast": optimizer._fast_obj is not None,
        "wall_s": wall_s,
        "maxiter": maxiter,
        "s_per_iter": wall_s / maxiter,
        "gpu_util_pct": util.mean,
    }


def sweep(case_fn, sizes, configs, **kwargs) -> pd.DataFrame:
    """Run ``case_fn`` over every (size, device/dtype) combination."""
    rows = []
    for n in sizes:
        for device, dtype in configs:
            row = case_fn(n, device=device, dtype=dtype, **kwargs)
            rows.append(row)
            metric = row.get("s_per_eval", row.get("s_per_iter"))
            util = row["gpu_util_pct"]
            util_txt = f"{util:5.1f}% GPU-busy" if util == util else "  cpu"
            print(
                f"{row['case']:>12} | N={n:>3} | {device}/{row['dtype']:<7} | "
                f"total {row['wall_s']:7.1f} s | {metric:7.3f} s/step-unit | "
                f"{util_txt} | fast={row['fast']}"
            )
    return pd.DataFrame(rows)


def breakeven(df: pd.DataFrame, metric: str, gpu_config=("cuda", "float64")):
    """Smallest N where the GPU config beats cpu/float64 (None if never)."""
    cpu = df[(df.device == "cpu")].set_index("n_zones")[metric]
    gpu = df[
        (df.device == gpu_config[0]) & (df.dtype == gpu_config[1])
    ].set_index("n_zones")[metric]
    for n in cpu.index:
        if n in gpu.index and gpu[n] < cpu[n]:
            return n
    return None


if __name__ == "__main__":
    configs = [("cpu", torch.float64)]
    if torch.cuda.is_available():
        configs += [("cuda", torch.float64), ("cuda", torch.float32)]
    print("== estimation ==")
    df_e = sweep(run_estimation_case, [1, 2, 4], configs)
    print("== optimization ==")
    df_o = sweep(run_optimization_case, [1, 2, 4], configs, maxiter=3)
    print(df_e.to_string(index=False))
    print(df_o.to_string(index=False))
