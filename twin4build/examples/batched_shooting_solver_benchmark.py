"""CPU/A100 benchmark for the experimental custom shooting solvers.

Run, for example::

    python -m twin4build.examples.batched_shooting_solver_benchmark \
        --arm batched-bfgs --hours 24 --maxiter 20 --device cpu

The full A100 study uses ``--hours 120 --n-starts 8``.  SciPy SLSQP is the
single-start reference and always uses the identical canonical initial point.
"""

from __future__ import annotations

import argparse
import datetime
import json
import platform
import runpy
import time
from pathlib import Path

import numpy as np
import torch
from dateutil import tz

import twin4build as tb
import twin4build.examples.utils as utils

fcn = runpy.run_path(str(Path(__file__).with_name("full_workflow_example.py")))["fcn"]


START = datetime.datetime(2023, 12, 2, tzinfo=tz.gettz("Europe/Copenhagen"))
STEP = 1200


def build_problem(device="cpu"):
    model = tb.Model(id="batched_shooting_benchmark")
    filename = utils.get_path(["estimator_example", "one_room_example_model.xlsm"])
    model.load(semantic_model_filename=filename, fcn=fcn)
    model.to(device, torch.float64)
    c = model.components
    space = c["office"]
    heater = c["office_space_heater"]
    heat_ctrl = c["office_temperature_heating_controller"]
    co2_ctrl = c["office_co2_controller"]
    valve = c["office_space_heater_valve"]
    supply = c["office_supply_damper"]
    exhaust = c["office_exhaust_damper"]
    occupancy = c["office_occupancy"]
    detector = c["office_occupancy_detector"]
    wall = c["office_boundary_wall"]
    parameters = [
        (space, "thermal.C_air", 5e5, 1e4, 5e5),
        (space, "thermal.C_wall", 1e6, 1e5, 3e6),
        (wall, "C", 1e6, 1e4, 1e7),
        (space, "thermal.R_out", 0.5, 0.01, 1),
        (space, "thermal.R_in", 0.1, 0.01, 1),
        (wall, "R_a", 0.04, 1e-4, 1),
        (wall, "R_b", 0.04, 1e-4, 1),
        (space, "thermal.f_wall", 0.1, 0, 10),
        (space, "thermal.f_air", 0.1, 0, 10),
        (space, "thermal.Q_occ_gain", 100.0, 10, 200),
        (heater, "thermalMassHeatCapacity", 1e4, 1e3, 2e5),
        (heater, "UA", None, 1, 100),
        (heat_ctrl, "kp", 0.005, 1e-5, 1, "private"),
        (co2_ctrl, "kp", 0.0001, 1e-5, 1, "private"),
        ([heat_ctrl, co2_ctrl], "Ti", 30, 1, 300, "private"),
        ([heat_ctrl, co2_ctrl], "Td", 0, 0, 1, "private"),
        (valve, "waterFlowRateMax", 0.001, 1e-6, 0.1),
        (valve, "valveAuthority", 1, 0.4, 1),
        ([supply, occupancy.supply_damper], "a", 1, 1, 10, "shared"),
        (
            [supply, occupancy.supply_damper],
            "nominalAirFlowRate",
            0.1,
            1e-5,
            1,
            "shared",
        ),
        ([exhaust, occupancy.exhaust_damper], "a", 1, 1, 10, "shared"),
        (
            [exhaust, occupancy.exhaust_damper],
            "nominalAirFlowRate",
            0.1,
            1e-5,
            1,
            "shared",
        ),
        ([space, occupancy], "mass.V", 65, 50, 80, "shared"),
        ([space, occupancy], "mass.G_occ", 1e-6, 1e-6, 1e-5, "shared"),
        ([space, occupancy], "mass.m_inf", 0.001, 1e-4, 0.01, "shared"),
        (detector, "threshold", 1.0, 0.02, 5.0),
    ]
    measurements = [
        (c["office_valve_position_sensor"], 0.025),
        (c["office_temperature_sensor"], 0.05),
        (c["office_damper_position_sensor"], 0.025),
        (c["office_co2_sensor"], 15.0),
    ]
    return model, parameters, measurements


def run(arm, hours, maxiter, n_starts, batch_size, device, capture):
    model, parameters, measurements = build_problem(device)
    simulator = tb.Simulator(model, execution_mode="composed")
    estimator = tb.Estimator(simulator)
    if arm == "slsqp":
        method = ("scipy", "SLSQP", "ad")
        options = {"maxiter": maxiter}
    else:
        method = ("custom", arm, "ad")
        options = {
            "maxiter": maxiter,
            "n_starts": n_starts,
            "batch_size": batch_size,
            "start_seed": 42,
            "start_spread": 0.15,
            "capture": capture,
        }
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    result = estimator.estimate(
        parameters=parameters,
        measurements=measurements,
        start_time=[START],
        end_time=[START + datetime.timedelta(hours=hours)],
        step_size=[STEP],
        n_warmup=min(20, max(0, int(hours * 3600 / STEP) // 4)),
        method=method,
        options=options,
    )
    if device == "cuda":
        torch.cuda.synchronize()
    seconds = time.perf_counter() - t0
    score_started = time.perf_counter()
    simulator.simulate(
        start_time=[START],
        end_time=[START + datetime.timedelta(hours=hours)],
        step_size=[STEP],
        show_progress_bar=False,
        execution_mode="object_graph",
    )
    warmup = min(20, max(0, int(hours * 3600 / STEP) // 4))
    sensor_rmse = {}
    weighted_sse = 0.0
    scored_values = 0
    for measuring_device, sd in measurements:
        predicted = (
            measuring_device.input["measuredValue"]
            .history(i_t=slice(warmup, None), i_s=0, i_c=0)
            .detach()
            .cpu()
            .numpy()
        )
        actual = np.asarray(
            estimator.actual_readings[measuring_device.id][0].to_numpy(),
            dtype=np.float64,
        )[warmup:warmup + len(predicted)]
        sensor_rmse[measuring_device.id] = float(
            np.sqrt(np.mean((actual - predicted) ** 2))
        )
        weighted_sse += float(np.sum(((actual - predicted) / float(sd)) ** 2))
        scored_values += len(predicted)
    score_seconds = time.perf_counter() - score_started
    row = {
        "arm": arm,
        "device": device,
        "hardware": (
            torch.cuda.get_device_name() if device == "cuda" else platform.processor()
        ),
        "hours": hours,
        "maxiter": maxiter,
        "n_starts": 1 if arm == "slsqp" else n_starts,
        "seconds": seconds,
        "score_seconds": score_seconds,
        "success": bool(result["success"]),
        "iterations": result["iterations"],
        "objective": float(result["final_objective"]),
        "peak_cuda_memory_gb": (
            torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else None
        ),
        "sensor_rmse": sensor_rmse,
        "rollout_weighted_mse": weighted_sse / max(1, scored_values),
        "multistart_audit": result.get("multistart_audit"),
        "derivative_stats": result.get("derivative_stats"),
        "iteration_history": result.get("iteration_history"),
    }
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arm",
        choices=("slsqp", "batched-bfgs", "batched-lm", "batched-newton"),
        required=True,
    )
    parser.add_argument("--hours", type=int, default=24)
    parser.add_argument("--maxiter", type=int, default=20)
    parser.add_argument("--n-starts", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--capture", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    row = run(
        args.arm,
        args.hours,
        args.maxiter,
        args.n_starts,
        args.batch_size,
        args.device,
        args.capture,
    )
    text = json.dumps(row, indent=2, default=lambda value: value.tolist())
    print(text)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(text)


if __name__ == "__main__":
    main()
