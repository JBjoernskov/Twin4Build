"""Shared utilities for the paper experiment suite.

Everything here is platform- and experiment-agnostic: model construction,
timing, metrics, CSV I/O, and environment capture.  Experiment scripts only
assemble configurations out of these pieces.
"""

from __future__ import annotations

import csv
import datetime
import json
import os
import platform
import resource
import subprocess
import time
from pathlib import Path
from statistics import median
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import pytz
import torch

import twin4build as tb
from twin4build.model.model import Model
from twin4build.simulator.simulator import Simulator


# ---------------------------------------------------------------------------
# Defaults reused by synthetic models
# ---------------------------------------------------------------------------
TZ = pytz.UTC
DEFAULT_START = datetime.datetime(2023, 1, 15, 0, 0, 0, tzinfo=TZ)


# ---------------------------------------------------------------------------
# Platform control
# ---------------------------------------------------------------------------
def set_cpu_threads(device_key: str, thread_count: int) -> None:
    """Pin torch / OpenMP thread count for deterministic CPU timings.

    ``thread_count <= 0`` means "leave thread counts as-is" -- useful for a
    future GPU entry where CPU threading is not the bottleneck.
    """
    if device_key.startswith("cpu") and thread_count > 0:
        torch.set_num_threads(thread_count)
        # Best-effort: OpenMP and MKL pick up these env vars only if set
        # before the relevant libs are loaded, but setting them here does
        # not hurt and helps nested subprocesses.
        os.environ["OMP_NUM_THREADS"] = str(thread_count)
        os.environ["MKL_NUM_THREADS"] = str(thread_count)


def resolve_device(device_key: str) -> torch.device:
    """Map an experiment-facing platform key onto a torch.device.

    Kept as a single chokepoint so the GPU pass only needs to add a branch.
    """
    if device_key.startswith("cpu"):
        return torch.device("cpu")
    if device_key == "gpu":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "DEVICES contains 'gpu' but torch.cuda.is_available() is False."
            )
        return torch.device("cuda")
    raise ValueError(f"Unknown device key: {device_key!r}")


def peak_memory_mb(device_key: str) -> float:
    """Peak process memory in MB for the given platform.

    On Linux ``ru_maxrss`` is in KB; we normalize to MB.  For GPU runs this
    returns the CUDA allocator's peak.
    """
    if device_key.startswith("cpu"):
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    if device_key == "gpu":
        return torch.cuda.max_memory_allocated() / (1024.0 ** 2)
    return float("nan")


def current_rss_mb(device_key: str) -> float:
    """Current (live) process memory in MB -- unlike :func:`peak_memory_mb`
    this decreases as memory is freed.

    On CPU we read ``VmRSS`` from ``/proc/self/status`` so we can sample the
    resident set during a specific phase of work (e.g. just the compiled
    simulate loop) rather than the lifetime high-water mark that
    ``ru_maxrss`` tracks.  Falls back to ``ru_maxrss`` if ``/proc`` is
    unavailable (non-Linux).  For GPU we read the CUDA allocator's current
    allocation.
    """
    if device_key.startswith("cpu"):
        try:
            with open("/proc/self/status") as fh:
                for line in fh:
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1]) / 1024.0
        except OSError:
            pass
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    if device_key == "gpu":
        return torch.cuda.memory_allocated() / (1024.0 ** 2)
    return float("nan")


def reset_peak_memory(device_key: str) -> None:
    """Reset per-device peak trackers before a measured run."""
    if device_key == "gpu":
        torch.cuda.reset_peak_memory_stats()


# ---------------------------------------------------------------------------
# Synthetic multi-room builder
# ---------------------------------------------------------------------------
def make_weather_df(
    start: datetime.datetime,
    end: datetime.datetime,
    step_size: int,
) -> pd.DataFrame:
    """Synthetic outdoor conditions: diurnal temperature + solar, flat CO2."""
    freq = f"{step_size}s"
    idx = pd.date_range(
        start - datetime.timedelta(hours=1),
        end + datetime.timedelta(hours=1),
        freq=freq,
        tz=TZ,
    )
    hours = np.array([(t - start).total_seconds() / 3600 for t in idx])
    temperature = 2.0 + 5.0 * np.sin(2 * np.pi * hours / 24 - np.pi / 2)
    irradiation = np.maximum(0.0, 400.0 * np.sin(2 * np.pi * (hours - 6) / 24))
    co2 = np.full_like(hours, 420.0)
    return pd.DataFrame(
        {
            "outdoorTemperature": temperature,
            "globalIrradiation": irradiation,
            "outdoorCo2Concentration": co2,
        },
        index=idx,
    )


def build_multi_room_model(
    n_rooms: int,
    *,
    start: datetime.datetime = DEFAULT_START,
    horizon_days: int = 7,
    step_size: int = 600,
    heater_nelements: int | List[int] = 3,
    model_id: str = "paper_multi_room",
) -> Model:
    """Build an ``n_rooms``-zone model with air-side and water-side heating.

    A room-level replica of :mod:`twin4build.examples.multi_room_batching_test`
    adapted so ``heater_nelements`` can be either a scalar (uniform) or a
    per-room list (heterogeneous).  The list form is what E2 uses to split
    the batch group by varying ``n_states``.
    """
    if isinstance(heater_nelements, int):
        nelements_per_room = [heater_nelements] * n_rooms
    else:
        assert len(heater_nelements) == n_rooms, (
            "heater_nelements list must have length n_rooms"
        )
        nelements_per_room = list(heater_nelements)

    end = start + datetime.timedelta(days=horizon_days)
    model = Model(id=model_id)

    weather_df = make_weather_df(start, end, step_size)
    outdoor = tb.OutdoorEnvironmentSystem(df=weather_df, id="outdoor_environment")
    model.add_component(outdoor)

    damper_schedule = tb.ScheduleSystem(
        weekDayRulesetDict={
            "ruleset_default_value": 0.0,
            "ruleset_start_minute": [0, 0],
            "ruleset_end_minute":   [0, 0],
            "ruleset_start_hour":   [6, 18],
            "ruleset_end_hour":     [18, 24],
            "ruleset_value":        [0.8, 0.0],
        },
        id="damper_schedule",
    )
    valve_schedule = tb.ScheduleSystem(
        weekDayRulesetDict={
            "ruleset_default_value": 0.0,
            "ruleset_start_minute": [0, 0],
            "ruleset_end_minute":   [0, 0],
            "ruleset_start_hour":   [6, 18],
            "ruleset_end_hour":     [18, 24],
            "ruleset_value":        [0.6, 0.0],
        },
        id="valve_schedule",
    )
    supply_air_temp_schedule = tb.ScheduleSystem(
        weekDayRulesetDict={
            "ruleset_default_value": 18.0,
            "ruleset_start_minute": [0],
            "ruleset_end_minute":   [0],
            "ruleset_start_hour":   [6],
            "ruleset_end_hour":     [18],
            "ruleset_value":        [21.0],
        },
        id="supply_air_temp_schedule",
    )
    supply_water_temp_schedule = tb.ScheduleSystem(
        weekDayRulesetDict={
            "ruleset_default_value": 40.0,
            "ruleset_start_minute": [0],
            "ruleset_end_minute":   [0],
            "ruleset_start_hour":   [0],
            "ruleset_end_hour":     [24],
            "ruleset_value":        [60.0],
        },
        id="supply_water_temp_schedule",
    )
    occupancy_schedule = tb.ScheduleSystem(
        weekDayRulesetDict={
            "ruleset_default_value": 0.0,
            "ruleset_start_minute": [0, 0],
            "ruleset_end_minute":   [0, 0],
            "ruleset_start_hour":   [8, 17],
            "ruleset_end_hour":     [17, 24],
            "ruleset_value":        [5.0, 0.0],
        },
        id="occupancy_schedule",
    )
    for sched in (
        damper_schedule,
        valve_schedule,
        supply_air_temp_schedule,
        supply_water_temp_schedule,
        occupancy_schedule,
    ):
        model.add_component(sched)

    for i in range(n_rooms):
        tag = f"room_{i}"
        nelements_i = nelements_per_room[i]

        supply_damper = tb.DamperTorchSystem(
            a=1.0, nominalAirFlowRate=0.1, id=f"{tag}_supply_damper",
        )
        return_damper = tb.DamperTorchSystem(
            a=1.0, nominalAirFlowRate=0.1, id=f"{tag}_return_damper",
        )
        coil = tb.CoilTorchSystem(id=f"{tag}_coil")
        valve = tb.ValveTorchSystem(
            waterFlowRateMax=0.05, valveAuthority=0.5, id=f"{tag}_valve",
        )
        heater = tb.SpaceHeaterTorchSystem(
            Q_flow_nominal_sh=2000.0,
            T_a_nominal_sh=60.0,
            T_b_nominal_sh=45.0,
            TAir_nominal_sh=21.0,
            thermalMassHeatCapacity=50000.0,
            nelements=nelements_i,
            id=f"{tag}_space_heater",
        )
        room = tb.BuildingSpaceTorchSystem(
            thermal_kwargs={
                "C_air": 100000.0,
                "C_wall": 500000.0,
                "C_int": 100000.0,
                "C_boundary": 100000.0,
                "R_out": 0.01,
                "R_in": 0.001,
                "R_int": 0.005,
                "R_boundary": 0.01,
                "f_wall": 0.5,
                "f_air": 0.5,
                "Q_occ_gain": 80.0,
            },
            mass_kwargs={"V": 150.0, "G_occ": 8.18e-6, "m_inf": 0.005},
            id=tag,
        )

        model.add_connection(damper_schedule, supply_damper, "scheduleValue", "damperPosition")
        model.add_connection(damper_schedule, return_damper, "scheduleValue", "damperPosition")
        model.add_connection(supply_damper, coil, "airFlowRate", "airFlowRate")
        model.add_connection(outdoor, coil, "outdoorTemperature", "inletAirTemperature")
        model.add_connection(
            supply_air_temp_schedule, coil,
            "scheduleValue", "outletAirTemperatureSetpoint",
        )
        model.add_connection(coil, room, "outletAirTemperature", "supplyAirTemperature")
        model.add_connection(supply_damper, room, "airFlowRate", "supplyAirFlowRate")
        model.add_connection(return_damper, room, "airFlowRate", "exhaustAirFlowRate")

        model.add_connection(valve_schedule, valve, "scheduleValue", "valvePosition")
        model.add_connection(valve, heater, "waterFlowRate", "waterFlowRate")
        model.add_connection(
            supply_water_temp_schedule, heater,
            "scheduleValue", "supplyWaterTemperature",
        )
        model.add_connection(room, heater, "indoorTemperature", "indoorTemperature")
        model.add_connection(heater, room, "Power", "heatGain")

        model.add_connection(outdoor, room, "outdoorTemperature", "outdoorTemperature")
        model.add_connection(outdoor, room, "globalIrradiation", "globalIrradiation")
        model.add_connection(outdoor, room, "outdoorCo2Concentration", "outdoorCO2")
        model.add_connection(occupancy_schedule, room, "scheduleValue", "numberOfPeople")

    return model


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------
def timed(fn: Callable[[], Any]) -> Tuple[Any, float]:
    """Call ``fn()`` and return ``(result, elapsed_seconds)``."""
    t0 = time.perf_counter()
    result = fn()
    return result, time.perf_counter() - t0


def iqr(values: List[float]) -> Tuple[float, float, float]:
    """Return ``(q25, q50, q75)`` computed with numpy's linear interpolation."""
    if not values:
        return (float("nan"), float("nan"), float("nan"))
    arr = np.asarray(values, dtype=float)
    q25, q50, q75 = np.percentile(arr, [25, 50, 75])
    return float(q25), float(q50), float(q75)


def time_sim(
    simulate_callable: Callable[[], None],
    repeats: int,
    warmup: int = 1,
) -> Dict[str, Any]:
    """Run ``simulate_callable`` ``(repeats + warmup)`` times, discard warmup.

    Returns a dict with the raw measurements and summary statistics.
    """
    all_times: List[float] = []
    for _ in range(warmup + repeats):
        _, dt = timed(simulate_callable)
        all_times.append(dt)
    measured = all_times[warmup:]
    q25, q50, q75 = iqr(measured)
    return {
        "times_raw_s": all_times,
        "times_measured_s": measured,
        "median_s": q50,
        "iqr_low_s": q25,
        "iqr_high_s": q75,
        "min_s": float(np.min(measured)) if measured else float("nan"),
        "max_s": float(np.max(measured)) if measured else float("nan"),
    }


# ---------------------------------------------------------------------------
# Simulation wrapper
# ---------------------------------------------------------------------------
def simulate_once(
    model: Model,
    start: datetime.datetime,
    end: datetime.datetime,
    step_size: int,
) -> Simulator:
    simulator = Simulator(model)
    simulator.simulate(
        start_time=start,
        end_time=end,
        step_size=step_size,
        show_progress_bar=False,
    )
    return simulator


# ---------------------------------------------------------------------------
# Graph metrics
# ---------------------------------------------------------------------------
def graph_size(model: Model) -> Tuple[int, int]:
    """Return ``(n_components, n_connections)`` for a loaded model."""
    return (
        model.simulation_model.count_components(),
        model.simulation_model.count_connections(),
    )


def execution_group_sizes(model: Model) -> List[int]:
    """Return the sizes of each execution group in topological order."""
    return [len(group) for group in model.simulation_model.execution_order]


def compressed_group_stats(
    model_orig: Model, model_compiled: Model
) -> pd.DataFrame:
    """Per-execution-group compression: n_components before vs after compile.

    Returns a long-form DataFrame with columns
    ``group_idx, n_original, n_meta, batch_sizes`` where ``batch_sizes`` is a
    comma-joined list of ``n_c`` values belonging to that group.
    """
    rows = []
    orig_groups = model_orig.simulation_model.execution_order
    comp_groups = model_compiled.simulation_model.execution_order
    # The compiler processes groups in the same order; their indices align.
    for idx, (o_group, c_group) in enumerate(zip(orig_groups, comp_groups)):
        batch_sizes = [
            getattr(meta, "_n_c_compiled", 1) for meta in c_group
        ]
        rows.append(
            {
                "group_idx": idx,
                "n_original": len(o_group),
                "n_meta": len(c_group),
                "batch_sizes": ",".join(str(b) for b in batch_sizes),
                "mean_batch_size": (
                    float(np.mean(batch_sizes)) if batch_sizes else 0.0
                ),
                "max_batch_size": max(batch_sizes) if batch_sizes else 0,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Connection-wiring classification
# ---------------------------------------------------------------------------
def connection_wiring_stats(model_compiled: Model) -> Dict[str, int]:
    """Classify compiled-model connections into fast-aligned vs gather paths.

    A connection uses the *gather* path iff the compiler stored an
    ``index_select`` tensor for its ``(inputPort, Connection)`` pair; if no
    tensor is stored the source/receiver ``n_c`` were aligned and identity-
    mapped (see ``build_compiled_model`` in ``twin4build/model/model.py``).
    """
    n_aligned = 0
    n_gather = 0
    n_total = 0

    for comp in model_compiled.simulation_model.components.values():
        for cp in comp.connects_at:
            for conn in cp.connects_system_through:
                n_total += 1
                out_idx = cp.output_component_index.get(conn)
                in_idx = cp.input_component_index.get(conn)
                if isinstance(out_idx, torch.Tensor) or isinstance(
                    in_idx, torch.Tensor
                ):
                    n_gather += 1
                else:
                    n_aligned += 1
    return {
        "n_connections_total": n_total,
        "n_connections_aligned": n_aligned,
        "n_connections_gather": n_gather,
        "fraction_aligned": (n_aligned / n_total) if n_total else float("nan"),
    }


# ---------------------------------------------------------------------------
# Accuracy metrics
# ---------------------------------------------------------------------------
def _port_history_np(port: Any, i_c: int = 0) -> np.ndarray:
    """Return a port's history as a numpy array in ``(n_t, n_s, ...)`` form.

    Handles both Scalar (shape ``(n_t, n_s, n_c)``) and Vector (shape
    ``(n_t, n_s, n_c, n_v)``) ports uniformly by slicing out ``i_c``.
    """
    hist = port.history()  # torch tensor
    # Scalar: (n_t, n_s, n_c); Vector: (n_t, n_s, n_c, n_v)
    arr = hist[:, :, i_c].detach().cpu().numpy()
    return arr


def port_errors(
    model_orig: Model,
    model_compiled: Model,
    io_types: Iterable[str] = ("output",),
) -> pd.DataFrame:
    """Per-port ``max_abs`` and ``max_rel`` errors between orig and compiled.

    Iterates every component in the original model, resolves the
    corresponding ``(meta_component, i_c)`` via
    ``Model.get_compiled_component_info``, and compares each port's history
    in place.  Components without a compiled mapping are skipped with a
    reason flag.
    """
    rows = []
    for comp_id, comp in model_orig.simulation_model.components.items():
        meta_info = model_orig.get_compiled_component_info(comp_id)
        if meta_info is None:
            # Compiler did not register this component -- record it so the
            # audit is complete rather than silently dropping ports.
            rows.append(
                {
                    "component_id": comp_id,
                    "io_type": "",
                    "port_name": "",
                    "n_samples": 0,
                    "max_abs_err": float("nan"),
                    "max_rel_err": float("nan"),
                    "note": "no_compiled_mapping",
                }
            )
            continue
        meta, i_c = meta_info
        comp_meta = model_compiled.simulation_model.components[meta.id]

        for io_type in io_types:
            ports_orig = getattr(comp, io_type, {}) or {}
            ports_meta = getattr(comp_meta, io_type, {}) or {}
            for port_name, port_orig in ports_orig.items():
                port_meta = ports_meta.get(port_name)
                if port_meta is None:
                    continue
                try:
                    a = _port_history_np(port_orig, i_c=0)
                    b = _port_history_np(port_meta, i_c=i_c)
                except AssertionError:
                    # History not populated (log_history=False); skip.
                    continue
                if a.shape != b.shape:
                    rows.append(
                        {
                            "component_id": comp_id,
                            "io_type": io_type,
                            "port_name": port_name,
                            "n_samples": 0,
                            "max_abs_err": float("nan"),
                            "max_rel_err": float("nan"),
                            "note": f"shape_mismatch_{a.shape}_vs_{b.shape}",
                        }
                    )
                    continue
                diff = np.abs(a - b)
                denom = np.maximum(np.abs(a), 1e-12)
                max_abs = float(np.max(diff)) if diff.size else 0.0
                max_rel = float(np.max(diff / denom)) if diff.size else 0.0
                rows.append(
                    {
                        "component_id": comp_id,
                        "io_type": io_type,
                        "port_name": port_name,
                        "n_samples": int(a.size),
                        "max_abs_err": max_abs,
                        "max_rel_err": max_rel,
                        "note": "",
                    }
                )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CSV I/O
# ---------------------------------------------------------------------------
def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    """Write a list of dicts to CSV.  Creates the parent directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    for row in rows[1:]:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def append_csv(path: Path, row: Dict[str, Any]) -> None:
    """Append a single row to a CSV, writing the header on first call."""
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


# ---------------------------------------------------------------------------
# Environment snapshot
# ---------------------------------------------------------------------------
def _cpu_model() -> str:
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown"


def env_snapshot() -> Dict[str, Any]:
    """Capture the software + hardware environment for the results folder."""
    info: Dict[str, Any] = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "git_sha": _git_sha(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_model": _cpu_model(),
        "cpu_count": os.cpu_count(),
        "ram_total_mb": _total_ram_mb(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "torch_num_threads": torch.get_num_threads(),
        "torch_default_dtype": str(torch.get_default_dtype()),
        "cuda_available": torch.cuda.is_available(),
    }
    if info["cuda_available"]:
        info["cuda_device"] = torch.cuda.get_device_name(0)
    return info


def _total_ram_mb() -> Optional[float]:
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return kb / 1024.0
    except OSError:
        return None
    return None


def write_env_snapshot(results_dir: Path) -> None:
    info = env_snapshot()
    (results_dir / "env.json").write_text(json.dumps(info, indent=2))
