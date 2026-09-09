"""Shared implementation for the three authoritative scaling notebooks.

Every benchmark starts from the translated model configured by
``twin4build/examples/full_workflow_example.py``.  Scaling is a disjoint union
of deep-copied, prefixed instances of that exact 23-component graph.
"""

from __future__ import annotations

import copy
import datetime as dt
import importlib.util
import json
import os
import platform
import random
import statistics
import subprocess
import threading
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd
import torch
from dateutil import tz

try:
    import psutil
except ImportError:  # host memory stats then degrade to None
    psutil = None

import twin4build as tb
import twin4build.examples.utils as example_utils
from twin4build.simulator._functional_simulation import (
    FUNCTIONAL_MATERIALIZATION_REVISION,
)
from twin4build.utils.rgetattr import rgetattr

SEED = 20260902
BENCHMARK_IMPLEMENTATION_REVISION = "post-run-finiteness-v1"
HISTORICAL_BENCHMARK_IMPLEMENTATION_REVISION = "per-edge-nan-validation-v1"
ZONE_COUNTS = [1, 10, 50, 100]
DEVICES = ["cpu", "cuda"]
SIMULATION_MATRIX = [
    {
        "device": device,
        "model_layout": model_layout,
        "execution_mode": execution_mode,
        "execution_backend": execution_backend,
    }
    for device, model_layout, execution_mode, execution_backend in (
        ("cpu", "standard", "object", "eager"),
        ("cuda", "standard", "object", "eager"),
        ("cpu", "batched", "object", "eager"),
        ("cuda", "batched", "object", "eager"),
        ("cpu", "batched", "functional", "eager"),
        ("cuda", "batched", "functional", "eager"),
        ("cuda", "batched", "functional", "cuda_graph"),
    )
]
# Valid five-repetition medians from the interrupted 2026-09-02 run. That run
# measured object execution before the functional-mode bug was fixed, so its
# former labels map exactly to the standard/batched object cases below. The
# raw checkpoint was lost; retain the measured aggregates without fabricating
# individual repetitions.
RETAINED_SIMULATION_SUMMARIES = {
    (1, "cpu", "standard", "object", "eager"): {
        "seconds_median": 2.033,
        "max_parity_error": 0.0,
    },
    (1, "cuda", "standard", "object", "eager"): {
        "seconds_median": 9.865,
        "max_parity_error": 0.0,
    },
    (1, "cpu", "batched", "object", "eager"): {
        "seconds_median": 2.082,
        "max_parity_error": 0.0,
    },
    (1, "cuda", "batched", "object", "eager"): {
        "seconds_median": 10.134,
        "max_parity_error": 0.0,
    },
    (10, "cpu", "standard", "object", "eager"): {
        "seconds_median": 24.178,
        "max_parity_error": 6.89e-5,
    },
    (10, "cuda", "standard", "object", "eager"): {
        "seconds_median": 108.986,
        "max_parity_error": 6.89e-5,
    },
    (10, "cpu", "batched", "object", "eager"): {
        "seconds_median": 5.806,
        "max_parity_error": 6.89e-5,
    },
    (10, "cuda", "batched", "object", "eager"): {
        "seconds_median": 26.804,
        "max_parity_error": 6.89e-5,
    },
    (50, "cpu", "standard", "object", "eager"): {
        "seconds_median": 175.449,
        "max_parity_error": 7.90e-5,
    },
    (50, "cuda", "standard", "object", "eager"): {
        "seconds_median": 845.530,
        "max_parity_error": 7.90e-5,
    },
    (50, "cpu", "batched", "object", "eager"): {
        "seconds_median": 23.401,
        "max_parity_error": 7.90e-5,
    },
    (50, "cuda", "batched", "object", "eager"): {
        "seconds_median": 100.370,
        "max_parity_error": 7.90e-5,
    },
    (100, "cpu", "standard", "object", "eager"): {
        "seconds_median": 1062.323,
        "seconds_min": 690.996,
        "seconds_max": 2184.549,
        "seconds_spread": 1493.553,
    },
}
RETAINED_SIMULATION_REPETITIONS = {
    (100, "cpu", "batched", "object", "eager", 0): 39.483,
}
ESTIMATION_MATRIX = [
    ("cpu", "slsqp-single-shooting", 1),
    ("cuda", "slsqp-single-shooting", 1),
    ("cuda", "custom-batched-sqp", 1),
    ("cuda", "custom-batched-sqp", 8),
    # Block trust region (issue #142): same starts, per-zone blocks from the
    # model structure, per-block acceptance/radius; kept alongside the batched
    # SQP rows so its line-search failures at 50/100 zones stay as the reference.
    ("cuda", "custom-batched-tr", 1),
    ("cuda", "custom-batched-tr", 8),
    ("cuda", "ipopt-collocation", 1),
    # "slsqp5-ipopt-collocation" (5 SLSQP iterations, then collocation) was
    # dropped from the matrix: it was a workaround for the +/-6 boundary-state
    # box (fixed) and buys nothing over a data-seeded cold collocation start.
]
# CPU single-shooting is only run up to this many zones: at 10 zones it already
# loses to every CUDA arm (2271 s vs 1443 s for the same SLSQP), and the 50/100
# zone cases would cost hours for a row that carries no information.
ESTIMATION_CPU_MAX_ZONES = 10
# Solvers listed (comma-separated) in this environment variable are DEFERRED by
# run_estimation_scaling: no row is written for them, so a later run without
# the variable picks them up from the checkpoint as still-queued cases.
ESTIMATION_DEFER_SOLVERS_ENV = "T4B_BENCHMARK_DEFER_SOLVERS"
# Same for devices (e.g. ``cpu``): no row is written, the case stays queued.
ESTIMATION_DEFER_DEVICES_ENV = "T4B_BENCHMARK_DEFER_DEVICES"
# A tag appended to every checkpoint / results file name, so a run under a
# different configuration (another machine, the compiled functional step)
# keeps its own checkpoint instead of resuming from rows measured elsewhere.
RESULTS_TAG_ENV = "T4B_BENCHMARK_RESULTS_TAG"


def _tagged(benchmark: str) -> str:
    tag = os.environ.get(RESULTS_TAG_ENV, "").strip()
    return f"{benchmark}_{tag}" if tag else benchmark
ESTIMATION_METHODS = {
    "slsqp-single-shooting": ("scipy", "SLSQP", "ad"),
    "ipopt-collocation": ("casadi", "ipopt", "ad", "collocation"),
    "slsqp5-ipopt-collocation": ("staged", "slsqp5-ipopt-collocation", "ad"),
    "custom-batched-sqp": ("custom", "batched-sqp", "ad"),
    "custom-batched-tr": ("custom", "batched-tr", "ad"),
}
COLLOCATION_SOLVERS = frozenset(
    {"ipopt-collocation", "slsqp5-ipopt-collocation"}
)
ESTIMATION_SLSQP_WARMSTART_ITERS = 5
ESTIMATION_CASE_MODULE = "benchmarks.estimation_case"
ESTIMATION_BENCHMARK_IMPLEMENTATION_REVISION = 3
OPTIMIZATION_MATRIX = [(device, "SLSQP") for device in DEVICES]
PARETO_SOLVERS = ("SLSQP", "ipopt")
_TRANSLATED_TEMPLATE_DIRECTORIES: list[tempfile.TemporaryDirectory] = []
ESTIMATION_SOLVER_BUDGET = 300
OPTIMIZATION_SOLVER_BUDGET = 300
PARETO_SOLVER_BUDGET = 300
MAX_COLLOCATION_DENSE_HESSIAN_BYTES = 8 * 1024**3
# Measured peak VRAM of the exact-Hessian collocation solve on an A100 40 GB
# (torch max_memory_allocated): 1.25 GiB at 1 zone, 11.7 GiB at 10 zones,
# out-of-memory at 50 zones (>30.7 GiB allocated, 39.5 GiB in use).  The AD
# bundles (Jacobian / Hessian tapes over 360 steps) dominate, not the sparse
# storage the old preflight counted, so the fit is ~1.16 GiB per zone.
COLLOCATION_VRAM_GIB_PER_ZONE = 1.16
COLLOCATION_VRAM_GIB_BASE = 0.6  # CUDA context + 1-zone intercept
COLLOCATION_VRAM_SAFETY_FRACTION = 0.85
MAX_PARETO_SCALAR_CONSTRAINTS = 150_000
MAX_PARETO_DENSE_JACOBIAN_BYTES = 2 * 1024**3

START = dt.datetime(2023, 12, 2, tzinfo=tz.gettz("Europe/Copenhagen"))
OPT_START = dt.datetime(2023, 12, 1, tzinfo=tz.gettz("Europe/Copenhagen"))
STEP_SIZE = 1200
CANONICAL_COMPONENTS_PER_ZONE = 23
CANONICAL_CONNECTIONS_PER_ZONE = 32
CANONICAL_STATES_PER_ZONE = 13
CANONICAL_AUGMENTED_STATES_PER_ZONE = 16
CANONICAL_GLOBAL_AUGMENTED_STATES = 1
CANONICAL_PARAMETER_GROUPS_PER_ZONE = 26
CANONICAL_THETA_PER_ZONE = 28
CANONICAL_MEASUREMENTS_PER_ZONE = 4
RESULTS_DIR = Path("generated_files") / "canonical_benchmarks"
THETA_KEYS = (
    "thermal.C_air",
    "thermal.C_wall",
    "wall.C",
    "thermal.R_out",
    "thermal.R_in",
    "wall.R_a",
    "wall.R_b",
    "thermal.f_wall",
    "thermal.f_air",
    "thermal.Q_occ_gain",
    "heater.thermalMassHeatCapacity",
    "heater.UA",
    "heating_pid.kp",
    "co2_pid.kp",
    "heating_pid.Ti",
    "co2_pid.Ti",
    "heating_pid.Td",
    "co2_pid.Td",
    "valve.waterFlowRateMax",
    "valve.valveAuthority",
    "occupancy_detector.threshold",
    "supply_damper.a",
    "supply_damper.nominalAirFlowRate",
    "exhaust_damper.a",
    "exhaust_damper.nominalAirFlowRate",
    "mass.V",
    "mass.G_occ",
    "mass.m_inf",
)

ROLE_NAMES = (
    "office_supply_damper",
    "office",
    "office_exhaust_damper",
    "office_space_heater",
    "outdoor_environment",
    "supply_air_temperature_sensor",
    "office_co2_controller",
    "office_co2_sensor",
    "office_co2_setpoint",
    "office_temperature_sensor",
    "office_temperature_heating_controller",
    "office_temperature_heating_setpoint",
    "office_damper_position_sensor",
    "office_valve_position_sensor",
    "office_space_heater_valve",
    "office_boundary_wall",
    "boundary_temp_schedule",
    "supply_water_schedule",
    "office_occupancy",
    "office_occupancy_detector",
    "office_occupancy_controller",
    "office_occupancy_controller_setpoint",
    "office_damper_max",
)


@dataclass(frozen=True)
class BenchmarkConfig:
    mode: str = "smoke"
    seed: int = SEED

    def __post_init__(self) -> None:
        if self.mode not in {"smoke", "full"}:
            raise ValueError("mode must be 'smoke' or 'full'")

    @property
    def hours(self) -> int:
        return 2 if self.mode == "smoke" else 120

    @property
    def optimization_hours(self) -> int:
        return 2 if self.mode == "smoke" else 72

    @property
    def repeats(self) -> int:
        return 1 if self.mode == "smoke" else 5

    @property
    def estimation_repeats(self) -> int:
        return 1

    @property
    def maxiter(self) -> int:
        """Compatibility smoke budget; scaling runners use named budgets."""
        return 1 if self.mode == "smoke" else ESTIMATION_SOLVER_BUDGET

    @property
    def estimation_maxiter(self) -> int:
        return 1 if self.mode == "smoke" else ESTIMATION_SOLVER_BUDGET

    @property
    def optimization_maxiter(self) -> int:
        return 1 if self.mode == "smoke" else OPTIMIZATION_SOLVER_BUDGET

    @property
    def pareto_maxiter(self) -> int:
        return 1 if self.mode == "smoke" else PARETO_SOLVER_BUDGET

    @property
    def pareto_points(self) -> int:
        return 3 if self.mode == "smoke" else 11

    @property
    def zone_counts(self) -> list[int]:
        return [1] if self.mode == "smoke" else ZONE_COUNTS.copy()


def seed_everything(seed: int = SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def synchronize(device: str) -> None:
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


MEMORY_SAMPLER_INTERVAL_SECONDS = 1.0
# A heartbeat gap longer than this (beyond the sampler interval) is treated as
# the process having been frozen.  Long GIL holds in native code are seconds,
# a lid close is minutes to hours.
FREEZE_GAP_SECONDS = 60.0
# Peak memory of the most recent ``timed`` region; row builders copy it via
# ``memory_stats_of_last_timed()`` right after the timed call they report.
LAST_TIMED_MEMORY: dict[str, Any] = {}


class _PeakMemoryMonitor:
    """Peak host and device memory over a timed region, at negligible cost.

    Three sources, none of which touch the timed code path:

    * PyTorch's caching-allocator peak counters (``max_memory_allocated`` /
      ``max_memory_reserved``): bookkeeping the allocator keeps anyway, reset
      at region start and read at region end.  This is the process's own
      tensor memory (CUDA Graph pools are reserved memory).
    * A daemon thread sampling the caching allocator's *bookkeeping*
      (``memory_reserved`` / ``memory_allocated``, pure host-side counters)
      and the process RSS once per ``MEMORY_SAMPLER_INTERVAL_SECONDS``.  The
      thread deliberately makes NO CUDA runtime/driver call: PyTorch captures
      CUDA Graphs in ``capture_error_mode="global"``, where a potentially
      unsafe CUDA API call from ANY thread (``cudaMemGetInfo`` included)
      can invalidate the capture in flight.  An earlier version of this
      sampler polled ``torch.cuda.mem_get_info``; every multi-zone shooting
      case on an A100 (torch 2.11 / CUDA 12.8) then died with an illegal
      memory access right after graph capture, while the same cases had
      passed on a laptop before the sampler existed.  A local reproduction
      (torch 2.13 / CUDA 13, Windows) did NOT trigger it, so the sampler is a
      suspect by timing rather than a confirmed cause; keeping the thread
      free of CUDA calls is the safe contract either way.  Device-wide used
      memory is read on the main thread before and after the region instead.
    * Windows' process-lifetime peak working set (``peak_wset``), which the
      OS tracks for free.  Each benchmark case runs in a fresh interpreter,
      so the lifetime peak is the case peak.
    """

    def __init__(self, device: str) -> None:
        self.cuda = device == "cuda" and torch.cuda.is_available()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._proc = psutil.Process() if psutil is not None else None
        self.device_used_baseline: int | None = None
        self.device_used_end: int | None = None
        self.reserved_peak_sampled = 0
        self.rss_peak = 0
        self.samples = 0
        self.stats: dict[str, Any] = {}
        # Heartbeat: a frozen process (classic sleep OR a Modern Standby
        # desktop-activity-moderator freeze, which Windows does not count as
        # sleep) shows up as a gap between consecutive samples far longer
        # than the interval.  Gaps above FREEZE_GAP_SECONDS are summed.
        self._last_beat: float | None = None
        self.frozen_seconds = 0.0

    def _sample(self) -> None:
        now = time.perf_counter()
        if self._last_beat is not None:
            gap = now - self._last_beat - MEMORY_SAMPLER_INTERVAL_SECONDS
            if gap > FREEZE_GAP_SECONDS:
                self.frozen_seconds += gap
        self._last_beat = now
        try:
            if self.cuda:
                # Allocator bookkeeping only -- safe during a graph capture.
                self.reserved_peak_sampled = max(
                    self.reserved_peak_sampled, int(torch.cuda.memory_reserved())
                )
            if self._proc is not None:
                self.rss_peak = max(self.rss_peak, int(self._proc.memory_info().rss))
            self.samples += 1
        except Exception:  # never let telemetry break a benchmark case
            pass

    def _run(self) -> None:
        while not self._stop.wait(MEMORY_SAMPLER_INTERVAL_SECONDS):
            self._sample()

    @staticmethod
    def _device_used_bytes() -> int | None:
        """Device-wide used bytes; MAIN THREAD ONLY (CUDA runtime call)."""
        try:
            free, total = torch.cuda.mem_get_info()
            return int(total - free)
        except Exception:
            return None

    def __enter__(self) -> "_PeakMemoryMonitor":
        if self.cuda:
            torch.cuda.reset_peak_memory_stats()
            self.device_used_baseline = self._device_used_bytes()
        self._sample()
        self._thread = threading.Thread(
            target=self._run, name="t4b-benchmark-memory", daemon=True
        )
        self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._sample()
        if self.cuda:
            self.device_used_end = self._device_used_bytes()
        stats: dict[str, Any] = {
            "sampler_interval_seconds": MEMORY_SAMPLER_INTERVAL_SECONDS,
            "sampler_samples": self.samples,
            "host_rss_peak_sampled_bytes": self.rss_peak or None,
            "host_peak_working_set_bytes": None,
            "host_total_bytes": None,
            "torch_cuda_max_allocated_bytes": None,
            "torch_cuda_max_reserved_bytes": None,
            "cuda_device_used_baseline_bytes": self.device_used_baseline,
            "cuda_device_used_end_bytes": self.device_used_end,
            # Process footprint sampled from allocator bookkeeping (no CUDA
            # call); the allocator's own max_memory_reserved is the exact peak.
            "torch_cuda_reserved_peak_sampled_bytes": self.reserved_peak_sampled or None,
            "cuda_device_total_bytes": None,
        }
        try:
            if self._proc is not None:
                info = self._proc.memory_info()
                stats["host_peak_working_set_bytes"] = int(
                    getattr(info, "peak_wset", None) or info.rss
                )
                stats["host_total_bytes"] = int(psutil.virtual_memory().total)
            if self.cuda:
                stats["torch_cuda_max_allocated_bytes"] = int(
                    torch.cuda.max_memory_allocated()
                )
                stats["torch_cuda_max_reserved_bytes"] = int(
                    torch.cuda.max_memory_reserved()
                )
                stats["cuda_device_total_bytes"] = int(torch.cuda.mem_get_info()[1])
                # WDDM lets the allocator reserve more than the card holds by
                # spilling into system RAM ("shared GPU memory"), which pages
                # over PCIe and stalls both CPU and GPU.  mem_get_info only
                # reports dedicated memory, so compare the allocator's reserved
                # peak against the card: the 50-zone CUDA SLSQP case committed
                # 10.8 GB on an 8.2 GB card and ran at ~1 % CPU / 10 % GPU.
                stats["cuda_oversubscribed"] = bool(
                    stats["torch_cuda_max_reserved_bytes"] > stats["cuda_device_total_bytes"]
                )
        except Exception:
            pass
        self.stats = stats
        LAST_TIMED_MEMORY.clear()
        LAST_TIMED_MEMORY.update(stats)


def memory_stats_of_last_timed() -> dict[str, Any]:
    """Peak memory recorded around the most recent ``timed`` call."""
    return dict(LAST_TIMED_MEMORY)


def _unbiased_seconds() -> float | None:
    """Seconds of interrupt time EXCLUDING system sleep/hibernate (Windows).

    ``time.perf_counter`` keeps counting through a lid-close sleep, so a case
    that spans one reports inflated seconds (the 2026-09-07 morning hybrid run
    logged 3681 s for ~12 CPU-minutes).  ``QueryUnbiasedInterruptTime`` does
    not advance during sleep; the difference between the two clocks over a
    region is the time the machine was suspended.  Returns None off Windows.
    """
    if os.name != "nt":
        return None
    try:
        import ctypes

        value = ctypes.c_ulonglong()
        if ctypes.windll.kernel32.QueryUnbiasedInterruptTime(ctypes.byref(value)):
            return value.value / 1e7  # 100 ns units
    except Exception:
        pass
    return None


# Clock record of the most recent ``timed`` call; see timing_of_last_timed().
LAST_TIMED_CLOCKS: dict[str, Any] = {}


def timing_of_last_timed() -> dict[str, Any]:
    """Wall / suspended / frozen / CPU seconds of the most recent ``timed`` call.

    ``timed`` returns ACTIVE seconds: wall-clock time minus the larger of the
    two sleep estimates below, so a case that spans a lid close reports the
    time the machine actually spent computing instead of being discarded.

    * ``suspended_seconds`` -- wall minus the unbiased interrupt clock, which
      Windows stops during classic sleep/hibernate.
    * ``frozen_seconds`` -- heartbeat gaps of the 1 Hz sampler thread, which
      also catches a Modern Standby (S0 low-power idle) freeze that the
      unbiased clock may not count as sleep.
    * ``cpu_seconds`` -- process CPU time over the region, for sanity checks.

    Both sleep estimates and the raw wall time are kept in the row, so the
    correction can be audited or redone after the fact.
    """
    return dict(LAST_TIMED_CLOCKS)


def timed(device: str, fn: Callable[[], Any]) -> tuple[Any, float]:
    with _PeakMemoryMonitor(device) as monitor:
        synchronize(device)
        unbiased_started = _unbiased_seconds()
        cpu_started = time.process_time()
        started = time.perf_counter()
        value = fn()
        synchronize(device)
        wall = time.perf_counter() - started
        cpu = time.process_time() - cpu_started
        unbiased_ended = _unbiased_seconds()
    if unbiased_started is None or unbiased_ended is None:
        suspended = None
    else:
        suspended = max(0.0, wall - (unbiased_ended - unbiased_started))
    frozen = float(monitor.frozen_seconds)
    inactive = max(suspended or 0.0, frozen)
    active = max(0.0, wall - inactive)
    LAST_TIMED_CLOCKS.clear()
    LAST_TIMED_CLOCKS.update(
        {
            "wall_seconds": wall,
            "suspended_seconds": suspended,
            "frozen_seconds": frozen,
            "inactive_seconds": inactive,
            "cpu_seconds": cpu,
            "resumed_from_sleep": inactive > 1.0,
        }
    )
    return value, active


def available_devices() -> tuple[list[str], list[dict[str, str]]]:
    if torch.cuda.is_available():
        return DEVICES.copy(), []
    return ["cpu"], [
        {
            "status": "skipped",
            "device": "cuda",
            "reason": "torch.cuda.is_available() is false",
        }
    ]


def environment_metadata(git_ref: str | None = None) -> dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        commit = None
    return {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git_ref_requested": git_ref,
        "git_commit": commit,
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "cuda_device": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
        "twin4build": getattr(tb, "__version__", "source checkout"),
        "seed": SEED,
        "timing_clock": "time.perf_counter; CUDA synchronized before and after",
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (dt.datetime, dt.date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def serialize_results(
    benchmark: str,
    config: BenchmarkConfig,
    rows: Iterable[dict[str, Any]],
    git_ref: str | None = None,
) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{_tagged(benchmark)}_{dt.datetime.now():%Y%m%d_%H%M%S}.json"
    payload = {
        "schema_version": 3,
        "benchmark": benchmark,
        "config": asdict(config),
        "environment": environment_metadata(git_ref),
        "rows": list(rows),
    }
    path.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
    return path


def checkpoint_results(
    benchmark: str,
    config: BenchmarkConfig,
    rows: Iterable[dict[str, Any]],
    git_ref: str | None = None,
) -> Path:
    """Atomically retain all completed rows after every benchmark case."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "in_progress" if config.mode == "full" else "smoke_in_progress"
    path = RESULTS_DIR / f"{_tagged(benchmark)}_{suffix}.json"
    temporary = path.with_suffix(".tmp")
    payload = {
        "schema_version": 3,
        "benchmark": benchmark,
        "checkpoint": True,
        "config": asdict(config),
        "environment": environment_metadata(git_ref),
        "rows": list(rows),
    }
    temporary.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
    temporary.replace(path)
    return path


def resume_simulation_rows(config: BenchmarkConfig) -> list[dict[str, Any]]:
    """Load only rows matching the current benchmark horizon and schema."""
    suffix = "in_progress" if config.mode == "full" else "smoke_in_progress"
    path = RESULTS_DIR / f"{_tagged('simulation_scaling')}_{suffix}.json"
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for original in payload.get("rows", []):
        if original.get("status") not in {"ok", "retained_summary"}:
            continue
        if original.get("current_publication_eligible") is False:
            rows.append(dict(original))
            continue
        if all(
            field in original
            for field in (
                "model_layout",
                "execution_mode",
                "execution_backend",
            )
        ):
            if (
                original.get("benchmark_implementation_revision")
                != BENCHMARK_IMPLEMENTATION_REVISION
            ):
                continue
            if (
                original.get("horizon_hours") == config.hours
                and original.get("step_size_seconds") == STEP_SIZE
                and original.get("n_zones") in config.zone_counts
            ):
                rows.append(dict(original))
    return rows


def resume_estimation_rows(config: BenchmarkConfig) -> list[dict[str, Any]]:
    """Load matching estimation attempts, including failed audit rows."""
    suffix = "in_progress" if config.mode == "full" else "smoke_in_progress"
    path = RESULTS_DIR / f"{_tagged('estimation_scaling')}_{suffix}.json"
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    valid_cases = set(ESTIMATION_MATRIX)
    for original in payload.get("rows", []):
        case = (
            original.get("device"),
            original.get("solver"),
            original.get("n_starts"),
        )
        if (
            original.get("status") in {"ok", "nonconverged", "failed"}
            and case in valid_cases
            and original.get("horizon_hours") == config.hours
            and original.get("step_size_seconds") == STEP_SIZE
            and original.get("n_zones") in config.zone_counts
            and original.get("solver_budget") == config.estimation_maxiter
            and original.get("model_layout") == "batched"
            and original.get("execution_mode") == "functional"
            and original.get("benchmark_implementation_revision")
            == ESTIMATION_BENCHMARK_IMPLEMENTATION_REVISION
        ):
            row = dict(original)
            if row["status"] == "failed":
                for field in (
                    "seconds_median",
                    "seconds_min",
                    "seconds_max",
                    "seconds_spread",
                ):
                    row.pop(field, None)
            rows.append(row)
    return rows


def _full_workflow_configurator() -> Callable:
    """Load the configurator from the canonical example script, not its package."""
    path = Path(tb.__file__).resolve().parent / "examples" / "full_workflow_example.py"
    spec = importlib.util.spec_from_file_location("_t4b_full_workflow_source", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load canonical full-workflow source at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.fcn


@lru_cache(maxsize=1)
def _translated_template() -> tb.Model:
    # Isolated benchmark children must not share translator output on disk.
    # A fixed generated_files model id allowed concurrent/aborted processes to
    # expose a partially written semantic model to a fresh child.
    template_directory = tempfile.TemporaryDirectory(
        prefix=f"twin4build_template_{os.getpid()}_"
    )
    _TRANSLATED_TEMPLATE_DIRECTORIES.append(template_directory)
    model = tb.Model(id="canonical_full_workflow_template")
    model.dir_conf = [template_directory.name]
    model.load(
        semantic_model_filename=example_utils.get_path(
            ["estimator_example", "one_room_example_model.xlsm"]
        ),
        fcn=_full_workflow_configurator(),
        draw_semantic_model=False,
        draw_simulation_model=False,
        verbose=0,
        enable_fusion=False,
    )
    if set(model.components) != set(ROLE_NAMES):
        missing = set(ROLE_NAMES) - set(model.components)
        extra = set(model.components) - set(ROLE_NAMES)
        raise AssertionError(
            f"Canonical topology drift: missing={missing}, extra={extra}"
        )
    if len(model.components) != CANONICAL_COMPONENTS_PER_ZONE:
        raise AssertionError("Canonical component count changed")
    if _connection_count(model) != CANONICAL_CONNECTIONS_PER_ZONE:
        raise AssertionError("Canonical connection count changed")
    return model


def _connection_count(model: tb.Model) -> int:
    return sum(
        len(connection.connects_system_at)
        for component in model.components.values()
        for connection in component.connected_through
    )


def _disconnect(sender: Any, receiver: Any, output_port: str, input_port: str) -> None:
    """Detach one copied graph edge without relying on a semantic RDF mirror."""
    connection = next(
        c for c in sender.connected_through if c.output_port == output_port
    )
    point = next(
        p
        for p in connection.connects_system_at
        if p.connection_point_of is receiver and p.input_port == input_port
    )
    connection.connects_system_at.remove(point)
    point.connects_system_through.remove(connection)
    if not connection.connects_system_at:
        sender.connected_through.remove(connection)
    if not point.connects_system_through:
        receiver.connects_at.remove(point)


def _prefix_copy(
    template: tb.Model, zone_index: int
) -> tuple[list[Any], dict[str, Any]]:
    copied = copy.deepcopy(template)
    prefix = f"zone_{zone_index}__"
    components = list(copied.components.values())
    roles = {name: copied.components[name] for name in ROLE_NAMES}
    for component in components:
        component.id = prefix + component.id
    occupancy = roles["office_occupancy"]
    occupancy.supply_damper.id = prefix + "occupancy_supply_damper"
    occupancy.exhaust_damper.id = prefix + "occupancy_exhaust_damper"
    return components, roles


def physical_zone_parameters(n_zones: int, seed: int) -> list[dict[str, float]]:
    """Distinct deterministic truth for every one of the 28 physical theta."""
    rng = np.random.default_rng(seed)
    centers = {
        "thermal.C_air": (2.8e5, 0.12),
        "thermal.C_wall": (1.2e6, 0.15),
        "wall.C": (1.0e6, 0.15),
        "thermal.R_out": (0.35, 0.18),
        "thermal.R_in": (0.12, 0.18),
        "wall.R_a": (0.04, 0.2),
        "wall.R_b": (0.04, 0.2),
        "thermal.f_wall": (0.12, 0.2),
        "thermal.f_air": (0.12, 0.2),
        "thermal.Q_occ_gain": (100.0, 0.12),
        "heater.thermalMassHeatCapacity": (3.0e4, 0.18),
        "heater.UA": (35.0, 0.18),
        "heating_pid.kp": (0.02, 0.2),
        "co2_pid.kp": (0.001, 0.2),
        "heating_pid.Ti": (45.0, 0.15),
        "co2_pid.Ti": (55.0, 0.15),
        "heating_pid.Td": (0.08, 0.2),
        "co2_pid.Td": (0.05, 0.2),
        "valve.waterFlowRateMax": (0.002, 0.18),
        "valve.valveAuthority": (0.7, 0.12),
        "supply_damper.a": (2.0, 0.15),
        "supply_damper.nominalAirFlowRate": (0.18, 0.15),
        "exhaust_damper.a": (2.2, 0.15),
        "exhaust_damper.nominalAirFlowRate": (0.17, 0.15),
        "mass.V": (65.0, 0.08),
        "mass.G_occ": (5.0e-6, 0.15),
        "mass.m_inf": (0.0015, 0.15),
        "occupancy_detector.threshold": (0.40, 0.18),
    }
    rows = []
    phases = rng.uniform(-np.pi, np.pi, len(centers))
    for i in range(n_zones):
        row = {}
        for j, (name, (center, spread)) in enumerate(centers.items()):
            # Irrational phase increments prevent repeated values at large n.
            factor = 1 + spread * np.sin(phases[j] + (i + 1) * (j + 1) * np.sqrt(2))
            row[name] = float(center * factor)
        rows.append(row)
    return rows


def _set_truth(roles: dict[str, Any], values: dict[str, float]) -> None:
    space = roles["office"]
    wall = roles["office_boundary_wall"]
    heater = roles["office_space_heater"]
    heat_pid = roles["office_temperature_heating_controller"]
    co2_pid = roles["office_co2_controller"]
    valve = roles["office_space_heater_valve"]
    supply = roles["office_supply_damper"]
    exhaust = roles["office_exhaust_damper"]
    occupancy = roles["office_occupancy"]
    detector = roles["office_occupancy_detector"]
    for key in (
        "thermal.C_air",
        "thermal.C_wall",
        "thermal.R_out",
        "thermal.R_in",
        "thermal.f_wall",
        "thermal.f_air",
        "thermal.Q_occ_gain",
        "mass.V",
        "mass.G_occ",
        "mass.m_inf",
    ):
        rgetattr(space, key).set(values[key])
    for obj, prefix, names in (
        (wall, "wall", ("C", "R_a", "R_b")),
        (heater, "heater", ("thermalMassHeatCapacity", "UA")),
        (heat_pid, "heating_pid", ("kp", "Ti", "Td")),
        (co2_pid, "co2_pid", ("kp", "Ti", "Td")),
        (valve, "valve", ("waterFlowRateMax", "valveAuthority")),
    ):
        for name in names:
            rgetattr(obj, name).set(values[f"{prefix}.{name}"])
    heater.initialize_UA = False
    for damper, internal, prefix in (
        (supply, occupancy.supply_damper, "supply_damper"),
        (exhaust, occupancy.exhaust_damper, "exhaust_damper"),
    ):
        for name in ("a", "nominalAirFlowRate"):
            value = values[f"{prefix}.{name}"]
            rgetattr(damper, name).set(value)
            rgetattr(internal, name).set(value)
    for name in ("V", "G_occ", "m_inf"):
        value = values[f"mass.{name}"]
        rgetattr(occupancy, f"mass.{name}").set(value)
    detector.threshold.set(values["occupancy_detector.threshold"])


def build_multizone_model(
    n_zones: int,
    *,
    model_id: str,
    parameter_seed: int = SEED,
    parameter_overrides: list[dict[str, float]] | None = None,
) -> tuple[tb.Model, dict[str, Any]]:
    """Replicate the complete translated full-workflow graph per zone."""
    if n_zones < 1:
        raise ValueError("n_zones must be positive")
    truth = parameter_overrides or physical_zone_parameters(n_zones, parameter_seed)
    if len(truth) != n_zones:
        raise ValueError("parameter_overrides must contain one row per zone")
    model = tb.Model(id=model_id)
    zones = []
    for i in range(n_zones):
        components, roles = _prefix_copy(_translated_template(), i)
        _set_truth(roles, truth[i])
        for component in components:
            model.add_component(component)
        zones.append(roles)
    model.load(
        draw_semantic_model=False,
        draw_simulation_model=False,
        verbose=0,
        enable_fusion=False,
    )
    expected_connections = CANONICAL_CONNECTIONS_PER_ZONE * n_zones
    if len(model.components) != CANONICAL_COMPONENTS_PER_ZONE * n_zones:
        raise AssertionError("Replicated component count collapsed")
    if _connection_count(model) != expected_connections:
        raise AssertionError("Replicated connection graph collapsed")
    return model, {
        "zone_parts": zones,
        "zones": [z["office"] for z in zones],
        "parameters": truth,
        "topology": topology_metrics(model, n_zones),
    }


def topology_metrics(model: tb.Model, n_zones: int) -> dict[str, int]:
    return {
        "n_zones": n_zones,
        "n_components": len(model.components),
        "n_connections": _connection_count(model),
        "n_states": CANONICAL_STATES_PER_ZONE * n_zones,
        "n_parameter_groups": CANONICAL_PARAMETER_GROUPS_PER_ZONE * n_zones,
        "n_theta": CANONICAL_THETA_PER_ZONE * n_zones,
        "n_measurements": CANONICAL_MEASUREMENTS_PER_ZONE * n_zones,
    }


def _align_repeated_roles_for_batch(
    model: tb.Model, zone_parts: list[dict[str, Any]]
) -> None:
    """Place equivalent non-data roles in aligned batching layers."""
    groups = model.simulation_model.execution_order
    candidates = []
    for role in ROLE_NAMES:
        components = [zone[role] for zone in zone_parts]
        if any(
            isinstance(
                component,
                (tb.SensorSystem, tb.ScheduleSystem, tb.OutdoorEnvironmentSystem),
            )
            for component in components
        ):
            continue
        source_priority = min(
            index for index, group in enumerate(groups) if components[0] in group
        )
        candidates.append((source_priority, role, components))
    all_components = {
        component for _, _, components in candidates for component in components
    }
    for group in groups:
        group[:] = [component for component in group if component not in all_components]
    for source_priority, _, components in candidates:
        groups[source_priority].extend(components)
    groups[:] = [group for group in groups if group]


def batch_model(
    model: tb.Model, *, measure: bool = True
) -> tuple[tb.Model, float | None]:
    if measure:
        batched, seconds = timed("cpu", model.batch_components)
    else:
        batched, seconds = model.batch_components(), None
    batched.load(
        draw_semantic_model=False,
        draw_simulation_model=False,
        verbose=0,
        enable_fusion=False,
    )
    return batched, seconds


def batching_mapping_audit(
    original: tb.Model, expected_ids: Iterable[str]
) -> dict[str, Any]:
    batching_mapping: dict[str, dict[str, Any]] = {}
    for component_id in expected_ids:
        info = original.get_batched_component_info(component_id)
        if info is None:
            raise AssertionError(f"Missing batched batching_mapping for {component_id}")
        meta, component_index = info
        batch_component_id = original.get_batch_id_for_component(component_id)
        if batch_component_id != meta.id:
            raise AssertionError(f"Inconsistent batch id for {component_id}")
        row = {
            "batch_component_id": batch_component_id,
            "component_index": component_index,
            "batch_size": int(meta._n_c_batched),
        }
        if not 0 <= component_index < row["batch_size"]:
            raise AssertionError(
                f"Invalid batch batching_mapping for {component_id}: {row}"
            )
        batching_mapping[component_id] = row
    return batching_mapping


def _batched_roles(
    original: tb.Model, batched: tb.Model, zone_parts: list[dict[str, Any]]
) -> dict[str, Any]:
    aliases = {}
    for role in ROLE_NAMES:
        infos = [original.get_batched_component_info(z[role].id) for z in zone_parts]
        if any(info is None for info in infos):
            raise AssertionError(f"Missing batched role {role}")
        mapped = [info[0] for info in infos]
        component_indices = [info[1] for info in infos]
        meta_ids = {meta.id for meta in mapped}
        if len(meta_ids) == 1:
            if sorted(component_indices) != list(range(len(zone_parts))):
                raise AssertionError(
                    f"Role {role} broadcast/collapse: {component_indices}"
                )
            aliases[role] = batched.components[mapped[0].id]
        elif len(meta_ids) == len(zone_parts) and all(
            i_c == 0 for i_c in component_indices
        ):
            aliases[role] = [batched.components[meta.id] for meta in mapped]
        else:
            raise AssertionError(f"Role {role} batched ambiguously: {meta_ids}")
    return aliases


def batched_parity_audit(
    n_zones: int, *, hours: int = 1, atol: float = 1e-4, rtol: float = 1e-5
) -> dict[str, Any]:
    """Check batched temperatures within 0.1 mK plus float-order tolerance."""
    original, parts = build_multizone_model(n_zones, model_id=f"parity_{n_zones}")
    _align_repeated_roles_for_batch(original, parts["zone_parts"])
    batched, batch_seconds = batch_model(original)
    _batched_roles(original, batched, parts["zone_parts"])
    _simulate(original, device="cpu", execution_mode="object", hours=hours)
    _simulate(batched, device="cpu", execution_mode="functional", hours=hours)
    max_abs = 0.0
    for zone in parts["zones"]:
        expected = zone.output["indoorTemperature"].history().detach().cpu().reshape(-1)
        meta, i_c = original.get_batched_component_info(zone.id)
        actual = (
            batched.components[meta.id]
            .output["indoorTemperature"]
            .history(i_c=i_c)
            .detach()
            .cpu()
            .reshape(-1)
        )
        max_abs = max(max_abs, float(torch.max(torch.abs(expected - actual))))
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    return {
        "status": "passed",
        "batch_seconds": batch_seconds,
        "max_abs_error": max_abs,
        "atol": atol,
        "rtol": rtol,
        **parts["topology"],
    }


def _simulate(
    model: tb.Model,
    *,
    device: str,
    execution_mode: str,
    execution_backend: str = "eager",
    hours: int,
    start: dt.datetime = START,
    step_size: int = STEP_SIZE,
) -> tuple[tb.Simulator, float]:
    model.to(device, torch.float64)
    simulator = tb.Simulator(
        model,
        execution_mode=execution_mode,
        execution_backend=execution_backend,
    )
    _, seconds = timed(
        device,
        lambda: simulator.simulate(
            start_time=start,
            end_time=start + dt.timedelta(hours=hours),
            step_size=step_size,
            show_progress_bar=False,
        ),
    )
    return simulator, seconds


def _result_counters(result: Any) -> dict[str, Any]:
    data = result if isinstance(result, dict) else getattr(result, "__dict__", {})
    iterations = data.get("nit", data.get("iterations"))
    evaluations = data.get("nfev", data.get("evaluations"))
    success = data.get("success", data.get("converged", False))
    if isinstance(success, (list, tuple, np.ndarray, torch.Tensor)):
        success = bool(np.asarray(_jsonable(success)).all())
    return {
        "iterations": _jsonable(iterations),
        "evaluations": _jsonable(evaluations),
        "converged": bool(success),
        "message": data.get("message", data.get("status")),
    }


def _repeat_summary(seconds: list[float]) -> dict[str, float]:
    median = statistics.median(seconds)
    return {
        "seconds_median": median,
        "seconds_min": min(seconds),
        "seconds_max": max(seconds),
        "seconds_spread": max(seconds) - min(seconds),
    }


def _base_metrics(
    n_zones: int, hours: int, step_size: int = STEP_SIZE
) -> dict[str, Any]:
    return {
        "horizon_hours": hours,
        "step_size_seconds": step_size,
        "timesteps": int(hours * 3600 / step_size),
        "n_zones": n_zones,
        "n_components": CANONICAL_COMPONENTS_PER_ZONE * n_zones,
        "n_states": CANONICAL_STATES_PER_ZONE * n_zones,
        "n_parameter_groups": CANONICAL_PARAMETER_GROUPS_PER_ZONE * n_zones,
        "n_theta": CANONICAL_THETA_PER_ZONE * n_zones,
        "n_measurements": CANONICAL_MEASUREMENTS_PER_ZONE * n_zones,
    }


def run_simulation_matrix(config: BenchmarkConfig) -> list[dict[str, Any]]:
    rows = resume_simulation_rows(config)
    completed = {
        (
            int(row["n_zones"]),
            row["device"],
            row["model_layout"],
            row["execution_mode"],
            row["execution_backend"],
            int(row["repetition"]),
        )
        for row in rows
        if row.get("status") == "ok" and "repetition" in row
    }
    devices, skips = available_devices()
    for n_zones in config.zone_counts:
        parity = batched_parity_audit(n_zones, hours=min(config.hours, 1))
        for case in SIMULATION_MATRIX:
            device = case["device"]
            model_layout = case["model_layout"]
            execution_mode = case["execution_mode"]
            execution_backend = case["execution_backend"]
            case_key = (
                n_zones,
                device,
                model_layout,
                execution_mode,
                execution_backend,
            )
            retained_summary = (
                RETAINED_SIMULATION_SUMMARIES.get(case_key)
                if config.mode == "full"
                else None
            )
            if retained_summary is not None:
                if not any(
                    row.get("retained_summary")
                    and (
                        row["n_zones"],
                        row["device"],
                        row["model_layout"],
                        row["execution_mode"],
                        row["execution_backend"],
                    )
                    == case_key
                    for row in rows
                ):
                    rows.append(
                        {
                            "status": "retained_summary",
                            **case,
                            **retained_summary,
                            "repetitions": config.repeats,
                            "raw_timings_available": False,
                            "retained_summary": True,
                            "benchmark_implementation_revision": (
                                HISTORICAL_BENCHMARK_IMPLEMENTATION_REVISION
                            ),
                            "current_publication_eligible": False,
                            "parity": parity,
                            **_base_metrics(n_zones, config.hours),
                        }
                    )
                    checkpoint_results("simulation_scaling", config, rows)
            if device not in devices:
                rows.append(
                    {
                        **skips[0],
                        "n_zones": n_zones,
                        **case,
                        "parity": parity,
                    }
                )
                checkpoint_results("simulation_scaling", config, rows)
                continue
            elapsed = []
            start_index = len(rows)
            for repetition in range(config.repeats):
                key = (
                    n_zones,
                    device,
                    model_layout,
                    execution_mode,
                    execution_backend,
                    repetition,
                )
                retained_seconds = (
                    RETAINED_SIMULATION_REPETITIONS.get(key)
                    if config.mode == "full"
                    else None
                )
                if retained_seconds is not None and key not in completed:
                    if not any(
                        row.get("retained_repetition")
                        and (
                            row["n_zones"],
                            row["device"],
                            row["model_layout"],
                            row["execution_mode"],
                            row["execution_backend"],
                            row.get("repetition"),
                        )
                        == key
                        for row in rows
                    ):
                        rows.append(
                            {
                                "status": "retained_summary",
                                **case,
                                "repetition": repetition,
                                "seconds": retained_seconds,
                                "retained_repetition": True,
                                "raw_timings_available": True,
                                "benchmark_implementation_revision": (
                                    HISTORICAL_BENCHMARK_IMPLEMENTATION_REVISION
                                ),
                                "current_publication_eligible": False,
                                "parity": parity,
                                "iterations": int(config.hours * 3600 / STEP_SIZE),
                                "evaluations": None,
                                "converged": True,
                                **_base_metrics(n_zones, config.hours),
                            }
                        )
                if key in completed:
                    elapsed.append(
                        next(
                            row["seconds"]
                            for row in rows
                            if (
                                row["n_zones"],
                                row["device"],
                                row["model_layout"],
                                row["execution_mode"],
                                row["execution_backend"],
                                row.get("repetition"),
                            )
                            == key
                        )
                    )
                    continue
                model, parts = build_multizone_model(
                    n_zones,
                    model_id=(
                        f"simulation_{n_zones}_{device}_{model_layout}_"
                        f"{execution_mode}_{execution_backend}_{repetition}"
                    ),
                )
                batch_seconds = None
                batching_mapping = {}
                if model_layout == "batched":
                    _align_repeated_roles_for_batch(model, parts["zone_parts"])
                    source = model
                    model, batch_seconds = batch_model(source)
                    batching_mapping = batching_mapping_audit(source, source.components)
                model.to(device, torch.float64)
                simulator = tb.Simulator(
                    model,
                    execution_mode=execution_mode,
                    execution_backend=execution_backend,
                )
                call = lambda: simulator.simulate(
                    start_time=START,
                    end_time=START + dt.timedelta(hours=config.hours),
                    step_size=STEP_SIZE,
                    show_progress_bar=False,
                )
                first_call_seconds = None
                if execution_mode == "functional":
                    _, first_call_seconds = timed(device, call)
                _, seconds = timed(device, call)
                elapsed.append(seconds)
                rows.append(
                    {
                        "status": "ok",
                        "device": device,
                        "model_layout": model_layout,
                        "execution_mode": execution_mode,
                        "execution_backend": execution_backend,
                        "repetition": repetition,
                        "seconds": seconds,
                        "memory": memory_stats_of_last_timed(),
                        "first_call_seconds": first_call_seconds,
                        "batch_seconds": batch_seconds,
                        "functional_setup_seconds": getattr(
                            simulator, "_functional_setup_seconds", None
                        ),
                        "model_initialization_seconds": getattr(
                            simulator, "_functional_initialization_seconds", None
                        ),
                        "exogenous_recording_seconds": getattr(
                            simulator, "_exogenous_recording_seconds", None
                        ),
                        "functional_rollout_seconds": getattr(
                            simulator, "_functional_rollout_seconds", None
                        ),
                        "output_materialization_seconds": getattr(
                            simulator, "_output_materialization_seconds", None
                        ),
                        "input_materialization_seconds": getattr(
                            simulator, "_input_materialization_seconds", None
                        ),
                        "validation_seconds": getattr(
                            simulator, "_functional_validation_seconds", None
                        ),
                        "validation_check_count": getattr(
                            simulator, "_functional_validation_check_count", None
                        ),
                        "validation_host_sync_count": getattr(
                            simulator,
                            "_functional_validation_host_sync_count",
                            None,
                        ),
                        "functional_materialization_revision": (
                            FUNCTIONAL_MATERIALIZATION_REVISION
                            if execution_mode == "functional"
                            else None
                        ),
                        "benchmark_implementation_revision": (
                            BENCHMARK_IMPLEMENTATION_REVISION
                        ),
                        "current_publication_eligible": True,
                        "graph_capture_seconds": getattr(
                            simulator, "_cuda_graph_capture_seconds", None
                        ),
                        "graph_replay_seconds": getattr(
                            simulator, "_cuda_graph_replay_seconds", None
                        ),
                        "graph_capture_count": getattr(
                            simulator, "_cuda_graph_capture_count", 0
                        ),
                        "graph_replay_count": getattr(
                            simulator, "_cuda_graph_replay_count", 0
                        ),
                        "batching_mapping": batching_mapping,
                        "parity": parity,
                        "iterations": len(simulator.date_time_steps[0]),
                        "evaluations": None,
                        "converged": True,
                        **_base_metrics(n_zones, config.hours),
                    }
                )
                completed.add(key)
                checkpoint_results("simulation_scaling", config, rows)
            if elapsed:
                summary = _repeat_summary(elapsed)
                for row in rows[start_index:]:
                    if (
                        row.get("model_layout") == model_layout
                        and row.get("execution_mode") == execution_mode
                        and row.get("execution_backend") == execution_backend
                        and row.get("n_zones") == n_zones
                        and row.get("current_publication_eligible", True)
                    ):
                        row.update(summary)
            checkpoint_results("simulation_scaling", config, rows)
    return rows


def _parameter_entries(roles: dict[str, Any], vector: Callable[[str, float], Any]):
    s = roles["office"]
    w = roles["office_boundary_wall"]
    h = roles["office_space_heater"]
    hp = roles["office_temperature_heating_controller"]
    cp = roles["office_co2_controller"]
    v = roles["office_space_heater_valve"]
    sd = roles["office_supply_damper"]
    ed = roles["office_exhaust_damper"]
    occ = roles["office_occupancy"]
    det = roles["office_occupancy_detector"]
    return [
        (s, "thermal.C_air", vector("thermal.C_air", 5e5), 1e4, 5e5),
        (s, "thermal.C_wall", vector("thermal.C_wall", 1e6), 1e5, 3e6),
        (w, "C", vector("wall.C", 1e6), 1e4, 1e7),
        (s, "thermal.R_out", vector("thermal.R_out", 0.5), 0.01, 1),
        (s, "thermal.R_in", vector("thermal.R_in", 0.1), 0.01, 1),
        (w, "R_a", vector("wall.R_a", 0.04), 1e-4, 1),
        (w, "R_b", vector("wall.R_b", 0.04), 1e-4, 1),
        (s, "thermal.f_wall", vector("thermal.f_wall", 0.1), 0, 10),
        (s, "thermal.f_air", vector("thermal.f_air", 0.1), 0, 10),
        (s, "thermal.Q_occ_gain", vector("thermal.Q_occ_gain", 100), 10, 200),
        (
            h,
            "thermalMassHeatCapacity",
            vector("heater.thermalMassHeatCapacity", 1e4),
            1e3,
            2e5,
        ),
        (h, "UA", vector("heater.UA", 30), 1, 100),
        (hp, "kp", vector("heating_pid.kp", 0.005), 1e-5, 1, "private"),
        (cp, "kp", vector("co2_pid.kp", 0.0001), 1e-5, 1, "private"),
        ([hp, cp], "Ti", vector("heating_pid.Ti", 30), 1, 300, "private"),
        ([hp, cp], "Td", vector("heating_pid.Td", 0.01), 0, 1, "private"),
        (v, "waterFlowRateMax", vector("valve.waterFlowRateMax", 0.001), 1e-6, 0.1),
        (v, "valveAuthority", vector("valve.valveAuthority", 0.8), 0.4, 1),
        ([sd, occ.supply_damper], "a", vector("supply_damper.a", 1), 1, 10, "shared"),
        (
            [sd, occ.supply_damper],
            "nominalAirFlowRate",
            vector("supply_damper.nominalAirFlowRate", 0.1),
            1e-5,
            1,
            "shared",
        ),
        ([ed, occ.exhaust_damper], "a", vector("exhaust_damper.a", 1), 1, 10, "shared"),
        (
            [ed, occ.exhaust_damper],
            "nominalAirFlowRate",
            vector("exhaust_damper.nominalAirFlowRate", 0.1),
            1e-5,
            1,
            "shared",
        ),
        ([s, occ], "mass.V", vector("mass.V", 65), 50, 80, "shared"),
        ([s, occ], "mass.G_occ", vector("mass.G_occ", 1e-6), 1e-6, 1e-5, "shared"),
        ([s, occ], "mass.m_inf", vector("mass.m_inf", 0.001), 1e-4, 0.01, "shared"),
        (det, "threshold", vector("occupancy_detector.threshold", 1.0), 0.02, 5.0),
    ]


MEASUREMENT_ROLES = (
    ("office_valve_position_sensor", 0.025),
    ("office_temperature_sensor", 0.05),
    ("office_damper_position_sensor", 0.025),
    ("office_co2_sensor", 15.0),
)
ESTIMATION_SIGNAL_NAMES = ("valve", "temperature", "damper", "co2")
ESTIMATION_QUALITY_QUANTILES = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)


def generate_estimation_data(
    n_zones: int, config: BenchmarkConfig
) -> tuple[dict[str, pd.DataFrame], list[dict[str, float]]]:
    truth = physical_zone_parameters(n_zones, config.seed + 1000)
    model, parts = build_multizone_model(
        n_zones,
        model_id=f"true_estimation_{n_zones}",
        parameter_overrides=truth,
    )
    simulator, _ = _simulate(
        model, device="cpu", execution_mode="object", hours=config.hours
    )
    rng = np.random.default_rng(config.seed + 2000)
    data = {}
    for zone in parts["zone_parts"]:
        for role, sd in MEASUREMENT_ROLES:
            sensor = zone[role]
            values = (
                sensor.output["measuredValue"].history()[:, 0, 0].detach().cpu().numpy()
            )
            noise = rng.normal(0.0, sd, len(values))
            data[sensor.id] = pd.DataFrame(
                {"value": values + noise}, index=simulator.date_time_steps[0]
            )
    return data, truth


def _estimation_problem(n_zones: int, config: BenchmarkConfig, *, batched: bool):
    data, truth = generate_estimation_data(n_zones, config)
    fit_initial = physical_zone_parameters(n_zones, config.seed + 3000)
    model, parts = build_multizone_model(
        n_zones,
        model_id=f"fit_estimation_{n_zones}",
        parameter_overrides=fit_initial,
    )
    for zone in parts["zone_parts"]:
        for role, _ in MEASUREMENT_ROLES:
            source_id = zone[role].id.replace("fit_estimation_", "true_estimation_")
            # Prefixes are identical; model id is not part of component ids.
            zone[role].df = data[zone[role].id]
    if not batched:
        if n_zones != 1:
            raise ValueError("Unbatched canonical setup is only used for one zone")
        roles = parts["zone_parts"][0]
        parameters = _parameter_entries(roles, lambda _name, default: default)
        measurements = [(roles[role], sd) for role, sd in MEASUREMENT_ROLES]
        return {
            "model": model,
            "parameters": parameters,
            "measurements": measurements,
            "truth": truth,
            "truth_lookup": {},
            "batch_seconds": None,
            "topology": parts["topology"],
            "batching_mapping": {},
        }
    _align_repeated_roles_for_batch(model, parts["zone_parts"])
    batched_model, batch_seconds = batch_model(model)
    roles = _batched_roles(model, batched_model, parts["zone_parts"])
    parameters = _parameter_entries(
        roles,
        lambda _name, default: [default] * n_zones,
    )
    measurements = []
    for zone in parts["zone_parts"]:
        for role, sd in MEASUREMENT_ROLES:
            meta, component_index = model.get_batched_component_info(zone[role].id)
            if component_index != 0:
                raise AssertionError(f"{role} data source must remain independent")
            measurements.append((batched_model.components[meta.id], sd))
    batching_mapping = batching_mapping_audit(model, model.components)
    return {
        "model": batched_model,
        "parameters": parameters,
        "measurements": measurements,
        "truth": truth,
        "batch_seconds": batch_seconds,
        "topology": parts["topology"],
        "batching_mapping": batching_mapping,
    }


def batched_estimation_problem(
    n_zones: int, config: BenchmarkConfig, *, measure_batch: bool = True
):
    setup = _estimation_problem(n_zones, config, batched=True)
    if not measure_batch:
        setup["batch_seconds"] = None
    setup["batched_model"] = setup["model"]
    setup["parameter_groups"] = [
        {"component": entry[0], "attribute": entry[1]} for entry in setup["parameters"]
    ]
    return setup


def recovery_metrics(
    result: dict[str, Any], truth: list[dict[str, float]]
) -> dict[str, Any]:
    estimated = np.asarray(result.get("result_x", []), dtype=float).reshape(-1)
    expected = np.asarray(
        [row[key] for key in THETA_KEYS for row in truth], dtype=float
    )
    n = min(len(estimated), len(expected))
    if n == 0:
        return {
            "parameter_rmse": np.nan,
            "parameter_mape": np.nan,
            "recovered_theta": 0,
        }
    error = estimated[:n] - expected[:n]
    scale = np.maximum(np.abs(expected[:n]), 1e-12)
    return {
        "parameter_rmse": float(np.sqrt(np.mean(error**2))),
        "parameter_mape": float(np.mean(np.abs(error) / scale)),
        "recovered_theta": n,
        "fit_objective": float(result.get("final_objective", np.nan)),
    }


def _quantiles(values: np.ndarray) -> dict[str, float]:
    return {
        f"q{int(q * 100):02d}": float(np.quantile(values, q))
        for q in ESTIMATION_QUALITY_QUANTILES
    }


def _postfit_prediction_quality(
    estimator: tb.Estimator,
    config: BenchmarkConfig,
    *,
    n_zones: int,
    device: str,
    solver: str,
    n_starts: int,
    repetition: int,
) -> dict[str, Any]:
    """Run an untimed post-fit rollout and retain every prediction/observation."""
    simulator = estimator.simulator
    configured_backend = simulator.execution_backend
    try:
        # Prediction quality is intentionally outside the solver timing.  Eager
        # execution avoids invalidating or re-capturing a solver CUDA graph.
        simulator.simulate(
            start_time=START,
            end_time=START + dt.timedelta(hours=config.hours),
            step_size=STEP_SIZE,
            show_progress_bar=False,
            execution_mode="functional",
            execution_backend="eager",
        )
    finally:
        # The public per-call override is non-mutating today.  Keep this guard
        # so benchmark behavior remains stable if Simulator ever changes it.
        simulator.execution_backend = configured_backend
    measurements = estimator._measurements
    expected_columns = n_zones * len(ESTIMATION_SIGNAL_NAMES)
    if len(measurements) != expected_columns:
        raise AssertionError(
            f"Expected {expected_columns} measurement columns, got {len(measurements)}"
        )

    predicted_columns = []
    observed_columns = []
    for measuring_device, _ in measurements:
        predicted_columns.append(
            measuring_device.input["measuredValue"]
            .history()[:, 0, 0]
            .detach()
            .cpu()
            .numpy()
        )
        observed_columns.append(
            np.asarray(
                estimator.actual_readings[measuring_device.id][0].to_numpy(),
                dtype=np.float64,
            ).reshape(-1)
        )
    n_timesteps = min(
        min(len(values) for values in predicted_columns),
        min(len(values) for values in observed_columns),
    )
    predicted = np.stack(
        [values[:n_timesteps] for values in predicted_columns], axis=1
    ).reshape(n_timesteps, n_zones, len(ESTIMATION_SIGNAL_NAMES))
    observed = np.stack(
        [values[:n_timesteps] for values in observed_columns], axis=1
    ).reshape(n_timesteps, n_zones, len(ESTIMATION_SIGNAL_NAMES))
    error = predicted - observed
    score_start = min(20, max(0, n_timesteps - 1))
    scored_prediction = predicted[score_start:]
    scored_observation = observed[score_start:]
    scored_error = error[score_start:]

    quality = {}
    for signal_index, signal in enumerate(ESTIMATION_SIGNAL_NAMES):
        prediction_values = scored_prediction[..., signal_index].reshape(-1)
        observation_values = scored_observation[..., signal_index].reshape(-1)
        error_values = scored_error[..., signal_index].reshape(-1)
        absolute_error = np.abs(error_values)
        per_zone_rmse = np.sqrt(
            np.mean(scored_error[..., signal_index] ** 2, axis=0)
        )
        quality[signal] = {
            "prediction_quantiles": _quantiles(prediction_values),
            "observation_quantiles": _quantiles(observation_values),
            "error_quantiles": _quantiles(error_values),
            "absolute_error_quantiles": _quantiles(absolute_error),
            "mae": float(np.mean(absolute_error)),
            "rmse": float(np.sqrt(np.mean(error_values**2))),
            "bias": float(np.mean(error_values)),
            "per_zone_rmse_quantiles": _quantiles(per_zone_rmse),
        }

    artifact_dir = RESULTS_DIR / "estimation_predictions"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact = artifact_dir / (
        f"zones_{n_zones}_{device}_{solver}_starts_{n_starts}_rep_{repetition}.npz"
    )
    temporary_artifact = artifact.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary_artifact,
        predicted=predicted,
        observed=observed,
        error=error,
        signal_names=np.asarray(ESTIMATION_SIGNAL_NAMES),
        timestamps=np.asarray(
            [
                value.isoformat()
                for value in estimator.simulator.date_time_steps[0][:n_timesteps]
            ]
        ),
        score_start=np.asarray(score_start, dtype=np.int64),
    )
    temporary_artifact.replace(artifact)
    return {
        "prediction_artifact": str(artifact),
        "prediction_shape": list(predicted.shape),
        "quality_score_start": score_start,
        "quality_scored_points_per_signal": int(
            scored_prediction.shape[0] * n_zones
        ),
        "prediction_quality": quality,
    }


def _clip_parameter_x0(values: list[float], x0: Any, lo: float, hi: float):
    """Keep a fitted vector inside the estimate() box, matching the input shape."""
    eps = 1e-9 * (hi - lo)

    def _clip(value: float) -> float:
        return min(max(float(value), lo + eps), hi - eps)

    if isinstance(x0, (list, tuple)):
        if len(values) == len(x0):
            return [_clip(value) for value in values]
        if len(values) == 1:
            return [_clip(values[0]) for _ in x0]
        raise ValueError(
            f"result_x slice has {len(values)} values but x0 length {len(x0)}"
        )
    return _clip(values[0])


def _result_x_by_component(result: dict[str, Any]) -> dict[tuple[str, str], list[float]]:
    """Map ``(component_id, attr)`` to the denormalized ``result_x`` slice."""
    required = ("result_x", "component_id", "component_attr", "theta_mask")
    missing = [key for key in required if key not in result]
    if missing:
        raise ValueError(
            "EstimationResult is missing "
            + ", ".join(missing)
            + "; cannot warm-start the next stage from the solver contract"
        )
    result_x = np.asarray(result["result_x"], dtype=float).reshape(-1)
    widths = result.get("unique_param_n_c")
    if widths is None:
        slices = result.get("theta_slices")
        if not slices:
            raise ValueError(
                "EstimationResult needs unique_param_n_c or theta_slices "
                "to unpack result_x"
            )
        widths = [int(end) - int(start) for start, end in slices]
    unique_values: dict[int, list[float]] = {}
    offset = 0
    for unique_idx, width in enumerate(widths):
        width = int(width)
        unique_values[int(unique_idx)] = result_x[offset : offset + width].tolist()
        offset += width
    if offset != len(result_x):
        raise ValueError(
            f"result_x length {len(result_x)} does not match unique_param_n_c "
            f"sum {offset}"
        )
    lookup = {}
    for component_id, attr, unique_idx in zip(
        result["component_id"],
        result["component_attr"],
        np.asarray(result["theta_mask"]).reshape(-1),
    ):
        lookup[(str(component_id), str(attr))] = unique_values[int(unique_idx)]
    return lookup


def _parameter_entries_from_estimation_result(
    entries: list[tuple], result: dict[str, Any]
) -> list[tuple]:
    """Rebuild estimate() parameter tuples from an EstimationResult.

    ``result_x`` is the fitted theta.  ``component_id``, ``component_attr``,
    ``theta_mask``, and ``unique_param_n_c`` map each input group back onto
    that vector after the estimator's private/shared expansion.  Private
    lists are expanded so each component keeps its own unique slice.
    """
    lookup = _result_x_by_component(result)
    updated = []
    for entry in entries:
        comps, attr, x0, lo, hi = entry[:5]
        extra = entry[5:]
        parameter_type = extra[0] if extra else "private"
        component_list = comps if isinstance(comps, list) else [comps]
        if parameter_type == "private" and len(component_list) > 1:
            for component in component_list:
                key = (component.id, attr)
                if key not in lookup:
                    raise KeyError(
                        f"{component.id}.{attr} is missing from EstimationResult"
                    )
                fitted = _clip_parameter_x0(lookup[key], x0, lo, hi)
                updated.append((component, attr, fitted, lo, hi, *extra))
            continue
        key = (component_list[0].id, attr)
        if key not in lookup:
            raise KeyError(
                f"{component_list[0].id}.{attr} is missing from EstimationResult"
            )
        fitted = _clip_parameter_x0(lookup[key], x0, lo, hi)
        updated.append((comps, attr, fitted, lo, hi, *extra))
    return updated


def _estimation_window(config: BenchmarkConfig) -> dict[str, Any]:
    return {
        "start_time": [START],
        "end_time": [START + dt.timedelta(hours=config.hours)],
        "step_size": [STEP_SIZE],
        "n_warmup": 20,
    }


def _make_estimation_estimator(model: Any, device: str) -> tb.Estimator:
    return tb.Estimator(
        tb.Simulator(
            model,
            execution_mode="functional",
            execution_backend="cuda_graph" if device == "cuda" else "eager",
        )
    )


def _run_slsqp5_then_collocation(
    setup: dict[str, Any],
    config: BenchmarkConfig,
    device: str,
) -> tuple[dict[str, Any], float, tb.Estimator]:
    if device != "cuda":
        raise ValueError("SLSQP5+collocation is CUDA-only in this suite")
    model = setup["model"]
    model.to(device, torch.float64)
    estimator = _make_estimation_estimator(model, device)
    slsqp_iters = min(
        ESTIMATION_SLSQP_WARMSTART_ITERS, config.estimation_maxiter
    )
    window = _estimation_window(config)

    def stage1() -> Any:
        return estimator.estimate(
            parameters=setup["parameters"],
            measurements=setup["measurements"],
            method=("scipy", "SLSQP", "ad"),
            options={"maxiter": slsqp_iters},
            **window,
        )

    result1, seconds1 = timed(device, stage1)
    estimator.simulator.close()
    parameters2 = _parameter_entries_from_estimation_result(
        setup["parameters"], result1
    )

    def stage2() -> Any:
        return estimator.estimate(
            parameters=parameters2,
            measurements=setup["measurements"],
            method=("casadi", "ipopt", "ad", "collocation"),
            options={
                "maxiter": config.estimation_maxiter,
                "hessian": "exact",
                "boundary_state_init": "rollout",
                "early_stopping": False,
            },
            **window,
        )

    result2, seconds2 = timed(device, stage2)
    counters1 = _result_counters(result1)
    result2["slsqp_warmstart_iters"] = slsqp_iters
    result2["boundary_state_init"] = "rollout"
    result2["stage1_seconds"] = seconds1
    result2["stage2_seconds"] = seconds2
    result2["stage1_iterations"] = counters1["iterations"]
    result2["stage1_evaluations"] = counters1["evaluations"]
    result2["stage1_success"] = counters1["converged"]
    result2["stage1_message"] = counters1["message"]
    result2["stage1_final_objective"] = (
        result1.get("final_objective")
        if isinstance(result1, dict)
        else getattr(result1, "final_objective", None)
    )
    return result2, seconds1 + seconds2, estimator


def _run_estimation(
    setup: dict[str, Any],
    config: BenchmarkConfig,
    device: str,
    method: tuple,
    n_starts: int,
) -> tuple[dict[str, Any], float, tb.Estimator]:
    if method[1] == "slsqp5-ipopt-collocation":
        return _run_slsqp5_then_collocation(setup, config, device)
    model = setup["model"]
    model.to(device, torch.float64)
    estimator = _make_estimation_estimator(model, device)
    options = {"maxiter": config.estimation_maxiter}
    if method[-1] == "collocation":
        if device != "cuda":
            raise ValueError("IPOPT collocation is CUDA-only in this suite")
        options["hessian"] = "exact"
    if method[0] == "custom":
        options.update(
            {
                "n_starts": n_starts,
                "batch_size": n_starts,
                "start_seed": config.seed,
                "start_strategy": "local",
                "start_spread": 0.1,
            }
        )
        if method[1] == "batched-sqp":
            n_zones = int(setup["topology"]["n_zones"])
            options["sqp_curvature_blocks"] = [
                [zone + group * n_zones for group in range(CANONICAL_THETA_PER_ZONE)]
                for zone in range(n_zones)
            ]
    result, seconds = timed(
        device,
        lambda: estimator.estimate(
            parameters=setup["parameters"],
            measurements=setup["measurements"],
            method=method,
            options=options,
            **_estimation_window(config),
        ),
    )
    return result, seconds, estimator


def run_estimation_matrix(config: BenchmarkConfig) -> list[dict[str, Any]]:
    """Compatibility alias for the authoritative estimation scaling matrix."""
    return run_estimation_scaling(config)


def _collocation_preflight(n_zones: int, hours: int) -> dict[str, Any]:
    n_steps = int(hours * 3600 / STEP_SIZE)
    d_zone = CANONICAL_AUGMENTED_STATES_PER_ZONE
    d_global = CANONICAL_GLOBAL_AUGMENTED_STATES
    p_zone = CANONICAL_THETA_PER_ZONE
    p_global = 0
    d_total = d_global + d_zone * n_zones
    n_state_variables = d_total * n_steps
    n_nlp_variables = n_state_variables + CANONICAL_THETA_PER_ZONE * n_zones
    n_links = max(0, n_steps - 1)
    jacobian_nnz = n_links * (
        d_global * (p_global + d_global + 1)
        + n_zones * d_zone * (p_global + p_zone + d_global + d_zone + 1)
    )
    theta_nnz = p_global * (p_global + 1) // 2 + n_zones * (
        p_global * p_zone + p_zone * (p_zone + 1) // 2
    )
    segment_hessian_nnz = (
        p_global * d_global
        + n_zones * p_global * d_zone
        + n_zones * p_zone * d_global
        + n_zones * p_zone * d_zone
        + d_global * (d_global + 1) // 2
        + n_zones * d_global * d_zone
        + n_zones * d_zone * (d_zone + 1) // 2
    )
    hessian_nnz = theta_nnz + n_steps * segment_hessian_nnz
    # Two int64 coordinates plus one float64 value per declared nonzero.
    jacobian_bytes = jacobian_nnz * 24
    hessian_bytes = hessian_nnz * 24
    storage_safe = (
        jacobian_bytes <= MAX_PARETO_DENSE_JACOBIAN_BYTES
        and hessian_bytes <= MAX_COLLOCATION_DENSE_HESSIAN_BYTES
    )
    implementation_supported = True
    # Empirical VRAM model against the actual card (the storage count above
    # under-estimated the 50-zone case by two orders of magnitude).
    estimated_vram_bytes = int(
        (COLLOCATION_VRAM_GIB_BASE + COLLOCATION_VRAM_GIB_PER_ZONE * n_zones) * 1024**3
    )
    device_total_bytes = None
    if torch.cuda.is_available():
        try:
            device_total_bytes = int(torch.cuda.mem_get_info()[1])
        except Exception:
            device_total_bytes = None
    vram_safe = (
        device_total_bytes is None
        or estimated_vram_bytes <= COLLOCATION_VRAM_SAFETY_FRACTION * device_total_bytes
    )
    safe = storage_safe and vram_safe
    if not storage_safe:
        reason = (
            "replica-sparse estimator collocation storage exceeds "
            "the configured safety cap"
        )
    elif not vram_safe:
        reason = (
            f"estimated peak VRAM {estimated_vram_bytes / 1024**3:.1f} GiB "
            f"({COLLOCATION_VRAM_GIB_PER_ZONE} GiB/zone, measured on A100) exceeds "
            f"{COLLOCATION_VRAM_SAFETY_FRACTION:.0%} of the "
            f"{device_total_bytes / 1024**3:.1f} GiB card"
        )
    else:
        reason = None
    return {
        "estimated_peak_vram_bytes": estimated_vram_bytes,
        "cuda_device_total_bytes": device_total_bytes,
        "vram_safe": vram_safe,
        "preflight_kind": "collocation_hessian_exact",
        "n_collocation_state_variables": n_state_variables,
        "n_collocation_nlp_variables": n_nlp_variables,
        "replica_sparse_jacobian_nnz": jacobian_nnz,
        "replica_sparse_hessian_nnz": hessian_nnz,
        "replica_sparse_jacobian_storage_bytes": jacobian_bytes,
        "replica_sparse_hessian_storage_bytes": hessian_bytes,
        "replica_count": n_zones,
        "replica_theta_width": p_zone,
        "global_theta_width": p_global,
        "replica_state_width": d_zone,
        "global_state_width": d_global,
        "implementation_supported": implementation_supported,
        "safety_limit_bytes": MAX_COLLOCATION_DENSE_HESSIAN_BYTES,
        "mathematically_safe": safe,
        "preflight_reason": reason,
    }


def _estimation_case_base(
    config: BenchmarkConfig,
    n_zones: int,
    device: str,
    solver: str,
    n_starts: int,
) -> dict[str, Any]:
    preflight = (
        _collocation_preflight(n_zones, config.hours)
        if solver in COLLOCATION_SOLVERS
        else {}
    )
    if solver in ("custom-batched-sqp", "custom-batched-tr"):
        solver_variant = f"{n_starts}-start"
        budget_basis = (
            "run to native convergence with maxiter=300; fixed across scaling sizes"
        )
    elif solver == "slsqp5-ipopt-collocation":
        solver_variant = "slsqp5-then-collocation"
        budget_basis = (
            "5 CUDA Graph SLSQP iterations, then IPOPT collocation to native "
            "convergence or maxiter=300; rollout boundary-state init"
        )
    else:
        solver_variant = "single-start"
        budget_basis = (
            "run to native convergence with maxiter=300; fixed across scaling sizes"
        )
    return {
        "device": device,
        "model_layout": "batched",
        "execution_mode": "functional",
        "execution_backend": "cuda_graph" if device == "cuda" else "eager",
        # Simulator(compile_step="auto") compiles the functional step where the
        # torch build has Triton (Linux wheels); rows from such runs are not
        # comparable with eager-step rows, so record which one this is.
        "step_compiled": bool(
            tb.Simulator(None, execution_mode="functional").step_compilation_active(device)
        ),
        "solver": solver,
        "solver_variant": solver_variant,
        "n_starts": n_starts,
        "method": ESTIMATION_METHODS[solver],
        "solver_budget": config.estimation_maxiter,
        "benchmark_implementation_revision": (
            ESTIMATION_BENCHMARK_IMPLEMENTATION_REVISION
        ),
        "budget_basis": budget_basis,
        **_base_metrics(n_zones, config.hours),
        **preflight,
    }


def run_one_estimation_case(
    config: BenchmarkConfig,
    *,
    n_zones: int,
    device: str,
    solver: str,
    n_starts: int,
    repetition: int,
) -> dict[str, Any]:
    """Execute one estimation attempt; intended for the isolated child CLI."""
    seed_everything(config.seed)
    base = _estimation_case_base(config, n_zones, device, solver, n_starts)
    setup = batched_estimation_problem(n_zones, config)
    result, seconds, estimator = _run_estimation(
        setup, config, device, ESTIMATION_METHODS[solver], n_starts
    )
    counters = _result_counters(result)
    stage_fields = {
        key: result[key]
        for key in (
            "slsqp_warmstart_iters",
            "boundary_state_init",
            "stage1_seconds",
            "stage2_seconds",
            "stage1_iterations",
            "stage1_evaluations",
            "stage1_success",
            "stage1_message",
            "stage1_final_objective",
        )
        if isinstance(result, dict) and key in result
    }
    prediction_quality = _postfit_prediction_quality(
        estimator,
        config,
        n_zones=n_zones,
        device=device,
        solver=solver,
        n_starts=n_starts,
        repetition=repetition,
    )
    return {
        **base,
        "status": "ok" if counters["converged"] else "nonconverged",
        "repetition": repetition,
        # ``seconds`` is ACTIVE time (sleep / freeze subtracted); the raw
        # clocks are kept alongside so the correction is auditable.
        "seconds": seconds,
        **timing_of_last_timed(),
        "memory": memory_stats_of_last_timed(),
        "batch_seconds": setup["batch_seconds"],
        "n_warmup": 20,
        "batching_mapping": setup["batching_mapping"],
        "ground_truth": setup["truth"],
        **counters,
        **stage_fields,
        **recovery_metrics(result, setup["truth"]),
        **prediction_quality,
        "quality_label": (
            "converged" if counters["converged"] else "fixed-budget-nonconverged"
        ),
        "raw_result": result,
        "child_pid": os.getpid(),
        "seed": config.seed,
    }


def _run_estimation_case_subprocess(
    config: BenchmarkConfig,
    *,
    n_zones: int,
    device: str,
    solver: str,
    n_starts: int,
    repetition: int,
) -> dict[str, Any]:
    """Run one case in a fresh interpreter and normalize all child failures."""
    base = _estimation_case_base(config, n_zones, device, solver, n_starts)
    request = {
        "config": asdict(config),
        "n_zones": n_zones,
        "device": device,
        "solver": solver,
        "n_starts": n_starts,
        "repetition": repetition,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    attempt_started = time.perf_counter()
    with tempfile.TemporaryDirectory(
        prefix="estimation_case_", dir=RESULTS_DIR
    ) as temporary_dir:
        temporary_path = Path(temporary_dir)
        request_path = temporary_path / "request.json"
        result_path = temporary_path / "result.json"
        request_path.write_text(json.dumps(request), encoding="utf-8")
        environment = os.environ.copy()
        environment["PYTHONHASHSEED"] = str(config.seed)
        command = [
            sys.executable,
            "-m",
            ESTIMATION_CASE_MODULE,
            "--request",
            str(request_path),
            "--result",
            str(result_path),
        ]
        try:
            completed = subprocess.run(
                command,
                cwd=Path(__file__).resolve().parents[1],
                env=environment,
                text=True,
                check=False,
            )
        except Exception as exc:
            return {
                **base,
                "status": "failed",
                "repetition": repetition,
                "seconds": time.perf_counter() - attempt_started,
                "quality_label": "failed",
                "error": f"could not start estimation child: {exc!r}",
                "child_pid": None,
                "child_returncode": None,
                "child_stderr": "",
                "seed": config.seed,
            }
        process_seconds = time.perf_counter() - attempt_started
        envelope = None
        if result_path.exists():
            try:
                envelope = json.loads(result_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                envelope = None
        if (
            completed.returncode == 0
            and isinstance(envelope, dict)
            and envelope.get("ok") is True
            and isinstance(envelope.get("row"), dict)
            and envelope["row"].get("status") in {"ok", "nonconverged"}
        ):
            return envelope["row"]
        error = (
            envelope.get("error")
            if isinstance(envelope, dict) and envelope.get("error")
            else f"estimation child exited with code {completed.returncode}"
        )
        return {
            **base,
            "status": "failed",
            "repetition": repetition,
            "seconds": process_seconds,
            "quality_label": "failed",
            "error": error,
            "child_pid": (
                envelope.get("child_pid") if isinstance(envelope, dict) else None
            ),
            "child_returncode": completed.returncode,
            "child_stderr": (completed.stderr or "")[-4000:],
            # The child writes its own traceback into result.json; without it
            # a CUDA fault only leaves the one-line message in the row.
            "child_traceback": (
                (envelope.get("traceback") or "")[-6000:]
                if isinstance(envelope, dict)
                else ""
            ),
            "seed": config.seed,
        }


def run_estimation_scaling(config: BenchmarkConfig) -> list[dict[str, Any]]:
    rows = resume_estimation_rows(config)
    completed_cases = {
        (
            int(row["n_zones"]),
            row["device"],
            row["solver"],
            int(row["n_starts"]),
            int(row.get("repetition", 0)),
        )
        for row in rows
        if row.get("status") in {"ok", "nonconverged"}
    }
    devices, skips = available_devices()
    deferred = {
        name.strip()
        for name in os.environ.get(ESTIMATION_DEFER_SOLVERS_ENV, "").split(",")
        if name.strip()
    }
    if deferred:
        print(f"[estimation_scaling] deferring solvers: {sorted(deferred)}", flush=True)
    deferred_devices = {
        name.strip()
        for name in os.environ.get(ESTIMATION_DEFER_DEVICES_ENV, "").split(",")
        if name.strip()
    }
    if deferred_devices:
        print(f"[estimation_scaling] deferring devices: {sorted(deferred_devices)}", flush=True)
    for n_zones in config.zone_counts:
        for device, solver, n_starts in ESTIMATION_MATRIX:
            if solver in deferred or device in deferred_devices:
                continue
            base = _estimation_case_base(
                config, n_zones, device, solver, n_starts
            )
            preflight = (
                _collocation_preflight(n_zones, config.hours)
                if solver in COLLOCATION_SOLVERS
                else {}
            )
            if device not in devices:
                rows.append({**skips[0], **base})
                checkpoint_results("estimation_scaling", config, rows)
                continue
            if device == "cpu" and n_zones > ESTIMATION_CPU_MAX_ZONES:
                rows.append(
                    {
                        **base,
                        "status": "skipped",
                        "reason": (
                            f"cpu arm capped at {ESTIMATION_CPU_MAX_ZONES} zones "
                            "(loses to every CUDA arm at 10 zones)"
                        ),
                    }
                )
                checkpoint_results("estimation_scaling", config, rows)
                continue
            if preflight and not preflight["mathematically_safe"]:
                rows.append(
                    {
                        **base,
                        "status": "skipped",
                        "reason": preflight["preflight_reason"],
                    }
                )
                checkpoint_results("estimation_scaling", config, rows)
                continue
            for repetition in range(config.estimation_repeats):
                case_key = (n_zones, device, solver, n_starts, repetition)
                if case_key in completed_cases:
                    continue
                row = _run_estimation_case_subprocess(
                    config,
                    n_zones=n_zones,
                    device=device,
                    solver=solver,
                    n_starts=n_starts,
                    repetition=repetition,
                )
                rows.append(row)
                if row.get("status") in {"ok", "nonconverged"}:
                    row.update(_repeat_summary([row["seconds"]]))
                    completed_cases.add(case_key)
                checkpoint_results("estimation_scaling", config, rows)
    return rows


def _optimization_problem(n_zones: int, seed: int, *, batch_it: bool = False):
    truth = physical_zone_parameters(n_zones, seed + 1000)
    model, parts = build_multizone_model(
        n_zones, model_id=f"optimization_{n_zones}", parameter_overrides=truth
    )
    price_path = example_utils.get_path(["estimator_example", "electricity_price.csv"])
    shared_valve_schedule = tb.ScheduleSystem(
        weekday_ruleset={
            "ruleset_default_value": 0,
            "ruleset_start_minute": [0, 0],
            "ruleset_end_minute": [0, 0],
            "ruleset_start_hour": [8, 19],
            "ruleset_end_hour": [16, 20],
            "ruleset_value": [0.5, 0.5],
        },
        id="shared_valve_position_schedule",
    )
    opt_parts = []
    for i, zone in enumerate(parts["zone_parts"]):
        prefix = f"zone_{i}__"
        price = tb.ScheduleSystem(
            filename=price_path,
            date_column=0,
            value_column=1,
            id=prefix + "price_schedule",
        )
        cost = tb.ScalarProductSystem(
            scale_factor=STEP_SIZE / 3600 / 1000, id=prefix + "costs_sensor"
        )
        cooling = tb.ScheduleSystem(
            weekday_ruleset={
                "ruleset_default_value": 0,
                "ruleset_start_minute": [0, 0, 0],
                "ruleset_end_minute": [0, 0, 0],
                "ruleset_start_hour": [0, 8, 17],
                "ruleset_end_hour": [8, 17, 24],
                "ruleset_value": [30, 25, 30],
            },
            id=prefix + "cooling_setpoint",
        )
        for schedule in (
            price,
            cooling,
            zone["office_temperature_heating_setpoint"],
        ):
            schedule._allow_component_batching = True
        _disconnect(
            zone["office_temperature_heating_controller"],
            zone["office_space_heater_valve"],
            "inputSignal",
            "valvePosition",
        )
        model.add_connection(
            shared_valve_schedule,
            zone["office_space_heater_valve"],
            "scheduleValue",
            "valvePosition",
        )
        model.add_connection(zone["office_space_heater"], cost, "Power", "input_1")
        model.add_connection(price, cost, "scheduleValue", "input_2")
        model.add_component(cooling)
        opt_parts.append(
            {
                **zone,
                "valve_schedule": shared_valve_schedule,
                "price": price,
                "cost": cost,
                "cooling": cooling,
            }
        )
    model.load(
        draw_semantic_model=False,
        draw_simulation_model=False,
        verbose=0,
        enable_fusion=False,
    )
    if not batch_it:
        return model, opt_parts, truth, None, {}
    # These source schedules and terminal cost sensors are stateless,
    # equivalent components. Co-locate each role in the batching execution
    # layers so disjoint feedback-cycle ordering cannot fragment its n_c batch.
    groups = model.simulation_model.execution_order
    for role in (
        "price",
        "cooling",
        "office_temperature_heating_setpoint",
        "cost",
    ):
        role_components = [zone[role] for zone in opt_parts]
        for group in groups:
            group[:] = [
                component for component in group if component not in role_components
            ]
        groups.append(role_components)
    groups[:] = [group for group in groups if group]
    batched, batch_seconds = batch_model(model)
    role_keys = ("price", "cost", "cooling")
    batched_parts = {}
    required_roles = role_keys + ("office", "office_temperature_heating_setpoint")
    for role in required_roles:
        infos = [model.get_batched_component_info(z[role].id) for z in opt_parts]
        metas = [info[0] for info in infos]
        meta_ids = {meta.id for meta in metas}
        indices = sorted(info[1] for info in infos)
        if len(meta_ids) != 1 or indices != list(range(n_zones)):
            raise AssertionError(f"Optimization role {role} broadcast/collapse")
        batched_parts[role] = batched.components[metas[0].id]
    valve_meta, valve_component_index = model.get_batched_component_info(
        shared_valve_schedule.id
    )
    if valve_component_index != 0:
        raise AssertionError("Shared valve schedule must remain one broadcast source")
    batched_parts["valve_schedule"] = batched.components[valve_meta.id]
    return (
        batched,
        batched_parts,
        truth,
        batch_seconds,
        batching_mapping_audit(model, model.components),
    )


def _optimization_dimensions(n_zones: int, hours: int) -> dict[str, int | str]:
    n_steps = int(hours * 3600 / STEP_SIZE)
    # The canonical batched graph has one delayed heater-power feedback state
    # per zone in addition to its physical states.
    # Observed canonical augmented layout: 13 physical states and up to three
    # branch-local feedback carries per zone, plus one shared feedback carry.
    pareto_state_width = (CANONICAL_STATES_PER_ZONE + 3) * n_zones + 1
    pareto_links = max(0, n_steps - 1)
    return {
        "valve_schedule_semantics": (
            "single_zone_exact" if n_zones == 1 else "shared_broadcast"
        ),
        "n_base_full_workflow_components": CANONICAL_COMPONENTS_PER_ZONE * n_zones,
        "n_components_with_optimization": 26 * n_zones + 1,
        "independent_valve_schedules": 1,
        "decision_variables_per_timestep": 1,
        "n_decision_variables": n_steps,
        "counterfactual_independent_zone_valve_schedules": n_zones,
        "counterfactual_independent_n_decision_variables": n_steps * n_zones,
        # Comfort limits are soft relu terms in the objective, not NLP rows.
        "n_soft_comfort_penalty_samples": 2 * n_steps * n_zones,
        "pareto_control_variables": n_steps,
        "pareto_augmented_state_width": pareto_state_width,
        "pareto_boundary_state_variables": n_steps * pareto_state_width,
        "pareto_exact_polish_decision_variables": n_steps * (1 + pareto_state_width),
        "pareto_dynamics_constraints": pareto_links * pareto_state_width,
        "pareto_hard_epsilon_constraints_per_point": 1,
    }


def _constraint_metrics(zones: list[dict[str, Any]]) -> dict[str, Any]:
    upper = 0.0
    lower = 0.0
    for zone in zones:
        temperature = zone["office"].output["indoorTemperature"].history()
        cooling = zone["cooling"].output["scheduleValue"].history()
        heating = (
            zone["office_temperature_heating_setpoint"]
            .output["scheduleValue"]
            .history()
        )
        upper = max(
            upper,
            float(torch.relu(temperature - cooling).max().detach().cpu()),
        )
        lower = max(
            lower,
            float(torch.relu(heating - temperature).max().detach().cpu()),
        )
    maximum = max(upper, lower)
    return {
        "max_upper_constraint_violation": upper,
        "max_lower_constraint_violation": lower,
        "max_constraint_violation": maximum,
        "constraints_feasible": maximum <= 1e-6,
    }


def _solver_options(solver: str, config: BenchmarkConfig) -> dict[str, Any]:
    if solver == "SLSQP":
        return {"maxiter": config.optimization_maxiter, "ftol": 1e-9}
    raise ValueError(f"Unsupported constrained optimization solver: {solver}")


def _run_optimization(model, parts, config, device, solver):
    model.to(device, torch.float64)
    optimizer = tb.Optimizer(
        tb.Simulator(
            model,
            execution_mode="functional",
            execution_backend="eager",
        )
    )
    zones = parts if isinstance(parts, list) else [parts]
    valve_schedules = {z["valve_schedule"].id: z["valve_schedule"] for z in zones}
    variables = [
        (schedule, "scheduleValue", 0.0, 1.0) for schedule in valve_schedules.values()
    ]
    objectives = [(z["cost"], "output", "min") for z in zones]
    constraints = [
        constraint
        for z in zones
        for constraint in (
            (z["office"], "indoorTemperature", "upper", z["cooling"]),
            (
                z["office"],
                "indoorTemperature",
                "lower",
                z["office_temperature_heating_setpoint"],
            ),
        )
    ]
    result, seconds = timed(
        device,
        lambda: optimizer.optimize(
            start_time=OPT_START,
            end_time=OPT_START + dt.timedelta(hours=config.optimization_hours),
            step_size=STEP_SIZE,
            variables=variables,
            objectives=objectives,
            eq_cons=None,
            ineq_cons=constraints,
            method=("scipy", solver, "ad"),
            options=_solver_options(solver, config),
        ),
    )
    return result, seconds, _constraint_metrics(zones)


def run_optimization_matrix(config: BenchmarkConfig) -> list[dict[str, Any]]:
    """Run authoritative full-workflow constrained optimization scaling."""
    rows = []
    devices, skips = available_devices()
    for n_zones in config.zone_counts:
        for device, solver in OPTIMIZATION_MATRIX:
            base = {
                "device": device,
                "model_layout": "batched",
                "execution_mode": "functional",
                "execution_backend": "eager",
                "solver": solver,
                "method": ("scipy", solver, "ad"),
                "solver_options": _solver_options(solver, config),
                "solver_budget": config.optimization_maxiter,
                "budget_basis": (
                    "full_workflow_example constrained optimization " "uses maxiter=300"
                ),
                **_base_metrics(n_zones, config.optimization_hours),
                **_optimization_dimensions(n_zones, config.optimization_hours),
            }
            if device not in devices:
                rows.append({**skips[0], **base})
                checkpoint_results("optimization_scaling", config, rows)
                continue
            elapsed = []
            start_index = len(rows)
            for repetition in range(config.repeats):
                attempt_started = time.perf_counter()
                try:
                    model, parts, truth, batch_seconds, batching_mapping = (
                        _optimization_problem(n_zones, config.seed, batch_it=True)
                    )
                    result, seconds, constraint_metrics = _run_optimization(
                        model, parts, config, device, solver
                    )
                    counters = _result_counters(result)
                    speedup_eligible = (
                        counters["converged"]
                        and constraint_metrics["constraints_feasible"]
                    )
                    rows.append(
                        {
                            **base,
                            "status": "ok" if speedup_eligible else "nonconverged",
                            "successful_speedup_eligible": speedup_eligible,
                            "model_layout": "batched",
                            "execution_mode": "functional",
                            "execution_backend": "eager",
                            "repetition": repetition,
                            "seconds": seconds,
                            "memory": memory_stats_of_last_timed(),
                            "batch_seconds": batch_seconds,
                            "batching_mapping": batching_mapping,
                            "ground_truth": truth,
                            "quality": result.get("fun"),
                            "quality_label": (
                                "converged-feasible"
                                if speedup_eligible
                                else "not-speedup-eligible"
                            ),
                            **counters,
                            **constraint_metrics,
                            "raw_result": result,
                        }
                    )
                except Exception as exc:
                    seconds = time.perf_counter() - attempt_started
                    rows.append(
                        {
                            **base,
                            "status": "failed",
                            "successful_speedup_eligible": False,
                            "model_layout": "batched",
                            "execution_mode": "functional",
                            "execution_backend": "eager",
                            "repetition": repetition,
                            "seconds": seconds,
                            "quality_label": "failed",
                            "error": repr(exc),
                        }
                    )
                elapsed.append(seconds)
                checkpoint_results("optimization_scaling", config, rows)
            summary = _repeat_summary(elapsed)
            for row in rows[start_index:]:
                row.update(summary)
            checkpoint_results("optimization_scaling", config, rows)
    return rows


def batched_pareto_problem(
    n_zones: int, seed: int = SEED, *, measure_batch: bool = True
) -> dict[str, Any]:
    model, parts, truth, batch_seconds, batching_mapping = _optimization_problem(
        n_zones, seed, batch_it=True
    )
    if not measure_batch:
        batch_seconds = None
    return {
        "batched_model": model,
        "truth": truth,
        "batch_seconds": batch_seconds,
        "batching_mapping": batching_mapping,
        "variables": [(parts["valve_schedule"], "scheduleValue", 0.0, 1.0)],
        "objective1": (parts["cost"], "output", "min"),
        "objective2": (parts["office"], "indoorTemperature", "max"),
        "ineq_cons": [
            (parts["office"], "indoorTemperature", "upper", parts["cooling"]),
            (
                parts["office"],
                "indoorTemperature",
                "lower",
                parts["office_temperature_heating_setpoint"],
            ),
        ],
        "topology": _base_metrics(n_zones, 0),
        "full_mode_dimensions": _optimization_dimensions(n_zones, 72),
    }


def _pareto_preflight(n_zones: int, hours: int) -> dict[str, Any]:
    dimensions = _optimization_dimensions(n_zones, hours)
    n_variables = int(dimensions["pareto_exact_polish_decision_variables"])
    da = int(dimensions["pareto_augmented_state_width"])
    n_steps = int(hours * 3600 / STEP_SIZE)
    n_constraints = int(dimensions["pareto_dynamics_constraints"]) + 1
    d_zone = CANONICAL_STATES_PER_ZONE + 3
    d_global = da - n_zones * d_zone
    n_control = 1
    n_links = max(0, n_steps - 1)
    # Defect rows see controls and global state; replica rows additionally see
    # only their own local state block. The epsilon row remains dense.
    jacobian_per_link = d_global * (n_control + d_global + 1) + n_zones * d_zone * (
        n_control + d_global + d_zone + 1
    )
    jacobian_nnz = n_links * jacobian_per_link + n_variables
    # Per segment: shared (control/global) arrowhead plus one local triangle
    # per replica. No cross-replica local/local entries are declared.
    shared = n_control + d_global
    hessian_per_segment = (
        shared * (shared + 1) // 2
        + n_control * n_zones * d_zone
        + d_global * n_zones * d_zone
        + n_zones * d_zone * (d_zone + 1) // 2
    )
    hessian_nnz = n_steps * hessian_per_segment
    sparse_jacobian_bytes = jacobian_nnz * 16
    sparse_hessian_bytes = hessian_nnz * 16
    safe = (
        sparse_jacobian_bytes <= MAX_PARETO_DENSE_JACOBIAN_BYTES
        and sparse_hessian_bytes <= MAX_COLLOCATION_DENSE_HESSIAN_BYTES
    )
    return {
        "preflight_kind": "pareto_sparse_collocation",
        "preflight_n_variables": n_variables,
        "preflight_n_scalar_constraints": n_constraints,
        "sparse_jacobian_nnz": jacobian_nnz,
        "sparse_hessian_nnz": hessian_nnz,
        "sparse_jacobian_storage_bytes": sparse_jacobian_bytes,
        "sparse_hessian_storage_bytes": sparse_hessian_bytes,
        "replica_count": n_zones,
        "replica_state_width": d_zone,
        "global_state_width": d_global,
        "mathematically_safe": safe,
        "preflight_reason": (
            None
            if safe
            else (
                "replica block-arrowhead Jacobian/Hessian storage exceeds the "
                "configured safety cap"
            )
        ),
    }


def _run_pareto_scaling_solver(
    config: BenchmarkConfig, solver: str
) -> list[dict[str, Any]]:
    rows = []
    devices, _ = available_devices()
    device = "cuda" if "cuda" in devices else "cpu"
    for n_zones in config.zone_counts:
        preflight = _pareto_preflight(n_zones, config.optimization_hours)
        base = {
            "n_zones": n_zones,
            "device": device,
            "model_layout": "batched",
            "execution_mode": "functional",
            "execution_backend": "cuda_graph" if device == "cuda" else "eager",
            "solver": solver,
            "method": (
                ("scipy", "SLSQP", "ad")
                if solver == "SLSQP"
                else ("casadi", "ipopt", "ad", "collocation")
            ),
            "preferred_device": "cuda",
            "cpu_fallback": device == "cpu",
            "solver_budget": config.pareto_maxiter,
            "budget_basis": ("full_workflow constrained optimizer uses maxiter=300"),
            **_base_metrics(n_zones, config.optimization_hours),
            **_optimization_dimensions(n_zones, config.optimization_hours),
            **preflight,
        }
        if not preflight["mathematically_safe"]:
            rows.append(
                {
                    **base,
                    "status": "skipped",
                    "reason": preflight["preflight_reason"],
                }
            )
            checkpoint_results("pareto_scaling", config, rows)
            continue
        elapsed = []
        start_index = len(rows)
        for repetition in range(config.repeats):
            setup = batched_pareto_problem(n_zones, config.seed)
            model = setup["batched_model"]
            model.to(device, torch.float64)
            optimizer = tb.Optimizer(
                tb.Simulator(
                    model,
                    execution_mode="functional",
                    execution_backend=base["execution_backend"],
                )
            )
            attempt_started = time.perf_counter()
            try:
                result, seconds = timed(
                    device,
                    lambda: optimizer.pareto_front(
                        start_time=OPT_START,
                        end_time=OPT_START
                        + dt.timedelta(hours=config.optimization_hours),
                        step_size=STEP_SIZE,
                        variables=setup["variables"],
                        objective1=setup["objective1"],
                        objective2=setup["objective2"],
                        ineq_cons=setup["ineq_cons"],
                        n_points=config.pareto_points,
                        method=base["method"],
                        batched_prepass=True,
                        prepass_options={"max_iter": config.pareto_maxiter},
                        options={
                            "maxiter": config.pareto_maxiter,
                            "ftol": 1e-9,
                            "hessian": (
                                "exact" if solver == "ipopt" else "limited_memory"
                            ),
                        },
                    ),
                )
            except Exception as exc:
                seconds = time.perf_counter() - attempt_started
                elapsed.append(seconds)
                rows.append(
                    {
                        **base,
                        "status": "failed",
                        "successful_speedup_eligible": False,
                        "repetition": repetition,
                        "seconds": seconds,
                        "error": repr(exc),
                        "model_layout": "batched",
                        "execution_mode": "functional",
                        "execution_backend": base["execution_backend"],
                    }
                )
                checkpoint_results("pareto_scaling", config, rows)
                continue
            elapsed.append(seconds)
            counters = _result_counters(result)
            point_violations = []
            for point_index in range(len(result.theta)):
                result.apply(point_index)
                point_violations.append(
                    _constraint_metrics(
                        [
                            {
                                "office": setup["objective2"][0],
                                "cooling": setup["ineq_cons"][0][3],
                                "office_temperature_heating_setpoint": setup[
                                    "ineq_cons"
                                ][1][3],
                            }
                        ]
                    )["max_constraint_violation"]
                )
            max_violation = max(point_violations, default=float("nan"))
            constraints_feasible = bool(point_violations and max_violation <= 1e-6)
            speedup_eligible = counters["converged"] and constraints_feasible
            rows.append(
                {
                    **base,
                    "status": "ok" if speedup_eligible else "nonconverged",
                    "successful_speedup_eligible": speedup_eligible,
                    "model_layout": "batched",
                    "execution_mode": "functional",
                    "execution_backend": base["execution_backend"],
                    "batched_prepass": True,
                    "exact_polish": True,
                    "repetition": repetition,
                    "seconds": seconds,
                    "memory": memory_stats_of_last_timed(),
                    "batch_seconds": setup["batch_seconds"],
                    "batching_mapping": setup["batching_mapping"],
                    "ground_truth": setup["truth"],
                    "pareto_points": config.pareto_points,
                    "capture": result.capture,
                    "hessian": result.hessian,
                    "callback_shapes": getattr(result, "callback_shapes", None),
                    "max_collocation_defect": (
                        float(np.max(result.max_defect))
                        if getattr(result, "max_defect", None) is not None
                        else None
                    ),
                    "rollout_parity": getattr(result, "rollout_parity", None),
                    "quality": getattr(result, "objective_values", None),
                    "constraint_violation_by_point": point_violations,
                    "max_constraint_violation": max_violation,
                    "constraints_feasible": constraints_feasible,
                    **counters,
                    "raw_result": result,
                }
            )
            checkpoint_results("pareto_scaling", config, rows)
        summary = _repeat_summary(elapsed)
        for row in rows[start_index:]:
            row.update(summary)
        checkpoint_results("pareto_scaling", config, rows)
    return rows


def run_pareto_scaling(config: BenchmarkConfig) -> list[dict[str, Any]]:
    """Benchmark both captured-CUDA Pareto solver paths (CPU fallback)."""
    rows = []
    for solver in PARETO_SOLVERS:
        rows.extend(_run_pareto_scaling_solver(config, solver))
    checkpoint_results("pareto_scaling", config, rows)
    return rows


def run_optimization_scaling(
    config: BenchmarkConfig,
) -> list[dict[str, Any]]:
    """Run standard constrained and Pareto studies for one notebook."""
    standard = [
        {"study": "constrained_slsqp", **row} for row in run_optimization_matrix(config)
    ]
    pareto = [{"study": "pareto", **row} for row in run_pareto_scaling(config)]
    rows = standard + pareto
    checkpoint_results("optimization_scaling", config, rows)
    return rows
