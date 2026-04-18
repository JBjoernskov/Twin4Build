"""E6 -- Ablations around the batching compiler.

Three complementary ablations, each writing one row to
``results/e6_ablations.csv``:

* **A1 NO_BATCH_CLASSES** -- rerun the compilation with the default
  exclusion set, then rerun with the exclusion effectively disabled
  (schedules, sensors, and the outdoor environment become eligible for
  batching).  Reports whether ``compiled.load()`` still succeeds, the
  resulting speedup, and the post-compile graph size.

* **A2 Aligned-vs-gather** -- no model change.  Builds the compiled
  ``N_E6`` case once and reports the fraction of connections routed
  through the index-gather path vs. the aligned fast path using
  :func:`common.connection_wiring_stats`.

* **A3 Dtype float64 -> float32** -- rebuild the model under
  ``torch.set_default_dtype(torch.float32)`` and measure the speedup as
  well as the worst per-port accuracy drop relative to float64.
"""

from __future__ import annotations

import datetime
import gc
import sys
from contextlib import contextmanager
from typing import Dict, Iterator, List

import torch
from unittest.mock import patch

from twin4build.examples.paper_experiments import common, config


CSV_PATH = config.RESULTS_DIR / "e6_ablations.csv"


class _SentinelClass:
    """Placeholder class used to neutralize the compiler exclusion list."""


@contextmanager
def _no_exclusions() -> Iterator[None]:
    """Patch the three excluded classes so ``isinstance`` always returns False.

    ``build_compiled_model`` imports the exclusion list fresh every call, so
    patching the source classes with an unreachable sentinel disables the
    exclusion without editing library code.
    """
    targets = [
        "twin4build.systems.outdoor_environment.outdoor_environment_system."
        "OutdoorEnvironmentSystem",
        "twin4build.systems.schedule.schedule_system.ScheduleSystem",
        "twin4build.systems.sensor.sensor_system.SensorSystem",
    ]
    with patch(targets[0], _SentinelClass), \
            patch(targets[1], _SentinelClass), \
            patch(targets[2], _SentinelClass):
        yield


def _measure_speedup(
    model_orig: "common.Model",
    compiled: "common.Model",
    start: datetime.datetime,
    end: datetime.datetime,
    step: int,
    repeats: int = 3,
    warmup: int = 1,
) -> Dict[str, float]:
    orig_times: List[float] = []
    comp_times: List[float] = []
    for rep in range(warmup + repeats):
        _, dt_o = common.timed(
            lambda: common.simulate_once(model_orig, start, end, step)
        )
        _, dt_c = common.timed(
            lambda: common.simulate_once(compiled, start, end, step)
        )
        if rep >= warmup:
            orig_times.append(dt_o)
            comp_times.append(dt_c)
    _, o_med, _ = common.iqr(orig_times)
    _, c_med, _ = common.iqr(comp_times)
    return {
        "t_sim_orig_median": o_med,
        "t_sim_comp_median": c_med,
        "speedup_median": (o_med / c_med) if c_med > 0 else float("inf"),
    }


def _worst_abs_err(
    model_orig: "common.Model",
    compiled: "common.Model",
) -> float:
    df = common.port_errors(model_orig, compiled, io_types=("output",))
    df = df.dropna(subset=["max_abs_err"])
    df = df[df["n_samples"] > 0]
    if df.empty:
        return float("nan")
    return float(df["max_abs_err"].max())


# ---------------------------------------------------------------------------
# A1 -- NO_BATCH_CLASSES
# ---------------------------------------------------------------------------
def _ablation_A1(n_rooms: int) -> List[Dict[str, object]]:
    start = common.DEFAULT_START
    end = start + datetime.timedelta(days=config.HORIZON_DAYS_DEFAULT)
    step = config.STEP_SIZE_DEFAULT

    rows: List[Dict[str, object]] = []

    # -- Default exclusion ---------------------------------------------------
    print("[E6][A1] default exclusion ...")
    model = common.build_multi_room_model(
        n_rooms,
        start=start,
        horizon_days=config.HORIZON_DAYS_DEFAULT,
        step_size=step,
        model_id=f"e6_a1_default_n{n_rooms}",
    )
    model.load(draw_semantic_model=False, draw_simulation_model=False, verbose=0)
    compiled = model.build_compiled_model()
    compiled.load(
        draw_semantic_model=False, draw_simulation_model=False, verbose=0
    )
    n_comp_orig, n_conn_orig = common.graph_size(model)
    n_comp_comp, n_conn_comp = common.graph_size(compiled)
    timings = _measure_speedup(model, compiled, start, end, step)
    rows.append(
        {
            "ablation": "A1_no_batch_classes",
            "variant": "default",
            "n_rooms": n_rooms,
            "load_ok": True,
            "n_comp_orig": n_comp_orig,
            "n_comp_comp": n_comp_comp,
            "n_conn_orig": n_conn_orig,
            "n_conn_comp": n_conn_comp,
            **timings,
            "note": "_NO_BATCH_CLASSES = (OutdoorEnv, Schedule, Sensor)",
        }
    )
    del compiled
    gc.collect()

    # -- Exclusion disabled --------------------------------------------------
    print("[E6][A1] exclusion disabled ...")
    with _no_exclusions():
        try:
            compiled_no_excl = model.build_compiled_model()
            compiled_no_excl.load(
                draw_semantic_model=False,
                draw_simulation_model=False,
                verbose=0,
            )
            load_ok = True
            n_comp_comp_x, n_conn_comp_x = common.graph_size(compiled_no_excl)
            try:
                timings_x = _measure_speedup(
                    model, compiled_no_excl, start, end, step
                )
                err_x = _worst_abs_err(model, compiled_no_excl)
            except Exception as exc:  # noqa: BLE001
                timings_x = {
                    "t_sim_orig_median": float("nan"),
                    "t_sim_comp_median": float("nan"),
                    "speedup_median": float("nan"),
                }
                err_x = float("nan")
                note = f"simulate failed: {exc}"
            else:
                note = "exclusion disabled via sentinel patch"
        except Exception as exc:  # noqa: BLE001
            load_ok = False
            n_comp_comp_x = -1
            n_conn_comp_x = -1
            timings_x = {
                "t_sim_orig_median": float("nan"),
                "t_sim_comp_median": float("nan"),
                "speedup_median": float("nan"),
            }
            err_x = float("nan")
            note = f"compile/load failed: {exc}"

    rows.append(
        {
            "ablation": "A1_no_batch_classes",
            "variant": "no_exclusion",
            "n_rooms": n_rooms,
            "load_ok": load_ok,
            "n_comp_orig": n_comp_orig,
            "n_comp_comp": n_comp_comp_x,
            "n_conn_orig": n_conn_orig,
            "n_conn_comp": n_conn_comp_x,
            **timings_x,
            "max_abs_err": err_x,
            "note": note,
        }
    )
    del model
    gc.collect()
    return rows


# ---------------------------------------------------------------------------
# A2 -- Aligned-vs-gather
# ---------------------------------------------------------------------------
def _ablation_A2(n_rooms: int) -> List[Dict[str, object]]:
    print(f"[E6][A2] connection-path classification at N={n_rooms} ...")
    model = common.build_multi_room_model(
        n_rooms,
        horizon_days=config.HORIZON_DAYS_DEFAULT,
        step_size=config.STEP_SIZE_DEFAULT,
        model_id=f"e6_a2_n{n_rooms}",
    )
    model.load(draw_semantic_model=False, draw_simulation_model=False, verbose=0)
    compiled = model.build_compiled_model()
    compiled.load(
        draw_semantic_model=False, draw_simulation_model=False, verbose=0
    )
    wiring = common.connection_wiring_stats(compiled)
    n_comp_orig, n_conn_orig = common.graph_size(model)
    n_comp_comp, n_conn_comp = common.graph_size(compiled)
    del model, compiled
    gc.collect()

    return [
        {
            "ablation": "A2_aligned_vs_gather",
            "variant": f"n_rooms={n_rooms}",
            "n_rooms": n_rooms,
            "n_comp_orig": n_comp_orig,
            "n_comp_comp": n_comp_comp,
            "n_conn_orig": n_conn_orig,
            "n_conn_comp": n_conn_comp,
            "n_conn_total": wiring["n_connections_total"],
            "n_conn_aligned": wiring["n_connections_aligned"],
            "n_conn_gather": wiring["n_connections_gather"],
            "frac_aligned": wiring["fraction_aligned"],
            "note": "counts in compiled-graph connections only",
        }
    ]


# ---------------------------------------------------------------------------
# A3 -- dtype
# ---------------------------------------------------------------------------
def _run_dtype(
    dtype: torch.dtype, n_rooms: int
) -> Dict[str, object]:
    prev = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        start = common.DEFAULT_START
        end = start + datetime.timedelta(days=config.HORIZON_DAYS_DEFAULT)
        step = config.STEP_SIZE_DEFAULT
        model = common.build_multi_room_model(
            n_rooms,
            start=start,
            horizon_days=config.HORIZON_DAYS_DEFAULT,
            step_size=step,
            model_id=f"e6_a3_{str(dtype).replace('torch.', '')}_n{n_rooms}",
        )
        model.load(
            draw_semantic_model=False, draw_simulation_model=False, verbose=0
        )
        compiled = model.build_compiled_model()
        compiled.load(
            draw_semantic_model=False, draw_simulation_model=False, verbose=0
        )
        try:
            timings = _measure_speedup(model, compiled, start, end, step)
            err = _worst_abs_err(model, compiled)
            note = ""
        except Exception as exc:  # noqa: BLE001
            timings = {
                "t_sim_orig_median": float("nan"),
                "t_sim_comp_median": float("nan"),
                "speedup_median": float("nan"),
            }
            err = float("nan")
            note = f"run failed: {exc}"
        row = {
            "ablation": "A3_dtype",
            "variant": str(dtype),
            "n_rooms": n_rooms,
            "default_dtype": str(dtype),
            **timings,
            "max_abs_err": err,
            "note": note,
        }
        del model, compiled
        gc.collect()
        return row
    finally:
        torch.set_default_dtype(prev)


def _ablation_A3(n_rooms: int) -> List[Dict[str, object]]:
    print("[E6][A3] dtype float64 ...")
    r64 = _run_dtype(torch.float64, n_rooms)
    print("[E6][A3] dtype float32 ...")
    r32 = _run_dtype(torch.float32, n_rooms)
    return [r64, r32]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    n_rooms = config.N_E6
    # Set a consistent threading regime for all ablations.
    common.set_cpu_threads("cpu-mt", config.DEVICES.get("cpu-mt", 1))
    print(f"[E6] n_rooms={n_rooms} (CPU multi-thread)")

    rows: List[Dict[str, object]] = []
    rows.extend(_ablation_A1(n_rooms))
    rows.extend(_ablation_A2(n_rooms))
    rows.extend(_ablation_A3(n_rooms))

    common.write_csv(CSV_PATH, rows)
    print(f"\n[E6] wrote {len(rows)} rows -> {CSV_PATH}")


if __name__ == "__main__":
    sys.exit(main())
