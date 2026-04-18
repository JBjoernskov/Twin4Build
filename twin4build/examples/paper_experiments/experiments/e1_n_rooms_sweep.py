"""E1 -- N_ROOMS scalability sweep (CPU).

For each combination of ``device`` in :data:`config.DEVICES` and ``n_rooms``
in :data:`config.N_ROOMS_SWEEP`, build the synthetic multi-room model,
compile it, and measure simulate wall-clock time for both the uncompiled
reference and the compiled (batched) version.  Each measured cell reports
``N_REPEATS`` runs plus ``N_WARMUP`` discarded warm-ups.

Writes ``results/e1_n_rooms.csv`` with one row per individual run (device,
n_rooms, replicate index, and wallclock breakdown) so downstream plots can
aggregate to medians/IQRs without losing the raw data.
"""

from __future__ import annotations

import datetime
import gc
import sys
from pathlib import Path
from typing import Dict, List

import torch

from twin4build.examples.paper_experiments import common, config


CSV_PATH = config.RESULTS_DIR / "e1_n_rooms.csv"
SUMMARY_PATH = config.RESULTS_DIR / "e1_n_rooms_summary.csv"


def _run_cell(
    device_key: str,
    thread_count: int,
    n_rooms: int,
) -> List[Dict[str, object]]:
    """Measure one (device, n_rooms) cell and return one row per run."""
    common.set_cpu_threads(device_key, thread_count)
    common.reset_peak_memory(device_key)

    start = common.DEFAULT_START
    end = start + datetime.timedelta(days=config.HORIZON_DAYS_DEFAULT)
    step = config.STEP_SIZE_DEFAULT
    repeats = config.repeats_for(n_rooms)
    warmup = config.N_WARMUP

    # Build once (not measured per repeat -- we only care about simulate wall).
    _, t_build_orig = common.timed(
        lambda: common.build_multi_room_model(
            n_rooms,
            start=start,
            horizon_days=config.HORIZON_DAYS_DEFAULT,
            step_size=step,
            model_id=f"e1_n{n_rooms}_{device_key}_orig",
        )
    )
    model_orig = common.build_multi_room_model(
        n_rooms,
        start=start,
        horizon_days=config.HORIZON_DAYS_DEFAULT,
        step_size=step,
        model_id=f"e1_n{n_rooms}_{device_key}",
    )
    model_orig.load(
        draw_semantic_model=False,
        draw_simulation_model=False,
        verbose=0,
    )
    n_comp_orig, n_conn_orig = common.graph_size(model_orig)

    compiled, t_compile = common.timed(model_orig.build_compiled_model)
    compiled.load(
        draw_semantic_model=False,
        draw_simulation_model=False,
        verbose=0,
    )
    n_comp_comp, n_conn_comp = common.graph_size(compiled)
    wiring = common.connection_wiring_stats(compiled)

    # Run warmups (discarded) then measured repeats, alternating orig/comp
    # on the same repeat index so each replicate sees comparable system load.
    orig_times: List[float] = []
    comp_times: List[float] = []
    for rep in range(warmup + repeats):
        _, dt_o = common.timed(
            lambda: common.simulate_once(model_orig, start, end, step)
        )
        _, dt_c = common.timed(
            lambda: common.simulate_once(compiled, start, end, step)
        )
        orig_times.append(dt_o)
        comp_times.append(dt_c)

    rss_peak = common.peak_memory_mb(device_key)

    rows: List[Dict[str, object]] = []
    for rep in range(warmup + repeats):
        rows.append(
            {
                "device": device_key,
                "thread_count": thread_count,
                "n_rooms": n_rooms,
                "horizon_days": config.HORIZON_DAYS_DEFAULT,
                "step_size_s": step,
                "repeat": rep,
                "is_warmup": rep < warmup,
                "t_build_s": t_build_orig,
                "t_compile_s": t_compile,
                "t_sim_orig_s": orig_times[rep],
                "t_sim_comp_s": comp_times[rep],
                "n_comp_orig": n_comp_orig,
                "n_comp_comp": n_comp_comp,
                "n_conn_orig": n_conn_orig,
                "n_conn_comp": n_conn_comp,
                "frac_aligned": wiring["fraction_aligned"],
                "n_conn_aligned": wiring["n_connections_aligned"],
                "n_conn_gather": wiring["n_connections_gather"],
                "rss_peak_mb": rss_peak,
                "torch_default_dtype": str(torch.get_default_dtype()),
                "notes": "",
            }
        )

    # Explicitly release so the next cell doesn't accumulate RSS.
    del compiled, model_orig
    gc.collect()
    return rows


def main() -> None:
    print(f"[E1] MAX_N_ROOMS = {config.MAX_N_ROOMS}")
    print(f"[E1] sweep       = {config.N_ROOMS_SWEEP}")
    print(f"[E1] devices     = {config.DEVICES}")

    all_rows: List[Dict[str, object]] = []
    for device_key, thread_count in config.DEVICES.items():
        for n_rooms in config.N_ROOMS_SWEEP:
            print(
                f"[E1] device={device_key:>6s} n_rooms={n_rooms:>4d} ...",
                end=" ",
                flush=True,
            )
            try:
                cell_rows = _run_cell(device_key, thread_count, n_rooms)
            except Exception as exc:  # noqa: BLE001 -- want to log and move on
                print(f"FAILED: {exc}")
                all_rows.append(
                    {
                        "device": device_key,
                        "thread_count": thread_count,
                        "n_rooms": n_rooms,
                        "notes": f"failed: {exc}",
                    }
                )
                continue
            median_orig = sorted(
                r["t_sim_orig_s"] for r in cell_rows if not r["is_warmup"]
            )[len(cell_rows) // 2]
            median_comp = sorted(
                r["t_sim_comp_s"] for r in cell_rows if not r["is_warmup"]
            )[len(cell_rows) // 2]
            speedup = median_orig / median_comp if median_comp > 0 else float("inf")
            print(
                f"orig={median_orig:.3f}s  comp={median_comp:.3f}s  "
                f"speedup={speedup:.2f}x"
            )
            all_rows.extend(cell_rows)

    common.write_csv(CSV_PATH, all_rows)
    print(f"\n[E1] wrote {len(all_rows)} rows -> {CSV_PATH}")

    # Summary: median + IQR per (device, n_rooms).
    summary_rows = _build_summary(all_rows)
    common.write_csv(SUMMARY_PATH, summary_rows)
    print(f"[E1] wrote {len(summary_rows)} summary rows -> {SUMMARY_PATH}")


def _build_summary(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    from collections import defaultdict

    buckets: Dict[tuple, List[Dict[str, object]]] = defaultdict(list)
    for r in rows:
        if r.get("is_warmup") or "t_sim_orig_s" not in r:
            continue
        key = (r["device"], r["n_rooms"])
        buckets[key].append(r)

    out: List[Dict[str, object]] = []
    for (device, n_rooms), bucket in sorted(buckets.items()):
        orig_vals = [float(r["t_sim_orig_s"]) for r in bucket]
        comp_vals = [float(r["t_sim_comp_s"]) for r in bucket]
        orig_q = common.iqr(orig_vals)
        comp_q = common.iqr(comp_vals)
        ref = bucket[0]
        n_steps = int(
            config.HORIZON_DAYS_DEFAULT * 24 * 3600 / config.STEP_SIZE_DEFAULT
        )
        comp_steps_per_s = (
            ref["n_comp_orig"] * n_steps / comp_q[1] if comp_q[1] > 0 else float("nan")
        )
        out.append(
            {
                "device": device,
                "n_rooms": n_rooms,
                "n_samples": len(bucket),
                "t_sim_orig_q25": orig_q[0],
                "t_sim_orig_median": orig_q[1],
                "t_sim_orig_q75": orig_q[2],
                "t_sim_comp_q25": comp_q[0],
                "t_sim_comp_median": comp_q[1],
                "t_sim_comp_q75": comp_q[2],
                "speedup_median": (
                    orig_q[1] / comp_q[1] if comp_q[1] > 0 else float("inf")
                ),
                "n_comp_orig": ref["n_comp_orig"],
                "n_comp_comp": ref["n_comp_comp"],
                "n_conn_orig": ref["n_conn_orig"],
                "n_conn_comp": ref["n_conn_comp"],
                "frac_aligned": ref["frac_aligned"],
                "component_steps_per_s_compiled": comp_steps_per_s,
                "t_compile_s": ref["t_compile_s"],
                "rss_peak_mb": ref["rss_peak_mb"],
            }
        )
    return out


if __name__ == "__main__":
    sys.exit(main())
