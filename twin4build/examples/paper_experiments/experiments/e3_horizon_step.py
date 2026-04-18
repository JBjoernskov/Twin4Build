"""E3 -- Horizon x step-size grid (what kind of speedup, arithmetic or dispatch).

Holds ``N_ROOMS = N_E3`` fixed and sweeps the Cartesian product of
``HORIZON_GRID_DAYS x STEP_GRID_S``.  At small step size the simulation is
dominated by *per-step* dispatch overhead (many short calls), so we expect
the compiler's speedup to peak there; at long step size and long horizon
the bottleneck shifts toward arithmetic and speedup should flatten.

Writes ``results/e3_horizon_step.csv`` with one row per grid cell.
"""

from __future__ import annotations

import datetime
import gc
import sys
from typing import Dict, List

from twin4build.examples.paper_experiments import common, config


CSV_PATH = config.RESULTS_DIR / "e3_horizon_step.csv"


def _run_cell(
    n_rooms: int,
    horizon_days: int,
    step_size: int,
    device_key: str,
    thread_count: int,
) -> Dict[str, object]:
    common.set_cpu_threads(device_key, thread_count)
    common.reset_peak_memory(device_key)

    start = common.DEFAULT_START
    end = start + datetime.timedelta(days=horizon_days)
    # Long horizons + small dt are expensive; trim repeats accordingly.
    n_steps = int(horizon_days * 24 * 3600 / step_size)
    if n_steps > 20_000:
        repeats = 1
    elif n_steps > 5_000:
        repeats = 2
    else:
        repeats = config.repeats_for(n_rooms)
    warmup = min(1, config.N_WARMUP)

    model = common.build_multi_room_model(
        n_rooms,
        start=start,
        horizon_days=horizon_days,
        step_size=step_size,
        model_id=f"e3_h{horizon_days}_dt{step_size}",
    )
    model.load(
        draw_semantic_model=False, draw_simulation_model=False, verbose=0
    )
    n_comp_orig, n_conn_orig = common.graph_size(model)

    compiled, t_compile = common.timed(model.build_compiled_model)
    compiled.load(
        draw_semantic_model=False, draw_simulation_model=False, verbose=0
    )
    n_comp_comp, n_conn_comp = common.graph_size(compiled)

    orig_measurements: List[float] = []
    comp_measurements: List[float] = []
    for rep in range(warmup + repeats):
        _, dt_o = common.timed(
            lambda: common.simulate_once(model, start, end, step_size)
        )
        _, dt_c = common.timed(
            lambda: common.simulate_once(compiled, start, end, step_size)
        )
        if rep >= warmup:
            orig_measurements.append(dt_o)
            comp_measurements.append(dt_c)

    o_q25, o_q50, o_q75 = common.iqr(orig_measurements)
    c_q25, c_q50, c_q75 = common.iqr(comp_measurements)
    speedup = o_q50 / c_q50 if c_q50 > 0 else float("inf")

    del model, compiled
    gc.collect()

    return {
        "device": device_key,
        "thread_count": thread_count,
        "n_rooms": n_rooms,
        "horizon_days": horizon_days,
        "step_size_s": step_size,
        "n_steps": n_steps,
        "n_repeats": repeats,
        "t_compile_s": t_compile,
        "t_sim_orig_q25": o_q25,
        "t_sim_orig_median": o_q50,
        "t_sim_orig_q75": o_q75,
        "t_sim_comp_q25": c_q25,
        "t_sim_comp_median": c_q50,
        "t_sim_comp_q75": c_q75,
        "speedup_median": speedup,
        "n_comp_orig": n_comp_orig,
        "n_comp_comp": n_comp_comp,
        "n_conn_orig": n_conn_orig,
        "n_conn_comp": n_conn_comp,
    }


def main() -> None:
    n_rooms = config.N_E3
    device_key = "cpu-mt"
    thread_count = config.DEVICES.get(device_key, 1)
    print(
        f"[E3] n_rooms={n_rooms}  horizons={config.HORIZON_GRID_DAYS}  "
        f"steps={config.STEP_GRID_S}"
    )

    rows: List[Dict[str, object]] = []
    for h in config.HORIZON_GRID_DAYS:
        for dt in config.STEP_GRID_S:
            # Skip infeasible cells (simulation way too long at laptop scale).
            n_steps = int(h * 24 * 3600 / dt)
            if n_steps > 50_000:
                print(
                    f"[E3] skipping h={h}d dt={dt}s ({n_steps} steps, "
                    f"too expensive)"
                )
                continue
            print(
                f"[E3] h={h:>3d}d dt={dt:>5d}s ({n_steps:>6d} steps) ...",
                end=" ",
                flush=True,
            )
            try:
                row = _run_cell(n_rooms, h, dt, device_key, thread_count)
            except Exception as exc:  # noqa: BLE001
                print(f"FAILED: {exc}")
                continue
            print(
                f"orig={row['t_sim_orig_median']:.3f}s  "
                f"comp={row['t_sim_comp_median']:.3f}s  "
                f"speedup={row['speedup_median']:.2f}x"
            )
            rows.append(row)

    common.write_csv(CSV_PATH, rows)
    print(f"\n[E3] wrote {len(rows)} rows -> {CSV_PATH}")


if __name__ == "__main__":
    sys.exit(main())
