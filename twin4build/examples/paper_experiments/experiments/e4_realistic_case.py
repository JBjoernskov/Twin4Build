"""E4 -- Accuracy audit on the scaling multi-room building.

Reuses the synthetic multi-room model that drives E1 (see
``common.build_multi_room_model``) and records, for each of a small and a
large configuration, the numerical agreement between the uncompiled
reference and the compiled (batched) version:

* Full-port accuracy: max absolute and max relative error between the
  uncompiled reference and the compiled model, for every output port on
  every component that the compiler registered.  Written to
  ``results/e4_port_errors.csv`` (one file, with an ``n_rooms`` column).
* Per-execution-group compression: how many original components fused
  into how many meta components, and the batch sizes involved.  Written
  to ``results/e4_compression.csv`` (with an ``n_rooms`` column).
* Per-case timing summary: ``results/e4_timing.csv`` (one row per size).
* Residual timeseries for three representative ports at each size (for
  F5).  Saved under ``results/e4_residuals/n{N_ROOMS}/<component>__<port>.csv``.

Building a small instance (``N_ROOMS_SMALL``) exercises the code path with
modest batch widths, while the large instance (``N_ROOMS_LARGE``) exercises
the deep-batching regime that motivates the compiler in the first place.
Comparing both keeps the accuracy claim honest across scales.
"""

from __future__ import annotations

import datetime
import gc
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

import twin4build as tb

from twin4build.examples.paper_experiments import common, config


CSV_ERRORS = config.RESULTS_DIR / "e4_port_errors.csv"
CSV_COMPRESSION = config.RESULTS_DIR / "e4_compression.csv"
CSV_TIMING = config.RESULTS_DIR / "e4_timing.csv"
RESIDUALS_DIR = config.RESULTS_DIR / "e4_residuals"

# Small and large cases.  Both are clamped against the global scale cap so
# this experiment never grows past what the rest of the suite allows.
N_ROOMS_SMALL: int = min(5, config.MAX_N_ROOMS)
N_ROOMS_LARGE: int = min(128, config.MAX_N_ROOMS)
N_ROOMS_CASES: List[int] = [N_ROOMS_SMALL, N_ROOMS_LARGE]

HORIZON_DAYS: int = config.HORIZON_DAYS_DEFAULT
STEP_SIZE: int = config.STEP_SIZE_DEFAULT


def _representative_ports(n_rooms: int) -> List[tuple]:
    """Representative output ports for the F5 residual plot.

    We always pick the first room so the plot is comparable across sizes.
    """
    _ = n_rooms  # kept for symmetry: size-dependent picks can slot in here
    return [
        ("room_0", "indoorTemperature", "output"),
        ("room_0_space_heater", "Power", "output"),
        ("room_0_supply_damper", "airFlowRate", "output"),
    ]


def _build_and_load(n_rooms: int, model_id: str) -> "tb.Model":
    model = common.build_multi_room_model(
        n_rooms,
        start=common.DEFAULT_START,
        horizon_days=HORIZON_DAYS,
        step_size=STEP_SIZE,
        model_id=model_id,
    )
    model.load(
        draw_semantic_model=False,
        draw_simulation_model=False,
        verbose=0,
    )
    return model


def _simulate(model: "tb.Model") -> float:
    start = common.DEFAULT_START
    end = start + datetime.timedelta(days=HORIZON_DAYS)
    _, dt = common.timed(
        lambda: common.simulate_once(model, start, end, STEP_SIZE)
    )
    return dt


def _save_residual_timeseries(
    model_orig: "tb.Model",
    model_compiled: "tb.Model",
    representative_ports: List[tuple],
    out_dir: Path,
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for comp_id, port_name, io_type in representative_ports:
        meta_info = model_orig.get_compiled_component_info(comp_id)
        if meta_info is None:
            continue
        meta, i_c = meta_info
        try:
            port_orig = getattr(model_orig.components[comp_id], io_type)[port_name]
            port_meta = getattr(
                model_compiled.components[meta.id], io_type
            )[port_name]
        except (AttributeError, KeyError):
            continue
        a = port_orig.history()[:, :, 0].detach().cpu().numpy()  # (n_t, n_s)
        b = port_meta.history()[:, :, i_c].detach().cpu().numpy()
        n_t, n_s = a.shape
        df = pd.DataFrame(
            {
                "t_index": np.tile(np.arange(n_t), n_s),
                "sim_index": np.repeat(np.arange(n_s), n_t),
                "value_orig": a.T.reshape(-1),
                "value_compiled": b.T.reshape(-1),
                "residual": (a - b).T.reshape(-1),
            }
        )
        df.to_csv(out_dir / f"{comp_id}__{port_name}.csv", index=False)
        written += 1
    return written


def _run_case(n_rooms: int) -> Dict[str, object]:
    """Run the full accuracy audit for a single ``n_rooms`` case."""
    print(f"\n[E4] ==== N_ROOMS = {n_rooms} ====")
    print(f"[E4] building original ...")
    model_orig = _build_and_load(n_rooms, model_id=f"e4_n{n_rooms}_orig")
    n_comp_orig, n_conn_orig = common.graph_size(model_orig)
    print(f"[E4]   original: {n_comp_orig} components, {n_conn_orig} connections")

    print("[E4] compiling ...")
    compiled, t_compile = common.timed(model_orig.build_compiled_model)
    compiled.load(
        draw_semantic_model=False,
        draw_simulation_model=False,
        verbose=0,
    )
    n_comp_comp, n_conn_comp = common.graph_size(compiled)
    print(f"[E4]   compiled: {n_comp_comp} components, {n_conn_comp} connections")
    print(f"[E4]   compile time: {t_compile:.3f}s")

    print("[E4] simulating original ...")
    t_orig = _simulate(model_orig)
    print(f"[E4]   original simulate: {t_orig:.3f}s")

    print("[E4] simulating compiled ...")
    t_comp = _simulate(compiled)
    print(f"[E4]   compiled simulate: {t_comp:.3f}s")

    speedup = t_orig / t_comp if t_comp > 0 else float("inf")
    print(f"[E4]   speedup: {speedup:.2f}x")

    # -- Accuracy audit --------------------------------------------------
    print("[E4] computing per-port errors ...")
    errors_df = common.port_errors(model_orig, compiled, io_types=("output",))
    errors_df.insert(0, "n_rooms", n_rooms)
    valid = errors_df.dropna(subset=["max_abs_err"])
    valid = valid[valid["n_samples"] > 0]
    if len(valid):
        worst = valid.sort_values("max_abs_err", ascending=False).head(5)
        print(
            f"[E4]   ports audited: {len(valid)};  "
            f"global max_abs_err = {valid['max_abs_err'].max():.3e}"
        )
        print("[E4]   top-5 ports by abs error:")
        for _, row in worst.iterrows():
            print(
                f"           {row['component_id']:30s} {row['port_name']:25s}  "
                f"abs={row['max_abs_err']:.3e}  rel={row['max_rel_err']:.3e}"
            )
    else:
        print("[E4]   WARNING: no valid port comparisons produced.")

    # -- Compression audit -----------------------------------------------
    print("[E4] computing per-group compression ...")
    comp_df = common.compressed_group_stats(model_orig, compiled)
    comp_df.insert(0, "n_rooms", n_rooms)
    print(
        f"[E4]   groups total: {len(comp_df)};  "
        f"orig total={int(comp_df['n_original'].sum())}, "
        f"meta total={int(comp_df['n_meta'].sum())}"
    )

    # -- Residual timeseries for F5 -------------------------------------
    print("[E4] saving representative residual timeseries ...")
    case_residuals_dir = RESIDUALS_DIR / f"n{n_rooms}"
    n_residuals = _save_residual_timeseries(
        model_orig, compiled, _representative_ports(n_rooms), case_residuals_dir
    )

    wiring = common.connection_wiring_stats(compiled)
    timing_row: Dict[str, object] = {
        "n_rooms": n_rooms,
        "horizon_days": HORIZON_DAYS,
        "step_size_s": STEP_SIZE,
        "t_compile_s": t_compile,
        "t_sim_orig_s": t_orig,
        "t_sim_comp_s": t_comp,
        "speedup": speedup,
        "n_comp_orig": n_comp_orig,
        "n_comp_comp": n_comp_comp,
        "n_conn_orig": n_conn_orig,
        "n_conn_comp": n_conn_comp,
        "frac_aligned": wiring["fraction_aligned"],
        "n_conn_aligned": wiring["n_connections_aligned"],
        "n_conn_gather": wiring["n_connections_gather"],
        "n_residual_files": n_residuals,
    }

    # Release models before moving to the next (larger) case so peak RSS
    # is dominated by a single case at a time.
    del compiled, model_orig
    gc.collect()

    return {
        "errors_df": errors_df,
        "compression_df": comp_df,
        "timing_row": timing_row,
    }


def main() -> None:
    all_errors: List[pd.DataFrame] = []
    all_compression: List[pd.DataFrame] = []
    timing_rows: List[Dict[str, object]] = []

    for n_rooms in N_ROOMS_CASES:
        result = _run_case(n_rooms)
        all_errors.append(result["errors_df"])
        all_compression.append(result["compression_df"])
        timing_rows.append(result["timing_row"])

    pd.concat(all_errors, ignore_index=True).to_csv(CSV_ERRORS, index=False)
    pd.concat(all_compression, ignore_index=True).to_csv(
        CSV_COMPRESSION, index=False
    )
    common.write_csv(CSV_TIMING, timing_rows)

    print("\n[E4] summary")
    for row in timing_rows:
        print(
            f"  n_rooms={row['n_rooms']:>4d}  "
            f"orig={row['t_sim_orig_s']:.3f}s  "
            f"comp={row['t_sim_comp_s']:.3f}s  "
            f"speedup={row['speedup']:.2f}x"
        )
    print(f"\n[E4] wrote:\n  {CSV_ERRORS}\n  {CSV_COMPRESSION}\n  {CSV_TIMING}")
    print(f"  {RESIDUALS_DIR}/")


if __name__ == "__main__":
    sys.exit(main())
