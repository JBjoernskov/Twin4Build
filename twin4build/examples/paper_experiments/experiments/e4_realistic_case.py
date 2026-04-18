"""E4 -- Realistic case study (full-port accuracy + graph-compression audit).

Runs the estimator-example model used by
:mod:`twin4build.examples.model_compilator` and records:

* Full-port accuracy: max absolute and max relative error between the
  uncompiled reference and the compiled (batched) model, for every output
  port on every component that the compiler registered.  Written to
  ``results/e4_port_errors.csv``.
* Per-execution-group compression: how many original components fused into
  how many meta components, and the batch sizes involved.  Written to
  ``results/e4_compression.csv``.
* A one-row timing summary: ``results/e4_timing.csv``.
* Residual timeseries for three representative ports (for F5).  Saved
  under ``results/e4_residuals/<component>__<port>.csv``.

The reference implementation is :mod:`twin4build.examples.model_compilator`;
this script imports its ``setup_sensor_filenames`` helper so the data-source
wiring stays in one place.
"""

from __future__ import annotations

import datetime
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from dateutil import tz

import twin4build as tb
import twin4build.examples.utils as t4b_utils
from twin4build.examples.model_compilator import setup_sensor_filenames

from twin4build.examples.paper_experiments import common, config


CSV_ERRORS = config.RESULTS_DIR / "e4_port_errors.csv"
CSV_COMPRESSION = config.RESULTS_DIR / "e4_compression.csv"
CSV_TIMING = config.RESULTS_DIR / "e4_timing.csv"
RESIDUALS_DIR = config.RESULTS_DIR / "e4_residuals"

TZ_CPH = tz.gettz("Europe/Copenhagen")
# Match the periods used by model_compilator.py so results are comparable.
START_TIME = [
    datetime.datetime(2023, 11, 27, 0, 0, 0, tzinfo=TZ_CPH),
    datetime.datetime(2023, 12, 2, 0, 0, 0, tzinfo=TZ_CPH),
]
END_TIME = [
    datetime.datetime(2023, 12, 1, 0, 0, 0, tzinfo=TZ_CPH),
    datetime.datetime(2023, 12, 5, 0, 0, 0, tzinfo=TZ_CPH),
]
STEP_SIZE = 1200  # 20 minutes


# A hand-picked set of "representative" ports for the residual timeseries
# plot.  Each tuple is (component_id, port_name, io_type).
REPRESENTATIVE_PORTS = [
    ("020B", "indoorTemperature", "output"),
    ("020B_space_heater", "Power", "output"),
    ("020B_room_supply_damper", "airFlowRate", "output"),
]


def _load_reference() -> "tb.Model":
    """Load the estimator-example model with sensor CSVs wired up."""
    model = tb.Model(id="e4_estimator_example")
    filename_simulation = t4b_utils.get_path(
        ["estimator_example", "instance_graph.ttl"]
    )
    model.load(
        simulation_model_filename=filename_simulation,
        draw_semantic_model=False,
        draw_simulation_model=False,
        verbose=0,
    )
    setup_sensor_filenames(model)
    return model


def _simulate(model: "tb.Model") -> float:
    simulator = tb.Simulator(model)
    t0_simulate = 0.0

    import time

    t0 = time.perf_counter()
    simulator.simulate(
        step_size=STEP_SIZE,
        start_time=START_TIME,
        end_time=END_TIME,
        show_progress_bar=False,
    )
    t0_simulate = time.perf_counter() - t0
    return t0_simulate


def _save_residual_timeseries(
    model_orig: "tb.Model", model_compiled: "tb.Model"
) -> None:
    RESIDUALS_DIR.mkdir(parents=True, exist_ok=True)
    for comp_id, port_name, io_type in REPRESENTATIVE_PORTS:
        meta_info = model_orig.get_compiled_component_info(comp_id)
        if meta_info is None:
            continue
        meta, i_c = meta_info
        port_orig = getattr(model_orig.components[comp_id], io_type)[port_name]
        port_meta = getattr(model_compiled.components[meta.id], io_type)[port_name]
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
        path = RESIDUALS_DIR / f"{comp_id}__{port_name}.csv"
        df.to_csv(path, index=False)


def main() -> None:
    print("[E4] loading estimator example ...")
    model_orig = _load_reference()
    n_comp_orig, n_conn_orig = common.graph_size(model_orig)
    print(f"[E4]   original: {n_comp_orig} components, {n_conn_orig} connections")

    print("[E4] compiling ...")
    compiled, t_compile = common.timed(model_orig.build_compiled_model)
    compiled.simulation_model.load(verbose=0, validate_model=True)
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
    errors_df.to_csv(CSV_ERRORS, index=False)
    # Summary: aggregate across all valid ports.
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
    comp_df.to_csv(CSV_COMPRESSION, index=False)
    print(
        f"[E4]   groups total: {len(comp_df)};  "
        f"orig total={int(comp_df['n_original'].sum())}, "
        f"meta total={int(comp_df['n_meta'].sum())}"
    )

    # -- Residual timeseries for F5 -------------------------------------
    print("[E4] saving representative residual timeseries ...")
    _save_residual_timeseries(model_orig, compiled)

    # -- Timing summary --------------------------------------------------
    wiring = common.connection_wiring_stats(compiled)
    common.write_csv(
        CSV_TIMING,
        [
            {
                "model": "estimator_example",
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
                "step_size_s": STEP_SIZE,
                "n_periods": len(START_TIME),
            }
        ],
    )

    print(f"\n[E4] wrote:\n  {CSV_ERRORS}\n  {CSV_COMPRESSION}\n  {CSV_TIMING}")
    print(f"  {RESIDUALS_DIR}/ ({len(REPRESENTATIVE_PORTS)} files)")


if __name__ == "__main__":
    sys.exit(main())
