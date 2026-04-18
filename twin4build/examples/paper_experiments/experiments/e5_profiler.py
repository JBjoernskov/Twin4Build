"""E5 -- torch.profiler instrumentation (CPU dispatch breakdown).

Runs one measured simulate pass through the reference and compiled models
at ``N_E5`` rooms under :pymod:`torch.profiler`, exports a Chrome trace per
run, and aggregates CPU self-time per top-level op family ("aten::...",
Python interpreter, "other").  The resulting bar chart (F6) makes visible
*why* the compiler is faster: fewer dispatches, not faster math.

Writes:

* ``results/e5_profiler_summary.csv`` -- aggregated per-family self-time.
* ``results/e5_trace_orig.json`` / ``results/e5_trace_comp.json`` --
  Chrome trace files.  Readable at https://ui.perfetto.dev/ or
  ``chrome://tracing``.

The script is CPU-only in this pass.  A commented switch at the top of
``_profile_once`` shows how to add CUDA activity for a future GPU pass.
"""

from __future__ import annotations

import datetime
import sys
from pathlib import Path
from typing import Dict, List, Optional

from torch.profiler import ProfilerActivity, profile

from twin4build.examples.paper_experiments import common, config


CSV_PATH = config.RESULTS_DIR / "e5_profiler_summary.csv"
TRACE_ORIG = config.RESULTS_DIR / "e5_trace_orig.json"
TRACE_COMP = config.RESULTS_DIR / "e5_trace_comp.json"


def _classify_op(name: str) -> str:
    """Coarse grouping for the stacked-bar plot."""
    if name.startswith("aten::"):
        return "aten"
    if name.startswith("autograd::"):
        return "autograd"
    if "python" in name.lower():
        return "python"
    if name.startswith("cudaLaunchKernel") or name.startswith("cuda"):
        return "cuda_launch"
    return "other"


def _profile_once(
    simulate_callable,
    label: str,
    trace_path: Path,
) -> List[Dict[str, object]]:
    """Profile one run and return per-family aggregate rows."""
    # To enable CUDA profiling in a future GPU pass, uncomment the CUDA
    # activity below and ensure the model's tensors live on a cuda device.
    activities = [
        ProfilerActivity.CPU,
        # ProfilerActivity.CUDA,
    ]
    with profile(
        activities=activities,
        record_shapes=False,
        with_stack=False,
        profile_memory=False,
    ) as prof:
        simulate_callable()

    # Save Chrome trace (human-readable in perfetto / chrome://tracing).
    prof.export_chrome_trace(str(trace_path))

    # Aggregate: sum CPU self-time by family.
    family_totals: Dict[str, float] = {}
    family_counts: Dict[str, int] = {}
    for evt in prof.key_averages():
        family = _classify_op(evt.key)
        family_totals[family] = (
            family_totals.get(family, 0.0)
            + float(evt.cpu_time_total) / 1e6  # microseconds -> seconds
        )
        family_counts[family] = (
            family_counts.get(family, 0) + int(evt.count)
        )

    rows: List[Dict[str, object]] = []
    for family, total in sorted(family_totals.items()):
        rows.append(
            {
                "label": label,
                "family": family,
                "cpu_self_time_s": total,
                "n_events": family_counts.get(family, 0),
            }
        )
    return rows


def main() -> None:
    n_rooms = config.N_E5
    print(f"[E5] profiling at N_ROOMS={n_rooms} on CPU")
    common.set_cpu_threads("cpu-mt", config.DEVICES.get("cpu-mt", 1))

    start = common.DEFAULT_START
    horizon = config.HORIZON_DAYS_DEFAULT
    step = config.STEP_SIZE_DEFAULT
    end = start + datetime.timedelta(days=horizon)

    print("[E5] building model ...")
    model_orig = common.build_multi_room_model(
        n_rooms,
        start=start,
        horizon_days=horizon,
        step_size=step,
        model_id=f"e5_n{n_rooms}",
    )
    model_orig.load(
        draw_semantic_model=False, draw_simulation_model=False, verbose=0
    )

    print("[E5] compiling model ...")
    compiled = model_orig.build_compiled_model()
    compiled.load(
        draw_semantic_model=False, draw_simulation_model=False, verbose=0
    )

    # Warm-up both models once before profiling to avoid first-run setup
    # drowning the dispatch breakdown.
    print("[E5] warm-up pass ...")
    common.simulate_once(model_orig, start, end, step)
    common.simulate_once(compiled, start, end, step)

    all_rows: List[Dict[str, object]] = []

    print("[E5] profiling original ...")
    all_rows.extend(
        _profile_once(
            lambda: common.simulate_once(model_orig, start, end, step),
            label="original",
            trace_path=TRACE_ORIG,
        )
    )
    print(f"[E5]   trace -> {TRACE_ORIG}")

    print("[E5] profiling compiled ...")
    all_rows.extend(
        _profile_once(
            lambda: common.simulate_once(compiled, start, end, step),
            label="compiled",
            trace_path=TRACE_COMP,
        )
    )
    print(f"[E5]   trace -> {TRACE_COMP}")

    common.write_csv(CSV_PATH, all_rows)
    print(f"\n[E5] wrote summary -> {CSV_PATH}")

    # Pretty print so the script is useful on its own.
    totals: Dict[str, Dict[str, float]] = {}
    for r in all_rows:
        totals.setdefault(r["label"], {})[r["family"]] = r["cpu_self_time_s"]
    print("\n[E5] CPU self-time (s) per family:")
    families = sorted({f for fam in totals.values() for f in fam})
    header = "  " + "label".ljust(10) + "  " + "  ".join(f.ljust(12) for f in families)
    print(header)
    for label, fam_map in totals.items():
        row = "  " + label.ljust(10) + "  " + "  ".join(
            f"{fam_map.get(f, 0.0):12.3f}" for f in families
        )
        print(row)


if __name__ == "__main__":
    sys.exit(main())
