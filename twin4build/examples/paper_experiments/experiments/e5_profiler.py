"""E5 -- torch.profiler instrumentation (CPU dispatch breakdown).

Runs one measured simulate pass through the reference and compiled models
at ``N_E5`` rooms under :pymod:`torch.profiler`, exports a Chrome trace per
run, and aggregates CPU **self** time per top-level op family ("aten::...",
"autograd::...", "other").  The complementary ``python_overhead`` row is
derived as ``wall_time - sum(self_cpu_time)`` and captures everything the
profiler did *not* attribute to a recorded op -- i.e. Python interpreter
time, twin4build's per-step dispatch glue, tensor-construction boxing, and
a small (symmetric) profiler-instrumentation overhead.  That decomposition
is what F6 visualises: the compiler's win is expected to show up as a
collapse of ``python_overhead`` + event count rather than faster aten math.

Methodology notes
-----------------
* We aggregate ``self_cpu_time_total`` (not ``cpu_time_total``).  Parent
  events -- for example user annotations or higher-level autograd scopes
  -- include their children's CPU time in ``cpu_time_total``; summing that
  across ``key_averages`` would double-count.  Self-time is additive.
* Wall time is measured with ``perf_counter`` *inside* the profiler
  context so profiler ``__enter__``/``__exit__`` is excluded, and the
  derived ``python_overhead`` row is ``max(0, wall - Σ self_cpu_time)``.
* ``with_stack`` and ``profile_memory`` are off: Python-frame events are
  not needed to *quantify* Python overhead (the derived row does that),
  and shape/memory tracking inflates the event buffer significantly.

Writes:

* ``results/e5_profiler_summary.csv`` -- aggregated per-family self-time
  with an explicit ``python_overhead`` row per label.  Each row also
  carries the profiled wall time so the decomposition can be
  sanity-checked (``python_overhead + Σ aten/autograd/other ≈ wall_s``).
* ``results/e5_trace_orig.json`` / ``results/e5_trace_comp.json`` --
  Chrome trace files.  Readable at https://ui.perfetto.dev/ or
  ``chrome://tracing``.

Memory discipline
-----------------
A full-horizon profile buffers one event per op dispatch for every step,
which at the default 7-day / 600 s setting produces on the order of a
million events per pass.  With two back-to-back passes plus the chrome
trace JSON held in memory on export, peak RSS can easily exceed a couple
of GB and OOM-kill under WSL.

This script therefore:

* only profiles ``config.PROFILE_N_STEPS`` steps of size
  ``config.PROFILE_STEP_SIZE`` (the op-family aggregate is an average,
  so a short slice is sufficient);
* runs the two profile passes in fully isolated scopes so the profiler,
  its event buffer, and the model under measurement are released with an
  explicit ``del`` + ``gc.collect()`` between passes;
* optionally skips the chrome-trace export (``PROFILE_EXPORT_TRACE``),
  which is the single largest in-memory step of a pass.

The script is CPU-only in this pass.  A commented switch at the top of
``_profile_once`` shows how to add CUDA activity for a future GPU pass.
"""

from __future__ import annotations

import datetime
import gc
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Tuple

from torch.profiler import ProfilerActivity, profile

from twin4build.examples.paper_experiments import common, config


CSV_PATH = config.RESULTS_DIR / "e5_profiler_summary.csv"
TRACE_ORIG = config.RESULTS_DIR / "e5_trace_orig.json"
TRACE_COMP = config.RESULTS_DIR / "e5_trace_comp.json"


def _classify_op(name: str) -> str:
    """Coarse grouping for the stacked-bar plot.

    The profiler records CPU-side *operator* events (ATen dispatches,
    autograd scopes, user annotations, and on GPU builds CUDA kernel
    launches).  Python interpreter time is NOT recorded as events here --
    it is accounted for in the derived ``python_overhead`` row computed
    from ``wall_time - sum(self_cpu_time)``, so no Python bucket is
    needed in this classifier.
    """
    if name.startswith("aten::"):
        return "aten"
    if name.startswith("autograd::"):
        return "autograd"
    if name.startswith("cudaLaunchKernel") or name.startswith("cuda"):
        return "cuda_launch"
    return "other"


def _profile_once(
    simulate_callable: Callable[[], object],
    label: str,
    trace_path: Path,
    export_trace: bool,
) -> List[Dict[str, object]]:
    """Profile one run and return per-family aggregate rows.

    Uses ``self_cpu_time_total`` (not ``cpu_time_total``) so parent events
    do not double-count their children, and times simulate wall clock
    inside the profiler scope so a synthetic ``python_overhead`` row can
    be derived as ``wall - Σ self_cpu_time``.  That derived row is what
    makes the Python-vs-tensor comparison honest: PyTorch's CPU profiler
    does not record events for Python interpreter time, so "Python
    overhead" never appears as its own event key -- it is the residual
    between wall clock and the sum of attributed op self-times.

    The heavy profiler object is deliberately scoped to this function so
    it is released as soon as aggregation and (optional) trace export are
    done.  ``gc.collect()`` is called before returning so the caller sees
    the smallest possible residual footprint.
    """
    # To enable CUDA profiling in a future GPU pass, uncomment the CUDA
    # activity below and ensure the model's tensors live on a cuda device.
    activities = [
        ProfilerActivity.CPU,
        # ProfilerActivity.CUDA,
    ]
    family_self_us: Dict[str, float] = {}
    family_counts: Dict[str, int] = {}

    with profile(
        activities=activities,
        record_shapes=False,
        with_stack=False,
        profile_memory=False,
        with_flops=False,
    ) as prof:
        # Time the simulate call INSIDE the profiler context so profiler
        # __enter__/__exit__ setup is excluded from wall time.  The small
        # profiler per-op instrumentation overhead is still inside
        # ``wall_s``; it is symmetric across the original/compiled runs so
        # their Python-overhead *delta* remains meaningful.
        t0 = time.perf_counter()
        simulate_callable()
        wall_s = time.perf_counter() - t0

    # Aggregate FIRST (cheap; ``key_averages`` returns a small summary list)
    # so we can drop the trace JSON and the profiler object promptly on the
    # error path as well.  ``self_cpu_time_total`` is the per-event time
    # with child-event time subtracted out -- additive across the event
    # list, unlike ``cpu_time_total`` which recursively includes children.
    total_self_us = 0.0
    for evt in prof.key_averages():
        family = _classify_op(evt.key)
        self_us = float(evt.self_cpu_time_total)
        family_self_us[family] = family_self_us.get(family, 0.0) + self_us
        family_counts[family] = (
            family_counts.get(family, 0) + int(evt.count)
        )
        total_self_us += self_us

    if export_trace:
        # The chrome-trace export is the single largest in-memory step of a
        # profile pass: kineto serializes the full event list to JSON in a
        # buffer before writing.  Do it last so ``prof`` can be released
        # immediately afterwards.
        prof.export_chrome_trace(str(trace_path))

    # Explicitly drop the profiler so its internal event buffer is freed
    # before we return to the caller.
    del prof
    gc.collect()

    # Derived Python/glue overhead.  Everything the profiler did not
    # attribute to a recorded op -- Python dispatch, twin4build per-step
    # bookkeeping, tensor-arg boxing, plus a small symmetric profiler
    # instrumentation cost -- falls here.
    attributed_self_s = total_self_us / 1e6
    python_overhead_s = max(0.0, wall_s - attributed_self_s)

    rows: List[Dict[str, object]] = []
    for family, total_us in sorted(family_self_us.items()):
        rows.append(
            {
                "label": label,
                "family": family,
                "cpu_self_time_s": total_us / 1e6,
                "n_events": family_counts.get(family, 0),
                "wall_s": wall_s,
            }
        )
    rows.append(
        {
            "label": label,
            "family": "python_overhead",
            "cpu_self_time_s": python_overhead_s,
            "n_events": 0,
            "wall_s": wall_s,
        }
    )
    return rows


def _profile_pass(
    build_model: Callable[[], object],
    label: str,
    trace_path: Path,
    start: datetime.datetime,
    end: datetime.datetime,
    step_size: int,
    export_trace: bool,
) -> Tuple[List[Dict[str, object]], float]:
    """Build one model, warm it, profile it, release it.

    Everything model- or profiler-related is held in local scope so the
    reference count drops to zero on return.  This keeps peak RSS bounded
    to one model + one profile buffer at a time instead of two of each.
    """
    model = build_model()
    print(f"[E5]   warm-up ({label}) ...")
    common.simulate_once(model, start, end, step_size)

    print(f"[E5]   profiling {label} ...")
    rows = _profile_once(
        simulate_callable=lambda: common.simulate_once(
            model, start, end, step_size
        ),
        label=label,
        trace_path=trace_path,
        export_trace=export_trace,
    )
    if export_trace:
        print(f"[E5]     trace -> {trace_path}")

    peak = common.peak_memory_mb("cpu-mt")

    # Release the model (and its histories/tensors) before returning so the
    # next pass starts from a clean slate.
    del model
    gc.collect()
    return rows, peak


def main() -> None:
    n_rooms = config.N_E5
    n_steps = config.PROFILE_N_STEPS
    step = config.PROFILE_STEP_SIZE
    export_trace = config.PROFILE_EXPORT_TRACE

    horizon_seconds = n_steps * step
    start = common.DEFAULT_START
    end = start + datetime.timedelta(seconds=horizon_seconds)
    # Weather DataFrame is built against ``horizon_days``; round up so it
    # covers the profile window even when it is sub-daily.
    horizon_days = max(1, (horizon_seconds + 86399) // 86400)

    print(
        f"[E5] profiling at N_ROOMS={n_rooms} on CPU "
        f"(profile window: {n_steps} steps of {step}s "
        f"= {horizon_seconds/3600:.2f}h; "
        f"export_trace={export_trace})"
    )
    common.set_cpu_threads("cpu-mt", config.DEVICES.get("cpu-mt", 1))

    # Each pass gets a *fresh* model built in its own scope.  We could share
    # the reference model to avoid rebuilding, but keeping one model alive
    # across both passes was doubling peak RSS (model + 2x histories + both
    # profile buffers) and OOM-killing WSL at N_E5 = 8.  Rebuilding is cheap
    # compared to a simulation pass.
    def _build_original() -> object:
        print("[E5]   building original model ...")
        m = common.build_multi_room_model(
            n_rooms,
            start=start,
            horizon_days=int(horizon_days),
            step_size=step,
            model_id=f"e5_n{n_rooms}",
        )
        m.load(
            draw_semantic_model=False,
            draw_simulation_model=False,
            verbose=0,
        )
        return m

    def _build_compiled() -> object:
        print("[E5]   building compiled model ...")
        base = common.build_multi_room_model(
            n_rooms,
            start=start,
            horizon_days=int(horizon_days),
            step_size=step,
            model_id=f"e5_n{n_rooms}",
        )
        base.load(
            draw_semantic_model=False,
            draw_simulation_model=False,
            verbose=0,
        )
        compiled = base.build_compiled_model()
        compiled.load(
            draw_semantic_model=False,
            draw_simulation_model=False,
            verbose=0,
        )
        # Drop the non-compiled base before returning so we do not hold two
        # fully-loaded models at the same time during the compiled pass.
        del base
        gc.collect()
        return compiled

    all_rows: List[Dict[str, object]] = []

    rows_orig, peak_orig = _profile_pass(
        build_model=_build_original,
        label="original",
        trace_path=TRACE_ORIG,
        start=start,
        end=end,
        step_size=step,
        export_trace=export_trace,
    )
    all_rows.extend(rows_orig)
    print(f"[E5]   peak RSS after original pass: {peak_orig:.1f} MB")

    rows_comp, peak_comp = _profile_pass(
        build_model=_build_compiled,
        label="compiled",
        trace_path=TRACE_COMP,
        start=start,
        end=end,
        step_size=step,
        export_trace=export_trace,
    )
    all_rows.extend(rows_comp)
    print(f"[E5]   peak RSS after compiled pass: {peak_comp:.1f} MB")

    common.write_csv(CSV_PATH, all_rows)
    print(f"\n[E5] wrote summary -> {CSV_PATH}")

    # Pretty print so the script is useful on its own.  The ``wall_s``
    # row is the profiled wall time per label; the family columns sum to
    # approximately ``wall_s`` by construction (python_overhead closes the
    # gap).
    totals: Dict[str, Dict[str, float]] = {}
    walls: Dict[str, float] = {}
    for r in all_rows:
        totals.setdefault(r["label"], {})[r["family"]] = r["cpu_self_time_s"]
        walls[r["label"]] = float(r.get("wall_s", float("nan")))
    families = sorted({f for fam in totals.values() for f in fam})
    print("\n[E5] CPU self-time (s) per family  (python_overhead = wall - Σ op self-time):")
    header = (
        "  " + "label".ljust(10) + "  "
        + "  ".join(f.ljust(16) for f in families)
        + "  " + "wall_s".ljust(10)
    )
    print(header)
    for label, fam_map in totals.items():
        row = (
            "  " + label.ljust(10) + "  "
            + "  ".join(f"{fam_map.get(f, 0.0):16.3f}" for f in families)
            + f"  {walls.get(label, float('nan')):10.3f}"
        )
        print(row)

    # Also print counts so the reader can see the dispatch-volume story.
    counts: Dict[str, Dict[str, int]] = {}
    for r in all_rows:
        counts.setdefault(r["label"], {})[r["family"]] = int(r["n_events"])
    print("\n[E5] dispatch counts per family:")
    print(header)
    for label, fam_map in counts.items():
        row = (
            "  " + label.ljust(10) + "  "
            + "  ".join(f"{fam_map.get(f, 0):16d}" for f in families)
            + f"  {'':>10s}"
        )
        print(row)


if __name__ == "__main__":
    sys.exit(main())
