"""F2 -- Speedup & memory tradeoff vs N_ROOMS.

Reads ``results/e1_n_rooms.csv``, aggregates per-cell medians + IQRs, and
emits a 2x2 figure that exposes both axes of the compiler tradeoff:

* (0,0) log-log wallclock time vs N, original vs compiled.
* (0,1) speedup vs N (original / compiled), with a dashed 1x line.
* (1,0) peak memory vs N, original vs compiled (MB, log y).
* (1,1) memory-overhead ratio compiled / original vs N, with a dashed 1x
  line -- values >1 mean the compiled model uses more RAM than the
  reference for the same simulation.

Peak memory is aggregated as the *max across replicates* (not median),
matching how a user would have to provision for the workload.  If the CSV
predates the memory instrumentation, the memory panels are skipped and the
old 1x2 layout is emitted for backwards compatibility.

Saves to ``figures/f2_speedup_vs_n.pdf`` (and .png alongside).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from twin4build.examples.paper_experiments import config


CSV = config.RESULTS_DIR / "e1_n_rooms.csv"
OUT = config.FIGURES_DIR / "f2_speedup_vs_n"


def _aggregate(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["is_warmup"] == False].copy()  # noqa: E712
    has_mem = {"rss_peak_orig_mb", "rss_peak_comp_mb"}.issubset(df.columns)
    agg_spec: Dict[str, tuple] = {
        "orig_q25": ("t_sim_orig_s", lambda s: np.percentile(s, 25)),
        "orig_median": ("t_sim_orig_s", "median"),
        "orig_q75": ("t_sim_orig_s", lambda s: np.percentile(s, 75)),
        "comp_q25": ("t_sim_comp_s", lambda s: np.percentile(s, 25)),
        "comp_median": ("t_sim_comp_s", "median"),
        "comp_q75": ("t_sim_comp_s", lambda s: np.percentile(s, 75)),
    }
    if has_mem:
        # Peak memory is a high-water mark: use max across replicates, not
        # median, because that is the allocation a user has to provision.
        agg_spec["mem_orig_max"] = ("rss_peak_orig_mb", "max")
        agg_spec["mem_comp_max"] = ("rss_peak_comp_mb", "max")

    agg = df.groupby(["device", "n_rooms"]).agg(**agg_spec).reset_index()
    agg["speedup_median"] = agg["orig_median"] / agg["comp_median"]
    if has_mem:
        agg["mem_ratio"] = agg["mem_comp_max"] / agg["mem_orig_max"]
    return agg


def _device_colors(devices) -> Dict[str, tuple]:
    return {
        d: c
        for d, c in zip(
            devices,
            plt.cm.tab10(np.linspace(0, 0.9, max(len(devices), 2))),
        )
    }


def _plot_wallclock(ax, agg: pd.DataFrame, colors: Dict[str, tuple]) -> None:
    for device in sorted(agg["device"].unique()):
        sub = agg[agg["device"] == device].sort_values("n_rooms")
        color = colors[device]
        ax.errorbar(
            sub["n_rooms"],
            sub["orig_median"],
            yerr=[sub["orig_median"] - sub["orig_q25"],
                  sub["orig_q75"] - sub["orig_median"]],
            fmt="o-",
            color=color,
            label=f"{device} original",
        )
        ax.errorbar(
            sub["n_rooms"],
            sub["comp_median"],
            yerr=[sub["comp_median"] - sub["comp_q25"],
                  sub["comp_q75"] - sub["comp_median"]],
            fmt="s--",
            color=color,
            label=f"{device} compiled",
        )

    if len(agg):
        first = agg.sort_values(["device", "n_rooms"]).iloc[0]
        ns = np.asarray(sorted(agg["n_rooms"].unique()))
        ideal = first["orig_median"] * ns / first["n_rooms"]
        ax.plot(ns, ideal, ":", color="grey", label="ideal linear (original)")

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Number of rooms")
    ax.set_ylabel("Simulate wallclock [s]")
    ax.set_title("Simulation wallclock vs problem size")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)


def _plot_speedup(ax, agg: pd.DataFrame, colors: Dict[str, tuple]) -> None:
    for device in sorted(agg["device"].unique()):
        sub = agg[agg["device"] == device].sort_values("n_rooms")
        ax.plot(
            sub["n_rooms"],
            sub["speedup_median"],
            "o-",
            color=colors[device],
            label=device,
        )
    ax.axhline(1.0, color="grey", linestyle="--", alpha=0.6)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Number of rooms")
    ax.set_ylabel("Speedup (original / compiled)")
    ax.set_title("Compiler speedup vs problem size")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)


def _plot_memory(ax, agg: pd.DataFrame, colors: Dict[str, tuple]) -> None:
    for device in sorted(agg["device"].unique()):
        sub = agg[agg["device"] == device].sort_values("n_rooms")
        color = colors[device]
        ax.plot(
            sub["n_rooms"],
            sub["mem_orig_max"],
            "o-",
            color=color,
            label=f"{device} original",
        )
        ax.plot(
            sub["n_rooms"],
            sub["mem_comp_max"],
            "s--",
            color=color,
            label=f"{device} compiled",
        )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Number of rooms")
    ax.set_ylabel("Peak memory [MB]")
    ax.set_title("Peak memory vs problem size")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)


def _plot_memory_ratio(ax, agg: pd.DataFrame, colors: Dict[str, tuple]) -> None:
    for device in sorted(agg["device"].unique()):
        sub = agg[agg["device"] == device].sort_values("n_rooms")
        ax.plot(
            sub["n_rooms"],
            sub["mem_ratio"],
            "o-",
            color=colors[device],
            label=device,
        )
    ax.axhline(1.0, color="grey", linestyle="--", alpha=0.6)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Number of rooms")
    ax.set_ylabel("Memory ratio (compiled / original)")
    ax.set_title("Compile memory overhead vs problem size")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)


def main() -> None:
    if not CSV.exists():
        print(f"[F2] missing {CSV} -- run E1 first", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(CSV)
    agg = _aggregate(df)
    if agg.empty:
        print(f"[F2] no measured rows in {CSV}", file=sys.stderr)
        sys.exit(1)

    colors = _device_colors(sorted(agg["device"].unique()))
    has_mem = {"mem_orig_max", "mem_comp_max", "mem_ratio"}.issubset(agg.columns)

    if has_mem:
        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        _plot_wallclock(axes[0, 0], agg, colors)
        _plot_speedup(axes[0, 1], agg, colors)
        _plot_memory(axes[1, 0], agg, colors)
        _plot_memory_ratio(axes[1, 1], agg, colors)
    else:
        print(
            "[F2] memory columns missing from e1_n_rooms.csv -- "
            "falling back to 1x2 layout (re-run E1 to enable memory panels)",
            file=sys.stderr,
        )
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        _plot_wallclock(axes[0], agg, colors)
        _plot_speedup(axes[1], agg, colors)

    fig.tight_layout()
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".png"), bbox_inches="tight", dpi=160)
    print(f"[F2] wrote {OUT}.pdf / .png")


if __name__ == "__main__":
    sys.exit(main())
