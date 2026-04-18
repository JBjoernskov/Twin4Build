"""F2 -- Speedup vs N_ROOMS (left: wallclock, right: speedup).

Reads ``results/e1_n_rooms.csv``, aggregates per-cell medians + IQRs, and
emits a two-panel figure:

* Left:  log-log wallclock time vs N, original and compiled, one colour
         per device.
* Right: speedup vs N, one curve per device, with a dashed 1x line.

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
    agg = (
        df.groupby(["device", "n_rooms"])
        .agg(
            orig_q25=("t_sim_orig_s", lambda s: np.percentile(s, 25)),
            orig_median=("t_sim_orig_s", "median"),
            orig_q75=("t_sim_orig_s", lambda s: np.percentile(s, 75)),
            comp_q25=("t_sim_comp_s", lambda s: np.percentile(s, 25)),
            comp_median=("t_sim_comp_s", "median"),
            comp_q75=("t_sim_comp_s", lambda s: np.percentile(s, 75)),
        )
        .reset_index()
    )
    agg["speedup_median"] = agg["orig_median"] / agg["comp_median"]
    return agg


def main() -> None:
    if not CSV.exists():
        print(f"[F2] missing {CSV} -- run E1 first", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(CSV)
    agg = _aggregate(df)
    if agg.empty:
        print(f"[F2] no measured rows in {CSV}", file=sys.stderr)
        sys.exit(1)

    fig, (ax_t, ax_s) = plt.subplots(1, 2, figsize=(11, 4.5))
    devices = sorted(agg["device"].unique())
    colors: Dict[str, str] = {
        d: c
        for d, c in zip(
            devices,
            plt.cm.tab10(np.linspace(0, 0.9, max(len(devices), 2))),
        )
    }

    for device in devices:
        sub = agg[agg["device"] == device].sort_values("n_rooms")
        color = colors[device]
        # Wallclock curves
        ax_t.errorbar(
            sub["n_rooms"],
            sub["orig_median"],
            yerr=[sub["orig_median"] - sub["orig_q25"],
                  sub["orig_q75"] - sub["orig_median"]],
            fmt="o-",
            color=color,
            label=f"{device} original",
        )
        ax_t.errorbar(
            sub["n_rooms"],
            sub["comp_median"],
            yerr=[sub["comp_median"] - sub["comp_q25"],
                  sub["comp_q75"] - sub["comp_median"]],
            fmt="s--",
            color=color,
            label=f"{device} compiled",
        )
        # Speedup
        ax_s.plot(
            sub["n_rooms"],
            sub["speedup_median"],
            "o-",
            color=color,
            label=device,
        )

    # Ideal-linear reference (original wallclock grows linearly with N).
    if len(agg):
        first = agg.sort_values(["device", "n_rooms"]).iloc[0]
        ns = np.asarray(sorted(agg["n_rooms"].unique()))
        ideal = first["orig_median"] * ns / first["n_rooms"]
        ax_t.plot(ns, ideal, ":", color="grey", label="ideal linear (original)")

    ax_t.set_xscale("log", base=2)
    ax_t.set_yscale("log")
    ax_t.set_xlabel("Number of rooms")
    ax_t.set_ylabel("Simulate wallclock [s]")
    ax_t.set_title("Simulation wallclock vs problem size")
    ax_t.grid(True, which="both", alpha=0.3)
    ax_t.legend(fontsize=8)

    ax_s.axhline(1.0, color="grey", linestyle="--", alpha=0.6)
    ax_s.set_xscale("log", base=2)
    ax_s.set_xlabel("Number of rooms")
    ax_s.set_ylabel("Speedup (original / compiled)")
    ax_s.set_title("Compiler speedup vs problem size")
    ax_s.grid(True, which="both", alpha=0.3)
    ax_s.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".png"), bbox_inches="tight", dpi=160)
    print(f"[F2] wrote {OUT}.pdf / .png")


if __name__ == "__main__":
    sys.exit(main())
