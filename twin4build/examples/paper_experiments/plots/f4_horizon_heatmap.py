"""F4 -- Speedup heatmap over (horizon, step_size).

Reads ``results/e3_horizon_step.csv`` and draws a 2D heatmap of median
speedup across the (horizon_days, step_size_s) grid.  Cells that were
skipped (infeasible) appear as NaN / grey.
"""

from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from twin4build.examples.paper_experiments import config


CSV = config.RESULTS_DIR / "e3_horizon_step.csv"
OUT = config.FIGURES_DIR / "f4_horizon_heatmap"


def main() -> None:
    if not CSV.exists():
        print(f"[F4] missing {CSV} -- run E3 first", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(CSV)
    if df.empty:
        print(f"[F4] no rows in {CSV}", file=sys.stderr)
        sys.exit(1)

    pivot = df.pivot_table(
        index="horizon_days",
        columns="step_size_s",
        values="speedup_median",
        aggfunc="mean",
    )

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    im = ax.imshow(
        pivot.values, aspect="auto", origin="lower", cmap="viridis"
    )
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{int(c)} s" for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{int(r)} d" for r in pivot.index])
    ax.set_xlabel("Step size")
    ax.set_ylabel("Horizon")
    ax.set_title(
        f"Speedup heatmap (N_ROOMS={int(df['n_rooms'].iloc[0])}, CPU)"
    )

    # Annotate cells with numeric values.
    for i, r in enumerate(pivot.index):
        for j, c in enumerate(pivot.columns):
            val = pivot.values[i, j]
            if np.isnan(val):
                continue
            ax.text(
                j,
                i,
                f"{val:.2f}x",
                ha="center",
                va="center",
                color="white" if val < np.nanmedian(pivot.values) else "black",
                fontsize=9,
            )

    fig.colorbar(im, ax=ax, label="Speedup")
    fig.tight_layout()
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".png"), bbox_inches="tight", dpi=160)
    print(f"[F4] wrote {OUT}.pdf / .png")


if __name__ == "__main__":
    sys.exit(main())
