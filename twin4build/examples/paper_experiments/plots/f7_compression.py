"""F7 -- Per-execution-group compression ratio (estimator-example case).

Reads ``results/e4_compression.csv`` and plots one bar per execution
group showing ``n_original / n_meta``.  Groups with compression == 1
(singletons -- typically schedules / outdoor env / sensors) appear at
the baseline.
"""

from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from twin4build.examples.paper_experiments import config


CSV = config.RESULTS_DIR / "e4_compression.csv"
OUT = config.FIGURES_DIR / "f7_compression"


def main() -> None:
    if not CSV.exists():
        print(f"[F7] missing {CSV} -- run E4 first", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(CSV).sort_values("group_idx")
    if df.empty:
        print(f"[F7] no rows in {CSV}", file=sys.stderr)
        sys.exit(1)

    compression = df["n_original"] / df["n_meta"].replace(0, np.nan)

    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    x = df["group_idx"].to_numpy()
    ax.bar(x, compression.to_numpy(), alpha=0.8, label="compression ratio")
    ax.axhline(1.0, color="grey", linestyle="--", alpha=0.6, label="no compression")

    for xi, r, m in zip(x, df["n_original"], df["n_meta"]):
        ax.text(
            xi,
            (r / m) if m else 1.0,
            f"{int(r)}/{int(m)}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_xlabel("Execution group index")
    ax.set_ylabel("n_original / n_meta")
    ax.set_title("Per-group compression ratio (estimator-example)")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".png"), bbox_inches="tight", dpi=160)
    print(f"[F7] wrote {OUT}.pdf / .png")


if __name__ == "__main__":
    sys.exit(main())
