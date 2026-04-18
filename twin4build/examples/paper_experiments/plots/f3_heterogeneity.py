"""F3 -- Speedup vs heterogeneity (k archetypes).

Reads ``results/e2_heterogeneity.csv`` and produces a single-panel bar
chart of median speedup per ``k``, with IQR whiskers.
"""

from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from twin4build.examples.paper_experiments import config


CSV = config.RESULTS_DIR / "e2_heterogeneity.csv"
OUT = config.FIGURES_DIR / "f3_heterogeneity"


def main() -> None:
    if not CSV.exists():
        print(f"[F3] missing {CSV} -- run E2 first", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(CSV).sort_values("k_archetypes")
    if df.empty:
        print(f"[F3] no rows in {CSV}", file=sys.stderr)
        sys.exit(1)

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    x = np.arange(len(df))
    medians = df["speedup_median"].to_numpy()
    # Back out asymmetric whiskers from the compiled IQR only, since orig
    # is approximately constant across k.
    comp_low = df["t_sim_comp_q75"].to_numpy()
    comp_high = df["t_sim_comp_q25"].to_numpy()
    orig_med = df["t_sim_orig_median"].to_numpy()
    speedup_low = orig_med / comp_low
    speedup_high = orig_med / comp_high
    # Whiskers can come out slightly inverted when the sample is tiny; clip
    # to non-negative so matplotlib's errorbar doesn't abort the whole plot.
    yerr_low = np.clip(medians - speedup_low, 0.0, None)
    yerr_high = np.clip(speedup_high - medians, 0.0, None)

    ax.bar(x, medians, yerr=[yerr_low, yerr_high], capsize=4, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"k={int(k)}\n(n={int(n)})" for k, n in zip(df["k_archetypes"], df["n_per_archetype"])]
    )
    ax.set_ylabel("Speedup (original / compiled)")
    ax.set_xlabel("Number of archetypes (and rooms per archetype)")
    ax.set_title(
        f"Speedup vs heterogeneity (N_ROOMS={int(df['n_rooms'].iloc[0])})"
    )
    ax.axhline(1.0, color="grey", linestyle="--", alpha=0.6)
    ax.grid(True, axis="y", alpha=0.3)

    # Annotate compiled-model size per bar to explain the drop.
    for xi, (_, row) in zip(x, df.iterrows()):
        ax.text(
            xi,
            medians[xi],
            f"meta={int(row['n_comp_comp'])}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".png"), bbox_inches="tight", dpi=160)
    print(f"[F3] wrote {OUT}.pdf / .png")


if __name__ == "__main__":
    sys.exit(main())
