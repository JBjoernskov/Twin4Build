"""F6 -- Profiler stacked bars (CPU self-time per op family).

Reads ``results/e5_profiler_summary.csv`` and plots a two-bar stacked
chart (original vs compiled) with one segment per op family.  The
complementary view -- event counts -- is drawn in the right-hand panel so
readers can see both "time spent" and "number of dispatches".
"""

from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from twin4build.examples.paper_experiments import config


CSV = config.RESULTS_DIR / "e5_profiler_summary.csv"
OUT = config.FIGURES_DIR / "f6_profiler"


def main() -> None:
    if not CSV.exists():
        print(f"[F6] missing {CSV} -- run E5 first", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(CSV)
    if df.empty:
        print(f"[F6] no rows in {CSV}", file=sys.stderr)
        sys.exit(1)

    labels = ["original", "compiled"]
    labels = [l for l in labels if l in df["label"].unique()]
    families = sorted(df["family"].unique())

    # ``python_overhead`` is a derived residual, not a dispatched op.  Keep
    # it in the self-time panel (that is the whole point of the
    # decomposition) but drop it from the dispatch-count panel so it does
    # not sit at zero-height in the legend and color budget.
    families_time = families
    families_count = [f for f in families if f != "python_overhead"]

    # Pivot: rows = family, cols = label, values = cpu_self_time_s
    time_pivot = (
        df.pivot_table(index="family", columns="label", values="cpu_self_time_s", fill_value=0.0)
        .reindex(index=families_time, columns=labels, fill_value=0.0)
    )
    count_pivot = (
        df.pivot_table(index="family", columns="label", values="n_events", fill_value=0)
        .reindex(index=families_count, columns=labels, fill_value=0)
    )

    # Stable family->color across both panels -- same family gets the same
    # hue even though the count panel skips ``python_overhead``.
    palette = plt.cm.tab10(np.linspace(0, 0.9, max(len(families_time), 2)))
    fam_color = {fam: palette[i] for i, fam in enumerate(families_time)}

    fig, (ax_t, ax_c) = plt.subplots(1, 2, figsize=(10, 4.5))
    x = np.arange(len(labels))
    bar_width = 0.6

    bottom = np.zeros(len(labels))
    for fam in families_time:
        vals = time_pivot.loc[fam].to_numpy()
        ax_t.bar(
            x, vals, bar_width, bottom=bottom, label=fam, color=fam_color[fam]
        )
        bottom += vals
    ax_t.set_xticks(x)
    ax_t.set_xticklabels(labels)
    ax_t.set_ylabel("CPU self-time [s]")
    ax_t.set_title("Where wall time goes (one simulate pass)")
    ax_t.legend(fontsize=8, loc="upper right")
    ax_t.grid(True, axis="y", alpha=0.3)

    # Reference marker: the profiled wall time per label.  If the
    # decomposition is well-formed, the top of each stack should land on
    # (or very near) this tick -- i.e. python_overhead + Σ op self-time
    # ≈ wall_s.  Drawn as a short horizontal tick per bar so it reads as
    # a sanity check rather than a second data series.
    if "wall_s" in df.columns:
        wall_by_label = (
            df.drop_duplicates(subset=["label"])
            .set_index("label")["wall_s"]
        )
        for xi, lbl in zip(x, labels):
            wall = float(wall_by_label.get(lbl, float("nan")))
            if np.isfinite(wall):
                ax_t.hlines(
                    wall,
                    xi - bar_width / 2,
                    xi + bar_width / 2,
                    colors="black",
                    linewidth=1.2,
                    linestyles="-",
                    label="wall_s" if xi == x[0] else None,
                )
        ax_t.legend(fontsize=8, loc="upper right")

    bottom = np.zeros(len(labels))
    for fam in families_count:
        vals = count_pivot.loc[fam].to_numpy()
        ax_c.bar(
            x, vals, bar_width, bottom=bottom, label=fam, color=fam_color[fam]
        )
        bottom += vals
    ax_c.set_xticks(x)
    ax_c.set_xticklabels(labels)
    ax_c.set_ylabel("Number of events (dispatches)")
    ax_c.set_title("Op dispatch counts")
    ax_c.set_yscale("log")
    ax_c.legend(fontsize=8, loc="upper right")
    ax_c.grid(True, axis="y", which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".png"), bbox_inches="tight", dpi=160)
    print(f"[F6] wrote {OUT}.pdf / .png")


if __name__ == "__main__":
    sys.exit(main())
