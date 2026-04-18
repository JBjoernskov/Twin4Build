"""F5 -- Accuracy: residual timeseries + CDF of per-port max errors.

Reads ``results/e4_port_errors.csv`` and ``results/e4_residuals/*.csv``
and produces a two-panel figure:

* Top:    residual timeseries for the three representative ports.
* Bottom: CDF (really: sorted values) of per-port max absolute error
          across every output port of the estimator-example model.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from twin4build.examples.paper_experiments import config


CSV_ERRORS = config.RESULTS_DIR / "e4_port_errors.csv"
RESIDUALS_DIR = config.RESULTS_DIR / "e4_residuals"
OUT = config.FIGURES_DIR / "f5_accuracy"


def main() -> None:
    if not CSV_ERRORS.exists():
        print(f"[F5] missing {CSV_ERRORS} -- run E4 first", file=sys.stderr)
        sys.exit(1)

    err_df = pd.read_csv(CSV_ERRORS)
    valid = err_df.dropna(subset=["max_abs_err"])
    valid = valid[valid["n_samples"] > 0]

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(8.5, 7))

    # Top: residual timeseries
    residual_files = sorted(RESIDUALS_DIR.glob("*.csv")) if RESIDUALS_DIR.exists() else []
    if not residual_files:
        ax_top.text(
            0.5,
            0.5,
            "No residual timeseries found\n(rerun E4 to populate e4_residuals/).",
            transform=ax_top.transAxes,
            ha="center",
            va="center",
        )
    else:
        for path in residual_files:
            df = pd.read_csv(path)
            # One line per (sim_index); use sim_index == 0 only for clarity.
            first_sim = df[df["sim_index"] == df["sim_index"].min()]
            label = path.stem.replace("__", " / ")
            ax_top.plot(
                first_sim["t_index"],
                first_sim["residual"],
                label=label,
                linewidth=0.9,
            )
        ax_top.axhline(0.0, color="grey", linestyle=":", alpha=0.6)
        ax_top.set_xlabel("Timestep index")
        ax_top.set_ylabel("Residual (original - compiled)")
        ax_top.set_title("Per-port residual timeseries (sim 0)")
        ax_top.legend(fontsize=8)
        ax_top.grid(True, alpha=0.3)

    # Bottom: sorted per-port max abs error (pseudo-CDF on log-scale y).
    if valid.empty:
        ax_bot.text(
            0.5,
            0.5,
            "No valid port comparisons.",
            transform=ax_bot.transAxes,
            ha="center",
            va="center",
        )
    else:
        sorted_err = np.sort(valid["max_abs_err"].to_numpy())
        cdf = np.arange(1, len(sorted_err) + 1) / len(sorted_err)
        # If any positive errors exist use log-x; otherwise fall back to
        # linear so a perfect-match run still produces a readable panel.
        if np.any(sorted_err > 0):
            plot_err = np.where(sorted_err > 0, sorted_err, np.nextafter(0, 1))
            ax_bot.plot(plot_err, cdf, marker=".")
            ax_bot.set_xscale("log")
        else:
            ax_bot.plot(sorted_err, cdf, marker=".")
            ax_bot.text(
                0.02, 0.5,
                "All ports matched to machine precision (max abs err = 0).",
                transform=ax_bot.transAxes,
                va="center",
                fontsize=9,
            )
        ax_bot.set_xlabel("Per-port max absolute error")
        ax_bot.set_ylabel("CDF across ports")
        ax_bot.set_title(
            f"Per-port max abs error across {len(valid)} output ports"
        )
        ax_bot.grid(True, which="both", alpha=0.3)
        if np.any(sorted_err > 0):
            # Reference tolerance used by the library's validation harness.
            ax_bot.axvline(
                1e-3, color="red", linestyle="--", alpha=0.6,
                label="1e-3 tolerance",
            )
            ax_bot.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".png"), bbox_inches="tight", dpi=160)
    print(f"[F5] wrote {OUT}.pdf / .png")


if __name__ == "__main__":
    sys.exit(main())
