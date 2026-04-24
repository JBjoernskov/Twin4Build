"""F5 -- Accuracy: residual timeseries + CDF of per-port max errors.

Reads ``results/e4_port_errors.csv`` and ``results/e4_residuals/n{N}/*.csv``
and produces a two-panel figure that compares the small and large multi-
room cases audited by E4:

* Top:    residual timeseries for the three representative ports, one
          panel-row per ``n_rooms`` case.
* Bottom: CDF (sorted values) of per-port max absolute error across every
          output port, one line per ``n_rooms`` case.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from twin4build.examples.paper_experiments import config


CSV_ERRORS = config.RESULTS_DIR / "e4_port_errors.csv"
RESIDUALS_DIR = config.RESULTS_DIR / "e4_residuals"
OUT = config.FIGURES_DIR / "f5_accuracy"


def _discover_residual_dirs() -> Dict[int, Path]:
    """Return ``{n_rooms: dir}`` for each ``n{N}`` subfolder with CSVs.

    Falls back to the legacy flat layout (``e4_residuals/*.csv`` with no
    ``n_rooms`` subfolder) under key ``-1`` so re-running F5 against an
    older E4 result set still produces a plot.
    """
    if not RESIDUALS_DIR.exists():
        return {}
    dirs: Dict[int, Path] = {}
    for sub in sorted(RESIDUALS_DIR.iterdir()):
        if sub.is_dir() and sub.name.startswith("n"):
            try:
                n = int(sub.name[1:])
            except ValueError:
                continue
            if any(sub.glob("*.csv")):
                dirs[n] = sub
    if not dirs and any(RESIDUALS_DIR.glob("*.csv")):
        dirs[-1] = RESIDUALS_DIR
    return dirs


def _plot_residuals(
    ax: plt.Axes, case_label: str, residual_dir: Path
) -> None:
    files = sorted(residual_dir.glob("*.csv"))
    if not files:
        ax.text(
            0.5,
            0.5,
            f"No residuals for {case_label}.",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )
        return
    for path in files:
        df = pd.read_csv(path)
        first_sim = df[df["sim_index"] == df["sim_index"].min()]
        label = path.stem.replace("__", " / ")
        ax.plot(
            first_sim["t_index"],
            first_sim["residual"],
            label=label,
            linewidth=0.9,
        )
    ax.axhline(0.0, color="grey", linestyle=":", alpha=0.6)
    ax.set_xlabel("Timestep index")
    ax.set_ylabel("Residual (orig - comp)")
    ax.set_title(f"Residuals ({case_label}, sim 0)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)


def _plot_error_cdf(
    ax: plt.Axes, valid_by_case: Dict[int, pd.DataFrame]
) -> None:
    if not valid_by_case:
        ax.text(
            0.5,
            0.5,
            "No valid port comparisons.",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )
        return

    any_positive = False
    total_ports = 0
    for n_rooms, valid in sorted(valid_by_case.items()):
        sorted_err = np.sort(valid["max_abs_err"].to_numpy())
        cdf = np.arange(1, len(sorted_err) + 1) / len(sorted_err)
        label = (
            f"n_rooms={n_rooms}  (n={len(sorted_err)})"
            if n_rooms >= 0
            else f"all  (n={len(sorted_err)})"
        )
        plot_err = np.where(sorted_err > 0, sorted_err, np.nextafter(0, 1))
        ax.plot(plot_err, cdf, marker=".", linewidth=1.0, label=label)
        if np.any(sorted_err > 0):
            any_positive = True
        total_ports += len(sorted_err)

    if any_positive:
        ax.set_xscale("log")
        ax.axvline(
            1e-3, color="red", linestyle="--", alpha=0.6, label="1e-3 tolerance"
        )
    else:
        ax.text(
            0.02,
            0.5,
            "All ports matched to machine precision (max abs err = 0).",
            transform=ax.transAxes,
            va="center",
            fontsize=9,
        )
    ax.set_xlabel("Per-port max absolute error")
    ax.set_ylabel("CDF across ports")
    ax.set_title(
        f"Per-port max abs error across {total_ports} output ports"
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)


def main() -> None:
    if not CSV_ERRORS.exists():
        print(f"[F5] missing {CSV_ERRORS} -- run E4 first", file=sys.stderr)
        sys.exit(1)

    err_df = pd.read_csv(CSV_ERRORS)
    valid_all = err_df.dropna(subset=["max_abs_err"])
    valid_all = valid_all[valid_all["n_samples"] > 0]

    # Split by n_rooms when E4 produced the multi-case layout; otherwise
    # fall back to a single all-rows bucket so legacy CSVs still plot.
    valid_by_case: Dict[int, pd.DataFrame] = {}
    if "n_rooms" in valid_all.columns:
        for n_rooms, group in valid_all.groupby("n_rooms"):
            valid_by_case[int(n_rooms)] = group
    elif not valid_all.empty:
        valid_by_case[-1] = valid_all

    residual_dirs = _discover_residual_dirs()
    case_keys: List[int] = sorted(
        set(valid_by_case.keys()) | set(residual_dirs.keys())
    )
    # Drop the legacy sentinel if real cases are present.
    if any(k >= 0 for k in case_keys):
        case_keys = [k for k in case_keys if k >= 0]

    n_residual_rows = max(1, len(case_keys))
    fig, axes = plt.subplots(
        n_residual_rows + 1, 1, figsize=(8.5, 3.2 * n_residual_rows + 3.2)
    )
    if n_residual_rows + 1 == 1:
        axes = [axes]

    if case_keys:
        for ax, n_rooms in zip(axes[:-1], case_keys):
            case_label = f"n_rooms={n_rooms}" if n_rooms >= 0 else "legacy"
            residual_dir = residual_dirs.get(n_rooms)
            if residual_dir is None:
                ax.text(
                    0.5,
                    0.5,
                    f"No residuals for {case_label}.",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                )
                continue
            _plot_residuals(ax, case_label, residual_dir)
    else:
        axes[0].text(
            0.5,
            0.5,
            "No residual timeseries found\n(rerun E4 to populate e4_residuals/).",
            transform=axes[0].transAxes,
            ha="center",
            va="center",
        )

    _plot_error_cdf(axes[-1], valid_by_case)

    fig.tight_layout()
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".png"), bbox_inches="tight", dpi=160)
    print(f"[F5] wrote {OUT}.pdf / .png")


if __name__ == "__main__":
    sys.exit(main())
