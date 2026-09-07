"""Run the canonical benchmark implementations as local correctness smoke tests.

This runner deliberately uses ``BenchmarkConfig(mode="smoke")``. Its timing
values are diagnostic only and must not be published as benchmark results.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

# Direct ``python benchmarks/run_smoke.py`` puts only ``benchmarks/`` on
# sys.path. Add the repository root so the package import below works in both
# direct-script and ``python -m benchmarks.run_smoke`` forms.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.common import (
    BenchmarkConfig,
    run_estimation_scaling,
    run_optimization_scaling,
    run_simulation_matrix,
    seed_everything,
    serialize_results,
)

RUNNERS: dict[str, Callable[[BenchmarkConfig], list[dict[str, Any]]]] = {
    "simulation_scaling": run_simulation_matrix,
    "estimation_scaling": run_estimation_scaling,
    "optimization_scaling": run_optimization_scaling,
}


def _validate_rows(name: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise AssertionError(f"{name} produced no result rows")

    valid_statuses = {"ok", "nonconverged", "failed", "skipped"}
    statuses = {row.get("status") for row in rows}
    if not statuses <= valid_statuses:
        raise AssertionError(f"{name} produced invalid statuses: {statuses}")
    failed = [row for row in rows if row.get("status") == "failed"]
    if failed:
        details = [
            {
                "device": row.get("device"),
                "solver": row.get("solver"),
                "error": row.get("error"),
            }
            for row in failed
        ]
        raise AssertionError(f"{name} produced failed cases: {details}")
    if not statuses.intersection({"ok", "nonconverged"}):
        raise AssertionError(f"{name} did not execute any supported case")

    for row in rows:
        for field in ("model_layout", "execution_mode", "execution_backend"):
            if field not in row:
                raise AssertionError(f"{name} row is missing {field}")
        if row.get("status") not in {"ok", "nonconverged", "failed"}:
            continue
        for field in ("seconds", "batch_seconds"):
            value = row.get(field)
            if value is not None and (
                not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise AssertionError(f"{name} produced invalid {field}={value!r}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run canonical benchmark code in smoke mode. Timings are not "
            "publication-quality benchmark results."
        )
    )
    parser.add_argument(
        "benchmarks",
        nargs="*",
        choices=sorted(RUNNERS),
        help="Subset to run; defaults to all three scaling notebooks.",
    )
    args = parser.parse_args()

    config = BenchmarkConfig(mode="smoke")
    seed_everything(config.seed)
    selected = args.benchmarks or list(RUNNERS)

    for name in selected:
        print(f"[smoke] running {name}")
        rows = RUNNERS[name](config)
        _validate_rows(name, rows)
        result_path = serialize_results(f"{name}_smoke", config, rows)
        print(f"[smoke] {name}: passed ({result_path})")


if __name__ == "__main__":
    main()
