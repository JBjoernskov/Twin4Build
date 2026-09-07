"""Fresh-process entry point for one canonical estimation benchmark case."""

from __future__ import annotations

import argparse
import json
import os
import traceback
from pathlib import Path
from typing import Any

from benchmarks.common import (
    BenchmarkConfig,
    _jsonable,
    run_one_estimation_case,
)


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_jsonable(payload), indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run exactly one estimation benchmark case."
    )
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()

    request = json.loads(args.request.read_text(encoding="utf-8"))
    child_pid = os.getpid()
    try:
        row = run_one_estimation_case(
            BenchmarkConfig(**request["config"]),
            n_zones=int(request["n_zones"]),
            device=request["device"],
            solver=request["solver"],
            n_starts=int(request["n_starts"]),
            repetition=int(request["repetition"]),
        )
    except BaseException as exc:
        _atomic_write(
            args.result,
            {
                "ok": False,
                "child_pid": child_pid,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            },
        )
        return 1

    _atomic_write(
        args.result,
        {"ok": True, "child_pid": child_pid, "row": row},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
