"""Dump the environment / hardware snapshot to ``results/env.json``.

Invoked once at the start of :file:`run_all.sh` so every paper result can
be traced back to an exact torch/numpy/commit/hardware triple.
"""

from __future__ import annotations

import json
import sys

from twin4build.examples.paper_experiments import common, config


def main() -> None:
    info = common.env_snapshot()
    out = config.RESULTS_DIR / "env.json"
    out.write_text(json.dumps(info, indent=2))
    print(f"[env] wrote {out}")
    for k, v in info.items():
        print(f"  {k:24s} = {v}")


if __name__ == "__main__":
    sys.exit(main())
