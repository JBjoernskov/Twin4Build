"""Measure the ``small_matmul_max`` setting's crossover on this machine's GPU.

``python benchmarks/small_matmul_crossover.py [--dtype float32] [--matrices 16 960] [--sizes 8 12 16 24 32 64]``

Times one matrix exponential (``discrete_statespace_system._expm_ss``, the one
a compiled functional step uses) of random stable ``N x N`` matrices, forward
and backward, compiled, with its products written both ways: as a broadcast
multiply and sum (which Inductor fuses) and as ``bmm``.  ``N`` is the matrix
the setting is compared with (a block of ``n`` states: ``n + 1`` with
``zoh_expm="aug"``, ``n`` with ``"phi1"``); ``--matrices`` is how many are
exponentiated per call (blocks x shooting windows of the runs to tune for).
Prints the times and, per batch, the largest size up to which the fused form
is faster: the value to give ``small_matmul_max`` (``T4B_SMALL_MATMUL_MAX`` or
``tb.settings.set(small_matmul_max=...)``).
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence

# Direct ``python benchmarks/small_matmul_crossover.py`` puts only
# ``benchmarks/`` on sys.path: add the repository root for ``twin4build``.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from twin4build import settings
from twin4build.systems.utils import discrete_statespace_system as dss


def _matrices(n: int, batch: int, dtype, device, seed: int = 0) -> torch.Tensor:
    g = torch.Generator(device=device).manual_seed(seed)
    M = torch.rand(batch, n, n, generator=g, device=device, dtype=dtype) * (30.0 / n)
    return M - torch.diag_embed(M.sum(-1) + 1.0)  # diagonally dominant: stable, |M| ~ 60 like a stiff step


def _time(n: int, batch: int, products: str, dtype, device, reps: int) -> float:
    """Median ms of one compiled exponential, forward and backward."""
    try:
        # the setting is read while torch.compile traces: every size fused, or none
        with settings.override(small_matmul_max=10**6 if products == "small" else 0):
            torch._dynamo.reset()
            M = _matrices(n, batch, dtype, device).requires_grad_(True)
            f = torch.compile(lambda m: dss._expm_ss(m).square().sum(), fullgraph=True, dynamic=False)

            def call():
                return torch.autograd.grad(f(M), M)

            for _ in range(3):
                call()
            torch.cuda.synchronize(device)
            times = []
            for _ in range(reps):
                t0 = time.perf_counter()
                call()
                torch.cuda.synchronize(device)
                times.append((time.perf_counter() - t0) * 1e3)
            return statistics.median(times)
    finally:
        torch._dynamo.reset()


def measure(
    sizes: Sequence[int] = (8, 12, 16, 24, 32, 64),
    matrices: Sequence[int] = (16, 960),
    dtype=torch.float32,
    device: str = "cuda",
    reps: int = 20,
) -> List[Dict[str, float]]:
    """``[{"n", "matrices", "small_ms", "bmm_ms"}]``: ms per exponential (forward + backward) by product form."""
    if not torch.cuda.is_available():
        raise RuntimeError("the crossover is a GPU's: no CUDA device")
    rows = []
    for batch in matrices:
        for n in sizes:
            rows.append(
                {
                    "n": n,
                    "matrices": batch,
                    "small_ms": _time(n, batch, "small", dtype, device, reps),
                    "bmm_ms": _time(n, batch, "bmm", dtype, device, reps),
                }
            )
    return rows


def recommend(rows: List[Dict[str, float]], matrices: int) -> int:
    """The largest measured size up to which the fused form is faster at every measured size (0: never)."""
    best = 0
    for row in sorted((r for r in rows if r["matrices"] == matrices), key=lambda r: r["n"]):
        if row["small_ms"] >= row["bmm_ms"]:
            break
        best = int(row["n"])
    return best


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dtype", default="float32", choices=("float32", "float64"))
    parser.add_argument("--matrices", type=int, nargs="+", default=[16, 960])
    parser.add_argument("--sizes", type=int, nargs="+", default=[8, 12, 16, 24, 32, 64])
    parser.add_argument("--reps", type=int, default=20)
    args = parser.parse_args(argv)
    rows = measure(args.sizes, args.matrices, getattr(torch, args.dtype), reps=args.reps)
    current = settings.snapshot()["small_matmul_max"]
    print(f"{torch.cuda.get_device_name()}, torch {torch.__version__}, {args.dtype}: "
          "ms per exponential (forward + backward)")
    for batch in args.matrices:
        print(f"  {batch} matrices per call")
        for r in (r for r in rows if r["matrices"] == batch):
            faster = "fused" if r["small_ms"] < r["bmm_ms"] else "bmm"
            print(f"    N={r['n']:4d}: fused {r['small_ms']:8.3f}  bmm {r['bmm_ms']:8.3f}  ({faster})")
        print(f"  -> small_matmul_max = {recommend(rows, batch)} for {batch} matrices per call "
              f"(now {current['value']}, from {current['source']})")


if __name__ == "__main__":
    main()
