"""Global configuration for the paper experiment suite.

All knobs that control experiment scope live here so the rest of the code
stays platform- and scale-agnostic.  Change these values and re-run to
expand the study (for example, raise ``MAX_N_ROOMS`` when moving from a
laptop to a workstation, or uncomment ``"gpu"`` in ``DEVICES`` once the
CUDA pass is ready).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

import torch


# ---------------------------------------------------------------------------
# Scale caps
# ---------------------------------------------------------------------------
# Hard cap on the largest multi-room configuration any experiment will build.
# Every sweep that scales with room count clamps itself against this value,
# so a single edit here widens the study without per-script changes.
MAX_N_ROOMS: int = 256


# ---------------------------------------------------------------------------
# Timing repetitions
# ---------------------------------------------------------------------------
# Discarded runs used to warm up caches / torch dispatcher before measurement.
N_WARMUP: int = 1

# Default measured repeats; experiments may override with ``repeats_for(n)``.
N_REPEATS_DEFAULT: int = 3


def repeats_for(n_rooms: int) -> int:
    """Return a sensible repeat count given the problem size.

    Small models converge on median wallclock fast; big ones are expensive,
    so we spend fewer samples on them.
    """
    if n_rooms <= 64:
        return 5
    return N_REPEATS_DEFAULT


# ---------------------------------------------------------------------------
# Platforms
# ---------------------------------------------------------------------------
# Each entry maps a logical platform key to a thread count.  A value of 0
# means "do not pin torch threads" (useful for a future GPU row).  Add or
# remove entries to grow / shrink the per-experiment platform matrix.
DEVICES: Dict[str, int] = {
    "cpu-st": 1,  # single-thread baseline -- isolates per-step overhead
    "cpu-mt": max(1, (os.cpu_count() or 1)),  # all cores -- realistic CPU
    # "gpu": 0,   # uncomment when the CUDA pass lands; see common.set_device
}


# ---------------------------------------------------------------------------
# Sweeps
# ---------------------------------------------------------------------------
# E1 N_ROOMS sweep (auto-capped against MAX_N_ROOMS).
_N_ROOMS_SWEEP_FULL: List[int] = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
N_ROOMS_SWEEP: List[int] = [n for n in _N_ROOMS_SWEEP_FULL if n <= MAX_N_ROOMS]

# E2 heterogeneity sweep: number of archetypes, each replicated N_E2//k times.
# Archetypes are produced by varying SpaceHeaterTorchSystem.nelements, which
# changes the component's n_states and therefore its signature (see
# twin4build.model.model._component_signature).
N_E2: int = min(128, MAX_N_ROOMS)
K_ARCHETYPES_SWEEP: List[int] = [1, 2, 4, 8]
# Pool of structural variants; first k entries are used for k archetypes.
HETEROGENEITY_NELEMENTS_POOL: List[int] = [3, 4, 5, 6, 7, 8, 9, 10]

# E3 horizon x step-size grid, kept laptop-safe.
N_E3: int = min(64, MAX_N_ROOMS)
HORIZON_GRID_DAYS: List[int] = [1, 3, 7, 14]
STEP_GRID_S: List[int] = [60, 300, 600, 1800]

# Default single-value horizon/step used by E1, E2, E5.
HORIZON_DAYS_DEFAULT: int = 7
STEP_SIZE_DEFAULT: int = 600

# E5 profiler case size.
N_E5: int = min(32, MAX_N_ROOMS)

# E6 ablation base case size (small, to keep repeated runs cheap).
N_E6: int = min(64, MAX_N_ROOMS)


# ---------------------------------------------------------------------------
# Numerics
# ---------------------------------------------------------------------------
# Library default is float64.  The E6 dtype ablation temporarily flips this.
DTYPE_DEFAULT: torch.dtype = torch.float64


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PACKAGE_DIR: Path = Path(__file__).resolve().parent
RESULTS_DIR: Path = PACKAGE_DIR / "results"
FIGURES_DIR: Path = PACKAGE_DIR / "figures"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
