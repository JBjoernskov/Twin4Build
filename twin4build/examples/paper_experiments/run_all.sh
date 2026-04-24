#!/usr/bin/env bash
# Reproduce every paper result (CPU only) in a single pass.
#
# Intended to be invoked from the repo root so that relative imports
# (twin4build.examples.paper_experiments.*) resolve against the installed
# package.  Adjust PYTHON to point at the right interpreter if you are not
# using a venv that is on PATH.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../" && pwd)"
PYTHON="${PYTHON:-python}"

cd "${REPO_ROOT}"

#echo "=== paper_experiments :: environment snapshot ==="
#"${PYTHON}" -m twin4build.examples.paper_experiments.hardware_info

echo "=== E1 :: N_ROOMS sweep (CPU single/multi-thread) ==="
"${PYTHON}" -m twin4build.examples.paper_experiments.experiments.e1_n_rooms_sweep

#echo "=== E2 :: heterogeneity sweep ==="
#"${PYTHON}" -m twin4build.examples.paper_experiments.experiments.e2_heterogeneity

#echo "=== E3 :: horizon x step-size grid ==="
#"${PYTHON}" -m twin4build.examples.paper_experiments.experiments.e3_horizon_step

#echo "=== E4 :: estimator-example accuracy + compression audit ==="
#"${PYTHON}" -m twin4build.examples.paper_experiments.experiments.e4_realistic_case

#echo "=== E5 :: torch.profiler breakdown ==="
#"${PYTHON}" -m twin4build.examples.paper_experiments.experiments.e5_profiler

#echo "=== E6 :: ablations ==="
#"${PYTHON}" -m twin4build.examples.paper_experiments.experiments.e6_ablations

#echo "=== plots F2..F7 ==="
"${PYTHON}" -m twin4build.examples.paper_experiments.plots.f2_speedup_vs_n
#"${PYTHON}" -m twin4build.examples.paper_experiments.plots.f3_heterogeneity
#"${PYTHON}" -m twin4build.examples.paper_experiments.plots.f4_horizon_heatmap
#"${PYTHON}" -m twin4build.examples.paper_experiments.plots.f5_accuracy
#"${PYTHON}" -m twin4build.examples.paper_experiments.plots.f6_profiler
#"${PYTHON}" -m twin4build.examples.paper_experiments.plots.f7_compression

echo "=== F1 :: system-diagram visualisation (manual follow-up may be needed) ==="
"${PYTHON}" -m twin4build.examples.paper_experiments.plots.f1_visualize

echo "=== done.  See results/ and figures/ ==="
