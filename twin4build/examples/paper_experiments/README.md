# Paper experiments: batched component execution in twin4build

Reproducible, CPU-only experiment suite backing a BS/IBPSA-style conference
paper on the `Model.build_compiled_model` batching compiler in twin4build.
Everything is driven from CSVs under `results/`; figures are written to
`figures/`.  Both folders are gitignored -- the scripts are what gets
committed, not the outputs.

## Layout

```
paper_experiments/
├── config.py                # All knobs: MAX_N_ROOMS, sweep lists, devices, ...
├── common.py                # Model builder, timing, metrics, CSV I/O, env capture
├── experiments/
│   ├── e1_n_rooms_sweep.py  # N_ROOMS sweep (CPU single + multi-thread)
│   ├── e2_heterogeneity.py  # k-archetype sweep at N_ROOMS=128
│   ├── e3_horizon_step.py   # horizon x step_size grid at N_ROOMS=64
│   ├── e4_realistic_case.py # estimator_example full-port accuracy audit
│   ├── e5_profiler.py       # torch.profiler breakdown at N_ROOMS=256
│   └── e6_ablations.py      # NO_BATCH_CLASSES / aligned-gather / dtype
├── plots/
│   ├── f1_visualize.py      # System-diagram visualization hook (manual)
│   ├── f2_speedup_vs_n.py   # log-log wallclock + speedup curves
│   ├── f3_heterogeneity.py  # speedup vs k bar chart
│   ├── f4_horizon_heatmap.py# speedup heatmap over (horizon, step)
│   ├── f5_accuracy.py       # residual timeseries + per-port error CDF
│   ├── f6_profiler.py       # CPU self-time + dispatch-count stacked bars
│   └── f7_compression.py    # per-execution-group compression ratio
├── hardware_info.py         # Writes env.json (git SHA, torch, CPU, RAM, ...)
├── run_all.sh               # End-to-end reproducer (experiments + plots)
├── requirements.txt         # Pinned versions for the CPU pass
└── README.md
```

## Running

Use the same Python environment that has `twin4build` installed (the
scripts assume `import twin4build as tb` works).

Single experiment:

```bash
python -m twin4build.examples.paper_experiments.experiments.e1_n_rooms_sweep
python -m twin4build.examples.paper_experiments.plots.f2_speedup_vs_n
```

End-to-end (writes env.json first so results are traceable):

```bash
./twin4build/examples/paper_experiments/run_all.sh
```

## Scaling the study

All scale / scope knobs live in `config.py`.  The most important ones:

- `MAX_N_ROOMS` -- hard cap on every sweep that scales with room count.
  Raise to go bigger.  `N_ROOMS_SWEEP` auto-truncates against this cap,
  so one edit widens the study everywhere.
- `DEVICES` -- a dict mapping logical platform keys to torch thread
  counts.  A commented `"gpu"` entry documents how to enable the future
  CUDA pass; the experiment scripts are already device-key-driven and
  use `common.resolve_device` as the single chokepoint.
- `HORIZON_GRID_DAYS`, `STEP_GRID_S` -- E3 grid.  Kept laptop-safe; add
  longer horizons here when you have more compute.
- `K_ARCHETYPES_SWEEP`, `HETEROGENEITY_NELEMENTS_POOL` -- E2 knobs.
- `N_E5`, `N_E6` -- problem sizes for E5 profiler and E6 ablations.

## Claims -> artefacts map

| Claim | Experiment | Figure |
|---|---|---|
| C1 Correctness            | E4 (port_errors), E6-A1, E6-A3 | F5         |
| C2 Scalability            | E1, E2, E3                     | F2, F3, F4 |
| C3 Mechanism              | E5, E6-A2                      | F6         |
| C4 Realistic topology     | E4                             | F7, F5     |

## What is NOT in this pass (future work)

- GPU timings, CPU-vs-GPU crossover, kernel-launch counts -- scripts are
  device-key-driven; uncomment `"gpu"` in `DEVICES` and extend
  `common.resolve_device` to enable.
- Horizons beyond 14 days (E3) -- the 365-day cell is skipped to keep
  the full sweep laptop-safe.  `HORIZON_GRID_DAYS` accepts longer entries.
- `torch.compile` / JIT ablation -- not covered; could be added as an
  `A4` in `e6_ablations.py`.
