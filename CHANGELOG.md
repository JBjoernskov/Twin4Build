# Changelog

## 2.0.0

API-quality major release. Preferred forms are documented below; new soft-compat aliases warn and will be **removed in 2.1.0**.

### Removed (previously deprecated)

- camelCase kwargs already routed through deprecation helpers (`startTime` / `endTime` / `stepSize`, `useSpreadsheet` / `useDatabase` / `usedf` / `usedict`, …)
- `Optimizer` option `fast`; select composed execution with
  `Simulator(model, execution_mode="composed")`
- Estimator options `fast` and `fast_validate`; select composed execution on
  `Simulator`
- Estimator `lambda_schedule`; use the unified `schedule` argument
- `plot_component`, the old `Entry(attribute=...)` form, and tuple plot entries;
  use `plot(..., entries=[Entry(data=..., label=...)])`
- `CascadePIDControllerSystem` alias
- Estimator legacy `parameters` dict (`private` / `shared`)
- `set_parameters_from_array` → use `set_parameters`
- `windows-curses` dependency (curses LOGGER TUI removed)
- unknown public keyword arguments are rejected instead of being silently
  ignored; removed camelCase time/source keywords now raise `TypeError`

### Deprecated (removed in 2.1)

- `*TorchSystem` / `fmuSystem` public names → prefer `*System` / `FmuSystem` / `SmoothOnOffControllerSystem`
- Schedule/controller camelCase kwargs → snake_case (`weekday_ruleset`, `is_reverse`, `date_column`, …)
- `use_spreadsheet` / `use_database` / `use_dict` → prefer `source=`
- `Model.load(semantic_model_filename=...)` auto-translate → `SemanticModel` + `Translator.translate`; prefer `Model.load(filename=...)`
- `Model.load(simulation_model_filename=...)` → `Model.load(filename=...)`
- `Model.set_initial_values(dict_)` → parallel `values`, `components`, and
  `output_names` lists
- `Translator.translate(systems_=...)` → `systems=`
- Public `verbose=` kwargs → configure `LOGGER.verbose` / `LOGGER.logfile`
- `get_component_by_class(dict_, ...)` → `get_components_by_class(Cls)`
- `twin4build.utils.print_progress` → `twin4build.utils.logger`

### Changed

- `Simulator(compile_step=...)`: the functional transform-mode step can be
  compiled with `torch.compile` (Inductor) before it is captured or run
  eagerly.  `"auto"` (default) enables it on CUDA when the torch build has
  Triton (Linux wheels; Windows wheels have none and keep the eager step),
  `True` requires it, `False` disables it.  Inside `torch.compile` the
  state-space matrix exponential uses pointwise products (`_expm_ss_fused`)
  so Inductor fuses them; eager keeps the cuBLAS form.  On the one-zone
  shooting benchmark the captured graph replays in 0.35 s instead of 1.41 s,
  its driver-side executable shrinks from 1.9 GiB to 0.36 GiB, and values
  and gradients match eager to 1e-15 (issue #134); the first call pays a
  one-time compile of about 45 s.  `System.state_size()` is cached after the
  first call (Dynamo cannot trace the `vars()` walk it used every step).
  The batched single-shooting bundles (multi-start SQP, its line search)
  roll the batch out with a compiled `vmap` of the step
  (`FunctionalModel.compiled_batched_step`, `Simulator.rollout_functional_batched`)
  because `vmap` applied from eager code to a compiled function is not
  supported; functorch transforms over a compiled step fall back to the
  eager step automatically.

- `_expm_ss` (fixed-schedule scaling-and-squaring matrix exponential used by
  every state-space component in transform mode) evaluates the Taylor part in
  Horner form and each squaring as one fused `baddbmm`, cutting it from ~100
  to ~30 kernels per call; same schedule, same accuracy (2e-12 relative
  against `torch.matrix_exp`), same derivatives.  Forward kernels per
  rollout step on the one-zone benchmark drop from 819 to 707 (issue #134).

- Cycle removal breaks ties deterministically (by component id) when several
  edges break the same number of cycles and carry the same priority.  The
  winner used to follow the cycle enumeration over a set-based graph, i.e.
  Python's hash seed, so two processes could cut different edges of the same
  algebraic loop and place the one-step Gauss-Seidel lag on different signals
  -- different discrete-time models, with trajectories differing in the third
  significant digit for the canonical one-zone benchmark.

- Collocation estimator: the box on the boundary states is now each
  dimension's warm-start range widened by `boundary_state_margin` (default
  6 std), instead of a fixed +/-6 std box.  The fixed box clipped the warm
  start's initial transient, so IPOPT started infeasible and could not
  represent the true trajectory; on the canonical 1-zone benchmark
  collocation converged to a spurious optimum at 8.8x the shooting objective,
  and now matches shooting from both cold and warm starts.
- Collocation estimator: a "rollout" (feasible) start applies IPOPT
  warm-start defaults (`mu_init=1e-6`, `warm_start_init_point`, tiny bound
  push); any `ipopt.*` key passed by the caller still wins.
- Single-shooting value/gradient bundle uses plain reverse-mode autograd on
  CUDA instead of functorch's `grad_and_value` transform.  The functorch
  version recorded a CUDA graph that was valid for exactly one launch on
  torch 2.11+cu128 (A100): every multi-zone shooting benchmark case died
  with an illegal memory access at the first replay.  Plain autograd over
  the same rollout replays correctly and captures in about half the time;
  batches keep a `vmap` forward rollout (capture-safe on the same A100) so
  eight starts replay in about the time of one.
- `CudaGraphCallable` records the capture phase it is in (`last_phase`) and
  attaches it as a note to any exception raised during capture; with
  `T4B_CUDA_GRAPH_SYNC_PHASES=1` the device is synchronized after every phase
  so an asynchronous CUDA fault is attributed to the phase that launched it.
- Benchmark harness: the peak-memory sampler thread no longer calls the CUDA
  runtime (allocator bookkeeping only; device-wide memory is read on the main
  thread before/after each timed region), failed cases keep the child's
  traceback, and the collocation preflight uses a measured VRAM model
  (~1.16 GiB per zone on an A100) against the actual card instead of a
  sparse-storage count.  Adds `benchmarks/colab_debug_multizone.ipynb`, a
  focused in-process reproducer for the multi-zone CUDA fault seen on A100.
- IPOPT early stopping no longer counts infeasible iterates as stagnation
  (the rule is exposed as `twin4build.solvers.ipopt.early_stopping_step`);
  cold starts were being cut mid-descent at their first feasible incumbent.

- Preferred system names without `Torch` suffix; `OnOffControllerTorchSystem` → `SmoothOnOffControllerSystem`
- Module files renamed accordingly (`*_torch_system.py` → `*_system.py`; smooth on-off → `smooth_on_off_controller_system.py`)
- `Model.load(..., enable_fusion=True)` first-class
- Simulation policy has independent dimensions: `execution_mode` is `object`
  or `functional`, while `execution_backend` is `eager` (default) or
  `cuda_graph`. CUDA Graph is a backend for functional execution, not a mode.
- Component batching is an independent model-layout transformation through
  `Model.batch_components()`, with source mappings exposed by
  `get_batched_component_info()` and `get_batch_id_for_component()`.
- Functional workflows expose `build_functional_model`,
  `record_exogenous_inputs`, and `rollout_functional`.
- Estimator and optimizer Hessian selection uses
  `hessian="exact"|"gauss_newton"|"limited_memory"`; the default is `exact`.
- `Optimizer.optimize` returns `OptimizationResult` (SciPy fields preserved)
- Shared method parsing via `twin4build.utils.method_spec`
- LOGGER: dual sinks — ANSI+indent on stdout, plain (no ANSI) logfile always (`progress.log` by default)
- Top-level `tb.types` / `Vector` / `Scalar` / `Parameter` / `State`
- Package version `2.0.0`
- Canonical simulation, estimation, optimization, and scaling benchmarks now
  use the complete translated `full_workflow` topology; batching parity covers
  batched full-workflow controls, nested parameters, and data-source settings.
- The constrained optimization matrix compares AD SLSQP and trust-constr.
  Pareto scaling uses one shared broadcast valve schedule and reports exact
  decision/constraint dimensions plus convergence and comfort violations.

### Migration notes

1. Replace `BuildingSpaceTorchSystem` with `BuildingSpaceSystem` (etc.).
2. Use `weekday_ruleset=` / `is_reverse=` / `date_column=` instead of camelCase.
3. Prefer `source="spreadsheet"|"dict"|"database"|"df"`.
4. Translate explicitly: `model = Translator().translate(SemanticModel(...)); model.load()`.
5. Configure logging with `LOGGER.verbose` / `LOGGER.logfile`, not `verbose=` kwargs.
6. Historical execution labels `object_graph`, `composed`, and `cuda_graph`
   (as a mode), compound labels such as `compiled_composed`, and
   `build_compiled_model` migrate to the independent model-layout,
   `execution_mode`, and `execution_backend` APIs above. Historical
   `fast`, `public_composed`, and `exact_hessian`/`capture_hessian` options
   likewise migrate to functional execution and `hessian=...`.
7. Migrate before **2.1.0**, when the soft-compat aliases above are deleted.

### Release metadata

- `pyproject.toml` is the package-version source of truth and declares
  `2.0.0`.
- Release tags use the matching `vMAJOR.MINOR.PATCH` form. Pushing such a tag
  triggers the trusted-publishing PyPI workflow for the configured maintainer;
  branch pushes do not publish or auto-increment versions.
