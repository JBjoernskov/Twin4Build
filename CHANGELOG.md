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
