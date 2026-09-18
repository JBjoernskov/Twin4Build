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

### Added

- Make-up air on the zones: `BuildingSpaceMassSystem` takes `makeUpAirCO2`
  and `BuildingSpaceThermalSystem` `makeUpAirTemperature` for the air that
  replaces an exhaust surplus (default: outdoor), and `exchangeCO2Gain`
  (fusable, one slot per connected opening) for interzonal exchange, so a
  zone's CO2 balance can be closed against a neighbouring space instead of
  outdoor air.
- `discrete_statespace_system.effective_matrices(A, B, E, F, u)`: the
  bilinear matrices evaluated at the current input, factored out of
  `_discretize_onestep` so a caller that needs only part of the
  discretization can form the same matrices instead of re-deriving them.
- `BuildingSpaceMassSystem.mass_matrices`: the CO2 mass balance's
  `(A, B, C, D, E, F)` as a module-level pure function of `(V, G_occ, m_inf,
  n_c)`, with `N_STATES` / `N_INPUTS` / `OCCUPANCY_SLOT` naming the matrix
  contract.  `_build_matrices` is now a thin wrapper around it.

- Plug-in solvers (`twin4build.solvers.registry`): an object with a
  `method` tuple and `solve(problem, options)` can be registered
  (`register_solver`) and used by name, or passed as `method=` to
  `Estimator.estimate` directly.  The estimator hands it an
  `EstimationProblem` (normalized start and bounds, the composed objective
  with its batched bundles, device, dtype, transcription) and records its
  SciPy-like result exactly as for the built-in backends; a registered
  method shadows a built-in one of the same name.  `Estimator.estimation_problem()`
  exposes the prepared problem; `solve_batched_multistart(...,
  chunk_solver=)` lets a plug-in step reuse the multistart chunking and
  result assembly.  `Optimizer.pareto_front` accepts a registered Pareto
  route (`register_pareto_route`: `anchors` + `sweep`) under its own method
  tuple in place of the host-solver anchors and epsilon sweep.

- Post-fit identifiability report. `Estimator.estimate` now ends with a local
  identifiability analysis of the residual Jacobian at the optimum
  (`twin4build/estimator/_identifiability.py`): parameters no residual reacts
  to, flat directions (singular vectors of the unit-column Jacobian with a
  relative singular value below 1e-3, listed as the parameter combination
  that is the only thing the data determine), pairs whose Gauss-Newton
  correlation exceeds 0.95 (trade-offs), parameters whose standard error
  exceeds their whole admissible range, and parameters sitting on a bound.
  Findings are logged as warnings and attached to the result as
  `result["identifiability"]`.  `identifiability="auto"` (default) runs it
  whenever the residual Jacobian is cheap (functional single-shooting
  objective, or object-mode AD with at most 20 parameters); `True` forces
  it, `False` skips it.  A dead or flat parameter is left where the solver
  happened to stop, so its value carries no information -- the report says
  which ones.
- `AirHandlingUnitSystem(exhaust_follows_supply=True, exhaustFlowRatio=r)`:
  every branch's exhaust flow is `r` times its supply flow and the exhaust
  damper model is bypassed; `exhaustFlowRatio` is estimable (0.3-1.5) and the
  exhaust damper parameters leave theta.  `OccupancySystem` takes the same
  two arguments so its CO2 inversion uses the same flows (share the ratio
  with the AHU's in the estimator).  With one exhaust meter per AHU the
  per-branch exhaust dampers are not identifiable; the ratio is.
- `AirHandlingUnitSystem` output `preheatSupplyAirTemperature`: the
  heat-recovery outlet before the coil.  Patterns for the AHU-side measured
  points that read these outputs are left to the user's pattern set (#200).
- Translator `Node(cls=..., exclude=...)`: classes an instance must not be
  (subclasses included) to bind to a pattern node, so a pattern on a base
  class can step aside for a more specific pattern on a subclass.  Used by
  the AHU supply-air-temperature patterns, which now exclude the
  `Preheat_` subclass.
- Signature patterns are user-defined and passed explicitly (#200):
  `Translator.translate(semantic_model, patterns=[...])`, each pattern bound
  to the `System` class it models (`SignaturePattern(id, system=cls)` or
  `sp.bind(cls)`).  `twin4build.examples.patterns` is the public *example* set
  (`default_patterns()` and one helper per class); it is not a standard, a
  deployment composes it with its own patterns.  `System.sp` and
  `System.add_signature_pattern` are gone, and no system module registers
  patterns at import time.  `translate(patterns=None)` warns and uses the
  example set for one minor version; `systems=` is now an allow-list.
- `[gpu]` extra and a documented CUDA torch install. `pip install twin4build`
  still follows PyPI's default `torch` wheel (CPU-only on Windows).
  `pip install twin4build[gpu] --extra-index-url https://download.pytorch.org/whl/cu128`
  pulls a CUDA 12.8+ build and Triton on Linux. Compiled / CUDA-graph paths
  need Linux or WSL (Triton has no Windows wheels). `model.to("cuda")`
  raises with that install line when the process has no CUDA (issue #167).

### Changed

- The block trust-region step (`("custom", "batched-tr", "ad")`, on `dev`
  since September) and the batched trust-region Pareto route left the
  library; a solver of that shape now plugs in through
  `twin4build.solvers.registry`.  The benchmark matrices drop their rows.

- Room air models balance their ventilation flows. `BuildingSpaceThermalSystem`
  and `BuildingSpaceMassSystem` used to charge the supply at supply state and
  the exhaust at room state independently, so a mismatch between the two
  measured flows left a fictitious `(m_sup - m_exh) * cp * T_i` storage term
  (an exhaust meter reading 20% low heated the room out of nothing).  The room
  air mass is constant, so every stream entering is balanced by air leaving at
  room state: the supply term is `m_sup * cp * (T_sup - T_i)`, and the exhaust
  enters only as the outdoor **make-up flow** `max(m_exh - m_sup, 0)` drawn
  through the envelope, `m_mu * cp * (T_out - T_i)` (same for CO2).  Supply in
  excess of the exhaust leaves through the envelope and the exhaust flow drops
  out; the CO2 balance can no longer be driven below outdoor by ventilation.
  Ports are unchanged; the transform is applied to the `exhaustAirFlowRate`
  slot at input assembly on the object, functional and fused paths
  (`twin4build/systems/building_space/air_balance.py`; state-space units may
  declare `_ss_transform_inputs` / `SS_TRANSFORM_PORTS`, which
  `FusedStateSpaceSystem` applies before stacking the joint input).  The
  constant infiltration parameter `m_inf` stays additive (EnergyPlus
  convention).
  `OccupancySystem`'s CO2 inversion uses the same balanced equation, so
  the people it books reproduce the measured CO2 through the forward model
  for any supply/exhaust pair.
- `System.get_estimable_parameters` skips parameters the owner reports as
  inactive (`_inactive_parameters()`); `BuildingSpaceThermalSystem` reports
  `C_boundary` / `R_boundary` unless a `boundaryTemperature` is connected,
  so rooms without the deprecated in-zone boundary wall no longer put two
  dead entries per room into theta.
- `BuildingSpaceThermalSystem` `C_air` upper bound 1e6 -> 3e6 J/K: the air
  node stands for air plus furniture, and 1e6 was binding on classrooms.
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
