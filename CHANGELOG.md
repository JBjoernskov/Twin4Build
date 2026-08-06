# Changelog

## 2.0.0

API-quality major release. Preferred forms are documented below; new soft-compat aliases warn and will be **removed in 2.1.0**.

### Removed (previously deprecated)

- camelCase kwargs already routed through deprecation helpers (`startTime` / `endTime` / `stepSize`, `useSpreadsheet` / `useDatabase` / `usedf` / `usedict`, …)
- `plot_component` and legacy `Entry` / tuple plot formats
- `CascadePIDControllerSystem` alias
- Estimator legacy `parameters` dict (`private` / `shared`)
- `windows-curses` dependency (curses LOGGER TUI removed)

### Deprecated (removed in 2.1)

- `*TorchSystem` / `fmuSystem` public names → prefer `*System` / `FmuSystem` / `SmoothOnOffControllerSystem`
- Schedule/controller camelCase kwargs → snake_case (`weekday_ruleset`, `is_reverse`, `date_column`, …)
- `use_spreadsheet` / `use_database` / `use_dict` → prefer `source=`
- `Model.load(semantic_model_filename=...)` auto-translate → `SemanticModel` + `Translator.translate`; prefer `Model.load(filename=...)`
- `Translator.translate(systems_=...)` → `systems=`
- Public `verbose=` kwargs → configure `LOGGER.verbose` / `LOGGER.logfile`
- `get_component_by_class(dict_, ...)` → `get_components_by_class(Cls)`
- `twin4build.utils.print_progress` → `twin4build.utils.logger`

### Changed

- Preferred system names without `Torch` suffix; `OnOffControllerTorchSystem` → `SmoothOnOffControllerSystem`
- Module files renamed accordingly (`*_torch_system.py` → `*_system.py`; smooth on-off → `smooth_on_off_controller_system.py`)
- `Model.load(..., enable_fusion=True)` first-class
- `Optimizer.optimize` returns `OptimizationResult` (SciPy fields preserved)
- Shared method parsing via `twin4build.utils.method_spec`
- LOGGER: dual sinks — ANSI+indent on stdout, plain (no ANSI) logfile always (`progress.log` by default)
- Top-level `tb.types` / `Vector` / `Scalar` / `Parameter` / `State`
- Package version `2.0.0`

### Migration notes

1. Replace `BuildingSpaceTorchSystem` with `BuildingSpaceSystem` (etc.).
2. Use `weekday_ruleset=` / `is_reverse=` / `date_column=` instead of camelCase.
3. Prefer `source="spreadsheet"|"dict"|"database"|"df"`.
4. Translate explicitly: `model = Translator().translate(SemanticModel(...)); model.load()`.
5. Configure logging with `LOGGER.verbose` / `LOGGER.logfile`, not `verbose=` kwargs.
6. Migrate before **2.1.0**, when the soft-compat aliases above are deleted.
