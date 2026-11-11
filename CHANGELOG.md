# Changelog

## 2.0.0

API-quality major release. Preferred forms are documented below; new soft-compat aliases warn and will be **removed in 2.1.0**.

### A System is its declarations and one forward step (ADR 0001)

`docs/adr/0001-system-stepping-and-declarations.md`, implemented by #282 to
#299 and the clean-up after them.  A System is written as class-level
declarations plus `forward(x, inputs, p, step_size)`; the base builds,
sizes, steps and checks it.  There is no deprecation period for the System
interface.  Where an old keyword or attribute could be translated, it is,
with a `DeprecationWarning` until 2.1; where it cannot, reading, setting or
passing it raises and names its replacement.

#### Added

- Declarations (`twin4build.systems.saref4syst.declarations`, exported from
  `twin4build.core`): `Input`, `Output`, `State`, `Parameter`, `Setting`,
  `Child` and `REQUIRED`, with `size=` (an int, `"wired"`, another port or a
  setting), `default=` (a value or another input), `alias=` (a renamed
  port's old name), `init=` (a state's or an output's start: a number, a
  setting or a state) and `per_slot=` (a child or a value per slot of an
  input).  One name may serve one input and one output; any other repeated
  name raises `TypeError` at class definition (#285, #292 to #295,
  #298).
- `derive(p, step_size)`: what depends only on the parameters and the step
  size, evaluated once per run (object mode) or per rollout (functional
  engine) and handed to `forward` in `p` (#286).
- Data sources: a System that defines `series(timestamps, step_size)` is
  filled once at the run start and never stepped.  Readers `CsvSource`,
  `DataFrameSource` and `DatabaseSource`
  (`twin4build.utils.data_loaders.sources`) carry the reading options
  (resampling, gap limit, fill, cache) (#287).
- Optional parameters, `Parameter(None)`: absent until given, then not in
  `p`, not estimated and not saved (#291).  Parameters of several values
  per copy, `Parameter(..., size=...)` (#298).
- Value settings batch per copy and reach `forward` in `p`;
  `Setting(..., structural=True)` splits batches instead (#296).
- `twin4build.testing.check_system(system, inputs)`, the conformance check
  every System passes: save and reload, a deepcopy, object mode against the
  functional engine, two batched copies against the copies stepped
  separately, finite gradients, and CUDA when a GPU is present.
  `twin4build/tests/test_check_system.py` holds an example per core System.

#### Removed and renamed, with the migration

- `forward(..., sample_time)` → `forward(x, inputs, p, step_size)` (#284);
  `forward`'s `transform_mode`, `SUPPORTS_TRANSFORM_MODE`,
  `step_constants` and `StepParams` → `derive(p, step_size)`; the PID's
  per-simulation step-size tensor → one step size per run (#286).
- `FunctionSystem` → a small System of its own (declared inputs, an output
  and a `forward`), which also saves and loads (#290).
- `PiecewiseLinearScheduleSystem` → `PiecewiseLinearSystem(X=..., Y=...)`,
  output `y` (#288).  `PiecewiseLinearSystem(Y_bounds=...)` →
  `tb.Variable(curve.Y, lb=..., ub=...)` (#298).
- `ScheduleSystem`'s file and database keywords (`filename`, `date_column`,
  `value_column`, `uuid`, `name`, `dbconfig`, `fill`, the `use_*` flags,
  `source="dict"|...`) → `source=CsvSource(...)` / `DatabaseSource(...)`
  (#288).
- `OutdoorEnvironmentSystem`'s `df`, `filename_*`, `datecolumn_*`,
  `valuecolumn_*`, `uuid_*`, `dbconfig_*` and `use_*` → one reader per output
  (`outdoorTemperature_source=`, `globalIrradiation_source=`,
  `outdoorCo2Concentration_source=`) or
  `OutdoorEnvironmentSystem.from_dataframe(df)`; `transformation_ref_*` →
  `outdoorTemperature_transformation` / `globalIrradiation_transformation`
  (old saved names load, with a warning) (#289).  `TimeSeriesInputSystem`
  takes one `source=` reader (#287).
- `DamperSystem`'s `c_tied` and `set_c` → `damper.c = value` or
  `DamperSystem(c=...)`; an absent `c` is tied to `-a` (#291).
- The branch -> room maps: `AirHandlingUnitSystem._exhaust_branch_map` /
  `_branch_room_index`, `ReturnFlowJunctionSystem.branch_temperature_slots`
  and `partition._repair_return_junctions` → wire each room's temperature
  into the slot of each branch serving it, `input_port_index=[...]` (#291).
  `AirHandlingUnitSystem(n_branches=...)` → the wired supply damper slots;
  the `*_kwargs` → the children's names (`supply_fan=dict(...)`, with a
  warning) (#292).
- `SpaceHeaterSystem` / `FanCoilUnitSystem` `initialize_UA` →
  `unit.UA = unit.solve_UA()` (UA holds the nominal solve from
  construction).  The fan-coil's `_valve` → `valve`; its flat
  `waterFlowRateMax=` / `valveAuthority=` move into the child, with a
  warning (#294).
- The room's in-zone boundary wall (`C_boundary`, `R_boundary`,
  `boundaryTemperature`, `n_boundary_temperature`) → a `WallSystem` on
  `wallHeatGain`; `thermal_kwargs` / `mass_kwargs` → `thermal=dict(...)`,
  `mass=dict(...)`, with a warning (#295).
- `CascadeControllerSystem(controller_a=..., controller_b=...)` (classes) →
  a subclass declaring other children; the flat `kp_a` ... `isReverse_b`
  and `controller_a_kwargs` / `controller_b_kwargs` → `ctrl_a=dict(...)`,
  `ctrl_b=dict(...)`, with a warning.  `BandGate(threshold_high=...)` →
  `band = threshold_high - threshold` (#297).
- `CoilSystem.specificHeatCapacityAir` (unused; the coil reads
  `constants.CP_AIR`) (#290).
- `SensorSystem`'s `filename`, `datecolumn`, `valuecolumn`, `df`,
  `dbconfig`, `use_spreadsheet`, `use_database`, `use_df`, `resample_method`,
  `dt_limit` and `fill` → one reader, `source=CsvSource(path,
  date_column=..., value_column=...)`, `DataFrameSource(df)` or
  `DatabaseSource(uuid, dbconfig)`, which carries the reading options;
  `set_series(series)` and `set_dbconfig(dbconfig)` stay.  `uuid` names the
  point (and is the database point read when there is no `source`); setting
  it no longer switches a sensor with another reader to the database.
  `transformation_ref` → `transformation` (old saved names load, with a
  warning).  `del sensor.scoring_mask` → `sensor.scoring_mask = None`.
- `ScheduleSwitchControllerSystem` and its
  `ScheduleSwitchControllerTorchSystem` alias: no users.
- `SpaceHeaterSystem`'s output `Power` → `toRoomPower`, and the room's input
  `outdoorCO2` → `outdoorCo2Concentration` (on `BuildingSpaceSystem` and
  `BuildingSpaceMassSystem`): the aliases deprecated earlier in 2.0 are
  gone, so the old names no longer connect, read or load.  A saved
  `instance_graph.ttl` that names them: replace the port names in the file.
  `Output(alias=...)` / `Input(alias=...)` remain for later renames.
- `SimulationModel.validate`, `validate_components`, `validate_connections`,
  `_validate_ids`, the `_validated_for_*` flags, `is_validated`,
  `Model.validate`, `Model.is_validated`, `load(validate_model=...)` and the
  sensor's `validate` / `validate_connections` hooks → nothing to call: a
  load checks the model and raises (below).  `Model.load(validate_model=...)`
  raises a `TypeError` saying so.

#### Changed

- A load raises one `ValueError` that names every problem: an invalid
  component id, an input without a default that nothing is wired to (a
  `size="wired"` input is as wide as what is wired, none included), a
  declared System wired beyond its declared sizes, a declared parameter
  without a value.  These were warnings, and a model with a missing input
  loaded and ran.  Wire the input, give it a default (`Input(default=...)`)
  or give the parameter a value.
- `SensorSystem` is declared: `measuredData`, the recorded series read once
  at the run start, and `measuredValue`, what it reads, the value wired to
  its input or, with nothing wired, its recorded series
  (`measuredValue = Input(default=measuredData)`).  A System that needs the
  measurement connects to `measuredData`.  An `Input(default=...)` may name
  one of the System's own outputs that its `series` fills (the input then
  holds what it reads), and a System may define both `series` (for some
  outputs) and `forward` (for the others).
- `Model.check(start_time, end_time, step_size)` and
  `Simulator.simulate(..., check=True)` probe every declared System, on
  request: one `derive` and `forward` call each on the run-start values,
  writing nothing; one `ValueError` names every System whose `derive` gives
  no dict or whose `forward` leaves out an output, gives one it does not
  declare (or one `series` fills), an output at another size or a state of
  another shape.  `check_system` makes the request.  A `series` that gives
  an output the System does not declare, or a vector output at another
  size, raises at every run start, where it fills the outputs.
- `OccupancySystem` is declared and keeps its parameters estimable through
  the inversion: the children `mass` (`V`, `G_occ`, `m_inf`),
  `supply_damper` and `exhaust_damper` (`a`, `nominalAirFlowRate`, `c`, one
  per damper slot) keep their paths, so `([room, occupancy], "mass.V", ...,
  "shared")` and `([damper, occupancy.supply_damper], "a", ..., "shared")`
  work as before.  It reads its measurements through its own readers (ADR
  0001, decision 6, amended): `co2=` for the room's CO2 and
  `damper_position=` for the damper positions, one reader per terminal
  serving the room, such as the sensors' own
  (`co2=co2_sensor.source, damper_position=damper_sensor.source`).  Its
  `series` reads them at the run start into the outputs
  `indoorCo2Measured`, `previousIndoorCo2Measured` (at a period's first step
  its own, so the first step takes no change of CO2) and
  `damperPositionMeasured`, which its inputs of the same names hand to
  `forward`.  Its file mode goes: `co2_filename`, `damper_filename` and
  their column keywords raise and name the readers.  Wiring those inputs
  from the sensors' `measuredData` raises at the run start: it closes a
  loop through the model (occupancy, room, CO2 controller, damper, damper
  position sensor), which the loader cuts at the damper sensor, so its
  modelled reading lagged its damper by one step.  The flat `V=`, `G_occ=`,
  `m_inf=`, `supply_damper_a=` ... keywords move into the children
  (`mass=dict(V=...)`), with a warning; `c_tied` / `set_c` on its dampers
  raise (give `c` a value).  The full-workflow example, the canonical
  benchmarks and the estimator example give the occupancy the sensors'
  readers (the estimator example's saved model drops the file paths); the
  canonical topology is unchanged, and those models step exactly as with
  the file mode, in object mode, the functional engine and batched.
- A System class whose `forward` cannot take `(x, inputs, p, step_size)`,
  whose `derive` cannot take `(p, step_size)` or whose `series` cannot take
  `(timestamps, step_size)` raises `TypeError` when it is defined (they are
  called positionally, so the argument names are the System's own).
- A `SensorSystem`'s input is optional: a sensor holding data reads it while
  nothing is wired (its run start raises when it has neither); `is_leaf` is
  whether nothing is wired to it.
- A setting that holds a tuple (a piecewise-linear curve's points) is saved
  as JSON and reloads; a saved model with a `PiecewiseLinearSystem` did not
  load.  So does a setting that holds several readers (the occupancy's
  `damper_position`).
- `ToDegCFromDegK` and `ToDegKFromDegC` are declared Systems (they could not
  be built before).
- PIDs of different directions batch into one copy, each acting in its own
  direction, and the gains are no longer `abs()`-ed (#297); walls and
  thermal-mass nodes with different `T_init` start at their own values when
  batched (#293, #296); a fan's flow range is saved with the model (#292).

#### Not yet converted

Three Systems still write their own `do_step` and `initialize`; a test keeps
the list from growing and states each reason.  `ControllerIdentificationSystem`
(and `ControllerIdentificationPISystem`) is exempt for 2.0 and converted in
2.1: it builds its candidates and gates from its wiring.  `FusedStateSpaceSystem`
is exempt: the loader builds it.  `DiscreteStatespaceSystem` is converted
once the numerics work lands.  Until then the base does not raise on an
override.

### Removed

Unused and untested; 2.0 drops them without a deprecation period.

- The estimator's finite-difference route: the `("scipy", <optimizer>, "fd")`
  methods and `Estimator.estimate(n_cores=...)` (the worker pools that
  computed the Jacobian).  Every component is a torch module; use
  `method=(..., "ad")`.
- `TWIN4BUILD_NO_DATA_WARMSTART`, which made collocation start the boundary
  states from a rollout whatever its options said: it duplicated the
  `boundary_state_init` option, and the variable is now ignored.  Use
  `estimate(..., options={"boundary_state_init": "rollout"})`.
- `FmuSystem` (and its `fmuSystem` alias): FMU components and the `fmpy`
  dependency.
- `Model.make_pickable`, `SimulationModel.make_pickable`,
  `SimulationModel.reset_torch_tensors` and `Vector.make_pickable`: they
  prepared a model for the finite-difference worker pools.
- `Model.set_initial_values` / `SimulationModel.set_initial_values`: they
  wrote an output port's tensor (in a shape the port does not hold), which
  the next `initialize` overwrote, so a simulation never saw it, and they
  set no component's state.  Use `Model.set_state` for the state the
  simulations start from.
- The site-specific research scripts in `twin4build/examples/` (the Mortar
  buildings 1 and 13, BTS Site A, another case-study building, the ESIM 2026 calibration and
  optimization plots, the optimization runners and a pseudo-code `ex.py`):
  they read private files and local databases and are no maintained
  examples. They moved to `scripts/research/` in the repository and are no
  longer installed with the package.
- `ClassificationAnnControllerSystem`, `NeuralPolicyControllerSystem` and
  `ShadingDeviceSystem`: the ANN controller could not be built (its weights
  are not shipped), the neural policy's default policy could not step, and
  the shading device only passed a shade position on that no component
  reads.  The SAREF `ShadingDevice` class stays in the semantic model.
- The on-off controllers `OnOffControllerSystem` and
  `SmoothOnOffControllerSystem` (with its deprecated alias
  `OnOffControllerTorchSystem`).  On-off control is a proportional loop
  saturated to the off and on values: `PIDControllerSystem(kp=k, Ti=1e12,
  output_min=off, output_max=on)`, `u = clamp(k * e, off, on)`.  The removed
  smooth controller with steepness `s` about setpoint `sp` is exactly
  `kp = (on - off) * s` about `sp - 0.5 / s` (as `full_workflow_example`
  and the estimator example's saved model now do).  A serialized model that
  names one of the removed classes no longer loads; replace that node with
  a `PIDControllerSystem`.  The controller identification's `onOffSignal`
  gate input is unchanged.

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
- `set_parameters_from_array` → use `set_parameters({(component or id, attr): value})`
- `windows-curses` dependency (curses LOGGER TUI removed)
- unknown public keyword arguments are rejected instead of being silently
  ignored, component constructors included (`System.__init__` dropped any
  keyword it did not know: `PIDControllerSystem(isReverse=True)` built a
  direct-acting loop, `BuildingSpaceSystem(C_air=...)` kept the default; the
  room's submodel values go in `thermal_kwargs` / `mass_kwargs`); removed
  camelCase time/source keywords now raise `TypeError`.  The translator and the
  model loader pass a component only the keywords its constructor takes
  (`System.constructor_keywords`), so a graph's other literals and keys an
  older saved model recorded are left out

### Deprecated (removed in 2.1)

- `*TorchSystem` public names → prefer `*System`
- Schedule/controller camelCase kwargs → snake_case (`weekday_ruleset`, `is_reverse`, `date_column`, …)
- `Model.load(semantic_model_filename=...)` auto-translate → `SemanticModel` + `Translator.translate`; prefer `Model.load(filename=...)`
- `Model.load(simulation_model_filename=...)` → `Model.load(filename=...)`
- `Translator.translate(systems_=...)` → `systems=`
- Public `verbose=` kwargs → configure `LOGGER.verbose` / `LOGGER.logfile`
- `get_component_by_class(dict_, ...)` → `get_components_by_class(Cls)`
- `twin4build.utils.print_progress` → `twin4build.utils.logger`
- Room input `outdoorCO2` → `outdoorCo2Concentration` (#195), the name
  `OccupancySystem` and `OutdoorEnvironmentSystem` use; both default to
  400 ppmv while unwired (the occupancy's 0 ppmv default turned the whole
  indoor concentration into people).  The old name, and the space heater's
  `Power` (→ `toRoomPower`), are removed in 2.0 (see the ADR 0001 section).

### Added

- One scalar output can fill several slots of a Vector input:
  `model.add_connection(room, ahu, "indoorTemperature", "exhaustTemperature",
  input_port_index=[1, 2])` puts the room's temperature on the exhaust slot of
  both AHU branches serving it, as one connection (wiring the same pair again
  on other slots adds them to it; a slot listed twice is an error).  Both
  engines route it, the port is sized by its highest slot, and it serializes
  as a list.  A Vector input can then be wired one slot per branch where it
  used to be one slot per room (the AHU's branch -> room map is then not
  needed: it comes out `None`).
- `SpaceHeaterSystem.toRadiatorPower`: the heat the radiator takes from the
  heating circuit, `waterFlowRate * c_p * (supplyWaterTemperature -
  outletWaterTemperature)`, which is what a heat meter on the circuit
  measures.  `toRoomPower` (formerly `Power`) is the heat the radiator gives
  to the room.  The two differ by the rate of change of the heat stored in
  the radiator: equal in steady state and in the long-run mean, apart while
  it warms up or cools down.  Both are evaluated at the end-of-step state.
  `toRadiatorPower` is bilinear, so it is no row of the linear output
  equation; a fused state-space block computes it after the joint step from
  the member's inputs and its outlet temperature (a unit declares such
  outputs with `SS_DERIVED_OUTPUT_PORTS` and `_ss_derived_outputs`), so it is
  available fused and unfused, batched, in the object and the functional
  rollout, and differentiable.
- Renamed output ports.  `System.OUTPUT_PORT_ALIASES` maps a deprecated
  output port name to the one that replaced it; the outputs of such a class
  are an `AliasedPorts` (a `dict` that also answers to the old name) and
  `System.resolve_output_port` gives the current name.
- Multiple shooting (#91), a transcription:
  `method=(library, optimizer, mode, "multiple_shooting")` makes the initial
  state of every period a decision variable, the first period's too (a fit
  needs no warm-up), and ties it to the end of the period before it by a
  continuity defect.  The periods must be contiguous and of equal length
  (`ValueError` otherwise), and the fit has one phase.  A state's start is
  boxed at 25 % of its magnitude around its recorded value, widened to
  include the end of the period before it.  Its tolerance is a fraction
  (`continuity_sd_rel`, default 0.01) of how much the state varies over the
  periods, and a state that stores heat is held to at most
  `continuity_energy_tol / C` (J, default 3.6e5, 0.1 kWh), `C` its heat
  capacity: a jump in a wall or the shared interior costs by the heat it
  creates, room air and radiators keep their range tolerance.  The two
  options hold for the whole fit (a schedule phase may not set them).  The
  result carries the estimated states (`estimated_initial_state`,
  `estimated_initial_state_instances`) and the jumps at the boundaries
  (`continuity_jumps_instances`, `continuity_tolerance_instances`); the fit
  logs the largest jumps, and
  `twin4build.estimator._continuity.continuity_summary` ranks them and marks
  the states re-set to one side at every boundary.
- Multiple shooting across gaps (#262): with `continuity_gaps=True` the
  periods may leave gaps between them, so one fit can cover several spans of
  data (two seasons or two years).  A period that does not start where the
  one before ends is not tied to it (no defect, its box not widened by the
  other span's end): it starts free like the first period, while the periods
  within each span stay tied.  Contiguous periods behave as before.
- `System.state_heat_capacities()` gives every state's capacity [J/K]
  (`NaN` for a state that holds no heat); the room, wall, radiator and
  thermal-mass node declare theirs, composites and fused blocks concatenate
  their units'.
- Model state.  `Model.set_state(values, period_starts=None)` sets the
  initial state by component id; with `period_starts` a row is matched to the
  simulated period that starts at that time.
  `Model.load_estimation_result(filename, parameters=True,
  initial_state=True)` sets the estimated initial state with the parameters,
  so a simulation of a fit starts where the fit started.
- Window batching.  Several periods are rolled out as one batch
  (`Simulator.rollout_functional_windows`,
  `rollout_functional_batched_windows`), one initial state per period.
- `Simulator(cuda_graph_scope="step")`: one CUDA graph per step, replayed along
  the rollout, for models whose whole rollout does not fit in one captured
  graph.
- `ThermalMassNodeSystem`: one hidden thermal mass that many walls share.  A
  component declares the outputs that are read one step late (the class
  attribute `LAGGED_OUTPUT_PORTS`, or a method `lagged_output_ports()`), and
  the loader cuts the algebraic loop there.
- `FunctionSystem` and `WeightedSumSystem` batch.
- System registry (`twin4build.systems.registry`, #132): a `System` class
  that lives in another package is registered under a stable type id,
  `tb.system_registry.register(cls, type_id="acme:CoilSystem@1",
  provider=..., version=...)`.  A serialized component of a registered class
  records the type id, the provider and the provider version, and
  `Model.load(filename=...)` resolves the class from the type id through the
  registry of the model -- no class has to be set on `twin4build.systems`.
  A type that is not registered raises `UnknownSystemTypeError` naming the
  component, the type id and the provider to install.  Built-in classes need
  no registration and are serialized as before; a model that records only
  class names still loads.  `tb.system_registry` is the default registry;
  `Model(id=..., system_registry=...)` and
  `Translator(system_registry=...)` take a registry of their own, which the
  translator also uses as the whitelist of the external classes it accepts.
- `twin4build.model.partition.keep_groups(model, partition, groups,
  with_senders=True, stop_at=())` keeps some groups of a measured partition
  (a few rooms of a building) and removes the rest of the model: the groups
  named, with `with_senders` the groups that send into them over a measured
  edge (not followed into the groups of `stop_at`, e.g. the air handling
  unit's), and every leaf that feeds a kept component.  A
  `ReturnFlowJunctionSystem`'s `branch_temperature_slots` follow the
  surviving connections.

- `Simulator.measurement_frames()` and
  `Simulator.measurement_errors(skip=0)` compare the simulation that just
  ran with the measurements: for every sensor that reads a computed port
  and holds data, a frame (`measured`, `simulated`, `scored`) and a row of
  errors (`sensor`, `port`, `n`, `mae`, `rmse`, `bias`) over the samples
  the sensor's `scoring_mask` scores, after the first `skip` steps of each
  period.  They work on the model the simulator ran, unbatched or batched.

- Parameters by component id, across batchings.  A result also carries
  `parameter_instances_fixed`: every estimable parameter the fit held fixed
  (pinned, fixed, left out of the selection) at the value it ran with, and
  `load_estimation_result(..., fixed=True)` (the default) sets them before
  the estimated ones, so a model simulates as it was fitted whatever its own
  setup gave those parameters.
  `Model.get_source_component_ids(component)` gives the ids of the
  components a batched meta stands for;
  One public pair reads and writes parameters:
  `Model.get_parameters(parameters=None)` reads the estimator's parameter
  entries (`(component or component id, attr, ...)`; `None`: every
  parameter the components list in their configuration) as
  `{(component id, attr): array}` in physical units, one key per instance
  of a meta; `Model.set_parameters(values, strict=False)` writes
  `{(component or component id, attr): values}` to the model whatever its
  batching (the unbatched model, or a meta's instance, the other instances
  left alone) and returns `{"applied": n, "missing": m}`.  The parallel-list
  writer with normalization, bounds and `overwrite` is the estimator's own
  (private).  `get_parameter_values` / `set_parameter_values`, the names
  these had during 2.0's development, warn and are removed in 2.1.  A saved fit carries
  `parameter_instances`, `parameter_instances_x0` (the values the fit
  started from), `parameter_instance_bounds` and `component_source_ids`,
  and `load_estimation_result` applies the values by component id when the
  result was fitted on another batching of the model (its ids do not
  resolve, or resolve to metas that stand for other components) instead of
  failing with a `KeyError`.

- `SensorSystem`: `scoring_mask` (the samples an estimation scores, a
  boolean Series indexed by time) and `measurement_sd` (the standard
  deviation the sensor is scored with) are declared attributes and
  constructor arguments, and `Model.batch_components` carries them to the
  sensor's copy in the batched model.  `set_series(series, uuid=None)` puts
  a sensor on an in-memory series and names it without switching it to
  database mode, and `SensorSystem(df=frame, uuid="X")` is an in-memory
  sensor named "X".  Assigning `uuid`, `dbconfig` or `filename` the value
  the sensor already has no longer switches its source (`Model.load`
  restores the saved configuration by assignment and used to switch a
  sensor on a series back to the database); a new value switches as before.
- `Estimator.estimate`: an entry of `measurements` may be the sensor alone
  or `(sensor, None)`, in which case the sensor's `measurement_sd` is used;
  an entry with neither raises a `ValueError` that names the sensor.

- Workflow accessors, so that a workflow in an add-on package need not
  reach into private attributes:
  `twin4build.utils.get_main_dir.set_main_dir(path)` sets the folder models
  keep their files in (instead of overwriting the module's `_main_dir`);
  `Model.serialize()` returns the path of the saved instance graph and
  `Model.instance_graph_path` gives it without serializing;
  `discretize_onestep` is the public name of the one-step zero-order-hold
  discretization (`_discretize_onestep` is unchanged);
  `SpaceHeaterSystem.solve_UA()` returns the `UA` that meets the nominal
  sizing, so a radiator can be sized before the model is initialized.

- `Optimizer`: a decision variable may name a component's ``tps.Parameter``
  instead of an output port (`(component, "Y", lb, ub)`), so a handful of
  numbers (the points of a compensation curve, a gain, a setpoint) can be
  optimized next to, or instead of, actuator trajectories.  The decision
  vector is the trajectories followed by the parameters (normalized to
  [0, 1] by their bounds); the functional objective hands the parameter
  partition to the composed map as its theta, exactly as the Estimator does
  with estimated parameters, and `pareto_front` sweeps over it unchanged.
  Parameter decision variables need the functional objective.
- `PiecewiseLinearSystem`: the Y points are a `tps.Parameter` (with
  optional `Y_bounds` making them estimable) and the interpolation is a
  pure, differentiable function of them, so an outdoor-temperature
  compensation curve can be estimated or optimized.

- Example patterns (`twin4build/examples/patterns`): the AHU pattern reads
  the supply fan's speed command and the AHU-level flow meters; the space
  heater pattern takes the supply water temperature from the heating
  circuit that feeds the radiator (`isFedBy`) instead of a constant; valve
  loop patterns for a room's heating command with and without an explicit
  radiator node; a room-CO2 occupancy pattern (`occupancy_patterns()`)
  feeding `OccupancySystem` from the room's CO2 sensor, its VAVs' damper
  commands and the AHU's fan speed; radiator-loop controller patterns for
  both graph shapes.

- Controller identification, playback: `rewire(mode="playback")` opens the
  identified loops and replays each loop's measured command; an opened
  loop's command sensor (`actuatorMeasured`) is its actuator measurement;
  an already opened loop is recognised on reload; a loop without
  excitation replays its measured command in simulate mode.
- Controller identification, rewire: the identified action sign (direct /
  reverse) reaches the PI; the identified gate stays active in simulate
  mode and a loop whose gate seeding could not run keeps its pinned gate;
  level-based setpoint tie-break and the park value taken from the off
  regime; the radiator loop is gated on the VAV flow setpoints.
- `SpaceHeaterSystem`: `UA` and `thermalMassHeatCapacity` carry bounds and
  are estimable through the base `get_estimable_parameters` contract.

- `OccupancySystem` inverts the zone's CO2 step exactly. It now builds the
  same matrices (`mass_matrices`) and runs the same ZOH discretization
  (`_discretize_onestep`) on the same four-slot input vector as
  `BuildingSpaceMassSystem.forward`, with the occupancy slot zeroed, and
  divides by that slot's column of `Bd`.  Only `ACTIVE_INPUT_SLOTS` of
  `B_eff` are ever nonzero and `Bd` is linear in `B_eff` column by column,
  so the inversion drops the other columns first and exponentiates a 3x3
  block instead of a 5x5; the dropped columns enter the Taylor/squaring
  recursion as exact zeros, so `Ad` and the kept `Bd` entries are bit
  identical (pinned by tests).  This matters because the captured
  reverse-mode rollout, not the model, is what exhausts device memory:
  one capture of a 3-window 288-step rollout measured 5.2 GB on an 8 GB
  card, and a replay with no headroom left aborts as a segfault or an
  illegal memory access rather than a clean out-of-memory error.  The previous explicit-Euler
  rearrangement is a different map at AHU sample times (`λ Δt ~ O(1)` at
  600 s), so the people it booked did not reproduce the measured CO2 when
  pushed back through the zone.  `SUPPORTS_TRANSFORM_MODE = True`, so the
  composed and compiled paths get the `torch.matrix_exp` / `_expm_ss` /
  `_expm_ss_fused` switch the zone gets.  Note that an exact inverse makes
  measured-vs-predicted CO2 uninformative about `V`, `G_occ` and `m_inf`
  (only their effect through `N_occ -> Q_occ N_occ -> T` identifies them,
  and only the product `Q_occ V / G_occ` unless one factor is pinned).
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
- `OccupancySystem`: optional `fanSpeedMeasured` input gating its damper
  flows (the inverse balance sees no air when the fan is off), optional
  smoothing of the inverted occupancy, and the damper flows derived from
  the historised damper COMMAND through the same characteristic the AHU
  uses, so the inverse and the forward balance see one and the same flow.

- `AirHandlingUnitSystem`: optional `supplyFanSpeed` input gating the branch
  flows (a stopped fan moves no air whatever the dampers say);
  `totalSupplyAirFlowRate` / `totalExhaustAirFlowRate` outputs for the
  AHU-level flow meters; `exhaust_ratio_per_branch=True` gives
  `exhaustFlowRatio` one entry per branch (sized at `initialize`, like the
  other per-branch parameters) instead of one per AHU.
- `DamperSystem.set_c(...)` unties the damper characteristic's offset `c`
  from `a` so a minimum flow at a closed damper is an estimable parameter;
  the AHU lists a supply damper's untied `c` among its estimable parameters
  and the offset survives serialization.
- `FanSystem`: the temperature rise across the fan is bounded.

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
- `twin4build.settings` (`tb.settings`): the library's knobs in one place,
  each with its environment variable, type and default (the manual's
  *Settings* page lists them).  A value comes from `tb.settings.set(...)`,
  else the environment variable, else the default; `override(...)` sets
  values for a `with` block, `reset(...)` undoes `set`, and `snapshot()`
  gives every setting with its value and source.  The results of
  `Estimator.estimate` (and the file it saves) and `Optimizer.optimize`
  carry that snapshot as `result["settings"]`, and `ParetoResult` as
  `settings`, so a result records how it was computed.
  The settings off their defaults are logged when an estimation, a
  simulation, an optimization or a Pareto front starts.  The library reads
  a setting where it uses it, no longer when a module is imported, so a
  variable set after `import twin4build` takes effect.  The environment
  variables keep their names; their values are now checked: a boolean takes
  `1`/`true`/`yes`/`on` or `0`/`false`/`no`/`off`, an empty one is unset,
  and other text raises (`TWIN4BUILD_JAC_CHECK=0` and
  `TWIN4BUILD_BENCH_TIMESTEP=0` used to switch them on; `T4B_ZOH_EXPM`
  other than `phi1` or `block` used to mean
  `block`).  The module attributes that held the values read at import
  (`discrete_statespace_system.ZOH_EXPM` and `EXPM_CHECKPOINT`,
  `_functional.ROLLOUT_CHECKPOINT`, the translator's `_MATCH_DIAG_PATH` and
  `_MATCH_DIAG_PATTERN_FILTER`, `_cuda_graph.SYNC_PHASES_ENV`) are gone.

### Changed

- Translator: a list sender whose slots come from another list of the same
  match takes the slots at its own positions.  In an AHU's pattern,
  `vavs` is every VAV the unit feeds and `spaces`, reached by
  `StepRule(subject=vavs, object=spaces)`, the room of each VAV; with
  `sp.add_connection(spaces, "indoorTemperature", "exhaustTemperature",
  input_port_index=vavs)` each room lands on the slots of its own VAVs, the
  same slots the VAVs' damper commands take.  Every room used to get every
  VAV's slot, and one of them kept all of them.  Two lists are paired when
  they are filled from the same set rule (`SignaturePattern._paired`), not
  when they have the same length; a connection between lists the pattern
  does not pair raises `ValueError`, as does a slot read from a list of
  several values with nothing pairing it to the sender (it used to take the
  first).
- `FanSystem`: the power curve in Bernstein form over a fixed flow range.
  `beta0` .. `beta3` [W] (the power at the curve's control points, each
  `>= 0`, so the power is `>= 0` over the whole range) replace
  `nominalPowerRate`, `nominalAirFlowRate` and `c1` .. `c4`;
  `flowRangeLower` / `flowRangeUpper` [kg/s] fix the range (not estimated).
  `set_flow_range` re-maps the betas so the curve stays the same;
  `polynomial_to_bernstein` / `bernstein_to_polynomial` convert between the
  two forms (the same cubic).  The betas are size quantities:
  `size_to(m, P)` scales the curve through the operating point and
  `curve_power(m)` evaluates it.

- A model loaded with `Model.load(filename=...)` serializes to a file that
  loads again.  The loader rebuilt the connections but the semantic model
  kept the instances it had cached while reading the file, and
  `serialize()` wrote their triples back: every connection twice and JSON
  literals as Python reprs, which the next load could not parse.

- `twin4build.examples.patterns` is trimmed to the patterns the library's
  own translator example matches on its one-room model (16 patterns: the
  SAREF-shaped zone, outdoor environment, PID controller, schedule, damper,
  space heater and valve patterns and the example's sensor patterns).  The
  Brick/BMS-shaped patterns (VAV zones, damper commands, room-attached
  sensors, AHU flow meters, radiator loops, identified PI loops, room-CO2
  occupancy, fan-coil units, flow junctions, on/off controllers, air-to-air
  heat recovery) and their tests are gone from the library: patterns are a
  deployment's own, written next to the graphs they describe and passed to
  `Translator.translate(patterns=...)`.  The SAREF valve pattern now has an
  explicit id (`valve_signature_pattern`).

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
- A CUDA graph's first replay is checked against the eager run within the
  bound on reordered floating-point sums, `8 * n * u * scale` element-wise
  (`n` the widest dimension of the function's tensors, `u` the dtype's unit
  roundoff, `scale` the output tensor's largest magnitude), instead of a
  fixed `rtol=1e-9, atol=1e-11`.  In float32 a replay that sums in another
  order (atomic accumulation) differs by a few ulp, which the fixed
  tolerance rejected at random; float64 stays strict (about `1e-13`
  relative at `n = 100`).  The error names the difference observed and the
  one allowed; a passing check logs both at debug level.  The per-step
  graphs (`cuda_graph_scope="step"`) and `CudaGraphCallable` both use it.
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

- Preferred system names without `Torch` suffix
- Module files renamed accordingly (`*_torch_system.py` → `*_system.py`)
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
- LOGGER: the indented tree is printed on stdout (ANSI colors on a
  terminal), or appended as plain text to `LOGGER.logfile` when one is set.
  `logfile=None` means no file: the implicit `progress.log` in the working
  directory is gone (#138). A logfile that cannot be written gives a
  `RuntimeWarning` once instead of failing silently. `load(logfile=None)`
  leaves a configured `LOGGER.logfile` as it is.
- `LOGGER.use_stdlib_logging()`: opt-in forwarding of every line as a
  `logging.LogRecord` on `logging.getLogger("twin4build")` (level from the
  badge; badge, nesting depth and line number in `t4b_*` attributes;
  `change_status=True` arrives as `"<message> - OK"`). Nothing is printed
  while it is on, and the root logger and `logging.disable` are not touched.
- `LOGGER.is_interactive()` asks `sys.stdout.isatty()` instead of inspecting
  `__main__`; `LOGGER.interactive = True | False` overrides it.
- Top-level `tb.types` / `Vector` / `Scalar` / `Parameter` / `State`
- Package version `2.0.0`
- Canonical simulation, estimation, optimization, and scaling benchmarks now
  use the complete translated `full_workflow` topology; batching parity covers
  batched full-workflow controls, nested parameters, and data-source settings.
- The constrained optimization matrix compares AD SLSQP and trust-constr.
  Pareto scaling uses one shared broadcast valve schedule and reports exact
  decision/constraint dimensions plus convergence and comfort violations.
- A port that does not log (`set_save_simulation_result(False)`) and holds
  no data keeps its current value only: it allocates no time history at its
  next `initialize`, scalar and vector ports alike, in object and functional
  runs.  `set_save_simulation_result` now sets vector ports too (it set the
  scalar ones only).  An estimation logs its measuring devices only, so the
  histories of every other port no longer take memory; a functional run
  still fills the history of an output wired into a logged input (a
  sensor's `measuredValue`).  What is logged is unchanged, bit for bit.
- `Estimator.estimate` puts every port's logging back as it was when it
  returns or raises.  Its simulations log the measuring devices only, and it
  left them so: a simulation after it kept no history of any other port,
  vector ports included now that `set_save_simulation_result` sets them
  (`history()` raised "History is not populated").
- A fused state-space block builds its matrices unit by unit, each unit its
  own rows, and holds its bilinear terms (`E`, `F`) as their entries
  (`discrete_statespace_system.SparseBilinear`): a few thousand of the tens
  of millions a dense `(n_c, m, n, n)` / `(n_c, m, n, m)` tensor has for a
  storey-wide block, where padding every term to the joint shape ran an
  8 GB GPU out of memory and broadcasting the inputs over the dense terms
  took most of a step.  The entries' indices and the fixed order they are
  summed in (`bilinear_plan`) are built once from the block's structure, so
  a step gives the same bits on every call and in a CUDA graph, and also
  when the first assembly runs inside a `torch.func` transform.  The new
  `fused_sparse` setting (`T4B_FUSED_SPARSE`, on) switches back to dense
  terms; the matrices are the same.  The per-step CUDA graphs' adjoint
  gives an integer constant a zero cotangent instead of asking for its
  gradient.
- `zoh_expm="aug"` (`T4B_ZOH_EXPM=aug`): a transform-mode state-space step
  reads the next state off ONE matrix exponential of size `n + 1`,
  `exp([[A T, B T u], [0, 0]])`, instead of the `phi1` pair and an `n x m`
  product: about half the products, and the most accurate of the three on
  stiff blocks (about 1e-15 of the state against `torch.matrix_exp`, `phi1`
  about 1e-6).  The default stays `phi1`.
- The `small_matmul_max` setting (`T4B_SMALL_MATMUL_MAX`, 16): while
  `torch.compile` traces, a matrix exponential up to that size is written
  with broadcast products for Inductor to fuse, a larger one with `bmm` (a
  159-state fused block's step took 26 ms with the broadcast products, 5.8
  with `bmm`).  Every exponential used the broadcast products before.  A
  setting is a `bool`, an `int` or a `str` (`Setting(kind=int)`).
- `fusable` (default `True`), a structural setting of the Systems with
  fusable ports (`WallSystem`, `BuildingSpaceSystem`,
  `BuildingSpaceThermalSystem`, `SpaceHeaterSystem`): an instance set
  `fusable=False` joins no fused block and couples to its neighbours one
  step late, as with fusion off (walls between rooms that should not fuse
  the rooms into one block).  Batched copies share it: walls that fuse and
  walls that do not batch apart.  The partition (`split_worlds`) sees no
  fusable arc through such an instance.
- A fused block keeps a member's vector input fed partly inside the block
  and partly from outside (a room's `wallHeatGain` from a fused wall toward
  a hallway and from an unfused wall toward another room): the inside slots
  are substituted exactly, and only the outside slots enter the port's
  column.  Such a cluster could not be simulated before (the fused block
  had no input of that name).
- The rooms' internal gains and openings: `BuildingSpaceThermalSystem` and
  `BuildingSpaceSystem` take `plugLoad` [W] (electrical plug load, all of
  it heat in the air; 0 while unwired) and `exchangeHeatGain` [W] (fusable,
  one slot per connected opening to another zone: a port apart from
  `wallHeatGain`, so the walls between rooms can stay out of fusion while
  the openings fuse).  Both enter the air node as `heatGain` does.
- `BuildingSpaceThermalSystem.eta_inf` (0 to 1, default 0): the share of
  the indoor-outdoor temperature difference the space's outdoor air has
  recovered before it enters.  The infiltration and an outdoor make-up
  stream arrive at `T_o + eta_inf (T_i - T_o)`; a make-up stream from a
  neighbouring zone (`makeUpAirTemperature` wired) and the CO2 are not
  touched.  A composite's children see the composite's wiring
  (`System.input_defaults()` of a room's `thermal` is the room's).
- `split_worlds` provides a lagged hub instead of copying it: a component
  with free parameters whose outputs the other worlds read are all lagged
  (`LAGGED_OUTPUT_PORTS`, a building's hidden interior node) and that one
  world needs only through measured or exogenous signals lives in that
  world (its home), and every other world reads it through a cut
  connection: `Model.add_connection(..., detached=True)`, the value without
  its gradient (`Worlds.provided` lists them as `(output port, reader,
  input port)`).  Both engines pass the value on detached and still
  evaluate the sender every step, the partition reads the signal as data,
  and the estimator's structure walk (`FunctionalModel.index_coupling`) lets
  no parameter through, so the readers' blocks stay apart from the home's.
  A batched meta holds one flag per pair (its instances may read the sender
  in different ways), and a saved model keeps the flag.
- `System.JOINS_WORLDS` (default `False`): a component whose class sets it
  (an opening between the zones of one room) joins the worlds of the scored
  spaces it reads or feeds into one world (`Worlds.merged`) instead of being
  copied with a measurement of the other side.

### Migration notes

1. Replace `BuildingSpaceTorchSystem` with `BuildingSpaceSystem` (etc.).
2. Use `weekday_ruleset=` / `is_reverse=` / `date_column=` instead of camelCase.
3. Give a schedule, an outdoor environment, a time-series input or a sensor a reader:
   `source=CsvSource(path, date_column=..., value_column=...)`,
   `DataFrameSource(df)` or `DatabaseSource(...)`.
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
