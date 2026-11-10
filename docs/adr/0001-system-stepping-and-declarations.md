# ADR 0001: A System is its declarations and one forward step

- **Status:** accepted, 2026-11-05; decisions 7 and 10 amended 2026-11-10
  (`OccupancySystem` is converted, not deleted; the probe runs on request).
  To be implemented in 2.0.
- **Scope:** how a System is written, stepped, sized, fed with data and
  checked, for every System in the core package and in the packages that
  build on it.
- **Terms:** as defined in [GLOSSARY.md](../../GLOSSARY.md).

## Context

34 of the 37 core System classes write their own `do_step`, and 30 also
write a `forward`. So the same physics is written twice: `do_step` for object
mode and `forward` for the functional engine, plus run-start code in each
System's `initialize`. Only about six Systems have a test that the two
agree, each written by hand.

Systems also do work that belongs to the base:

- they read their own connections at run start to size their vector ports
  and to pick a mode (the AHU, the junctions, the sensor, the air transfer
  node);
- data sources read files inside `initialize` and are stepped like any other
  System, although their values are known before the run;
- parameter-derived values (a PID's discrete coefficients, a state-space
  System's discretised matrices) are cached inside `forward` through
  `transform_mode` and `step_constants`;
- three Systems keep a branch -> room map, because one scalar output could
  not fill several slots of a vector input.

Every check at load only logs a warning. The flags it sets are read by
nothing, and the closing assert in `SimulationModel.validate` is commented
out, so a model with a missing input still runs.

## Decision

### 1. Stepping

- A System's stepping logic and physics live only in
  `forward(self, x, inputs, p, step_size)`. It returns the next state and the
  outputs.
- `do_step` and `initialize` belong to the base. A subclass that defines
  either raises `TypeError` at class definition. There is no deprecation
  period.
- One step size per run, the same for every simulation in a batch. The PID's
  per-simulation step-size tensor goes.
- `forward` gets no clock or date. Anything that depends on the time reaches
  it through an input port.
- Every tensor `forward` receives has one form: the first dimension runs over
  the copies being stepped, then the port's or state's own size. Object mode
  folds simulations times copies into that dimension. The functional engine
  passes the copies of one simulation and keeps its `vmap` over rows
  (windows, multi-starts, collocation segments).

### 2. Declarations

A System is written as flat class-level declarations, with no `__init__`:

```python
class DamperSystem(System):
    damperPosition = Input()
    damperPosition = Output()
    airFlowRate = Output()
    a = Parameter(5.0, lb=0.1, ub=10.0)

    def forward(self, x, inputs, p, step_size):
        ...
```

- `Input()`, `Output()`, `State()`, `Parameter()`, `Setting()` and `Child()`.
- `size=` takes an int, `"wired"`, or another port or setting
  (`exhaustTemperature = Input(size=exhaustDamperPosition)`).
- `default=` replaces `optional`. It names another input, a value, or one
  of the System's own data-source outputs
  (`measuredValue = Input(default=measuredData)`).
- The metaclass's `__prepare__` files ports into separate input and output
  registries as the class body runs, so one name may serve one input and one
  output. Nothing is renamed, and ports are not class attributes. Any other
  repeated name is a `TypeError`. Subclasses inherit and may redeclare.
- At runtime, ports have attribute access as well as `["name"]`:
  `damper.input.damperPosition`, `damper.output.damperPosition`. Parameters
  and settings stay plain attributes (`damper.a`).
- A renamed port declares its old name: `Output(alias="old")`. The two
  aliases that exist today are removed in 2.0: the space heater's `Power`
  (now `toRoomPower`) and the room's `outdoorCO2` (now
  `outdoorCo2Concentration`).

### 3. Composite Systems

- A composite declares its children with `Child(Cls, **values)`. A user
  overrides a child's values with `name=dict(...)`, or redeclares the child in
  a subclass.
- A composite's ports are its own declarations, never borrowed from its
  children. The building space writes out its 19 ports.
- A child's parameters keep their dotted path (`space.thermal.C_air`).

### 4. Settings

- `Setting(...)` declares a constructor value that is never estimated: a
  size, a modelling choice, a constant.
- Changing a setting after construction marks the model for reload, like
  editing a connection.
- A saved model that names a predicate no class takes raises on load.

### 5. Derived values

- `derive(self, p, step_size)` computes values that depend only on the
  parameters and the step size.
- The caller evaluates it inside autograd: once per run in object mode, once
  per rollout in the functional engine. The results reach `forward` in `p`.
- `forward` caches nothing. `transform_mode`, `SUPPORTS_TRANSFORM_MODE` (14
  Systems) and `step_constants` go. `StateSpaceSystem` becomes a
  base-written `derive` plus `forward`.

### 6. Inputs and data sources

- Inputs arrive only through connections. No System reads files for its own
  inputs. The occupancy System's file mode (`co2_filename`,
  `damper_filename` and their column keywords) goes.
  `ScheduleSwitchControllerSystem` is deleted: it has no users.
- A data source is a System whose outputs are a series known before the run.
  It defines `series(self, timestamps, step_size)`. The base calls it once at
  run start with the `(n_s, n_t)` timestamps that `get_simulation_timesteps`
  returns, and fills the outputs from it. Nothing steps a data source.
- A data source takes one `source` setting holding a reader: `CsvSource`,
  `DataFrameSource` or `DatabaseSource`. The outdoor environment takes one
  reader per output. Reading options (resampling, gaps, fill) live on the
  reader. A file reader saves its path relative to the saved model. The
  `use_spreadsheet` / `use_database` / `use_df` / `use_dict` switches go.
- Data sources: outdoor environment, schedules, the piecewise-linear
  schedule, time-series inputs, the recorded series of a sensor, and a new
  core `TimeOfDayBasisSystem`. The last replaces the DataFrame basis
  leaves that downstream code writes for itself.
- A sensor has two outputs. `measuredData` is the recorded series, filled by
  `series`. `measuredValue` is an identity `forward` of its input, which
  defaults to `measuredData` when nothing is wired. A System that needs the
  measurement connects to `measuredData`.
- The examples switch from `PiecewiseLinearScheduleSystem` to
  `PiecewiseLinearSystem`.

### 7. Occupancy

- A forward `OccupancyProfileSystem`, written downstream so far, moves into core: people in
  a room are a time-of-day basis weighted by estimated knot values, and the
  room's CO2 sensor scores the fit.
- The inverse `OccupancySystem` is deleted (amended: it is converted, see
  the Implementation, step 9). Its inversion survives as an
  `invert_zoh_occupancy` helper for start values.

### 8. Load

- Load is the once-per-model step, repeated after any edit to the
  components, connections or settings. It is owned by the base: no System
  looks at its own connections.
- Each vector port's width comes from its declared `size=` and its
  connections. Ports sized after the same port or setting must agree.
- An unwired input falls back to its `default=`.
- The thermal model's deprecated in-zone boundary wall (`C_boundary`,
  `R_boundary`, `_inactive_parameters`) is removed.

### 9. Wiring to several slots

- One scalar output may fill several slots of one vector input:
  `add_connection(room, ahu, "indoorTemperature", "exhaustTemperature",
  input_port_index=[1, 2])`. The slots merge into one connection, and a slot
  wired twice is rejected.
- Room temperature inputs become one slot per branch. The three
  branch -> room maps go: the AHU's `_exhaust_branch_map`, the return
  junction's `branch_temperature_slots` and a downstream air transfer
  node's own map. So do `partition._repair_return_junctions` and the
  room lookup in `factor_units`.
- In a translator pattern, a sender list and a slot list are paired only
  when the pattern's rules fill one from the other, which means both are
  reached from the same set rule. Position k sends `spaces[k]` into the slot
  of `vavs[k]`. Unpaired lists, or paired lists of different lengths, raise.

### 10. Checks

Every check runs automatically, once, at the first moment its information
exists, and raises, except the probe, which runs on request. Nothing is
checked during the steps themselves.

| When | What is checked |
|---|---|
| Class definition | no name declared twice (except one input and one output); no `do_step` or `initialize` override; `size=` and `default=` name a real port or setting; `forward`, `derive` and `series` take the arguments above |
| Construction | no unknown keyword; every setting without a default is given |
| Load | every input without a default is wired; wired widths match declared sizes; ports sized after the same port or setting agree; translator lists are paired; every parameter has a value after the config is read |
| Every run start | `series` returns every output it fills at `(n_s, n_t, size)` and no other |
| On request: `model.check(start_time, end_time, step_size)` or `simulate(check=True)`; `check_system` requests it | one probe per System on the run-start values: `forward` returns exactly the declared outputs at their declared sizes and a state of the state's shape; `derive` returns a dict; every output is filled by exactly one of `series` or `forward` |
| After each run | every connected input stayed finite (as today) |

The per-System `validate(self, p)` hooks (schedule, sensor, outdoor
environment) and the sensor's `validate_connections(self, p)` go. So do the
model-level `SimulationModel.validate`, `validate_components`,
`validate_connections`, the `_validated_for_*` flags, `is_validated` and
`load(validate_model=...)`.

The probe runs on request (amended 2026-11-10, the maintainer's decision;
the draft ran it at the first run start after every load). Run at every
load, it would catch a wrong output name, size or state shape before it
reaches a fit. It would also cost one `derive` and one `forward` call per
System on every load of every simulation, estimation and optimization, for
a check that a System's author needs and `check_system` already requests.

### 11. Conformance check

`twin4build.testing.check_system(system, inputs)` is the test every System
passes, in core and in other packages. It checks:

- save and reload give the same settings and parameters;
- a deepcopy steps the same as the original and shares nothing with it;
- a few steps in object mode and in the functional engine agree;
- two copies with different parameter values, batched, match the same two
  copies stepped separately;
- the gradient of each output with respect to each parameter is finite;
- the System runs on CUDA when a GPU is present.

Example inputs live in the tests, not in the System class. The core tests
keep one table with an example per core System, and fail when a core System
class has no entry. Downstream packages call `check_system` on their own
Systems.

### 12. Names

`step_size` everywhere: `forward(x, inputs, p, step_size)`,
`derive(p, step_size)`, `series(timestamps, step_size)`,
`simulate(step_size=...)`. Not `sample_time` or `dt`.

## Rejected alternatives

| Alternative | Why not |
|---|---|
| Keep `do_step` overridable, or deprecate overrides over 2.x | Two places to write the same physics, kept in step only by hand-written tests. 2.0 is the major release that can break the System interface. |
| A step size per System, or per simulation | Only the PID carries a per-simulation step-size tensor. One step size per run keeps `derive` once per run. |
| A `forward` that handles one copy, batched by `vmap` per instance | Measured at about 190 µs per call against about 15 µs for one batched call. |
| Nested `class Inputs:` / `class Outputs:` blocks | Harder to read than flat declarations. The metaclass registries allow one name for an input and an output without them. |
| Renaming same-name ports (`damperPosition_in`) | Changes port names that saved models and patterns use. |
| One per-System load method for wiring-dependent choices | That is a System looking at its own connections. `size=` and `default=` express every case found: widths, unwired fallbacks and width agreement. |
| The AHU's branch -> room map as a setting | Indexing must come only from the connections. A scalar output filling several slots removes the need for the map. |
| Pairing any two translator lists of the same length | Silent and wrong for the AHU pattern, where it put every room on every slot. Pairing must come from the pattern's rules. |
| Declared self-fed inputs: a System reading a file for one of its inputs | A second way for data to reach a System. A data source plus a connection already does it. |
| Stepped data sources (a `do_step` that reads the next sample) | Their values are known before the run. Filling them once is cheaper and needs no state. |
| Data sources keeping their own `initialize` | Run start belongs to the base (decision 1). `series` is the one thing a data source adds. |
| A clock data source feeding schedules through ports | A schedule's values follow from the timestamps alone. Computing them once at run start is simpler than stepping calendar logic. |
| A `run` object, or a `DatetimeIndex` with a frequency, as the argument to `series` | Wraps what a System needs, the timestamps and the step size, in something it must unpack. |
| Caching derived values inside `forward` (`transform_mode`) | `forward` stays a pure function. The caller evaluates `derive` once, inside autograd. |
| Keeping the inverse `OccupancySystem` | The forward profile fits occupancy as a model. The inversion stays only as a helper for start values. |
| Keeping the two port aliases until 2.1 | Each alias is a second name for the same port, and 2.0 already breaks the System interface. |
| Checks that only warn | Today's checks only warn, and models with missing inputs run. |
| All checks only on `model.check()` | A model with a missing input then loads and runs. |
| The probe at the first run start after every load (the draft of decision 10) | Overridden by the maintainer: it would catch a wrong output name or size before a fit, at one `derive` and `forward` call per System on every load; the probe runs on request, and `check_system` requests it. |
| A conformance test that is internal to core, or none | Downstream packages' Systems are the ones with the most to rewrite in 2.0. Without a shared check they can drift from core unseen. |
| Example inputs declared on the System (`Input(example=...)`) | Adds to the declarations, which should stay as simple as possible. |

## Consequences

- **Every core System is rewritten** as declarations plus `forward`,
  `derive` or `series`.
- **Renamed:** `sample_time` becomes `step_size` in every `forward` (39
  files today) and in the engine's callers.
- **Removed:** `ScheduleSwitchControllerSystem`, the
  `Power` and `outdoorCO2` aliases, `C_boundary` / `R_boundary`,
  `transform_mode` / `SUPPORTS_TRANSFORM_MODE` / `step_constants`, the
  occupancy file mode, the `use_*` switches, the per-System `validate`
  hooks, `is_validated` and `load(validate_model=...)`.
- **Saved models:** a predicate no class takes raises on load; file paths in
  readers are relative to the saved model.
- **Examples:** `bems_example_lecture.ipynb` (cell 3) defines its own
  System with `initialize` and `do_step`, and becomes a `forward`. The
  minimal example's text says the simulator calls `do_step`, and it uses the
  deprecated `weekDayRulesetDict`.
- **Tests:** `tests/optimizer/test_parameter_decision_variables.py` uses
  the `Power` alias.
- **Downstream packages:** Systems that override `do_step` are rewritten.
  Patterns that index the exhaust temperature by terminal unit, and air
  transfer nodes that keep their own branch -> room map, change with the
  paired slots. DataFrame basis leaves are replaced by
  `TimeOfDayBasisSystem`.

## Implementation

A series of pull requests against `dev`, each keeping the tests passing:

1. One scalar output fills several slots; the translator pairs list slots
   by the pattern's rules (decision 9, without removing the maps).
2. A downstream AHU pattern indexes the exhaust temperature by terminal unit.
3. This ADR and the glossary.
4. `sample_time` -> `step_size`.
5. Declarations (decisions 2-4 and 8). Systems written the old way keep
   working in `dev` until they are converted.
6. `derive` (decision 5).
7. Data sources and readers (decision 6).
8. Convert the Systems in groups: simple Systems; flows, junctions and the
   AHU, where the branch maps go; rooms and composites, where the boundary
   wall goes; controllers.
9. Deletions (the aliases, the occupancy file mode). `OccupancySystem` is
   converted rather than removed (amended 2026-11-10, the maintainer's
   decision): converted, it keeps the zone's `V`, `G_occ` and `m_inf` and
   the dampers' `a`, `nominalAirFlowRate` and `c` estimable through the
   inversion, shared with the room and the dampers.
10. Lock-in: overrides raise, the checks raise and `validate` goes, and
    `check_system` with the core examples table (decisions 1, 10 and 11).

Then downstream packages convert their own Systems.
