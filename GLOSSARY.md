# Glossary

Domain terms for twin4build, one entry per term. Started during the 2.0
architecture review (2026-10-29). Terms listed under _Avoid_ name the same idea
and should not be used for it.

**forward**
A System's one-step map, `forward(x, inputs, p, step_size)`: from its state,
inputs, parameters and the step size to its next state and outputs. The only place a System's stepping logic or
physics lives. Object-mode simulation and the functional engine both step a
System through it. It gets no clock or date: anything that depends on the time
reaches it through an input port. Every tensor it receives has the same form:
the first dimension runs over the copies being stepped (see batched System),
then the port's or state's own size.
_Avoid_: stepping logic in `do_step`.

**Derived values**
Values that depend only on a System's parameters and the step size, such as
a PID's discrete coefficients or a state-space System's discretised matrices.
The System computes them in `derive(p, step_size)`; the caller evaluates it once per
run (object mode) or once per rollout (functional engine) and hands the results
to `forward` in `p`. `forward` caches nothing.
_Avoid_: step constants, transform mode, identity caches.

**Batched System**
One System that steps several copies of the same class at once: Systems with
the same class, ports and parameters are merged at load, and each parameter
holds one value per copy. The number of copies is `n_c`. In object mode the
first dimension `forward` sees is copies times simulations; in the functional
engine it is the copies of one simulation, the engine's `vmap` handling the
rest.
_Avoid_: meta component, parallel components.

**Composite System**
A System whose `forward` steps child Systems it declares with `Child(...)`,
routing inputs, parameters and state to them by hand. Its ports are its own
declarations, like any System's, never borrowed from its children. A child's
parameters keep their dotted path (`space.thermal.C_air`).
_Avoid_: owned sub-system, sub-model (for a child).

**Setting**
A constructor value of a System that is never estimated: a size, a modelling
choice, a constant. Declared with `Setting(...)`, given by keyword, saved with
the model. Changing a setting after construction marks the model for reload,
like editing a connection; changing a parameter does not.
_Avoid_: config, kwargs.

**Simulations side by side**
Several periods simulated in one call, one per start time (`n_s`). They share
parameters and the sample time.

**Data source**
A System whose outputs are a series known before the run, not computed from
inputs: outdoor environment, schedules, time-series inputs, the time-of-day
basis of an occupancy profile, and the recorded series of a sensor. It defines
`series(timestamps, step_size)`, which the base calls at run start with the
run's `(n_s, n_t)` timestamps and step size and which returns each output's
values for the whole run; nothing steps it. Measured data that another System needs reaches it through a connection
from a data source. A System never reads files for its own inputs.
_Avoid_: self-fed input, publisher, exogenous producer.

**Reader**
The object in a data source's `source` setting that fetches the raw series: a
CSV file, a DataFrame or a database. Its class says where the data comes from;
options about reading (resampling, gaps, fill) live on it. A file reader saves
its path relative to the saved model.
_Avoid_: loader (Load is the once-per-model step), `use_spreadsheet` /
`use_database` / `use_df` switches.

**Recorded data, modelled value**
A sensor's two outputs. `measuredData` is the recorded series; `measuredValue`
is what the model says the sensor reads. A System that needs the measurement
connects to `measuredData`.

**Load**
The once-per-model step, repeated after any edit to the components,
connections or settings: execution order, fusion, batching, and sizing each
vector port from its declared `size=` and its connections, checked against
the ports and settings it refers to. Owned by the base; no System looks at its
own connections.

**Run start**
The step at the beginning of every simulation: time span, batch size, port
allocation, parameter expansion, initial states, cache resets. Owned by the
base System; no System adds to it. The base also evaluates each System's
derived values here (object mode) and fills each data source's outputs from its
`series`.
_Avoid_: "initialize" for either step without saying which.

**Occupancy profile**
Occupancy as a forward model: the number of people in a room is a time-of-day
basis (a data source) weighted by estimated knot values. The room's CO2 sensor
scores the fit rather than being inverted for people. The inverse occupancy
model is gone in 2.0; its inversion survives only as a helper for start values.

**Conformance check**
`twin4build.testing.check_system(system, inputs)`: the test every System
passes, in core and in other packages. It checks the save/reload round trip,
deepcopy independence, agreement between object mode and the functional
engine, batched copies against separate copies, finite parameter gradients,
and a CUDA run when a GPU is present. Example inputs live in the tests, not
in the System class.

**Step size**
The time between two steps of a run, in seconds: `simulate(step_size=...)`.
One value per run, the same for every simulation in a batch. `forward`,
`derive` and `series` all receive it as `step_size`.
_Avoid_: sample time, `sample_time`, `dt`.
