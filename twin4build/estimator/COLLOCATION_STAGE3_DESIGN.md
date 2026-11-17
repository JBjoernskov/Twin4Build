# Stage 3 design — `forward` + `tps.State` for fast collocation

**Status:** design draft (not implemented). **Builds on** Stages 0–2
(`_casadi_ipopt.py`, `_transcription.py`, `stateful_system.py`). **Goal:** make the
collocation constraint Jacobian come from a single `torch.func.vmap(jacrev(...))`
call instead of Stage-2's `D` backward passes plus `n_theta` finite-difference
re-simulations — by expressing each component's one-step dynamics the *native
PyTorch way* (`forward` + registered state/params) so functorch can trace it.

---

## 1. Why (the measured problem)

Stage 2 implements simultaneous transcription correctly: continuity becomes hard
equality defects `d_t = x_{t+1} − F(x_t, u_t, θ) = 0`, and IPOPT consumes the
block-bidiagonal constraint Jacobian. But **assembling that Jacobian is the
bottleneck**: state blocks need `D` reverse passes, and the θ columns are done by
`n_theta` **finite differences**, each a full `simulator.simulate` that re-pays
`model.initialize`. Measured on a representative one-step (200 segments, `D=13`,
`n_theta=19`): a single `vmap(jacrev)` over a *pure* one-step map = **194 ms** vs
`D` backward + `n_theta` FD = **1657 ms (~8.5×)** — and far more on the real model.

**The obstacle:** `torch.func` traces cleanly through pure math (`matrix_exp`,
`matmul`) but **fails through `do_step`** — the port machinery uses in-place
history writes (`Scalar._set(..., i_t=...)`) and object-attribute access (`id`)
that functorch cannot trace/`vmap`.

**The fix:** give each component a *pure* `forward` that computes one step without
touching ports, with state and parameters as native, registered tensors; compose
those `forward`s over the model graph into one differentiable map
`F(x, θ; u) → x'`; then the collocation Jacobian is `vmap(jacrev(F))`.

---

## 2. Scope

**In scope:** the smooth, torch-based component subset needed to run the
estimation example under collocation (thermal, mass, space-heater, PID, valve,
damper, and exogenous sources — outdoor environment, schedules, occupancy).

**Fallbacks (auto):** FMU components (sequential C, not `vmap`-batchable), and
genuinely discrete/hysteretic logic, are not collocation-eligible — such models
fall back to Stage-1 multiple-shooting or Stage-2 (slow-Jacobian) collocation.

**Non-goal:** matching JModelica wall-clock — torch stays interpreted. The target
is removing the *redundant-simulation* overhead so full collocation becomes
competitive with / faster than single-shooting on the pure-torch subset.

---

## 3. Two native primitives

Everything below rests on making state and dynamics first-class in the way the
framework and PyTorch already express things.

### 3.1 `tps.State` — state as a first-class type

A `System` holds three kinds of things; today only two are first-class:

| kind | type |
|---|---|
| I/O ports | `tps.Scalar` / `tps.Vector` |
| parameters | `tps.Parameter` (subclasses `nn.Parameter`) |
| **state** | **ad-hoc plain attributes** (`self.x`, `self.err_prev`, `ss_model.x`) — the gap |

Introduce **`tps.State`**, alongside `tps.Parameter` in `twin4build.utils.types`:

- **Declared in `__init__`** like ports/params: `self.x = tps.State(...)`.
- **`initialize(n_s, n_c, n_v)`** — same batch convention as ports; this is also
  what supplies collocation's per-segment `n_s = n_segments` boundary states.
- **`get()` / `set()`**.
- **Registered** as persistent module state (in `state_dict`, moves with
  `.to(device)`) — mirroring `tps.Parameter`.
- **Initial-condition hook** — the one thing ports/params don't model: a constant
  or a callable evaluated at `initialize()` time, so e.g. the thermal component
  can still seed air/wall temperatures from `T_air_start` / output ports (its
  current `_get_initial_state_tensor` logic moves into this hook).

**Design decisions (from review):**
- **Granularity:** one `tps.State` per coherent physical state vector (PID → one
  of width 3; SS core → one of width `n_states`). Cleaner layout, fewer objects.
- **Ownership of nested state:** state lives where the dynamics live —
  `DiscreteStatespaceSystem` declares `self.x = tps.State(...)`; the SS-backed
  wrappers (thermal/mass/heater) inherit it via recursive enumeration (no
  duplication). The composite `BuildingSpaceTorchSystem` gets its state = union of
  submodels' `tps.State`, for free.
- **Ordering:** decision-vector layout uses deterministic declaration / `named_*`
  order, so warm-start and reload line up.

### 3.2 `forward` — dynamics the native torch way

Each component gets a **pure** `forward` (the `nn.Module` idiom: `forward` = the
math, everything else = plumbing):

```python
def forward(self, state, inputs):        # pure, functorch-safe
    """One timestep. (state, inputs) -> (new_state, outputs).
    CONTRACT: no self/port mutation, no in-place on traced tensors, no .item(),
    no value-dependent branching; vmap-safe numerics (matmul, matrix_exp,
    smooth_saturation). Params are read from self and substituted by
    functional_call for the differentiated path.
    """
    ...
    return new_state, outputs
```

- **State is an explicit argument** (it's the per-segment quantity that varies).
- **Parameters are substituted via `torch.func.functional_call`** — because
  `tps.Parameter` subclasses `nn.Parameter`, `functional_call(c, θ_overrides,
  (state, inputs))` runs `forward` with the optimizer's (jacrev-traced) θ while
  leaving non-estimated params at their module defaults, including nested
  `thermal.C_air` via dotted names. So θ needs no manual threading.
- **`jacrev(..., has_aux=True)`** carries the `outputs` dict alongside the
  differentiated `new_state`.

`do_step` becomes a thin port shell over `forward` (§7 rollout):

```python
def do_step(self, ...):
    new_state, outputs = self(self.x.get(), gather_inputs_from_ports())  # __call__ -> forward
    self.x.set(new_state)
    for name, val in outputs.items():
        self.output[name]._set(val, i_t=step_index)
```

### 3.3 What happens to `StatefulSystem`

It dissolves. "Is stateful" = "declares any `tps.State`". `get_state`/`set_state`
= a **generic gather/scatter** over a System's `tps.State` members (recursing into
submodules). The five hand-written implementations — including the composite's
concat boilerplate — collapse into one generic walk. `StatefulSystem` reduces to,
at most, a marker check; the `StateLayout` helper keeps only the flatten/unflatten
of `tps.State` values into the decision vector.

---

## 4. The composed one-step map

`F(x_flat, θ; u_t, carry_t) → (x_next_flat, carry_next)` runs the model's
components' `forward`s in dependency order, threading outputs → inputs, reading
exogenous drivers from precomputed `u_t` and cycle-broken feedback from `carry_t`.

### 4.1 Traverse the existing graph — no precomputed plan

Because `torch.func` is **eager** (it runs your Python with wrapped tensors), the
composition can **walk the live `System`/`Connection`/`ConnectionPoint` objects
directly** inside the traced function — the traversal is topology-driven, not
value-driven, so it traces fine, and under `vmap` it runs **once at trace time**
(no per-segment cost). No frozen side structure, single source of truth.

The wiring is read from the *same* fields `_assign_component_inputs` uses:

| composer needs | existing field |
|---|---|
| a component's input ports | `component.connects_at` (→ `ConnectionPoint`s) |
| the input-port name | `connection_point.input_port` |
| connections feeding a port | `connection_point.connects_system_through` |
| the **producer** component | `connection.connects_system` |
| producer's output port | `connection.output_port` |
| vector-slot alignment | `connection_point.input_port_index[conn]`, `.output_port_index[conn]` |
| dependency order | `model.execution_order` / `_flat_execution_order` |
| cycle-broken (carried) edges | model cycle bookkeeping: `_required_initialization_connections` / `_components_no_cycles` |

### 4.2 Roots and the influence cone

`F` only needs to produce what is actually differentiated/observed:

- **State roots** = components declaring `tps.State` (their `new_state` are the
  constraint-Jacobian outputs).
- **Objective roots** = the estimator's measuring devices (`self._measurements`).

Restrict evaluation to the **reverse-reachable cone** of those roots — a backward
walk `connects_at → connects_system_through → connects_system` over the **acyclic**
graph (`_components_no_cycles`), collecting upstream components and stopping at:
exogenous sources, carried (cycle-broken) edges, or already-visited nodes.
Instrumentation that neither feeds a state nor is a measurement is never
evaluated — shrinking the trace and the batched graph.

### 4.3 Edge classification (per input, static topology)

For component `c` reading from upstream `u`:

- **exogenous** — `u` is a pure time-source (weather/schedule/occupancy) → its
  outputs are precomputed once per timestep (`u_t`), constant to `jacrev`.
- **fresh** — `pos(u) < pos(c)` in the order → use `u`'s just-computed `forward`
  output.
- **carried** — the edge was removed to break a cycle (feedback) → read `carry_t`
  (the value from the previous step). Fold carried signals into the boundary
  decision vector with their own continuity defect, so `F` stays a clean
  `(x, θ) → x'`. *(For the estimation example the carry set is effectively empty —
  the PID's `err_prev` state already holds the delayed zone temperature — but the
  mechanism is general.)*

Vector ports keep the `(input_slot, out_slot)` index mapping so multi-source
vectors assemble exactly as they do today.

### 4.4 Composition strategy (two equivalent forms)

Both are functorch-fine (eager tracing); pick per taste:

- **Forward sweep over the cone** — iterate `execution_order` restricted to the
  cone; for each component gather inputs (exogenous / fresh producer outputs /
  carry), call `forward`, keep its outputs for downstream. Trivially comparable to
  `_do_system_time_step` for the equivalence test. *(Recommended.)*
- **Recursive pull from roots** — `compute(component, port)` recurses into
  producers via the same objects, memoizing by `(component, port)` and bottoming
  out at exogenous / carried / memoized nodes. Naturally demand-driven; watch
  Python recursion depth on deep chains.

---

## 5. Collocation integration

Only the **Jacobian source** in `_solve_sparse_collocation` changes:

```python
# Stage 2 (now):   blocks via D backward passes + n_theta FD sims
# Stage 3:
def F_seg(x_seg, theta, u_seg, carry_seg):        # one segment; vmapped below
    x_next, _outs = compose_forward(model, x_seg, theta, u_seg, carry_seg)
    return x_next                                  # (aux = _outs via has_aux)

Jx = vmap(jacrev(F_seg, argnums=0))(X, theta, U, C)   # (n_seg, D, D)
Jt = vmap(jacrev(F_seg, argnums=1))(X, theta, U, C)   # (n_seg, D, n_theta)
```

- Defects `g` come from a single `vmap(F_seg)` forward (already batched).
- `solve_ipopt_constrained`, the sparsity pattern, and the `(jac_rows, jac_cols)`
  assembly are **unchanged** — they just receive `Jx`/`Jt` from `F`.
- Objective (data-fit) can reuse the same `compose_forward` outputs or keep the
  existing simulate-based objective (it is not the bottleneck).
- **Eligibility gate:** `model_supports_functional_collocation(model)` — every
  component in the cone has a `forward` and is smooth, no FMU. The estimator picks
  fast-Jacobian collocation vs Stage-2 vs Stage-1 automatically.

---

## 6. Handling the hard cases

- **Coupling / gauss-seidel** — §4.3/§4.4: functional sweep in dependency order +
  carried feedback (previous-step values). No algebraic loops, mirroring the
  simulator.
- **Discrete / non-smooth** — `forward` must be smooth; saturations use
  `smooth_saturation`. Hard clamps / mode switches / hysteresis are not
  collocation-compatible → component declares itself non-smooth → model falls back.
- **FMU** — one sequential C instance, not `vmap`-batchable → fall back.
- **Physics-violating iterates** — mid-solve states are arbitrary; keep per-state
  box bounds (`tps.State` can carry `lb`/`ub`) and ensure `forward` is NaN-safe
  over the boxed range.

---

## 7. Phased rollout (additive first, then invert)

De-risk by proving the fast path before touching the working simulator.

- **P0 — `tps.State`.** Add the type (init/get/set, registration, initial-condition
  hook). Migrate the example's stateful components to declare `tps.State` (SS core
  `self.x`, PID's three scalars → one width-3 state). Generic gather/scatter
  replaces `StatefulSystem`. Existing sims/tests must stay green (do_step still
  reads/writes via the state, just typed now). *Additive, low risk.*
- **P1 — `forward` on the SS core (thermal/mass/heater), additive.** Add `forward`
  *alongside* the existing `do_step`; validate functorch-traceability and that
  `vmap(jacrev)` gives the fast Jacobian for the *uncoupled* case (measured
  inputs, no controller = Blum et al.'s RC identification). *Highest confidence.*
- **P2 — algebraic + PID + composer.** `forward` for valve/damper and the
  velocity-form PID; the cone + gauss-seidel composition + carry set over the live
  graph. Enables the full coupled estimation example. Main risk: faithful
  execution-order / cycle-break reproduction.
- **P3 — invert `do_step` onto `forward`.** Once `forward` is proven, refactor each
  `do_step` to delegate to `forward` (deleting duplicated math); the test suites
  are the safety net. Add the eligibility gate + auto-fallback + benchmark.

Touches the *dynamics* (additive `forward`, then a delegating `do_step`) but not
the simulator's core loop; ordinary `simulate` is unaffected until P3, and even
then only via a thin, test-guarded shell.

---

## 8. Validation plan

1. **`forward` ≡ `do_step`** — each component's `forward` reproduces `do_step`'s
   next-state/outputs to ~1e-9 on random-but-physical inputs. After P3 they share
   code, so this becomes a cheap regression guard rather than a correctness proof.
2. **Composed `F` ≡ `_do_system_time_step`** — one composed step matches the
   simulator's one-step over the whole model (drives out wiring/order/carry bugs).
3. **Jacobian correctness** — `vmap(jacrev(F))` blocks vs (a) dense
   `torch.autograd.functional.jacobian` and (b) central finite differences, on a
   small model, before trusting IPOPT.
4. **End-to-end** — collocation on the estimation example reaches the same optimum
   as Stage-2, with Jacobian assembly ≥5× faster and the full solve competitive
   with / faster than single-shooting.
5. **Regression** — estimator / building_space / controller suites stay green
   through P0–P3.

---

## 9. Risks & open questions

- **`tps.State` initial conditions (#3.1).** The genuinely new modelling concern:
  how components declare/seed initial state (constant vs callable at
  `initialize()`), and how boundary decision variables override it. Get this right
  up front — it's the part ports/params don't already cover.
- **Execution-order fidelity.** The composer must reproduce `execution_order` +
  `_assign_component_inputs` + cycle-break semantics exactly; mitigated by reading
  the *same* objects and validating against `_do_system_time_step` (§8.2).
- **functorch coverage per component.** Some ops may be non-`vmap`-able (certain
  `linalg` paths, boolean indexing); audit each `forward` against `vmap` early —
  the SS core is known-good from the spike.
- **Carry-set generality.** Folding feedback into the decision vector enlarges the
  NLP; confirm it stays small (feedback edges are few) for real models.
- **Payoff ceiling.** Interpreted-torch per-step cost remains; expect "competitive
  with / faster than single-shooting on the pure-torch subset," not JModelica
  seconds. That is the honest, defensible framing.
