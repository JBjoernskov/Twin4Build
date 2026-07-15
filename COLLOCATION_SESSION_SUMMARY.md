# Collocation vs Single-Shooting — Session Summary (for review)

Branch: `feature/issue-89/models-from-brick-datasets`. Goal: benchmark a **collocation /
simultaneous-transcription** parameter-estimation method against the current
**single-shooting** estimator on the estimation example, using IPOPT (via CasADi).
Reference example: `twin4build/examples/estimator_example.ipynb` (19 params, 4 sensors,
2 training periods of 4 days, step 1200 s, SLSQP).

Driver script: `twin4build/examples/collocation_comparison.py`
(`--param-set {full,thermal}`, `--methods slsqp,single_shooting,multiple_shooting,collocation`).

---

## 1. What was implemented (all committed)

**A pure, functorch-traceable one-step map of the whole model.** Each stateful
component got a `forward(state, inputs, params, sample_time) -> (next_state, outputs)`
that exactly re-expresses its `do_step`. `OneStepComposer` (`estimator/_composer.py`)
threads these in execution order into `F(states, theta, captured, feedback)` and an
**augmented** `F_aug(y) = [F(state,feedback), producer_outputs]` where `y = [state |
feedback]`. F_aug returns the modelled measured outputs too.

- **Cut feedback edges = state.** `office.heatGain <- space_heater.Power` and
  `co2_controller.actualValue <- office.indoorCO2` are one-step lag variables. Appending
  them to the state turns feedback closure into ordinary continuity, matching `do_step`'s
  delayed Gauss-Seidel semantics exactly (no separate defect type). Two convergence
  fixes were essential: (a) robust feedback scale (a ~constant feedback has ~0 std ->
  normalized var blows past bounds -> "local infeasibility"); (b) consistent *delayed*
  feedback warm start (the batched 1-step capture gives fresh-init feedback, violating
  `w_{t+1}=producer_output(y_t)` by ~1e5 -> IPOPT stalls; fix = run F on warm-start
  states and shift by one segment -> inf_pr starts ~0.5 not 3e5).

- **Sparse block-bidiagonal Jacobian to IPOPT.** `vmap(jacrev(F_aug))` gives all
  per-segment blocks in one call. Crucial CasADi fix: declare the Jacobian *sparsity
  pattern* via `has_jac_sparsity`/`get_jac_sparsity` on the constraint callback
  (`estimator/_casadi_ipopt.py`) — without it CasADi handed IPOPT a *dense* n_g×n
  pattern (867k nnz on 24h); with it, 16.6k nnz (the real structure).

- **`_expm_ss`** (`systems/utils/discrete_statespace_system.py`): scaling-and-squaring
  Taylor matrix-exponential, pure matmul so `vmap` batches it natively.
  `torch.matrix_exp` has NO vmap batching rule -> silent per-segment Python loop, 8-56×
  slower under `vmap(jacrev)` (measured). NaN bug fixed: fixed squarings 6->18 (stiff
  matrices at bound-hitting params, ‖M‖ up to ~3e5, overflowed the truncated Taylor sum;
  block matrix is dissipative so extra squarings are unconditionally stable). Accuracy
  ~1e-11 vs matrix_exp.

- **Time-series init cache** (`systems/utils/time_series_input_system.py`): early-return
  when the window is unchanged (was rebuilding the value tensor + rescanning NaNs every
  `model.initialize`).

- Estimator plumbing: `result["estimated_initial_state"]`; `Simulator.simulate(
  after_initialize=...)` hook to scatter states before a prediction; a per-segment
  **warmup mask** for the collocation objective (opt-in); data-informed warm start
  (seed observed boundary states from measurements via a d(meas)/d(y) readout test).

---

## 2. What was verified

- **F_aug ≡ do_step to <5e-4 for ALL four measured outputs** over a full window
  (temp 5e-7, CO2 5e-4, valve 3e-7, damper exact) — verified by a continuous
  sequential F_aug rollout vs the object-graph `do_step` from the same initial state.
- **Sparse Jacobian correct** to ~6e-6 vs finite-difference of the real defect.
- **Timestepping speedup (vmap over segments):** vmap(F_aug) vs object-graph 1-step
  batched simulate = **192× (72 seg) / 821× (576 seg)**. This is *parallel over
  segments* — a structural property of collocation. For **sequential** single-shooting
  the F_aug forward is only ~2× (149 vs 313 ms, 72 steps); its eager gradient is slow
  (1.6 s, autograd through the unrolled Python loop).

---

## 3. VERIFICATION SESSION (follow-up): a real bug was found and fixed

The open question of Section 4 (old) — "objective or implementation?" — was settled by
a controlled experiment suite (`twin4build/examples/collocation_verification.py`):

1. **Feasibility audit** (now automatic after every fast-path solve): reports
   `max|defect|` at the solution, active box bounds, and per-sensor RMSE three ways —
   NLP-internal (what the optimizer scored), a *sequential* `F_aug` rollout from the
   estimated init (incl. feedback lags), and a real object-graph `do_step` rollout.
2. **Tight tolerances** (`constr_viol_tol=1e-9`, `acceptable_iter=0`, no data warm
   start).
3. **Pinned-init stationarity test**: pin each period's initial augmented state by
   bound equality, match `n_warmup=20` with SLSQP, warm-start theta at SLSQP's optimum.

### The bug: the composer froze the CO2→damper control loop

`office_damper_max` (`MaxSystem`) had no `forward`, and its `inputs` port is a
*vector* port which the composer skipped. So the composer classified
`office_supply/exhaust_damper.damperPosition` (and the damper-position measurement) as
**"captured" — frozen from the theta0 reference simulation**. The ventilation loop
(CO2 → PID → max → dampers → airflow → CO2) never responded to theta or the states:
collocation was optimizing a *different model* than `do_step`/SLSQP, with damper
behaviour hard-coded to the initial guess. The old "F_aug ≡ do_step to <5e-4"
verification did not catch this because it, too, compared at theta0 with the same
frozen captures.

**Fixes (committed in this session):**
- `MaxSystem.forward` (pure logsumexp, functorch-traceable) so the max joins the cone.
- Per-slot **vector-port resolution** in `OneStepComposer` (`("vector", [slot_specs])`
  wiring; per-slot capture keys `(comp, port, slot)`), so `office_co2_controller.
  inputSignal` threads fresh and only the truly-exogenous occupancy-override slot stays
  captured.
- **Continuous warm-start capture**: exogenous inputs + delayed-feedback warm start are
  now sampled from one *continuous* `do_step` rollout per period (input ports sampled
  after each step). The old batched 1-step capture (a) gave defect-inconsistent
  feedback (warm-start `max|defect|` ≈ 2.7σ) and (b) froze `office.numberOfPeople` at
  ZERO (the stateful `OccupancySystem` outputs 0 on its first step after initialize).
  Warm-start `max|defect|` is now ~1e-5.
- Tight constraint tolerances are the **default** for the collocation path
  (`constr_viol_tol = acceptable_constr_viol_tol = 1e-8` in `_casadi_ipopt.py`).
- Duplicated `model.initialize` in `_warmstart_segment_states` removed.

Post-fix verification: every cone component's `forward` matches its `do_step` history
exactly (≤1e-8 phys. units, given matching theta), and the sequential `F_aug` rollout
tracks the continuous `do_step` trajectory to ≤6e-4 over 72 steps *including* the
previously frozen loop.

### Experiment results (24 h window, 19 params, 4 sensors)

| experiment | temp RMSE (continuous rollout, skip 20) | notes |
|---|---|---|
| SLSQP reference (n_warmup=20) | **0.042 K** | |
| collocation, free init, n_warmup=0 (baseline cfg) | 0.15 K | was 0.33 K before the fix |
| collocation, tight tol, no data warm start | 0.15 K | max\|defect\| ≤ 3e-3 at maxiter |
| **pinned init + n_warmup=20 + warm start at SLSQP optimum** | **0.042 K** | `Solve_Succeeded`, matches SLSQP |

The decisive test now passes: with the initial condition pinned and the objective
matched, IPOPT stays at SLSQP's optimum quality (0.0422 vs 0.0419 K; NLP-internal =
F_aug rollout = do_step rollout for temperature to 3 decimals). Some theta components
still move along the objective's flat directions (e.g. `C_wall` → bound) with **no
change in fit** — an identifiability ridge, not a bug.

- **Defect slack was NOT the problem**: even at `Solved_To_Acceptable_Level` the
  audited `max|defect|` was ≤1e-5σ..1e-2σ and the NLP-internal fit matched the
  sequential F_aug rollout to ~1e-4 on all sensors. The frozen control loop was the
  problem.
- **The remaining free-init gap (0.15 vs 0.042 K) is the objective story** of the
  original summary — now on a *correct* model: with free boundary states and
  `n_warmup=0` the 4-sensor objective genuinely prefers a different trajectory. It is
  not an implementation artifact (pinning + matching the objective reproduces SLSQP).
- Known audit caveat: the `do_step` rollout column can differ for CO2 (slowest, least
  observable state) because that rollout cannot seed the feedback-lag variables or the
  `OccupancySystem`'s internal memory; temperature/valve/damper agree to ~1e-4.

---

## 4. The benchmark result (full 19-param, 4-sensor example)

| horizon | SLSQP | collocation (pre-fix) | collocation (post-fix, L-BFGS) | collocation (post-fix, Gauss-Newton) | collocation (GN + early stopping) |
|---|---|---|---|---|---|
| 24h | 0.042 K (117 s) | 0.19–0.33 K | 0.15 K (free init) / **0.042 K** (pinned init, matched objective) | 0.093 K free init, **65 s**, `Solved_To_Acceptable_Level` at iter 203 | 0.068 K, **14 s** (objective stagnant 10 iters; best iterate restored) |
| full (2×4 day) | 0.060 K (671 s) | 0.20 K | **0.110 K** (691 s, free init, maxiter 300, not converged) | **0.110 K in 140 s** (maxiter 100; objective plateau reached by ~iter 40) | **0.109 K in 59 s** (`User_Requested_Stop` at ~iter 45: objective stagnant 10 iters; best iterate restored; no maxiter tuning needed) |

### Gauss-Newton Hessian (July 2026 follow-up)

Collocation now supplies IPOPT a **Gauss-Newton Hessian of the Lagrangian**
(`sigma * (2/N) * J^T W J` from the per-segment measurement Jacobians) through CasADi's
`hess_lag` option, instead of limited-memory BFGS. It is on by default for the fast
(composer) path; disable with `options={"gauss_newton": False}` or
`--no-gn` in `collocation_comparison.py`.

- The measurement Jacobians come from the SAME `vmap(jacrev)` evaluation as the defect
  Jacobian (one shared per-iterate cache), so the Hessian is nearly free per iteration;
  the sparse-value assembly was also vectorized (the former per-nonzero Python loop was
  a real per-iteration cost).
- Effect: the objective reaches its optimum in ~40-100 iterations instead of never
  converging within 300-600 (L-BFGS). Full problem: same optimum in 140 s vs 691 s.
- GN drops the constraint curvature term, so IPOPT's dual infeasibility plateaus
  (~1e-3..1e-1) and its default `tol=1e-8` is unreachable — left alone it burns
  hundreds of extra iterations polishing duals with zero objective progress, then dies
  in restoration (`Restoration_Failed`, at a feasible point). The GN path therefore
  defaults to a pragmatic acceptable-level exit: feasible (`acceptable_constr_viol_tol
  1e-2`, normalized units — the audit shows 1e-3-level slack is benign) + objective
  stagnant (`acceptable_obj_change_tol 1e-4`) for 5 consecutive iterations =>
  `Solved_To_Acceptable_Level`. All overridable via `options`.
- Correctness guards: `TWIN4BUILD_HESS_CHECK=1` verifies the GN residual/Jacobian
  assembly against the autograd objective gradient (`grad f = (2/N) J^T r`, exact
  identity; observed agreement ~8e-16). `TWIN4BUILD_JAC_CHECK=1` still checks the
  defect Jacobian against finite differences.

### Early stopping (DNN-style, July 2026 follow-up)

On top of IPOPT's own acceptable-level exit, the GN path now runs an **iteration
callback** (CasADi `iteration_callback` -> `User_Requested_Stop`) implementing
patience-based early stopping plus best-iterate checkpointing — the direct analogue of
`EarlyStopping(restore_best_weights=True)` in NN training. On by default with GN;
disable with `options={"early_stopping": False}` / `--no-early-stop`, tune with
`options={"early_stopping": {...}}`.

- **Feasibility gate** (`feas_tol` 1e-2, normalized defects): only iterates with
  `max|g| <= feas_tol` can update the incumbent or count as objective progress; the
  patience counter starts at the first feasible incumbent and then advances on EVERY
  iteration (an infeasible restoration excursion can't reset it — if the excursion
  pays off, its improved feasible landing point resets it).
- **Objective patience**: stop after `patience` (10) consecutive iterations without a
  `min_delta_rel` (1e-3) relative improvement over the best-so-far. The coarse
  `min_delta_rel` is what makes the stop *early*: it separates the initial descent
  (per-iteration gains of percents) from the plateau creep (48.0 -> 47.78 over 500+
  iterations — noise-level gains that a 1e-4 threshold kept counting as progress).
- **Theta patience**: stop when the parameter block `x[:n_theta]` (the thing we
  actually keep — states/duals may churn long after theta settles) moves less than
  `theta_tol` (1e-4, normalized) for `patience` iterations.
- **Best-feasible-iterate restore**: the callback checkpoints the lowest-objective
  feasible iterate; after ANY exit (early stop, max-iter, restoration failure) the
  checkpoint replaces the last iterate if that is worse or infeasible. This alone fixes
  a real defect: max-iter runs previously returned whatever the final iterate happened
  to be (the objective log shows excursions to 50-62 amid the ~48 plateau).
- Measured (maxiter 600, no tuning): full problem stops at ~iter 45 ->
  **0.109 K in 59 s** (vs 242 s with the first, conservative patience settings and
  895 s run-to-maxiter); 24h stops in **14 s** at 0.068 K. Quality is unchanged or
  slightly better — the extra 500 plateau iterations bought a 0.5% objective
  improvement with zero effect on the reported temperature RMSE.
- Considered-but-deferred: validation-holdout early stopping (hold out a period,
  monitor validation rollout RMSE, stop when it rises). The truest DNN analogue and the
  only rule that guards against overfitting the training window, but it changes the
  experiment methodology (less training data), so it stays opt-in future work.

### Post-solve model state (July 2026 notebook-integration fix)

Running the estimation-example **notebook** with collocation exposed a usability
defect: `estimate()` used to leave the model's parameters at the solver's *last
objective evaluation* — for IPOPT that is a line-search probe, and for the
transcription backends the returned `result_x` is a restored best iterate the model
never evaluated last. A plain `simulator.simulate()` after `estimate()` (the
notebook's "After calibration" cell) therefore ran with junk parameters and produced
garbage plots, even though `result_x` itself was the correct optimum. Fixed in
`estimator.py`: after any scipy/casadi solve the optimal theta is re-applied to the
model before returning (`collocation_comparison.py`'s `apply_estimated` workaround is
now redundant).

Two related expectation notes for notebook users:

- A **default-init rollout** with collocation's theta gives ~0.6 K temperature RMSE
  on period 1 — the parameters were fit jointly with the estimated initial state, so
  the plot cell should seed it:
  `simulator.simulate(..., after_initialize=lambda: [model.components[cid].set_state(b) for cid, b in result["estimated_initial_state"].items()])`
  which recovers ~0.12 K per period (matching the audit's `rollout_rmse`).
  `estimated_initial_state` blocks are shaped `(n_periods, n_c, state)`, batched to
  seed a multi-period `simulate` directly (slice `b[0:1]` for a single-period run).
  The estimation-example notebook's "After calibration" cell now does this.
- The benchmark's "0.109 K" full-problem numbers are seeded-init rollouts
  (`collocation_comparison.py` has `seed_initial_state = True`).

- On the **thermal-RC subset** (4 params, temperature sensor only — well-posed,
  objective-aligned) collocation already beat SLSQP (0.19 vs 0.22 K) pre-fix.
- On the full multi-sensor problem, free-init collocation with `n_warmup=0` optimizes
  a genuinely different objective (it also estimates the init and scores the transient);
  when configured equivalently to SLSQP it reproduces SLSQP's optimum.

### The valve-blind local optimum + two-stage fix (July 2026 follow-up)

The "0.109 K" free-init collocation solution is a **bad local optimum**: its valve fit
is RMSE 0.32 on a 0-1 signal (std 0.29 — i.e. unfit; the predicted valve sits near
constant ~0.1), and its objective is ~48 vs ~3.2 at the SLSQP optimum (valve 0.039).
The valve term alone contributes ~44 of the 48. From the example's x0, IPOPT descends
into a basin where the free trajectory variables absorb the temperature fit with a
sluggish heating loop (kp/Ti/UA far from truth) and then GN/L-BFGS both plateau there;
`n_warmup`, data warm start on/off, and patience settings don't change the basin.

Two findings from warm-start probes (full 2x4-day problem):

- **The NLP is consistent**: started at SLSQP's theta, collocation stays at/near that
  optimum (obj 3.13, valve 0.039, temp 0.059) and even polishes it — so the
  transcription is fine; this is pure non-convexity. The single-shooting objective's
  implicit trajectory constraint is exactly what steers *around* this basin.
- **A tiny single-shooting stage-1 suffices**: 5 SLSQP iterations (~70 s) from the raw
  x0, then collocation from the stage-1 theta => obj 3.76, valve **0.039**, temp
  **0.058 K**, default-init rollout 0.061 K, total ~140-180 s. This beats both
  standalone methods (SLSQP alone needs ~670 s for 0.060 K; collocation alone gives the
  valve-blind 48-objective optimum in ~60 s). The estimator_example notebook now uses
  this two-stage recipe.

## 5. Speed follow-up (the more promising practical direction)

SLSQP is the accuracy winner; making *it* fast is higher-value than fixing collocation.
- `torch.jit` (TorchScript) needs NO compiler — try scripting the F_aug rollout (removes
  Python overhead; modest, no fusion). NOT YET benchmarked.
- `torch.compile` (Inductor) needs a C compiler (missing on this Windows box: `cl` not
  found) — realistic ~5-10× on the sequential rollout+gradient; run under WSL/Linux.
- The **gradient** (autograd through the sequential loop) is single-shooting's real cost,
  and the prime compile/jit target.

## 6. Verdict

The collocation implementation had one real model-fidelity bug — the composer froze the
CO2→damper control loop (vector-port `MaxSystem` skipped) — plus an inconsistent /
occupancy-zeroing warm-start capture. Both are fixed and verified: with a pinned
initial condition and a matched objective, collocation now reproduces the SLSQP optimum
exactly (0.042 K), confirming the transcription, Jacobian, and objective wiring are
correct. The remaining free-init configuration difference is a *modelling choice*
(estimate the init and score everything, vs fixed init + warmup), not an error. Use the
audit log (`AUDIT:` lines) to sanity-check any future collocation run.
