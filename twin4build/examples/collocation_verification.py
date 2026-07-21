"""Verify whether the collocation-vs-SLSQP gap is implementation or objective.

Three controlled experiments on the full estimation example (19 params,
4 sensors), by default on a 24 h window:

1. **Feasibility audit (baseline)** -- run collocation exactly as the benchmark
   did (default IPOPT tolerances, data-informed warm start ON, ``n_warmup=0``)
   and audit the solution: max defect violation, active box bounds, and a
   per-sensor comparison of the NLP's internal fit vs. an honest *sequential*
   ``F_aug`` rollout from the estimated initial state.  A gap between the two
   fits means the solution leans on defect slack (``Solved_To_Acceptable_Level``
   allows constraint violations up to 1e-2 per normalized state per step).

2. **Tight tolerances** -- same problem with ``constr_viol_tol=1e-9``,
   ``acceptable_iter=0`` (acceptable-level termination disabled) and the
   data-informed warm start off.  If the baseline's gap closes and the rollout
   RMSE improves, the slack was the problem.

3. **Equivalence / stationarity test** -- run SLSQP first (``n_warmup=20``),
   then warm-start collocation at SLSQP's optimum with the initial augmented
   state PINNED (bound equality) and the SAME warmup mask.  With a pinned
   initial condition and tight defects the feasible set is exactly the
   single-shooting trajectory manifold and the objective matches, so SLSQP's
   optimum must be (near-)stationary.  If IPOPT still moves theta materially,
   there is a real implementation bug.

Usage::

    python -m twin4build.examples.collocation_verification --hours 24
    python -m twin4build.examples.collocation_verification --experiments 1,2

Results (RMSE table + audits) are printed; per-run plots go to ``--outdir``.
"""

import argparse
import datetime
import os
import time

import matplotlib

matplotlib.use("Agg")
import numpy as np

import twin4build as tb
from twin4build.examples.collocation_comparison import (
    EXAMPLE_END,
    EXAMPLE_START,
    STEP_SIZE,
    example_measurements,
    example_parameters,
    load_model,
    apply_estimated,
)

TIGHT_OPTIONS = {
    "ipopt.constr_viol_tol": 1e-9,
    "ipopt.acceptable_iter": 0,  # disable Solved_To_Acceptable_Level termination
    "ipopt.tol": 1e-8,
}

WARMUP_SKIP = 20  # rollout-evaluation transient skip (matches the benchmark)


def _norm_theta(params, values):
    """Linear [0,1] normalization of physical values against the spec bounds
    (diagnostic only -- ignores log scaling, adequate for drift reporting)."""
    out = []
    for (_, _, _x0, lo, hi), v in zip(params, values):
        out.append((float(v) - lo) / (hi - lo))
    return np.asarray(out, dtype=np.float64)


def _rollout_temp_rmse(model, simulator, start, end):
    """Continuous rollout from the default init; temperature RMSE after skip."""
    model.set_save_simulation_result(flag=True)
    simulator.simulate(
        start_time=start, end_time=end, step_size=STEP_SIZE, show_progress_bar=False
    )
    space = model.components["office"]
    pred = space.output["indoorTemperature"].history(i_s=0, i_c=0).detach().numpy()
    actual = np.asarray(
        model.components["office_temperature_sensor"].time_series_input.values
    ).flatten()
    n = min(len(pred), len(actual))
    s = min(WARMUP_SKIP, max(0, n - 1))
    return float(np.sqrt(np.mean((pred[s:n] - actual[s:n]) ** 2)))


def run_estimation(method, options, periods, n_warmup, x0_override=None,
                   data_warmstart=True):
    """One estimation run on a fresh model; returns (result, temp_rmse, secs)."""
    prev = os.environ.pop("TWIN4BUILD_NO_DATA_WARMSTART", None)
    if not data_warmstart:
        os.environ["TWIN4BUILD_NO_DATA_WARMSTART"] = "1"
    try:
        model = load_model()
        params = example_parameters(model)
        if x0_override is not None:
            # SLSQP may leave parameters exactly on a bound; nudge inside the
            # open interval so the estimator's x0 > lb / x0 < ub checks pass.
            params = [
                (comp, attr,
                 min(max(float(v), lo + 1e-9 * (hi - lo)), hi - 1e-9 * (hi - lo)),
                 lo, hi)
                for (comp, attr, _x0, lo, hi), v in zip(params, x0_override)
            ]
        measurements = example_measurements(model)
        est = tb.Estimator(tb.Simulator(model))
        t0 = time.time()
        result = est.estimate(
            parameters=params,
            measurements=measurements,
            start_time=[s for s, _ in periods],
            end_time=[e for _, e in periods],
            step_size=STEP_SIZE,
            n_warmup=n_warmup,
            method=method,
            options=options,
        )
        elapsed = time.time() - t0
        apply_estimated(model, params, result.get("result_x", []))
        rmse = _rollout_temp_rmse(model, est.simulator, periods[0][0], periods[0][1])
        return result, rmse, elapsed
    finally:
        os.environ.pop("TWIN4BUILD_NO_DATA_WARMSTART", None)
        if prev is not None:
            os.environ["TWIN4BUILD_NO_DATA_WARMSTART"] = prev


def _print_audit(result):
    audit = result.get("transcription_audit")
    if not audit:
        print("    (no transcription audit attached)")
        return
    print(f"    status={audit['return_status']}  max|defect|={audit['max_defect']:.3e}  "
          f"state vars at box bound: {audit['n_active_state_bounds']}/{audit['n_free_state_vars']}  "
          f"theta at bound: {audit['n_theta_at_bounds']}")
    for mid, e in audit["per_sensor"].items():
        gap = e["rollout_rmse"] - e["nlp_rmse"]
        ds = e.get("do_step_rmse", float("nan"))
        print(f"    {mid:<34} NLP={e['nlp_rmse']:.4f}  F_aug rollout={e['rollout_rmse']:.4f}  "
              f"(gap={gap:+.4f})  do_step rollout={ds:.4f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hours", type=int, default=24,
                    help="window length (0 = the example's two full periods)")
    ap.add_argument("--maxiter", type=int, default=300)
    ap.add_argument("--slsqp-maxiter", type=int, default=100)
    ap.add_argument("--experiments", default="1,2,3")
    args = ap.parse_args()

    if args.hours > 0:
        periods = [(EXAMPLE_START[0], EXAMPLE_START[0] + datetime.timedelta(hours=args.hours))]
    else:
        periods = list(zip(EXAMPLE_START, EXAMPLE_END))
    which = {e.strip() for e in args.experiments.split(",")}
    colloc = ("casadi", "ipopt", "ad", "collocation")
    summary = []

    # ---- Experiment 1: baseline audit --------------------------------------
    if "1" in which:
        print("\n=== Experiment 1: baseline collocation (default tol, data warm start ON, n_warmup=0)")
        res1, rmse1, t1 = run_estimation(
            colloc, {"maxiter": args.maxiter}, periods, n_warmup=0,
            data_warmstart=True,
        )
        print(f"    rollout temp RMSE = {rmse1:.4f} K   ({t1:.0f} s)")
        _print_audit(res1)
        summary.append(("1 baseline collocation", rmse1, t1))

    # ---- Experiment 2: tight tolerances ------------------------------------
    if "2" in which:
        print("\n=== Experiment 2: tight tolerances (constr_viol_tol=1e-9, acceptable_iter=0, no data warm start)")
        res2, rmse2, t2 = run_estimation(
            colloc, {"maxiter": args.maxiter, **TIGHT_OPTIONS}, periods, n_warmup=0,
            data_warmstart=False,
        )
        print(f"    rollout temp RMSE = {rmse2:.4f} K   ({t2:.0f} s)")
        _print_audit(res2)
        summary.append(("2 tight-tol collocation", rmse2, t2))

    # ---- Experiment 3: pinned-init stationarity test ------------------------
    if "3" in which:
        print("\n=== Experiment 3: SLSQP optimum -> pinned-init collocation (matched n_warmup=20)")
        res_s, rmse_s, t_s = run_estimation(
            ("scipy", "SLSQP", "ad"), {"maxiter": args.slsqp_maxiter}, periods,
            n_warmup=WARMUP_SKIP,
        )
        theta_slsqp = np.asarray(res_s["result_x"], dtype=np.float64)
        print(f"    SLSQP rollout temp RMSE = {rmse_s:.4f} K   ({t_s:.0f} s)")
        summary.append(("3a SLSQP reference", rmse_s, t_s))

        res3, rmse3, t3 = run_estimation(
            colloc,
            {"maxiter": args.maxiter, "pin_initial_state": True, **TIGHT_OPTIONS},
            periods, n_warmup=WARMUP_SKIP,
            x0_override=theta_slsqp, data_warmstart=False,
        )
        theta_colloc = np.asarray(res3["result_x"], dtype=np.float64)
        print(f"    pinned collocation rollout temp RMSE = {rmse3:.4f} K   ({t3:.0f} s)")
        _print_audit(res3)
        summary.append(("3b pinned collocation @SLSQP", rmse3, t3))

        # Stationarity: how far did theta move from the SLSQP optimum?
        model_ref = load_model()
        params_ref = example_parameters(model_ref)
        dn = _norm_theta(params_ref, theta_colloc) - _norm_theta(params_ref, theta_slsqp)
        print(f"\n    theta drift (normalized): max|d|={np.abs(dn).max():.4f}  "
              f"L2={np.linalg.norm(dn):.4f}")
        order = np.argsort(-np.abs(dn))
        for i in order[:5]:
            comp, attr, *_ = params_ref[i]
            print(f"      {comp.id}.{attr:<28} {theta_slsqp[i]:.5g} -> {theta_colloc[i]:.5g} "
                  f"(dnorm={dn[i]:+.4f})")

    print(f"\n{'experiment':<36}{'temp RMSE [K]':>14}{'time [s]':>10}")
    print("-" * 60)
    for tag, rmse, secs in summary:
        print(f"{tag:<36}{rmse:>14.4f}{secs:>10.0f}")


if __name__ == "__main__":
    main()
