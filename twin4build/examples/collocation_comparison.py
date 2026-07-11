"""Reproduce the estimation-example calibration under single-shooting vs. collocation.

Runs the *full* parameter-estimation example -- all 19 parameters, all four
measurement sensors, and the example's two training periods -- through both a
single-shooting and a multiple-shooting/collocation estimation.  The two use the
identical ``Estimator`` API; only the ``method`` tuple differs (the 4th, optional
tuple element selects the transcription).  A timing/RMSE table is printed, and
for each method the estimated parameters are re-applied, the first period is
simulated, and the estimation example's "After calibration" plot is saved -- plus
an overlay of each method's predicted indoor temperature vs. the measurement.

IPOPT is provided by CasADi (``pip install casadi``).

Usage::

    # Full example (2 periods, 19 parameters) -- slow, faithful to the notebook:
    python -m twin4build.examples.collocation_comparison --methods single_shooting,multiple_shooting

    # Quick single-window run for iteration:
    python -m twin4build.examples.collocation_comparison --hours 24 --segments 6 --maxiter 30

Figures are written as PNGs to ``--outdir`` (default ``collocation_plots``).
"""

import argparse
import datetime
import os
import time

import matplotlib

matplotlib.use("Agg")  # headless: save figures to files
import matplotlib.pyplot as plt
import numpy as np
from dateutil import tz

import twin4build as tb
import twin4build.examples.utils as utils
from twin4build.utils.rgetattr import rgetattr

STEP_SIZE = 1200  # 20 min
_TZ = tz.gettz("Europe/Copenhagen")

# The estimation example's two training periods.
EXAMPLE_START = [
    datetime.datetime(2023, 11, 27, 0, 0, 0, tzinfo=_TZ),
    datetime.datetime(2023, 12, 2, 0, 0, 0, tzinfo=_TZ),
]
EXAMPLE_END = [
    datetime.datetime(2023, 12, 1, 0, 0, 0, tzinfo=_TZ),
    datetime.datetime(2023, 12, 6, 0, 0, 0, tzinfo=_TZ),
]

METHODS = {
    "single_shooting": ("IPOPT single-shooting", ("casadi", "ipopt", "ad")),
    "multiple_shooting": ("IPOPT multiple-shooting", ("casadi", "ipopt", "ad", "multiple_shooting")),
    "collocation": ("IPOPT collocation", ("casadi", "ipopt", "ad", "collocation")),
    "slsqp": ("SLSQP single-shooting", ("scipy", "SLSQP", "ad")),
}


def load_model():
    """Load the estimation-example model and point sensors at the bundled CSVs."""
    model = tb.Model(id="collocation_comparison")
    model.load(
        simulation_model_filename=utils.get_path(["estimator_example", "instance_graph.ttl"])
    )
    c = model.components
    p = lambda name: utils.get_path(["estimator_example", name])
    c["office_temperature_sensor"].filename = p("temperature_sensor.csv")
    c["office_co2_sensor"].filename = p("co2_sensor.csv")
    c["office_valve_position_sensor"].filename = p("valve_position_sensor.csv")
    c["office_damper_position_sensor"].filename = p("damper_position_sensor.csv")
    c["supply_air_temperature_sensor"].filename = p("supply_air_temperature.csv")
    c["office_temperature_heating_setpoint"].filename = p("temperature_heating_setpoint.csv")
    c["outdoor_environment"].filename_outdoorTemperature = p("outdoor_environment.csv")
    c["outdoor_environment"].filename_globalIrradiation = p("outdoor_environment.csv")
    c["outdoor_environment"].filename_outdoorCo2Concentration = p("outdoor_environment.csv")
    c["office_occupancy"].co2_filename = p("co2_sensor.csv")
    c["office_occupancy"].damper_filename = p("damper_position_sensor.csv")
    return model


def example_parameters(model):
    """The full 19-parameter set from the estimation example."""
    c = model.components
    space = c["office"]
    space_heater = c["office_space_heater"]
    heating_controller = c["office_temperature_heating_controller"]
    valve = c["office_space_heater_valve"]
    supply_damper = c["office_supply_damper"]
    exhaust_damper = c["office_exhaust_damper"]
    return [
        (space, "thermal.C_air", 2e6, 1e6, 1e7),
        (space, "thermal.C_wall", 2e6, 1e6, 1e7),
        (space, "thermal.C_boundary", 5e5, 1e4, 1e6),
        (space, "thermal.R_out", 0.01, 1e-3, 0.1),
        (space, "thermal.R_in", 0.05, 1e-3, 0.5),
        (space, "thermal.R_boundary", 0.05, 1e-3, 0.5),
        (space, "thermal.f_wall", 0.5, 0.01, 0.99),
        (space, "thermal.f_air", 0.5, 0.01, 0.99),
        (space, "mass.V", 80, 10, 300),
        (space, "mass.G_occ", 5e-6, 1e-6, 1e-5),
        (space, "mass.m_inf", 1e-3, 1e-4, 1e-2),
        (space_heater, "thermalMassHeatCapacity", 7000, 100, 100000),
        (space_heater, "UA", 100, 10, 10000),
        (heating_controller, "kp", 0.001, 1e-5, 1.0),
        (heating_controller, "Ti", 3600, 60, 86400),
        (valve, "waterFlowRateMax", 0.016, 1e-4, 1.0),
        (valve, "valveAuthority", 0.5, 0.01, 0.99),
        (supply_damper, "nominalAirFlowRate", 0.1, 0.001, 1.0),
        (exhaust_damper, "nominalAirFlowRate", 0.1, 0.001, 1.0),
    ]


def thermal_parameters(model):
    """A temperature-focused 4-parameter subset (RC identification, Blum-et-al
    style).  This is where collocation applies cleanly -- the composer Jacobian
    is near-exact, so the fast path shines."""
    space = model.components["office"]
    return [
        (space, "thermal.C_air", 2e6, 1e6, 1e7),
        (space, "thermal.C_wall", 2e6, 1e6, 1e7),
        (space, "thermal.R_out", 0.01, 1e-3, 0.1),
        (space, "thermal.R_in", 0.05, 1e-3, 0.5),
    ]


def example_measurements(model):
    """The four calibration sensors from the estimation example."""
    c = model.components
    percentile = 2
    return [
        (c["office_valve_position_sensor"], 0.05 / percentile),
        (c["office_temperature_sensor"], 0.1 / percentile),
        (c["office_co2_sensor"], 30 / percentile),
        (c["office_damper_position_sensor"], 0.05 / percentile),
    ]


def temperature_measurement(model):
    """Just the zone-temperature sensor (for the thermal-RC parameter set)."""
    return [(model.components["office_temperature_sensor"], 0.05)]


def apply_estimated(model, params, result_x):
    """Write the estimated (denormalized) values back onto the components.

    ``estimate`` leaves the model at the *last* objective evaluation, which for
    the transcription solvers is not necessarily the optimum -- so re-apply
    ``result_x`` explicitly before the calibrated simulation.
    """
    for (comp, attr, *_), value in zip(params, result_x):
        rgetattr(comp, attr).set(float(value), normalized=False)


def estimate_and_predict(tag, method, options, periods, param_set="full"):
    """Estimate over ``periods`` (list of (start, end)), then simulate period 0."""
    start_list = [s for s, _ in periods]
    end_list = [e for _, e in periods]
    model = load_model()
    est = tb.Estimator(tb.Simulator(model))
    if param_set == "thermal":
        params = thermal_parameters(model)
        measurements = temperature_measurement(model)
    else:
        params = example_parameters(model)
        measurements = example_measurements(model)
    is_transcription = len(method) == 4

    t0 = time.time()
    result = est.estimate(
        parameters=params,
        measurements=measurements,
        start_time=start_list,
        end_time=end_list,
        step_size=STEP_SIZE,
        n_warmup=0 if is_transcription else 20,
        method=method,
        options=options,
    )
    elapsed = time.time() - t0

    # Calibrated prediction over the first period (continuous rollout).
    apply_estimated(model, params, result.get("result_x", []))
    model.set_save_simulation_result(flag=True)

    # Both methods are evaluated identically: a continuous rollout from the model's
    # default initial state, with the first WARMUP_SKIP steps excluded from the
    # RMSE (below) so the initial transient -- which depends on the unknown true
    # initial state, not the estimated parameters -- doesn't bias the comparison.
    #
    # NOTE on ``seed_initial_state``: transcription also estimates the boundary
    # states (available as ``result["estimated_initial_state"]``), but seeding the
    # *full* estimated state here makes the rollout WORSE, not better: the observed
    # air temperature is estimated well (~0.15 K), but the weakly-observable wall
    # temperature and PID integral states settle at non-physical values that fit
    # the one-step defects yet contaminate the multi-step rollout (the wall sits at
    # ~27 C with C_wall pinned at its upper bound, a frozen phantom heat source).
    # A fair, apples-to-apples metric therefore uses the same default init + warmup
    # for both methods.
    seed_initial_state = False
    est_x0 = result.get("estimated_initial_state", None)

    def _seed_initial_state():
        if seed_initial_state and est_x0:
            for cid, block in est_x0.items():
                model.components[cid].set_state(block.unsqueeze(0))

    est.simulator.simulate(
        start_time=start_list[0], end_time=end_list[0], step_size=STEP_SIZE,
        show_progress_bar=False, after_initialize=_seed_initial_state,
    )
    # Temperature RMSE on the continuous prediction -- a meaningful, comparable
    # metric in K.  (The estimator's internal objective aggregates all four
    # sensors across mixed units -- K, ppm, [0-1] -- so it is dominated by CO2
    # and not a clean per-method quality measure.)
    #
    # Fairness: every method rolls out from the SAME default initial state, so
    # the first ~WARMUP_SKIP steps are a settling transient that depends on the
    # (unknown) true initial state, not on the estimated parameters.  Score only
    # the settled portion so single-shooting (fit with n_warmup>0) and the
    # transcription methods (which optimise+discard their own initial states) are
    # judged on equal footing.
    WARMUP_SKIP = 20
    space = model.components["office"]
    pred = space.output["indoorTemperature"].history(i_s=0, i_c=0).detach().numpy()
    actual = np.asarray(model.components["office_temperature_sensor"].time_series_input.values).flatten()
    n = min(len(pred), len(actual))
    s = min(WARMUP_SKIP, max(0, n - 1))
    temp_rmse = float(np.sqrt(np.mean((pred[s:n] - actual[s:n]) ** 2)))
    return {
        "tag": tag, "time": elapsed, "rmse": temp_rmse,
        "x": list(result.get("result_x", [])),
        "model": model, "simulator": est.simulator,
    }


def save_after_calibration_plot(res, outdir):
    """Duplicate the estimation example's 'After calibration' plot for one run."""
    model, sim = res["model"], res["simulator"]
    heating_controller = model.components["office_temperature_heating_controller"]
    temp_sensor = model.components["office_temperature_sensor"]
    fig, _ = tb.plot.plot_component(
        sim,
        components_1axis=[
            ("office", "indoorTemperature", "output"),
            (heating_controller.id, "setpointValue", "input"),
            (temp_sensor.time_series_input.values, "Actual temperature"),
        ],
        components_2axis=[("office_space_heater", "Power", "output")],
        components_3axis=[("office_space_heater_valve", "valvePosition", "output")],
        ylabel_1axis=r"Temperature [$^\circ$C]",
        ylabel_2axis="Power [W]",
        ylabel_3axis="Valve position [0-1]",
        title=f"After calibration - {res['tag']}",
        show=False, nticks=11,
    )
    path = os.path.join(outdir, f"after_calibration_{res['tag'].replace(' ', '_')}.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def save_overlay(results, outdir):
    """Overlay each method's predicted indoor temperature vs. the measurement."""
    fig, ax = plt.subplots(figsize=(11, 5))
    ref = results[0]
    sensor = ref["model"].components["office_temperature_sensor"]
    actual = np.asarray(sensor.time_series_input.values).flatten()
    time_axis = ref["simulator"].date_time_steps[0]
    n = min(len(actual), len(time_axis))
    ax.plot(time_axis[:n], actual[:n], "k.", ms=4, label="Measured")
    for res in results:
        space = res["model"].components["office"]
        pred = space.output["indoorTemperature"].history(i_s=0, i_c=0).detach().numpy()
        m = min(len(pred), len(time_axis))
        ax.plot(time_axis[:m], pred[:m], lw=1.8,
                label=f"{res['tag']} (RMSE={res['rmse']:.3f} K, {res['time']:.0f} s)")
    ax.set_ylabel(r"Indoor temperature [$^\circ$C]")
    ax.set_xlabel("Time")
    ax.set_title("Calibrated prediction vs. measurement (period 1)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    path = os.path.join(outdir, "overlay_temperature.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hours", type=int, default=0,
                    help="if >0, use a single window of this length instead of the "
                         "example's two periods (for quick runs)")
    ap.add_argument("--segments", type=int, default=10, help="multiple-shooting segments per period")
    ap.add_argument("--maxiter", type=int, default=100)
    ap.add_argument("--methods", default="single_shooting,multiple_shooting",
                    help="comma list of: single_shooting, multiple_shooting, collocation, slsqp")
    ap.add_argument("--param-set", default="full", choices=["full", "thermal"],
                    help="'full' = 19 params + 4 sensors; 'thermal' = 4 RC params + "
                         "temperature sensor (where collocation applies cleanly)")
    ap.add_argument("--outdir", default="collocation_plots")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    if args.hours > 0:
        periods = [(EXAMPLE_START[0], EXAMPLE_START[0] + datetime.timedelta(hours=args.hours))]
    else:
        periods = list(zip(EXAMPLE_START, EXAMPLE_END))
    print(f"periods={len(periods)}  step={STEP_SIZE}s  segments/period={args.segments}  "
          f"maxiter={args.maxiter}\n")

    results = []
    for key in [k.strip() for k in args.methods.split(",")]:
        if key not in METHODS:
            print(f"  (skipping unknown method '{key}')")
            continue
        tag, method = METHODS[key]
        options = {"maxiter": args.maxiter}
        if key == "multiple_shooting":
            options["n_segments"] = args.segments
        res = estimate_and_predict(tag, method, options, periods, args.param_set)
        results.append(res)
        save_after_calibration_plot(res, args.outdir)

    print(f"\n{'method':<28}{'time [s]':>12}{'Temp RMSE [K]':>16}")
    print("-" * 56)
    for res in results:
        print(f"{res['tag']:<28}{res['time']:>12.1f}{res['rmse']:>12.4f}")

    if results:
        save_overlay(results, args.outdir)
        print(f"\nPlots written to '{args.outdir}/'")


if __name__ == "__main__":
    main()
