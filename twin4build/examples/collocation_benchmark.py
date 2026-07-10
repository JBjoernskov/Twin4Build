"""Benchmark: single-shooting vs. multiple-shooting / collocation estimation.

Compares parameter-estimation approaches on the parameter-estimation example
model (a single-zone RC thermal + CO2 building), reporting wall-clock time and
data-fit RMSE.  All approaches use the identical ``Estimator`` API on a plain
``twin4build`` ``Model`` -- only the ``method`` tuple changes:

    single-shooting  IPOPT   -> ("casadi", "ipopt", "ad")
    single-shooting  SLSQP   -> ("scipy",  "SLSQP", "ad")
    multiple-shooting IPOPT  -> ("casadi", "ipopt", "ad", "multiple_shooting")
    collocation       IPOPT  -> ("casadi", "ipopt", "ad", "collocation")

The 4th (optional) tuple element selects the *transcription*: single-shooting
runs one forward simulation per objective evaluation and backpropagates through
the whole horizon; multiple-shooting splits the horizon into segments whose
initial states become decision variables stitched by continuity penalties
(``options={"n_segments": K}``); collocation is the one-segment-per-timestep
limit.  Gradients then flow only through short segments, improving conditioning
(the motivation is documented in ``twin4build.estimator._transcription``).

IPOPT is provided by CasADi (``pip install casadi``), an optional dependency.

Run e.g.::

    python -m twin4build.examples.collocation_benchmark --hours 24 --segments 6 \
        --maxiter 50 --methods ipopt_ss,ms
"""

import argparse
import datetime
import time

from dateutil import tz

import twin4build as tb
import twin4build.examples.utils as utils

STEP_SIZE = 1200  # 20 min
START = datetime.datetime(2023, 11, 27, 0, 0, 0, tzinfo=tz.gettz("Europe/Copenhagen"))


def load_model():
    """Load the estimation-example model and point sensors at the bundled CSVs."""
    model = tb.Model(id="collocation_benchmark")
    model.load(
        simulation_model_filename=utils.get_path(
            ["estimator_example", "instance_graph.ttl"]
        )
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


def thermal_parameters(model):
    space = model.components["office"]
    return [
        (space, "thermal.C_air", 2e6, 1e6, 1e7),
        (space, "thermal.C_wall", 2e6, 1e6, 1e7),
        (space, "thermal.R_out", 0.01, 1e-3, 0.1),
        (space, "thermal.R_in", 0.05, 1e-3, 0.5),
    ]


def run(tag, method, options, hours):
    model = load_model()
    est = tb.Estimator(tb.Simulator(model))
    is_transcription = len(method) == 4
    t0 = time.time()
    result = est.estimate(
        parameters=thermal_parameters(model),
        measurements=[(model.components["office_temperature_sensor"], 0.1)],
        start_time=START,
        end_time=START + datetime.timedelta(hours=hours),
        step_size=STEP_SIZE,
        n_warmup=0 if is_transcription else 5,
        method=method,
        options=options,
    )
    dt = time.time() - t0
    rmse = getattr(est, "_last_rmse", float("nan"))
    xs = ", ".join(f"{v:.3g}" for v in result.get("result_x", []))
    print(f"[{tag:<28}] time={dt:7.1f}s   RMSE={rmse:.4f} K   x=[{xs}]")
    return dt, rmse


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hours", type=int, default=24, help="estimation horizon length")
    ap.add_argument("--segments", type=int, default=6, help="multiple-shooting segments")
    ap.add_argument("--maxiter", type=int, default=50)
    ap.add_argument(
        "--methods",
        default="ipopt_ss,ms",
        help="comma list of: slsqp_ss, ipopt_ss, ms, colloc",
    )
    args = ap.parse_args()
    methods = args.methods.split(",")
    print(
        f"window={args.hours}h  step={STEP_SIZE}s  segments={args.segments}  "
        f"maxiter={args.maxiter}\n"
    )
    if "slsqp_ss" in methods:
        run("SLSQP single-shooting", ("scipy", "SLSQP", "ad"), {"maxiter": args.maxiter}, args.hours)
    if "ipopt_ss" in methods:
        run("IPOPT single-shooting", ("casadi", "ipopt", "ad"), {"maxiter": args.maxiter}, args.hours)
    if "ms" in methods:
        run(
            "IPOPT multiple-shooting",
            ("casadi", "ipopt", "ad", "multiple_shooting"),
            {"maxiter": args.maxiter, "n_segments": args.segments},
            args.hours,
        )
    if "colloc" in methods:
        run(
            "IPOPT collocation",
            ("casadi", "ipopt", "ad", "collocation"),
            {"maxiter": args.maxiter},
            args.hours,
        )


if __name__ == "__main__":
    main()
