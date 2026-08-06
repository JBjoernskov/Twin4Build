"""
Run optimization on a calibrated model – minimize heating energy only.

Loads the serialized model and estimation results, rewires the model for
optimization (replacing the heating controller with an optimizable valve
schedule), and minimizes total heating power subject to comfort constraints.

Usage:
    python run_optimization_min_energy.py
"""

# Standard library imports
import datetime
import os
import pickle

# Third party imports
from dateutil import tz

# Local application imports
import twin4build as tb
import twin4build.examples.utils as utils

# ── Configuration ─────────────────────────────────────────────────────────
RESULT_PICKLE = (
    r"generated_files\models\full_workflow_example"
    r"\model_parameters\estimation_results"
    r"\20260303_103841_scipy_SLSQP_ad.pickle"
)

STEP_SIZE = 1200  # 20 minutes

OPT_START = datetime.datetime(
    year=2023,
    month=12,
    day=2,
    hour=0,
    minute=0,
    second=0,
    tzinfo=tz.gettz("Europe/Copenhagen"),
)
OPT_END = datetime.datetime(
    year=2023,
    month=12,
    day=7,
    hour=0,
    minute=0,
    second=0,
    tzinfo=tz.gettz("Europe/Copenhagen"),
)

OPT_OPTIONS = {
    "maxiter": 300,
    "tol": 1e-15,
    "disp": True,
}


def main():
    # ── 1. Load model & apply calibrated parameters ───────────────────────
    model = tb.Model(id="full_workflow_example")
    filename_simulation, _ = model._simulation_model._semantic_model.get_dir(
        filename="instance_graph.ttl"
    )
    model.load(simulation_model_filename=filename_simulation)

    result_path = utils.get_path([RESULT_PICKLE])
    model.load_estimation_result(filename=result_path)
    print("Calibrated model loaded.")

    # ── 2. Component references ───────────────────────────────────────────
    space = model.components["office"]
    space_heater = model.components["office_space_heater"]
    heating_controller = model.components["office_temperature_heating_controller"]
    space_heater_valve = model.components["office_space_heater_valve"]
    heating_setpoint = model.components["office_temperature_heating_setpoint"]

    # ── 3. Create optimization components ─────────────────────────────────
    valve_position_schedule = tb.ScheduleSystem(
        weekday_ruleset={
            "ruleset_default_value": 0,
            "ruleset_start_minute": [0, 0],
            "ruleset_end_minute": [0, 0],
            "ruleset_start_hour": [8, 19],
            "ruleset_end_hour": [16, 20],
            "ruleset_value": [0.5, 0.5],
        },
        id="valve_position_schedule",
    )

    cooling_setpoint = tb.ScheduleSystem(
        weekday_ruleset={
            "ruleset_default_value": 0,
            "ruleset_start_minute": [0, 0, 0],
            "ruleset_end_minute": [0, 0, 0],
            "ruleset_start_hour": [0, 8, 17],
            "ruleset_end_hour": [8, 17, 24],
            "ruleset_value": [30, 25, 30],
        },
        id="CoolingSetpoint",
    )

    # ── 4. Rewire model for optimization ──────────────────────────────────
    model.remove_connection(
        heating_controller, space_heater_valve, "inputSignal", "valvePosition"
    )
    model.add_connection(
        valve_position_schedule, space_heater_valve, "scheduleValue", "valvePosition"
    )
    model.load()
    print("Model rewired for optimization.")

    # ── 5. Run optimization ───────────────────────────────────────────────
    opt_start_time = [OPT_START]
    opt_end_time = [OPT_END]

    simulator_opt = tb.Simulator(model)
    simulator_opt.simulate(
        step_size=STEP_SIZE, start_time=opt_start_time, end_time=opt_end_time
    )

    variables = [(valve_position_schedule, "scheduleValue", 0, 1)]
    objectives = [(space_heater, "Power", "min")]
    ineq_cons = [
        (space, "indoorTemperature", "upper", cooling_setpoint),
        (space, "indoorTemperature", "lower", heating_setpoint),
    ]

    optimizer = tb.Optimizer(simulator_opt)
    optimizer.optimize(
        start_time=opt_start_time,
        end_time=opt_end_time,
        step_size=STEP_SIZE,
        variables=variables,
        objectives=objectives,
        eq_cons=None,
        ineq_cons=ineq_cons,
        method="scipy",
        options=OPT_OPTIONS,
    )

    # ── 6. Save results ───────────────────────────────────────────────────
    model.add_component(cooling_setpoint)
    model.add_component(heating_setpoint)

    opt_results = {
        "date_time_steps": simulator_opt.date_time_steps,
        "indoor_temperature": space.output["indoorTemperature"]
        .history()
        .detach()
        .clone(),
        "heating_setpoint": heating_setpoint.output["scheduleValue"]
        .history()
        .detach()
        .clone(),
        "cooling_setpoint": cooling_setpoint.output["scheduleValue"]
        .history()
        .detach()
        .clone(),
        "power": space_heater.output["Power"].history().detach().clone(),
        "valve_position": space_heater_valve.output["valvePosition"]
        .history()
        .detach()
        .clone(),
        "step_size": STEP_SIZE,
        "opt_start_time": opt_start_time,
        "opt_end_time": opt_end_time,
    }

    results_dir = utils.get_path(
        ["generated_files", "models", "full_workflow_example", "optimization_results"]
    )
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(
        results_dir, f"{timestamp}_optimization_min_energy.pickle"
    )
    with open(results_path, "wb") as f:
        pickle.dump(opt_results, f)

    power_1d = opt_results["power"][:, 0, 0]
    energy_kwh = power_1d.sum() * STEP_SIZE / 3600 / 1000
    print(f"\nOptimization results saved to: {results_path}")
    print(f"Total energy: {energy_kwh:.2f} kWh")
    print(f"Peak power: {power_1d.max():.1f} W")


if __name__ == "__main__":
    main()
