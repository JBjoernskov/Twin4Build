"""
Run optimization on a calibrated model.

Loads the serialized model and estimation results, rewires the model for
optimization (replacing the heating controller with an optimizable valve
schedule), runs the optimizer, and saves results to a pickle file.

Usage:
    python run_optimization.py
"""

# Standard library imports
import datetime
import os
import pickle

# Third party imports
import pandas as pd
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
    "maxiter": 2000,
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
    model.load_estimation_result(filename=result_path, verbose=1)
    print("Calibrated model loaded.")

    # ── 2. Component references ───────────────────────────────────────────
    space = model.components["office"]
    space_heater = model.components["office_space_heater"]
    heating_controller = model.components["office_temperature_heating_controller"]
    space_heater_valve = model.components["office_space_heater_valve"]
    heating_setpoint = model.components["office_temperature_heating_setpoint"]

    # ── 3. Create optimization components ─────────────────────────────────
    valve_position_schedule = tb.ScheduleSystem(
        weekDayRulesetDict={
            "ruleset_default_value": 0,
            "ruleset_start_minute": [0, 0],
            "ruleset_end_minute": [0, 0],
            "ruleset_start_hour": [8, 19],
            "ruleset_end_hour": [16, 20],
            "ruleset_value": [0.5, 0.5],
        },
        id="valve_position_schedule",
    )

    # Load real Elspot prices from Dec 11-13, 2024 and remap to Dec 1-3, 2023
    elspot_raw = pd.read_csv(
        utils.get_path(["Elspotprices2023-2025.csv"]), sep=";", decimal=","
    )
    elspot_raw["HourDK"] = pd.to_datetime(elspot_raw["HourDK"], dayfirst=True)

    src_start = pd.Timestamp("2024-12-11")
    src_end = pd.Timestamp("2024-12-14")
    mask = (elspot_raw["HourDK"] >= src_start) & (elspot_raw["HourDK"] < src_end)
    elspot_subset = elspot_raw.loc[mask, ["HourDK", "SpotPriceDKK"]].copy()

    date_offset = pd.DateOffset(years=1, days=7)
    elspot_subset["HourDK"] = elspot_subset["HourDK"] - date_offset
    elspot_subset.columns = ["time", "value"]
    elspot_subset["value"] = elspot_subset["value"] / 1000  # DKK/MWh -> DKK/kWh
    elspot_subset = elspot_subset.sort_values("time").reset_index(drop=True)

    elspot_clean_path = utils.get_path(["estimator_example", "electricity_price.csv"])
    elspot_subset.to_csv(elspot_clean_path, index=False)
    print(
        f"Price range: {elspot_subset['value'].min():.4f} – {elspot_subset['value'].max():.4f} DKK/kWh"
    )

    price_schedule = tb.ScheduleSystem(
        filename=elspot_clean_path, datecolumn=0, valuecolumn=1, id="price_schedule"
    )

    costs_sensor = tb.ScalarProductSystem(
        scale_factor=STEP_SIZE / 3600 / 1000, id="costs_sensor"
    )

    cooling_setpoint = tb.ScheduleSystem(
        weekDayRulesetDict={
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
    model.add_connection(space_heater, costs_sensor, "Power", "input_1")
    model.add_connection(price_schedule, costs_sensor, "scheduleValue", "input_2")
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
    objectives = [(costs_sensor, "output", "min")]
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
        "electricity_price": price_schedule.output["scheduleValue"]
        .history()
        .detach()
        .clone(),
        "cost_per_step": costs_sensor.output["output"].history().detach().clone(),
        "step_size": STEP_SIZE,
        "opt_start_time": opt_start_time,
        "opt_end_time": opt_end_time,
        # Diagnostic: building space inputs (heat sources)
        "supply_air_temperature": space.input["supplyAirTemperature"]
        .history()
        .detach()
        .clone(),
        "supply_air_flow_rate": space.input["supplyAirFlowRate"]
        .history()
        .detach()
        .clone(),
        "outdoor_temperature": space.input["outdoorTemperature"]
        .history()
        .detach()
        .clone(),
        "number_of_people": space.input["numberOfPeople"].history().detach().clone(),
        "heat_gain": space.input["heatGain"].history().detach().clone(),
        "global_irradiation": space.input["globalIrradiation"]
        .history()
        .detach()
        .clone(),
    }

    results_dir = utils.get_path(
        ["generated_files", "models", "full_workflow_example", "optimization_results"]
    )
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(results_dir, f"{timestamp}_optimization.pickle")
    with open(results_path, "wb") as f:
        pickle.dump(opt_results, f)

    cost_1d = opt_results["cost_per_step"][:, 0, 0]
    print(f"\nOptimization results saved to: {results_path}")
    print(f"Total cost: {cost_1d.sum():.2f} DKK")
    print(f"Peak cost/step: {cost_1d.max():.4f} DKK")
    print(f"Peak power: {opt_results['power'][:, 0, 0].max():.1f} W")


if __name__ == "__main__":
    main()
