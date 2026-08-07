"""
Twin4Build: Full Workflow Example

Demonstrates the complete Twin4Build workflow on a single model:

1. Semantic Model -> Simulation Model (Translator): Load a semantic description
   from an Excel file and automatically generate a simulation model
2. Simulation Model -> Calibrated Model (Estimator): Calibrate the simulation
   model parameters against real sensor data
3. Calibrated Model -> Optimized Control (Optimizer): Optimize the valve position
   schedule to minimize energy use while maintaining thermal comfort

The example uses a single-room office model with:
- A building space with thermal and mass balance
- A space heater with valve control
- Temperature and CO2 controllers with dampers
- Sensors for temperature, CO2, valve position, and damper position
"""

# Standard library imports
import datetime
import os
import pickle

# Third party imports
import matplotlib.pyplot as plt
import pandas as pd
import torch
from dateutil import tz

# Local application imports
import twin4build as tb
import twin4build.examples.utils as utils

# ---------------------------------------------------------------------------
# Part 1: Semantic Model -> Simulation Model (Translator)
# ---------------------------------------------------------------------------


def fcn(self):
    """Post-translation configuration.

    Called after the translator generates the simulation model. Adds:
    - Supply water temperature and boundary temperature schedules
    - A wall component coupling the office to the boundary temperature
    - Sensor data file connections
    - Control setpoints
    - Outdoor environment data
    """
    supply_water_schedule = tb.PiecewiseLinearScheduleSystem(
        default_x=[-12, 5, 20],
        default_y=[60, 50, 20],
        id="supply_water_schedule",
    )

    boundary_temp_schedule = tb.ScheduleSystem(
        weekday_ruleset={
            "ruleset_default_value": 21,
            "ruleset_start_minute": [],
            "ruleset_end_minute": [],
            "ruleset_start_hour": [],
            "ruleset_end_hour": [],
            "ruleset_value": [],
        },
        weekend_ruleset={
            "ruleset_default_value": 21,
            "ruleset_start_minute": [],
            "ruleset_end_minute": [],
            "ruleset_start_hour": [],
            "ruleset_end_hour": [],
            "ruleset_value": [],
        },
        id="boundary_temp_schedule",
    )

    # The wall toward the neighbouring (unmodeled) space: the office couples
    # to the boundary-temperature schedule through a 2R1C WallSystem.
    # The wall owns the wall state, so the heat exchange is energy-consistent
    # by construction (see the WallSystem docstring).
    boundary_wall = tb.WallSystem(
        C=1e6,
        R_a=0.02,
        R_b=0.02,
        id="office_boundary_wall",
    )
    self.add_connection(
        self.components["office"],
        boundary_wall,
        "indoorTemperature",
        "temperatureA",
    )
    self.add_connection(
        boundary_temp_schedule,
        boundary_wall,
        "scheduleValue",
        "temperatureB",
    )
    self.add_connection(
        boundary_wall,
        self.components["office"],
        "heatFlowRateA",
        "wallHeatGain",
        input_port_index=0,
    )
    self.add_connection(
        self.components["outdoor_environment"],
        supply_water_schedule,
        "outdoorTemperature",
        "x",
    )
    self.add_connection(
        supply_water_schedule,
        self.components["office_space_heater"],
        "scheduleValue",
        "supplyWaterTemperature",
    )

    # Packaged example CSVs (same schema as the case-study PLC exports:
    # datecolumn=2 / valuecolumn=4). Resolved via utils.get_path so the
    # workflow is machine-independent (Colab, CI, any checkout).
    self.components["office_temperature_sensor"].filename = utils.get_path(
        ["estimator_example", "temperature_sensor.csv"]
    )
    self.components["office_temperature_sensor"].datecolumn = 2
    self.components["office_temperature_sensor"].valuecolumn = 4

    self.components["office_co2_sensor"].filename = utils.get_path(
        ["estimator_example", "co2_sensor.csv"]
    )
    self.components["office_co2_sensor"].datecolumn = 2
    self.components["office_co2_sensor"].valuecolumn = 4

    self.components["office_valve_position_sensor"].filename = utils.get_path(
        ["estimator_example", "valve_position_sensor.csv"]
    )
    self.components["office_valve_position_sensor"].datecolumn = 2
    self.components["office_valve_position_sensor"].valuecolumn = 4

    self.components["office_damper_position_sensor"].filename = utils.get_path(
        ["estimator_example", "damper_position_sensor.csv"]
    )
    self.components["office_damper_position_sensor"].datecolumn = 2
    self.components["office_damper_position_sensor"].valuecolumn = 4

    self.components["supply_air_temperature_sensor"].filename = utils.get_path(
        ["estimator_example", "supply_air_temperature.csv"]
    )
    self.components["supply_air_temperature_sensor"].datecolumn = 2
    self.components["supply_air_temperature_sensor"].valuecolumn = 4

    self.components["office_co2_setpoint"].weekDayRulesetDict = {
        "ruleset_default_value": 900,
        "ruleset_start_minute": [],
        "ruleset_end_minute": [],
        "ruleset_end_hour": [],
        "ruleset_start_hour": [],
        "ruleset_value": [],
    }

    # Replace constant-zero occupancy schedule with inverse-CO2 estimator.
    # Indoor CO2 and damper position are read directly from CSV (real measurements)
    # to avoid a gradient feedback loop through simulated sensor outputs.
    self.remove_component(self.components["office_occupancy_profile"])
    occupancy_system = tb.OccupancySystem(
        V=100,
        G_occ=5e-6,
        m_inf=0.001,
        co2_filename=utils.get_path(["estimator_example", "co2_sensor.csv"]),
        co2_date_column=2,
        co2_value_column=4,
        damper_filename=utils.get_path(
            ["estimator_example", "damper_position_sensor.csv"]
        ),
        damper_date_column=2,
        damper_value_column=4,
        id="office_occupancy",
    )
    self.add_connection(
        self.components["outdoor_environment"],
        occupancy_system,
        "outdoorCo2Concentration",
        "outdoorCo2Concentration",
    )
    self.add_connection(
        occupancy_system, self.components["office"], "scheduleValue", "numberOfPeople"
    )

    # --- Occupancy-driven minimum-ventilation controller ---
    # Remove existing CO2 controller → damper connections (from semantic model)
    self.remove_connection(
        self.components["office_co2_controller"],
        self.components["office_supply_damper"],
        "inputSignal",
        "damperPosition",
    )
    self.remove_connection(
        self.components["office_co2_controller"],
        self.components["office_exhaust_damper"],
        "inputSignal",
        "damperPosition",
    )

    # Occupancy detector: continuous N_occ → smooth binary (0/1).
    # The threshold must sit between the unoccupied noise floor (~0) and the
    # occupied-hours signal: this room's CO2 elevation is weak (~50 ppm at
    # damper 0.3), which back-solves to only ~0.25 inferred occupants, so a
    # threshold of 1 person would never trigger and the ventilation branch
    # would stay off (with a saturated sigmoid the calibration gradient dies
    # and no solver can recover it).  A moderate steepness keeps the sigmoid
    # differentiable near the threshold instead of a hard step.
    occupancy_detector = tb.OccupancyDetectorSystem(
        threshold=0.15,
        steepness=30.0,
        id="office_occupancy_detector",
    )
    self.add_connection(
        occupancy_system, occupancy_detector, "scheduleValue", "occupancy"
    )

    # On/off controller: binary occupancy → minimum damper position (0.3 when occupied)
    occupancy_controller = tb.SmoothOnOffControllerSystem(
        on_value=0.3,
        off_value=0.0,
        is_reverse=False,
        # steepness=10.0,
        id="office_occupancy_controller",
    )
    self.add_connection(
        occupancy_detector, occupancy_controller, "occupancySignal", "actualValue"
    )

    # Constant setpoint at 0.5 (midpoint of the 0-1 detector signal)
    occupancy_controller_setpoint = tb.ScheduleSystem(
        weekday_ruleset={
            "ruleset_default_value": 0.5,
            "ruleset_start_minute": [],
            "ruleset_end_minute": [],
            "ruleset_start_hour": [],
            "ruleset_end_hour": [],
            "ruleset_value": [],
        },
        id="office_occupancy_controller_setpoint",
    )
    self.add_connection(
        occupancy_controller_setpoint,
        occupancy_controller,
        "scheduleValue",
        "setpointValue",
    )

    # Max selector: final damper position = max(CO2 PID output, occupancy on/off output)
    damper_max = tb.MaxSystem(id="office_damper_max")
    self.add_connection(
        self.components["office_co2_controller"],
        damper_max,
        "inputSignal",
        "inputs",
        input_port_index=0,
    )
    self.add_connection(
        occupancy_controller, damper_max, "inputSignal", "inputs", input_port_index=1
    )
    self.add_connection(
        damper_max, self.components["office_supply_damper"], "value", "damperPosition"
    )
    self.add_connection(
        damper_max, self.components["office_exhaust_damper"], "value", "damperPosition"
    )

    # Rewire damper position sensor to read actual damper position (after MaxSystem)
    self.remove_connection(
        self.components["office_co2_controller"],
        self.components["office_damper_position_sensor"],
        "inputSignal",
        "measuredValue",
    )
    self.add_connection(
        damper_max,
        self.components["office_damper_position_sensor"],
        "value",
        "measuredValue",
    )

    self.components["office_temperature_heating_setpoint"].filename = utils.get_path(
        ["estimator_example", "temperature_heating_setpoint.csv"]
    )
    self.components["office_temperature_heating_setpoint"].datecolumn = 2
    self.components["office_temperature_heating_setpoint"].valuecolumn = 4

    self.components["outdoor_environment"].use_spreadsheet = True
    self.components["outdoor_environment"].filename_outdoorTemperature = utils.get_path(
        ["estimator_example", "outdoor_environment.csv"]
    )
    self.components["outdoor_environment"].datecolumn_outdoorTemperature = 0
    self.components["outdoor_environment"].valuecolumn_outdoorTemperature = 1

    self.components["outdoor_environment"].filename_globalIrradiation = utils.get_path(
        ["estimator_example", "outdoor_environment.csv"]
    )
    self.components["outdoor_environment"].datecolumn_globalIrradiation = 0
    self.components["outdoor_environment"].valuecolumn_globalIrradiation = 2

    self.components["outdoor_environment"].filename_outdoorCo2Concentration = (
        utils.get_path(["estimator_example", "outdoor_environment.csv"])
    )
    self.components["outdoor_environment"].datecolumn_outdoorCo2Concentration = 0
    self.components["outdoor_environment"].valuecolumn_outdoorCo2Concentration = 3

    # UA will be estimated — disable the automatic re-computation from nominal
    # values so initialize() preserves the calibrated/set value.
    # This is serialized via the config so loading from RDF also gets False.
    self.components["office_space_heater"].initialize_UA = False


def main():
    # Re-serialize: uncomment this block to rebuild from the semantic model
    model = tb.Model(id="full_workflow_example")
    filename = utils.get_path(["estimator_example", "one_room_example_model.xlsm"])
    model.load(semantic_model_filename=filename, fcn=fcn)
    print("Serializing model...")
    # model.serialize()

    # aa

    # print(model)

    # Load from serialized model (comment out the block above after re-serializing)
    # model = tb.Model(id="full_workflow_example")
    # filename_simulation, _ = model._simulation_model._semantic_model.get_dir(
    #     filename="instance_graph.ttl"
    # )
    # model.load(simulation_model_filename=filename_simulation)
    print(model)

    # --- 2.1 Set Up Simulation Parameters ---
    simulator = tb.Simulator(model)
    step_size = 1200  # 20 minutes in seconds

    start_time = [
        datetime.datetime(
            year=2023,
            month=12,
            day=2,
            hour=0,
            minute=0,
            second=0,
            tzinfo=tz.gettz("Europe/Copenhagen"),
        ),
    ]
    end_time = [
        datetime.datetime(
            year=2023,
            month=12,
            day=7,
            hour=0,
            minute=0,
            second=0,
            tzinfo=tz.gettz("Europe/Copenhagen"),
        ),
    ]

    print(start_time[0].weekday())
    print(end_time[0].weekday())

    # --- 2.2 Identify Model Components ---
    space = model.components["office"]
    space_heater = model.components["office_space_heater"]
    heating_controller = model.components["office_temperature_heating_controller"]
    co2_controller = model.components["office_co2_controller"]
    space_heater_valve = model.components["office_space_heater_valve"]
    supply_damper = model.components["office_supply_damper"]
    exhaust_damper = model.components["office_exhaust_damper"]

    occupancy_system = model.components["office_occupancy"]
    occupancy_detector = model.components["office_occupancy_detector"]
    occupancy_controller = model.components["office_occupancy_controller"]
    boundary_wall = model.components["office_boundary_wall"]

    print("Key components:")
    for name, comp in [
        ("Building space", space),
        ("Space heater", space_heater),
        ("Heating controller", heating_controller),
        ("CO2 controller", co2_controller),
        ("Valve", space_heater_valve),
        ("Supply damper", supply_damper),
        ("Exhaust damper", exhaust_damper),
        ("Occupancy", occupancy_system),
        ("Occupancy detector", occupancy_detector),
    ]:
        print(f"  {name}: {comp.id}")

    # --- 2.3 Define Target Parameters for Estimation ---
    parameters = [
        # Thermal parameters
        (space, "thermal.C_air", 5e5, 1e4, 5e5),
        (space, "thermal.C_wall", 1e6, 1e5, 3e6),
        (space, "thermal.R_out", 0.5, 0.01, 1),
        (space, "thermal.R_in", 0.1, 0.01, 1),
        # Boundary wall (WallSystem toward the boundary-temperature schedule)
        (boundary_wall, "C", 1e6, 1e4, 1e7),
        (boundary_wall, "R_a", 0.04, 0.0001, 1),
        (boundary_wall, "R_b", 0.04, 0.0001, 1),
        (space, "thermal.f_wall", 0.1, 0, 10),
        (space, "thermal.f_air", 0.1, 0, 10),
        (space, "thermal.Q_occ_gain", 100.0, 10, 200),
        # Space heater parameters
        (space_heater, "thermalMassHeatCapacity", 1e4, 1000, 2e5),
        (space_heater, "UA", 30, 1, 100),
        # Controller PID parameters (private = each controller gets its own values)
        (heating_controller, "kp", 0.005, 1e-5, 1, "private"),
        (co2_controller, "kp", 0.0001, 1e-5, 1, "private"),
        ([heating_controller, co2_controller], "Ti", 30, 1, 300, "private"),
        ([heating_controller, co2_controller], "Td", 0, 0, 1, "private"),
        # Valve parameters
        (space_heater_valve, "waterFlowRateMax", 0.001, 1e-6, 0.1),  # 0.003
        (space_heater_valve, "valveAuthority", 1, 0.4, 1),
        # Damper parameters — shared between model dampers and occupancy's internal dampers
        ([supply_damper, occupancy_system.supply_damper], "a", 1, 1, 10, "shared"),
        (
            [supply_damper, occupancy_system.supply_damper],
            "nominalAirFlowRate",
            0.1,
            1e-5,
            1,
            "shared",
        ),
        ([exhaust_damper, occupancy_system.exhaust_damper], "a", 1, 1, 10, "shared"),
        (
            [exhaust_damper, occupancy_system.exhaust_damper],
            "nominalAirFlowRate",
            0.1,
            1e-5,
            1,
            "shared",
        ),
        # Mass-balance / occupancy parameters (shared between space and occupancy
        # estimator).  G_occ and m_inf must stay in physically justified ranges:
        # per-person CO2 generation is well-known physics (~5e-6 kg/s per
        # person), and if infiltration can grow freely the estimator explains
        # the indoor-outdoor CO2 elevation with air exchange instead of
        # occupants -- inferred occupancy drops below the detection threshold,
        # the ventilation branch never fires (its sigmoid gradient dies), and
        # the simulated damper/CO2 collapse even though temperature fits.
        ([space, occupancy_system], "mass.V", 65, 50, 80, "shared"),
        ([space, occupancy_system], "mass.G_occ", 5e-6, 4e-6, 7e-6, "shared"),
        ([space, occupancy_system], "mass.m_inf", 0.001, 1e-4, 2e-3, "shared"),
        # Occupancy detector threshold
        # (occupancy_detector, "threshold", 0.5, 0.001, 5.0),
        # NOTE: the occupancy controller's onValue (minimum damper position
        # when occupied) is NOT estimated: it is a known BMS constant (0.3 in
        # the measured damper data), and leaving it free lets the solver
        # strand it at an arbitrary value once the detection branch is quiet.
    ]

    print(f"Total parameter groups: {len(parameters)}")

    # --- 2.4 Configure Measuring Devices ---
    percentile = 2
    measurements = [
        (model.components["office_valve_position_sensor"], 0.05 / percentile),
        (model.components["office_temperature_sensor"], 0.1 / percentile),
        (model.components["office_damper_position_sensor"], 0.05 / percentile),
        (model.components["office_co2_sensor"], 30 / percentile),
    ]

    print("Measuring devices for calibration:")
    for device, sd in measurements:
        print(f"  {device.id} (sd={sd})")

    # --- 2.5 Run Initial Simulation (Before Calibration) ---
    x0_values = []
    x0_components = []
    x0_names = []
    x0_min = []
    x0_max = []
    for entry in parameters:
        comp, name, val, lo, hi = entry[0], entry[1], entry[2], entry[3], entry[4]
        if isinstance(comp, list):
            for c in comp:
                x0_values.append(val)
                x0_components.append(c)
                x0_names.append(name)
                x0_min.append(lo)
                x0_max.append(hi)
        else:
            x0_values.append(val)
            x0_components.append(comp)
            x0_names.append(name)
            x0_min.append(lo)
            x0_max.append(hi)

    model.set_parameters(
        values=x0_values,
        components=x0_components,
        parameter_names=x0_names,
    )
    print(f"Set {len(x0_values)} initial parameter values.")

    simulator.simulate(step_size=step_size, start_time=start_time, end_time=end_time)
    print("Initial simulation completed.")

    # --- 2.6 Plot Initial Results (Before Calibration) ---
    fig, axes = tb.plot.plot(
        simulator.date_time_steps,
        [
            tb.plot.Entry(
                model.components["office_temperature_sensor"].time_series_input.values,
                label="Temperature measured",
                color=tb.plot.Colors.green,
                linewidth=2,
                fmt="--",
            ),
            tb.plot.Entry(
                model.components["office_temperature_sensor"]
                .output["measuredValue"]
                .history(),
                label="Temperature simulated",
                color=tb.plot.Colors.green,
                linewidth=1.5,
            ),
            tb.plot.Entry(
                model.components[
                    "office_valve_position_sensor"
                ].time_series_input.values,
                label="Valve position measured",
                color=tb.plot.Colors.blue,
                linewidth=2,
                fmt="--",
                axis=2,
            ),
            tb.plot.Entry(
                model.components["office_valve_position_sensor"]
                .output["measuredValue"]
                .history(),
                label="Valve position simulated",
                color=tb.plot.Colors.blue,
                linewidth=1.5,
                axis=2,
            ),
            tb.plot.Entry(
                heating_controller.input["setpointValue"].history(),
                label="Heating setpoint",
                color=tb.plot.Colors.red,
                linewidth=1,
            ),
        ],
        ylabel_1axis=r"Temperature [$^\circ$C]",
        ylabel_2axis="Valve position [0-1]",
        title="Before calibration",
        show=False,
        nticks=11,
    )

    fig, axes = tb.plot.plot(
        simulator.date_time_steps,
        [
            tb.plot.Entry(
                model.components["office_co2_sensor"].time_series_input.values,
                label=r"CO$_2$ concentration measured",
                color=tb.plot.Colors.green,
                linewidth=2,
                fmt="--",
            ),
            tb.plot.Entry(
                model.components["office_co2_sensor"].output["measuredValue"].history(),
                label=r"CO$_2$ concentration simulated",
                color=tb.plot.Colors.green,
                linewidth=1.5,
            ),
            tb.plot.Entry(
                model.components[
                    "office_damper_position_sensor"
                ].time_series_input.values,
                label=r"Damper position measured",
                color=tb.plot.Colors.blue,
                linewidth=2,
                fmt="--",
                axis=2,
            ),
            tb.plot.Entry(
                model.components["office_damper_position_sensor"]
                .output["measuredValue"]
                .history(),
                label=r"Damper position simulated",
                color=tb.plot.Colors.blue,
                linewidth=1.5,
                axis=2,
            ),
            tb.plot.Entry(
                model.components["office"].input["numberOfPeople"].history(),
                label=r"Number of people",
                color=tb.plot.Colors.red,
                linewidth=1.5,
                axis=2,
            ),
            tb.plot.Entry(
                model.components["office_occupancy_detector"]
                .output["occupancySignal"]
                .history(),
                label=r"Occupancy signal",
                color=tb.plot.Colors.purple,
                linewidth=1.5,
                axis=2,
            ),
        ],
        # tb.plot.Entry(heating_controller.input["setpointValue"].history(), label="Heating setpoint", color=tb.plot.Colors.red, linewidth=1),],
        ylabel_1axis=r"CO$_2$ concentration [ppmv]",
        ylabel_2axis="Damper position [0-1]",
        title="Before calibration",
        show=True,
        nticks=11,
    )

    # --- 2.7 Run Parameter Estimation ---
    estimator = tb.Estimator(simulator)
    options = {"maxiter": 300, "ftol": 1e-15}

    result = estimator.estimate(
        start_time,
        end_time,
        step_size,
        parameters,
        measurements,
        n_warmup=72,
        method=("scipy", "SLSQP", "ad"),
        options=options,
    )
    print(result)

    theta_mask = result["theta_mask"]
    theta_slices = result["theta_slices"]
    for id_, attr, param_idx in zip(
        result["component_id"], result["component_attr"], theta_mask
    ):
        start, end = theta_slices[param_idx]
        x = result["result_x"][start:end]
        print(f"{id_}.{attr} = {x[0] if len(x) == 1 else x}")

    # --- 2.8 Plot Calibrated Results ---
    model.set_save_simulation_result(flag=True)

    # Collocation estimates the boundary states along with theta; its RMSEs
    # are for a simulation starting from the ESTIMATED initial state.  Seed it,
    # otherwise the default initial conditions (e.g. wall temperature 20 degC,
    # with day-scale wall time constants) bias the whole horizon.
    # Single-shooting results carry no estimated state, so skip seeding there.
    _init_state = result.get("estimated_initial_state")

    def _seed_estimated_initial_state():
        # get_component: the office+wall pair executes as one fused
        # state-space block, so the state keys are executing-component ids.
        for comp_id, x0 in _init_state.items():
            model.get_component(comp_id).set_state(x0)

    simulator.simulate(
        step_size=step_size,
        start_time=start_time,
        end_time=end_time,
        after_initialize=_seed_estimated_initial_state if _init_state else None,
    )
    print("Calibration complete.")

    print(len(simulator.date_time_steps[0]))
    print(
        model.components["office_temperature_sensor"]
        .time_series_input.values[:, :, 0]
        .shape
    )
    print(
        model.components["office_temperature_sensor"]
        .output["measuredValue"]
        .history()[:, :, 0]
        .shape
    )

    fig, axes = tb.plot.plot(
        simulator.date_time_steps,
        [
            tb.plot.Entry(
                model.components["office_temperature_sensor"].time_series_input.values,
                label="measured",
                color=tb.plot.Colors.green,
                linewidth=1,
                fmt="--",
            ),
            tb.plot.Entry(
                model.components["office_temperature_sensor"]
                .output["measuredValue"]
                .history(),
                label="simulated",
                color=tb.plot.Colors.green,
                linewidth=1,
            ),
            tb.plot.Entry(
                heating_controller.input["setpointValue"].history(),
                label="indoor temperature",
                color=tb.plot.Colors.red,
                linewidth=1,
            ),
            tb.plot.Entry(
                model.components[
                    "office_valve_position_sensor"
                ].time_series_input.values,
                label="measured",
                color=tb.plot.Colors.purple,
                linewidth=1,
                fmt="--",
                axis=2,
            ),
            tb.plot.Entry(
                model.components["office_valve_position_sensor"]
                .output["measuredValue"]
                .history(),
                label="simulated",
                color=tb.plot.Colors.purple,
                linewidth=1,
                axis=2,
            ),
        ],
        ylabel_1axis=r"Temperature [$^\circ$C]",
        ylabel_2axis="Valve position [0-1]",
        title="After calibration",
        show=False,
        nticks=11,
    )

    fig, axes = tb.plot.plot(
        simulator.date_time_steps,
        [
            tb.plot.Entry(
                model.components["office_co2_sensor"].time_series_input.values,
                label=r"CO$_2$ concentration measured",
                color=tb.plot.Colors.green,
                linewidth=2,
                fmt="--",
            ),
            tb.plot.Entry(
                model.components["office_co2_sensor"].output["measuredValue"].history(),
                label=r"CO$_2$ concentration simulated",
                color=tb.plot.Colors.green,
                linewidth=1.5,
            ),
            tb.plot.Entry(
                model.components[
                    "office_damper_position_sensor"
                ].time_series_input.values,
                label=r"Damper position measured",
                color=tb.plot.Colors.blue,
                linewidth=2,
                fmt="--",
                axis=2,
            ),
            tb.plot.Entry(
                model.components["office_damper_position_sensor"]
                .output["measuredValue"]
                .history(),
                label=r"Damper position simulated",
                color=tb.plot.Colors.blue,
                linewidth=1.5,
                axis=2,
            ),
        ],
        # tb.plot.Entry(heating_controller.input["setpointValue"].history(), label="Heating setpoint", color=tb.plot.Colors.red, linewidth=1),],
        ylabel_1axis=r"CO$_2$ concentration [ppmv]",
        ylabel_2axis="Damper position [0-1]",
        title="After calibration",
        show=True,
        nticks=11,
    )

    # --- 3.1 Create Optimizable Valve Position Schedule ---
    df_valve = pd.DataFrame(
        model.components["office_valve_position_sensor"]
        .output["measuredValue"]
        .history()[:, 0, 0]
        .detach()
        .cpu()
        .numpy(),
        index=simulator.date_time_steps[0],
    )
    valve_position_schedule = tb.SensorSystem(df=df_valve, id="ValvePositionSchedule")

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

    # Load real Elspot prices from Dec 11-13, 2024 and remap dates to Dec 1-3, 2023
    elspot_raw = pd.read_csv(
        utils.get_path(["Elspotprices2023-2025.csv"]), sep=";", decimal=","
    )
    elspot_raw["HourDK"] = pd.to_datetime(elspot_raw["HourDK"], dayfirst=True)

    src_start = pd.Timestamp("2024-12-11")
    src_end = pd.Timestamp("2024-12-14")  # exclusive
    mask = (elspot_raw["HourDK"] >= src_start) & (elspot_raw["HourDK"] < src_end)
    elspot_subset = elspot_raw.loc[mask, ["HourDK", "SpotPriceDKK"]].copy()

    date_offset = pd.DateOffset(years=1, days=10)  # 2024-12-11 - 1y10d = 2023-12-01
    elspot_subset["HourDK"] = elspot_subset["HourDK"] - date_offset
    elspot_subset.columns = ["time", "value"]
    elspot_subset["value"] = elspot_subset["value"] / 1000  # DKK/MWh -> DKK/kWh
    elspot_subset = elspot_subset.sort_values("time").reset_index(drop=True)

    elspot_clean_path = utils.get_path(["estimator_example", "electricity_price.csv"])
    elspot_subset.to_csv(elspot_clean_path, index=False)
    print(f"Using real Elspot prices from Dec 11-13, 2024 (remapped to Dec 1-3, 2023)")
    print(
        f"Price range: {elspot_subset['value'].min():.4f} - {elspot_subset['value'].max():.4f} DKK/kWh"
    )
    print(elspot_subset.head())

    price_schedule = tb.ScheduleSystem(
        filename=elspot_clean_path, date_column=0, value_column=1, id="price_schedule"
    )

    # Multiplies Power (W) by price (DKK/kWh), with scale_factor converting W*s to kWh
    costs_sensor = tb.ScalarProductSystem(
        scale_factor=step_size / 3600 / 1000, id="costs_sensor"  # W -> kWh per timestep
    )

    heating_setpoint = model.components["office_temperature_heating_setpoint"]

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

    # --- 3.2 Rewire the Model for Optimization ---
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
    print(model)

    # --- 3.3 Set Up Optimization Parameters ---
    opt_step_size = step_size
    opt_start_time = [
        datetime.datetime(
            year=2023,
            month=12,
            day=1,
            hour=0,
            minute=0,
            second=0,
            tzinfo=tz.gettz("Europe/Copenhagen"),
        ),
    ]
    opt_end_time = [
        datetime.datetime(
            year=2023,
            month=12,
            day=4,
            hour=0,
            minute=0,
            second=0,
            tzinfo=tz.gettz("Europe/Copenhagen"),
        ),
    ]

    print(f"Optimization period: {opt_start_time[0]} to {opt_end_time[0]} (3 days)")
    print(f"Step size: {opt_step_size} seconds ({opt_step_size / 60} minutes)")
    print(f"Using real Elspot prices from Dec 11-13, 2024")

    # --- 3.4 Run Initial Simulation (Before Optimization) ---
    simulator_opt = tb.Simulator(model)
    simulator_opt.simulate(
        step_size=opt_step_size, start_time=opt_start_time, end_time=opt_end_time
    )

    fig, axes = tb.plot.plot(
        simulator_opt.date_time_steps,
        [
            tb.plot.Entry(
                space.output["indoorTemperature"].history(), label="Indoor temperature"
            ),
            tb.plot.Entry(
                heating_setpoint.output["scheduleValue"].history(),
                label="Heating setpoint",
            ),
            tb.plot.Entry(
                space_heater.output["Power"].history(),
                label="Space heater power",
                axis=2,
            ),
            tb.plot.Entry(
                space_heater_valve.output["valvePosition"].history(),
                label="Valve position",
                axis=3,
            ),
        ],
        ylabel_1axis=r"Temperature [$^\circ$C]",
        ylabel_2axis="Heating Power [W]",
        ylabel_3axis="Valve Position [0-1]",
        title="Before optimization (Dec 1-3, 2023)",
        show=False,
    )

    # --- 3.5 Run Optimization ---
    variables = [(valve_position_schedule, "scheduleValue", 0, 1)]

    objectives = [(costs_sensor, "output", "min")]

    ineq_cons = [
        (space, "indoorTemperature", "upper", cooling_setpoint),
        (space, "indoorTemperature", "lower", heating_setpoint),
    ]

    optimizer = tb.Optimizer(simulator_opt)

    opt_options = {"maxiter": 300, "tol": 1e-15, "disp": True}

    optimizer.optimize(
        start_time=opt_start_time,
        end_time=opt_end_time,
        step_size=opt_step_size,
        variables=variables,
        objectives=objectives,
        eq_cons=None,
        ineq_cons=ineq_cons,
        method="scipy",
        options=opt_options,
    )

    # --- 3.6 Save & Plot Optimization Results ---
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
        "step_size": opt_step_size,
        "opt_start_time": opt_start_time,
        "opt_end_time": opt_end_time,
    }

    results_dir = utils.get_path(
        ["generated_files", "models", "full_workflow_example", "optimization_results"]
    )
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(results_dir, f"{timestamp}_optimization.pickle")
    with open(results_path, "wb") as f:
        pickle.dump(opt_results, f)
    print(f"Optimization results saved to: {results_path}")

    cost_1d = opt_results["cost_per_step"][:, 0, 0]
    print(f"Total cost: {cost_1d.sum():.2f} DKK")
    print(f"Peak cost/step: {cost_1d.max():.4f} DKK")
    print(f"Peak power: {opt_results['power'][:, 0, 0].max():.1f} W")

    fig, axes = tb.plot.plot(
        simulator_opt.date_time_steps,
        [
            tb.plot.Entry(
                opt_results["indoor_temperature"], label="Indoor temperature"
            ),
            tb.plot.Entry(opt_results["heating_setpoint"], label="Heating setpoint"),
            tb.plot.Entry(opt_results["power"], label="Space heater power", axis=2),
            tb.plot.Entry(
                opt_results["valve_position"], label="Valve position", axis=3
            ),
            tb.plot.Entry(
                opt_results["electricity_price"],
                label="Electricity price",
                fmt="--",
                axis=3,
            ),
        ],
        ylabel_1axis=r"Temperature [$^\circ$C]",
        ylabel_2axis="Power [W]",
        ylabel_3axis="Valve [0-1] / Price [DKK/kWh]",
        title="Optimization results",
        show=True,
    )


if __name__ == "__main__":
    main()
