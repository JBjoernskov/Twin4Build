"""
Twin4Build: Full Workflow Example

Demonstrates the complete Twin4Build workflow on a single model:

1. Semantic Model -> Simulation Model (Translator): Load a semantic description
   from an Excel file and automatically generate a simulation model
2. Simulation Model -> Calibrated Model (Estimator): Calibrate the simulation
   model parameters against real sensor data
3. Calibrated Model -> Optimized Control (Optimizer): Optimize the valve position
   schedule to minimize electricity cost (Danish Elspot prices) while maintaining
   thermal comfort — matching ``full_workflow_example.ipynb``.

Calibration uses a two-stage estimator warm-start (fast SciPy SLSQP, then
CasADi/IPOPT collocation), as in the notebook.
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
from twin4build.utils.rgetattr import rgetattr

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

    # Original DP37 PLC exports (room OD095_01_011A / HF04), packaged under
    # full_workflow_example/ and resolved via utils.get_path (same pattern as
    # estimator_example — not the estimator CSVs themselves).
    self.components["office_temperature_sensor"].filename = utils.get_path(
        ["full_workflow_example", "temperature_sensor.csv"]
    )
    self.components["office_temperature_sensor"].datecolumn = 2
    self.components["office_temperature_sensor"].valuecolumn = 4

    self.components["office_co2_sensor"].filename = utils.get_path(
        ["full_workflow_example", "co2_sensor.csv"]
    )
    self.components["office_co2_sensor"].datecolumn = 2
    self.components["office_co2_sensor"].valuecolumn = 4

    self.components["office_valve_position_sensor"].filename = utils.get_path(
        ["full_workflow_example", "valve_position_sensor.csv"]
    )
    self.components["office_valve_position_sensor"].datecolumn = 2
    self.components["office_valve_position_sensor"].valuecolumn = 4

    self.components["office_damper_position_sensor"].filename = utils.get_path(
        ["full_workflow_example", "damper_position_sensor.csv"]
    )
    self.components["office_damper_position_sensor"].datecolumn = 2
    self.components["office_damper_position_sensor"].valuecolumn = 4

    self.components["supply_air_temperature_sensor"].filename = utils.get_path(
        ["full_workflow_example", "supply_air_temperature.csv"]
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
        co2_filename=utils.get_path(["full_workflow_example", "co2_sensor.csv"]),
        co2_date_column=2,
        co2_value_column=4,
        damper_filename=utils.get_path(
            ["full_workflow_example", "damper_position_sensor.csv"]
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
    #
    # The threshold must sit between the unoccupied noise floor and the
    # occupied-hours signal -- but where that is depends on ``mass.G_occ``,
    # which is ESTIMATED: the occupancy inverse-model divides the CO2 balance
    # by G_occ, so N_occ scales as 1/G_occ.  A hard-coded threshold is
    # therefore only valid for one particular G_occ, and the CO2/temperature
    # data pull G_occ to the physical ~5e-6 kg/s/person no matter where it
    # starts.  Threshold and G_occ are identifiable only as a *pair*, so the
    # threshold is estimated alongside it (see the parameter list in main()).
    # This value is just the starting point.
    #
    # ``steepness`` is not merely cosmetic.  ``SigmoidGate`` is a clamped
    # linear ramp of width 1/steepness with a power-law tail, so the window in
    # which the gate has usable gradient is +-1/steepness occupants wide.  At
    # steepness=30 the threshold itself becomes hard to move (measured: it
    # crawls 0.15 -> 0.085 and the collocation NLP hits its iteration limit);
    # at steepness=10 it travels 1.0 -> 0.38 and converges.  Keep it low
    # enough that the threshold stays estimable.
    occupancy_detector = tb.OccupancyDetectorSystem(
        threshold=1,
        steepness=10,
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
        ["full_workflow_example", "temperature_heating_setpoint.csv"]
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

    # Space heater UA is computed from nominal conditions during initialize()
    # (initialize_UA=True). Omit x0 so estimation starts from that solved value.
    self.components["office_space_heater"].initialize_UA = True


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
        (boundary_wall, "C", 1e6, 1e4, 1e7),
        (space, "thermal.R_out", 0.5, 0.01, 1),
        (space, "thermal.R_in", 0.1, 0.01, 1),
        (boundary_wall, "R_a", 0.04, 1e-4, 1),
        (boundary_wall, "R_b", 0.04, 1e-4, 1),
        (space, "thermal.f_wall", 0.1, 0, 10),
        (space, "thermal.f_air", 0.1, 0, 10),
        (space, "thermal.Q_occ_gain", 100.0, 10, 200),
        # Space heater parameters
        (space_heater, "thermalMassHeatCapacity", 1e4, 1e3, 2e5),
        (space_heater, "UA", None, 1, 100),  # x0 from initialize_UA
        # Controller PID parameters (private = each controller gets its own values)
        (heating_controller, "kp", 0.005, 1e-5, 1, "private"),
        (co2_controller, "kp", 0.0001, 1e-5, 1, "private"),
        ([heating_controller, co2_controller], "Ti", 30, 1, 300, "private"),
        ([heating_controller, co2_controller], "Td", 0, 0, 1, "private"),
        # Valve parameters
        (space_heater_valve, "waterFlowRateMax", 0.001, 1e-6, 0.1),
        (space_heater_valve, "valveAuthority", 1, 0.4, 1),
        # Damper parameters -- shared between model dampers and occupancy's internal dampers
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
        # Mass-balance / occupancy parameters (shared between space and
        # occupancy estimator).
        ([space, occupancy_system], "mass.V", 65, 50, 80, "shared"),
        ([space, occupancy_system], "mass.G_occ", 1e-6, 1e-6, 1e-5, "shared"),
        ([space, occupancy_system], "mass.m_inf", 0.001, 1e-4, 0.01, "shared"),
        # Occupancy-detector threshold -- MUST be estimated together with
        # G_occ, which sets the scale of the inferred occupancy it is compared
        # against (N_occ ~ 1/G_occ).  Leaving it fixed makes the ventilation
        # branch a degenerate direction of the objective: the solver moves
        # G_occ to fit CO2/temperature, N_occ slides out from under the fixed
        # threshold, the gate saturates, and the damper prediction collapses
        # to "always off" -- at almost no cost in the pooled objective, so
        # nothing pushes back.  Once saturated the gate's gradient is ~4
        # orders of magnitude down (power-law tail), so no solver recovers it.
        #
        # Measured over the two-stage estimation on Dec 2-7 (damper RMSE):
        #   threshold fixed at 1.0    -> 0.164   (gate never fires)
        #   threshold fixed at 0.15   -> 0.108   (gate chatters: FP 11%)
        #   threshold estimated       -> 0.076   (fits 0.384; FP 0.9%, FN 6.5%)
        # Estimating it also lowers the POOLED objective 69.7 -> 41.2 and
        # recovers G_occ = 5.0e-6, i.e. the textbook per-person CO2 generation
        # rate, without needing hand-tightened bounds to force it there.
        (occupancy_detector, "threshold", 1.0, 0.02, 5.0),
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
        if val is None:
            # Omitted x0: leave the component's current / initialize()-computed value.
            continue
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
        show=False,
        nticks=11,
    )

    # --- 2.7 Run Parameter Estimation ---
    #
    # SciPy SLSQP single-shooting, run to convergence.  This is the path that
    # produced the published results.
    #
    # ``USE_COLLOCATION_REFINEMENT`` additionally runs the CasADi/IPOPT
    # collocation stage as a stage-2 refinement.  It is OFF by default because
    # on this model it makes every sensor's fit worse, not better.  Measured on
    # Dec 2-7 with an already-converged (100-iteration) SLSQP warm start:
    #
    #                       temp    valve   damper   CO2    pooled*
    #   after SLSQP        0.190    0.104    0.022   14.3    33.3
    #   + collocation      0.167    0.100    0.167   15.2    72.9
    #
    #   *pooled = sum over sensors of (RMSE/sd)^2, i.e. the quantity BOTH
    #    stages minimize -- so stage 2 returns a point twice as bad as the one
    #    it was handed, and reports ``Solved_To_Acceptable_Level``.
    #
    # The damper is what collapses (the occupancy gate switches fully off:
    # true-positive rate 0.344 -> 0.003).  Two independent causes, both in the
    # estimator rather than in this example:
    #
    #   1. The collocation warm start is corrupted before IPOPT ever runs.
    #      ``_transcription.py`` seeds boundary states as ``ACT / coeff`` with
    #      ``coeff = d(meas_t)/d(y_t)`` -- a one-step transition factor (0.79
    #      here), not a readout gain -- so every segment's air-temperature
    #      state starts ~5.6 K too hot.  Measured at the handoff: objective
    #      1875.8 and max|defect| 7.7, against 8.2 / 7e-5 for the same warm
    #      start with the seeding disabled (TWIN4BUILD_NO_DATA_WARMSTART=1).
    #   2. Termination is effectively optimality-free: with the Gauss-Newton
    #      Hessian on (the default) the module sets acceptable_tol=1e3,
    #      acceptable_iter=5, acceptable_compl_inf_tol=1e3, leaving "feasible
    #      and objective stagnant for 5 iterations" as the only exit -- which
    #      a flat ridge satisfies while still sliding sideways.
    #
    # Neither knob rescues it: tightening the tolerances alone gets pooled 59
    # instead of 73; disabling the seeding alone stalls infeasible; doing both
    # diverges outright (max|defect| 13.9 after 1500 iterations).  Turn this on
    # only to exercise the collocation code path, not to improve a fit.
    USE_COLLOCATION_REFINEMENT = True

    estimator = tb.Estimator(simulator)

    result = estimator.estimate(
        start_time,
        end_time,
        step_size,
        parameters,
        measurements,
        n_warmup=20,
        method=("scipy", "SLSQP", "ad"),
        # maxiter=5 is NOT converged: it leaves the pooled objective at ~70
        # where ~29 is reachable, and the extra iterations are cheap on the
        # ``fast`` composed-rollout path.
        options={"maxiter": 5, "fast": True},
    )

    if USE_COLLOCATION_REFINEMENT:
        # Continue from stage 1's optimum.  estimate() leaves the model's
        # parameters at the fitted values, so read each group's current value
        # straight off the model.  (Do NOT zip the input list with
        # stage1["result_x"]: the estimator reorders parameters -- private
        # first, then shared -- expands private component lists and collapses
        # shared groups, so result_x is not aligned with `parameters`.)
        def _from_model(entry):
            comps, attr, _x0, lo, hi = entry[:5]
            comp = comps[0] if isinstance(comps, list) else comps
            v = float(rgetattr(comp, attr).get().reshape(-1)[0])
            eps = 1e-9 * (hi - lo)  # nudge off the bounds for the x0 checks
            return (comps, attr, min(max(v, lo + eps), hi - eps), lo, hi, *entry[5:])

        parameters_stage2 = [_from_model(entry) for entry in parameters]

        result = estimator.estimate(
            start_time,
            end_time,
            step_size,
            parameters_stage2,
            measurements,
            n_warmup=20,
            method=("casadi", "ipopt", "ad", "collocation"),
            options={
                "maxiter": 600,
                # This is a REFINEMENT of a converged single-shooting solution,
                # so start ON that solution's trajectory instead of on data
                # planted into the boundary states.  Measured here at the
                # handoff: data_warmstart=True (the cold-start default) begins
                # at max|defect| 4.9 with the fit it is meant to refine already
                # lost, against 3.4e-5 unseeded -- i.e. stage 1's own
                # trajectory, which is the point of a warm start.
                "data_warmstart": False,
                # With that feasible warm start, early stopping's
                # best-feasible-iterate checkpoint adopts stage 1's solution as
                # its incumbent -- so this stage can improve on stage 1 or
                # leave it alone, but not return something worse.
                "early_stopping": True,
            },
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

    # Collocation estimates the trajectory's boundary states along with theta, so
    # the reported RMSEs are for a simulation STARTING FROM the estimated initial
    # state.  A plain simulate() would instead start from the component defaults
    # (e.g. wall temperature 20 degC) -- with day-scale wall time constants that
    # initial-condition error biases the whole horizon.  Seed the estimated
    # initial state to reproduce the fitted trajectory.
    #
    # Single-shooting carries no estimated state (it rolls out from the defaults,
    # which is exactly what its own objective scored), so seeding is a no-op
    # there -- hence .get() rather than [...].
    _init_state = result.get("estimated_initial_state") or {}

    def _seed_estimated_initial_state():
        # get_component: the office+wall pair executes as one fused state-space
        # block, so the state keys are executing-component ids.
        for comp_id, x0 in _init_state.items():
            model.get_component(comp_id).set_state(x0)  # (n_periods, n_c, state_size)

    simulator.simulate(
        step_size=step_size,
        start_time=start_time,
        end_time=end_time,
        after_initialize=_seed_estimated_initial_state,
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

    # Packaged Elspot subset (Dec 11-13 2024 remapped to Dec 1-3 2023), DKK/kWh
    elspot_clean_path = utils.get_path(["estimator_example", "electricity_price.csv"])
    elspot_subset = pd.read_csv(elspot_clean_path)
    print("Using packaged Elspot prices (Dec 11-13 2024 remapped to Dec 1-3 2023)")
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

    opt_options = {"maxiter": 300, "tol": 1e-15, "disp": True, "fast": True}

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
