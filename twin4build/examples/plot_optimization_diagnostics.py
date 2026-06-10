"""
Optimization diagnostics: re-simulate with the optimized valve schedule
and plot all available model outputs.

Loads the calibrated model, extracts the optimized valve-position time series
from the optimization pickle, feeds it back as a ScheduleSystem, and runs a
forward simulation so every internal variable is available for plotting.

Usage:
    python plot_optimization_diagnostics.py
"""

# Standard library imports
import datetime
import os
import pickle
import tempfile

# Third party imports
import matplotlib.pyplot as plt
import pandas as pd
from dateutil import tz
from plot_calibration_results import add_hourly_ticks, compute_metrics, move_legend

# Local application imports
import twin4build as tb
import twin4build.examples.utils as utils

# ── Configuration ─────────────────────────────────────────────────────────
ESTIMATION_PICKLE = (
    r"generated_files\models\full_workflow_example"
    r"\model_parameters\estimation_results"
    r"\20260301_104344_scipy_SLSQP_ad.pickle"
)

OPT_PICKLE = (
    r"generated_files\models\full_workflow_example"
    r"\optimization_results"
    r"\20260304_083429_optimization_min_energy.pickle"
)

STEP_SIZE = 1200  # must match the optimization run


def _valve_schedule_csv(opt_result, tmp_dir):
    """Write the optimized valve position to a CSV that ScheduleSystem can load."""
    dt = opt_result["date_time_steps"]
    dt = dt.flatten().tolist()
    valve = opt_result["valve_position"][:, 0, 0].detach().cpu()

    print(f"  Valve from pickle: shape={opt_result['valve_position'].shape}")
    print(f"  min={valve.min():.6f}  max={valve.max():.6f}  mean={valve.mean():.6f}")
    print(f"  First 10 values: {valve[:10].tolist()}")
    print(
        f"  Unique values (up to 20): {sorted(set(round(v, 6) for v in valve.tolist()))[:20]}"
    )

    df = pd.DataFrame({"time": dt, "value": valve.tolist()})
    path = os.path.join(tmp_dir, "optimized_valve_schedule.csv")
    df.to_csv(path, index=False)
    return path


def main():
    # ── 1. Load optimization pickle ───────────────────────────────────────
    opt_path = utils.get_path([OPT_PICKLE])
    with open(opt_path, "rb") as f:
        opt = pickle.load(f)

    opt_start_time = opt["opt_start_time"]
    opt_end_time = opt["opt_end_time"]
    print(f"Optimization period: {opt_start_time[0]} → {opt_end_time[0]}")

    # ── 2. Export optimized valve schedule as CSV ─────────────────────────
    tmp_dir = tempfile.mkdtemp(prefix="t4b_opt_diag_")
    valve_csv = _valve_schedule_csv(opt, tmp_dir)
    print(f"Valve schedule CSV: {valve_csv}")

    # ── 3. Load model & apply calibrated parameters ───────────────────────
    model = tb.Model(id="full_workflow_example")
    filename_simulation, _ = model._simulation_model._semantic_model.get_dir(
        filename="instance_graph.ttl"
    )
    model.load(simulation_model_filename=filename_simulation)

    est_path = utils.get_path([ESTIMATION_PICKLE])
    model.load_estimation_result(filename=est_path, verbose=1)
    print("Calibrated model loaded.")

    # ── 4. Rewire: replace heating controller with optimized schedule ─────
    heating_controller = model.components["office_temperature_heating_controller"]
    space_heater_valve = model.components["office_space_heater_valve"]

    valve_position_schedule = tb.ScheduleSystem(
        filename=valve_csv,
        datecolumn=0,
        valuecolumn=1,
        id="valve_position_schedule",
    )

    model.remove_connection(
        heating_controller, space_heater_valve, "inputSignal", "valvePosition"
    )
    model.add_connection(
        valve_position_schedule, space_heater_valve, "scheduleValue", "valvePosition"
    )
    model.load()
    print("Model rewired with optimized valve schedule.")

    # ── 5. Simulate ───────────────────────────────────────────────────────
    model.set_save_simulation_result(flag=True)
    simulator = tb.Simulator(model)
    simulator.simulate(
        step_size=STEP_SIZE, start_time=opt_start_time, end_time=opt_end_time
    )
    print("Simulation complete.\n")

    # ── 6. Extract signals ────────────────────────────────────────────────
    dt = simulator.date_time_steps
    space = model.components["office"]
    space_heater = model.components["office_space_heater"]

    indoor_temp = space.output["indoorTemperature"].history()
    wall_temp = space.output["wallTemperature"].history()
    outdoor_temp = space.input["outdoorTemperature"].history()
    supply_air_t = space.input["supplyAirTemperature"].history()
    supply_air_q = space.input["supplyAirFlowRate"].history()
    n_people = space.input["numberOfPeople"].history()
    heat_gain = space.input["heatGain"].history()
    irradiation = space.input["globalIrradiation"].history()

    heater_power = space_heater.output["Power"].history()
    water_t_out = space_heater.output["outletWaterTemperature"].history()
    water_t_in = space_heater.input["supplyWaterTemperature"].history()
    water_flow = space_heater.input["waterFlowRate"].history()

    valve_pos = space_heater_valve.output["valvePosition"].history()
    valve_flow = space_heater_valve.output["waterFlowRate"].history()

    temp_sensor = model.components["office_temperature_sensor"]
    temp_meas = temp_sensor.time_series_input.values
    temp_sim = temp_sensor.output["measuredValue"].history()

    co2_sensor = model.components["office_co2_sensor"]
    co2_meas = co2_sensor.time_series_input.values
    co2_sim = co2_sensor.output["measuredValue"].history()

    valve_sensor = model.components["office_valve_position_sensor"]
    valve_meas = valve_sensor.time_series_input.values

    damper_sensor = model.components["office_damper_position_sensor"]
    damper_meas = damper_sensor.time_series_input.values
    damper_sim = damper_sensor.output["measuredValue"].history()

    heating_sp = heating_controller.input["setpointValue"].history()
    occ_signal = (
        model.components["office_occupancy_detector"]
        .output["occupancySignal"]
        .history()
    )

    # ── Quick summary ─────────────────────────────────────────────────────
    power_1d = heater_power[:, 0, 0]
    energy_kwh = power_1d.sum() * STEP_SIZE / 3600 / 1000
    print("=" * 80)
    print(f"  Total energy:    {energy_kwh:.2f} kWh")
    print(f"  Peak power:      {power_1d.max():.1f} W")
    print(
        f"  Temp range:      {indoor_temp[:, 0, 0].min():.2f} – {indoor_temp[:, 0, 0].max():.2f} °C"
    )
    print("=" * 80 + "\n")

    # =====================================================================
    # ── 7. PLOTS — Customize freely ──────────────────────────────────────
    # =====================================================================
    C = tb.plot.Colors
    tz_cph = tz.gettz("Europe/Copenhagen")

    # --- Plot 1: Temperature & valve position (with measured overlay) ---
    fig, axes = tb.plot.plot(
        dt,
        [
            tb.plot.Entry(
                indoor_temp, label="Indoor temp (opt)", color="#d95f02", linewidth=2
            ),
            tb.plot.Entry(
                temp_meas,
                label="Indoor temp (meas)",
                color=C.green,
                linewidth=1.5,
                fmt="--",
            ),
            tb.plot.Entry(
                valve_pos,
                label="Valve position (opt)",
                color="#e7298a",
                linewidth=2,
                axis=2,
            ),
            tb.plot.Entry(
                valve_meas,
                label="Valve position (meas)",
                color="#1f78b4",
                linewidth=1.5,
                fmt="--",
                axis=2,
            ),
            tb.plot.Entry(
                heating_sp,
                label="Heating setpoint",
                color=C.black,
                linewidth=1.5,
                fmt=":",
            ),
        ],
        ylim_1axis=(17, 25),
        ylim_2axis=(0, 1.2),
        ylabel_1axis=r"Temperature [$^\circ$C]",
        ylabel_2axis="Valve position [0-1]",
        show=False,
        nticks=7,
    )
    add_hourly_ticks(axes, tz_info=tz_cph)
    move_legend(fig, y=1.01, fontsize=10, ncol=3)
    fig.set_size_inches(12, 4.5)
    fig.savefig("diag_temp_valve.png", dpi=300, bbox_inches="tight")

    # --- Plot 2: Heater power & water circuit ---
    fig, axes = tb.plot.plot(
        dt,
        [
            tb.plot.Entry(
                heater_power, label="Heater power", color="#d95f02", linewidth=2
            ),
            tb.plot.Entry(
                water_flow,
                label="Water flow rate",
                color="#7570b3",
                linewidth=1.5,
                axis=2,
            ),
            tb.plot.Entry(
                water_t_in,
                label="Water supply temp",
                color="#1b9e77",
                linewidth=1.5,
                fmt="--",
                axis=3,
            ),
            tb.plot.Entry(
                water_t_out,
                label="Water return temp",
                color="#e7298a",
                linewidth=1.5,
                fmt="--",
                axis=3,
            ),
        ],
        ylabel_1axis="Power [W]",
        ylabel_2axis="Flow rate [kg/s]",
        ylabel_3axis=r"Water temp [$^\circ$C]",
        show=False,
        nticks=7,
    )
    add_hourly_ticks(axes, tz_info=tz_cph)
    move_legend(fig, y=1.01, fontsize=10, ncol=4)
    fig.set_size_inches(12, 4.5)
    fig.savefig("diag_power_water.png", dpi=300, bbox_inches="tight")

    # --- Plot 3: Ventilation & CO2 ---
    fig, axes = tb.plot.plot(
        dt,
        [
            tb.plot.Entry(co2_sim, label=r"CO$_2$ (sim)", color="#d95f02", linewidth=2),
            tb.plot.Entry(
                co2_meas, label=r"CO$_2$ (meas)", color=C.green, linewidth=1.5, fmt="--"
            ),
            tb.plot.Entry(
                supply_air_q,
                label="Supply air flow",
                color="#7570b3",
                linewidth=1.5,
                axis=2,
            ),
            tb.plot.Entry(
                damper_sim, label="Damper (sim)", color="#e7298a", linewidth=1.5, axis=3
            ),
            tb.plot.Entry(
                damper_meas,
                label="Damper (meas)",
                color="#1f78b4",
                linewidth=1.5,
                fmt="--",
                axis=3,
            ),
        ],
        ylabel_1axis=r"CO$_2$ [ppmv]",
        ylabel_2axis="Air flow [kg/s]",
        ylabel_3axis="Damper position [0-1]",
        show=False,
        nticks=7,
    )
    add_hourly_ticks(axes, tz_info=tz_cph)
    move_legend(fig, y=1.01, fontsize=10, ncol=3)
    fig.set_size_inches(12, 4.5)
    fig.savefig("diag_ventilation_co2.png", dpi=300, bbox_inches="tight")

    # --- Plot 4: Disturbances (weather, occupancy, solar) ---
    fig, axes = tb.plot.plot(
        dt,
        [
            tb.plot.Entry(
                outdoor_temp, label="Outdoor temp", color="#1b9e77", linewidth=1.5
            ),
            tb.plot.Entry(
                irradiation,
                label="Solar irradiation",
                color="#e6ab02",
                linewidth=1.5,
                axis=2,
            ),
            tb.plot.Entry(
                n_people,
                label="Occupancy (people)",
                color="#7570b3",
                linewidth=1.5,
                axis=3,
            ),
        ],
        ylabel_1axis=r"Outdoor temp [$^\circ$C]",
        ylabel_2axis=r"Irradiation [W/m$^2$]",
        ylabel_3axis="Number of people",
        show=False,
        nticks=7,
    )

    print(irradiation)
    add_hourly_ticks(axes, tz_info=tz_cph)
    move_legend(fig, y=1.01, fontsize=10, ncol=3)
    fig.set_size_inches(12, 4.5)
    fig.savefig("diag_disturbances.png", dpi=300, bbox_inches="tight")

    plt.show()
    print("All diagnostic plots saved.")


if __name__ == "__main__":
    main()
