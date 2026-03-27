"""
Publication-quality calibration plots.

Loads a saved estimation result (.pickle), applies the calibrated parameters
to the model, runs a forward simulation, and produces customizable plots.

Usage:
    python plot_calibration_results.py
"""

# Standard library imports
import datetime

# Third party imports
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import torch
from dateutil import tz

# Local application imports
import twin4build as tb
import twin4build.examples.utils as utils


def add_hourly_ticks(axes, tz_info, hour_interval=6):
    """Add day-level major ticks and hour-level minor ticks to all axes."""
    for ax in axes:
        ax.xaxis.set_major_locator(mdates.DayLocator(tz=tz_info))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %b %d", tz=tz_info))
        ax.tick_params(axis="x", which="major", pad=12)
        ax.xaxis.set_minor_locator(
            mdates.HourLocator(interval=hour_interval, tz=tz_info)
        )
        ax.xaxis.set_minor_formatter(mdates.DateFormatter("%H", tz=tz_info))
        ax.tick_params(axis="x", which="minor", labelsize=12)


def compute_metrics(measured, simulated, name=""):
    """Compute error metrics between measured and simulated time series."""
    if isinstance(measured, torch.Tensor):
        measured = measured.detach().cpu().numpy()
    if isinstance(simulated, torch.Tensor):
        simulated = simulated.detach().cpu().numpy()
    measured = measured.flatten()
    simulated = simulated.flatten()

    mask = ~(np.isnan(measured) | np.isnan(simulated))
    m, s = measured[mask], simulated[mask]

    err = s - m
    rmse = np.sqrt(np.mean(err**2))
    mae = np.mean(np.abs(err))
    me = np.mean(err)
    r2 = 1 - np.sum(err**2) / np.sum((m - np.mean(m)) ** 2)
    cvrmse = rmse / np.mean(np.abs(m)) * 100

    print(
        f"  {name:<30s}  RMSE={rmse:8.4f}  MAE={mae:8.4f}  ME={me:+8.4f}  R²={r2:.4f}  CV-RMSE={cvrmse:.1f}%"
    )
    return {"rmse": rmse, "mae": mae, "me": me, "r2": r2, "cvrmse": cvrmse}


def move_legend(
    fig,
    y=1.0,
    fontsize=11,
    labelspacing=0.2,
    handlelength=1.5,
    columnspacing=1.0,
    ncol=3,
):
    """Remove the default legend and recreate it with tighter spacing."""
    old = fig.legends[0]
    handles = old.legend_handles
    labels = [t.get_text() for t in old.get_texts()]
    old.remove()
    fig.legend(
        handles,
        labels,
        ncol=ncol,
        bbox_to_anchor=(0.5, y),
        loc="upper center",
        fontsize=fontsize,
        labelspacing=labelspacing,
        handlelength=handlelength,
        columnspacing=columnspacing,
    )


# ── Estimation result to load ────────────────────────────────────────────
RESULT_PICKLE = (
    r"generated_files\models\full_workflow_example"
    r"\model_parameters\estimation_results"
    r"\20260301_104344_scipy_SLSQP_ad.pickle"
)

RESULT_PICKLE = (
    r"generated_files\models\full_workflow_example"
    r"\model_parameters\estimation_results"
    r"\20260303_103841_scipy_SLSQP_ad.pickle"
)


def main():
    # ── 1. Load model from serialized RDF ────────────────────────────────
    model = tb.Model(id="full_workflow_example")
    filename_simulation, _ = model._simulation_model._semantic_model.get_dir(
        filename="instance_graph.ttl"
    )
    model.load(simulation_model_filename=filename_simulation)
    model.simulation_model.visualize(
        format="png", include_full_uri=False, literals=False, dpi=800, compressed=True
    )

    # ── 2. Apply calibrated parameters from pickle ───────────────────────
    result_path = utils.get_path([RESULT_PICKLE])
    model.load_estimation_result(filename=result_path, verbose=1)

    # ── 3. Simulate ──────────────────────────────────────────────────────
    simulator = tb.Simulator(model)
    step_size = 1200

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

    model.set_save_simulation_result(flag=True)
    simulator.simulate(step_size=step_size, start_time=start_time, end_time=end_time)
    print("Simulation with calibrated parameters complete.")

    # ── 5. Convenience aliases ───────────────────────────────────────────
    dt = simulator.date_time_steps
    temp_meas = model.components["office_temperature_sensor"].time_series_input.values
    temp_sim = (
        model.components["office_temperature_sensor"].output["measuredValue"].history()
    )
    co2_meas = model.components["office_co2_sensor"].time_series_input.values
    co2_sim = model.components["office_co2_sensor"].output["measuredValue"].history()
    valve_meas = model.components[
        "office_valve_position_sensor"
    ].time_series_input.values
    valve_sim = (
        model.components["office_valve_position_sensor"]
        .output["measuredValue"]
        .history()
    )
    damper_meas = model.components[
        "office_damper_position_sensor"
    ].time_series_input.values
    damper_sim = (
        model.components["office_damper_position_sensor"]
        .output["measuredValue"]
        .history()
    )
    heating_setpoint = (
        model.components["office_temperature_heating_controller"]
        .input["setpointValue"]
        .history()
    )
    n_people = model.components["office"].input["numberOfPeople"].history()
    occ_signal = (
        model.components["office_occupancy_detector"]
        .output["occupancySignal"]
        .history()
    )

    # ── Calibration metrics ─────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("  Calibration Metrics (measured vs. simulated)")
    print("=" * 90)
    compute_metrics(temp_meas, temp_sim, "Temperature [°C]")
    compute_metrics(co2_meas, co2_sim, "CO₂ [ppm]")
    compute_metrics(valve_meas, valve_sim, "Valve position [0-1]")
    compute_metrics(damper_meas, damper_sim, "Damper position [0-1]")
    print("=" * 90 + "\n")

    # =====================================================================
    # ── 6. PLOTS — Customize everything below for your publication ───────
    # =====================================================================

    C = tb.plot.Colors

    # --- Plot 1: Temperature & valve position ---
    fig, axes = tb.plot.plot(
        dt,
        [
            tb.plot.Entry(
                temp_sim, label="Temperature simulated", color="#d95f02", linewidth=2
            ),
            tb.plot.Entry(
                temp_meas,
                label="Temperature measured",
                color=C.green,
                linewidth=1.5,
                fmt="--",
            ),
            tb.plot.Entry(
                valve_sim, label="Valve simulated", color="#e7298a", linewidth=2, axis=2
            ),
            tb.plot.Entry(
                valve_meas,
                label="Valve measured",
                color="#1f78b4",
                linewidth=1.5,
                fmt="--",
                axis=2,
            ),
            tb.plot.Entry(
                heating_setpoint,
                label="Heating setpoint",
                color=C.black,
                linewidth=1.5,
                fmt=":",
            ),
        ],
        ylim_1axis=(18.4, 23.6),
        ylim_2axis=(0, 1.5),
        ylabel_1axis=r"Temperature [$^\circ$C]",
        ylabel_2axis="Valve position [0-1]",
        # title="Temperature & valve (calibrated)",
        show=False,
        nticks=7,
    )
    add_hourly_ticks(axes, tz_info=tz.gettz("Europe/Copenhagen"))
    move_legend(fig, y=1.01, fontsize=11)
    fig.set_size_inches(11, 4)
    fig.savefig("cal_t_v.png", dpi=400, bbox_inches="tight")

    # --- Plot 2: CO2 & damper position ---
    fig, axes = tb.plot.plot(
        dt,
        [
            tb.plot.Entry(
                co2_sim, label=r"CO$_2$ simulated", color="#d95f02", linewidth=2
            ),
            tb.plot.Entry(
                co2_meas,
                label=r"CO$_2$ measured",
                color=C.green,
                linewidth=1.5,
                fmt="--",
            ),
            tb.plot.Entry(
                damper_sim,
                label="Damper simulated",
                color="#e7298a",
                linewidth=2,
                axis=2,
            ),
            tb.plot.Entry(
                damper_meas,
                label="Damper measured",
                color="#1f78b4",
                linewidth=1.5,
                fmt="--",
                axis=2,
            ),
            tb.plot.Entry(
                occ_signal,
                label="Occupancy signal (binary)",
                color=C.black,
                linewidth=1.5,
                fmt=":",
                axis=2,
            ),
            tb.plot.Entry(
                n_people,
                label="Occupancy count",
                color=C.blue,
                linewidth=1.5,
                fmt=":",
                axis=3,
            ),
        ],
        ylim_1axis=(400, 800),
        ylim_2axis=(0, 1.5),
        ylim_3axis=(0, 6),
        ylabel_1axis=r"CO$_2$ [ppmv]",
        ylabel_2axis="Position / Signal [0-1]",
        ylabel_3axis="Occupancy count",
        # title=r"CO$_2$ & damper (calibrated)",
        show=False,
        nticks=7,
    )
    add_hourly_ticks(axes, tz_info=tz.gettz("Europe/Copenhagen"))
    move_legend(fig, y=1.01, fontsize=11)
    fig.set_size_inches(11, 4)
    fig.savefig("cal_c_d.png", dpi=400, bbox_inches="tight")

    # --- Plot 3: Occupancy chain ---
    fig, axes = tb.plot.plot(
        dt,
        [
            tb.plot.Entry(
                n_people, label="Estimated occupancy", color="#d95f02", linewidth=1.5
            ),
            tb.plot.Entry(
                occ_signal,
                label="Occupancy signal (binary)",
                color="#7570b3",
                linewidth=1.5,
                axis=2,
            ),
            tb.plot.Entry(
                damper_sim,
                label="Damper simulated",
                color="#e7298a",
                linewidth=1.5,
                axis=2,
            ),
        ],
        ylim_2axis=(0, 1),
        ylabel_1axis="Number of people",
        ylabel_2axis="Signal / Position [0-1]",
        title="Occupancy estimation & control chain",
        show=True,
        nticks=11,
    )
    add_hourly_ticks(axes, tz_info=tz.gettz("Europe/Copenhagen"))
    move_legend(fig, y=1.02)

    energy_kwh = (
        model.components["office_space_heater"].output["Power"].history().sum()
        * step_size
        / 3600
        / 1000
    )
    print(f"Energy: {energy_kwh:.2f} kWh")


if __name__ == "__main__":
    main()
