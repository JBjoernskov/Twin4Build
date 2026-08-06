"""
Replay optimized simulation and produce diagnostic plots.

Loads the saved optimization result (.pickle), replays the simulation with
the optimized valve positions on the full model, then plots temperatures,
valve position, and all heat-source diagnostics.

Usage:
    python plot_optimization_results.py
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
from plot_calibration_results import add_hourly_ticks, compute_metrics, move_legend

# Local application imports
import twin4build as tb
import twin4build.examples.utils as utils

# ── Configuration ─────────────────────────────────────────────────────────
ESTIMATION_PICKLE = (
    r"generated_files\models\full_workflow_example"
    r"\model_parameters\estimation_results"
    r"\20260303_103841_scipy_SLSQP_ad.pickle"
)

OPT_RESULT_PICKLE = (
    r"generated_files\models\full_workflow_example"
    r"\optimization_results"
    r"\20260326_225247_optimization.pickle"
)


def main():
    # ── 1. Load saved optimization results ────────────────────────────────
    result_path = utils.get_path([OPT_RESULT_PICKLE])
    with open(result_path, "rb") as f:
        r = pickle.load(f)

    dt = r["date_time_steps"]
    opt_start_time = r["opt_start_time"]
    opt_end_time = r["opt_end_time"]
    step_size = r["step_size"]
    saved_valve = r["valve_position"]  # (n_t, n_s, n_c)

    print(f"Period : {opt_start_time[0]} → {opt_end_time[0]}")
    print(f"Step   : {step_size}s  |  Timesteps: {saved_valve.shape[0]}")

    # ── 2. Save optimized valve positions to CSV ──────────────────────────
    valve_csv_path = utils.get_path(
        ["estimator_example", "optimized_valve_position.csv"]
    )
    valve_df = pd.DataFrame(
        {
            "time": dt[0],
            "value": saved_valve[:, 0, 0].numpy(),
        }
    )
    valve_df.to_csv(valve_csv_path, index=False)

    # ── 3. Load model & apply calibrated parameters ───────────────────────
    model = tb.Model(id="full_workflow_example")
    filename_simulation, _ = model._simulation_model._semantic_model.get_dir(
        filename="instance_graph.ttl"
    )
    model.load(simulation_model_filename=filename_simulation)

    estimation_path = utils.get_path([ESTIMATION_PICKLE])
    model.load_estimation_result(filename=estimation_path, verbose=1)
    print("Calibrated model loaded.")

    # ── 4. Component references ───────────────────────────────────────────
    space = model.components["office"]
    space_heater = model.components["office_space_heater"]
    heating_controller = model.components["office_temperature_heating_controller"]
    space_heater_valve = model.components["office_space_heater_valve"]
    heating_setpoint = model.components["office_temperature_heating_setpoint"]

    # ── 5. Rewire: replace controller with saved valve schedule ───────────
    valve_position_schedule = tb.ScheduleSystem(
        filename=valve_csv_path,
        date_column=0,
        value_column=1,
        id="valve_position_schedule",
    )

    model.remove_connection(
        heating_controller, space_heater_valve, "inputSignal", "valvePosition"
    )
    model.add_connection(
        valve_position_schedule, space_heater_valve, "scheduleValue", "valvePosition"
    )
    model.load()
    print("Model rewired with optimized valve positions.")

    # ── 6. Replay simulation ──────────────────────────────────────────────
    simulator = tb.Simulator(model)
    simulator.simulate(
        step_size=step_size,
        start_time=opt_start_time,
        end_time=opt_end_time,
    )
    print("Replay simulation complete.")

    # ── 7. Extract all signals ────────────────────────────────────────────
    indoor_temp = space.output["indoorTemperature"].history().detach()
    valve = space_heater_valve.output["valvePosition"].history().detach()
    power = space_heater.output["Power"].history().detach()
    heating_sp = heating_setpoint.output["scheduleValue"].history().detach()

    supply_air_temp = space.input["supplyAirTemperature"].history().detach()
    supply_air_flow = space.input["supplyAirFlowRate"].history().detach()
    outdoor_temp = space.input["outdoorTemperature"].history().detach()
    n_people = space.input["numberOfPeople"].history().detach()
    heat_gain = space.input["heatGain"].history().detach()
    irradiation = space.input["globalIrradiation"].history().detach()

    print(f"Peak power      : {power[:, 0, 0].max():.1f} W")
    print(
        f"Temp range      : {indoor_temp[:, 0, 0].min():.1f} – {indoor_temp[:, 0, 0].max():.1f} °C"
    )
    print(
        f"Supply air range: {supply_air_temp[:, 0, 0].min():.1f} – {supply_air_temp[:, 0, 0].max():.1f} °C"
    )
    print(
        f"Outdoor range   : {outdoor_temp[:, 0, 0].min():.1f} – {outdoor_temp[:, 0, 0].max():.1f} °C"
    )
    print(f"Max occupancy   : {n_people[:, 0, 0].max():.1f}")

    tz_cph = tz.gettz("Europe/Copenhagen")
    C = tb.plot.Colors

    # =====================================================================
    # ── 8. PLOTS ─────────────────────────────────────────────────────────
    # =====================================================================

    # --- Plot 1: Temperature + valve + setpoint (main result) ---
    fig1, axes1 = tb.plot.plot(
        dt,
        [
            tb.plot.Entry(
                indoor_temp, label="Temperature simulated", color="#d95f02", linewidth=2
            ),
            tb.plot.Entry(
                valve, label="Valve simulated", color="#e7298a", linewidth=2, axis=2
            ),
            tb.plot.Entry(
                heating_sp,
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
        show=False,
        nticks=7,
    )
    add_hourly_ticks(axes1, tz_info=tz_cph)
    move_legend(fig1, y=1.01, fontsize=11)
    fig1.set_size_inches(11, 4)
    fig1.savefig("opt_t_v.png", dpi=400, bbox_inches="tight")

    # --- Plot 2: Diagnostic – temperature comparison ---
    fig2, axes2 = tb.plot.plot(
        dt,
        [
            tb.plot.Entry(
                indoor_temp, label="Indoor temperature", color="#d95f02", linewidth=2
            ),
            tb.plot.Entry(
                supply_air_temp,
                label="Supply air temperature",
                color="#1b9e77",
                linewidth=2,
            ),
            tb.plot.Entry(
                outdoor_temp, label="Outdoor temperature", color="#7570b3", linewidth=2
            ),
            tb.plot.Entry(
                heating_sp,
                label="Heating setpoint",
                color=C.black,
                linewidth=1.5,
                fmt=":",
            ),
        ],
        ylabel_1axis=r"Temperature [$^\circ$C]",
        title="Diagnostic: temperatures (supply air above indoor = AHU heating)",
        show=False,
        nticks=7,
    )
    add_hourly_ticks(axes2, tz_info=tz_cph)
    move_legend(fig2, y=1.01, fontsize=11)
    fig2.set_size_inches(11, 4)
    fig2.savefig("opt_diagnostic_temps.png", dpi=400, bbox_inches="tight")

    # --- Plot 3: Diagnostic – heat sources ---
    fig3, axes3 = tb.plot.plot(
        dt,
        [
            tb.plot.Entry(
                heat_gain, label="Space heater power", color="#d95f02", linewidth=2
            ),
            tb.plot.Entry(
                irradiation,
                label="Solar irradiation",
                color="#e6ab02",
                linewidth=1.5,
                fmt="--",
            ),
            tb.plot.Entry(
                supply_air_flow,
                label="Supply air flow rate",
                color="#1b9e77",
                linewidth=2,
                axis=2,
            ),
            tb.plot.Entry(
                n_people, label="Number of people", color="#7570b3", linewidth=2, axis=3
            ),
        ],
        ylabel_1axis=r"Power [W] / Irradiation [W/m$^2$]",
        ylabel_2axis="Air flow [kg/s]",
        ylabel_3axis="People",
        title="Diagnostic: heat sources",
        show=False,
        nticks=7,
    )
    add_hourly_ticks(axes3, tz_info=tz_cph)
    move_legend(fig3, y=1.01, fontsize=11)
    fig3.set_size_inches(11, 4)
    fig3.savefig("opt_diagnostic_sources.png", dpi=400, bbox_inches="tight")

    plt.show()

    # =====================================================================
    # ── 9. PERTURBATION TEST: Does opening valve reduce total loss? ──────
    # =====================================================================
    print("\n" + "=" * 70)
    print("PERTURBATION TEST")
    print("Does increasing valve at a violated timestep reduce the total loss?")
    print("=" * 70)

    temp_1d = indoor_temp[:, 0, 0]
    sp_1d = heating_sp[:, 0, 0]
    power_1d = power[:, 0, 0]
    violations = (sp_1d - temp_1d).clamp(min=0)

    if violations.max() <= 0:
        print("No constraint violations detected — nothing to test.")
    else:
        t_worst = violations.argmax().item()
        n_t = temp_1d.shape[0]

        # Perturb a 4-hour block (12 × 20-min steps) centred on worst violation
        BLOCK = 12
        t_start = max(0, t_worst - BLOCK // 2)
        t_end = min(n_t, t_start + BLOCK)

        print(f"\nWorst violation: t={t_worst} ({dt[0][t_worst]})")
        print(f"  Temperature : {temp_1d[t_worst]:.2f} °C")
        print(f"  Setpoint    : {sp_1d[t_worst]:.2f} °C")
        print(f"  Gap         : {violations[t_worst]:.2f} °C")
        print(f"  Valve       : {saved_valve[t_worst, 0, 0]:.4f}")
        print(
            f"\nPerturbing valve → 1.0 for t={t_start}–{t_end - 1}  "
            f"({dt[0][t_start]:%a %H:%M} – {dt[0][t_end - 1]:%a %H:%M})"
        )

        perturbed_valve = saved_valve[:, 0, 0].clone()
        perturbed_valve[t_start:t_end] = 1.0

        perturbed_csv = utils.get_path(
            ["estimator_example", "perturbed_valve_position.csv"]
        )
        pd.DataFrame({"time": dt[0], "value": perturbed_valve.numpy()}).to_csv(
            perturbed_csv, index=False
        )

        valve_position_schedule.filename = perturbed_csv
        simulator.simulate(
            step_size=step_size,
            start_time=opt_start_time,
            end_time=opt_end_time,
        )
        print("Perturbed simulation complete.\n")

        pert_temp = space.output["indoorTemperature"].history().detach()[:, 0, 0]
        pert_power = space_heater.output["Power"].history().detach()[:, 0, 0]
        pert_sp = heating_setpoint.output["scheduleValue"].history().detach()[:, 0, 0]
        pert_violations = (pert_sp - pert_temp).clamp(min=0)

        # ── Loss computation (mirrors optimizer's __obj_ad) ──────────────
        # Normalization ranges fixed from the baseline (same as optimizer cache)
        temp_lo, temp_hi = temp_1d.min().item(), temp_1d.max().item()
        if temp_hi == temp_lo:
            temp_hi = temp_lo + 1.0
        pow_lo, pow_hi = power_1d.min().item(), power_1d.max().item()
        if pow_hi == pow_lo:
            pow_hi = pow_lo + 1.0

        def loss_components(temp, sp, pwr, k=1000):
            t_n = (temp - temp_lo) / (temp_hi - temp_lo)
            s_n = (sp - temp_lo) / (temp_hi - temp_lo)
            p_n = (pwr - pow_lo) / (pow_hi - pow_lo)
            obj = p_n.mean()
            pen = k * torch.relu(s_n - t_n).mean()
            return obj.item(), pen.item()

        b_obj, b_pen = loss_components(temp_1d, sp_1d, power_1d)
        p_obj, p_pen = loss_components(pert_temp, pert_sp, pert_power)
        b_tot, p_tot = b_obj + b_pen, p_obj + p_pen

        hdr = f"{'':34s} {'Baseline':>12s} {'Perturbed':>12s} {'Δ':>12s}"
        sep = "-" * 74
        print(hdr)
        print(sep)
        print(
            f"{'Objective (norm. mean power)':34s} {b_obj:12.6f} {p_obj:12.6f} {p_obj - b_obj:+12.6f}"
        )
        print(
            f"{'Lower-constraint penalty (k=1000)':34s} {b_pen:12.6f} {p_pen:12.6f} {p_pen - b_pen:+12.6f}"
        )
        print(f"{'Total loss':34s} {b_tot:12.6f} {p_tot:12.6f} {p_tot - b_tot:+12.6f}")
        print(sep)

        if p_tot < b_tot:
            print("RESULT: Total loss DECREASED → opening the valve helps.")
            print("  The optimizer should find this direction.  If it still")
            print("  violates constraints, increase  constraint_penalty.")
        else:
            print("RESULT: Total loss INCREASED → cost increase outweighs")
            print("  the penalty reduction.  Raise  constraint_penalty  in")
            print("  OPT_OPTIONS to make constraint satisfaction worthwhile.")

        # Physical summary around perturbation window
        print(f"\nPhysical changes (perturbed block t={t_start}–{t_end - 1}):")
        print(
            f"  Mean power : {power_1d[t_start:t_end].mean():.1f} → "
            f"{pert_power[t_start:t_end].mean():.1f} W  "
            f"(Δ = {pert_power[t_start:t_end].mean() - power_1d[t_start:t_end].mean():+.1f})"
        )
        print(
            f"  Mean temp  : {temp_1d[t_start:t_end].mean():.2f} → "
            f"{pert_temp[t_start:t_end].mean():.2f} °C  "
            f"(Δ = {pert_temp[t_start:t_end].mean() - temp_1d[t_start:t_end].mean():+.3f})"
        )
        print(
            f"  Mean viol. : {violations[t_start:t_end].mean():.3f} → "
            f"{pert_violations[t_start:t_end].mean():.3f} °C  "
            f"(Δ = {pert_violations[t_start:t_end].mean() - violations[t_start:t_end].mean():+.3f})"
        )

        print(f"\nTimestep-level detail:")
        for t in range(t_start, min(t_end + 6, n_t)):
            tag = " *" if t_start <= t < t_end else ""
            print(
                f"  t={t:3d} {dt[0][t]:%a %H:%M}  "
                f"T={temp_1d[t]:5.2f}→{pert_temp[t]:5.2f} "
                f"(Δ{pert_temp[t] - temp_1d[t]:+.3f})  "
                f"sp={sp_1d[t]:5.1f}  "
                f"viol={violations[t]:.2f}→{pert_violations[t]:.2f}{tag}"
            )


if __name__ == "__main__":
    main()
