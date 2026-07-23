"""
Multi-room HVAC batching benchmark.

Creates N identical rooms, each served by a full air-side and water-side
heating system, then compares simulation of the original model against the
compiled (n_c-batched) model for correctness and speed.

To test horizontal scalability, change N_ROOMS at the top of this file and
observe the timing difference.

Topology per room
=================

  Air side
  --------
  damper_schedule ─► supply_damper ──┬──► coil ──► room (supplyAirTemperature)
                                     ├──────────► room (supplyAirFlowRate)
  damper_schedule ─► return_damper ──────────────► room (exhaustAirFlowRate)
  outdoor_env ──────────────────────────► coil (inletAirTemperature)
  supply_air_temp_schedule ────────────► coil (outletAirTemperatureSetpoint)

  Water side
  ----------
  valve_schedule ──► valve ──► space_heater ──► room (heatGain)
  supply_water_temp_schedule ──────────────► space_heater (supplyWaterTemperature)
  room (indoorTemperature) ────────────────► space_heater (indoorTemperature)

  Boundary conditions
  -------------------
  outdoor_env ──────► room (outdoorTemperature, globalIrradiation, outdoorCO2)
  occupancy_schedule ─► room (numberOfPeople)

All per-room components share the same class and parameter structure, so the
compiled model batches them along the n_c axis.
"""

import datetime
import os
import shutil
import time
from typing import List

import numpy as np
import pandas as pd
import pytz
import torch

import twin4build as tb
from twin4build.model.model import Model
from twin4build.simulator.simulator import Simulator

# ═══════════════════════════════════════════════════════════════════════════
# Configuration — change these to scale the benchmark
# ═══════════════════════════════════════════════════════════════════════════
N_ROOMS = 5
STEP_SIZE = 600        # seconds (10 min)
SIM_DAYS = 2           # simulation duration

TZ = pytz.UTC
START = datetime.datetime(2023, 1, 15, 0, 0, 0, tzinfo=TZ)
END = START + datetime.timedelta(days=SIM_DAYS)


# ═══════════════════════════════════════════════════════════════════════════
# Synthetic weather data
# ═══════════════════════════════════════════════════════════════════════════
def make_weather_df(start, end, step_size):
    """Return a DataFrame with synthetic outdoor conditions."""
    freq = f"{step_size}s"
    idx = pd.date_range(
        start - datetime.timedelta(hours=1),
        end + datetime.timedelta(hours=1),
        freq=freq,
        tz=TZ,
    )
    hours = np.array([(t - start).total_seconds() / 3600 for t in idx])

    temperature = 2.0 + 5.0 * np.sin(2 * np.pi * hours / 24 - np.pi / 2)
    irradiation = np.maximum(0.0, 400.0 * np.sin(2 * np.pi * (hours - 6) / 24))
    co2 = np.full_like(hours, 420.0)

    return pd.DataFrame(
        {
            "outdoorTemperature": temperature,
            "globalIrradiation": irradiation,
            "outdoorCo2Concentration": co2,
        },
        index=idx,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Model builder
# ═══════════════════════════════════════════════════════════════════════════
def build_multi_room_model(n_rooms: int, model_id: str = "multi_room") -> Model:
    """Build an *n_rooms*-zone model with air-side and water-side heating."""

    model = Model(id=model_id)

    # ── Shared boundary conditions ────────────────────────────────────────
    weather_df = make_weather_df(START, END, STEP_SIZE)
    outdoor = tb.OutdoorEnvironmentSystem(df=weather_df, id="outdoor_environment")
    model.add_component(outdoor)

    damper_schedule = tb.ScheduleSystem(
        weekDayRulesetDict={
            "ruleset_default_value": 0.0,
            "ruleset_start_minute": [0, 0],
            "ruleset_end_minute":   [0, 0],
            "ruleset_start_hour":   [6, 18],
            "ruleset_end_hour":     [18, 24],
            "ruleset_value":        [0.8, 0.0],
        },
        id="damper_schedule",
    )
    valve_schedule = tb.ScheduleSystem(
        weekDayRulesetDict={
            "ruleset_default_value": 0.0,
            "ruleset_start_minute": [0, 0],
            "ruleset_end_minute":   [0, 0],
            "ruleset_start_hour":   [6, 18],
            "ruleset_end_hour":     [18, 24],
            "ruleset_value":        [0.6, 0.0],
        },
        id="valve_schedule",
    )
    supply_air_temp_schedule = tb.ScheduleSystem(
        weekDayRulesetDict={
            "ruleset_default_value": 18.0,
            "ruleset_start_minute": [0],
            "ruleset_end_minute":   [0],
            "ruleset_start_hour":   [6],
            "ruleset_end_hour":     [18],
            "ruleset_value":        [21.0],
        },
        id="supply_air_temp_schedule",
    )
    supply_water_temp_schedule = tb.ScheduleSystem(
        weekDayRulesetDict={
            "ruleset_default_value": 40.0,
            "ruleset_start_minute": [0],
            "ruleset_end_minute":   [0],
            "ruleset_start_hour":   [0],
            "ruleset_end_hour":     [24],
            "ruleset_value":        [60.0],
        },
        id="supply_water_temp_schedule",
    )
    occupancy_schedule = tb.ScheduleSystem(
        weekDayRulesetDict={
            "ruleset_default_value": 0.0,
            "ruleset_start_minute": [0, 0],
            "ruleset_end_minute":   [0, 0],
            "ruleset_start_hour":   [8, 17],
            "ruleset_end_hour":     [17, 24],
            "ruleset_value":        [5.0, 0.0],
        },
        id="occupancy_schedule",
    )
    model.add_component(damper_schedule)
    model.add_component(valve_schedule)
    model.add_component(supply_air_temp_schedule)
    model.add_component(supply_water_temp_schedule)
    model.add_component(occupancy_schedule)

    # ── Per-room HVAC ─────────────────────────────────────────────────────
    for i in range(n_rooms):
        tag = f"room_{i}"

        supply_damper = tb.DamperTorchSystem(
            a=1.0,
            nominalAirFlowRate=0.1,
            id=f"{tag}_supply_damper",
        )
        return_damper = tb.DamperTorchSystem(
            a=1.0,
            nominalAirFlowRate=0.1,
            id=f"{tag}_return_damper",
        )
        coil = tb.CoilTorchSystem(id=f"{tag}_coil")
        valve = tb.ValveTorchSystem(
            waterFlowRateMax=0.05,
            valveAuthority=0.5,
            id=f"{tag}_valve",
        )
        heater = tb.SpaceHeaterTorchSystem(
            Q_flow_nominal_sh=2000.0,
            T_a_nominal_sh=60.0,
            T_b_nominal_sh=45.0,
            TAir_nominal_sh=21.0,
            thermalMassHeatCapacity=50000.0,
            nelements=3,
            id=f"{tag}_space_heater",
        )
        room = tb.BuildingSpaceTorchSystem(
            thermal_kwargs={
                "C_air": 100000.0,
                "C_wall": 500000.0,
                "C_boundary": 100000.0,
                "R_out": 0.01,
                "R_in": 0.001,
                "R_boundary": 0.01,
                "f_wall": 0.5,
                "f_air": 0.5,
                "Q_occ_gain": 80.0,
            },
            mass_kwargs={
                "V": 150.0,
                "G_occ": 8.18e-6,
                "m_inf": 0.005,
            },
            id=tag,
        )

        # ── Air-side wiring ──────────────────────────────────────────────
        model.add_connection(
            damper_schedule, supply_damper,
            "scheduleValue", "damperPosition",
        )
        model.add_connection(
            damper_schedule, return_damper,
            "scheduleValue", "damperPosition",
        )
        model.add_connection(
            supply_damper, coil,
            "airFlowRate", "airFlowRate",
        )
        model.add_connection(
            outdoor, coil,
            "outdoorTemperature", "inletAirTemperature",
        )
        model.add_connection(
            supply_air_temp_schedule, coil,
            "scheduleValue", "outletAirTemperatureSetpoint",
        )
        model.add_connection(
            coil, room,
            "outletAirTemperature", "supplyAirTemperature",
        )
        model.add_connection(
            supply_damper, room,
            "airFlowRate", "supplyAirFlowRate",
        )
        model.add_connection(
            return_damper, room,
            "airFlowRate", "exhaustAirFlowRate",
        )

        # ── Water-side wiring ────────────────────────────────────────────
        model.add_connection(
            valve_schedule, valve,
            "scheduleValue", "valvePosition",
        )
        model.add_connection(
            valve, heater,
            "waterFlowRate", "waterFlowRate",
        )
        model.add_connection(
            supply_water_temp_schedule, heater,
            "scheduleValue", "supplyWaterTemperature",
        )
        model.add_connection(
            room, heater,
            "indoorTemperature", "indoorTemperature",
        )
        model.add_connection(
            heater, room,
            "Power", "heatGain",
        )

        # ── Boundary-condition wiring ────────────────────────────────────
        model.add_connection(
            outdoor, room,
            "outdoorTemperature", "outdoorTemperature",
        )
        model.add_connection(
            outdoor, room,
            "globalIrradiation", "globalIrradiation",
        )
        model.add_connection(
            outdoor, room,
            "outdoorCo2Concentration", "outdoorCO2",
        )
        model.add_connection(
            occupancy_schedule, room,
            "scheduleValue", "numberOfPeople",
        )

    return model


# ═══════════════════════════════════════════════════════════════════════════
# Simulation helpers
# ═══════════════════════════════════════════════════════════════════════════
def run_simulation(model: Model, label: str) -> tuple:
    """Simulate and return ``(simulator, elapsed_seconds)``."""
    simulator = Simulator(model)
    t0 = time.perf_counter()
    simulator.simulate(
        start_time=START,
        end_time=END,
        step_size=STEP_SIZE,
        show_progress_bar=True,
    )
    elapsed = time.perf_counter() - t0
    print(f"  [{label}] completed in {elapsed:.3f}s")
    return simulator, elapsed


def compare_results(model_orig: Model, model_compiled: Model):
    """Check that compiled outputs match the originals within tolerance."""
    print("\n=== Result Comparison ===")
    max_temp_diff = 0.0
    max_co2_diff = 0.0

    for i in range(N_ROOMS):
        tag = f"room_{i}"
        meta_info = model_orig.get_compiled_component_info(tag)
        if meta_info is None:
            print(f"  {tag}: not found in compiled-model mapping — skipping")
            continue
        meta, i_c = meta_info

        orig_temp = (
            model_orig.components[tag]
            .output["indoorTemperature"]
            .history()[:, :, 0]
            .detach()
            .numpy()
        )
        compiled_temp = (
            model_compiled.components[meta.id]
            .output["indoorTemperature"]
            .history()[:, :, i_c]
            .detach()
            .numpy()
        )

        orig_co2 = (
            model_orig.components[tag]
            .output["indoorCO2"]
            .history()[:, :, 0]
            .detach()
            .numpy()
        )
        compiled_co2 = (
            model_compiled.components[meta.id]
            .output["indoorCO2"]
            .history()[:, :, i_c]
            .detach()
            .numpy()
        )

        temp_diff = np.max(np.abs(orig_temp - compiled_temp))
        co2_diff = np.max(np.abs(orig_co2 - compiled_co2))
        max_temp_diff = max(max_temp_diff, temp_diff)
        max_co2_diff = max(max_co2_diff, co2_diff)

        print(
            f"  {tag}: "
            f"temp max_diff={temp_diff:.6f} C  "
            f"CO2 max_diff={co2_diff:.6f} ppm"
        )

    tol_temp = 1e-3
    tol_co2 = 1e-2
    if max_temp_diff < tol_temp and max_co2_diff < tol_co2:
        print(f"\n  PASS: All outputs match (temp<{tol_temp} C, CO2<{tol_co2}ppm)")
    else:
        print(
            f"\n  FAIL: Mismatch detected "
            f"(max temp diff={max_temp_diff:.6f}, max CO2 diff={max_co2_diff:.6f})"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Data extraction helpers
# ═══════════════════════════════════════════════════════════════════════════
def get_scalar_history(component, port_name, io_type="output", i_c=0):
    """Return ``(n_s, n_t)`` numpy array from a component port history."""
    port = component.output[port_name] if io_type == "output" else component.input[port_name]
    return port.history()[:, :, i_c].detach().cpu().numpy().T


def get_compiled_data(model_orig, compiled_model, orig_id, port_name, io_type="output"):
    """Look up the compiled meta-component for *orig_id* and extract its history."""
    meta_info = model_orig.get_compiled_component_info(orig_id)
    if meta_info is not None:
        meta, i_c = meta_info
        comp = compiled_model.components[meta.id]
    else:
        comp = compiled_model.components[orig_id]
        i_c = 0
    return get_scalar_history(comp, port_name, io_type, i_c)


# ═══════════════════════════════════════════════════════════════════════════
# Overlay plots
# ═══════════════════════════════════════════════════════════════════════════
def plot_comparison(
    model_orig: Model,
    model_compiled: Model,
    sim_orig: Simulator,
    n_rooms: int,
    room_indices: List[int] = None,
):
    """Create overlay plots comparing original vs compiled outputs.

    Parameters
    ----------
    model_orig : Model
        The original (non-batched) model — must already be simulated.
    model_compiled : Model
        The compiled (batched) model — must already be simulated.
    sim_orig : Simulator
        Simulator of the original model (used for the time axis).
    n_rooms : int
        Total number of rooms.
    room_indices : list[int], optional
        Which rooms to include in per-room plots.  Defaults to first,
        middle, and last.
    """
    Entry = tb.plot.Entry
    time_steps = sim_orig.date_time_steps

    if room_indices is None:
        room_indices = sorted({0, n_rooms // 2, n_rooms - 1})

    # ── Plot 1: Indoor temperature — selected rooms ───────────────────────
    temp_entries = []
    for idx in room_indices:
        tag = f"room_{idx}"
        orig = get_scalar_history(model_orig.components[tag], "indoorTemperature")
        comp = get_compiled_data(model_orig, model_compiled, tag, "indoorTemperature")
        temp_entries.append(Entry(data=orig, label=f"{tag} (original)", axis=1))
        temp_entries.append(Entry(data=comp, label=f"{tag} (compiled)", fmt="--", axis=1))

    outdoor_temp = get_scalar_history(
        model_orig.components["outdoor_environment"], "outdoorTemperature"
    )
    temp_entries.append(Entry(data=outdoor_temp, label="Outdoor", fmt=":", axis=1))

    tb.plot.plot(
        time=time_steps,
        entries=temp_entries,
        ylabel_1axis="Temperature [°C]",
        title="Indoor Temperature: Original vs Compiled",
        show=True,
        nticks=11,
    )

    # ── Plot 2: Indoor CO2 — selected rooms ───────────────────────────────
    co2_entries = []
    for idx in room_indices:
        tag = f"room_{idx}"
        orig = get_scalar_history(model_orig.components[tag], "indoorCO2")
        comp = get_compiled_data(model_orig, model_compiled, tag, "indoorCO2")
        co2_entries.append(Entry(data=orig, label=f"{tag} (original)", axis=1))
        co2_entries.append(Entry(data=comp, label=f"{tag} (compiled)", fmt="--", axis=1))

    tb.plot.plot(
        time=time_steps,
        entries=co2_entries,
        ylabel_1axis="CO2 [ppm]",
        title="Indoor CO2: Original vs Compiled",
        show=True,
        nticks=11,
    )

    # ── Plot 3: Heating system — room_0 ──────────────────────────────────
    r0 = "room_0"
    orig_temp_r0 = get_scalar_history(model_orig.components[r0], "indoorTemperature")
    comp_temp_r0 = get_compiled_data(model_orig, model_compiled, r0, "indoorTemperature")

    orig_power = get_scalar_history(
        model_orig.components[f"{r0}_space_heater"], "Power"
    )
    comp_power = get_compiled_data(
        model_orig, model_compiled, f"{r0}_space_heater", "Power"
    )

    orig_valve_flow = get_scalar_history(
        model_orig.components[f"{r0}_valve"], "waterFlowRate"
    )
    comp_valve_flow = get_compiled_data(
        model_orig, model_compiled, f"{r0}_valve", "waterFlowRate"
    )

    tb.plot.plot(
        time=time_steps,
        entries=[
            Entry(data=orig_temp_r0, label="Indoor Temp (original)", axis=1),
            Entry(data=comp_temp_r0, label="Indoor Temp (compiled)", fmt="--", axis=1),
            Entry(data=outdoor_temp, label="Outdoor Temp", fmt=":", axis=1),
            Entry(data=orig_power, label="Heater Power (original)", axis=2),
            Entry(data=comp_power, label="Heater Power (compiled)", fmt="--", axis=2),
            Entry(data=orig_valve_flow, label="Valve Flow (original)", axis=3),
            Entry(data=comp_valve_flow, label="Valve Flow (compiled)", fmt="--", axis=3),
        ],
        ylabel_1axis="Temperature [°C]",
        ylabel_2axis="Power [W]",
        ylabel_3axis="Water flow [m³/s]",
        title=f"Heating System ({r0}): Original vs Compiled",
        show=True,
        nticks=11,
    )

    # ── Plot 4: Damper system — room_0 ───────────────────────────────────
    orig_damper_flow = get_scalar_history(
        model_orig.components[f"{r0}_supply_damper"], "airFlowRate"
    )
    comp_damper_flow = get_compiled_data(
        model_orig, model_compiled, f"{r0}_supply_damper", "airFlowRate"
    )

    orig_damper_pos = get_scalar_history(
        model_orig.components[f"{r0}_supply_damper"], "damperPosition"
    )
    comp_damper_pos = get_compiled_data(
        model_orig, model_compiled, f"{r0}_supply_damper", "damperPosition"
    )

    tb.plot.plot(
        time=time_steps,
        entries=[
            Entry(data=orig_damper_flow, label="Air Flow (original)", axis=1),
            Entry(data=comp_damper_flow, label="Air Flow (compiled)", fmt="--", axis=1),
            Entry(data=orig_damper_pos, label="Damper Pos. (original)", axis=2),
            Entry(data=comp_damper_pos, label="Damper Pos. (compiled)", fmt="--", axis=2),
            Entry(data=orig_temp_r0, label="Indoor Temp (original)", axis=3),
            Entry(data=comp_temp_r0, label="Indoor Temp (compiled)", fmt="--", axis=3),
        ],
        ylabel_1axis="Air flow rate [m³/s]",
        ylabel_2axis="Damper position",
        ylabel_3axis="Temperature [°C]",
        title=f"Damper System ({r0}): Original vs Compiled",
        show=True,
        nticks=11,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"=== Multi-Room Batching Benchmark (N_ROOMS={N_ROOMS}) ===\n")

    # ── Build original model ──────────────────────────────────────────────
    print("Building original model ...")
    model = build_multi_room_model(N_ROOMS)
    model.load()
    n_comp = model.simulation_model.count_components()
    n_conn = model.simulation_model.count_connections()
    print(f"  {n_comp} components, {n_conn} connections\n")

    # ── Build compiled model ──────────────────────────────────────────────
    print("Compiling model (n_c batching) ...")
    compiled = model.build_compiled_model()
    compiled.load()
    n_comp_c = compiled.simulation_model.count_components()
    n_conn_c = compiled.simulation_model.count_connections()
    print(f"  {n_comp_c} components, {n_conn_c} connections")
    print(f"  Reduction: {n_comp} -> {n_comp_c} components\n")

    # ── Simulate original ─────────────────────────────────────────────────
    print(f"Simulating original model ({SIM_DAYS} days, dt={STEP_SIZE}s) ...")
    sim_orig, t_orig = run_simulation(model, "original")

    # ── Simulate compiled ─────────────────────────────────────────────────
    print(f"\nSimulating compiled model ({SIM_DAYS} days, dt={STEP_SIZE}s) ...")
    sim_compiled, t_compiled = run_simulation(compiled, "compiled")

    # ── Summary ───────────────────────────────────────────────────────────
    speedup = t_orig / t_compiled if t_compiled > 0 else float("inf")
    print(f"\n=== Timing Summary (N_ROOMS={N_ROOMS}) ===")
    print(f"  Original : {t_orig:.3f}s")
    print(f"  Compiled : {t_compiled:.3f}s")
    print(f"  Speedup  : {speedup:.2f}x")

    compare_results(model, compiled)

    # ── Overlay plots ─────────────────────────────────────────────────────
    plot_comparison(model, compiled, sim_orig, N_ROOMS)

    # ── Cleanup generated artefacts ───────────────────────────────────────
    for mid in [model.id, compiled.id]:
        p = os.path.join("generated_files", "models", mid)
        if os.path.exists(p):
            shutil.rmtree(p)
