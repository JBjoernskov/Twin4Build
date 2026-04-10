"""
Batching benchmark for the estimator example model.

Runs both the original (non-batched) and compiled (n_c-batched) simulations
of the estimator example model, compares wall-clock times and verifies that
key outputs (indoor temperature, damper positions, heater power) match
between the two.

The estimator example has two DamperTorchSystem components (supply + exhaust)
that share the same class and parameter structure, so the compiler merges them
into a single batched meta-component with n_c=2.
"""

import time
import datetime

import numpy as np
import twin4build as tb
from dateutil import tz
import twin4build.examples.utils as utils


# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════
STEP_SIZE = 1200  # 20 minutes in seconds
TZ_CPH = tz.gettz("Europe/Copenhagen")
START_TIME = [
    datetime.datetime(2023, 11, 27, 0, 0, 0, tzinfo=TZ_CPH),
    datetime.datetime(2023, 12, 2, 0, 0, 0, tzinfo=TZ_CPH),
]
END_TIME = [
    datetime.datetime(2023, 12, 1, 0, 0, 0, tzinfo=TZ_CPH),
    datetime.datetime(2023, 12, 5, 0, 0, 0, tzinfo=TZ_CPH),
]


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════
def setup_sensor_filenames(model):
    """Assign CSV data-source paths to sensor/schedule components."""
    model.components["020B_temperature_sensor"].filename = utils.get_path(
        ["estimator_example", "temperature_sensor.csv"]
    )
    model.components["020B_co2_sensor"].filename = utils.get_path(
        ["estimator_example", "co2_sensor.csv"]
    )
    model.components["020B_valve_position_sensor"].filename = utils.get_path(
        ["estimator_example", "valve_position_sensor.csv"]
    )
    model.components["020B_damper_position_sensor"].filename = utils.get_path(
        ["estimator_example", "damper_position_sensor.csv"]
    )
    model.components["BTA004"].filename = utils.get_path(
        ["estimator_example", "supply_air_temperature.csv"]
    )
    model.components["020B_temperature_heating_setpoint"].filename = utils.get_path(
        ["estimator_example", "temperature_heating_setpoint.csv"]
    )
    model.components["outdoor_environment"].filename_outdoorTemperature = utils.get_path(
        ["estimator_example", "outdoor_environment.csv"]
    )
    model.components["outdoor_environment"].filename_globalIrradiation = utils.get_path(
        ["estimator_example", "outdoor_environment.csv"]
    )
    model.components["outdoor_environment"].filename_outdoorCo2Concentration = utils.get_path(
        ["estimator_example", "outdoor_environment.csv"]
    )


def run_simulation(model, label):
    """Create a Simulator, run it, and return ``(simulator, elapsed_seconds)``."""
    simulator = tb.Simulator(model)
    t0 = time.perf_counter()
    simulator.simulate(
        step_size=STEP_SIZE,
        start_time=START_TIME,
        end_time=END_TIME,
    )
    elapsed = time.perf_counter() - t0
    print(f"  [{label}] completed in {elapsed:.3f}s")
    return simulator, elapsed


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
# Main
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("  Model Compilator — Batched vs Non-Batched Benchmark")
    print("=" * 60)

    # ── Load original model ───────────────────────────────────────────────
    print("\nLoading original model ...")
    model = tb.Model(id="model_compilator_example")
    filename_simulation = utils.get_path(["estimator_example", "instance_graph.ttl"])
    model.load(simulation_model_filename=filename_simulation, verbose=0)
    setup_sensor_filenames(model)
    print(model)

    # ── Build compiled model ──────────────────────────────────────────────
    print("\nCompiling model (n_c batching) ...")
    compiled_model = model.build_compiled_model()
    compiled_model.simulation_model.load(verbose=0, validate_model=True)
    compiled_model.simulation_model.visualize()
    print(compiled_model)

    # ── Simulate both ─────────────────────────────────────────────────────
    print(f"\nSimulation periods : {START_TIME}")
    print(f"                  -> {END_TIME}")
    print(f"Step size          : {STEP_SIZE}s ({STEP_SIZE / 60:.0f} min)\n")

    print("Simulating original model ...")
    sim_orig, t_orig = run_simulation(model, "original")

    print("\nSimulating compiled model ...")
    sim_compiled, t_compiled = run_simulation(compiled_model, "compiled")

    # ── Timing summary ────────────────────────────────────────────────────
    speedup = t_orig / t_compiled if t_compiled > 0 else float("inf")
    print(f"\n{'─' * 50}")
    print(f"  Original : {t_orig:.3f}s")
    print(f"  Compiled : {t_compiled:.3f}s")
    print(f"  Speedup  : {speedup:.2f}x")
    print(f"{'─' * 50}")

    # ── Extract data ──────────────────────────────────────────────────────
    signals = {
        "Indoor Temperature": ("020B", "indoorTemperature", "output", "°C"),
        "Outdoor Temperature": ("outdoor_environment", "outdoorTemperature", "output", "°C"),
        "Heating Setpoint": ("020B_temperature_heating_controller", "setpointValue", "input", "°C"),
        "Heater Power": ("020B_space_heater", "Power", "output", "W"),
        "Heat Gain": ("020B", "heatGain", "input", "W"),
        "Controller Signal": ("020B_temperature_heating_controller", "inputSignal", "output", "m³/s"),
        "Supply Damper Flow": ("020B_room_supply_damper", "airFlowRate", "output", "m³/s"),
        "Supply Damper Position": ("020B_room_supply_damper", "damperPosition", "output", ""),
    }

    orig_data = {}
    comp_data = {}
    for key, (comp_id, port, io, _unit) in signals.items():
        orig_data[key] = get_scalar_history(model.components[comp_id], port, io)
        comp_data[key] = get_compiled_data(model, compiled_model, comp_id, port, io)

    # ── Numerical comparison ──────────────────────────────────────────────
    print("\n=== Numerical Comparison ===")
    all_pass = True
    for key, (_cid, _port, _io, unit) in signals.items():
        diff = np.max(np.abs(orig_data[key] - comp_data[key]))
        ok = diff < 1e-3
        if not ok:
            all_pass = False
        suffix = f" {unit}" if unit else ""
        print(f"  {key:<27s}: max_diff = {diff:.8f}{suffix}  [{'OK' if ok else 'MISMATCH'}]")

    if all_pass:
        print("\n  PASS — all outputs match within tolerance (< 0.001)")
    else:
        print("\n  WARN — some outputs diverge beyond 0.001")

    # ── Plot 1: Temperatures & heating ────────────────────────────────────
    time_steps = sim_orig.date_time_steps
    Entry = tb.plot.Entry

    tb.plot.plot(
        time=time_steps,
        entries=[
            Entry(data=orig_data["Indoor Temperature"],
                  label="Indoor Temp (original)", axis=1),
            Entry(data=comp_data["Indoor Temperature"],
                  label="Indoor Temp (compiled)", fmt="--", axis=1),
            Entry(data=orig_data["Outdoor Temperature"],
                  label="Outdoor Temp", axis=1),
            Entry(data=orig_data["Heating Setpoint"],
                  label="Setpoint", fmt=":", axis=1),
            Entry(data=orig_data["Heater Power"],
                  label="Heater Power (original)", axis=2),
            Entry(data=comp_data["Heater Power"],
                  label="Heater Power (compiled)", fmt="--", axis=2),
            Entry(data=orig_data["Controller Signal"],
                  label="Controller Signal (original)", axis=3),
            Entry(data=comp_data["Controller Signal"],
                  label="Controller Signal (compiled)", fmt="--", axis=3),
        ],
        ylabel_1axis="Temperature [°C]",
        ylabel_2axis="Power [W]",
        ylabel_3axis="Water flow rate [m³/s]",
        title="Temperature & Heating: Original vs Compiled",
        show=True,
        nticks=11,
    )

    # ── Plot 2: Damper system ─────────────────────────────────────────────
    tb.plot.plot(
        time=time_steps,
        entries=[
            Entry(data=orig_data["Supply Damper Flow"],
                  label="Air Flow (original)", axis=1),
            Entry(data=comp_data["Supply Damper Flow"],
                  label="Air Flow (compiled)", fmt="--", axis=1),
            Entry(data=orig_data["Supply Damper Position"],
                  label="Damper Position (original)", axis=2),
            Entry(data=comp_data["Supply Damper Position"],
                  label="Damper Position (compiled)", fmt="--", axis=2),
            Entry(data=orig_data["Indoor Temperature"],
                  label="Indoor Temp (original)", axis=3),
            Entry(data=comp_data["Indoor Temperature"],
                  label="Indoor Temp (compiled)", fmt="--", axis=3),
        ],
        ylabel_1axis="Air flow rate [m³/s]",
        ylabel_2axis="Damper position",
        ylabel_3axis="Temperature [°C]",
        title="Damper System: Original vs Compiled",
        show=True,
        nticks=11,
    )
