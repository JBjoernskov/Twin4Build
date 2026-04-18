"""E2 -- Heterogeneity sweep (speedup vs archetype count).

Holds ``N_E2`` rooms fixed and partitions them into ``k`` archetypes.  Each
archetype is a *distinct subclass* of
:class:`twin4build.systems.building_space.BuildingSpaceTorchSystem` so the
compiler signature differs (it includes the class module + name -- see
``Model._component_signature``).  With ``k`` archetypes and ``N/k`` rooms
per archetype the compiler produces ``k`` room batches of size ``N/k``
instead of a single batch of size ``N``; speedup should decrease
monotonically with ``k``.

We chose dynamic subclassing (rather than varying numeric parameter values
or ``nelements``) because those knobs do **not** change the signature -- the
compiler groups by port structure + parameter *keys* + class, not by
parameter values or ``self.nelements``.  This is the minimal structural
change that honestly splits a batch group.

Writes ``results/e2_heterogeneity.csv`` (one row per ``k``) on CPU
multi-thread only; E2 is about batch structure, not CPU parallelism.
"""

from __future__ import annotations

import datetime
import gc
import sys
from typing import Dict, List, Type

import twin4build as tb
from twin4build.model.model import Model
from twin4build.systems.building_space.building_space_torch_system import (
    BuildingSpaceTorchSystem,
)

from twin4build.examples.paper_experiments import common, config


CSV_PATH = config.RESULTS_DIR / "e2_heterogeneity.csv"


# ---------------------------------------------------------------------------
# Archetype synthesis
# ---------------------------------------------------------------------------
def _make_archetype_subclasses(k: int) -> List[Type[BuildingSpaceTorchSystem]]:
    """Return ``k`` distinct BuildingSpaceTorchSystem subclasses.

    Each subclass is otherwise identical to its parent, but has a unique
    class name so the signature computed by ``Model._component_signature``
    (which keys on class module + name) assigns each archetype to its own
    batch group.
    """
    subclasses: List[Type[BuildingSpaceTorchSystem]] = []
    for i in range(k):
        cls = type(
            f"BuildingSpaceTorchSystem_Archetype{i}",
            (BuildingSpaceTorchSystem,),
            {
                "__module__": BuildingSpaceTorchSystem.__module__,
            },
        )
        subclasses.append(cls)
    return subclasses


def _build_heterogeneous_model(
    n_rooms: int,
    k: int,
    *,
    start: datetime.datetime,
    horizon_days: int,
    step_size: int,
    model_id: str,
) -> Model:
    """Same topology as :func:`common.build_multi_room_model` but with
    ``k`` distinct BuildingSpace subclasses rotated across rooms."""
    import pytz  # noqa: F401 -- ensures common.TZ works with pickled DFs
    import numpy as np  # noqa: F401

    end = start + datetime.timedelta(days=horizon_days)
    model = Model(id=model_id)

    weather_df = common.make_weather_df(start, end, step_size)
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
    for sched in (
        damper_schedule,
        valve_schedule,
        supply_air_temp_schedule,
        supply_water_temp_schedule,
        occupancy_schedule,
    ):
        model.add_component(sched)

    archetype_classes = _make_archetype_subclasses(k)

    for i in range(n_rooms):
        tag = f"room_{i}"
        archetype_idx = i % k
        RoomClass = archetype_classes[archetype_idx]

        supply_damper = tb.DamperTorchSystem(
            a=1.0, nominalAirFlowRate=0.1, id=f"{tag}_supply_damper",
        )
        return_damper = tb.DamperTorchSystem(
            a=1.0, nominalAirFlowRate=0.1, id=f"{tag}_return_damper",
        )
        coil = tb.CoilTorchSystem(id=f"{tag}_coil")
        valve = tb.ValveTorchSystem(
            waterFlowRateMax=0.05, valveAuthority=0.5, id=f"{tag}_valve",
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
        room = RoomClass(
            thermal_kwargs={
                "C_air": 100000.0,
                "C_wall": 500000.0,
                "C_int": 100000.0,
                "C_boundary": 100000.0,
                "R_out": 0.01,
                "R_in": 0.001,
                "R_int": 0.005,
                "R_boundary": 0.01,
                "f_wall": 0.5,
                "f_air": 0.5,
                "Q_occ_gain": 80.0,
            },
            mass_kwargs={"V": 150.0, "G_occ": 8.18e-6, "m_inf": 0.005},
            id=tag,
        )

        model.add_connection(damper_schedule, supply_damper, "scheduleValue", "damperPosition")
        model.add_connection(damper_schedule, return_damper, "scheduleValue", "damperPosition")
        model.add_connection(supply_damper, coil, "airFlowRate", "airFlowRate")
        model.add_connection(outdoor, coil, "outdoorTemperature", "inletAirTemperature")
        model.add_connection(
            supply_air_temp_schedule, coil,
            "scheduleValue", "outletAirTemperatureSetpoint",
        )
        model.add_connection(coil, room, "outletAirTemperature", "supplyAirTemperature")
        model.add_connection(supply_damper, room, "airFlowRate", "supplyAirFlowRate")
        model.add_connection(return_damper, room, "airFlowRate", "exhaustAirFlowRate")

        model.add_connection(valve_schedule, valve, "scheduleValue", "valvePosition")
        model.add_connection(valve, heater, "waterFlowRate", "waterFlowRate")
        model.add_connection(
            supply_water_temp_schedule, heater,
            "scheduleValue", "supplyWaterTemperature",
        )
        model.add_connection(room, heater, "indoorTemperature", "indoorTemperature")
        model.add_connection(heater, room, "Power", "heatGain")

        model.add_connection(outdoor, room, "outdoorTemperature", "outdoorTemperature")
        model.add_connection(outdoor, room, "globalIrradiation", "globalIrradiation")
        model.add_connection(outdoor, room, "outdoorCo2Concentration", "outdoorCO2")
        model.add_connection(occupancy_schedule, room, "scheduleValue", "numberOfPeople")

    return model


# ---------------------------------------------------------------------------
# Measurement loop
# ---------------------------------------------------------------------------
def _run_cell(
    k: int,
    n_rooms: int,
    device_key: str,
    thread_count: int,
) -> Dict[str, object]:
    common.set_cpu_threads(device_key, thread_count)
    common.reset_peak_memory(device_key)

    start = common.DEFAULT_START
    end = start + datetime.timedelta(days=config.HORIZON_DAYS_DEFAULT)
    step = config.STEP_SIZE_DEFAULT
    repeats = config.repeats_for(n_rooms)
    warmup = config.N_WARMUP

    model = _build_heterogeneous_model(
        n_rooms,
        k,
        start=start,
        horizon_days=config.HORIZON_DAYS_DEFAULT,
        step_size=step,
        model_id=f"e2_k{k}",
    )
    model.load(
        draw_semantic_model=False, draw_simulation_model=False, verbose=0
    )
    n_comp_orig, n_conn_orig = common.graph_size(model)

    compiled, t_compile = common.timed(model.build_compiled_model)
    compiled.load(
        draw_semantic_model=False, draw_simulation_model=False, verbose=0
    )
    n_comp_comp, n_conn_comp = common.graph_size(compiled)
    wiring = common.connection_wiring_stats(compiled)

    orig_measurements: List[float] = []
    comp_measurements: List[float] = []
    for rep in range(warmup + repeats):
        _, dt_o = common.timed(
            lambda: common.simulate_once(model, start, end, step)
        )
        _, dt_c = common.timed(
            lambda: common.simulate_once(compiled, start, end, step)
        )
        if rep >= warmup:
            orig_measurements.append(dt_o)
            comp_measurements.append(dt_c)

    o_q25, o_q50, o_q75 = common.iqr(orig_measurements)
    c_q25, c_q50, c_q75 = common.iqr(comp_measurements)
    speedup = o_q50 / c_q50 if c_q50 > 0 else float("inf")
    rss = common.peak_memory_mb(device_key)

    del model, compiled
    gc.collect()

    return {
        "device": device_key,
        "thread_count": thread_count,
        "k_archetypes": k,
        "n_rooms": n_rooms,
        "n_per_archetype": n_rooms // k,
        "horizon_days": config.HORIZON_DAYS_DEFAULT,
        "step_size_s": step,
        "n_repeats": repeats,
        "t_compile_s": t_compile,
        "t_sim_orig_q25": o_q25,
        "t_sim_orig_median": o_q50,
        "t_sim_orig_q75": o_q75,
        "t_sim_comp_q25": c_q25,
        "t_sim_comp_median": c_q50,
        "t_sim_comp_q75": c_q75,
        "speedup_median": speedup,
        "n_comp_orig": n_comp_orig,
        "n_comp_comp": n_comp_comp,
        "n_conn_orig": n_conn_orig,
        "n_conn_comp": n_conn_comp,
        "frac_aligned": wiring["fraction_aligned"],
        "n_conn_aligned": wiring["n_connections_aligned"],
        "n_conn_gather": wiring["n_connections_gather"],
        "rss_peak_mb": rss,
    }


def main() -> None:
    n_rooms = config.N_E2
    device_key = "cpu-mt"
    thread_count = config.DEVICES.get(device_key, 1)
    print(f"[E2] n_rooms={n_rooms}  device={device_key}  threads={thread_count}")
    print(f"[E2] k sweep = {config.K_ARCHETYPES_SWEEP}")

    rows: List[Dict[str, object]] = []
    for k in config.K_ARCHETYPES_SWEEP:
        if n_rooms % k != 0:
            print(f"[E2] skipping k={k}: {n_rooms} not divisible by {k}")
            continue
        print(f"[E2] k={k:>2d} ...", end=" ", flush=True)
        try:
            row = _run_cell(k, n_rooms, device_key, thread_count)
        except Exception as exc:  # noqa: BLE001
            print(f"FAILED: {exc}")
            continue
        print(
            f"orig={row['t_sim_orig_median']:.3f}s  "
            f"comp={row['t_sim_comp_median']:.3f}s  "
            f"speedup={row['speedup_median']:.2f}x  "
            f"meta={row['n_comp_comp']}"
        )
        rows.append(row)

    common.write_csv(CSV_PATH, rows)
    print(f"\n[E2] wrote {len(rows)} rows -> {CSV_PATH}")


if __name__ == "__main__":
    sys.exit(main())
