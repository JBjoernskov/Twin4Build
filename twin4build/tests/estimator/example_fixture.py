"""Shared estimation-example fixtures owned by the test suite."""

import datetime
import importlib
from pathlib import Path

from dateutil import tz

import twin4build as tb


STEP_SIZE = 1200
EXAMPLE_START = [
    datetime.datetime(2023, 11, 27, tzinfo=tz.gettz("Europe/Copenhagen")),
    datetime.datetime(2023, 12, 2, tzinfo=tz.gettz("Europe/Copenhagen")),
]


def load_model():
    """Load the bundled estimation model and its sensor data."""
    data_dir = Path(tb.__file__).resolve().parent / "examples" / "estimator_example"
    cache_root = str(data_dir.parents[2])
    importlib.import_module("twin4build.utils.get_main_dir")._main_dir = cache_root
    model = tb.Model(id="estimation_test_fixture")
    model.load(
        simulation_model_filename=str(data_dir / "instance_graph.ttl"),
        draw_semantic_model=False,
        draw_simulation_model=False,
    )
    components = model.components
    for component in components.values():
        if hasattr(component, "_cache_root"):
            component._cache_root = cache_root
        if hasattr(component, "cache_root"):
            component.cache_root = cache_root

    def data_path(name):
        return str(data_dir / name)

    components["office_temperature_sensor"].filename = data_path(
        "temperature_sensor.csv"
    )
    components["office_co2_sensor"].filename = data_path("co2_sensor.csv")
    components["office_valve_position_sensor"].filename = data_path(
        "valve_position_sensor.csv"
    )
    components["office_damper_position_sensor"].filename = data_path(
        "damper_position_sensor.csv"
    )
    components["supply_air_temperature_sensor"].filename = data_path(
        "supply_air_temperature.csv"
    )
    components["office_temperature_heating_setpoint"].filename = data_path(
        "temperature_heating_setpoint.csv"
    )
    outdoor = components["outdoor_environment"]
    outdoor.filename_outdoorTemperature = data_path("outdoor_environment.csv")
    outdoor.filename_globalIrradiation = data_path("outdoor_environment.csv")
    outdoor.filename_outdoorCo2Concentration = data_path("outdoor_environment.csv")
    components["office_occupancy"].co2_filename = data_path("co2_sensor.csv")
    components["office_occupancy"].damper_filename = data_path(
        "damper_position_sensor.csv"
    )
    return model


def example_parameters(model):
    """Return the full parameter set used by estimator regression tests."""
    c = model.components
    space = c["office"]
    heater = c["office_space_heater"]
    controller = c["office_temperature_heating_controller"]
    valve = c["office_space_heater_valve"]
    supply = c["office_supply_damper"]
    exhaust = c["office_exhaust_damper"]
    return [
        (space, "thermal.C_air", 2e6, 1e6, 1e7),
        (space, "thermal.C_wall", 2e6, 1e6, 1e7),
        (space, "thermal.C_boundary", 5e5, 1e4, 1e6),
        (space, "thermal.R_out", 0.01, 1e-3, 0.1),
        (space, "thermal.R_in", 0.05, 1e-3, 0.5),
        (space, "thermal.R_boundary", 0.05, 1e-3, 0.5),
        (space, "thermal.f_wall", 0.5, 0.01, 0.99),
        (space, "thermal.f_air", 0.5, 0.01, 0.99),
        (space, "mass.V", 80, 10, 300),
        (space, "mass.G_occ", 5e-6, 1e-6, 1e-5),
        (space, "mass.m_inf", 1e-3, 1e-4, 1e-2),
        (heater, "thermalMassHeatCapacity", 7000, 100, 100000),
        (heater, "UA", 100, 10, 10000),
        (controller, "kp", 0.001, 1e-5, 1.0),
        (controller, "Ti", 3600, 60, 86400),
        (valve, "waterFlowRateMax", 0.016, 1e-4, 1.0),
        (valve, "valveAuthority", 0.5, 0.01, 0.99),
        (supply, "nominalAirFlowRate", 0.1, 0.001, 1.0),
        (exhaust, "nominalAirFlowRate", 0.1, 0.001, 1.0),
    ]


def example_measurements(model):
    """Return the four sensors used by estimator regression tests."""
    c = model.components
    return [
        (c["office_valve_position_sensor"], 0.025),
        (c["office_temperature_sensor"], 0.05),
        (c["office_co2_sensor"], 15.0),
        (c["office_damper_position_sensor"], 0.025),
    ]
