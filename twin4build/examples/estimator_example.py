import sys
sys.path.append(r"C:\Users\jabj\Documents\python\Twin4Build")

import twin4build as tb
import datetime
from dateutil import tz
import twin4build.examples.utils as utils

def main():
    # Create a new model
    model = tb.Model(id="estimator_example")
    
    # Load the model from semantic file
    filename_simulation = utils.get_path(["generated_files", "models", "translator_example", "simulation_model", "semantic_model", "semantic_model.ttl"])
    print(filename_simulation)
    model.load(simulation_model_filename=filename_simulation, verbose=False)

    # Set up simulation parameters
    simulator = tb.Simulator(model)
    stepSize = 1200  # 40 minutes in seconds
    startTime = datetime.datetime(year=2023, month=11, day=27, hour=0, minute=0, second=0,
                                    tzinfo=tz.gettz("Europe/Copenhagen"))
    endTime = datetime.datetime(year=2023, month=12, day=1, hour=0, minute=0, second=0,
                                tzinfo=tz.gettz("Europe/Copenhagen"))

    space = model.components["020B"]
    space_heater = model.components["020B_space_heater"]
    heating_controller = model.components["020B_temperature_heating_controller"]
    co2_controller = model.components["020B_co2_controller"]
    space_heater_valve = model.components["020B_space_heater_valve"]
    supply_damper = model.components["020B_room_supply_damper"]
    exhaust_damper = model.components["020B_room_exhaust_damper"]

    targetParameters = {"private": {
                                # Thermal parameters
                                "thermal.C_air": {"components": [space], "x0": 2e+6, "lb": 1e+6, "ub": 1e+7},                # Thermal capacitance of indoor air [J/K]
                                "thermal.C_wall": {"components": [space], "x0": 2e+6, "lb": 1e+6, "ub": 1e+7},               # Thermal capacitance of exterior wall [J/K]
                                "thermal.C_boundary": {"components": [space], "x0": 5e+5, "lb": 1e+4, "ub": 1e+6},               # Thermal capacitance of exterior wall [J/K]
                                "thermal.R_out": {"components": [space], "x0": 0.05, "lb": 0.01, "ub": 1},                # Thermal resistance between wall and outdoor [K/W]
                                "thermal.R_in": {"components": [space], "x0": 0.05, "lb": 0.01, "ub": 1},                 # Thermal resistance between wall and indoor [K/W]
                                "thermal.R_boundary": {"components": [space], "x0": 0.01, "lb": 0.0001, "ub": 1},                 # Thermal resistance between wall and indoor [K/W]
                                "thermal.f_wall": {"components": [space], "x0": 0.3, "lb": 0, "ub": 1},         # Radiation factor for exterior wall
                                "thermal.f_air": {"components": [space], "x0": 0.1, "lb": 0, "ub": 1},          # Radiation factor for air
                                "thermal.Q_occ_gain": {"components": [space], "x0": 100.0, "lb": 10, "ub": 200},   # Heat gain per occupant [W]

                                # Mass parameters
                                "mass.V": {"components": [space], "x0": 100, "lb": 10, "ub": 1000}, # Volume of the space [m³]
                                "mass.G_occ": {"components": [space], "x0": 8.18e-6, "lb": 1e-8, "ub": 1e-4}, # CO2 generation rate per occupant [ppm·kg/s]
                                "mass.m_inf": {"components": [space], "x0": 0.001, "lb": 1e-6, "ub": 0.3}, # Infiltration rate [kg/s]

                                # Space heater parameters
                                "thermalMassHeatCapacity": {"components": [space_heater], "x0": 10000, "lb": 1000, "ub": 50000}, # Thermal mass heat capacity [J/K]
                                "UA": {"components": [space_heater], "x0": 30, "lb": 1, "ub": 100}, # Thermal conductance [W/K]

                                # Heating controller parameters
                                "kp": {"components": [heating_controller, co2_controller], "x0": [0.001, -0.001], "lb": [1e-5, -1], "ub": [1, -1e-5]}, # Proportional gain
                                "Ti": {"components": [heating_controller, co2_controller], "x0": [10, 10], "lb": [1, 1], "ub": [100, 100]}, # Integral gain
                                "Td": {"components": [heating_controller, co2_controller], "x0": [0, 0], "lb": [0, 0], "ub": [1, 1]}, # Derivative gain

                                # Space heater valve parameters
                                "waterFlowRateMax": {"components": [space_heater_valve], "x0": 0.01, "lb": 1e-6, "ub": 0.1}, # Maximum water flow rate [m³/s]
                                "valveAuthority": {"components": [space_heater_valve], "x0": 0.8, "lb": 0.4, "ub": 1}, # Valve authority

                                # Damper parameters
                                "a": {"components": [supply_damper, exhaust_damper], "x0": 1, "lb": 1, "ub": 10}, # Shape parameter
                                "nominalAirFlowRate": {"components": [supply_damper, exhaust_damper], "x0": 0.001, "lb": 1e-5, "ub": 1}, # Maximum water flow rate [m³/s]
                                }}

    targetMeasuringDevices = [model.components["020B_valve_position_sensor"],
                                model.components["020B_temperature_sensor"],
                                model.components["020B_co2_sensor"],
                                model.components["020B_damper_position_sensor"]]

    
    # # Run initial simulation for comparison
    simulator.simulate(
        stepSize=stepSize,
        startTime=startTime,
        endTime=endTime
    )

    
    # Plot initial results
    fig, axes = tb.plot.plot_component(
        simulator,
        components_1axis=[
            ("020B", "indoorTemperature", "output"),
            ("outdoor_environment", "outdoorTemperature", "output"),
            (heating_controller.id, "setpointValue", "input"),
            # (estimator.actual_readings[model.components["020B_temperature_sensor"].id], "Actual temperature"),
        ],
        components_2axis=[
            ("020B_space_heater", "Power", "output"),
            ("020B", "heatGain", "input"),

        ],
        components_3axis=[
            # ("020B_space_heater", "waterFlowRate", "input"),
            # (heating_controller.id, "setpointValue", "input"),
            (heating_controller.id, "inputSignal", "output"),
        ],
        ylabel_1axis="Temperature [°C]",
        ylabel_2axis="Power [W]",
        ylabel_3axis="Water flow rate [m³/s]",
        title="Before calibration",
        show=True,
        nticks=11
    )

    # Create estimator
    estimator = tb.Estimator(simulator)
    
    options = {"maxiter": 150,
               "disp": True}
    

    # 400 secs with scipy_solver
    # Time and run LS_AD method
    estimator.estimate(
        targetParameters=targetParameters,
        targetMeasuringDevices=targetMeasuringDevices,
        startTime=startTime,
        endTime=endTime,
        stepSize=stepSize,
        n_initialization_steps=20,
        method=("scipy", "SLSQP", "ad"),
        options=options,
    )

    # Plot results
    fig, axes = tb.plot.plot_component(
        simulator,
        components_1axis=[
            ("020B", "indoorTemperature", "output"),
            # ("outdoor_environment", "outdoorTemperature", "output"),
            (heating_controller.id, "setpointValue", "input"),
            (estimator.actual_readings[model.components["020B_temperature_sensor"].id], "Actual temperature"),
        ],
        components_2axis=[
            ("020B_space_heater", "Power", "output"),
            # ("020B", "heatGain", "input"),

        ],
        components_3axis=[
            ("020B_space_heater_valve", "valvePosition", "output"),
        ],
        ylabel_1axis="Temperature [°C]",
        ylabel_2axis="Power [W]",
        ylabel_3axis="Valve position [0-1]",
        title="After calibration",
        show=False,
        nticks=11
    )

    fig, axes = tb.plot.plot_component(
        simulator,
        components_1axis=[
            ("020B_valve_position_sensor", "measuredValue", "input"),
            (estimator.actual_readings[model.components["020B_valve_position_sensor"].id], "Actual valve position"),
        ],
        ylabel_1axis="Valve position [0-1]",
        title="Valve position comparison",
        show=False,
        nticks=11
    )

    fig, axes = tb.plot.plot_component(
        simulator,
        components_1axis=[
            ("020B_temperature_sensor", "measuredValue", "input"),
            (estimator.actual_readings[model.components["020B_temperature_sensor"].id], "Actual temperature"),
        ],
        ylabel_1axis="Temperature [°C]",
        title="Temperature comparison",
        show=True,
        nticks=11
    )


if __name__ == "__main__":
    main()