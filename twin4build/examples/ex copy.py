
import sys
sys.path.append(r"C:\Users\jabj\Documents\python\Twin4Build")
import twin4build as tb
import datetime
from dateutil import tz
import twin4build.utils.plot.plot as plot
import twin4build.examples.utils as utils
from dateutil.tz import gettz
import twin4build.systems as systems

if __name__ == "__main__":

    # Create a new model
    model = tb.Model(id="mymodel")
    filename = utils.get_path(["parameter_estimation_example", "one_room_example_model.xlsm"])


    def fcn(self):
        supply_water_schedule = systems.ScheduleSystem(
        weekDayRulesetDict = {
            "ruleset_default_value": 60,
            "ruleset_start_minute": [],
            "ruleset_end_minute": [],
            "ruleset_start_hour": [],
            "ruleset_end_hour": [],
            "ruleset_value": []
        },
        id="supply_water_schedule"
        )




        self.add_connection(supply_water_schedule, self.components["[020B][020B_space_heater]"], "scheduleValue", "supplyWaterTemperature") # Add missing input
        self.components["020B_temperature_sensor"].filename = utils.get_path(["parameter_estimation_example", "temperature_sensor.csv"])
        self.components["020B_co2_sensor"].filename = utils.get_path(["parameter_estimation_example", "co2_sensor.csv"])
        self.components["020B_valve_position_sensor"].filename = utils.get_path(["parameter_estimation_example", "valve_position_sensor.csv"])
        self.components["020B_damper_position_sensor"].filename = utils.get_path(["parameter_estimation_example", "damper_position_sensor.csv"])
        self.components["BTA004"].filename = utils.get_path(["parameter_estimation_example", "supply_air_temperature.csv"])
        self.components["020B_co2_setpoint"].weekDayRulesetDict = {"ruleset_default_value": 900,
                                                                        "ruleset_start_minute": [],
                                                                        "ruleset_end_minute": [],
                                                                        "ruleset_start_hour": [],
                                                                        "ruleset_end_hour": [],
                                                                        "ruleset_value": []}
        self.components["020B_occupancy_profile"].weekDayRulesetDict = {"ruleset_default_value": 0,
                                                                        "ruleset_start_minute": [],
                                                                        "ruleset_end_minute": [],
                                                                        "ruleset_start_hour": [],
                                                                        "ruleset_end_hour": [],
                                                                        "ruleset_value": []}
        self.components["020B_temperature_heating_setpoint"].useFile = True
        self.components["020B_temperature_heating_setpoint"].filename = utils.get_path(["parameter_estimation_example", "temperature_heating_setpoint.csv"])
        self.components["outdoor_environment"].filename = utils.get_path(["parameter_estimation_example", "outdoor_environment.csv"])


    model.load(semantic_model_filename=filename, fcn=fcn, verbose=False)




    stepSize = 600  # Seconds
    startTime = datetime.datetime(year=2023, month=11, day=27, hour=0, minute=0, second=0,
                                    tzinfo=gettz("Europe/Copenhagen"))
    endTime = datetime.datetime(year=2023, month=12, day=7, hour=0, minute=0, second=0,
                                tzinfo=gettz("Europe/Copenhagen"))
    space = model.components["[020B][020B_space_heater]"]
    heating_controller = model.components["020B_temperature_heating_controller"]
    co2_controller = model.components["020B_co2_controller"]
    space_heater_valve = model.components["020B_space_heater_valve"]
    supply_damper = model.components["020B_room_supply_damper"]
    exhaust_damper = model.components["020B_room_exhaust_damper"]

    space.CO2_start = 400
    space.fraRad_sh = 0.35
    space.T_a_nominal_sh = 333.15
    space.T_b_nominal_sh = 303.15
    space.TAir_nominal_sh = 293.15
    space.airVolume = 125.82


    targetParameters = {"private": {"C_wall": {"components": [space], "x0": 1.7e+6, "lb": 1e+6, "ub": 2e+6}, #1.5e+6
                                    "C_air": {"components": [space], "x0": 1.7e+6, "lb": 1e+4, "ub": 2e+6}, #3e+6
                                    "C_boundary": {"components": [space], "x0": 2e+4, "lb": 1e+4, "ub": 2e+5}, #1e+5
                                    "R_out": {"components": [space], "x0": 0.014, "lb": 1e-3, "ub": 0.5}, #0.2
                                    "R_in": {"components": [space], "x0": 0.024, "lb": 1e-3, "ub": 0.5}, #0.2
                                    "R_boundary": {"components": [space], "x0": 0.001, "lb": 9.9e-4, "ub": 0.3}, #0.005
                                    "f_wall": {"components": [space], "x0": 0.5, "lb": 0, "ub": 2}, #1
                                    "f_air": {"components": [space], "x0": 0.5, "lb": 0, "ub": 2}, #1
                                    "kp": {"components": [heating_controller, co2_controller], "x0": 1e-3, "lb": 1e-6, "ub": 3}, #1e-3
                                    "Ti": {"components": [heating_controller, co2_controller], "x0": 5, "lb": 1e-5, "ub": 10}, #3
                                    "m_flow_nominal": {"components": [space_heater_valve], "x0": 0.0202, "lb": 1e-3, "ub": 0.5}, #0.0202
                                    "dpFixed_nominal": {"components": [space_heater_valve], "x0": 1, "lb": 0, "ub": 10000}, #2000
                                    "T_boundary": {"components": [space], "x0": 21, "lb": 19, "ub": 24}, #20
                                    "a": {"components": [supply_damper, exhaust_damper], "x0": 5, "lb": 0.5, "ub": 8}, #2
                                    "infiltration": {"components": [space], "x0": 0.001, "lb": 1e-4, "ub": 0.3}, #0.001
                                    "Q_occ_gain": {"components": [space], "x0": 50, "lb": 10, "ub": 1000}, #100,
                                    "C_supply": {"components": [space], "x0": 400, "lb": 100, "ub": 600}, #400
                                    "Q_flow_nominal_sh": {"components": [space], "x0": 500, "lb": 10, "ub": 1000}, #100,
                                    "n_sh": {"components": [space], "x0": 1.24, "lb": 1, "ub": 2}, #1
                                    "CO2_occ_gain": {"components": [space], "x0": 8.18e-6, "lb": 1e-8, "ub": 1e-4}, #100,
                                    }}


    percentile = 2
    targetMeasuringDevices = {model.components["020B_valve_position_sensor"]: {"standardDeviation": 0.01/percentile, "scale_factor": 1},
                                model.components["020B_temperature_sensor"]: {"standardDeviation": 0.1/percentile, "scale_factor": 20},
                                model.components["020B_co2_sensor"]: {"standardDeviation": 10/percentile, "scale_factor": 400},
                                model.components["020B_damper_position_sensor"]: {"standardDeviation": 0.01/percentile, "scale_factor": 1},
                                }

    # Options for the estimation method. If the options argument is not supplied or None is supplied, default options are applied.
    options = {"ftol": 1e-10,
                "xtol": 1e-14,
                "verbose": 2}
    estimator = tb.Estimator(model)
    estimator.estimate(targetParameters=targetParameters,
                        targetMeasuringDevices=targetMeasuringDevices,
                        startTime=startTime,
                        endTime=endTime,
                        stepSize=stepSize,
                        n_initialization_steps=288,
                        method="LS", #Use Least Squares instead
                        options=options)
    model.load_estimation_result(estimator.result_savedir_pickle)

    print("SOLUTION")
    print("C_wall: ", space.C_wall)
    print("C_air: ", space.C_air)
    print("C_boundary: ", space.C_boundary)
    print("R_out: ", space.R_out)
    print("R_in: ", space.R_in)
    print("R_boundary: ", space.R_boundary)
    print("f_wall: ", space.f_wall)
    print("f_air: ", space.f_air)
    print("Q_occ_gain: ", space.Q_occ_gain)
    print("kp: ", heating_controller.kp)
    print("Ti: ", heating_controller.Ti)
    print("kp: ", co2_controller.kp)
    print("Ti: ", co2_controller.Ti)
    print("m_flow_nominal: ", space_heater_valve.m_flow_nominal)
    print("dpFixed_nominal: ", space_heater_valve.dpFixed_nominal)
    print("T_boundary: ", space.T_boundary)
    print("a: ", supply_damper.a)
    print("a: ", exhaust_damper.a)
    print("infiltration: ", space.infiltration)
    print("CO2_occ_gain: ", space.CO2_occ_gain)
    print("Q_flow_nominal_sh: ", space.Q_flow_nominal_sh)
    print("n_sh: ", space.n_sh)
    print("C_supply: ", space.C_supply)



    monitor = tb.Monitor(model) #Compares the simulation results with the measured results
    monitor.monitor(startTime=startTime,
                    endTime=endTime,
                    stepSize=stepSize,
                    show=True)



