import sys
sys.path.append(r"C:\Users\jabj\Documents\python\Twin4Build")
import twin4build as tb
import datetime
from dateutil import tz
import torch
from twin4build.optimizer.optimizer import Optimizer

def main():
    # Create a new model
    model = tb.Model(id="space_heater_test_model")

    # Create space heater
    space_heater = tb.SpaceHeaterStateSpace(
        Q_flow_nominal_sh=1000.0,
        T_a_nominal_sh=60.0,
        T_b_nominal_sh=30.0,
        TAir_nominal_sh=21.0,
        thermalMassHeatCapacity=10000.0,
        nelements=10,
        id="SpaceHeater"
    )

    # Create minimal required schedules
    supply_temp = tb.ScheduleSystem(
        weekDayRulesetDict={
            "ruleset_default_value": 60.0,
            "ruleset_start_minute": [0],
            "ruleset_end_minute": [0],
            "ruleset_start_hour": [0],
            "ruleset_end_hour": [24],
            "ruleset_value": [60.0]
        },
        id="SupplyTempSchedule"
    )

    indoor_temp = tb.ScheduleSystem(
        weekDayRulesetDict={
            "ruleset_default_value": 21.0,
            "ruleset_start_minute": [0],
            "ruleset_end_minute": [0],
            "ruleset_start_hour": [0],
            "ruleset_end_hour": [24],
            "ruleset_value": [21.0]
        },
        id="IndoorTempSchedule"
    )

    mf = space_heater.Q_flow_nominal_sh/4180/(space_heater.T_a_nominal_sh-space_heater.T_b_nominal_sh)
    waterflow_schedule = tb.ScheduleSystem(
        weekDayRulesetDict = {
            "ruleset_default_value": mf,
            "ruleset_start_minute": [0],
            "ruleset_end_minute": [0],
            "ruleset_start_hour": [0],
            "ruleset_end_hour": [24],
            "ruleset_value": [mf]
        },
        id="WaterflowSchedule"
    )

    # Connect schedules to space heater
    model.add_connection(supply_temp, space_heater, "scheduleValue", "supplyWaterTemperature")
    model.add_connection(waterflow_schedule, space_heater, "scheduleValue", "waterFlowRate")
    model.add_connection(indoor_temp, space_heater, "scheduleValue", "indoorTemperature")

    # Load the model
    model.load()

    # Set up simulation parameters
    simulator = tb.Simulator(model)
    stepSize = 600  # 10 minutes in seconds
    startTime = datetime.datetime(
        year=2024, month=1, day=1, hour=0, minute=0, second=0,
        tzinfo=tz.gettz("Europe/Copenhagen")
    )
    endTime = datetime.datetime(
        year=2024, month=1, day=2, hour=0, minute=0, second=0,
        tzinfo=tz.gettz("Europe/Copenhagen")
    )

    # Define optimization targets
    targetInputs = {
        waterflow_schedule: "scheduleValue"  # Optimize the water flow rate
    }
    
    targetOutputs = {
        space_heater: "Power"  # Target the power output
    }

    # Run optimization
    optimizer = Optimizer(simulator)
    optimizer.optimize(
        targetInputs=targetInputs,
        targetOutputs=targetOutputs,
        startTime=startTime,
        endTime=endTime,
        stepSize=stepSize
    )

    # Plot results
    tb.plot.plot_component(
        simulator,
        components_1axis=[
            ("SpaceHeater", "outletWaterTemperature", "output"),
            ("SpaceHeater", "indoorTemperature", "input"),
        ],
        components_2axis=[
            ("SpaceHeater", "Power", "output"),
        ],
        components_3axis=[
            ("SpaceHeater", "waterFlowRate", "input"),
        ],
        ylabel_1axis="Temperature [°C]",
        ylabel_2axis="Power [W]",
        ylabel_3axis="Water flow rate [m³/s]",
        show=True,
        nticks=11
    )

if __name__ == "__main__":
    main() 