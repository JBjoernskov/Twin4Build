import sys
sys.path.append(r"C:\Users\jabj\Documents\python\Twin4Build")
import twin4build as tb
import datetime
from dateutil import tz
import torch
from twin4build.optimizer.optimizer import Optimizer

def main():
    # Create a new model
    model = tb.Model(id="schedule_test_model")

    # Create a simple schedule
    schedule = tb.ScheduleSystem(
        weekDayRulesetDict={
            "ruleset_default_value": 0.0,
            "ruleset_start_minute": [0,0,0,0],
            "ruleset_end_minute": [0,0,0,0],
            "ruleset_start_hour": [1, 6, 12, 18],
            "ruleset_end_hour": [2,8,16,22],
            "ruleset_value": [0.0,0.5,0.3,0.1]
        },
        id="TestSchedule"
    )

    sensor = tb.SensorSystem(
        id="Sensor"
    )

    model.add_connection(schedule, sensor, "scheduleValue", "measuredValue")
    

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
        schedule: "scheduleValue"  # Optimize the schedule value
    }
    
    targetOutputs = {
        sensor: "measuredValue"  # Target the same schedule value
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
            ("TestSchedule", "scheduleValue", "output"),
        ],
        ylabel_1axis="Schedule Value",
        show=True,
        nticks=11
    )

if __name__ == "__main__":
    main() 