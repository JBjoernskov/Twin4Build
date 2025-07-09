#!/usr/bin/env python3
"""
Example demonstrating the use of the trust-constr optimizer in Twin4Build.

This example shows how to use the new streamlined trust-constr implementation
with automatic differentiation for gradients and hessians.
"""

import twin4build as tb
import datetime
import pytz
import numpy as np
import torch

def main():
    # Create a simple building model
    model = tb.SimulationModel(id="simple_building")
    
    # Create components
    space = tb.Space(id="space")
    space_heater = tb.SpaceHeater(id="space_heater")
    
    # Add components to model
    model.add_component(space)
    model.add_component(space_heater)
    
    # Create schedules
    occupancy_schedule = tb.ScheduleSystem(
        id="occupancy_schedule",
        weekDayRulesetDict={
            "ruleset_default_value": 0,
            "ruleset_start_minute": 0,
            "ruleset_end_minute": 0,
            "ruleset_start_hour": 0,
            "ruleset_end_hour": 0,
            "ruleset_value": 0,
            "ruleset_monday_value": 0,
            "ruleset_tuesday_value": 0,
            "ruleset_wednesday_value": 0,
            "ruleset_thursday_value": 0,
            "ruleset_friday_value": 0,
            "ruleset_saturday_value": 0,
            "ruleset_sunday_value": 0,
        },
        add_noise=False
    )
    
    outdoor_temp_schedule = tb.ScheduleSystem(
        id="outdoor_temp_schedule",
        weekDayRulesetDict={
            "ruleset_default_value": 20,
            "ruleset_start_minute": 0,
            "ruleset_end_minute": 0,
            "ruleset_start_hour": 0,
            "ruleset_end_hour": 0,
            "ruleset_value": 20,
            "ruleset_monday_value": 20,
            "ruleset_tuesday_value": 20,
            "ruleset_wednesday_value": 20,
            "ruleset_thursday_value": 20,
            "ruleset_friday_value": 20,
            "ruleset_saturday_value": 20,
            "ruleset_sunday_value": 20,
        },
        add_noise=False
    )
    
    # Connect components
    space_heater.connect_input("airFlowRate", space, "airFlowRate")
    space_heater.connect_input("inletWaterTemperature", space, "inletWaterTemperature")
    space.connect_input("indoorTemperature", space_heater, "outletAirTemperature")
    space.connect_input("numberOfPeople", occupancy_schedule, "scheduleValue")
    space.connect_input("outdoorTemperature", outdoor_temp_schedule, "scheduleValue")
    
    # Create simulator
    simulator = tb.Simulator(model)
    
    # Create optimizer
    optimizer = tb.Optimizer(simulator)
    
    # Define optimization parameters
    start_time = datetime.datetime(2024, 1, 1, tzinfo=pytz.UTC)
    end_time = datetime.datetime(2024, 1, 2, tzinfo=pytz.UTC)
    step_size = 3600  # 1 hour
    
    # Define decision variables (normalized values)
    # x0 = np.array([0.5, 0])  # Example initial values
    decision_variables = [
        (space_heater, "airFlowRate", 0.0, 1.0),  # Normalized bounds
        (space_heater, "inletWaterTemperature", 0.0, 1.0)  # Normalized bounds
    ]
    
    # Define minimization objectives
    minimize = [
        (space, "indoorTemperature")  # Minimize indoor temperature
    ]
    
    # Define constraints (optional)
    equality_constraints = [
        # (space, "indoorTemperature", 22.0)  # Keep temperature at 22°C
    ]
    
    inequality_constraints = [
        (space, "indoorTemperature", "upper", 25.0),  # Temperature <= 25°C
        (space, "indoorTemperature", "lower", 18.0)   # Temperature >= 18°C
    ]
    
    # Run optimization with trust-constr method
    print("Running trust-constr optimization...")
    result = optimizer.optimize(
        decisionVariables=decision_variables,
        minimize=minimize,
        equalityConstraints=equality_constraints,
        inequalityConstraints=inequality_constraints,
        startTime=start_time,
        endTime=end_time,
        stepSize=step_size,
        method="trust-constr",
        options={
            'verbose': 2,
            'maxiter': 50,
            'gtol': 1e-6,
            'xtol': 1e-6
        }
    )
    
    print("\nOptimization Results:")
    print(f"Success: {result.success}")
    print(f"Final objective value: {result.fun}")
    print(f"Number of iterations: {result.nit}")
    print(f"Number of function evaluations: {result.nfev}")
    print(f"Optimal decision variables: {result.x}")
    
    # Print final values
    print("\nFinal Decision Variable Values:")
    for i, (component, output_name, *bounds) in enumerate(decision_variables):
        if component.output[output_name].do_normalization:
            final_value = component.output[output_name].denormalize(torch.tensor(result.x[i]))
            print(f"{component.id}.{output_name}: {final_value.item():.4f}")
        else:
            print(f"{component.id}.{output_name}: {result.x[i]:.4f}")

if __name__ == "__main__":
    main() 