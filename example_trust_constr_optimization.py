"""
Example script demonstrating the use of Trust-Region Constrained Algorithm
in the twin4build optimizer.

This example shows how to use the new SciPy-based optimization methods
alongside the existing PyTorch-based methods.
"""

import twin4build as tb
import datetime
import pytz
import numpy as np

def example_trust_constr_optimization():
    """
    Example of using the Trust-Region Constrained Algorithm for optimization.
    """
    
    # Create a simple model (you would replace this with your actual model)
    model = tb.SimulationModel(id="example_model")
    
    # Add some components to your model
    # (This is a placeholder - you would add your actual components)
    
    # Create simulator
    simulator = tb.Simulator(model)
    
    # Create optimizer
    optimizer = tb.Optimizer(simulator)
    
    # Define decision variables (components to optimize)
    # Format: (component, 'output_name', lower_bound, upper_bound)
    decisionVariables = [
        # Example: optimize temperature setpoint between 20-25°C
        # (component, 'temperature_setpoint', 20.0, 25.0),
    ]
    
    # Define minimization objectives
    # Format: (component, 'output_name')
    minimize = [
        # Example: minimize energy consumption
        # (energy_meter, 'energy_consumption'),
    ]
    
    # Define equality constraints
    # Format: (component, 'output_name', desired_value)
    equalityConstraints = [
        # Example: maintain room temperature at 22°C
        # (room_sensor, 'temperature', 22.0),
    ]
    
    # Define inequality constraints
    # Format: (component, 'output_name', constraint_type, desired_value)
    inequalityConstraints = [
        # Example: keep humidity below 60%
        # (humidity_sensor, 'humidity', 'upper', 60.0),
        # Example: keep temperature above 18°C
        # (room_sensor, 'temperature', 'lower', 18.0),
    ]
    
    # Define time period
    start_time = datetime.datetime(2024, 1, 1, tzinfo=pytz.UTC)
    end_time = datetime.datetime(2024, 1, 2, tzinfo=pytz.UTC)
    step_size = 3600  # 1 hour in seconds
    
    # Example 1: Use Trust-Region Constrained Algorithm
    print("=== Example 1: Trust-Region Constrained Algorithm ===")
    result_trust_constr = optimizer.optimize(
        decisionVariables=decisionVariables,
        minimize=minimize,
        equalityConstraints=equalityConstraints,
        inequalityConstraints=inequalityConstraints,
        startTime=start_time,
        endTime=end_time,
        stepSize=step_size,
        method="trust-constr",
        options={
            'verbose': 2,  # Show optimization progress
            'maxiter': 100,  # Maximum iterations
            'gtol': 1e-6,  # Gradient tolerance
            'xtol': 1e-6,  # Parameter tolerance
        }
    )
    
    print(f"Trust-constr result: {result_trust_constr}")
    
    # Example 2: Use SLSQP (Sequential Least SQuares Programming)
    print("\n=== Example 2: SLSQP Algorithm ===")
    result_slsqp = optimizer.optimize(
        decisionVariables=decisionVariables,
        minimize=minimize,
        equalityConstraints=equalityConstraints,
        inequalityConstraints=inequalityConstraints,
        startTime=start_time,
        endTime=end_time,
        stepSize=step_size,
        method="SLSQP",
        options={
            'ftol': 1e-6,  # Function tolerance
            'maxiter': 100,  # Maximum iterations
            'disp': True,  # Show optimization progress
        }
    )
    
    print(f"SLSQP result: {result_slsqp}")
    
    # Example 3: Use COBYLA (derivative-free method)
    print("\n=== Example 3: COBYLA Algorithm ===")
    result_cobyla = optimizer.optimize(
        decisionVariables=decisionVariables,
        minimize=minimize,
        equalityConstraints=equalityConstraints,
        inequalityConstraints=inequalityConstraints,
        startTime=start_time,
        endTime=end_time,
        stepSize=step_size,
        method="COBYLA",
        options={
            'rhobeg': 1.0,  # Initial trust region radius
            'rhoend': 1e-6,  # Final trust region radius
            'maxiter': 100,  # Maximum iterations
            'disp': True,  # Show optimization progress
        }
    )
    
    print(f"COBYLA result: {result_cobyla}")
    
    # Example 4: Compare with PyTorch method (1st_order)
    print("\n=== Example 4: PyTorch Method (1st_order) ===")
    result_pytorch = optimizer.optimize(
        decisionVariables=decisionVariables,
        minimize=minimize,
        equalityConstraints=equalityConstraints,
        inequalityConstraints=inequalityConstraints,
        startTime=start_time,
        endTime=end_time,
        stepSize=step_size,
        method="1st_order",
        options={
            'lr': 0.01,  # Learning rate
            'iterations': 50,  # Number of iterations
            'optimizer_type': "Adam",  # Optimizer type
            'scheduler_type': "step",  # Scheduler type
            'scheduler_params': {
                'step_size': 20,
                'gamma': 0.5
            }
        }
    )
    
    print("PyTorch optimization completed")

def example_with_actual_components():
    """
    Example showing how to use the optimizer with actual twin4build components.
    This is a more realistic example that you would adapt to your specific model.
    """
    
    # Create model
    model = tb.SimulationModel(id="building_model")
    
    # Add components to your model
    # Example components (you would replace with your actual components):
    
    # 1. HVAC system
    # hvac = tb.HVACSystem(id="hvac_1")
    # model.add_component(hvac)
    
    # 2. Room
    # room = tb.Room(id="room_1")
    # model.add_component(room)
    
    # 3. Temperature sensor
    # temp_sensor = tb.TemperatureSensor(id="temp_sensor_1")
    # model.add_component(temp_sensor)
    
    # 4. Energy meter
    # energy_meter = tb.EnergyMeter(id="energy_meter_1")
    # model.add_component(energy_meter)
    
    # Connect components
    # model.connect(hvac, "supply_temperature", room, "heating_input")
    # model.connect(room, "temperature", temp_sensor, "temperature")
    # model.connect(hvac, "power_consumption", energy_meter, "power")
    
    # Create simulator
    simulator = tb.Simulator(model)
    
    # Create optimizer
    optimizer = tb.Optimizer(simulator)
    
    # Define optimization problem
    decisionVariables = [
        # Optimize HVAC setpoint temperature
        # (hvac, "setpoint_temperature", 18.0, 26.0),
    ]
    
    minimize = [
        # Minimize energy consumption
        # (energy_meter, "total_energy"),
    ]
    
    equalityConstraints = [
        # Maintain room temperature at 22°C
        # (temp_sensor, "temperature", 22.0),
    ]
    
    inequalityConstraints = [
        # Keep room temperature between 18-26°C
        # (temp_sensor, "temperature", "lower", 18.0),
        # (temp_sensor, "temperature", "upper", 26.0),
    ]
    
    # Time period
    start_time = datetime.datetime(2024, 1, 1, tzinfo=pytz.UTC)
    end_time = datetime.datetime(2024, 1, 7, tzinfo=pytz.UTC)  # One week
    step_size = 3600  # 1 hour
    
    # Run optimization with trust-constr
    print("Running optimization with Trust-Region Constrained Algorithm...")
    result = optimizer.optimize(
        decisionVariables=decisionVariables,
        minimize=minimize,
        equalityConstraints=equalityConstraints,
        inequalityConstraints=inequalityConstraints,
        startTime=start_time,
        endTime=end_time,
        stepSize=step_size,
        method="trust-constr",
        options={
            'verbose': 2,
            'maxiter': 200,
            'gtol': 1e-8,
            'xtol': 1e-8,
        }
    )
    
    print(f"Optimization completed!")
    print(f"Success: {result.success}")
    print(f"Final objective value: {result.fun}")
    print(f"Number of iterations: {result.nit}")
    print(f"Number of function evaluations: {result.nfev}")
    
    # Access the optimized solution
    if result.success:
        print("Optimization was successful!")
        print(f"Optimal decision variables: {result.x}")
        
        # You can now use the optimized values in your model
        # The optimizer has already updated the model with the optimal values
        
    else:
        print("Optimization failed or did not converge.")
        print(f"Message: {result.message}")

def example_method_comparison():
    """
    Example comparing different optimization methods.
    """
    
    # Create model and simulator
    model = tb.SimulationModel(id="comparison_model")
    simulator = tb.Simulator(model)
    optimizer = tb.Optimizer(simulator)
    
    # Define a simple optimization problem
    decisionVariables = [
        # (component, 'output_name', lower_bound, upper_bound)
    ]
    
    minimize = [
        # (component, 'output_name')
    ]
    
    equalityConstraints = [
        # (component, 'output_name', desired_value)
    ]
    
    inequalityConstraints = [
        # (component, 'output_name', 'upper', max_value),
        # (component, 'output_name', 'lower', min_value)
    ]
    
    # Time period
    start_time = datetime.datetime(2024, 1, 1, tzinfo=pytz.UTC)
    end_time = datetime.datetime(2024, 1, 2, tzinfo=pytz.UTC)
    step_size = 3600
    
    # Compare different methods
    methods = [
        ("1st_order", {
            'lr': 0.01,
            'iterations': 50,
            'optimizer_type': "Adam"
        }),
        ("trust-constr", {
            'verbose': 1,
            'maxiter': 100,
            'gtol': 1e-6,
            'xtol': 1e-6
        }),
        ("SLSQP", {
            'ftol': 1e-6,
            'maxiter': 100,
            'disp': False
        }),
        ("COBYLA", {
            'rhobeg': 1.0,
            'rhoend': 1e-6,
            'maxiter': 100,
            'disp': False
        })
    ]
    
    results = {}
    
    for method_name, method_options in methods:
        print(f"\n=== Testing {method_name} method ===")
        try:
            result = optimizer.optimize(
                decisionVariables=decisionVariables,
                minimize=minimize,
                equalityConstraints=equalityConstraints,
                inequalityConstraints=inequalityConstraints,
                startTime=start_time,
                endTime=end_time,
                stepSize=step_size,
                method=method_name,
                options=method_options
            )
            results[method_name] = result
            print(f"{method_name}: Success={result.success}, Fun={result.fun}, Iterations={result.nit}")
        except Exception as e:
            print(f"{method_name}: Failed with error - {e}")
            results[method_name] = None
    
    # Summary
    print("\n=== Method Comparison Summary ===")
    for method_name, result in results.items():
        if result is not None:
            print(f"{method_name}: Success={result.success}, Final Value={result.fun:.6f}, Iterations={result.nit}")
        else:
            print(f"{method_name}: Failed")

if __name__ == "__main__":
    print("Twin4Build Trust-Region Constrained Optimization Example")
    print("=" * 60)
    
    # Run the basic example
    example_trust_constr_optimization()
    
    print("\n" + "=" * 60)
    print("Note: The examples above use placeholder components.")
    print("To use with actual components, uncomment and adapt the code")
    print("in example_with_actual_components().")
    
    # Uncomment the lines below to run with actual components
    # example_with_actual_components()
    # example_method_comparison() 