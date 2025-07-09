# Trust-Region Constrained Optimization in Twin4Build

This document describes the implementation of SciPy-based optimization methods in the Twin4Build optimizer, including the Trust-Region Constrained Algorithm.

## Overview

The Twin4Build optimizer now supports multiple optimization methods:

1. **PyTorch-based methods** (original): 1st_order with automatic differentiation
2. **SciPy-based methods** (new): trust-constr, SLSQP, COBYLA, COBYQA

## New Optimization Methods

### 1. Trust-Region Constrained Algorithm (`trust-constr`)

The Trust-Region Constrained Algorithm is a robust optimization method that handles both equality and inequality constraints effectively. It's particularly well-suited for problems with:

- Nonlinear constraints
- Bounds on decision variables
- Equality and inequality constraints
- Problems where gradient information is available

**Key Features:**
- Handles equality constraints as `NonlinearConstraint` objects
- Handles inequality constraints as `NonlinearConstraint` objects with bounds
- Supports bounds on decision variables
- Uses gradient information when available
- Robust convergence properties

### 2. Sequential Least SQuares Programming (`SLSQP`)

SLSQP is another constrained optimization method that's efficient for problems with:

- Linear and nonlinear constraints
- Bounds on decision variables
- Problems where gradient information is available

**Key Features:**
- Handles equality constraints as `{'type': 'eq', 'fun': constraint_function}`
- Handles inequality constraints as `{'type': 'ineq', 'fun': constraint_function}`
- Supports bounds on decision variables
- Uses gradient information

### 3. COBYLA and COBYQA

These are derivative-free optimization methods suitable for:

- Problems where gradients are difficult to compute
- Problems with noisy objective functions
- Problems with constraints

**Key Features:**
- No gradient information required
- Handles constraints through penalty functions
- Robust to noisy objective functions

## Usage

### Basic Usage

```python
import twin4build as tb
import datetime
import pytz

# Create your model and simulator
model = tb.SimulationModel(id="my_model")
simulator = tb.Simulator(model)
optimizer = tb.Optimizer(simulator)

# Define optimization problem
decisionVariables = [
    (component, 'output_name', lower_bound, upper_bound)
]

minimize = [
    (component, 'output_name')
]

equalityConstraints = [
    (component, 'output_name', desired_value)
]

inequalityConstraints = [
    (component, 'output_name', 'upper', max_value),
    (component, 'output_name', 'lower', min_value)
]

# Time period
start_time = datetime.datetime(2024, 1, 1, tzinfo=pytz.UTC)
end_time = datetime.datetime(2024, 1, 2, tzinfo=pytz.UTC)
step_size = 3600

# Run optimization with trust-constr
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
        'maxiter': 100,
        'gtol': 1e-6,
        'xtol': 1e-6,
    }
)
```

### Method Comparison

| Method | Gradient Required | Constraint Support | Bounds Support | Best For |
|--------|------------------|-------------------|----------------|----------|
| `1st_order` | Yes (automatic) | Penalty-based | Clamping | Fast, differentiable models |
| `trust-constr` | Yes (finite diff) | Full support | Yes | Constrained problems |
| `SLSQP` | Yes (finite diff) | Full support | Yes | Constrained problems |
| `COBYLA` | No | Penalty-based | Limited | Noisy/derivative-free |
| `COBYQA` | No | Penalty-based | Limited | Noisy/derivative-free |

### Configuration Options

#### 1st_order (PyTorch) Options

```python
options = {
    'lr': 1.0,                    # Learning rate for optimizer
    'iterations': 100,            # Number of optimization iterations
    'optimizer_type': "SGD",      # Type of PyTorch optimizer ("SGD", "Adam", "LBFGS")
    'scheduler_type': "step",     # Type of learning rate scheduler
    'scheduler_params': {         # Parameters for learning rate scheduler
        'step_size': 30,
        'gamma': 0.1
    }
}
```

#### Trust-Region Constrained Algorithm Options

```python
options = {
    'verbose': 2,                 # Verbosity level (0-3)
    'maxiter': 100,              # Maximum iterations
    'gtol': 1e-6,               # Gradient tolerance
    'xtol': 1e-6,               # Parameter tolerance
    'barrier_tol': 1e-8,        # Barrier tolerance
    'initial_tr_radius': 1.0,   # Initial trust region radius
    'initial_constr_penalty': 1.0,  # Initial constraint penalty
}
```

#### SLSQP Options

```python
options = {
    'ftol': 1e-6,               # Function tolerance
    'maxiter': 100,             # Maximum iterations
    'disp': True,               # Display progress
    'eps': 1e-8,                # Step size for finite differences
}
```

#### COBYLA/COBYQA Options

```python
options = {
    'rhobeg': 1.0,              # Initial trust region radius
    'rhoend': 1e-6,             # Final trust region radius
    'maxiter': 100,             # Maximum iterations
    'disp': True,               # Display progress
}
```

## Implementation Details

### Objective Function

The objective function combines:
1. **Equality constraints**: Penalty for deviations from desired values
2. **Inequality constraints**: Penalty for constraint violations
3. **Minimization objectives**: Direct minimization of specified outputs

### Constraint Handling

#### Trust-Region Constrained Algorithm
- Equality constraints: `NonlinearConstraint(constraint_func, 0, 0)`
- Inequality constraints: `NonlinearConstraint(constraint_func, -inf, 0)` or `NonlinearConstraint(constraint_func, 0, inf)`

#### SLSQP
- Equality constraints: `{'type': 'eq', 'fun': constraint_func}`
- Inequality constraints: `{'type': 'ineq', 'fun': constraint_func}`

#### COBYLA/COBYQA
- Constraints are handled through penalty functions in the objective

### Gradient Computation

For SciPy methods, gradients are computed using finite differences:
```python
def gradient(x):
    h = 1e-6
    grad = np.zeros_like(x)
    
    for i in range(len(x)):
        x_plus = x.copy()
        x_plus[i] += h
        x_minus = x.copy()
        x_minus[i] -= h
        
        grad[i] = (objective(x_plus) - objective(x_minus)) / (2 * h)
    
    return grad
```

## Advantages and Disadvantages

### Trust-Region Constrained Algorithm

**Advantages:**
- Robust convergence properties
- Handles constraints naturally
- Good for problems with multiple constraints
- Supports bounds on decision variables

**Disadvantages:**
- Slower than PyTorch methods for large problems
- Requires gradient computation (finite differences)
- More function evaluations per iteration

### PyTorch Methods (1st_order)

**Advantages:**
- Fast for large problems
- Automatic differentiation
- Efficient gradient computation
- Good for differentiable models

**Disadvantages:**
- Requires differentiable models
- Constraint handling through penalties
- May not converge for highly constrained problems

## Example Use Cases

### 1. Building Energy Optimization

```python
# Optimize HVAC setpoints while maintaining comfort
decisionVariables = [
    (hvac, "setpoint_temperature", 18.0, 26.0),
    (hvac, "setpoint_humidity", 30.0, 70.0)
]

minimize = [
    (energy_meter, "total_energy")
]

equalityConstraints = [
    (comfort_sensor, "thermal_comfort", 0.8)  # Maintain 80% comfort
]

inequalityConstraints = [
    (room_sensor, "temperature", "lower", 18.0),
    (room_sensor, "temperature", "upper", 26.0),
    (room_sensor, "humidity", "lower", 30.0),
    (room_sensor, "humidity", "upper", 70.0)
]

# Run optimization
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
```

### 2. Demand Response Optimization

```python
# Optimize load shifting while maintaining grid stability
decisionVariables = [
    (battery, "charge_rate", 0.0, 100.0),
    (battery, "discharge_rate", 0.0, 100.0)
]

minimize = [
    (grid_meter, "peak_demand")
]

equalityConstraints = [
    (battery, "state_of_charge", 50.0)  # Maintain 50% SOC
]

inequalityConstraints = [
    (battery, "state_of_charge", "lower", 20.0),
    (battery, "state_of_charge", "upper", 80.0),
    (grid_meter, "power_draw", "upper", 1000.0)  # Grid limit
]

# Run optimization
result = optimizer.optimize(
    decisionVariables=decisionVariables,
    minimize=minimize,
    equalityConstraints=equalityConstraints,
    inequalityConstraints=inequalityConstraints,
    startTime=start_time,
    endTime=end_time,
    stepSize=step_size,
    method="SLSQP",
    options={
        'ftol': 1e-6,
        'maxiter': 100,
        'disp': True,
    }
)
```

### 3. Method Comparison Example

```python
# Compare different optimization methods
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
```

## Troubleshooting

### Common Issues

1. **Optimization not converging**
   - Try different initial values
   - Adjust tolerance parameters
   - Check constraint feasibility

2. **Constraints violated**
   - Verify constraint definitions
   - Check bounds on decision variables
   - Consider using penalty-based methods

3. **Slow convergence**
   - Use PyTorch methods for large problems
   - Adjust step sizes for finite differences
   - Consider using derivative-free methods

### Performance Tips

1. **For large problems**: Use 1st_order (PyTorch) methods
2. **For constrained problems**: Use trust-constr or SLSQP
3. **For noisy problems**: Use COBYLA or COBYQA
4. **For derivative-free optimization**: Use COBYLA or COBYQA

## Future Enhancements

Potential improvements to consider:

1. **Automatic differentiation for SciPy methods**: Use PyTorch gradients in SciPy optimization
2. **Parallel function evaluation**: Speed up finite difference gradient computation
3. **Adaptive constraint handling**: Automatically choose constraint handling method
4. **Multi-objective optimization**: Support for Pareto-optimal solutions
5. **Robust optimization**: Handle uncertainty in model parameters 