# import pygad
# Standard library imports
import datetime
from typing import Any, Dict, List, Tuple, Union

# Third party imports
import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import Bounds, least_squares, minimize

# Local application imports
import twin4build.core as core
import twin4build.systems as systems


def _min_max_normalize(x, min_val=None, max_val=None):
    if min_val is None:
        min_val = torch.min(x)
    if max_val is None:
        max_val = torch.max(x)
    return (x - min_val) / (max_val - min_val)


class Optimizer:
    r"""
    A class for optimizing building operation in the twin4build framework.

    This class optimizes model inputs (variables) (e.g., setpoints) by minimizing a loss function, using gradient-based or other optimization algorithms.
    The optimizer implements soft constraints on model outputs (embedded in the loss function) and hard constraints on variables.

    Mathematical Formulation
    =======================

    The general optimization problem is formulated as:

        .. math::

            \hat{\boldsymbol{U}} = \underset{\boldsymbol{U} \in \mathcal{U}}{\operatorname{argmin}} \; \mathcal{L}(\boldsymbol{U})

    where:
        - :math:`\hat{\boldsymbol{U}}` is the optimal control input matrix
        - :math:`\boldsymbol{U}` is the control input matrix
        - :math:`\mathcal{U} \subseteq \mathbb{R}^{n_u \times n_t}` is the set of feasible control inputs
        - :math:`\mathcal{L}(\boldsymbol{U})` is the loss function

    Dimensions
    ----------

    - :math:`n_t`: Number of time steps in the simulation period
    - :math:`n_u`: Number of control inputs (actuators)
    - :math:`n_d`: Number of disturbance inputs (weather, occupancy, etc.)
    - :math:`n_y`: Number of system outputs (sensors, performance metrics)

    Model Structure
    ---------------

    The building model :math:`\mathcal{M}` is represented as a directed graph where nodes are dynamic components
    and edges represent input/output connections as shown in a simple example below.

    .. figure:: /_static/optimizer_graph_.png
       :alt: System overview showing components and their relationships
       :align: center
       :width: 80%

    The model takes control inputs :math:`\boldsymbol{U} \in \mathbb{R}^{n_u \times n_t}`
    (the optimization variables) along with external inputs or disturbances :math:`\boldsymbol{D} \in \mathbb{R}^{n_d \times n_t}`, and produces system outputs for optimization
    :math:`\boldsymbol{\hat{Y}} \in \mathbb{R}^{n_y \times n_t}` with timesteps :math:`\boldsymbol{t} \in \mathbb{R}^{n_t}`:

    .. math::

            \boldsymbol{\hat{Y}} = \mathcal{M}(\boldsymbol{X}, \boldsymbol{t})

    where:

        .. math::

            \boldsymbol{X} = [\boldsymbol{U}, \boldsymbol{D}]

    and :math:`\mathcal{M}` represents the complete simulation model. See :class:`~twin4build.simulator.simulator.Simulator`
    for detailed explanation of the simulation process.

    Loss Function
    -------------

    The loss function :math:`\mathcal{L}(\boldsymbol{U})` is composed of the following terms:

    **Equality Constraints**

        .. math::

            \mathcal{L}_{eq} = \frac{1}{n_t} \sum_{t=1}^{n_t} \sum_{(j, \boldsymbol{y}) \in \mathcal{C}_{eq}} |\boldsymbol{\hat{Y}}_{j,t} - \boldsymbol{y}_{t}|

        where :math:`\mathcal{C}_{eq}` is the set of equality constraints, each element is (output index :math:`j`, desired value :math:`\boldsymbol{y}_{t}`).

    **Inequality Constraints**

        Upper constraints:

        .. math::

            \mathcal{L}_{ineq}^{upper} = \frac{1}{n_t} \sum_{t=1}^{n_t} \sum_{(j, \boldsymbol{y}) \in \mathcal{C}_{ineq}^{upper}} k \cdot \text{relu}\left(\boldsymbol{\hat{Y}}_{j,t} - \boldsymbol{y}_{t}\right)

        Lower constraints:

        .. math::

            \mathcal{L}_{ineq}^{lower} = \frac{1}{n_t} \sum_{t=1}^{n_t} \sum_{(j, \boldsymbol{y}) \in \mathcal{C}_{ineq}^{lower}} k \cdot \text{relu}\left(\boldsymbol{y}_{t} - \boldsymbol{\hat{Y}}_{j,t}\right)

        where :math:`\mathcal{C}_{ineq}^{upper}` and :math:`\mathcal{C}_{ineq}^{lower}` are the sets of upper and lower inequality constraints, and :math:`k` is a penalty factor.

        Combined inequality constraint loss:

        .. math::

            \mathcal{L}_{ineq} = \mathcal{L}_{ineq}^{upper} + \mathcal{L}_{ineq}^{lower}

    **Objective Terms**

        .. math::

            \mathcal{L}_{obj} = \frac{1}{n_t} \sum_{t=1}^{n_t} \sum_{(j, w) \in \mathcal{O}_{obj}} w \cdot \boldsymbol{\hat{Y}}_{j,t}

        where :math:`\mathcal{O}_{obj}` is the set of outputs to minimize or maximize, and :math:`w` is a weight (+1 for minimization, -1 for maximization).

    **Total Loss**

        .. math::

            \mathcal{L}(\boldsymbol{U}) = \mathcal{L}_{eq} + \mathcal{L}_{ineq} + \mathcal{L}_{obj}

    See method docstrings for details on the specific loss terms and optimization algorithms.

    Attributes
    ----------
    simulator : core.Simulator
        The simulator instance for running simulations.
    variables : List[Tuple[Any, str, float, float]]
        List of decision variables to optimize. Each tuple contains:
        (component, output_name, lower_bound, upper_bound).
    objectives : List[Tuple[Any, str, str]]
        List of objectives to minimize or maximize. Each tuple contains:
        (component, output_name, objective_type) where objective_type is "min" or "max".
    equalityConstraints : List[Tuple[Any, str, Any]]
        List of equality constraints. Each tuple contains:
        (component, output_name, desired_value).
    inequalityConstraints : List[Tuple[Any, str, str, Any]]
        List of inequality constraints. Each tuple contains:
        (component, output_name, constraint_type, desired_value) where constraint_type is "upper" or "lower".
    startTime : Union[datetime.datetime, List[datetime.datetime]]
        Start time(s) for optimization period(s).
    endTime : Union[datetime.datetime, List[datetime.datetime]]
        End time(s) for optimization period(s).
    stepSize : Union[float, List[float]]
        Step size(s) for simulation in seconds.

    Examples
    --------
    Basic optimization with PyTorch method:

    >>> import twin4build as tb
    >>> import datetime
    >>> import pytz
    >>>
    >>> # Create model and simulator
    >>> model = tb.SimulationModel(id="my_model")
    >>> simulator = tb.Simulator(model)
    >>> optimizer = tb.Optimizer(simulator)
    >>>
    >>> # Define decision variables (actuators to optimize)
    >>> variables = [
    ...     (heater_component, "setpointValue", 18.0, 25.0),  # Temperature setpoint bounds
    ...     (ventilation_component, "flowRate", 0.1, 1.0)    # Ventilation flow rate bounds
    ... ]
    >>>
    >>> # Define objectives (what to optimize)
    >>> objectives = [
    ...     (energy_meter, "powerConsumption", "min"),  # Minimize energy consumption
    ...     (comfort_sensor, "comfortIndex", "max")     # Maximize comfort
    ... ]
    >>>
    >>> # Set time period
    >>> start = datetime.datetime(2024, 1, 1, tzinfo=pytz.UTC)
    >>> end = datetime.datetime(2024, 1, 2, tzinfo=pytz.UTC)
    >>> step = 3600
    >>>
    >>> # Run optimization with PyTorch (default SGD)
    >>> optimizer.optimize(
    ...     variables=variables,
    ...     objectives=objectives,
    ...     startTime=start,
    ...     endTime=end,
    ...     stepSize=step,
    ...     method="torch",
    ...     options={"lr": 0.1, "iterations": 100}
    ... )

    Advanced PyTorch optimization with scheduler:

    >>> # Use Adam optimizer with cosine annealing scheduler
    >>> optimizer.optimize(
    ...     variables=variables,
    ...     objectives=objectives,
    ...     startTime=start,
    ...     endTime=end,
    ...     stepSize=step,
    ...     method=("torch", "Adam", "ad"),
    ...     options={
    ...         "lr": 0.01,
    ...         "iterations": 200,
    ...         "scheduler_type": "cosine",
    ...         "scheduler_params": {"T_max": 200, "eta_min": 1e-6}
    ...     }
    ... )

    SciPy optimization with constraints:

    >>> # Define equality constraints (maintain temperature at specific times)
    >>> equality_constraints = [
    ...     (room_temperature, "temperature", 21.0)  # Maintain 21°C
    ... ]
    >>>
    >>> # Define inequality constraints (comfort bounds)
    >>> inequality_constraints = [
    ...     (room_temperature, "temperature", "lower", 20.0),  # Not below 20°C
    ...     (room_temperature, "temperature", "upper", 24.0),  # Not above 24°C
    ...     (co2_sensor, "concentration", "upper", 1000.0)     # CO2 limit
    ... ]
    >>>
    >>> # Run SciPy optimization with SLSQP (preferred for constrained problems)
    >>> optimizer.optimize(
    ...     variables=variables,
    ...     objectives=objectives,
    ...     equalityConstraints=equality_constraints,
    ...     inequalityConstraints=inequality_constraints,
    ...     startTime=start,
    ...     endTime=end,
    ...     stepSize=step,
    ...     method=("scipy", "SLSQP", "ad"),
    ...     options={"verbose": 2, "maxiter": 1000}
    ... )

    Alternative SciPy methods:

    >>> # Use L-BFGS-B for unconstrained optimization
    >>> optimizer.optimize(
    ...     variables=variables,
    ...     objectives=objectives,
    ...     startTime=start,
    ...     endTime=end,
    ...     stepSize=step,
    ...     method=("scipy", "L-BFGS-B", "ad"),
    ...     options={"gtol": 1e-8, "maxiter": 500}
    ... )

    >>> # Use trust-region method for difficult constraints
    >>> optimizer.optimize(
    ...     variables=variables,
    ...     objectives=objectives,
    ...     equalityConstraints=equality_constraints,
    ...     inequalityConstraints=inequality_constraints,
    ...     startTime=start,
    ...     endTime=end,
    ...     stepSize=step,
    ...     method=("scipy", "trust-constr", "ad"),
    ...     options={"verbose": 1, "barrier_tol": 1e-8}
    ... )

    Schedule-based constraints:

    >>> # Use schedule systems for time-varying constraints
    >>> import twin4build.systems as systems
    >>>
    >>> # Create temperature schedule
    >>> temp_schedule = systems.ScheduleSystem(
    ...     id="temp_schedule",
    ...     schedule_filename="temperature_profile.csv"
    ... )
    >>>
    >>> # Use schedule as constraint
    >>> equality_constraints = [
    ...     (room_temperature, "temperature", temp_schedule)
    ... ]
    >>>
    >>> optimizer.optimize(
    ...     variables=variables,
    ...     objectives=objectives,
    ...     equalityConstraints=equality_constraints,
    ...     startTime=start,
    ...     endTime=end,
    ...     stepSize=step,
    ...     method=("scipy", "SLSQP", "ad")
    ... )

    Multi-objective optimization:

    >>> # Optimize multiple conflicting objectives
    >>> objectives = [
    ...     (energy_meter, "powerConsumption", "min"),     # Minimize energy
    ...     (comfort_sensor, "thermalComfort", "max"),     # Maximize comfort
    ...     (air_quality_sensor, "iaqIndex", "max"),       # Maximize air quality
    ... ]
    >>>
    >>> # Use multiple decision variables
    >>> variables = [
    ...     (heater_component, "setpointValue", 18.0, 25.0),
    ...     (cooler_component, "setpointValue", 22.0, 28.0),
    ...     (ventilation_component, "flowRate", 0.1, 2.0),
    ...     (window_actuator, "openingDegree", 0.0, 1.0)
    ... ]
    >>>
    >>> optimizer.optimize(
    ...     variables=variables,
    ...     objectives=objectives,
    ...     startTime=start,
    ...     endTime=end,
    ...     stepSize=step,
    ...     method=("scipy", "SLSQP", "ad"),
    ...     options={"ftol": 1e-9, "maxiter": 2000}
    ... )

    Legacy string format (still supported):

    >>> # Simple usage with default settings
    >>> optimizer.optimize(
    ...     variables=variables,
    ...     objectives=objectives,
    ...     startTime=start,
    ...     endTime=end,
    ...     stepSize=step,
    ...     method="scipy"  # Defaults to ("scipy", "SLSQP", "ad")
    ... )

    >>> # PyTorch method with defaults
    >>> optimizer.optimize(
    ...     variables=variables,
    ...     objectives=objectives,
    ...     startTime=start,
    ...     endTime=end,
    ...     stepSize=step,
    ...     method="torch"  # Defaults to ("torch", "SGD", "ad")
    ... )
    """

    def __init__(self, simulator: core.Simulator):
        self.simulator = simulator

    def _closure(self):
        self.optimizer.zero_grad()

        # Apply bounds to decision variables
        with torch.no_grad():
            for component, output_name, *bounds in self.variables:
                if len(bounds) > 0:
                    lower_bound = bounds[0] if len(bounds) > 0 else float("-inf")
                    upper_bound = bounds[1] if len(bounds) > 1 else float("inf")
                    if component.output[output_name].do_normalization:
                        lower_bound_ = component.output[output_name].normalize(
                            lower_bound
                        )
                        upper_bound_ = component.output[output_name].normalize(
                            upper_bound
                        )
                        # print("==========================")
                        # print(f"CLAMPED BEFORE: {component.id}.{output_name} to {component.output[output_name].denormalize(component.output[output_name].normalized_history)}")
                        component.output[output_name].normalized_history.clamp_(
                            min=lower_bound_, max=upper_bound_
                        )

                        # print("==========================")
                        # print(f"CLAMPED AFTER: {component.id}.{output_name} to {component.output[output_name].denormalize(component.output[output_name].normalized_history)}")
                    else:
                        component.output[output_name].history.clamp_(
                            min=lower_bound, max=upper_bound
                        )

        # Run simulation
        self.simulator.simulate(
            startTime=self.startTime,
            endTime=self.endTime,
            stepSize=self.stepSize,
            show_progress_bar=False,
        )

        self.loss = 0
        k = 100

        # Handle equality constraints
        if self.equalityConstraints is not None:
            eq_term = 0
            for constraint in self.equalityConstraints:
                component, output_name, desired_value = constraint
                y = component.output[output_name].history
                desired_tensor = self.equality_constraint_values[component, output_name]
                y = component.output[output_name].normalize(y)
                desired_tensor = component.output[output_name].normalize(desired_tensor)

                eq_term += torch.mean(torch.abs(y - desired_tensor))
            self.loss += eq_term

        # Handle inequality constraints
        if self.inequalityConstraints is not None:
            ineq_upper_term = 0
            ineq_lower_term = 0
            for constraint in self.inequalityConstraints:
                component, output_name, constraint_type, desired_value = constraint
                y = component.output[output_name].history
                desired_tensor = self.inequality_constraint_values[
                    (component, output_name, constraint_type)
                ]
                y_norm = component.output[output_name].normalize(y)
                desired_tensor_norm = component.output[output_name].normalize(
                    desired_tensor
                )

                if constraint_type == "upper":
                    # Penalize when y > desired_value
                    constraint_violations = torch.relu(y_norm - desired_tensor_norm)
                    constraint_term = torch.mean(k * constraint_violations)
                    ineq_upper_term += constraint_term

                elif constraint_type == "lower":
                    # Penalize when y < desired_value
                    constraint_violations = torch.relu(desired_tensor_norm - y_norm)
                    constraint_term = torch.mean(k * constraint_violations)
                    ineq_lower_term += constraint_term

            self.loss += ineq_upper_term + ineq_lower_term

        # Handle minimization objectives
        if self.objectives is not None:
            min_term = 0
            for minimize_obj in self.objectives:
                component, output_name = minimize_obj
                y = component.output[output_name].history
                y_norm = component.output[output_name].normalize(y)
                # print(f"NORMALIZED MINIMIZE OBJECTIVE BETWEEN: {component.output[output_name]._min_history} and {component.output[output_name]._max_history}")

                min_term += torch.mean(y_norm)
            self.loss += min_term  # Minimize the mean value

        # Compute gradients
        self.loss.backward()
        return self.loss

    def optimize(
        self,
        variables: List[Tuple[Any, str, float, float]] = None,
        objectives: List[Tuple[Any, str, str]] = None,
        equalityConstraints: List[Tuple[Any, str, Any]] = None,
        inequalityConstraints: List[Tuple[Any, str, str, Any]] = None,
        startTime: Union[datetime.datetime, List[datetime.datetime]] = None,
        endTime: Union[datetime.datetime, List[datetime.datetime]] = None,
        stepSize: Union[float, List[float]] = None,
        method: Union[str, Tuple[str, str, str]] = "scipy",
        options: Dict = None,
    ):
        """
        Optimize the model using various optimization methods.

        Args:
            variables: List of tuples (component, output_name, lower_bound, upper_bound)
            objectives: List of tuples (component, output_name, objective_type)
                where objective_type is "min" or "max"
            equalityConstraints: List of tuples (component, output_name, desired_value)
            inequalityConstraints: List of tuples (component, output_name, constraint_type, desired_value)
                where constraint_type is "upper" or "lower"
            startTime: Start time for simulation
            endTime: End time for simulation
            stepSize: Step size for simulation
            method: Optimization method specification. Can be specified in two formats:

                1. String format (legacy):
                   - "torch": Uses PyTorch-based gradient optimization
                   - "scipy": Uses SciPy's SLSQP solver with automatic differentiation

                2. Tuple format (recommended):
                   - (library, optimizer, mode) where:
                     - library: "torch" or "scipy"
                     - optimizer: The specific optimization algorithm
                     - mode: "ad" (automatic differentiation) or "fd" (finite difference)

                Supported optimizers by library:

                PyTorch-based methods (library="torch"):
                   - "SGD": Stochastic Gradient Descent (default)
                   - "Adam": Adam optimizer
                   - "LBFGS": Limited-memory BFGS
                   - Mode: Always "ad" (automatic differentiation)

                SciPy-based methods (library="scipy"):
                   - "SLSQP": Sequential Least Squares Programming (preferred for most problems)
                   - "L-BFGS-B": Limited-memory BFGS with bounds
                   - "TNC": Truncated Newton algorithm with bounds
                   - "trust-constr": Trust-region constrained optimization
                   - "trf": Trust Region Reflective (for least-squares problems)
                   - "dogbox": Dogleg algorithm (for least-squares problems)
                   - Mode: "ad" (automatic differentiation) or "fd" (finite difference)

                Method selection guidelines:
                   - PyTorch methods: Good for simple optimization problems, easy to configure
                   - SciPy SLSQP with AD: Preferred for most constrained optimization problems
                   - SciPy with FD: Use for non-PyTorch models or when AD is not available

                Examples:
                   - ("scipy", "SLSQP", "ad"): Preferred for most constrained optimization problems
                   - ("torch", "Adam", "ad"): Good for simple unconstrained problems
                   - ("scipy", "trf", "fd"): For non-PyTorch models with least-squares formulation
                   - "scipy": Legacy format, defaults to ("scipy", "SLSQP", "ad")

            options: Additional options for the chosen method:
                For PyTorch methods (library="torch"):
                    - "lr": Learning rate for optimizer (default: 1.0)
                    - "iterations": Number of optimization iterations (default: 100)
                    - "optimizer_type": Type of PyTorch optimizer ("SGD", "Adam", "LBFGS")
                    - "scheduler_type": Type of learning rate scheduler ("step", "exponential", "cosine", "reduce_on_plateau", None)
                    - "scheduler_params": Parameters for learning rate scheduler
                        - For "step": step_size (default: 30), gamma (default: 0.1)
                        - For "exponential": gamma (default: 0.95)
                        - For "cosine": T_max (default: 100), eta_min (default: 0)
                        - For "reduce_on_plateau": mode (default: "min"), factor (default: 0.9), patience (default: 10), threshold (default: 1e-4)

                For SciPy methods (library="scipy"):
                    - "verbose": Verbosity level (0-3)
                    - "maxiter": Maximum iterations
                    - "gtol": Gradient tolerance
                    - "xtol": Parameter tolerance
                    - "barrier_tol": Barrier tolerance
                    - "initial_tr_radius": Initial trust region radius
                    - "initial_constr_penalty": Initial constraint penalty
                    - Additional method-specific options as supported by SciPy optimizers
        """
        self.variables = variables or []
        self.objectives = objectives or []
        self.equalityConstraints = equalityConstraints or []
        self.inequalityConstraints = inequalityConstraints or []
        self.startTime = startTime
        self.endTime = endTime
        self.stepSize = stepSize

        self.max_values = {}

        # Validate input arguments
        # Check required simulation parameters
        assert startTime is not None, "startTime must be provided"
        assert endTime is not None, "endTime must be provided"
        assert stepSize is not None, "stepSize must be provided"

        # Check that we have something to optimize
        assert (
            len(self.variables) > 0
        ), "No decision variables specified for optimization"

        # Check that we have at least one objective (minimize or constraints)
        has_objective = (
            len(self.objectives) > 0
            or len(self.equalityConstraints) > 0
            or len(self.inequalityConstraints) > 0
        )
        assert (
            has_objective
        ), "No optimization objectives specified (minimize, equalityConstraints, or inequalityConstraints)"

        # Validate method
        # Define allowed optimization methods
        allowed_methods = [
            ("torch", "SGD", "ad"),
            ("torch", "Adam", "ad"),
            ("torch", "LBFGS", "ad"),
            ("scipy", "SLSQP", "ad"),
            ("scipy", "L-BFGS-B", "ad"),
            ("scipy", "TNC", "ad"),
            ("scipy", "trust-constr", "ad"),
            ("scipy", "trf", "ad"),
            ("scipy", "dogbox", "ad"),
            ("scipy", "trf", "fd"),
            ("scipy", "dogbox", "fd"),
        ]
        default_methods = [("scipy", "SLSQP", "ad")]
        default_mode = (
            "ad"  # Always choose automatic differentiation mode when ambiguous
        )

        # Process method specification
        if isinstance(method, str):
            valid_methods = list(
                set([l[0] for l in allowed_methods] + [l[1] for l in allowed_methods])
            )
            assert (
                method in valid_methods
            ), f"If a string is provided, the \"method\" argument must be one of the following: {', '.join(valid_methods)} - \"{method}\" was provided."

            # Try to match with default methods first
            matched = False
            for t in default_methods:
                if t[0] == method:
                    method = t
                    matched = True
                    break

            # If no match found, look for candidates
            if not matched:
                candidates = []
                for m in allowed_methods:
                    if m[1] == method:
                        candidates.append(m)

                if len(candidates) == 1:
                    method = candidates[0]
                elif len(candidates) > 1:
                    # Choose the one with default mode
                    for c in candidates:
                        if c[2] == default_mode:
                            method = c
                            break

        elif isinstance(method, tuple):
            assert (
                len(method) == 3
            ), f'If a tuple is provided, it must contain three elements, corresponding to the library, method, and mode (e.g. ("scipy", "SLSQP", "ad")) - "{method}" was provided.'
            assert method[0] in [
                l[0] for l in allowed_methods
            ], f"If a tuple is provided, the first element must be one of the following: {', '.join(list(set([l[0] for l in allowed_methods])))} - \"{method}\" was provided."
            assert method[1] in [
                l[1] for l in allowed_methods
            ], f"If a tuple is provided, the second element must be one of the following: {', '.join(list(set([l[1] for l in allowed_methods])))} - \"{method}\" was provided."
            assert method[2] in [
                l[2] for l in allowed_methods
            ], f"If a tuple is provided, the third element must be one of the following: {', '.join(list(set([l[2] for l in allowed_methods])))} - \"{method}\" was provided."

            # Validate the method tuple
            method_ = None
            for t in allowed_methods:
                if t[0] == method[0] and t[1] == method[1] and t[2] == method[2]:
                    method_ = t
                    break
            assert (
                method_ is not None
            ), f"The method {method} is not valid. Only the following methods are supported: {', '.join([str(t) for t in allowed_methods])}"
            method = method_
        else:
            raise ValueError(
                f'The "method" argument must be a string or a tuple - "{method}" was provided.'
            )

        # Validate format of decision variables
        for i, decision_var in enumerate(self.variables):
            assert (
                len(decision_var) >= 2
            ), f"Decision variable at index {i} must have at least component and output_name"
            component, output_name, *bounds = decision_var
            assert hasattr(
                component, "output"
            ), f"Component {component} at index {i} does not have 'output' attribute"
            assert (
                output_name in component.output
            ), f"Output '{output_name}' not found in component {component.id}"
            if len(bounds) >= 2:
                lower, upper = bounds[0], bounds[1]
                assert (
                    upper > lower
                ), f"Upper bound ({upper}) must be greater than lower bound ({lower}) for {component.id}.{output_name}"

        # Validate format of minimize objectives
        for i, min_obj in enumerate(self.objectives):
            assert (
                len(min_obj) == 3
            ), f"Minimize objective at index {i} must have component, output_name, and objective_type (min or max)"
            component, output_name, objective_type = min_obj
            assert hasattr(
                component, "output"
            ), f"Component {component} at index {i} does not have 'output' attribute"
            assert (
                output_name in component.output
            ), f"Output '{output_name}' not found in component {component.id}"

        # Validate format of equality constraints
        for i, eq_constraint in enumerate(self.equalityConstraints):
            assert (
                len(eq_constraint) == 3
            ), f"Equality constraint at index {i} must have component, output_name, and desired_value"
            component, output_name, desired_value = eq_constraint
            assert hasattr(
                component, "output"
            ), f"Component {component} at index {i} does not have 'output' attribute"
            assert (
                output_name in component.output
            ), f"Output '{output_name}' not found in component {component.id}"

        # Validate format of inequality constraints
        for i, ineq_constraint in enumerate(self.inequalityConstraints):
            assert (
                len(ineq_constraint) == 4
            ), f"Inequality constraint at index {i} must have component, output_name, constraint_type, and desired_value"
            component, output_name, constraint_type, desired_value = ineq_constraint
            assert hasattr(
                component, "output"
            ), f"Component {component} at index {i} does not have 'output' attribute"
            assert (
                output_name in component.output
            ), f"Output '{output_name}' not found in component {component.id}"
            assert constraint_type in [
                "upper",
                "lower",
            ], f"Constraint type must be 'upper' or 'lower', got '{constraint_type}'"

        # Check for conflicting constraints: can't minimize and have equality constraint on same output
        if self.objectives and self.equalityConstraints:
            minimize_pairs = {
                (component, output_name) for component, output_name in self.objectives
            }
            equality_pairs = {
                (component, output_name)
                for component, output_name, _ in self.equalityConstraints
            }

            conflicting_pairs = minimize_pairs.intersection(equality_pairs)
            if conflicting_pairs:
                conflict_info = [f"({c.id}, {o})" for c, o in conflicting_pairs]
                raise ValueError(
                    f"Cannot simultaneously minimize and apply equality constraints to the same outputs: {', '.join(conflict_info)}. "
                    f"These objectives conflict with each other."
                )

        # Check for decision variables that are also in equality constraints
        if self.variables and self.equalityConstraints:
            decision_pairs = {
                (component, output_name)
                for component, output_name, *_ in self.variables
            }
            equality_pairs = {
                (component, output_name)
                for component, output_name, _ in self.equalityConstraints
            }

            conflicting_pairs = decision_pairs.intersection(equality_pairs)
            if conflicting_pairs:
                conflict_info = [f"({c.id}, {o})" for c, o in conflicting_pairs]
                raise ValueError(
                    f"Cannot optimize and apply equality constraints to the same outputs: {', '.join(conflict_info)}. "
                    f"These objectives conflict with each other."
                )

        # allowed_methods = [("scipy", "trf", "fd"),
        #                     ("scipy", "dogbox", "fd"),
        #                     ("scipy", "trf", "ad"),
        #                     ("scipy", "dogbox", "ad"),
        #                     ("scipy", "L-BFGS-B", "ad"),
        #                     ("scipy", "TNC", "ad"),
        #                     ("scipy", "SLSQP", "ad"),
        #                     ("scipy", "trust-constr", "ad"),
        #                     # ("torch", "Adadelta", "ad"), # Currently, we do not support torch optimizers
        #                     # ("torch", "Adafactor", "ad"),
        #                     # ("torch", "Adagrad", "ad"),
        #                     # ("torch", "Adam", "ad"),
        #                     # ("torch", "AdamW", "ad"),
        #                     # ("torch", "SparseAdam", "ad"),
        #                     # ("torch", "Adamax", "ad"),
        #                     # ("torch", "ASGD", "ad"),
        #                     # ("torch", "LBFGS", "ad"),
        #                     # ("torch", "NAdam", "ad"),
        #                     # ("torch", "RAdam", "ad"),
        #                     # ("torch", "RMSprop", "ad"),
        #                     # ("torch", "Rprop", "ad"),
        #                     # ("torch", "SGD", "ad"),
        #                 ]
        # default_none_method = ("scipy", "SLSQP", "ad")
        # default_methods = [("scipy", "SLSQP", "ad")]#, ("torch", "SGD", "ad")]
        # default_mode = "ad" # Always choose automatic differentiation mode when ambiguous

        # Call the appropriate optimization method
        if method[0] == "torch":
            if options is None:
                options = {}
            # Extract optimizer type from method tuple
            optimizer_type = method[1]
            options["optimizer_type"] = optimizer_type
            return self._torch_solver(**options)
        elif method[0] == "scipy":
            if options is None:
                options = {}
            return self._scipy_solver(method=method, **options)

    def _torch_solver(
        self,
        lr: float = 1.0,
        iterations: int = 100,
        optimizer_type: str = "SGD",
        scheduler_type: str = "step",
        scheduler_params: Dict = None,
    ):
        """
        Perform optimization using PyTorch-based gradient optimization.

        This method uses PyTorch's automatic differentiation to compute gradients and
        applies gradient-based optimization algorithms to minimize the objective function.
        It supports various optimizers and learning rate schedulers for fine-tuning
        the optimization process.

        Args:
            lr: Learning rate for optimizer. Controls the step size in gradient descent.
                Higher values may converge faster but risk overshooting, while lower
                values are more stable but may converge slowly.
            iterations: Number of optimization iterations. More iterations generally
                lead to better convergence but take longer to compute.
            optimizer_type: Type of PyTorch optimizer:
                - "SGD": Stochastic Gradient Descent - simple, robust, good for most problems
                - "Adam": Adaptive learning rate optimizer - often faster convergence
                - "LBFGS": Limited-memory BFGS - good for smooth, well-behaved functions
            scheduler_type: Type of learning rate scheduler to adjust learning rate during optimization:
                - "step": Decreases learning rate by gamma every step_size iterations
                - "exponential": Decreases learning rate exponentially
                - "cosine": Uses cosine annealing schedule
                - "reduce_on_plateau": Reduces learning rate when loss stops improving
                - None: No scheduler, constant learning rate
            scheduler_params: Dictionary of parameters for the chosen scheduler:
                - For "step": {"step_size": int, "gamma": float}
                - For "exponential": {"gamma": float}
                - For "cosine": {"T_max": int, "eta_min": float}
                - For "reduce_on_plateau": {"mode": str, "factor": float, "patience": int, "threshold": float}

        Note:
            This method automatically handles gradient computation and parameter updates.
            It disables gradients for model parameters and only optimizes the decision variables.
            The optimization process is logged with current learning rate and loss values.
        """
        # Validate optimization parameters
        assert lr > 0, f"Learning rate must be positive, got {lr}"
        assert (
            iterations > 0
        ), f"Number of iterations must be positive, got {iterations}"

        # Validate scheduler type
        valid_scheduler_types = [
            "step",
            "exponential",
            "cosine",
            "reduce_on_plateau",
            None,
        ]
        assert (
            scheduler_type in valid_scheduler_types
        ), f"Invalid scheduler_type: {scheduler_type}. Must be one of {valid_scheduler_types}"

        # Disable gradients for all parameters since we're optimizing inputs.
        # It is VERY important to do this before initializing the model.
        # Otherwise, the model parameters and state space matrices will have requires_grad=True
        # and the backpropagate() call will fail.
        for component in self.simulator.model.components.values():
            if isinstance(component, nn.Module):
                for parameter in component.parameters():
                    parameter.requires_grad_(False)

        # Set before initializing the model
        for component, output_name, *bounds in self.variables:
            component.output[output_name].do_normalization = True

        self.simulator.get_simulation_timesteps(
            self.startTime, self.endTime, self.stepSize
        )
        self.simulator.model.initialize(
            startTime=self.startTime,
            endTime=self.endTime,
            stepSize=self.stepSize,
            simulator=self.simulator,
        )

        # Enable gradients only for the inputs we want to optimize
        opt_list = []
        for component, output_name, *bounds in self.variables:
            component.output[output_name].set_requires_grad(True)
            if component.output[output_name].do_normalization:
                opt_list.append(component.output[output_name].normalized_history)
            else:
                opt_list.append(component.output[output_name].history)

        if optimizer_type == "SGD":
            # Initialize optimizer
            self.optimizer = torch.optim.SGD(opt_list, lr=lr)
        elif optimizer_type == "Adam":
            self.optimizer = torch.optim.Adam(opt_list, lr=lr)
        elif optimizer_type == "LBFGS":
            self.optimizer = torch.optim.LBFGS(
                opt_list, lr=lr, line_search_fn=None, history_size=100
            )
        else:
            raise ValueError(
                f"Invalid optimizer type: {optimizer_type}. Must be one of {['SGD', 'Adam', 'LBFGS']}"
            )

        # Initialize scheduler
        if scheduler_params is None:
            scheduler_params = {}

        if scheduler_type == "step":
            # StepLR decreases learning rate by gamma every step_size epochs
            step_size = scheduler_params.get("step_size", 30)
            gamma = scheduler_params.get("gamma", 0.1)
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=step_size, gamma=gamma
            )
        elif scheduler_type == "exponential":
            # ExponentialLR decreases learning rate by gamma every epoch
            gamma = scheduler_params.get("gamma", 0.95)
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer, gamma=gamma
            )
        elif scheduler_type == "cosine":
            # CosineAnnealingLR uses a cosine schedule to decrease learning rate
            T_max = scheduler_params.get("T_max", 100)
            eta_min = scheduler_params.get("eta_min", 0)
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=T_max, eta_min=eta_min
            )
        elif scheduler_type == "reduce_on_plateau":
            # ReduceLROnPlateau reduces learning rate when a metric has stopped improving
            mode = scheduler_params.get("mode", "min")
            factor = scheduler_params.get("factor", 0.9)
            patience = scheduler_params.get("patience", 10)
            threshold = scheduler_params.get("threshold", 1e-4)
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=mode,
                factor=factor,
                patience=patience,
                threshold=threshold,
            )
        else:
            # Default: no scheduler
            self.scheduler = None

        def _get_constraint_value(component_or_value):
            """Helper function to get constraint value, handling both ScheduleSystem and scalar values"""
            if isinstance(component_or_value, (int, float)):
                return torch.tensor(component_or_value)
            elif isinstance(component_or_value, systems.ScheduleSystem):
                component_or_value.initialize(
                    startTime=self.startTime,
                    endTime=self.endTime,
                    stepSize=self.stepSize,
                    simulator=self.simulator,
                )
                return component_or_value.output["scheduleValue"].history
            elif isinstance(component_or_value, torch.Tensor):
                return component_or_value
            else:
                raise ValueError(
                    f"Invalid constraint value type: {type(component_or_value)}"
                )

        # Pre-compute all constraint values
        self.equality_constraint_values = {}
        if self.equalityConstraints is not None:
            for component, output_name, desired_value in self.equalityConstraints:
                self.equality_constraint_values[component, output_name] = (
                    _get_constraint_value(desired_value)
                )

        self.inequality_constraint_values = {}
        if self.inequalityConstraints is not None:
            for (
                component,
                output_name,
                constraint_type,
                desired_value,
            ) in self.inequalityConstraints:
                self.inequality_constraint_values[
                    (component, output_name, constraint_type)
                ] = _get_constraint_value(desired_value)

        for i in range(iterations):
            # Perform optimization step
            self.optimizer.step(self._closure)

            # Update learning rate with scheduler
            if self.scheduler is not None:
                if isinstance(
                    self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    # ReduceLROnPlateau needs the loss value
                    self.scheduler.step(self.loss)
                else:
                    # Other schedulers just need to be stepped
                    self.scheduler.step()

            # Log current learning rate
            current_lr = self.optimizer.param_groups[0]["lr"]
            print(f"Current learning rate: {current_lr}")
            print(f"Loss at step {i}: {self.loss.detach().item()}")

    def _scipy_solver(self, method: tuple = None, **options):
        """
        Perform optimization using SciPy's optimization algorithms.

        This method uses SciPy's optimization library to solve constrained and unconstrained
        optimization problems. It supports both automatic differentiation (AD) and finite
        difference (FD) modes for gradient computation. The method automatically handles
        constraint formulation and bounds specification.

        Args:
            method: Tuple of (library, optimizer, mode) specifying the optimization method:
                - library: Always "scipy" for this method
                - optimizer: The specific optimization algorithm:
                    - "SLSQP": Sequential Least Squares Programming - preferred for most constrained problems
                    - "L-BFGS-B": Limited-memory BFGS with bounds - good for unconstrained or bound-constrained problems
                    - "TNC": Truncated Newton algorithm with bounds - efficient for large-scale problems
                    - "trust-constr": Trust-region constrained optimization - robust for difficult constraints
                    - "trf": Trust Region Reflective - specialized for least-squares problems
                    - "dogbox": Dogleg algorithm - alternative for least-squares problems
                - mode: Differentiation mode:
                    - "ad": Automatic differentiation using PyTorch (recommended)
                    - "fd": Finite difference (not yet implemented)
            **options: Additional options passed to the SciPy optimizer:
                - "verbose": Verbosity level (0-3) for optimization output
                - "maxiter": Maximum number of iterations
                - "gtol": Gradient tolerance for convergence
                - "xtol": Parameter tolerance for convergence
                - "barrier_tol": Barrier tolerance for interior point methods
                - "initial_tr_radius": Initial trust region radius
                - "initial_constr_penalty": Initial constraint penalty
                - Additional method-specific options as supported by SciPy optimizers

        Note:
            This method automatically handles the conversion between PyTorch tensors and
            NumPy arrays required by SciPy. It uses caching to avoid redundant computations
            when the same parameters are evaluated multiple times. The method supports
            both equality and inequality constraints through the loss function formulation.
        """
        if method is None:
            method = ("scipy", "SLSQP", "ad")

        for component in self.simulator.model.components.values():
            if isinstance(component, nn.Module):
                for parameter in component.parameters():
                    parameter.requires_grad_(False)

        # Set before initializing the model
        for component, output_name, *bounds in self.variables:
            component.output[output_name].do_normalization = True

        self.simulator.get_simulation_timesteps(
            self.startTime, self.endTime, self.stepSize
        )
        self.simulator.model.initialize(
            startTime=self.startTime,
            endTime=self.endTime,
            stepSize=self.stepSize,
            simulator=self.simulator,
        )

        # Create initial guess vector
        x0 = []
        bounds_list = []

        n_timesteps = len(self.simulator.dateTimeSteps)

        # Create flattened vector of size N*M
        for t in range(n_timesteps):
            for component, output_name, *bounds in self.variables:
                component.output[output_name].set_requires_grad(True)
                if component.output[output_name].do_normalization:
                    x0.append(
                        component.output[output_name].normalized_history[t].item()
                    )
                else:
                    x0.append(component.output[output_name].history[t].item())

                # Set bounds (same for all timesteps for each actuator)
                if len(bounds) >= 2:
                    lower, upper = bounds[0], bounds[1]
                    if component.output[output_name].do_normalization:
                        lower = (
                            component.output[output_name]
                            .normalize(torch.tensor(lower))
                            .item()
                        )
                        upper = (
                            component.output[output_name]
                            .normalize(torch.tensor(upper))
                            .item()
                        )
                    bounds_list.append((lower, upper))
                else:
                    bounds_list.append((None, None))

        x0 = np.array(x0)

        # Create bounds object for SciPy
        if all(b[0] is not None and b[1] is not None for b in bounds_list):
            bounds_obj = Bounds(
                [b[0] for b in bounds_list], [b[1] for b in bounds_list]
            )
        else:
            bounds_obj = None

        # Pre-compute constraint values
        def _get_constraint_value(component_or_value):
            """Helper function to get constraint value, handling both ScheduleSystem and scalar values"""
            if isinstance(component_or_value, (int, float)):
                return torch.tensor(component_or_value)
            elif isinstance(component_or_value, systems.ScheduleSystem):
                component_or_value.initialize(
                    startTime=self.startTime,
                    endTime=self.endTime,
                    stepSize=self.stepSize,
                    simulator=self.simulator,
                )
                return component_or_value.output["scheduleValue"].history
            elif isinstance(component_or_value, torch.Tensor):
                return component_or_value
            else:
                raise ValueError(
                    f"Invalid constraint value type: {type(component_or_value)}"
                )

        self.equality_constraint_values = {}
        if self.equalityConstraints is not None:
            for component, output_name, desired_value in self.equalityConstraints:
                self.equality_constraint_values[component, output_name] = (
                    _get_constraint_value(desired_value)
                )

        self.inequality_constraint_values = {}
        if self.inequalityConstraints is not None:
            for (
                component,
                output_name,
                constraint_type,
                desired_value,
            ) in self.inequalityConstraints:
                self.inequality_constraint_values[
                    (component, output_name, constraint_type)
                ] = _get_constraint_value(desired_value)

        # Initialize caching variables for AD
        self._theta_jac = torch.nan * torch.ones_like(
            torch.tensor(x0, dtype=torch.float64)
        )
        self._theta_hes = torch.nan * torch.ones_like(
            torch.tensor(x0, dtype=torch.float64)
        )
        self._theta_obj = torch.nan * torch.ones_like(
            torch.tensor(x0, dtype=torch.float64)
        )

        # Run optimization based on method
        optimizer_name = method[1]
        mode = method[2]

        if mode == "ad":
            # Use automatic differentiation
            if optimizer_name in ["trf", "dogbox"]:
                # These are least-squares optimizers
                least_squares(
                    self._obj_ad,
                    x0,
                    jac=self._jac_ad,
                    bounds=bounds_obj,
                    method=optimizer_name,
                    **options,
                )
            else:
                # These are general optimization algorithms
                minimize(
                    self._obj_ad,
                    x0,
                    method=optimizer_name,
                    jac=self._jac_ad,
                    bounds=bounds_obj,
                    options=options,
                )
        else:
            # Use finite difference (not implemented yet for optimizer)
            raise NotImplementedError(
                "Finite difference mode is not yet implemented for the optimizer. Use automatic differentiation mode."
            )

    def __obj_ad(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Objective function for automatic differentiation.

        Args:
            theta (torch.Tensor): Flattened parameter vector of size N*M where
                                 N = number of timesteps, M = number of actuators.

        Returns:
            torch.Tensor: Objective value.
        """
        # Reshape theta from flattened vector (N*M) to matrix (N, M)
        n_timesteps = len(self.simulator.dateTimeSteps)
        n_actuators = len(self.variables)
        theta_matrix = theta.reshape(n_timesteps, n_actuators)
        # Update decision variables for each timestep using proper initialization
        for i, (component, output_name, *bounds) in enumerate(self.variables):
            # Extract values for this actuator across all timesteps
            values = component.output[output_name].denormalize(theta_matrix[:, i])
            # Initialize with the new values

            component.output[output_name].initialize(
                startTime=self.startTime,
                endTime=self.endTime,
                stepSize=self.stepSize,
                simulator=self.simulator,
                values=values,
                force=True,
            )

        # Run simulation
        self.simulator.simulate(
            startTime=self.startTime,
            endTime=self.endTime,
            stepSize=self.stepSize,
            show_progress_bar=False,
        )

        # Compute loss
        loss = 0
        k = 100

        # Handle equality constraints
        if self.equalityConstraints is not None:
            for constraint in self.equalityConstraints:
                component, output_name, desired_value = constraint
                y = component.output[output_name].history
                # print(f"{component.id}.{output_name}.history.grad_fn", y.grad_fn)
                desired_tensor = self.equality_constraint_values[component, output_name]
                y_norm = component.output[output_name].normalize(y)
                desired_tensor_norm = component.output[output_name].normalize(
                    desired_tensor
                )
                loss += torch.mean(torch.abs(y_norm - desired_tensor_norm))

        # Handle inequality constraints
        if self.inequalityConstraints is not None:
            ineq_upper_term = 0
            ineq_lower_term = 0
            for constraint in self.inequalityConstraints:
                component, output_name, constraint_type, desired_value = constraint
                y = component.output[output_name].history
                # print(f"{component.id}.{output_name}.history.grad_fn", y.grad_fn)
                desired_tensor = self.inequality_constraint_values[
                    (component, output_name, constraint_type)
                ]
                y_norm = component.output[output_name].normalize(y)
                desired_tensor_norm = component.output[output_name].normalize(
                    desired_tensor
                )

                if constraint_type == "upper":
                    # Penalize when y > desired_value
                    constraint_violations = torch.relu(y_norm - desired_tensor_norm)
                    ineq_upper_term += torch.mean(k * constraint_violations)
                elif constraint_type == "lower":
                    # Penalize when y < desired_value
                    constraint_violations = torch.relu(desired_tensor_norm - y_norm)
                    ineq_lower_term += torch.mean(k * constraint_violations)

            loss += ineq_upper_term + ineq_lower_term

        # Handle minimization objectives
        if self.objectives is not None:
            for component, output_name, objective_type in self.objectives:
                y = component.output[output_name].history
                y_norm = component.output[output_name].normalize(y)
                if objective_type == "min":
                    loss += torch.mean(y_norm)
                elif objective_type == "max":
                    loss += -torch.mean(y_norm)

        self.obj = loss
        return self.obj

    def _obj_ad(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Wrapper function for SciPy interface that converts numpy to torch and returns numpy.

        Args:
            theta (torch.Tensor): Parameter vector.

        Returns:
            torch.Tensor: Objective value as numpy array.
        """
        theta = torch.tensor(theta, dtype=torch.float64)
        if torch.equal(theta, self._theta_obj):
            return self.obj.detach().numpy()
        else:
            self._theta_obj = theta
            self.obj = self.__obj_ad(theta)

            # self._hes_ad(theta) # hes calls jac which calls obj.
            return self.obj.detach().numpy()

    def __jac_ad(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Compute the Jacobian matrix using automatic differentiation.

        Args:
            theta (torch.Tensor): Parameter vector.

        Returns:
            torch.Tensor: Jacobian matrix.
        """
        self.jac = torch.func.jacrev(self.__obj_ad, argnums=0)(theta)
        return self.jac

    def _jac_ad(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Compute the Jacobian matrix using automatic differentiation.

        Args:
            theta (torch.Tensor): Parameter vector.

        Returns:
            torch.Tensor: Jacobian matrix.
        """
        theta = torch.tensor(theta, dtype=torch.float64)

        if torch.equal(theta, self._theta_jac):
            return self.jac.detach().numpy()
        else:
            self._theta_jac = theta
            self.jac = self.__jac_ad(theta)
            return self.jac.detach().numpy()

    def __hes_ad(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Compute the Hessian matrix using automatic differentiation.

        Args:
            theta (torch.Tensor): Parameter vector.

        Returns:
            torch.Tensor: Hessian matrix.
        """
        self.hes = torch.func.jacfwd(self.__jac_ad, argnums=0)(theta)
        return self.hes

    def _hes_ad(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Compute the Hessian matrix using automatic differentiation.

        Args:
            theta (torch.Tensor): Parameter vector.

        Returns:
            torch.Tensor: Hessian matrix.
        """
        theta = torch.tensor(theta, dtype=torch.float64)

        if torch.equal(theta, self._theta_hes):
            return self.hes.detach().numpy()
        else:
            self._theta_hes = theta
            self.hes = self.__hes_ad(theta)
            return self.hes.detach().numpy()
