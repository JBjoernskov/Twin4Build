# import pygad
# Standard library imports
import datetime
import os
import time as time_module
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple, Union

# Third party imports
import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import Bounds, least_squares, minimize

# Local application imports
import twin4build.core as core
import twin4build.systems as systems
import twin4build.utils.types as tps
from twin4build.utils.deprecation import reject_unexpected_kwargs
from twin4build.utils.logger import LOGGER
from twin4build.utils.method_spec import parse_method
from twin4build.utils.result import ResultDict
from twin4build.utils.validate_period import validate_period
from twin4build.optimizer._pareto import pareto_front as _pareto_front
from twin4build.optimizer._single_shooting import (
    CapturedControlObjective,
    FunctionalControlObjective,
)


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

    For composable torch models, select
    ``Simulator(model, execution_mode="functional")`` to evaluate the loss with a
    pure one-step map. Exogenous inputs are captured once and each evaluation
    becomes a sequential torch rollout with one autograd pass. Models the
    functional model cannot express fall back to the object objective.

    Args:
        simulator: The simulator instance for running simulations.

    Mathematical Formulation
    ------------------------

    The general optimization problem is formulated as:

        .. math::

            \hat{\boldsymbol{U}} = \underset{\boldsymbol{U} \in \mathcal{U}}{\operatorname{argmin}} \; \mathcal{L}(\boldsymbol{U})

    where:
        - :math:`\hat{\boldsymbol{U}}` is the optimal control input matrix
        - :math:`\boldsymbol{U}` is the control input matrix
        - :math:`\mathcal{U} \subseteq \mathbb{R}^{n_u \times n_t}` is the set of feasible control inputs
        - :math:`\mathcal{L}(\boldsymbol{U})` is the loss function

    Dimensions
    ~~~~~~~~~~

    - :math:`n_t`: Number of time steps in the simulation period
    - :math:`n_u`: Number of control inputs (actuators)
    - :math:`n_d`: Number of disturbance inputs (weather, occupancy, etc.)
    - :math:`n_y`: Number of system outputs (sensors, performance metrics)

    Model Structure
    ~~~~~~~~~~~~~~~

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
    ~~~~~~~~~~~~~

    The loss function :math:`\mathcal{L}(\boldsymbol{U})` is composed of the following terms:

    Equality Constraints
    ^^^^^^^^^^^^^^^^^^^^

        .. math::

            \mathcal{L}_{eq} = \frac{1}{n_t} \sum_{t=1}^{n_t} \sum_{(j, \boldsymbol{y}) \in \mathcal{C}_{eq}} |\boldsymbol{\hat{Y}}_{j,t} - \boldsymbol{y}_{t}|

        where :math:`\mathcal{C}_{eq}` is the set of equality constraints, each element is (output index :math:`j`, desired value :math:`\boldsymbol{y}_{t}`).

    Inequality Constraints
    ^^^^^^^^^^^^^^^^^^^^^^

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

    Objective Terms
    ^^^^^^^^^^^^^^^

        .. math::

            \mathcal{L}_{obj} = \frac{1}{n_t} \sum_{t=1}^{n_t} \sum_{(j, w) \in \mathcal{O}_{obj}} w \cdot \boldsymbol{\hat{Y}}_{j,t}

        where :math:`\mathcal{O}_{obj}` is the set of outputs to minimize or maximize, and :math:`w` is a weight (+1 for minimization, -1 for maximization).

    Total Loss
    ^^^^^^^^^^

        .. math::

            \mathcal{L}(\boldsymbol{U}) = \mathcal{L}_{eq} + \mathcal{L}_{ineq} + \mathcal{L}_{obj}

    See method docstrings for details on the specific loss terms and optimization algorithms.

    Examples
    --------
    Basic optimization:

    >>> import twin4build as tb
    >>> import datetime
    >>> import pytz
    >>>
    >>> # Create model and simulator
    >>> model = tb.SimulationModel(id="my_model")
    >>> simulator = tb.Simulator(model)
    >>> optimizer = tb.Optimizer(simulator)
    >>>
    >>> # Define decision variables (actuators to optimize) with bounds
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
    >>> # Run optimization (SLSQP with automatic differentiation, the default)
    >>> optimizer.optimize(
    ...     variables=variables,
    ...     objectives=objectives,
    ...     start_time=start,
    ...     end_time=end,
    ...     step_size=step,
    ...     method=("scipy", "SLSQP", "ad")
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
    ...     eq_cons=equality_constraints,
    ...     ineq_cons=inequality_constraints,
    ...     start_time=start,
    ...     end_time=end,
    ...     step_size=step,
    ...     method=("scipy", "SLSQP", "ad"),
    ...     options={"verbose": 2, "maxiter": 1000}
    ... )

    Alternative SciPy methods:

    >>> # Use L-BFGS-B for unconstrained optimization
    >>> optimizer.optimize(
    ...     variables=variables,
    ...     objectives=objectives,
    ...     start_time=start,
    ...     end_time=end,
    ...     step_size=step,
    ...     method=("scipy", "L-BFGS-B", "ad"),
    ...     options={"gtol": 1e-8, "maxiter": 500}
    ... )

    >>> # Use trust-region method for difficult constraints
    >>> optimizer.optimize(
    ...     variables=variables,
    ...     objectives=objectives,
    ...     eq_cons=equality_constraints,
    ...     ineq_cons=inequality_constraints,
    ...     start_time=start,
    ...     end_time=end,
    ...     step_size=step,
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
    ...     eq_cons=equality_constraints,
    ...     start_time=start,
    ...     end_time=end,
    ...     step_size=step,
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
    ...     start_time=start,
    ...     end_time=end,
    ...     step_size=step,
    ...     method=("scipy", "SLSQP", "ad"),
    ...     options={"ftol": 1e-9, "maxiter": 2000}
    ... )

    Legacy string format (still supported):

    >>> # Simple usage with default settings
    >>> optimizer.optimize(
    ...     variables=variables,
    ...     objectives=objectives,
    ...     start_time=start,
    ...     end_time=end,
    ...     step_size=step,
    ...     method="scipy"  # Defaults to ("scipy", "SLSQP", "ad")
    ... )
    """

    def __init__(self, simulator: core.Simulator):
        assert isinstance(
            simulator, core.Simulator
        ), "Simulator must be a twin4build.core.Simulator instance"
        self.simulator = simulator
        self._functional_objective = None

    @property
    def _device(self) -> torch.device:
        """The model's device.  Solver-facing numpy boundaries convert inbound
        theta vectors to this device and outbound values via .cpu().numpy();
        scipy/IPOPT itself always runs on the CPU."""
        return self.simulator.model.device

    # def _closure(self):
    #     self.optimizer.zero_grad()
    #
    #     # Apply bounds to decision variables
    #     with torch.no_grad():
    #         for component, output_name, *bounds in self._variables:
    #             if len(bounds) > 0:
    #                 lower_bound = bounds[0] if len(bounds) > 0 else float("-inf")
    #                 upper_bound = bounds[1] if len(bounds) > 1 else float("inf")
    #                 if component.output[output_name].do_normalization:
    #                     lower_bound_ = component.output[output_name].normalize(
    #                         lower_bound
    #                     )
    #                     upper_bound_ = component.output[output_name].normalize(
    #                         upper_bound
    #                     )
    #                     # print("==========================")
    #                     # print(f"CLAMPED BEFORE: {component.id}.{output_name} to {component.output[output_name].denormalize(component.output[output_name].normalized_history)}")
    #                     component.output[output_name].normalized_history.clamp_(
    #                         min=lower_bound_, max=upper_bound_
    #                     )
    #
    #                     # print("==========================")
    #                     # print(f"CLAMPED AFTER: {component.id}.{output_name} to {component.output[output_name].denormalize(component.output[output_name].normalized_history)}")
    #                 else:
    #                     component.output[output_name].history.clamp_(
    #                         min=lower_bound, max=upper_bound
    #                     )
    #
    #     # Run simulation
    #     self.simulator.simulate(
    #         start_time=self._start_time,
    #         end_time=self._end_time,
    #         step_size=self._stepSize,
    #         show_progress_bar=False,
    #     )
    #
    #     self.loss = 0
    #     k = 100
    #
    #     # Handle equality constraints
    #     if self._eq_cons is not None:
    #         eq_term = 0
    #         for constraint in self._eq_cons:
    #             component, output_name, desired_value = constraint
    #             y = component.output[
    #                 output_name
    #             ].history  # Shape: [n_periods, n_timesteps]
    #             desired_tensor = self.equality_constraint_values[component, output_name]
    #             y = component.output[output_name].normalize(y)
    #             desired_tensor = component.output[output_name].normalize(desired_tensor)
    #
    #             # Aggregate loss across all periods
    #             eq_term += torch.nanmean(torch.abs(y - desired_tensor))
    #         self.loss += eq_term
    #
    #     # Handle inequality constraints
    #     if self._ineq_cons is not None:
    #         ineq_upper_term = torch.tensor(0.0, dtype=tps.float_dtype(), device=self._device)
    #         ineq_lower_term = torch.tensor(0.0, dtype=tps.float_dtype(), device=self._device)
    #         for constraint in self._ineq_cons:
    #             component, output_name, constraint_type, desired_value = constraint
    #             y = component.output[
    #                 output_name
    #             ].history  # Shape: [n_periods, n_timesteps]
    #             desired_tensor = self.inequality_constraint_values[
    #                 (component, output_name, constraint_type)
    #             ]
    #             y_norm = component.output[output_name].normalize(y)
    #             desired_tensor_norm = component.output[output_name].normalize(
    #                 desired_tensor
    #             )
    #
    #             if constraint_type == "upper":
    #                 # Penalize when y > desired_value
    #                 constraint_violations = torch.relu(y_norm - desired_tensor_norm)
    #                 constraint_term = torch.nanmean(k * constraint_violations)
    #                 ineq_upper_term += constraint_term
    #
    #             elif constraint_type == "lower":
    #                 # Penalize when y < desired_value
    #                 constraint_violations = torch.relu(desired_tensor_norm - y_norm)
    #                 constraint_term = torch.nanmean(k * constraint_violations)
    #                 ineq_lower_term += constraint_term
    #
    #         self.loss += ineq_upper_term + ineq_lower_term
    #
    #     # Handle minimization objectives
    #     if self._objectives is not None:
    #         min_term = 0
    #         for minimize_obj in self._objectives:
    #             component, output_name = minimize_obj
    #             y = component.output[
    #                 output_name
    #             ].history  # Shape: [n_periods, n_timesteps]
    #             y_norm = component.output[output_name].normalize(y)
    #             # print(f"NORMALIZED MINIMIZE OBJECTIVE BETWEEN: {component.output[output_name]._min_history} and {component.output[output_name]._max_history}")
    #
    #             # Aggregate loss across all periods
    #             min_term += torch.nanmean(y_norm)
    #         self.loss += min_term  # Minimize the mean value
    #
    #     # Compute gradients
    #     self.loss.backward()
    #     return self.loss

    def optimize(
        self,
        start_time: Union[datetime.datetime, List[datetime.datetime]] = None,
        end_time: Union[datetime.datetime, List[datetime.datetime]] = None,
        step_size: Union[int, List[int]] = None,
        variables: List[Tuple[Any, str, float, float]] = None,
        objectives: List[Tuple[Any, str, str]] = None,
        eq_cons: List[Tuple[Any, str, Any]] = None,
        ineq_cons: List[Tuple[Any, str, str, Any]] = None,
        method: Union[str, Tuple[str, str, str]] = "scipy",
        options: Dict = None,
        **kwargs,
    ):
        """
        Optimize the model control inputs using the specified optimization method.

        The decision variables are the full time series of the given actuator
        outputs (one value per timestep per variable), bounded by the supplied
        lower/upper bounds. Output constraints are handled as soft penalties in
        the loss function (see the class docstring's Loss Function section).

        Args:
            start_time: Start time(s) for simulation (timezone-aware). A single
                datetime or a list of datetimes for multiple periods.
            end_time: End time(s) for simulation. Same form as ``start_time``.
            step_size: Step size(s) for simulation in seconds.
            variables: List of tuples (component, output_name, lower_bound, upper_bound).
                The decision variables (actuator trajectories) to optimize.
            objectives: List of tuples (component, output_name, objective_type)
                where objective_type is "min" or "max".
            eq_cons: List of tuples (component, output_name, desired_value) where
                desired_value is a constant or a schedule component providing the
                time-varying target.
            ineq_cons: List of tuples (component, output_name, constraint_type, desired_value)
                where constraint_type is "upper" or "lower".

            method: Optimization method specification. Either the legacy
                string ``"scipy"`` (defaults to SLSQP with automatic
                differentiation) or, recommended, a tuple
                ``(library, optimizer, mode)`` where ``library`` is
                ``"scipy"`` (currently the only supported library),
                ``optimizer`` is the algorithm name, and ``mode`` is ``"ad"``
                (automatic differentiation) or ``"fd"`` (finite difference).

                Supported SciPy optimizers:

                - "SLSQP": Sequential Least Squares Programming (preferred for most problems)
                - "L-BFGS-B": Limited-memory BFGS with bounds
                - "TNC": Truncated Newton algorithm with bounds
                - "trust-constr": Trust-region constrained optimization

                Examples: ``("scipy", "SLSQP", "ad")`` is preferred for most
                constrained optimization problems.

            options: Additional options for the chosen method:

                - "verbose": Verbosity level (0-3)
                - "maxiter": Maximum iterations
                - "gtol": Gradient tolerance
                - "xtol": Parameter tolerance
                - "barrier_tol": Barrier tolerance
                - "initial_tr_radius": Initial trust region radius
                - "initial_constr_penalty": Initial constraint penalty
                - "constraint_penalty": Weight of the soft constraint penalty
                  terms in the loss (default 100)
                - Additional method-specific options as supported by SciPy optimizers

                Functional execution and CUDA graph capture are selected on
                ``Simulator`` via ``execution_mode`` and
                ``execution_backend``, not through these options.

        Returns:
            OptimizationResult: A dict-like result with SciPy-compatible
            mapping and attribute access. The optimized actuator
            trajectories are also applied to the model, so a subsequent
            ``simulator.simulate(...)`` runs with the optimal inputs.
        """

        for legacy_key, new_key in (
            ("startTime", "start_time"),
            ("endTime", "end_time"),
            ("stepSize", "step_size"),
        ):
            if legacy_key in kwargs:
                raise TypeError(
                    f"`{legacy_key}` has been removed. Use `{new_key}` instead."
                )
        reject_unexpected_kwargs("Optimizer.optimize", kwargs)

        self._variables = variables or []
        self._objectives = objectives or []
        self._eq_cons = eq_cons or []
        self._ineq_cons = ineq_cons or []

        start_time, end_time, step_size = validate_period(
            start_time, end_time, step_size
        )

        self._start_time = start_time
        self._end_time = end_time
        self._stepSize = step_size
        self._max_values = {}

        # Validate input arguments
        # Check required simulation parameters
        assert start_time is not None, "start_time must be provided"
        assert end_time is not None, "end_time must be provided"
        assert step_size is not None, "step_size must be provided"

        (
            self._second_time_steps,
            self._date_time_steps,
            self._max_timesteps,
            self._n_timesteps,
        ) = core.Simulator.get_simulation_timesteps(
            self._start_time, self._end_time, self._stepSize
        )

        timestep_mask = torch.ones(
            self._max_timesteps, len(self._start_time), dtype=torch.bool
        )
        for i_s, n_timesteps in enumerate(self._n_timesteps):
            timestep_mask[n_timesteps:, i_s] = False
        self._timestep_mask = timestep_mask  # .bool()

        # Check that we have something to optimize
        assert (
            len(self._variables) > 0
        ), "No decision variables specified for optimization"

        for obj in self._objectives:
            component, output_name, objective_type = obj
            assert objective_type in [
                "min",
                "max",
            ], f"Objective type must be 'min' or 'max', got '{objective_type}'"

        # Check that we have at least one objective (minimize or constraints)
        has_objective = (
            len(self._objectives) > 0
            or len(self._eq_cons) > 0
            or len(self._ineq_cons) > 0
        )
        assert (
            has_objective
        ), "No optimization objectives specified (minimize, eq_cons, or ineq_cons)"

        # Validate method
        # Define allowed optimization methods
        allowed_methods = [
            # ("torch", "SGD", "ad"),
            # ("torch", "Adam", "ad"),
            # ("torch", "LBFGS", "ad"),
            ("scipy", "SLSQP", "ad"),
            ("scipy", "L-BFGS-B", "ad"),
            ("scipy", "TNC", "ad"),
            ("scipy", "trust-constr", "ad"),
        ]
        default_methods = [("scipy", "SLSQP", "ad")]
        default_mode = (
            "ad"  # Always choose automatic differentiation mode when ambiguous
        )

        method, _transcription = parse_method(
            method,
            allowed_methods=allowed_methods,
            default_methods=default_methods,
            default_mode=default_mode,
            default_none_method=default_methods[0],
            allow_transcription=False,
        )

        # Validate format of decision variables
        for i, decision_var in enumerate(self._variables):
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
        for i, min_obj in enumerate(self._objectives):
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
        for i, eq_constraint in enumerate(self._eq_cons):
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
        for i, ineq_constraint in enumerate(self._ineq_cons):
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
        if self._objectives and self._eq_cons:
            minimize_pairs = {
                (component, output_name)
                for component, output_name, _ in self._objectives
            }
            equality_pairs = {
                (component, output_name) for component, output_name, _ in self._eq_cons
            }

            conflicting_pairs = minimize_pairs.intersection(equality_pairs)
            if conflicting_pairs:
                conflict_info = [f"({c.id}, {o})" for c, o in conflicting_pairs]
                raise ValueError(
                    f"Cannot simultaneously minimize and apply equality constraints to the same outputs: {', '.join(conflict_info)}. "
                    f"These objectives conflict with each other."
                )

        LOGGER.task("Running optimization")
        LOGGER.add_level()
        LOGGER.config("Method: %s", method)
        LOGGER.config("Variables: %d", len(self._variables))
        LOGGER.add_level()
        for component, output_name, *bounds in self._variables:
            bounds_str = (
                f" (lb={bounds[0]}, ub={bounds[1]})" if len(bounds) >= 2 else ""
            )
            LOGGER.debug("%s.%s%s", component.id, output_name, bounds_str)
        LOGGER.remove_level()
        LOGGER.config("Objectives: %d", len(self._objectives))
        LOGGER.add_level()
        for component, output_name, obj_type in self._objectives:
            LOGGER.debug("%s: %s.%s", obj_type, component.id, output_name)
        LOGGER.remove_level()
        if self._eq_cons:
            LOGGER.config("Equality constraints: %d", len(self._eq_cons))
        if self._ineq_cons:
            LOGGER.config("Inequality constraints: %d", len(self._ineq_cons))

        n_periods = len(self._start_time)
        LOGGER.config("Time periods: %d", n_periods)
        LOGGER.add_level()
        for i, (s, e, ss) in enumerate(
            zip(self._start_time, self._end_time, self._stepSize)
        ):
            LOGGER.config("Period %d: %s -> %s (step=%ss)", i + 1, s, e, ss)
        LOGGER.remove_level()

        # Check for decision variables that are also in equality constraints
        if self._variables and self._eq_cons:
            decision_pairs = {
                (component, output_name)
                for component, output_name, *_ in self._variables
            }
            equality_pairs = {
                (component, output_name) for component, output_name, _ in self._eq_cons
            }

            conflicting_pairs = decision_pairs.intersection(equality_pairs)
            if conflicting_pairs:
                conflict_info = [f"({c.id}, {o})" for c, o in conflicting_pairs]
                LOGGER.remove_level()
                LOGGER.error("Running optimization", change_status=True)
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
        # if method[0] == "torch":
        #     if options is None:
        #         options = {}
        #     # Extract optimizer type from method tuple
        #     optimizer_type = method[1]
        #     options["optimizer_type"] = optimizer_type
        #     return self._torch_solver(**options)
        if method[0] == "scipy":
            if options is None:
                options = {}
            # Fast-path for notebook example tests: keep the cell exercising
            # the full Optimizer API (so we still catch wiring / API
            # regressions) but stop the solver after a single iteration.
            # Honors the env var set by ``utils.test_notebook.test_notebook``;
            # callers in the regular test suite already pass small
            # ``maxiter`` values explicitly, so this is a no-op for them.
            if os.environ.get("TWIN4BUILD_TESTING", "").lower() in (
                "1",
                "true",
                "yes",
            ):
                options["maxiter"] = 1
            result = self._solve_scipy(method=method, **options)
        else:
            LOGGER.remove_level()
            LOGGER.error("Running optimization", change_status=True)
            raise ValueError("Unsupported optimization method: %s" % method[0])

        LOGGER.remove_level()
        LOGGER.ok("Running optimization", change_status=True)
        return OptimizationResult.from_scipy(result)

    def pareto_front(
        self,
        start_time: Union[datetime.datetime, List[datetime.datetime]] = None,
        end_time: Union[datetime.datetime, List[datetime.datetime]] = None,
        step_size: Union[int, List[int]] = None,
        variables: List[Tuple[Any, str, float, float]] = None,
        objective1: Tuple[Any, str, str] = None,
        objective2: Tuple[Any, str, str] = None,
        eq_cons: List[Tuple[Any, str, Any]] = None,
        ineq_cons: List[Tuple[Any, str, str, Any]] = None,
        n_points: int = 11,
        delta: float = 1e-3,
        method: tuple = ("scipy", "SLSQP", "ad"),
        batched_prepass: bool = True,
        prepass_options: Dict = None,
        options: Dict = None,
    ):
        """Trace a bi-objective front with the augmented epsilon-constraint method.

        Functional execution, including the optional batched prepass, is selected
        by constructing ``Simulator(model, execution_mode="functional")``.
        Optimizer options do not select execution mode.  Supported methods are
        ``("scipy", "SLSQP", "ad")`` and
        ``("casadi", "ipopt", "ad", "collocation")``.
        SLSQP and IPOPT remain host solvers; the CUDA Graph backend can capture
        and replay fixed-shape device derivatives. IPOPT receives the exact
        sparse Hessian of the collocation Lagrangian. The old three-element
        IPOPT spelling is rejected because it ambiguously implied direct shooting.

        ``("custom", "batched-tr", "ad")`` instead solves every
        epsilon-subproblem at once on the device with the block trust-region
        step: no host solver and no separate prepass.  It needs
        ``execution_mode="functional"`` and takes the trust-region options
        (``tr_radius``, ``tr_retries``, ...) rather than the SciPy/IPOPT ones.
        """
        if tuple(method) not in (
            ("scipy", "SLSQP", "ad"),
            ("casadi", "ipopt", "ad", "collocation"),
            ("custom", "batched-tr", "ad"),
        ):
            raise ValueError(
                "pareto_front requires exact AD derivatives with "
                '("scipy", "SLSQP", "ad"), '
                '("casadi", "ipopt", "ad", "collocation") or '
                f'("custom", "batched-tr", "ad"); got {method}.'
            )
        for name, obj in (("objective1", objective1), ("objective2", objective2)):
            if obj is None or len(obj) != 3:
                raise ValueError(
                    f"{name} must be a (component, output_name, 'min'|'max') tuple"
                )
            component, output_name, objective_type = obj
            if not hasattr(component, "output") or output_name not in component.output:
                raise ValueError(f"{name}: output '{output_name}' is not available")
            if objective_type not in ("min", "max"):
                raise ValueError(
                    f"{name}: objective type must be 'min' or 'max', got "
                    f"'{objective_type}'"
                )
        if n_points < 2:
            raise ValueError("n_points must be at least 2")
        if not variables:
            raise ValueError("No decision variables specified for optimization")

        self._variables = variables
        self._objectives = [tuple(objective1), tuple(objective2)]
        self._eq_cons = eq_cons or []
        self._ineq_cons = ineq_cons or []
        self._start_time, self._end_time, self._stepSize = validate_period(
            start_time, end_time, step_size
        )
        self._max_values = {}
        (
            self._second_time_steps,
            self._date_time_steps,
            self._max_timesteps,
            self._n_timesteps,
        ) = core.Simulator.get_simulation_timesteps(
            self._start_time, self._end_time, self._stepSize
        )
        self._timestep_mask = torch.ones(
            self._max_timesteps, len(self._start_time), dtype=torch.bool
        )
        for i_s, n_timesteps in enumerate(self._n_timesteps):
            self._timestep_mask[n_timesteps:, i_s] = False

        options = dict(options or {})
        prepass_options = dict(prepass_options or {})
        if os.environ.get("TWIN4BUILD_TESTING", "").lower() in ("1", "true", "yes"):
            options["maxiter"] = 1
            n_points = min(n_points, 3)
            prepass_options.setdefault("max_iter", 3)

        LOGGER.task("Generating Pareto front")
        LOGGER.add_level()
        try:
            result = _pareto_front(
                self,
                n_points=n_points,
                delta=delta,
                method=method,
                use_prepass=batched_prepass,
                prepass_options=prepass_options,
                options=options,
            )
        except Exception:
            LOGGER.remove_level()
            LOGGER.error("Generating Pareto front", change_status=True)
            raise
        LOGGER.remove_level()
        LOGGER.ok("Generating Pareto front", change_status=True)
        return result

    # def _torch_solver(
    #     self,
    #     lr: float = 1.0,
    #     iterations: int = 100,
    #     optimizer_type: str = "SGD",
    #     scheduler_type: str = "step",
    #     scheduler_params: Dict = None,
    # ):
    #     """
    #     Perform optimization using PyTorch-based gradient optimization.

    #     This method uses PyTorch's automatic differentiation to compute gradients and
    #     applies gradient-based optimization algorithms to minimize the objective function.
    #     It supports various optimizers and learning rate schedulers for fine-tuning
    #     the optimization process.

    #     Args:
    #         lr: Learning rate for optimizer. Controls the step size in gradient descent.
    #             Higher values may converge faster but risk overshooting, while lower
    #             values are more stable but may converge slowly.
    #         iterations: Number of optimization iterations. More iterations generally
    #             lead to better convergence but take longer to compute.
    #         optimizer_type: Type of PyTorch optimizer:
    #             - "SGD": Stochastic Gradient Descent - simple, robust, good for most problems
    #             - "Adam": Adaptive learning rate optimizer - often faster convergence
    #             - "LBFGS": Limited-memory BFGS - good for smooth, well-behaved functions
    #         scheduler_type: Type of learning rate scheduler to adjust learning rate during optimization:
    #             - "step": Decreases learning rate by gamma every step_size iterations
    #             - "exponential": Decreases learning rate exponentially
    #             - "cosine": Uses cosine annealing schedule
    #             - "reduce_on_plateau": Reduces learning rate when loss stops improving
    #             - None: No scheduler, constant learning rate
    #         scheduler_params: Dictionary of parameters for the chosen scheduler:
    #             - For "step": {"step_size": int, "gamma": float}
    #             - For "exponential": {"gamma": float}
    #             - For "cosine": {"T_max": int, "eta_min": float}
    #             - For "reduce_on_plateau": {"mode": str, "factor": float, "patience": int, "threshold": float}

    #     Note:
    #         This method automatically handles gradient computation and parameter updates.
    #         It disables gradients for model parameters and only optimizes the decision variables.
    #         The optimization process is logged with current learning rate and loss values.
    #     """
    #     # Validate optimization parameters
    #     assert lr > 0, f"Learning rate must be positive, got {lr}"
    #     assert (
    #         iterations > 0
    #     ), f"Number of iterations must be positive, got {iterations}"

    #     # Validate scheduler type
    #     valid_scheduler_types = [
    #         "step",
    #         "exponential",
    #         "cosine",
    #         "reduce_on_plateau",
    #         None,
    #     ]
    #     assert (
    #         scheduler_type in valid_scheduler_types
    #     ), f"Invalid scheduler_type: {scheduler_type}. Must be one of {valid_scheduler_types}"

    #     # Disable gradients for all parameters since we're optimizing inputs.
    #     # It is VERY important to do this before initializing the model.
    #     # Otherwise, the model parameters and state space matrices will have requires_grad=True
    #     # and the backpropagate() call will fail.
    #     for component in self.simulator.model.components.values():
    #         if isinstance(component, nn.Module):
    #             for parameter in component.parameters():
    #                 parameter.requires_grad_(False)

    #     # Set before initializing the model
    #     for component, output_name, *bounds in self._variables:
    #         component.output[output_name].do_normalization = True

    #     self.simulator.model.initialize(
    #         start_time=self._start_time,
    #         end_time=self._end_time,
    #         step_size=self._stepSize,
    #         simulator=self.simulator,
    #     )

    #     # Enable gradients only for the inputs we want to optimize
    #     opt_list = []
    #     for component, output_name, *bounds in self._variables:
    #         component.output[output_name].set_requires_grad(True)
    #         if component.output[output_name].do_normalization:
    #             opt_list.append(component.output[output_name].normalized_history)
    #         else:
    #             opt_list.append(component.output[output_name].history)

    #     if optimizer_type == "SGD":
    #         # Initialize optimizer
    #         self.optimizer = torch.optim.SGD(opt_list, lr=lr)
    #     elif optimizer_type == "Adam":
    #         self.optimizer = torch.optim.Adam(opt_list, lr=lr)
    #     elif optimizer_type == "LBFGS":
    #         self.optimizer = torch.optim.LBFGS(
    #             opt_list, lr=lr, line_search_fn=None, history_size=100
    #         )
    #     else:
    #         raise ValueError(
    #             f"Invalid optimizer type: {optimizer_type}. Must be one of {['SGD', 'Adam', 'LBFGS']}"
    #         )

    #     # Initialize scheduler
    #     if scheduler_params is None:
    #         scheduler_params = {}

    #     if scheduler_type == "step":
    #         # StepLR decreases learning rate by gamma every step_size epochs
    #         step_size = scheduler_params.get("step_size", 30)
    #         gamma = scheduler_params.get("gamma", 0.1)
    #         self.scheduler = torch.optim.lr_scheduler.StepLR(
    #             self.optimizer, step_size=step_size, gamma=gamma
    #         )
    #     elif scheduler_type == "exponential":
    #         # ExponentialLR decreases learning rate by gamma every epoch
    #         gamma = scheduler_params.get("gamma", 0.95)
    #         self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
    #             self.optimizer, gamma=gamma
    #         )
    #     elif scheduler_type == "cosine":
    #         # CosineAnnealingLR uses a cosine schedule to decrease learning rate
    #         T_max = scheduler_params.get("T_max", 100)
    #         eta_min = scheduler_params.get("eta_min", 0)
    #         self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #             self.optimizer, T_max=T_max, eta_min=eta_min
    #         )
    #     elif scheduler_type == "reduce_on_plateau":
    #         # ReduceLROnPlateau reduces learning rate when a metric has stopped improving
    #         mode = scheduler_params.get("mode", "min")
    #         factor = scheduler_params.get("factor", 0.9)
    #         patience = scheduler_params.get("patience", 10)
    #         threshold = scheduler_params.get("threshold", 1e-4)
    #         self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #             self.optimizer,
    #             mode=mode,
    #             factor=factor,
    #             patience=patience,
    #             threshold=threshold,
    #         )
    #     else:
    #         # Default: no scheduler
    #         self.scheduler = None

    #     def _get_constraint_value(component_or_value):
    #         """Helper function to get constraint value, handling both ScheduleSystem and scalar values"""
    #         if isinstance(component_or_value, (int, float)):
    #             return torch.tensor(component_or_value)
    #         elif isinstance(component_or_value, systems.ScheduleSystem):
    #             component_or_value.initialize(
    #                 start_time=self._start_time,
    #                 end_time=self._end_time,
    #                 step_size=self._stepSize,
    #             )
    #             return component_or_value.output["scheduleValue"].history
    #         elif isinstance(component_or_value, torch.Tensor):
    #             return component_or_value
    #         else:
    #             raise ValueError(
    #                 f"Invalid constraint value type: {type(component_or_value)}"
    #             )

    #     # Pre-compute all constraint values
    #     self.equality_constraint_values = {}
    #     if self._eq_cons is not None:
    #         for component, output_name, desired_value in self._eq_cons:
    #             self.equality_constraint_values[component, output_name] = (
    #                 _get_constraint_value(desired_value)
    #             )

    #     self.inequality_constraint_values = {}
    #     if self._ineq_cons is not None:
    #         for (
    #             component,
    #             output_name,
    #             constraint_type,
    #             desired_value,
    #         ) in self._ineq_cons:
    #             self.inequality_constraint_values[
    #                 (component, output_name, constraint_type)
    #             ] = _get_constraint_value(desired_value)

    #     for i in range(iterations):
    #         # Perform optimization step
    #         self.optimizer.step(self._closure)

    #         # Update learning rate with scheduler
    #         if self.scheduler is not None:
    #             if isinstance(
    #                 self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
    #             ):
    #                 # ReduceLROnPlateau needs the loss value
    #                 self.scheduler.step(self.loss)
    #             else:
    #                 # Other schedulers just need to be stepped
    #                 self.scheduler.step()

    #         # Log current learning rate
    #         current_lr = self.optimizer.param_groups[0]["lr"]
    #         print(f"Current learning rate: {current_lr}")
    #         print(f"Loss at step {i}: {self.loss.detach().item()}")

    def _solve_scipy(
        self,
        method: tuple = None,
        tol: float = None,
        scipy_constraints: list = None,
        **options,
    ):
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
                - "constraint_penalty": Penalty weight for soft constraint violations.
                  Higher values enforce constraints more strictly at the cost of slower
                  convergence. Increase when constraints are violated in the solution.
                  Defaults to 100.
                - "verbose": Verbosity level (0-3) for optimization output
                - "maxiter": Maximum number of iterations
                - "gtol": Gradient tolerance for convergence
                - "xtol": Parameter tolerance for convergence
                - "barrier_tol": Barrier tolerance for interior point methods
                - "initial_tr_radius": Initial trust region radius
                - "initial_constr_penalty": Initial constraint penalty
                - Additional method-specific options as supported by SciPy optimizers
            scipy_constraints: Optional SciPy constraint dictionaries or
                constraint objects passed to ``scipy.optimize.minimize`` as
                hard constraints.

        Note:
            This method automatically handles the conversion between PyTorch tensors and
            NumPy arrays required by SciPy. It uses caching to avoid redundant computations
            when the same parameters are evaluated multiple times. The method supports
            both equality and inequality constraints through the loss function formulation.
        """
        if method is None:
            method = ("scipy", "SLSQP", "ad")

        LOGGER.task("Starting scipy solver: %s (%s mode)", method[1], method[2])
        LOGGER.add_level()

        x0, bounds_obj = self._prepare_scipy_problem(method, options)

        # Run optimization based on method
        optimizer_name = method[1]
        mode = method[2]

        LOGGER.config("Decision vector size: %d", len(x0))
        LOGGER.task("Starting optimization")

        if mode == "ad":
            # Use automatic differentiation
            if optimizer_name in ["trf", "dogbox"]:
                # These are least-squares optimizers
                result = least_squares(
                    self._obj_ad,
                    x0,
                    jac=self._jac_ad,
                    bounds=bounds_obj,
                    method=optimizer_name,
                    **options,
                )
            else:
                # These are general optimization algorithms
                if (
                    optimizer_name == "SLSQP"
                    and self._functional_objective is not None
                    and self.simulator.execution_backend == "cuda_graph"
                ):
                    result = self._solve_scipy_captured_slsqp(
                        x0,
                        bounds_obj,
                        tol,
                        scipy_constraints,
                        options,
                    )
                else:
                    result = minimize(
                        self._obj_ad,
                        x0,
                        method=optimizer_name,
                        jac=self._jac_ad,
                        bounds=bounds_obj,
                        tol=tol,
                        constraints=scipy_constraints if scipy_constraints else (),
                        options=options,
                    )
        else:
            LOGGER.remove_level()
            LOGGER.error(
                "Starting scipy solver: %s (%s mode)",
                method[1],
                method[2],
                change_status=True,
                ignore_no_match=True,
            )
            raise NotImplementedError(
                "Finite difference mode is not yet implemented for the optimizer. Use automatic differentiation mode."
            )

        # Apply the solution to the model: one object evaluation at
        # result.x writes the optimal trajectories into the decision-variable
        # ports and re-runs the simulation, so component histories hold the
        # optimized signals (guaranteed -- SciPy's last internal evaluation is
        # not necessarily at the solution, and with the functional objective no
        # simulation ran during the solve at all).
        self.apply_solution(result.x)

        elapsed = time_module.time() - self._solver_start_time
        LOGGER.info(
            "Optimization finished in %.1fs (%d function evaluations)",
            elapsed,
            self._eval_count,
        )
        opt_success = getattr(result, "success", None)
        opt_message = getattr(result, "message", None)
        opt_nit = getattr(result, "nit", None)
        opt_fun = getattr(result, "fun", None)
        if opt_success is not None:
            if opt_success:
                LOGGER.ok(
                    "Solver result: success %s, iterations %s, final loss %s",
                    opt_success,
                    opt_nit,
                    opt_fun,
                )
            else:
                LOGGER.warning(
                    "Solver result: success %s, iterations %s, final loss %s.",
                    opt_success,
                    opt_nit,
                    opt_fun,
                )
        if opt_message:
            LOGGER.info("Solver message: %s", opt_message)

        LOGGER.remove_level()
        if opt_success is False:
            LOGGER.warning(
                "Starting scipy solver: %s (%s mode)",
                method[1],
                method[2],
                change_status=True,
                ignore_no_match=True,
            )
        else:
            LOGGER.ok(
                "Starting scipy solver: %s (%s mode)",
                method[1],
                method[2],
                change_status=True,
                ignore_no_match=True,
            )
        return result

    def _solve_scipy_captured_slsqp(
        self, x0, bounds_obj, tol, scipy_constraints, options
    ):
        """Serve SLSQP from one directly captured functional value/grad bundle."""
        evaluator = CapturedControlObjective(self._functional_objective)
        cache = {"key": None, "value": None, "gradient": None}

        def bundle(x):
            x_np = np.asarray(x, dtype=np.float64)
            key = x_np.tobytes()
            if key == cache["key"]:
                return cache["value"], cache["gradient"]
            z = torch.as_tensor(x_np, dtype=tps.float_dtype(), device=self._device)
            value_t, gradient_t = evaluator.value_and_grad(z)
            value = float(value_t.detach().cpu())
            gradient = gradient_t.detach().cpu().numpy().astype(np.float64)
            self.obj = value_t.detach().clone()
            self.jac = gradient_t.detach().clone()
            self._theta_obj = z.detach().clone()
            self._theta_jac = z.detach().clone()
            self._eval_count += 1
            LOGGER.iter(
                "Evaluation %d: loss %.6f (%.1fs)",
                self._eval_count,
                value,
                time_module.time() - self._solver_start_time,
            )
            cache.update(key=key, value=value, gradient=gradient)
            return value, gradient

        try:
            result = minimize(
                lambda x: bundle(x)[0],
                x0,
                method="SLSQP",
                jac=lambda x: bundle(x)[1],
                bounds=bounds_obj,
                tol=tol,
                constraints=scipy_constraints if scipy_constraints else (),
                options=options,
            )
            result.derivative_stats = {
                "scipy_value_gradient_bundle": dict(evaluator.stats),
                "scope": "functional objective and gradient; host SciPy callbacks",
            }
            return result
        finally:
            evaluator.close()

    def apply_solution(self, theta) -> None:
        """Write a solver decision vector into the model and re-simulate."""
        self.__obj_ad(
            torch.tensor(
                np.asarray(theta), dtype=tps.float_dtype(), device=self._device
            )
        )

    def _prepare_scipy_problem(self, method: tuple, options: dict):
        """Initialize and build the shared SciPy NLP state.

        This setup is shared by the ordinary solver and Pareto subproblems.
        Objective-related options are consumed in place.
        """
        self._eval_count = 0
        self._solver_start_time = time_module.time()
        self._constraint_penalty = options.pop("constraint_penalty", 100)

        if "fast" in options:
            raise TypeError(
                "Optimizer option 'fast' has been removed; construct "
                "Simulator(model, execution_mode='functional') instead."
            )

        for component in self.simulator.model.components.values():
            if isinstance(component, nn.Module):
                for parameter in component.parameters():
                    parameter.requires_grad_(False)

        for component, output_name, *bounds in self._variables:
            component.output[output_name].do_normalization = True

        LOGGER.task("Initializing model")
        self.simulator.model.initialize(
            start_time=self._start_time,
            end_time=self._end_time,
            step_size=self._stepSize,
        )

        bounds_list = []
        x0_tensors = []
        n_periods = len(self._start_time)
        for component, output_name, *bounds in self._variables:
            port = component.output[output_name]
            active_history = (
                port.normalized_history if port.do_normalization else port.history()
            )
            if not active_history.is_leaf:
                # A previous object AD evaluation may have installed a
                # differentiable trajectory in this leaf port. Start each NLP
                # from a detached trajectory so requires_grad can be enabled
                # again and repeated solves remain independent.
                port.initialize(
                    n_t=self._max_timesteps,
                    n_s=n_periods,
                    n_c=port.n_c,
                    values=port.history().detach(),
                    force=True,
                )
            port.set_requires_grad(True)
            history_tensor = (
                port.normalized_history.detach()
                if port.do_normalization
                else port.history().detach()
            )
            period_tensors = [
                history_tensor[: self._n_timesteps[i_s], i_s, :]
                for i_s in range(n_periods)
            ]
            flattened_history = torch.cat(period_tensors, dim=0)
            x0_tensors.append(flattened_history)

            for _ in range(flattened_history.numel()):
                if len(bounds) >= 2:
                    lower, upper = bounds[:2]
                    if port.do_normalization:
                        lower = port.normalize(torch.tensor(lower)).item()
                        upper = port.normalize(torch.tensor(upper)).item()
                    bounds_list.append((lower, upper))
                else:
                    bounds_list.append((None, None))

        if x0_tensors:
            x0 = (
                torch.stack(x0_tensors, dim=1)
                .flatten()
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
            )
        else:
            x0 = np.array([], dtype=np.float64)

        bounds_obj = None
        if bounds_list and all(
            lower is not None and upper is not None for lower, upper in bounds_list
        ):
            bounds_obj = Bounds(
                [lower for lower, _ in bounds_list],
                [upper for _, upper in bounds_list],
            )

        def _get_constraint_value(component, output_name, component_or_value):
            n_s = len(self._start_time)
            n_t = max(self._n_timesteps)
            port = component.output[output_name]
            if isinstance(port, tps.Scalar):
                desired_shape = (n_t, n_s, port.n_c)
            elif isinstance(port, tps.Vector):
                desired_shape = (n_t, n_s, port.n_c, port.n_v)
            else:
                raise ValueError(f"Invalid constraint value type: {type(port)}")

            if isinstance(component_or_value, (int, float)):
                return torch.full(
                    desired_shape,
                    component_or_value,
                    dtype=tps.float_dtype(),
                    device=self._device,
                )
            if isinstance(component_or_value, systems.ScheduleSystem):
                component_or_value.initialize(
                    start_time=self._start_time,
                    end_time=self._end_time,
                    step_size=self._stepSize,
                )
                return (
                    component_or_value.output["scheduleValue"]
                    .history()
                    .to(device=self._device, dtype=tps.float_dtype())
                )
            if isinstance(component_or_value, torch.Tensor):
                return component_or_value.to(
                    device=self._device, dtype=tps.float_dtype()
                )
            raise ValueError(
                f"Invalid constraint value type: {type(component_or_value)}"
            )

        self.equality_constraint_values = {
            (component, output_name): _get_constraint_value(
                component, output_name, desired_value
            )
            for component, output_name, desired_value in self._eq_cons
        }
        self.inequality_constraint_values = {
            (component, output_name, constraint_type): _get_constraint_value(
                component, output_name, desired_value
            )
            for component, output_name, constraint_type, desired_value in self._ineq_cons
        }

        theta0 = torch.tensor(x0, dtype=tps.float_dtype(), device=self._device)
        self._theta_jac = 1000000 * torch.ones_like(theta0)
        self._theta_hes = torch.nan * torch.ones_like(theta0)
        self._theta_obj = 1000000 * torch.ones_like(theta0)

        self._functional_objective = None
        removed = {"fast", "fast_validate"}.intersection(options)
        if removed:
            raise TypeError(
                f"Removed optimizer option(s): {', '.join(sorted(removed))}. "
                "Select functional execution on Simulator."
            )
        if self.simulator.execution_mode == "functional" and method[2] == "ad":
            self._setup_functional_objective(x0)

        return x0, bounds_obj

    def _setup_functional_objective(self, x0) -> None:
        """Build the functional single-shooting objective.

        On success sets ``self._functional_objective`` (consumed by
        :meth:`_obj_ad` /
        :meth:`_jac_ad`); on any structural incompatibility leaves it ``None``
        and the exact object objective is used. With ``validate=True``
        Structural incompatibilities leave the functional objective unset and
        use the object execution path.
        """
        t0 = time_module.time()
        try:
            functional = FunctionalControlObjective(self)
        except Exception as exc:
            LOGGER.config(
                "Functional objective unavailable (%s); using object objective",
                exc,
            )
            return

        self._functional_objective = functional
        LOGGER.config(
            "Functional objective enabled (built in %.1fs)", time_module.time() - t0
        )

    def _write_variables(self, theta: torch.Tensor) -> None:
        """Write an interleaved normalized decision vector to output ports."""
        n_actuators = len(self._variables)
        n_periods = len(self._start_time)
        total_actual_timesteps = int(len(theta) / n_actuators)
        theta_matrix = theta.reshape(total_actual_timesteps, n_actuators)

        for i, (component, output_name, *bounds) in enumerate(self._variables):
            actuator_values = theta_matrix[:, i]
            n_c = component.output[output_name].n_c
            reconstructed_tensor = torch.full(
                (self._max_timesteps, n_periods, n_c),
                0,
                dtype=tps.float_dtype(),
                device=self._device,
            )

            value_idx = 0
            for period_idx in range(n_periods):
                actual_timesteps = self._n_timesteps[period_idx]
                period_values = actuator_values[
                    value_idx : value_idx + actual_timesteps
                ]
                # Time-first: [t, s, c] where t=timestep, s=period, c=component
                reconstructed_tensor[:actual_timesteps, period_idx, 0] = period_values
                value_idx += actual_timesteps

            if component.output[output_name].do_normalization:
                values = component.output[output_name].denormalize(reconstructed_tensor)
            else:
                values = reconstructed_tensor

            component.output[output_name].initialize(
                n_t=self._max_timesteps,
                n_s=len(self._start_time),
                values=values,
                force=True,
            )

    def _graph_parts(self, theta: torch.Tensor) -> SimpleNamespace:
        """Evaluate and decompose the object-execution optimization loss."""
        self._write_variables(theta)
        self.simulator.simulate(
            start_time=self._start_time,
            end_time=self._end_time,
            step_size=self._stepSize,
            show_progress_bar=False,
            execution_mode="object",
            execution_backend="eager",
        )

        k = self._constraint_penalty
        mask = self._timestep_mask

        eq = []
        for component, output_name, desired_value in self._eq_cons:
            y = component.output[output_name].history()[mask]
            desired_tensor = self.equality_constraint_values[component, output_name][
                mask
            ]
            y_norm = component.output[output_name].normalize(y)
            desired_tensor_norm = component.output[output_name].normalize(
                desired_tensor
            )
            eq.append(k * torch.mean(torch.abs(y_norm - desired_tensor_norm)))

        ineq = None
        if self._ineq_cons:
            ineq_upper_term = torch.tensor(
                0.0, dtype=tps.float_dtype(), device=self._device
            )
            ineq_lower_term = torch.tensor(
                0.0, dtype=tps.float_dtype(), device=self._device
            )
            for (
                component,
                output_name,
                constraint_type,
                desired_value,
            ) in self._ineq_cons:
                y = component.output[output_name].history()[mask]
                desired_tensor = self.inequality_constraint_values[
                    (component, output_name, constraint_type)
                ][mask]
                y_norm = component.output[output_name].normalize(y)
                desired_tensor_norm = component.output[output_name].normalize(
                    desired_tensor
                )
                if constraint_type == "upper":
                    ineq_upper_term += torch.mean(
                        torch.relu(y_norm - desired_tensor_norm)
                    )
                else:
                    ineq_lower_term += torch.mean(
                        torch.relu(desired_tensor_norm - y_norm)
                    )
            ineq = k * (ineq_upper_term + ineq_lower_term)

        objs = []
        phys = []
        for component, output_name, objective_type in self._objectives:
            y = component.output[output_name].history()[mask]
            y_norm = component.output[output_name].normalize(y)
            mean_norm = torch.mean(y_norm)
            objs.append(mean_norm if objective_type == "min" else -mean_norm)
            phys.append(torch.mean(y))
        return SimpleNamespace(eq=eq, ineq=ineq, objs=objs, phys=phys)

    def __obj_ad(self, theta: torch.Tensor) -> torch.Tensor:
        """Evaluate the scalar object-execution objective."""
        parts = self._graph_parts(theta)
        loss = torch.tensor(0.0, dtype=tps.float_dtype(), device=self._device)
        for term in parts.eq:
            loss = loss + term
        if parts.ineq is not None:
            loss = loss + parts.ineq
        for objective in parts.objs:
            loss = loss + objective
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
        theta = torch.tensor(theta, dtype=tps.float_dtype(), device=self._device)
        if torch.equal(theta, self._theta_obj):
            # scipy (SLSQP in particular) requires float64 regardless of the
            # model dtype, so every solver-facing exit casts explicitly.
            return self.obj.detach().cpu().numpy().astype(np.float64)
        else:
            self._theta_obj = theta
            if self._functional_objective is not None:
                self.obj = self._functional_objective.loss(theta)
            else:
                self.obj = self.__obj_ad(theta)
            self._eval_count += 1
            elapsed = time_module.time() - self._solver_start_time
            LOGGER.iter(
                "Evaluation %d: loss %.6f (%.1fs)",
                self._eval_count,
                self.obj.detach().item(),
                elapsed,
            )
            return self.obj.detach().cpu().numpy().astype(np.float64)

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
        theta = torch.tensor(theta, dtype=tps.float_dtype(), device=self._device)

        if torch.equal(theta, self._theta_jac):
            return self.jac.detach().cpu().numpy().astype(np.float64)
        else:
            self._theta_jac = theta
            if self._functional_objective is not None:
                # One autograd pass yields value AND gradient: cache both so a
                # subsequent f(theta) query is free.
                f, g = self._functional_objective.value_and_grad(theta)
                self.obj = f
                self._theta_obj = theta
                self.jac = g
            else:
                self.jac = self.__jac_ad(theta)
            jac_numpy = self.jac.detach().cpu().numpy().astype(np.float64)

            # Check for NaN values in Jacobian and warn
            if np.isnan(jac_numpy).any():
                n_nans = np.isnan(jac_numpy).sum()
                raise ValueError(
                    f"WARNING: Jacobian contains {n_nans} NaN values out of {jac_numpy.size} total values"
                )

            return jac_numpy

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
        theta = torch.tensor(theta, dtype=tps.float_dtype(), device=self._device)

        if torch.equal(theta, self._theta_hes):
            return self.hes.detach().cpu().numpy().astype(np.float64)
        else:
            self._theta_hes = theta
            self.hes = self.__hes_ad(theta)
            return self.hes.detach().cpu().numpy().astype(np.float64)


class OptimizationResult(ResultDict):
    """Dict-like result of :meth:`Optimizer.optimize`, parallel to EstimationResult.

    Wraps the SciPy ``OptimizeResult`` fields and keeps attribute access
    (``result.x``, ``result.success``, …) for compatibility with SciPy-style code.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @classmethod
    def from_scipy(cls, result) -> "OptimizationResult":
        """Build an OptimizationResult from a SciPy OptimizeResult-like object."""
        data = {}
        # OptimizeResult supports dict-like access
        try:
            data.update(dict(result))
        except Exception:
            for key in (
                "x",
                "success",
                "status",
                "message",
                "fun",
                "jac",
                "hess",
                "hess_inv",
                "nfev",
                "njev",
                "nhev",
                "nit",
                "maxcv",
            ):
                if hasattr(result, key):
                    data[key] = getattr(result, key)
        return cls(**data)

    __copy__ = ResultDict.copy
