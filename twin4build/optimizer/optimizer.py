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
from twin4build.utils.deprecation import deprecate_args
from twin4build.utils.print_progress import LOGGER
from twin4build.utils.validate_period import validate_period


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

    For composable torch models the loss is evaluated (by default) with a fast
    composed objective: the components' ``forward`` methods are composed into a
    pure one-step map, the exogenous inputs are captured once from a reference
    simulation, and each evaluation is a plain sequential torch rollout with
    the gradient obtained in a single autograd pass -- identical values and
    gradients by construction, several times faster than simulating the full
    object graph per evaluation. Models the composed map cannot express fall
    back to the object-graph objective automatically (see the ``fast`` option
    of :meth:`optimize`).

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
        self._fast_obj = None

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
        step_size: Union[float, List[float]] = None,
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
                - "trf": Trust Region Reflective (for least-squares problems)
                - "dogbox": Dogleg algorithm (for least-squares problems)

                Examples: ``("scipy", "SLSQP", "ad")`` is preferred for most
                constrained optimization problems; ``("scipy", "trf", "fd")``
                for non-PyTorch models with a least-squares formulation.

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
                - "fast" (bool, default True): Evaluate the loss with the
                  composed one-step-map rollout (exogenous inputs captured
                  once, decision variables threaded through a pure torch
                  rollout, gradient via a single autograd pass) instead of a
                  full object-graph simulation per evaluation. Values and
                  gradients are identical by construction; the optimizer
                  silently falls back to the object-graph objective when the
                  model is not composable (components without ``forward``,
                  no stateful components, or a loss output the composed map
                  cannot produce).
                - "fast_validate" (bool, default False): Additionally
                  cross-check the fast loss and gradient against the
                  object-graph objective at the initial iterate (debugging
                  aid; costs ~2 object-graph evaluations).
                - Additional method-specific options as supported by SciPy optimizers

        Returns:
            The SciPy optimization result object. The optimized actuator
            trajectories are also applied to the model, so a subsequent
            ``simulator.simulate(...)`` runs with the optimal inputs.
        """

        deprecated_args = [
            "startTime",
            "endTime",
            "stepSize",
        ]
        new_args = ["start_time", "end_time", "step_size"]
        position = [1, 2, 3, 4, 5]
        value_map = deprecate_args(deprecated_args, new_args, position, kwargs)
        start_time = value_map.get("start_time", start_time)
        end_time = value_map.get("end_time", end_time)
        step_size = value_map.get("step_size", step_size)

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
                (component, output_name) for component, output_name in self._objectives
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
            bounds_str = f" (lb={bounds[0]}, ub={bounds[1]})" if len(bounds) >= 2 else ""
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
        for i, (s, e, ss) in enumerate(zip(self._start_time, self._end_time, self._stepSize)):
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
            result = self._scipy_solver(method=method, **options)
        else:
            LOGGER.remove_level()
            LOGGER.error("Running optimization", change_status=True)
            raise ValueError("Unsupported optimization method: %s" % method[0])

        LOGGER.remove_level()
        LOGGER.ok("Running optimization", change_status=True)
        return result

    def pareto_front(
        self,
        start_time: Union[datetime.datetime, List[datetime.datetime]] = None,
        end_time: Union[datetime.datetime, List[datetime.datetime]] = None,
        step_size: Union[float, List[float]] = None,
        variables: List[Tuple[Any, str, float, float]] = None,
        objective1: Tuple[Any, str, str] = None,
        objective2: Tuple[Any, str, str] = None,
        eq_cons: List[Tuple[Any, str, Any]] = None,
        ineq_cons: List[Tuple[Any, str, str, Any]] = None,
        n_points: int = 11,
        delta: float = 1e-3,
        method: Tuple[str, str, str] = ("scipy", "SLSQP", "ad"),
        batched_prepass: bool = True,
        prepass_options: Dict = None,
        options: Dict = None,
    ):
        """Trace the bi-objective Pareto front with the augmented
        epsilon-constraint method (AUGMECON).

        The front between ``objective1`` (kept as the scalar objective) and
        ``objective2`` (demoted to a hard constraint ``f2 <= eps``) is traced
        by sweeping ``eps`` between the two single-objective anchor solutions.
        Each subproblem is the regular optimization NLP plus ONE scalar
        inequality, solved exactly with gradients (SLSQP + AD) -- no
        evolutionary algorithm involved.  A small ``delta * f2`` objective
        term guarantees *properly* Pareto-optimal points, and, unlike a
        weighted sum, the sweep recovers non-convex front regions.

        With ``batched_prepass=True`` (default) and a composable model, all
        epsilon-subproblems are first solved approximately as ONE batched
        torch loss (independent copies, one backward per iteration --
        the batched workload shape where a GPU pays off), and the exact
        sequential solves merely polish the batched solutions.

        Args:
            start_time / end_time / step_size: Simulation period(s), exactly
                as in :meth:`optimize`.
            variables: Decision variables (component, output, lb, ub), as in
                :meth:`optimize`.
            objective1: ``(component, output_name, "min"|"max")`` -- the
                retained scalar objective (e.g. energy, "min").
            objective2: ``(component, output_name, "min"|"max")`` -- the
                objective swept via the epsilon constraint (e.g. comfort).
            eq_cons / ineq_cons: Additional constraints, handled as soft
                penalties exactly as in :meth:`optimize` (present in every
                subproblem including the anchors).
            n_points: Total number of front points INCLUDING the two anchor
                solutions.
            delta: AUGMECON augmentation coefficient on the normalized second
                objective (default 1e-3).
            method: ``("scipy", "SLSQP", "ad")`` (default) or
                ``("scipy", "trust-constr", "ad")`` -- the epsilon constraint
                requires a constraint-capable optimizer with AD gradients.
            batched_prepass: Solve all epsilon-subproblems approximately as
                one batched torch loss before the exact sweep (requires the
                fast composed objective; silently skipped otherwise).
            prepass_options: Prepass tuning: ``mu`` (penalty weight, default
                ``constraint_penalty``), ``lr``, ``max_iter``, ``patience``,
                ``rel_tol``.
            options: Solver options as in :meth:`optimize` (``maxiter``,
                ``ftol``, ``constraint_penalty``, ``fast``, ...), applied to
                every subproblem.

        Returns:
            :class:`~twin4build.optimizer._pareto.ParetoResult` with the
            physical objective values, decision trajectories, dominance mask,
            and finite-difference front slope (``-d f1 / d eps``, the marginal
            price of the second objective) per point.  ``result.apply(i)``
            writes point ``i`` back into the model; ``result.plot()`` draws
            the front.
        """
        from twin4build.optimizer._pareto import pareto_front as _pareto_front

        assert method[0] == "scipy" and method[2] == "ad" and method[1] in (
            "SLSQP",
            "trust-constr",
        ), (
            "pareto_front requires a constraint-capable scipy optimizer with "
            'AD gradients: ("scipy", "SLSQP", "ad") or '
            f'("scipy", "trust-constr", "ad") - {method} was provided.'
        )
        for name, obj in (("objective1", objective1), ("objective2", objective2)):
            assert (
                obj is not None and len(obj) == 3
            ), f"{name} must be a (component, output_name, 'min'|'max') tuple"
            component, output_name, objective_type = obj
            assert hasattr(
                component, "output"
            ), f"{name}: component {component} does not have an 'output' attribute"
            assert (
                output_name in component.output
            ), f"{name}: output '{output_name}' not found in component {component.id}"
            assert objective_type in ("min", "max"), (
                f"{name}: objective type must be 'min' or 'max', got "
                f"'{objective_type}'"
            )
        assert n_points >= 2, "n_points must be at least 2 (the two anchors)"

        # Same task setup as optimize(): attributes, periods, timestep mask.
        self._variables = variables or []
        self._objectives = [tuple(objective1), tuple(objective2)]
        self._eq_cons = eq_cons or []
        self._ineq_cons = ineq_cons or []
        assert (
            len(self._variables) > 0
        ), "No decision variables specified for optimization"

        start_time, end_time, step_size = validate_period(
            start_time, end_time, step_size
        )
        self._start_time = start_time
        self._end_time = end_time
        self._stepSize = step_size
        self._max_values = {}

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
        self._timestep_mask = timestep_mask

        options = dict(options or {})
        prepass_options = dict(prepass_options or {})
        # Notebook/CI fast path (same convention as optimize()): cap the work
        # so example notebooks exercise the full API without long solves.
        if os.environ.get("TWIN4BUILD_TESTING", "").lower() in ("1", "true", "yes"):
            options["maxiter"] = 1
            n_points = min(n_points, 3)
            prepass_options.setdefault("max_iter", 3)

        LOGGER.task("Generating Pareto front")
        LOGGER.add_level()
        LOGGER.config("Objective 1 (kept): %s.%s (%s)", objective1[0].id, objective1[1], objective1[2])
        LOGGER.config("Objective 2 (eps-constrained): %s.%s (%s)", objective2[0].id, objective2[1], objective2[2])
        LOGGER.config("Front points: %d | delta=%g | prepass=%s", n_points, delta, batched_prepass)
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

    def _scipy_solver(
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
            scipy_constraints: Optional list of SciPy constraint dicts (or
                ``LinearConstraint``/``NonlinearConstraint`` objects) passed
                through to ``scipy.optimize.minimize`` as HARD constraints.
                Only meaningful for constraint-capable optimizers (SLSQP,
                trust-constr); used by the Pareto epsilon-constraint sweep.

        Returns:
            The SciPy optimization result object.

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

        # Apply the solution to the model: one object-graph evaluation at
        # result.x writes the optimal trajectories into the decision-variable
        # ports and re-runs the simulation, so component histories hold the
        # optimized signals (guaranteed -- SciPy's last internal evaluation is
        # not necessarily at the solution, and with the fast objective no
        # simulation ran during the solve at all).
        self.apply_solution(result.x)

        elapsed = time_module.time() - self._solver_start_time
        LOGGER.info("Optimization finished in %.1fs (%d function evaluations)", elapsed, self._eval_count)
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

    def apply_solution(self, theta) -> None:
        """Write a solver decision vector into the model and re-simulate.

        One object-graph evaluation at ``theta`` writes the trajectories into
        the decision-variable ports and re-runs the simulation, so component
        histories hold the corresponding signals afterwards.
        """
        self.__obj_ad(
            torch.tensor(
                np.asarray(theta), dtype=tps.float_dtype(), device=self._device
            )
        )

    def _prepare_scipy_problem(self, method: tuple, options: dict):
        """Shared NLP setup for :meth:`_scipy_solver` and the Pareto sweep
        (:mod:`twin4build.optimizer._pareto`).

        Initializes the model at the current port trajectories, extracts the
        flattened/interleaved/normalized decision vector ``x0`` and its
        bounds, precomputes constraint target tensors, resets the evaluation
        caches and builds the fast composed objective (``options["fast"]``,
        default on).  Consumes the objective-related keys of ``options``
        in place (the remaining keys go to the SciPy solver).

        Returns:
            ``(x0, bounds_obj)``: the initial decision vector (float64 numpy)
            and the SciPy ``Bounds`` object (or ``None``).
        """
        self._eval_count = 0
        self._solver_start_time = time_module.time()
        self._constraint_penalty = options.pop("constraint_penalty", 100)

        for component in self.simulator.model.components.values():
            if isinstance(component, nn.Module):
                for parameter in component.parameters():
                    parameter.requires_grad_(False)

        # Set before initializing the model
        for component, output_name, *bounds in self._variables:
            component.output[output_name].do_normalization = True

        LOGGER.task("Initializing model")
        self.simulator.model.initialize(
            start_time=self._start_time,
            end_time=self._end_time,
            step_size=self._stepSize,
        )

        # Create initial guess vector
        x0 = []
        bounds_list = []

        n_periods = len(self._start_time)

        # Create flattened vector using vectorized operations, excluding padded values
        x0_tensors = []

        for component, output_name, *bounds in self._variables:
            component.output[output_name].set_requires_grad(True)

            # Get the full history tensor for this component
            if component.output[output_name].do_normalization:
                history_tensor = component.output[
                    output_name
                ].normalized_history.detach()
            else:
                history_tensor = component.output[output_name].history().detach()

            # Extract only actual timesteps (no padding) for each period
            # History shape is (n_t, n_s, n_c) - time-first layout
            period_tensors = []
            for period_idx in range(n_periods):
                actual_timesteps = self._n_timesteps[period_idx]
                # Slice time dimension, index scenario dimension, keep all components
                period_data = history_tensor[:actual_timesteps, period_idx, :]
                period_tensors.append(period_data)

            # Concatenate all periods for this variable
            flattened_history = torch.cat(period_tensors, dim=0)
            x0_tensors.append(flattened_history)

            # Set bounds for actual timesteps only
            total_actual_elements = flattened_history.numel()
            for _ in range(total_actual_elements):
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

        # Interleave the tensors: [var1_t0, var2_t0, var1_t1, var2_t1, ...]
        # This matches the expected structure for the theta vector
        if x0_tensors:
            # Stack tensors and transpose to interleave
            stacked = torch.stack(
                x0_tensors, dim=1
            )  # Shape: (total_actual_timesteps, n_variables)
            # scipy requires float64 regardless of the model dtype.
            x0 = (
                stacked.flatten().detach().cpu().numpy().astype(np.float64)
            )  # Flatten to get interleaved structure
        else:
            x0 = np.array([])

        # Create bounds object for SciPy
        if all(b[0] is not None and b[1] is not None for b in bounds_list):
            bounds_obj = Bounds(
                [b[0] for b in bounds_list], [b[1] for b in bounds_list]
            )
        else:
            bounds_obj = None

        # Pre-compute constraint values
        def _get_constraint_value(component, output_name, component_or_value):
            """Helper function to get constraint value, handling both ScheduleSystem and scalar values"""
            n_s = len(self._start_time)
            n_t = max(self._n_timesteps)
            n_c = component.output[output_name].n_c

            if isinstance(component.output[output_name], tps.Scalar):
                # Shape: (n_t, n_s, n_c) - time-first layout
                desired_shape = (n_t, n_s, n_c)
            elif isinstance(component.output[output_name], tps.Vector):
                # Shape: (n_t, n_s, n_c, n_v) - time-first layout
                desired_shape = (
                    n_t,
                    n_s,
                    n_c,
                    component.output[output_name].n_v,
                )
            else:
                raise ValueError(
                    f"Invalid constraint value type: {type(component.output[output_name])}"
                )

            if isinstance(component_or_value, (int, float)):
                return torch.full(
                    desired_shape,
                    component_or_value,
                    dtype=tps.float_dtype(),
                    device=self._device,
                )
            elif isinstance(component_or_value, systems.ScheduleSystem):
                # The schedule may be standalone (not part of the model), in
                # which case Model.to() never moved it -- align explicitly.
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
            elif isinstance(component_or_value, torch.Tensor):
                return component_or_value.to(
                    device=self._device, dtype=tps.float_dtype()
                )
            else:
                raise ValueError(
                    f"Invalid constraint value type: {type(component_or_value)}"
                )

        self.equality_constraint_values = {}
        if self._eq_cons is not None:
            for component, output_name, desired_value in self._eq_cons:
                self.equality_constraint_values[component, output_name] = (
                    _get_constraint_value(component, output_name, desired_value)
                )

        self.inequality_constraint_values = {}
        if self._ineq_cons is not None:
            for (
                component,
                output_name,
                constraint_type,
                desired_value,
            ) in self._ineq_cons:
                constraint_val = _get_constraint_value(
                    component, output_name, desired_value
                )
                self.inequality_constraint_values[
                    (component, output_name, constraint_type)
                ] = constraint_val

        # Initialize caching variables for AD
        self._theta_jac = 1000000 * torch.ones_like(
            torch.tensor(x0, dtype=tps.float_dtype(), device=self._device)  # torch.nan
        )
        self._theta_hes = torch.nan * torch.ones_like(
            torch.tensor(x0, dtype=tps.float_dtype(), device=self._device)
        )
        self._theta_obj = 1000000 * torch.ones_like(
            torch.tensor(x0, dtype=tps.float_dtype(), device=self._device)
        )

        # -- Fast composed objective (default ON) ------------------------------
        # Replaces the object-graph simulate-per-evaluation with a sequential
        # rollout of the composed pure one-step map (see _fast_objective.py):
        # exogenous inputs captured once, decision-variable slots driven by
        # theta, gradient via one autograd pass instead of jacrev around the
        # simulation.  Structural compatibility is checked at build time --
        # non-composable models silently fall back to the object-graph
        # objective.  Numerical equivalence holds by construction (each
        # composable component's do_step delegates to the same forward the
        # composer threads) and is regression-checked by
        # tests/optimizer/test_fast_objective.py.
        # options={"fast": False} opts out; options={"fast_validate": True}
        # additionally cross-checks value and gradient against the object-graph
        # objective at the initial iterate (debugging aid, costs ~2 evals).
        fast_requested = bool(options.pop("fast", True))
        fast_validate = bool(options.pop("fast_validate", False))
        self._fast_obj = None
        if fast_requested and method[2] == "ad":
            self._setup_fast_objective(x0, validate=fast_validate)

        return x0, bounds_obj

    def _setup_fast_objective(self, x0, validate: bool = False) -> None:
        """Build the composed-map objective (see ``_fast_objective.py``).

        On success sets ``self._fast_obj`` (consumed by :meth:`_obj_ad` /
        :meth:`_jac_ad`); on any structural incompatibility leaves it ``None``
        and the exact object-graph objective is used.  With ``validate=True``
        (``options={"fast_validate": True}``) the loss value AND gradient are
        additionally cross-checked against the object-graph objective at the
        initial iterate before the fast path is enabled -- a runtime debugging
        aid costing ~2 object-graph evaluations.
        """
        t0 = time_module.time()
        try:
            from twin4build.optimizer._fast_objective import FastControlObjective

            fast = FastControlObjective(self)
        except Exception as exc:
            LOGGER.config(
                "Fast objective unavailable (%s); using object-graph objective",
                exc,
            )
            return

        if validate:
            theta0 = torch.tensor(x0, dtype=tps.float_dtype(), device=self._device)
            f_fast, g_fast = fast.value_and_grad(theta0)
            f_slow = self.__obj_ad(theta0.clone())
            g_slow = torch.func.jacrev(self.__obj_ad, argnums=0)(theta0.clone())
            rel_f = float(
                abs(f_fast - f_slow) / max(1e-12, abs(float(f_slow)))
            )
            gscale = max(1e-12, float(g_slow.abs().max()))
            rel_g = float((g_fast - g_slow).abs().max()) / gscale
            # fp32 accumulates roundoff over the rollout; loosen the parity
            # thresholds accordingly (they only gate the fast-path opt-in).
            if tps.float_dtype() == torch.float64:
                tol_f, tol_g = 1e-6, 1e-4
            else:
                tol_f, tol_g = 1e-3, 1e-2
            if rel_f > tol_f or rel_g > tol_g:
                LOGGER.warning(
                    "Fast objective validation FAILED (value rel=%.3e, "
                    "gradient rel=%.3e); using object-graph objective",
                    rel_f,
                    rel_g,
                )
                return
            LOGGER.config(
                "Fast objective validated (value rel=%.3e, gradient rel=%.3e)",
                rel_f,
                rel_g,
            )

        self._fast_obj = fast
        LOGGER.config(
            "Fast objective enabled (built in %.1fs)", time_module.time() - t0
        )

    def _write_variables(self, theta: torch.Tensor) -> None:
        """Write a (normalized, interleaved) decision vector into the
        decision-variable ports, ready for a simulation."""
        # Reshape theta using vectorized operations
        n_actuators = len(self._variables)
        n_periods = len(self._start_time)

        # Reshape theta from interleaved format [var1_t0, var2_t0, var1_t1, var2_t1, ...]
        # to (total_actual_timesteps, n_variables) format
        total_actual_timesteps = int(len(theta) / n_actuators)
        theta_matrix = theta.reshape(total_actual_timesteps, n_actuators)

        # Update decision variables for each actuator
        for i, (component, output_name, *bounds) in enumerate(self._variables):
            # Extract values for this actuator across all actual timesteps
            actuator_values = theta_matrix[:, i]

            # Construct values tensor in time-first format: (n_t, n_s, n_c)
            # where n_t = max_timesteps, n_s = n_periods, n_c = 1
            n_c = component.output[output_name].n_c
            reconstructed_tensor = torch.full(
                (self._max_timesteps, n_periods, n_c),
                0,
                dtype=tps.float_dtype(),
                device=self._device,
            )  # FIX OF NAN JACOBIAN: 0 instead float('nan')

            # Fill in the actual values period by period
            value_idx = 0
            for period_idx in range(n_periods):
                actual_timesteps = self._n_timesteps[period_idx]
                period_values = actuator_values[
                    value_idx : value_idx + actual_timesteps
                ]
                # Time-first: [t, s, c] where t=timestep, s=period, c=component
                reconstructed_tensor[:actual_timesteps, period_idx, 0] = period_values
                value_idx += actual_timesteps

            # Denormalize if needed
            if component.output[output_name].do_normalization:
                values = component.output[output_name].denormalize(reconstructed_tensor)
            else:
                values = reconstructed_tensor

            # Initialize with the new values (time-first format)
            component.output[output_name].initialize(
                n_timesteps=self._max_timesteps,
                batch_size=len(self._start_time),
                values=values,
                force=True,
            )

    def _graph_parts(self, theta: torch.Tensor) -> SimpleNamespace:
        """Object-graph counterpart of ``FastControlObjective.parts``: write
        ``theta`` into the model, simulate, and return the loss decomposed
        into per-term components (same namespace layout -- ``eq`` list,
        ``ineq`` scalar or ``None``, min-oriented normalized ``objs``,
        physical ``phys`` means)."""
        self._write_variables(theta)

        # Run simulation
        self.simulator.simulate(
            start_time=self._start_time,
            end_time=self._end_time,
            step_size=self._stepSize,
            show_progress_bar=False,
        )

        k = self._constraint_penalty

        # Use boolean mask (n_t, n_s) to index 3D tensors (n_t, n_s, n_c) -> (num_valid, n_c)
        mask = self._timestep_mask

        eq = []
        if self._eq_cons is not None:
            for constraint in self._eq_cons:
                component, output_name, desired_value = constraint
                # History has shape (n_t, n_s, n_c) - index with mask to get valid entries
                y = component.output[output_name].history()[mask]
                desired_tensor = self.equality_constraint_values[
                    component, output_name
                ][mask]
                y_norm = component.output[output_name].normalize(y)
                desired_tensor_norm = component.output[output_name].normalize(
                    desired_tensor
                )
                eq.append(k * torch.mean(torch.abs(y_norm - desired_tensor_norm)))

        ineq = None
        if self._ineq_cons is not None:
            ineq_upper_term = torch.tensor(0.0, dtype=tps.float_dtype(), device=self._device)
            ineq_lower_term = torch.tensor(0.0, dtype=tps.float_dtype(), device=self._device)
            for constraint in self._ineq_cons:
                component, output_name, constraint_type, desired_value = constraint
                # History has shape (n_t, n_s, n_c) - index with mask to get valid entries
                y = component.output[output_name].history()[mask]
                desired_tensor = self.inequality_constraint_values[
                    (component, output_name, constraint_type)
                ][mask]
                y_norm = component.output[output_name].normalize(y)
                desired_tensor_norm = component.output[output_name].normalize(
                    desired_tensor
                )

                if constraint_type == "upper":
                    # Penalize when y > desired_value
                    constraint_violations = torch.relu(y_norm - desired_tensor_norm)
                    ineq_upper_term += torch.mean(constraint_violations)
                elif constraint_type == "lower":
                    # Penalize when y < desired_value
                    constraint_violations = torch.relu(desired_tensor_norm - y_norm)
                    ineq_lower_term += torch.mean(constraint_violations)

            ineq = k * (ineq_upper_term + ineq_lower_term)

        objs = []
        phys = []
        if self._objectives is not None:
            for component, output_name, objective_type in self._objectives:
                # History has shape (n_t, n_s, n_c) - index with mask to get valid entries
                y = component.output[output_name].history()[mask]
                y_norm = component.output[output_name].normalize(y)
                m = torch.mean(y_norm)
                objs.append(m if objective_type == "min" else -m)
                phys.append(torch.mean(y))

        return SimpleNamespace(eq=eq, ineq=ineq, objs=objs, phys=phys)

    def __obj_ad(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Objective function for automatic differentiation.

        Args:
            theta (torch.Tensor): Flattened parameter vector containing values for all periods,
                                 timesteps, and actuators.

        Returns:
            torch.Tensor: Objective value.
        """
        p = self._graph_parts(theta)

        # Compute loss - initialize as tensor to avoid NaN propagation issues
        # (same accumulation order as the original fused implementation).
        loss = torch.tensor(0.0, dtype=tps.float_dtype(), device=self._device)
        for e in p.eq:
            loss += e
        if p.ineq is not None:
            loss += p.ineq
        for o in p.objs:
            loss += o

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
            if self._fast_obj is not None:
                self.obj = self._fast_obj.loss(theta)
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
            if self._fast_obj is not None:
                # One autograd pass yields value AND gradient: cache both so a
                # subsequent f(theta) query is free.
                f, g = self._fast_obj.value_and_grad(theta)
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
