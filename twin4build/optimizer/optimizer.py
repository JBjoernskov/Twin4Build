# import pygad
import twin4build.core as core
import torch
import datetime
from typing import Dict, List, Union, Tuple, Any
import torch.nn as nn
import twin4build.systems as systems


def _min_max_normalize(x, min_val=None, max_val=None):
    if min_val is None:
        min_val = torch.min(x)
    if max_val is None:
        max_val = torch.max(x)
    return (x - min_val) / (max_val - min_val)

class Optimizer():
    r"""
    An Optimizer class for optimizing building operation through setpoint adjustments.

    The optimizer uses gradient-based optimization to minimize a loss function that combines:
       1. Equality constraints (exact matches)
       2. Inequality constraints (upper/lower bounds)
       3. Minimization objectives

    Mathematical Formulation:

    1. Normalization of Values:

        .. math::

            x_{norm} = \frac{x - x_{min}}{x_{max} - x_{min}}

        where :math:`x_{norm}` is the normalized value, :math:`x` is the original value,
        :math:`x_{min}` is the minimum value (defaults to 0), and :math:`x_{max}` is the maximum value.

    2. Loss Function Components:

        a. Equality Constraints:

            .. math::

                L_{eq} = \sum_{i=1}^{n_{eq}} \frac{1}{n_i} \sum_{j=1}^{n_i} |y_{ij} - y_{ij}^{desired}|

        b. Inequality Constraints:

            .. math::

                L_{ineq} = \sum_{i=1}^{n_{ineq}} \frac{1}{n_i} \sum_{j=1}^{n_i} k \cdot \max(0, y_{ij} - y_{ij}^{upper}) + k \cdot \max(0, y_{ij}^{lower} - y_{ij})

        c. Minimization Objectives:

            .. math::

                L_{min} = \sum_{i=1}^{n_{min}} \frac{1}{n_i} \sum_{j=1}^{n_i} y_{ij}

    3. Total Loss:

        .. math::

            L_{total} = L_{eq} + L_{ineq} + L_{min}

    The optimizer uses gradient descent to minimize :math:`L_{total}` by adjusting the decision variables.

    Examples
    --------
    >>> import twin4build as tb
    >>> simulator = tb.Simulator(model)
    >>> optimizer = tb.Optimizer(simulator)
    >>> decisionVariables = [(component, 'output_name', 0.0, 1.0)]
    >>> minimize = [(component, 'output_name')]
    >>> import datetime, pytz
    >>> start = datetime.datetime(2024, 1, 1, tzinfo=pytz.UTC)
    >>> end = datetime.datetime(2024, 1, 2, tzinfo=pytz.UTC)
    >>> step = 3600
    >>> optimizer.optimize(decisionVariables=decisionVariables, minimize=minimize, startTime=start, endTime=end, stepSize=step)
    """
    def __init__(self, simulator: core.Simulator):
        self.simulator = simulator

    def _closure(self):
        self.optimizer.zero_grad()

        # Run simulation
        self.simulator.simulate(
            startTime=self.startTime,
            endTime=self.endTime,
            stepSize=self.stepSize,
            show_progress_bar=False
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

                min_val = 0  # torch.min(y)
                if (component, output_name) not in self.max_values:
                    max_val = torch.max(y.clone().detach())
                    self.max_values[(component, output_name)] = max_val
                max_val = self.max_values[(component, output_name)]
                y = _min_max_normalize(y, min_val, max_val)
                desired_tensor = _min_max_normalize(desired_tensor, min_val, max_val)
                
                eq_term += torch.mean(torch.abs(y - desired_tensor))
            self.loss += eq_term

        # Handle inequality constraints
        if self.inequalityConstraints is not None:
            ineq_term = 0
            for constraint in self.inequalityConstraints:
                component, output_name, constraint_type, desired_value = constraint
                y = component.output[output_name].history
                desired_tensor = self.inequality_constraint_values[(component, output_name, constraint_type)]

                min_val = 0  # torch.min(y)
                if (component, output_name) not in self.max_values:
                    max_val = torch.max(y.clone().detach())
                    self.max_values[(component, output_name)] = max_val
                max_val = self.max_values[(component, output_name)]
                
                # Normalize values
                y_norm = _min_max_normalize(y, min_val, max_val)
                desired_tensor_norm = _min_max_normalize(desired_tensor, min_val, max_val)
                
                if constraint_type == "upper":
                    # Penalize when y > desired_value
                    constraint_violations = torch.relu(y_norm - desired_tensor_norm)
                    constraint_term = torch.mean(k*constraint_violations)
                    ineq_term += constraint_term
                    
                elif constraint_type == "lower":
                    # Penalize when y < desired_value
                    constraint_violations = torch.relu(desired_tensor_norm - y_norm)
                    constraint_term = torch.mean(k*constraint_violations)
                    ineq_term += constraint_term

            self.loss += ineq_term

        # Handle minimization objectives
        if self.minimize is not None:
            min_term = 0
            for minimize_obj in self.minimize:
                component, output_name = minimize_obj
                y = component.output[output_name].history
                
                min_val = 0
                if (component, output_name) not in self.max_values:
                    max_val = torch.max(y.clone().detach())
                    self.max_values[(component, output_name)] = max_val
                max_val = self.max_values[(component, output_name)]
                
                y_norm = _min_max_normalize(y, min_val, max_val)
                
                min_term += torch.mean(y_norm)
            self.loss += min_term  # Minimize the mean value

        print("min_term: ", min_term, "percentage of loss: ", min_term/self.loss)
        print("eq_term: ", eq_term, "percentage of loss: ", eq_term/self.loss)
        print("ineq_term: ", ineq_term, "percentage of loss: ", ineq_term/self.loss)
        
        # Compute gradients
        self.loss.backward()
        return self.loss

    def optimize(self, 
                 decisionVariables: List[Tuple[Any, str, float, float]] = None,
                 minimize: List[Tuple[Any, str]] = None,
                 equalityConstraints: List[Tuple[Any, str, Any]] = None,
                 inequalityConstraints: List[Tuple[Any, str, str, Any]] = None,
                 startTime: Union[datetime.datetime, List[datetime.datetime]] = None,
                 endTime: Union[datetime.datetime, List[datetime.datetime]] = None,
                 stepSize: Union[float, List[float]] = None,
                 lr: float = 4.0,
                 iterations: int = 100,
                 scheduler_type: str = "step",
                 scheduler_params: Dict = None):
        """
        Optimize the model using gradient descent.
        
        Args:
            decisionVariables: List of tuples (component, output_name, lower_bound, upper_bound)
            minimize: List of tuples (component, output_name) to minimize
            equalityConstraints: List of tuples (component, output_name, desired_value)
            inequalityConstraints: List of tuples (component, output_name, constraint_type, desired_value)
                where constraint_type is "upper" or "lower"
            startTime: Start time for simulation
            endTime: End time for simulation
            stepSize: Step size for simulation
            lr: Learning rate for optimizer
            iterations: Number of optimization iterations to run
            scheduler_type: Type of learning rate scheduler
            scheduler_params: Parameters for learning rate scheduler
        """
        self.decisionVariables = decisionVariables or []
        self.minimize = minimize or []
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
        
        # Validate optimization parameters
        assert lr > 0, f"Learning rate must be positive, got {lr}"
        assert iterations > 0, f"Number of iterations must be positive, got {iterations}"
        
        # Check that we have something to optimize
        assert len(self.decisionVariables) > 0, "No decision variables specified for optimization"
        
        # Check that we have at least one objective (minimize or constraints)
        has_objective = len(self.minimize) > 0 or len(self.equalityConstraints) > 0 or len(self.inequalityConstraints) > 0
        assert has_objective, "No optimization objectives specified (minimize, equalityConstraints, or inequalityConstraints)"
        
        # Validate scheduler type
        valid_scheduler_types = ["step", "exponential", "cosine", "reduce_on_plateau", None]
        assert scheduler_type in valid_scheduler_types, f"Invalid scheduler_type: {scheduler_type}. Must be one of {valid_scheduler_types}"
        
        # Validate format of decision variables
        for i, decision_var in enumerate(self.decisionVariables):
            assert len(decision_var) >= 2, f"Decision variable at index {i} must have at least component and output_name"
            component, output_name, *bounds = decision_var
            assert hasattr(component, 'output'), f"Component {component} at index {i} does not have 'output' attribute"
            assert output_name in component.output, f"Output '{output_name}' not found in component {component.id}"
            if len(bounds) >= 2:
                lower, upper = bounds[0], bounds[1]
                assert upper > lower, f"Upper bound ({upper}) must be greater than lower bound ({lower}) for {component.id}.{output_name}"
        
        # Validate format of minimize objectives
        for i, min_obj in enumerate(self.minimize):
            assert len(min_obj) == 2, f"Minimize objective at index {i} must have component and output_name"
            component, output_name = min_obj
            assert hasattr(component, 'output'), f"Component {component} at index {i} does not have 'output' attribute"
            assert output_name in component.output, f"Output '{output_name}' not found in component {component.id}"
        
        # Validate format of equality constraints
        for i, eq_constraint in enumerate(self.equalityConstraints):
            assert len(eq_constraint) == 3, f"Equality constraint at index {i} must have component, output_name, and desired_value"
            component, output_name, desired_value = eq_constraint
            assert hasattr(component, 'output'), f"Component {component} at index {i} does not have 'output' attribute"
            assert output_name in component.output, f"Output '{output_name}' not found in component {component.id}"
        
        # Validate format of inequality constraints
        for i, ineq_constraint in enumerate(self.inequalityConstraints):
            assert len(ineq_constraint) == 4, f"Inequality constraint at index {i} must have component, output_name, constraint_type, and desired_value"
            component, output_name, constraint_type, desired_value = ineq_constraint
            assert hasattr(component, 'output'), f"Component {component} at index {i} does not have 'output' attribute"
            assert output_name in component.output, f"Output '{output_name}' not found in component {component.id}"
            assert constraint_type in ["upper", "lower"], f"Constraint type must be 'upper' or 'lower', got '{constraint_type}'"
        
        # Check for conflicting constraints: can't minimize and have equality constraint on same output
        if self.minimize and self.equalityConstraints:
            minimize_pairs = {(component, output_name) for component, output_name in self.minimize}
            equality_pairs = {(component, output_name) for component, output_name, _ in self.equalityConstraints}
            
            conflicting_pairs = minimize_pairs.intersection(equality_pairs)
            if conflicting_pairs:
                conflict_info = [f"({c.id}, {o})" for c, o in conflicting_pairs]
                raise ValueError(
                    f"Cannot simultaneously minimize and apply equality constraints to the same outputs: {', '.join(conflict_info)}. "
                    f"These objectives conflict with each other."
                )
                
        # Check for decision variables that are also in equality constraints
        if self.decisionVariables and self.equalityConstraints:
            decision_pairs = {(component, output_name) for component, output_name, *_ in self.decisionVariables}
            equality_pairs = {(component, output_name) for component, output_name, _ in self.equalityConstraints}
            
            conflicting_pairs = decision_pairs.intersection(equality_pairs)
            if conflicting_pairs:
                conflict_info = [f"({c.id}, {o})" for c, o in conflicting_pairs]
                raise ValueError(
                    f"Cannot optimize and apply equality constraints to the same outputs: {', '.join(conflict_info)}. "
                    f"These objectives conflict with each other."
                )

        print("Using device: ", "cuda" if torch.cuda.is_available() else "cpu")

        # Disable gradients for all parameters since we're optimizing inputs.
        # It is VERY important to do this before initializing the model.
        # Otherwise, the model parameters and state space matrices will have requires_grad=True
        # and the backpropagate() call will fail.
        for component in self.simulator.model.components.values():
            if isinstance(component, nn.Module):
                for parameter in component.parameters():
                    parameter.requires_grad_(False)


        # Set before initializing the model
        for component, output_name, *bounds in decisionVariables:
            component.output[output_name].normalize = True

        self.simulator.get_simulation_timesteps(startTime, endTime, stepSize)
        self.simulator.model.initialize(startTime=startTime, endTime=endTime, stepSize=stepSize, simulator=self.simulator)

        # Enable gradients only for the inputs we want to optimize
        opt_list = []
        for component, output_name, *bounds in decisionVariables:
            component.output[output_name].set_requires_grad(True)
            if component.output[output_name].normalize:
                opt_list.append(component.output[output_name].normalized_history)
            else:
                opt_list.append(component.output[output_name].history)

        # Initialize optimizer
        self.optimizer = torch.optim.SGD(opt_list, lr=lr)
        
        # Initialize scheduler
        if scheduler_params is None:
            scheduler_params = {}
            
        if scheduler_type == "step":
            # StepLR decreases learning rate by gamma every step_size epochs
            step_size = scheduler_params.get("step_size", 30)
            gamma = scheduler_params.get("gamma", 0.1)
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        elif scheduler_type == "exponential":
            # ExponentialLR decreases learning rate by gamma every epoch
            gamma = scheduler_params.get("gamma", 0.95)
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=gamma)
        elif scheduler_type == "cosine":
            # CosineAnnealingLR uses a cosine schedule to decrease learning rate
            T_max = scheduler_params.get("T_max", 100)
            eta_min = scheduler_params.get("eta_min", 0)
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=T_max, eta_min=eta_min)
        elif scheduler_type == "reduce_on_plateau":
            # ReduceLROnPlateau reduces learning rate when a metric has stopped improving
            mode = scheduler_params.get("mode", "min")
            factor = scheduler_params.get("factor", 0.1)
            patience = scheduler_params.get("patience", 10)
            threshold = scheduler_params.get("threshold", 1e-4)
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode=mode, factor=factor, 
                patience=patience, threshold=threshold
            )
        else:
            # Default: no scheduler
            self.scheduler = None

        def _get_constraint_value(component_or_value):
            """Helper function to get constraint value, handling both ScheduleSystem and scalar values"""
            if isinstance(component_or_value, (int, float)):
                return torch.tensor(component_or_value)
            elif isinstance(component_or_value, systems.ScheduleSystem):
                component_or_value.initialize(startTime=startTime, endTime=endTime, stepSize=stepSize, simulator=self.simulator)
                return component_or_value.output["scheduleValue"].history
            elif isinstance(component_or_value, torch.Tensor):
                return component_or_value
            else:
                raise ValueError(f"Invalid constraint value type: {type(component_or_value)}")
            
        # Pre-compute all constraint values
        self.equality_constraint_values = {}
        if self.equalityConstraints is not None:
            for component, output_name, desired_value in self.equalityConstraints:
                self.equality_constraint_values[component, output_name] = _get_constraint_value(desired_value)

        self.inequality_constraint_values = {}
        if self.inequalityConstraints is not None:
            for component, output_name, constraint_type, desired_value in self.inequalityConstraints:
                self.inequality_constraint_values[(component, output_name, constraint_type)] = _get_constraint_value(desired_value)
                
        for i in range(iterations):
            # Perform optimization step
            self.optimizer.step(self._closure)
            
            # Update learning rate with scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    # ReduceLROnPlateau needs the loss value
                    self.scheduler.step(self.loss)
                else:
                    # Other schedulers just need to be stepped
                    self.scheduler.step()
            
            # Log current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"Current learning rate: {current_lr}")

            # Apply bounds to decision variables
            with torch.no_grad():
                for component, output_name, *bounds in decisionVariables:
                    print("--------------------------------")
                    print("component: ", component.id)
                    print("mean history: ", torch.mean(torch.abs(component.output[output_name]._normalized_history)))
                    print("mean grad: ", torch.mean(torch.abs(component.output[output_name]._normalized_history.grad)))
                    print("std grad: ", torch.std(torch.abs(component.output[output_name]._normalized_history.grad)))
                    if len(bounds) > 0:
                        lower_bound = bounds[0] if len(bounds) > 0 else float('-inf')
                        upper_bound = bounds[1] if len(bounds) > 1 else float('inf')
                        if component.output[output_name].normalize:
                            lower_bound = _min_max_normalize(lower_bound, component.output[output_name]._min_history, component.output[output_name]._max_history)
                            upper_bound = _min_max_normalize(upper_bound, component.output[output_name]._min_history, component.output[output_name]._max_history)
                            component.output[output_name].normalized_history.clamp_(min=lower_bound, max=upper_bound)
                        else:
                            component.output[output_name].history.clamp_(min=lower_bound, max=upper_bound)
            
            print(f"Loss at step {i}: {self.loss.detach().item()}")
            