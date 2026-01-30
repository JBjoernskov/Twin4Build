# Standard library imports
import datetime
from typing import Dict, List, Optional, Tuple, Type, Union

# Third party imports
import torch
import torch.nn as nn

# Local application imports
import twin4build.core as core
import twin4build.utils.types as tps
from twin4build.systems.controller.setpoint_controller.pid_controller.pid_controller_system import (
    PIDControllerSystem,
)


class ControllerIdentificationTorchSystem(core.System, nn.Module):
    r"""
    Controller Identification System using Continuous Relaxation.

    This class implements a differentiable controller identification framework
    that can estimate control logic from observed actuator data. It composes
    candidate controller Systems and uses continuous relaxation to identify:

    1. Which sensor(s) provide feedback (β weights)
    2. Which setpoint signal(s) are being tracked (γ weights)
    3. Which candidate controller(s) are active (α weights)
    4. The parameters of each candidate controller

    The system supports multiple actuators, each with its own set of candidate
    controllers. This is designed to work with the twin4build Estimator class.

    Mathematical Formulation
    ------------------------

    **Control Error (per actuator):**

    .. math::

        e_t = \sum_j \gamma_j \cdot sp_{jt} - \sum_i \beta_i \cdot y_{it}

    **Predicted Actuator Output (per actuator a):**

    .. math::

        \hat{u}_{a,t} = \sum_k \alpha_{a,k} \cdot C_k(e_t; \theta_k)

    where :math:`C_k` is the k-th candidate controller.

    **Binarization Penalty:**

    .. math::

        P(x) = x(1 - x)

    Args:
        n_sensors: Number of candidate feedback sensors (default: 1)
        n_setpoints: Number of candidate setpoint signals (default: 1)
        n_actuators: Number of actuator outputs (default: 1)
        candidate_controllers: List of controller System classes to use as candidates.
            If None, uses default PIDControllerSystem with different configurations.
        candidate_controller_kwargs: List of kwargs dicts for each candidate controller.
            If None, uses default configurations.
        isReverse: If True, controller output increases when error decreases (default: False)
        **kwargs: Additional keyword arguments passed to parent classes

    Example:
        >>> # Default usage with built-in candidates
        >>> controller = ControllerIdentificationTorchSystem(
        ...     n_sensors=1,
        ...     n_setpoints=1,
        ...     n_actuators=1,
        ...     id="my_controller"
        ... )
        >>>
        >>> # Custom candidates
        >>> controller = ControllerIdentificationTorchSystem(
        ...     candidate_controllers=[PIDControllerSystem, PIDControllerSystem],
        ...     candidate_controller_kwargs=[
        ...         {"kp": 0.1, "Ti": 100, "Td": 0},  # PI
        ...         {"kp": 0.1, "Ti": 100, "Td": 10}, # PID
        ...     ],
        ...     id="custom_controller"
        ... )
    """

    def __init__(
        self,
        n_sensors: int = 1,
        n_setpoints: int = 1,
        n_actuators: int = 1,
        candidate_controllers: Optional[List[Type[core.System]]] = None,
        candidate_controller_kwargs: Optional[List[dict]] = None,
        isReverse: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        nn.Module.__init__(self)

        self.n_sensors = n_sensors
        self.n_setpoints = n_setpoints
        self.n_actuators = n_actuators
        self.isReverse = isReverse

        # Setup candidate controllers
        if candidate_controllers is None:
            # Default: single PIDControllerSystem - parameters (kp, Ti, Td) will be estimated
            # The controller naturally represents P (Td=0, Ti→∞), PI (Td=0), or PID behavior
            candidate_controllers = [PIDControllerSystem]
            candidate_controller_kwargs = [
                {"kp": 0.3, "Ti": 5.0, "Td": 0.0, "isReverse": isReverse},
            ]

        self.candidate_controller_classes = candidate_controllers
        self.n_candidates = len(candidate_controllers)

        if candidate_controller_kwargs is None:
            candidate_controller_kwargs = [{} for _ in candidate_controllers]
        
        assert len(candidate_controller_kwargs) == self.n_candidates, \
            "candidate_controller_kwargs must match candidate_controllers length"

        # Store kwargs for later use
        self._candidate_controller_kwargs = candidate_controller_kwargs

        # Create candidate controller instances for each actuator
        # Set as attributes (e.g., self.candidate_0_0) for config system compatibility
        for a in range(n_actuators):
            for c, (CtrlClass, ctrl_kwargs) in enumerate(
                zip(candidate_controllers, candidate_controller_kwargs)
            ):
                ctrl_id = f"{kwargs.get('id', 'ctrl')}_a{a}_c{c}"
                controller = CtrlClass(id=ctrl_id, **ctrl_kwargs)
                setattr(self, f"candidate_{a}_{c}", controller)

        # Selection weights - all per-actuator:
        # - alpha_a: selects among candidates for actuator a (n_c = n_candidates)
        # - beta_a: selects feedback sensors for actuator a (n_c = n_sensors)
        # - gamma_a: selects setpoint signals for actuator a (n_c = n_setpoints)
        # All initialized uniformly
        alpha_init = 0.5
        beta_init = 0.5
        gamma_init = 0.5
        
        for a in range(n_actuators):
            # Alpha: candidate controller selection
            setattr(
                self,
                f"alpha_{a}",
                tps.Parameter(
                    torch.full((self.n_candidates,), alpha_init, dtype=torch.float64),
                    min_value=0.0,
                    max_value=1.0,
                    requires_grad=False,
                    n_c=self.n_candidates,
                ),
            )
            
            # Beta: feedback sensor selection (per-actuator)
            setattr(
                self,
                f"beta_{a}",
                tps.Parameter(
                    torch.full((n_sensors,), beta_init, dtype=torch.float64),
                    min_value=0.0,
                    max_value=1.0,
                    requires_grad=False,
                    n_c=n_sensors,
                ),
            )
            
            # Gamma: setpoint signal selection (per-actuator)
            setattr(
                self,
                f"gamma_{a}",
                tps.Parameter(
                    torch.full((n_setpoints,), gamma_init, dtype=torch.float64),
                    min_value=0.0,
                    max_value=1.0,
                    requires_grad=False,
                    n_c=n_setpoints,
                ),
            )

        # Build input dictionary
        self._input = {
            "sensorValue": tps.Vector(),  # Feedback sensor values (n_sensors)
            "setpointValue": tps.Vector(),  # Setpoint signals (n_setpoints)
        }

        # Output: one signal per actuator
        self._output = {"inputSignal": tps.Vector()}

        # Build config - include parameters from all candidate controllers
        config_params = [
            "n_sensors",
            "n_setpoints",
            "n_actuators",
            "isReverse",
        ]
        # Add candidate controller parameters with prefixes
        for a in range(n_actuators):
            for c in range(self.n_candidates):
                ctrl = getattr(self, f"candidate_{a}_{c}")
                if hasattr(ctrl, '_config') and 'parameters' in ctrl._config:
                    for param in ctrl._config['parameters']:
                        config_params.append(f"candidate_{a}_{c}.{param}")

        self._config = {"parameters": config_params}
        self.INITIALIZED = False

    @property
    def config(self):
        return self._config

    @property
    def input(self) -> dict:
        return self._input

    @property
    def output(self) -> dict:
        return self._output

    def _get_candidate(self, actuator: int, candidate: int) -> core.System:
        """Get candidate controller for actuator a, candidate c."""
        return getattr(self, f"candidate_{actuator}_{candidate}")

    def _get_alpha(self, actuator: int, candidate: int) -> torch.Tensor:
        """Get alpha parameter for candidate c of actuator a."""
        return getattr(self, f"alpha_{actuator}").get()[candidate]

    def _get_beta(self, actuator: int, sensor: int) -> torch.Tensor:
        """Get beta parameter for sensor i of actuator a."""
        return getattr(self, f"beta_{actuator}").get()[sensor]

    def _get_gamma(self, actuator: int, setpoint: int) -> torch.Tensor:
        """Get gamma parameter for setpoint j of actuator a."""
        return getattr(self, f"gamma_{actuator}").get()[setpoint]
    
    def _get_alpha_vector(self, actuator: int) -> torch.Tensor:
        """Get full alpha vector for actuator a."""
        return getattr(self, f"alpha_{actuator}").get()
    
    def _get_beta_vector(self, actuator: int) -> torch.Tensor:
        """Get full beta vector for actuator a."""
        return getattr(self, f"beta_{actuator}").get()
    
    def _get_gamma_vector(self, actuator: int) -> torch.Tensor:
        """Get full gamma vector for actuator a."""
        return getattr(self, f"gamma_{actuator}").get()

    def initialize(
        self,
        start_time: List[datetime.datetime],
        end_time: List[datetime.datetime],
        step_size: int,
    ) -> None:
        """Initialize the controller system and all candidate controllers."""
        _, _, max_timesteps, _ = core.Simulator.get_simulation_timesteps(
            start_time, end_time, step_size
        )
        batch_size = len(start_time)

        # Initialize Vector inputs with their sizes
        self.input["sensorValue"].initialize(
            n_timesteps=max_timesteps, batch_size=batch_size, size=self.n_sensors
        )
        self.input["setpointValue"].initialize(
            n_timesteps=max_timesteps, batch_size=batch_size, size=self.n_setpoints
        )

        # Initialize output
        self.output["inputSignal"].initialize(
            n_timesteps=max_timesteps, batch_size=batch_size, size=self.n_actuators
        )

        # Initialize all candidate controllers
        for a in range(self.n_actuators):
            for c in range(self.n_candidates):
                self._get_candidate(a, c).initialize(start_time, end_time, step_size)

        self.INITIALIZED = True

    def _compute_weighted_signals(self, actuator: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the weighted sensor and setpoint signals for a specific actuator.

        Uses vectorized operations with the per-actuator beta and gamma vectors.
        Weights are normalized by their sum so that selecting a single signal
        (e.g., gamma=[1,0,0,...]) gives 100% of that signal's value.

        Args:
            actuator: The actuator index to compute signals for.

        Returns:
            Tuple of (weighted_setpoint, weighted_feedback) tensors, each shape (n_s, n_c)
        """
        # Tensor shapes: (n_s, n_c, n_v) where n_v is n_sensors or n_setpoints
        sensor_values = self.input["sensorValue"].get()  # (n_s, n_c, n_sensors)
        setpoint_values = self.input["setpointValue"].get()  # (n_s, n_c, n_setpoints)
        
        # Get per-actuator weight vectors and normalize by sum
        gamma = self._get_gamma_vector(actuator)  # (n_setpoints,)
        gamma_sum = torch.sum(gamma) + 1e-8
        
        beta = self._get_beta_vector(actuator)  # (n_sensors,)
        beta_sum = torch.sum(beta) + 1e-8

        # Weighted setpoint: sum_j(gamma_j * sp_jt) / sum(gamma)
        # gamma broadcasts: (n_setpoints,) with (n_s, n_c, n_setpoints) -> (n_s, n_c, n_setpoints)
        # Sum over the last dimension (n_v = n_setpoints)
        weighted_setpoint = torch.sum(gamma * setpoint_values, dim=-1) / gamma_sum  # (n_s, n_c)

        # Weighted sensor feedback: sum_i(beta_i * y_it) / sum(beta)
        # beta broadcasts: (n_sensors,) with (n_s, n_c, n_sensors) -> (n_s, n_c, n_sensors)
        # Sum over the last dimension (n_v = n_sensors)
        weighted_feedback = torch.sum(beta * sensor_values, dim=-1) / beta_sum  # (n_s, n_c)
            
        return weighted_setpoint, weighted_feedback

    def do_step(
        self,
        second_time: float,
        date_time: datetime.datetime,
        step_size: int,
        step_index: int,
    ) -> None:
        """Perform one simulation step.

        For each actuator:
        1. Compute per-actuator weighted setpoint and feedback signals
        2. Run all candidate controllers with actuator-specific error signal
        3. Combine outputs using normalized alpha weights
        """
        # Get dimensions from input signals
        sensor_values = self.input["sensorValue"].get()
        if sensor_values.dim() <= 1:
            n_s, n_c = 1, 1
        elif sensor_values.dim() == 2:
            n_s = sensor_values.shape[0]
            n_c = 1
        else:
            n_s, n_c = sensor_values.shape[0], sensor_values.shape[1]

        # Process each actuator
        actuator_outputs = torch.zeros(n_s, n_c, self.n_actuators, dtype=torch.float64)

        for a in range(self.n_actuators):
            # Compute per-actuator weighted signals (each actuator has own beta/gamma)
            weighted_setpoint, weighted_feedback = self._compute_weighted_signals(a)
            
            # Run all candidate controllers and collect outputs
            candidate_outputs = []
            for c in range(self.n_candidates):
                ctrl = self._get_candidate(a, c)
                
                # Set inputs for candidate controller (same error signal for all candidates)
                ctrl.input["setpointValue"].set(weighted_setpoint, step_index)
                ctrl.input["actualValue"].set(weighted_feedback, step_index)
                
                # Run candidate controller step
                ctrl.do_step(second_time, date_time, step_size, step_index)
                
                # Collect candidate output - shape (n_s, n_c)
                candidate_outputs.append(ctrl.output["inputSignal"].get())
            
            # Stack outputs: (n_candidates, n_s, n_c)
            candidate_outputs = torch.stack(candidate_outputs, dim=0)
            
            # Flatten to (n_candidates, n_s * n_c) for einsum
            orig_shape = candidate_outputs.shape[1:]  # (n_s, n_c) or similar
            candidate_outputs_flat = candidate_outputs.reshape(self.n_candidates, -1)
            
            # Vectorized weighted sum using normalized alpha vector
            # Normalization ensures alpha=[1,0] gives 100% candidate 0, not 50%
            alpha = self._get_alpha_vector(a)
            alpha_sum = torch.sum(alpha) + 1e-8
            combined_output = torch.einsum('c,cb->b', alpha, candidate_outputs_flat) / alpha_sum
            
            # Reshape back and store
            combined_output = combined_output.reshape(orig_shape)
            actuator_outputs[..., a] = combined_output

        # Set final output - shape (n_s, n_c, n_actuators)
        self.output["inputSignal"].set(actuator_outputs, step_index)

    def compute_binarization_penalty(self) -> torch.Tensor:
        """Compute the binarization penalty P(x) = x(1-x) for all selection weights.

        Uses vectorized operations for efficient computation.
        All weights (alpha, beta, gamma) are per-actuator.

        Returns:
            torch.Tensor: Total binarization penalty
        """
        penalty = torch.tensor(0.0, dtype=torch.float64)

        # All weights are per-actuator
        for a in range(self.n_actuators):
            # Alpha penalties - vectorized
            alpha = self._get_alpha_vector(a)  # (n_candidates,)
            penalty = penalty + torch.sum(alpha * (1 - alpha))
            
            # Beta penalties - vectorized
            beta = self._get_beta_vector(a)  # (n_sensors,)
            penalty = penalty + torch.sum(beta * (1 - beta))
            
            # Gamma penalties - vectorized
            gamma = self._get_gamma_vector(a)  # (n_setpoints,)
            penalty = penalty + torch.sum(gamma * (1 - gamma))

        return penalty

    def compute_regularization_penalty(self) -> torch.Tensor:
        """Standard interface for Estimator to compute regularization penalty.
        
        This method is automatically called by the Estimator when 
        regularization_lambda > 0 is specified.

        Returns:
            torch.Tensor: Regularization penalty (binarization penalty for this component)
        """
        return self.compute_binarization_penalty()

    def get_selection_weights(self) -> Dict[str, torch.Tensor]:
        """Get all selection weights as a dictionary.
        
        All weights are per-actuator:
        - alpha_{a}_{c}: Individual alpha weights per actuator/candidate
        - alpha_{a}: Full alpha vector for actuator a
        - beta_{a}_{i}: Individual beta weights per actuator/sensor
        - beta_{a}: Full beta vector for actuator a
        - gamma_{a}_{j}: Individual gamma weights per actuator/setpoint
        - gamma_{a}: Full gamma vector for actuator a
        """
        weights = {}
        
        for a in range(self.n_actuators):
            # Alpha weights - individual and vector
            alpha_vec = self._get_alpha_vector(a)
            weights[f"alpha_{a}"] = alpha_vec.detach().clone()
            for c in range(self.n_candidates):
                weights[f"alpha_{a}_{c}"] = alpha_vec[c].detach().clone()

            # Beta weights - individual and vector (per-actuator)
            beta_vec = self._get_beta_vector(a)
            weights[f"beta_{a}"] = beta_vec.detach().clone()
            for i in range(self.n_sensors):
                weights[f"beta_{a}_{i}"] = beta_vec[i].detach().clone()

            # Gamma weights - individual and vector (per-actuator)
            gamma_vec = self._get_gamma_vector(a)
            weights[f"gamma_{a}"] = gamma_vec.detach().clone()
            for j in range(self.n_setpoints):
                weights[f"gamma_{a}_{j}"] = gamma_vec[j].detach().clone()

        return weights

    def get_candidate_controller(self, actuator: int, candidate: int) -> core.System:
        """Get a specific candidate controller instance."""
        return self._get_candidate(actuator, candidate)

    def get_identified_structure(
        self, threshold: float = 0.5
    ) -> Dict[str, Union[List, Dict]]:
        """Get the identified controller structure after thresholding.

        Args:
            threshold: Threshold for binary decision

        Returns:
            Dict describing the identified structure per actuator
        """
        weights = self.get_selection_weights()

        structure = {
            "actuators": {},
            "sensors": [],
            "setpoints": [],
        }

        # Identify active candidates per actuator
        for a in range(self.n_actuators):
            active_candidates = []
            for c in range(self.n_candidates):
                if weights.get(f"alpha_{a}_{c}", torch.tensor(0)).item() > threshold:
                    ctrl = self._get_candidate(a, c)
                    active_candidates.append({
                        "index": c,
                        "class": ctrl.__class__.__name__,
                        "id": ctrl.id,
                    })
            structure["actuators"][a] = active_candidates

        # Identify active sensors per actuator
        structure["sensors"] = {}
        for a in range(self.n_actuators):
            active_sensors = []
            for i in range(self.n_sensors):
                if weights.get(f"beta_{a}_{i}", torch.tensor(0)).item() > threshold:
                    active_sensors.append(i)
            structure["sensors"][a] = active_sensors

        # Identify active setpoints per actuator
        structure["setpoints"] = {}
        for a in range(self.n_actuators):
            active_setpoints = []
            for j in range(self.n_setpoints):
                if weights.get(f"gamma_{a}_{j}", torch.tensor(0)).item() > threshold:
                    active_setpoints.append(j)
            structure["setpoints"][a] = active_setpoints

        return structure

    def reset_state(self) -> None:
        """Reset the state of all candidate controllers."""
        for a in range(self.n_actuators):
            for c in range(self.n_candidates):
                ctrl = self._get_candidate(a, c)
                if hasattr(ctrl, 'reset_state'):
                    ctrl.reset_state()
                # Reset PID controller state
                if hasattr(ctrl, 'err_prev'):
                    ctrl.err_prev = torch.zeros_like(ctrl.err_prev)
                if hasattr(ctrl, 'err_prev_m1'):
                    ctrl.err_prev_m1 = torch.zeros_like(ctrl.err_prev_m1)
                if hasattr(ctrl, 'u_prev'):
                    ctrl.u_prev = torch.zeros_like(ctrl.u_prev)

    def summary(self) -> str:
        """Get a summary of the identified controller."""
        lines = ["=" * 60]
        lines.append("Controller Identification Summary")
        lines.append("=" * 60)
        lines.append(f"\nConfiguration:")
        lines.append(f"  Actuators: {self.n_actuators}")
        lines.append(f"  Sensors: {self.n_sensors}")
        lines.append(f"  Setpoints: {self.n_setpoints}")
        lines.append(f"  Candidates per actuator: {self.n_candidates}")

        # Selection weights
        lines.append("\nSelection Weights:")
        weights = self.get_selection_weights()
        
        # Group by type
        lines.append("  Alpha (candidate selection):")
        for a in range(self.n_actuators):
            for c in range(self.n_candidates):
                ctrl = self._get_candidate(a, c)
                val = weights[f"alpha_{a}_{c}"].item()
                lines.append(f"    α_{a},{c} ({ctrl.__class__.__name__}): {val:.4f}")
        
        lines.append("  Beta (sensor selection per actuator):")
        for a in range(self.n_actuators):
            lines.append(f"    Actuator {a}:")
            for i in range(self.n_sensors):
                val = weights[f"beta_{a}_{i}"].item()
                lines.append(f"      β_{a},{i}: {val:.4f}")
        
        lines.append("  Gamma (setpoint selection per actuator):")
        for a in range(self.n_actuators):
            lines.append(f"    Actuator {a}:")
            for j in range(self.n_setpoints):
                val = weights[f"gamma_{a}_{j}"].item()
                lines.append(f"      γ_{a},{j}: {val:.4f}")

        # Candidate controller parameters
        lines.append("\nCandidate Controller Parameters:")
        for a in range(self.n_actuators):
            lines.append(f"  Actuator {a}:")
            for c in range(self.n_candidates):
                ctrl = self._get_candidate(a, c)
                lines.append(f"    Candidate {c} ({ctrl.__class__.__name__}):")
                # PID controller parameters
                if hasattr(ctrl, 'kp'):
                    lines.append(f"      kp: {ctrl.kp.get().item():.6f}")
                if hasattr(ctrl, 'Ti'):
                    lines.append(f"      Ti: {ctrl.Ti.get().item():.6f}")
                if hasattr(ctrl, 'Td'):
                    lines.append(f"      Td: {ctrl.Td.get().item():.6f}")
                # On-Off controller parameters
                if hasattr(ctrl, 'offValue'):
                    lines.append(f"      offValue: {ctrl.offValue.get().item():.6f}")
                if hasattr(ctrl, 'onValue'):
                    lines.append(f"      onValue: {ctrl.onValue.get().item():.6f}")
                if hasattr(ctrl, 'steepness'):
                    lines.append(f"      steepness: {ctrl.steepness.get().item():.6f}")

        # Identified structure
        structure = self.get_identified_structure()
        lines.append("\nIdentified Structure (threshold=0.5):")
        for a in range(self.n_actuators):
            lines.append(f"  Actuator {a}:")
            lines.append(f"    Active sensors: {structure['sensors'].get(a, [])}")
            lines.append(f"    Active setpoints: {structure['setpoints'].get(a, [])}")
            candidates = structure['actuators'].get(a, [])
            lines.append(f"    Active controllers: {[c['class'] for c in candidates]}")

        lines.append("=" * 60)
        return "\n".join(lines)

    def get_estimator_parameters(self) -> List[tuple]:
        """Get parameter specifications for use with twin4build Estimator.

        Returns a list of parameter tuples for all selection weights and
        candidate controller parameters. Uses vector Parameters with n_c
        dimension for efficient handling of multiple weights.

        Returns:
            List of tuples: [(component, attr, x0, lb, ub), ...]
                - x0 can be scalar (broadcast to n_c) or array of shape (n_c,)
                - lb, ub can be scalar (broadcast) or array of shape (n_c,)
        """
        params = []

        weight_lb = 0.0
        weight_ub = 1.0

        # Initial values: uniform distribution (1/n)
        alpha_x0 = 0.5
        beta_x0 = 0.5
        gamma_x0 = 0.5

        # All selection weights are per-actuator:
        # - alpha_{a}: candidate selection (n_c = n_candidates)
        # - beta_{a}: sensor selection (n_c = n_sensors)  
        # - gamma_{a}: setpoint selection (n_c = n_setpoints)
        for a in range(self.n_actuators):
            params.append((self, f"alpha_{a}", alpha_x0, weight_lb, weight_ub))
            params.append((self, f"beta_{a}", beta_x0, weight_lb, weight_ub))
            params.append((self, f"gamma_{a}", gamma_x0, weight_lb, weight_ub))

        # Candidate controller parameters - accessed through parent using dot notation
        for a in range(self.n_actuators):
            for c in range(self.n_candidates):
                ctrl = self._get_candidate(a, c)
                prefix = f"candidate_{a}_{c}"
                # Add PID controller parameters if they exist
                if hasattr(ctrl, 'kp'):
                    params.append((self, f"{prefix}.kp", 0.3, 0.001, 1.0))
                if hasattr(ctrl, 'Ti'):
                    params.append((self, f"{prefix}.Ti", 5, 1.0, 100))
                if hasattr(ctrl, 'Td'):
                    params.append((self, f"{prefix}.Td", 0.0, 0.0, 100.0))
                # Add On-Off controller parameters if they exist
                if hasattr(ctrl, 'offValue'):
                    params.append((self, f"{prefix}.offValue", 0.0, 0, 1.0))
                if hasattr(ctrl, 'onValue'):
                    params.append((self, f"{prefix}.onValue", 1.0, 0.0, 1.0))
                if hasattr(ctrl, 'steepness'):
                    params.append((self, f"{prefix}.steepness", 100, 1, 100.0))

        return params

    def get_gradient_scales(self, weight_scale: float = 0.01) -> List[float]:
        """Get recommended gradient scaling factors for each parameter.
        
        Selection weights (alpha, beta, gamma) typically have gradients ~100x 
        larger than controller parameters due to directly multiplying signals.
        This method returns scaling factors that balance the gradients.
        
        Note: With vector Parameters, the scales are per-parameter (not per-element).
        The Estimator handles expanding scales to match n_c dimensions internally.
        
        Args:
            weight_scale: Scaling factor for selection weight gradients.
                Default 0.01 reduces weight gradients by 100x.
        
        Returns:
            List of gradient scales matching the order from get_estimator_parameters()
            
        Example:
            >>> params = controller.get_estimator_parameters()
            >>> scales = controller.get_gradient_scales()
            >>> result = estimator.estimate(
            ...     parameters=params,
            ...     gradient_scales=scales,
            ...     ...
            ... )
        """
        scales = []
        
        # Selection weights - all per-actuator, apply weight_scale
        # For each actuator: alpha, beta, gamma (3 parameters)
        for _ in range(self.n_actuators):
            scales.append(weight_scale)  # alpha
            scales.append(weight_scale)  # beta
            scales.append(weight_scale)  # gamma
        
        # Controller parameters - no scaling (1.0)
        for a in range(self.n_actuators):
            for c in range(self.n_candidates):
                ctrl = self._get_candidate(a, c)
                # PID controller parameters
                if hasattr(ctrl, 'kp'):
                    scales.append(1.0)
                if hasattr(ctrl, 'Ti'):
                    scales.append(1.0)
                if hasattr(ctrl, 'Td'):
                    scales.append(1.0)
                # On-Off controller parameters
                if hasattr(ctrl, 'offValue'):
                    scales.append(1.0)
                if hasattr(ctrl, 'onValue'):
                    scales.append(1.0)
                if hasattr(ctrl, 'steepness'):
                    scales.append(1.0)
        
        return scales
