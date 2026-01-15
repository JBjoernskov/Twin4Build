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

        # Selection weights (alpha) - one per candidate per actuator
        # α_a,c selects candidate c for actuator a
        # Initialize uniformly: 1/n_candidates
        alpha_init = 1.0 / self.n_candidates
        for a in range(n_actuators):
            for c in range(self.n_candidates):
                setattr(
                    self,
                    f"alpha_{a}_{c}",
                    tps.Parameter(
                        torch.tensor(alpha_init, dtype=torch.float64),
                        min_value=0.0,
                        max_value=1.0,
                        requires_grad=False,
                    ),
                )

        # Feedback sensor weights (beta) - one per sensor
        # Initialize uniformly: 1/n_sensors
        beta_init = 1.0 / n_sensors
        for i in range(n_sensors):
            setattr(
                self,
                f"beta_{i}",
                tps.Parameter(
                    torch.tensor(beta_init, dtype=torch.float64),
                    min_value=0.0,
                    max_value=1.0,
                    requires_grad=False,
                ),
            )

        # Setpoint signal weights (gamma) - one per setpoint
        # Initialize uniformly: 1/n_setpoints
        gamma_init = 1.0 / n_setpoints
        for j in range(n_setpoints):
            setattr(
                self,
                f"gamma_{j}",
                tps.Parameter(
                    torch.tensor(gamma_init, dtype=torch.float64),
                    min_value=0.0,
                    max_value=1.0,
                    requires_grad=False,
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
        return getattr(self, f"alpha_{actuator}_{candidate}").get()

    def _get_beta(self, i: int) -> torch.Tensor:
        """Get beta parameter for sensor i."""
        return getattr(self, f"beta_{i}").get()

    def _get_gamma(self, j: int) -> torch.Tensor:
        """Get gamma parameter for setpoint j."""
        return getattr(self, f"gamma_{j}").get()

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

    def _compute_weighted_signals(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the weighted sensor and setpoint signals.

        Returns:
            Tuple of (weighted_setpoint, weighted_feedback) tensors, each shape (batch,)
        """
        sensor_values = self.input["sensorValue"].get()
        setpoint_values = self.input["setpointValue"].get()
        
        batch_size = sensor_values.shape[0]

        # Weighted setpoint: sum_j(gamma_j * sp_jt)
        weighted_setpoint = torch.zeros(batch_size, dtype=sensor_values.dtype, device=sensor_values.device)
        for j in range(self.n_setpoints):
            sp = setpoint_values[:, j] if setpoint_values.dim() > 1 else setpoint_values.squeeze()
            gamma = self._get_gamma(j)
            weighted_setpoint = weighted_setpoint + gamma * sp

        # Weighted sensor feedback: sum_i(beta_i * y_it)
        weighted_feedback = torch.zeros(batch_size, dtype=sensor_values.dtype, device=sensor_values.device)
        for i in range(self.n_sensors):
            y = sensor_values[:, i] if sensor_values.dim() > 1 else sensor_values.squeeze()
            beta = self._get_beta(i)
            weighted_feedback = weighted_feedback + beta * y
            
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
        1. Compute weighted setpoint and feedback signals
        2. Run all candidate controllers
        3. Combine outputs using alpha weights
        """
        # Compute weighted signals
        weighted_setpoint, weighted_feedback = self._compute_weighted_signals()
        batch_size = weighted_setpoint.shape[0]

        # Process each actuator
        actuator_outputs = torch.zeros(batch_size, self.n_actuators, dtype=torch.float64)

        for a in range(self.n_actuators):
            # Combined output for this actuator
            combined_output = torch.zeros(batch_size, dtype=torch.float64)

            for c in range(self.n_candidates):
                ctrl = self._get_candidate(a, c)
                
                # Set inputs for candidate controller
                ctrl.input["setpointValue"].set(weighted_setpoint, step_index)
                ctrl.input["actualValue"].set(weighted_feedback, step_index)
                
                # Run candidate controller step
                ctrl.do_step(second_time, date_time, step_size, step_index)
                
                # Get candidate output and weight it
                candidate_output = ctrl.output["inputSignal"].get()
                alpha = self._get_alpha(a, c)
                combined_output = combined_output + alpha * candidate_output

            # Store output for this actuator
            actuator_outputs[:, a] = combined_output/self.n_candidates # take average of all candidates

        # Set final output
        self.output["inputSignal"].set(actuator_outputs, step_index)

    def compute_binarization_penalty(self) -> torch.Tensor:
        """Compute the binarization penalty P(x) = x(1-x) for all selection weights.

        Returns:
            torch.Tensor: Total binarization penalty
        """
        penalty = torch.tensor(0.0, dtype=torch.float64)

        # Alpha penalties (per actuator, per candidate)
        for a in range(self.n_actuators):
            for c in range(self.n_candidates):
                x = self._get_alpha(a, c)
                penalty = penalty + x * (1 - x)

        # Beta penalties
        for i in range(self.n_sensors):
            x = self._get_beta(i)
            penalty = penalty + x * (1 - x)

        # Gamma penalties
        for j in range(self.n_setpoints):
            x = self._get_gamma(j)
            penalty = penalty + x * (1 - x)

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
        """Get all selection weights as a dictionary."""
        weights = {}
        
        # Alpha weights
        for a in range(self.n_actuators):
            for c in range(self.n_candidates):
                weights[f"alpha_{a}_{c}"] = self._get_alpha(a, c).detach().clone()

        # Beta weights
        for i in range(self.n_sensors):
            weights[f"beta_{i}"] = self._get_beta(i).detach().clone()

        # Gamma weights
        for j in range(self.n_setpoints):
            weights[f"gamma_{j}"] = self._get_gamma(j).detach().clone()

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

        # Identify active sensors
        for i in range(self.n_sensors):
            if weights.get(f"beta_{i}", torch.tensor(0)).item() > threshold:
                structure["sensors"].append(i)

        # Identify active setpoints
        for j in range(self.n_setpoints):
            if weights.get(f"gamma_{j}", torch.tensor(0)).item() > threshold:
                structure["setpoints"].append(j)

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
        
        lines.append("  Beta (sensor selection):")
        for i in range(self.n_sensors):
            val = weights[f"beta_{i}"].item()
            lines.append(f"    β_{i}: {val:.4f}")
        
        lines.append("  Gamma (setpoint selection):")
        for j in range(self.n_setpoints):
            val = weights[f"gamma_{j}"].item()
            lines.append(f"    γ_{j}: {val:.4f}")

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
        lines.append(f"  Active sensors: {structure['sensors']}")
        lines.append(f"  Active setpoints: {structure['setpoints']}")
        for a, candidates in structure['actuators'].items():
            lines.append(f"  Actuator {a} active controllers: {[c['class'] for c in candidates]}")

        lines.append("=" * 60)
        return "\n".join(lines)

    def get_estimator_parameters(self) -> List[tuple]:
        """Get parameter specifications for use with twin4build Estimator.

        Returns a list of parameter tuples for all selection weights and
        candidate controller parameters.

        Returns:
            List of tuples: [(component, attr, x0, lb, ub), ...]
        """
        params = []

        weight_lb = 0.0
        weight_ub = 1.0

        # Initial values: uniform distribution (1/n)
        alpha_x0 = 1.0 / self.n_candidates
        beta_x0 = 1.0 / self.n_sensors
        gamma_x0 = 1.0 / self.n_setpoints

        # Selection weights (alpha) - per actuator, per candidate
        for a in range(self.n_actuators):
            for c in range(self.n_candidates):
                params.append((self, f"alpha_{a}_{c}", alpha_x0, weight_lb, weight_ub))

        # Beta weights (sensor selection)
        for i in range(self.n_sensors):
            params.append((self, f"beta_{i}", beta_x0, weight_lb, weight_ub))

        # Gamma weights (setpoint selection)
        for j in range(self.n_setpoints):
            params.append((self, f"gamma_{j}", gamma_x0, weight_lb, weight_ub))

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
        
        # Selection weights - apply weight_scale
        n_weights = (self.n_actuators * self.n_candidates +  # alpha
                     self.n_sensors +  # beta
                     self.n_setpoints)  # gamma
        scales.extend([weight_scale] * n_weights)
        
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
