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
from twin4build.systems.controller.setpoint_controller.cascade_controller.cascade_controller_system import (
    CascadeControllerSystem,
    CascadePIDControllerSystem,  # backward-compatible alias
)
from twin4build.systems.controller.rulebased_controller.sat_compensated_controller.sat_compensated_controller_torch_system import (
    SATCompensatedControllerTorchSystem,
)
from twin4build.translator.translator import (
    MultiPath,
    MultiPathRule,
    Node,
    Predicate,
    SignaturePattern,
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
        **kwargs,
    ):
        super().__init__(**kwargs)
        nn.Module.__init__(self)

        self.n_sensors = n_sensors
        self.n_setpoints = n_setpoints
        self.n_actuators = n_actuators

        # Setup candidate controllers
        if candidate_controllers is None:
            # Default candidates: PID (reverse), PID (non-reverse), Cascade PID,
            # SAT-Compensated cascade (SAT rule → flow setpoint, PID → damper)
            # PID naturally represents P (Td=0, Ti->inf), PI (Td=0), or PID behavior
            # Cascade controllers' B loop selects from the same sensor pool via beta_b
            candidate_controllers = [
                PIDControllerSystem,
                PIDControllerSystem,
                CascadeControllerSystem,
                SATCompensatedControllerTorchSystem,
            ]
            candidate_controller_kwargs = [
                {"kp": 0.3, "Ti": 5.0, "Td": 0.0, "isReverse": True},
                {"kp": 0.3, "Ti": 5.0, "Td": 0.0, "isReverse": False},
                {
                    "kp_a": 0.1, "Ti_a": 10.0, "Td_a": 0.0,
                    "kp_b": 0.5, "Ti_b": 5.0, "Td_b": 0.0,
                    "isReverse_a": True, "isReverse_b": True,
                },
                {
                    "base_position": 0.3, "sat_design": 13.0, "gain": 0.05,
                    "output_min_a": 0.0, "output_max_a": 1.0,
                    "kp_b": 0.5, "Ti_b": 5.0, "Td_b": 0.0,
                    "output_min_b": 0.0, "output_max_b": 1.0,
                    "isReverse_b": True,
                },
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

        # Detect whether any candidate is a cascade controller (has actualValue_b)
        # If so, create per-actuator beta_b weights to select B-loop feedback
        # from the same sensor pool as beta (no separate input needed)
        self._has_cascade = any(
            issubclass(cls, CascadeControllerSystem)
            for cls in candidate_controllers
        )
        if self._has_cascade:
            for a in range(n_actuators):
                setattr(
                    self,
                    f"beta_b_{a}",
                    tps.Parameter(
                        torch.full((n_sensors,), beta_init, dtype=torch.float64),
                        min_value=0.0,
                        max_value=1.0,
                        requires_grad=False,
                        n_c=n_sensors,
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

    def _get_beta_b_vector(self, actuator: int) -> torch.Tensor:
        """Get full beta_b vector (cascade B-loop sensor selection) for actuator a."""
        return getattr(self, f"beta_b_{actuator}").get()

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

    def _compute_weighted_signals(self, actuator: int) -> Tuple[torch.Tensor, ...]:
        """Compute weighted sensor and setpoint signals for a specific actuator.

        Uses ratio normalization (w_i / sum(w)) for beta, gamma, and beta_b weights.

        Args:
            actuator: The actuator index to compute signals for.

        Returns:
            Tuple of (weighted_setpoint, weighted_feedback[, weighted_feedback_b])
            tensors, each shape (n_s, n_c).  weighted_feedback_b is only included
            when cascade candidates exist.
        """
        # Tensor shapes: (n_s, n_c, n_v) where n_v is n_sensors or n_setpoints
        sensor_values = self.input["sensorValue"].get()  # (n_s, n_c, n_sensors)
        setpoint_values = self.input["setpointValue"].get()  # (n_s, n_c, n_setpoints)

        # Ratio-normalised weights
        gamma = self._get_gamma_vector(actuator)  # (n_setpoints,)
        gamma_norm = gamma / (torch.sum(gamma) + 1e-8)

        beta = self._get_beta_vector(actuator)  # (n_sensors,)
        beta_norm = beta / (torch.sum(beta) + 1e-8)

        # Weighted setpoint: sum_j gamma_norm_j * sp_jt
        weighted_setpoint = torch.sum(gamma_norm * setpoint_values, dim=-1)  # (n_s, n_c)

        # Weighted sensor feedback: sum_i beta_norm_i * y_it
        weighted_feedback = torch.sum(beta_norm * sensor_values, dim=-1)  # (n_s, n_c)

        if self._has_cascade:
            # Cascade B-loop feedback: second selection from the SAME sensor pool
            beta_b = self._get_beta_b_vector(actuator)  # (n_sensors,)
            beta_b_norm = beta_b / (torch.sum(beta_b) + 1e-8)
            weighted_feedback_b = torch.sum(beta_b_norm * sensor_values, dim=-1)  # (n_s, n_c)
            return weighted_setpoint, weighted_feedback, weighted_feedback_b

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
        n_s = self.input["sensorValue"].n_s
        n_c = self.input["sensorValue"].n_c

        # Process each actuator
        actuator_outputs = torch.zeros(n_s, n_c, self.n_actuators, dtype=torch.float64)

        for a in range(self.n_actuators):
            # Compute per-actuator weighted signals (each actuator has own beta/gamma/beta_b)
            signals = self._compute_weighted_signals(a)
            weighted_setpoint = signals[0]
            weighted_feedback = signals[1]
            weighted_feedback_b = signals[2] if len(signals) > 2 else None
            
            # Run all candidate controllers and collect outputs
            candidate_outputs = []
            for c in range(self.n_candidates):
                ctrl = self._get_candidate(a, c)
                
                # Route signals depending on controller type
                if "actualValue_b" in ctrl.input:
                    # Cascade controller: A loop gets setpoint/feedback, B loop gets beta_b selection
                    ctrl.input["setpointValue_a"].set(weighted_setpoint, step_index)
                    ctrl.input["actualValue_a"].set(weighted_feedback, step_index)
                    ctrl.input["actualValue_b"].set(weighted_feedback_b, step_index)
                elif "supplyAirTemp" in ctrl.input:
                    # SAT-compensated controller: uses weighted feedback as SAT input directly
                    # (beta selects which sensor to use as the supply air temperature)
                    ctrl.input["supplyAirTemp"].set(weighted_feedback, step_index)
                else:
                    # Standard controller (PID, on-off, etc.)
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
            
            # Ratio-normalised weighted sum of candidate outputs
            alpha = self._get_alpha_vector(a)
            alpha_norm = alpha / (torch.sum(alpha) + 1e-8)
            combined_output = torch.einsum('c,cb->b', alpha_norm, candidate_outputs_flat)
            
            # Reshape back and store
            combined_output = combined_output.reshape(orig_shape)
            actuator_outputs[..., a] = combined_output

        # Set final output - shape (n_s, n_c, n_actuators)
        self.output["inputSignal"].set(actuator_outputs, step_index)

    def compute_binarization_penalty(self) -> torch.Tensor:
        """Compute binarization penalty P(x) = x(1-x) for all selection weights.

        This penalty encourages weights toward 0 or 1, promoting crisp selection.

        Returns:
            torch.Tensor: Sum of x*(1-x) over all alpha, beta, gamma weights.
        """
        penalty = torch.tensor(0.0, dtype=torch.float64)
        for a in range(self.n_actuators):
            alpha = self._get_alpha_vector(a)
            penalty = penalty + torch.sum(alpha * (1 - alpha))
            beta = self._get_beta_vector(a)
            penalty = penalty + torch.sum(beta * (1 - beta))
            gamma = self._get_gamma_vector(a)
            penalty = penalty + torch.sum(gamma * (1 - gamma))
            if self._has_cascade:
                beta_b = self._get_beta_b_vector(a)
                penalty = penalty + torch.sum(beta_b * (1 - beta_b))
        return penalty

    def compute_regularization_penalty(self) -> torch.Tensor:
        """Standard interface for Estimator to compute regularization penalty.

        Returns the binarization penalty P(x) = x(1-x) over all selection weights.
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
            # Alpha weights
            alpha_vec = self._get_alpha_vector(a)
            weights[f"alpha_{a}"] = alpha_vec.detach().clone()
            for c in range(self.n_candidates):
                weights[f"alpha_{a}_{c}"] = alpha_vec[c].detach().clone()

            # Beta weights
            beta_vec = self._get_beta_vector(a)
            weights[f"beta_{a}"] = beta_vec.detach().clone()
            for i in range(self.n_sensors):
                weights[f"beta_{a}_{i}"] = beta_vec[i].detach().clone()

            # Gamma weights
            gamma_vec = self._get_gamma_vector(a)
            weights[f"gamma_{a}"] = gamma_vec.detach().clone()
            for j in range(self.n_setpoints):
                weights[f"gamma_{a}_{j}"] = gamma_vec[j].detach().clone()

            # Beta_b weights (cascade B-loop sensor selection)
            if self._has_cascade:
                beta_b_vec = self._get_beta_b_vector(a)
                weights[f"beta_b_{a}"] = beta_b_vec.detach().clone()
                for i in range(self.n_sensors):
                    weights[f"beta_b_{a}_{i}"] = beta_b_vec[i].detach().clone()

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

        # Identify active B-loop sensors per actuator (cascade controllers)
        if self._has_cascade:
            structure["sensors_b"] = {}
            for a in range(self.n_actuators):
                active_sensors_b = []
                for i in range(self.n_sensors):
                    if weights.get(f"beta_b_{a}_{i}", torch.tensor(0)).item() > threshold:
                        active_sensors_b.append(i)
                structure["sensors_b"][a] = active_sensors_b

        return structure

    def reset_state(self) -> None:
        """Reset the state of all candidate controllers."""
        for a in range(self.n_actuators):
            for c in range(self.n_candidates):
                ctrl = self._get_candidate(a, c)
                # Cascade controllers handle their own sub-PID reset
                if hasattr(ctrl, 'reset_state'):
                    ctrl.reset_state()
                # Reset PID controller state (direct PID candidates)
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
        lines.append(f"  Cascade candidates: {self._has_cascade}")
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

        if self._has_cascade:
            lines.append("  Beta_b (cascade B-loop sensor selection per actuator):")
            for a in range(self.n_actuators):
                lines.append(f"    Actuator {a}:")
                for i in range(self.n_sensors):
                    val = weights[f"beta_b_{a}_{i}"].item()
                    lines.append(f"      β_b_{a},{i}: {val:.4f}")

        # Candidate controller parameters
        lines.append("\nCandidate Controller Parameters:")
        for a in range(self.n_actuators):
            lines.append(f"  Actuator {a}:")
            for c in range(self.n_candidates):
                ctrl = self._get_candidate(a, c)
                lines.append(f"    Candidate {c} ({ctrl.__class__.__name__}):")
                if hasattr(ctrl, 'ctrl_a'):
                    # Composed cascade controller -- display both sub-controllers
                    for sub_name in ('ctrl_a', 'ctrl_b'):
                        sub = getattr(ctrl, sub_name)
                        lines.append(f"      {sub_name} ({sub.__class__.__name__}):")
                        # PID parameters
                        if hasattr(sub, 'kp'):
                            lines.append(f"        kp: {sub.kp.get().item():.6f}")
                        if hasattr(sub, 'Ti'):
                            lines.append(f"        Ti: {sub.Ti.get().item():.6f}")
                        if hasattr(sub, 'Td'):
                            lines.append(f"        Td: {sub.Td.get().item():.6f}")
                        if hasattr(sub, 'output_min'):
                            lines.append(f"        output_min: {sub.output_min.get().item():.6f}")
                        if hasattr(sub, 'output_max'):
                            lines.append(f"        output_max: {sub.output_max.get().item():.6f}")
                        if hasattr(sub, 'isReverse'):
                            lines.append(f"        isReverse: {sub.isReverse}")
                        # SAT-compensated parameters
                        if hasattr(sub, 'base_position'):
                            lines.append(f"        base_position: {sub.base_position.get().item():.6f}")
                        if hasattr(sub, 'sat_design'):
                            lines.append(f"        sat_design: {sub.sat_design.get().item():.6f}")
                        if hasattr(sub, 'gain'):
                            lines.append(f"        gain: {sub.gain.get().item():.6f}")
                else:
                    # Standard controllers (PID, on-off, etc.)
                    if hasattr(ctrl, 'kp'):
                        lines.append(f"      kp: {ctrl.kp.get().item():.6f}")
                    if hasattr(ctrl, 'Ti'):
                        lines.append(f"      Ti: {ctrl.Ti.get().item():.6f}")
                    if hasattr(ctrl, 'Td'):
                        lines.append(f"      Td: {ctrl.Td.get().item():.6f}")
                    if hasattr(ctrl, 'output_min'):
                        lines.append(f"      output_min: {ctrl.output_min.get().item():.6f}")
                    if hasattr(ctrl, 'output_max'):
                        lines.append(f"      output_max: {ctrl.output_max.get().item():.6f}")
                    if hasattr(ctrl, 'isReverse'):
                        lines.append(f"      isReverse: {ctrl.isReverse}")
                    # On-Off controller parameters
                    if hasattr(ctrl, 'offValue'):
                        lines.append(f"      offValue: {ctrl.offValue.get().item():.6f}")
                    if hasattr(ctrl, 'onValue'):
                        lines.append(f"      onValue: {ctrl.onValue.get().item():.6f}")
                    if hasattr(ctrl, 'steepness'):
                        lines.append(f"      steepness: {ctrl.steepness.get().item():.6f}")
                    # SAT-compensated controller parameters
                    if hasattr(ctrl, 'base_position'):
                        lines.append(f"      base_position: {ctrl.base_position.get().item():.6f}")
                    if hasattr(ctrl, 'sat_design'):
                        lines.append(f"      sat_design: {ctrl.sat_design.get().item():.6f}")
                    if hasattr(ctrl, 'gain'):
                        lines.append(f"      gain: {ctrl.gain.get().item():.6f}")

        # Identified structure
        structure = self.get_identified_structure()
        lines.append("\nIdentified Structure (threshold=0.5):")
        for a in range(self.n_actuators):
            lines.append(f"  Actuator {a}:")
            lines.append(f"    Active sensors: {structure['sensors'].get(a, [])}")
            lines.append(f"    Active setpoints: {structure['setpoints'].get(a, [])}")
            if self._has_cascade:
                lines.append(f"    Active B-loop sensors: {structure.get('sensors_b', {}).get(a, [])}")
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
            List of tuples: [(component, attr, x0, lb, ub, "private"), ...]
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
        for a in range(self.n_actuators): # NOTE: Commented out Temporary
            params.append((self, f"alpha_{a}", alpha_x0, weight_lb, weight_ub, "private"))
            params.append((self, f"beta_{a}", beta_x0, weight_lb, weight_ub, "private"))
            params.append((self, f"gamma_{a}", gamma_x0, weight_lb, weight_ub, "private"))
            if self._has_cascade:
                params.append((self, f"beta_b_{a}", beta_x0, weight_lb, weight_ub, "private"))

        # Candidate controller parameters - accessed through parent using dot notation
        for a in range(self.n_actuators):
            for c in range(self.n_candidates):
                ctrl = self._get_candidate(a, c)
                prefix = f"candidate_{a}_{c}"

                if hasattr(ctrl, 'ctrl_a'):
                    # Composed cascade controller -- expose both sub-controller params
                    # ctrl_a saturation = intermediate clamp, ctrl_b saturation = final output
                    for sub_name in ('ctrl_a', 'ctrl_b'):
                        sub = getattr(ctrl, sub_name)
                        sub_prefix = f"{prefix}.{sub_name}"
                        # PID parameters
                        if hasattr(sub, 'kp'):
                            params.append((self, f"{sub_prefix}.kp", 0.1, 0.001, 10, "private"))
                        if hasattr(sub, 'Ti'):
                            params.append((self, f"{sub_prefix}.Ti", 10, 1.0, 100, "private"))
                        if hasattr(sub, 'Td'):
                            params.append((self, f"{sub_prefix}.Td", 0.0, 0.0, 0.0001, "private"))
                        if hasattr(sub, 'output_min'):
                            params.append((self, f"{sub_prefix}.output_min", 0.0, 0.0, 1.0, "private"))
                        if hasattr(sub, 'output_max'):
                            params.append((self, f"{sub_prefix}.output_max", 1.0, 0.0, 1.0, "private"))
                        # SAT-compensated parameters
                        if hasattr(sub, 'base_position'):
                            params.append((self, f"{sub_prefix}.base_position", 0.3, 0.0, 1.0, "private"))
                        if hasattr(sub, 'sat_design'):
                            params.append((self, f"{sub_prefix}.sat_design", 13.0, 0.0, 30.0, "private"))
                        if hasattr(sub, 'gain'):
                            params.append((self, f"{sub_prefix}.gain", 0.05, -0.5, 0.5, "private"))
                else:
                    # Standard controllers (PID, on-off, etc.)
                    if hasattr(ctrl, 'kp'):
                        params.append((self, f"{prefix}.kp", 0.01, 0.001, 1, "private"))
                    if hasattr(ctrl, 'Ti'):
                        params.append((self, f"{prefix}.Ti", 10, 1.0, 100, "private"))
                    if hasattr(ctrl, 'Td'):
                        params.append((self, f"{prefix}.Td", 0.0, 0.0, 0.0001, "private"))
                    if hasattr(ctrl, 'output_min'):
                        params.append((self, f"{prefix}.output_min", 0.5, 0.0, 1.0, "private"))
                    # if hasattr(ctrl, 'output_max'):
                    #     params.append((self, f"{prefix}.output_max", 1.0, 0.0, 1.0, "private"))
                    # On-Off controller parameters
                    if hasattr(ctrl, 'offValue'):
                        params.append((self, f"{prefix}.offValue", 0.0, 0, 1.0, "private"))
                    if hasattr(ctrl, 'onValue'):
                        params.append((self, f"{prefix}.onValue", 1.0, 0.0, 1.0, "private"))
                    if hasattr(ctrl, 'steepness'):
                        params.append((self, f"{prefix}.steepness", 100, 1, 100.0, "private"))
                    # SAT-compensated controller parameters
                    if hasattr(ctrl, 'base_position'):
                        params.append((self, f"{prefix}.base_position", 0.3, 0.0, 1.0, "private"))
                    if hasattr(ctrl, 'sat_design'):
                        params.append((self, f"{prefix}.sat_design", 13.0, 0.0, 30.0, "private"))
                    if hasattr(ctrl, 'gain'):
                        params.append((self, f"{prefix}.gain", 0.05, -0.5, 0.5, "private"))

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
        # For each actuator: alpha, beta, gamma[, beta_b]
        for _ in range(self.n_actuators):
            scales.append(weight_scale)  # alpha
            scales.append(weight_scale)  # beta
            scales.append(weight_scale)  # gamma
            if self._has_cascade:
                scales.append(weight_scale)  # beta_b
        
        # Controller parameters - no scaling (1.0)
        # Must match the order and count of get_estimator_parameters()
        for a in range(self.n_actuators):
            for c in range(self.n_candidates):
                ctrl = self._get_candidate(a, c)

                if hasattr(ctrl, 'ctrl_a'):
                    # Composed cascade: count sub-controller params dynamically
                    for sub_name in ('ctrl_a', 'ctrl_b'):
                        sub = getattr(ctrl, sub_name)
                        for attr_name in ('kp', 'Ti', 'Td', 'output_min', 'output_max',
                                          'base_position', 'sat_design', 'gain'):
                            if hasattr(sub, attr_name):
                                scales.append(1.0)
                else:
                    # Standard controllers - must mirror get_estimator_parameters()
                    if hasattr(ctrl, 'kp'):
                        scales.append(1.0)
                    if hasattr(ctrl, 'Ti'):
                        scales.append(1.0)
                    if hasattr(ctrl, 'Td'):
                        scales.append(1.0)
                    if hasattr(ctrl, 'output_min'):
                        scales.append(1.0)
                    # On-Off controller parameters
                    if hasattr(ctrl, 'offValue'):
                        scales.append(1.0)
                    if hasattr(ctrl, 'onValue'):
                        scales.append(1.0)
                    if hasattr(ctrl, 'steepness'):
                        scales.append(1.0)
                    # SAT-compensated controller parameters
                    if hasattr(ctrl, 'base_position'):
                        scales.append(1.0)
                    if hasattr(ctrl, 'sat_design'):
                        scales.append(1.0)
                    if hasattr(ctrl, 'gain'):
                        scales.append(1.0)
        
        return scales


# def brick_signature_pattern():
#     """
#     BRICK signature pattern for AHU damper control identification.
    
#     Since BRICK doesn't include explicit control descriptions, this pattern
#     matches an AHU feeding spaces and creates a controller identification
#     system to infer the control logic from data.
    
#     The controller:
#     - Observes zone temperatures (sensorValue) from the fed spaces
#     - Tracks temperature setpoints (setpointValue)
#     - Outputs damper positions (inputSignal) to the AHU
#     """
#     sp = SignaturePattern(id="controller_identification_ahu_damper_brick")
    
#     # Match AHU that feeds spaces
#     ahu = Node(cls=core.namespace.BRICK.AHU)
#     spaces = Node(
#         cls=(
#             core.namespace.BRICK.Room,
#             core.namespace.BRICK.Enclosed_space,
#             core.namespace.BRICK.Open_space,
#         )
#     )
    
#     feeds = Predicate((core.namespace.BRICK.feeds, core.namespace.FSO.feedsFluidTo))
    
#     sp.add_triple(
#         MultiPathRule(subject=ahu, object=spaces, predicate=feeds)
#     )
    
#     # Connect space temperatures to controller's sensorValue (Vector input)
#     sp.add_connection(spaces, "indoorTemperature", "sensorValue", input_port_index=spaces)
    
#     # Connect controller's output to AHU's damper positions
#     sp.add_connection("inputSignal", ahu, "supplyDamperPosition", output_port_index=spaces)
#     sp.add_connection("inputSignal", ahu, "exhaustDamperPosition", output_port_index=spaces)
    
#     sp.add_modeled_node(ahu)  # Controller is created for each AHU
    
#     return sp


# ControllerIdentificationTorchSystem.add_signature_pattern(brick_signature_pattern())
