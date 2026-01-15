# Standard library imports
import datetime
from typing import List

# Third party imports
import torch
import torch.nn as nn

# Local application imports
import twin4build.core as core
import twin4build.utils.types as tps


class OnOffControllerTorchSystem(core.System, nn.Module):
    r"""
    Differentiable On-Off Controller System using smooth sigmoid approximation.

    This class implements a differentiable on-off (bang-bang) controller that can be
    used for gradient-based parameter estimation. Instead of a hard switch, it uses
    a sigmoid function to create a smooth transition between off and on states.

    Mathematical Formulation
    ------------------------

    **Error signal:**

    .. math::

        e_t = sp_t - y_t \quad \text{(reverse mode)}
        e_t = y_t - sp_t \quad \text{(normal mode)}

    **Smooth switching function:**

    .. math::

        \sigma(e_t) = \frac{1}{1 + e^{-k \cdot e_t}}

    where :math:`k` is the steepness parameter controlling transition sharpness.

    **Output signal:**

    .. math::

        u_t = u_{off} + \sigma(e_t) \cdot (u_{on} - u_{off})

    As :math:`k \to \infty`, this approaches the hard on-off behavior.

    Args:
        offValue: Output value when controller is OFF (default: 0.0)
        onValue: Output value when controller is ON (default: 1.0)
        steepness: Controls sharpness of transition (default: 10.0).
            Higher values = sharper transition (more like true on-off).
            Lower values = smoother transition.
        isReverse: If True, ON when actual < setpoint (heating mode).
            If False, ON when actual > setpoint (cooling mode). (default: False)
        **kwargs: Additional keyword arguments passed to parent classes

    Example:
        >>> # Heating controller: ON when temp < setpoint
        >>> controller = OnOffControllerTorchSystem(
        ...     offValue=0.0,
        ...     onValue=1.0,
        ...     steepness=10.0,
        ...     isReverse=True,
        ...     id="heater_controller"
        ... )
    """

    def __init__(
        self,
        offValue: float = 0.0,
        onValue: float = 1.0,
        steepness: float = 10.0,
        isReverse: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        nn.Module.__init__(self)

        self.isReverse = isReverse

        # Make offValue and onValue learnable parameters
        self.offValue = tps.Parameter(
            torch.tensor(float(offValue), dtype=torch.float64),
            requires_grad=False,
        )
        self.onValue = tps.Parameter(
            torch.tensor(float(onValue), dtype=torch.float64),
            requires_grad=False,
        )
        # Steepness controls transition sharpness
        # Higher = sharper (more like true on-off), Lower = smoother
        self.steepness = tps.Parameter(
            torch.tensor(float(steepness), dtype=torch.float64),
            requires_grad=False,
        )

        self.input = {"actualValue": tps.Scalar(), "setpointValue": tps.Scalar()}
        self.output = {"inputSignal": tps.Scalar()}
        self._config = {
            "parameters": ["offValue", "onValue", "steepness", "isReverse"],
        }

    @property
    def config(self):
        return self._config

    def initialize(
        self,
        start_time: List[datetime.datetime],
        end_time: List[datetime.datetime],
        step_size: int,
    ) -> None:
        """Initialize the controller for simulation."""
        _, _, max_timesteps, _ = core.Simulator.get_simulation_timesteps(
            start_time, end_time, step_size
        )
        batch_size = len(start_time)
        self.input["actualValue"].initialize(
            n_timesteps=max_timesteps,
            batch_size=batch_size,
        )
        self.input["setpointValue"].initialize(
            n_timesteps=max_timesteps,
            batch_size=batch_size,
        )
        self.output["inputSignal"].initialize(
            n_timesteps=max_timesteps,
            batch_size=batch_size,
        )

    def power_law_saturation(
        self,
        error: torch.Tensor,
        off_value: float = 0.0,
        on_value: float = 1.0,
        steepness: float = 100,
        curve_start: float = 0.1,
        a: float = 1.0,
        power_exp: float = 0.5,
    ) -> torch.Tensor:
        """
        Differentiable saturation function with linear passthrough and power law asymptotes.
        
        Maps error to output signal [off_value, on_value]:
            - error << 0 → off_value
            - error = 0 → midpoint between off_value and on_value
            - error >> 0 → on_value
        
        Args:
            error: Input error signal (setpoint - actual for reverse mode)
            off_value: Output value when controller is OFF
            on_value: Output value when controller is ON
            steepness: Slope/gain in the linear region (higher = sharper transition)
            curve_start: Width of linear region in normalized [0, 1] space
            a: Controls rate of asymptotic decay
            power_exp: Power law exponent (smaller = slower gradient decay)
        
        Returns:
            Output signal in [off_value, on_value]
        """
        # Compute switch signal in [0, 1]
        u = 0.5 + error * steepness
        
        lower_curve_point = curve_start
        upper_curve_point = 1.0 - curve_start
        
        def curve_function(x: torch.Tensor) -> torch.Tensor:
            scaled_x = a * x / curve_start
            return 1 - 1 / torch.pow(1 + scaled_x, power_exp)
        
        switch_signal = torch.where(
            u < lower_curve_point,
            # Lower region: curve toward 0
            curve_start * (1 - curve_function(lower_curve_point - u)),
            torch.where(
                u > upper_curve_point,
                # Upper region: curve toward 1
                1.0 - curve_start * (1 - curve_function(u - upper_curve_point)),
                # Linear region: passthrough
                u,
            ),
        )
    
        # Scale from [0, 1] to [off_value, on_value]
        return off_value + switch_signal * (on_value - off_value)

    def do_step(
        self,
        second_time: float,
        date_time: datetime.datetime,
        step_size: int,
        step_index: int,
    ) -> None:
        """
        Perform one simulation step with differentiable on-off switching.

        Uses sigmoid function for smooth, differentiable transition:
        - Reverse mode (heating): ON when actual < setpoint
        - Normal mode (cooling): ON when actual > setpoint
        """
        actual_value = self.input["actualValue"].get()
        setpoint_value = self.input["setpointValue"].get()
        
        k = self.steepness.get()
        off_val = self.offValue.get()
        on_val = self.onValue.get()

        # Compute error based on mode
        # Reverse: want to turn ON when actual < setpoint (error > 0)
        # Normal: want to turn ON when actual > setpoint (error > 0)
        if self.isReverse:
            error = setpoint_value - actual_value
        else:
            error = actual_value - setpoint_value

        # Smooth sigmoid switching
        # sigmoid(k * error) → 1 when error >> 0 (ON)
        # sigmoid(k * error) → 0 when error << 0 (OFF)
        output_signal = self.power_law_saturation(error, off_value=off_val, on_value=on_val, steepness=k)

        # Interpolate between off and on values
        # output_signal = off_val + switch_signal * (on_val - off_val)

        self.output["inputSignal"].set(output_signal, step_index)

    def get_switch_state(self, actual_value: torch.Tensor, setpoint_value: torch.Tensor) -> torch.Tensor:
        """
        Get the current switch state (0 to 1) for given inputs.
        
        Useful for debugging and visualization.
        
        Args:
            actual_value: Current measured value
            setpoint_value: Current setpoint value
            
        Returns:
            Switch state between 0 (OFF) and 1 (ON)
        """
        k = self.steepness.get()
        
        if self.isReverse:
            error = setpoint_value - actual_value
        else:
            error = actual_value - setpoint_value
            
        return torch.sigmoid(k * error)

    def reset_state(self) -> None:
        """Reset controller state (no-op for on-off controller)."""
        pass

