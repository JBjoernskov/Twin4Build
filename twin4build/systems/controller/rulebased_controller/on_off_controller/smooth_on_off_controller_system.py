# Standard library imports
import datetime
from typing import List

# Third party imports
import torch
import torch.nn as nn

# Local application imports
import twin4build.core as core
import twin4build.utils.types as tps
from twin4build.systems.utils.smooth_saturation import clamp
from twin4build.utils.deprecation import deprecate_args


class SmoothOnOffControllerSystem(core.System, nn.Module):
    r"""
    Differentiable On-Off Controller System using smooth sigmoid approximation.

    This class implements a differentiable on-off (bang-bang) controller that can be
    used for gradient-based parameter estimation. Instead of a hard switch, it uses
    a power law function to create a smooth transition between off and on states.

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
        >>> controller = SmoothOnOffControllerSystem(
        ...     offValue=0.0,
        ...     onValue=1.0,
        ...     steepness=10.0,
        ...     isReverse=True,
        ...     id="heater_controller"
        ... )
    """

    def __init__(
        self,
        off_value: float = 0.0,
        on_value: float = 1.0,
        steepness: float = 10.0,
        is_reverse: bool = False,
        **kwargs,
    ):
        legacy = deprecate_args(
            ["offValue", "onValue", "isReverse"],
            ["off_value", "on_value", "is_reverse"],
            [None, None, None],
            kwargs,
        )
        off_value = legacy.get("off_value", off_value)
        on_value = legacy.get("on_value", on_value)
        is_reverse = legacy.get("is_reverse", is_reverse)

        super().__init__(**kwargs)
        nn.Module.__init__(self)

        self.is_reverse = is_reverse

        self.off_value = tps.Parameter(
            torch.tensor(float(off_value), dtype=tps.float_dtype()),
            requires_grad=False,
        )
        self.on_value = tps.Parameter(
            torch.tensor(float(on_value), dtype=tps.float_dtype()),
            requires_grad=False,
        )
        self.steepness = tps.Parameter(
            torch.tensor(float(steepness), dtype=tps.float_dtype()),
            requires_grad=False,
        )

        self.input = {"actualValue": tps.Scalar(), "setpointValue": tps.Scalar()}
        self.output = {"inputSignal": tps.Scalar()}
        self._config = {
            "parameters": ["off_value", "on_value", "steepness", "is_reverse"],
        }
        # Deprecated attribute aliases until 2.1 (same Parameter objects)
        self.offValue = self.off_value
        self.onValue = self.on_value
        self.isReverse = self.is_reverse

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
            n_t=max_timesteps,
            n_s=batch_size,
        )
        self.input["setpointValue"].initialize(
            n_t=max_timesteps,
            n_s=batch_size,
        )
        self.output["inputSignal"].initialize(
            n_t=max_timesteps,
            n_s=batch_size,
        )

        # Expand parameters to n_c dimension for vectorization
        self.off_value = self.off_value.expand_to_n_c(self.n_c)
        self.on_value = self.on_value.expand_to_n_c(self.n_c)
        self.steepness = self.steepness.expand_to_n_c(self.n_c)

    @staticmethod
    def power_law_saturation(
        error: torch.Tensor,
        off_value: float = 0.0,
        on_value: float = 1.0,
        steepness: float = 100,
        curve_start: float = 0.1,
        a: float = 1.0,
        power_exp: float = 0.5,
    ) -> torch.Tensor:
        """Differentiable on/off switching via :func:`clamp` in smooth mode.

        Maps *error* → ``[off_value, on_value]``:
        ``error << 0`` → *off_value*,  ``error >> 0`` → *on_value*.

        Always uses ``mode="smooth"``: a hard step would zero the
        gradient with respect to the controller's setpoint and make
        the switching threshold structurally non-estimable.
        """
        u = 0.5 + error * steepness
        switch_signal = clamp(
            u,
            lower=0.0,
            upper=1.0,
            curve_start=curve_start,
            steepness=a,
            power_exp=power_exp,
            mode="smooth",
        )
        return off_value + switch_signal * (on_value - off_value)

    PARAM_NAMES = ("offValue", "onValue", "steepness")

    def forward(self, x, inputs, params, sample_time):
        """Pure one-step on-off switching (functorch-safe, stateless).

        - Reverse mode (heating): ON when actual < setpoint
        - Normal mode (cooling): ON when actual > setpoint

        ``isReverse`` is a structural (non-estimable) flag, fixed at
        construction.
        """
        # Reverse: want to turn ON when actual < setpoint (error > 0)
        # Normal: want to turn ON when actual > setpoint (error > 0)
        if self.is_reverse:
            error = inputs["setpointValue"] - inputs["actualValue"]
        else:
            error = inputs["actualValue"] - inputs["setpointValue"]

        # Smooth sigmoid switching
        # sigmoid(k * error) → 1 when error >> 0 (ON)
        # sigmoid(k * error) → 0 when error << 0 (OFF)
        output_signal = self.power_law_saturation(
            error,
            off_value=params["offValue"],
            on_value=params["onValue"],
            steepness=params["steepness"],
        )
        return x, {"inputSignal": output_signal}

    def do_step(
        self,
        second_time: float,
        date_time: datetime.datetime,
        step_size: int,
        step_index: int,
    ) -> None:
        """
        Perform one simulation step with differentiable on-off switching.

        Thin port-I/O wrapper delegating the math to :meth:`forward`.
        """
        inputs = {
            "actualValue": self.input["actualValue"].get(),
            "setpointValue": self.input["setpointValue"].get(),
        }
        _, outs = self.forward(
            None, inputs, self._forward_params(), self._scalar_sample_time(step_size)
        )
        self.output["inputSignal"].set(outs["inputSignal"], step_index)

    def get_switch_state(
        self, actual_value: torch.Tensor, setpoint_value: torch.Tensor
    ) -> torch.Tensor:
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

        if self.is_reverse:
            error = setpoint_value - actual_value
        else:
            error = actual_value - setpoint_value

        return torch.sigmoid(k * error)

    def reset_state(self) -> None:
        """Reset controller state (no-op for on-off controller)."""
        pass

# Deprecated aliases (removed in twin4build 2.1)
OnOffControllerTorchSystem = SmoothOnOffControllerSystem
