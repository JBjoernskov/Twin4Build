# Third party imports
import numpy as np
import torch

# Local application imports
import twin4build.core as core
from twin4build.systems.saref4syst.declarations import Input, Output, Parameter, Setting, State
from twin4build.systems.utils.smooth_saturation import (
    clamp,
    hardclamp_smooth_grad,
    smooth_saturation,
)


def _direction(value):
    """``is_reverse``: a bool, or one bool per copy (a bool tensor, a list);
    one value reads as a bool."""
    if not isinstance(value, (torch.Tensor, list, tuple, np.ndarray)):
        return bool(value)
    value = torch.as_tensor(value).detach().reshape(-1).to(torch.bool)
    return bool(value[0]) if value.numel() == 1 else value


class PIDControllerSystem(core.System):
    r"""
    PID Controller System.

    A positional PID with a differentiable output saturation and an
    integral that does not wind up:

    .. math::

        e_t = \pm(sp_t - y_t), \qquad
        p_t = k_p e_t + k_p \frac{T_d}{\Delta t} (e_t - e_{t-1})

        I_t = \min\bigl(\max(I_{t-1} + k_p \tfrac{\Delta t}{T_i} e_t,\;
              \min(I_{t-1}, u_{min} - p_t)),\; \max(I_{t-1}, u_{max} - p_t)\bigr)

        u_t = \mathrm{clamp}(p_t + I_t, u_{min}, u_{max})

    The integral moves until the output saturates and no further, and it
    holds its value while the output saturates on the error's side
    (conditional integration).  With :math:`T_i \to \infty` the integral
    keeps its start value (zero) and the controller is purely proportional
    about its setpoint, :math:`u = \mathrm{clamp}(k_p e)`: a loop that acts
    only above a threshold (a CO2 loop opening a damper above 800 ppm) is
    the same structure as a PI loop.  Unsaturated, the output equals the
    velocity-form PID's.

    Args:
        kp: Proportional gain
        Ti: Integral time constant
        Td: Derivative time constant
        output_min: Lower saturation limit for the controller output
        output_max: Upper saturation limit for the controller output
        is_reverse: The direction of action: ``True`` acts on ``setpoint -
            feedback`` (heating: a cold room opens the valve), ``False`` on
            ``feedback - setpoint``.  Batched copies may act in different
            directions (one value per copy).
    """

    #: The old name of ``is_reverse`` (removed in 2.1).
    DEPRECATED_KEYWORDS = {"isReverse": "is_reverse"}

    is_reverse = Setting(False, convert=_direction, doc="Acts on setpoint - feedback (True) or feedback - setpoint")
    actualValue = Input(doc="The measured value (feedback)")
    setpointValue = Input(doc="The setpoint")
    inputSignal = Output(init=0.0, doc="The control signal, between output_min and output_max")
    memory = State(size=2, init=0.0, names=["integral", "err_prev"], doc="The integral and the last error")
    kp = Parameter(0.001, lb=0.001, ub=10.0, scaling="log", doc="Proportional gain")
    Ti = Parameter(10.0, lb=0.1, ub=10000.0, scaling="log", doc="Integral time [s]")
    Td = Parameter(0.0, doc="Derivative time [s]")
    output_min = Parameter(0.0, lb=0.0, ub=1.0, doc="Lower limit of the control signal")
    output_max = Parameter(1.0, lb=0.0, ub=1.0, doc="Upper limit of the control signal")

    @property
    def isReverse(self):
        """Deprecated alias of ``is_reverse`` (until 2.1).  It reads and
        writes the same setting."""
        return self.is_reverse

    @isReverse.setter
    def isReverse(self, value) -> None:
        self.is_reverse = value

    @staticmethod
    def asymptotic_smooth_saturation(
        u,
        lower=0.0,
        upper=1.0,
        eps=0,
        curve_start=0.01,
        steepness=1,
        curve_type="power",
        power_exp=0.5,
    ):
        """Deprecated alias.  Delegates to :func:`clamp` with ``mode="smooth"``."""
        return smooth_saturation(
            u,
            lower=lower,
            upper=upper,
            eps=eps,
            curve_start=curve_start,
            steepness=steepness,
            curve_type=curve_type,
            power_exp=power_exp,
        )

    @staticmethod
    def hardclamp_smooth_grad(
        u,
        lower=0.0,
        upper=1.0,
        eps=0,
        curve_start=0.05,
        steepness=1,
        curve_type="power",
        power_exp=0.5,
    ):
        """Deprecated.  Use ``clamp(..., mode="hard")`` after a smooth
        warm-start instead.  See module docstring of
        :mod:`twin4build.systems.utils.smooth_saturation` for the
        recommended two-stage workflow.
        """
        return hardclamp_smooth_grad(
            u,
            lower=lower,
            upper=upper,
            eps=eps,
            curve_start=curve_start,
            steepness=steepness,
            curve_type=curve_type,
            power_exp=power_exp,
        )

    def _compute_pid_coefficients(self, kp, Ti, Td, step_size):
        """The gains per step: proportional ``kp``, integral ``kp * dt / Ti``
        and derivative ``kp * Td / dt`` (computed once per parameter set,
        not once per step)."""
        return kp, kp * step_size / Ti, kp * Td / step_size

    def derive(self, p, step_size):
        """The gains per step, from the parameters and the step size."""
        return {"coefficients": self._compute_pid_coefficients(p["kp"], p["Ti"], p["Td"], step_size)}

    def forward(self, x, inputs, p, step_size):
        """One step: ``x`` is the memory ``(n, 2)`` = ``[integral, err_prev]``;
        returns the next memory and ``{"inputSignal"}``."""
        kp, ki, kd = p["coefficients"]
        err = inputs["setpointValue"] - inputs["actualValue"]
        # one direction per copy on a batched PID
        err = torch.where(p["is_reverse"].to(device=err.device), err, -err)
        integral, err_prev = x[..., 0], x[..., 1]
        lower, upper = p["output_min"], p["output_max"]
        p_d = kp * err + kd * (err - err_prev)
        integral = torch.minimum(
            torch.maximum(integral + ki * err, torch.minimum(integral, lower - p_d)),
            torch.maximum(integral, upper - p_d),
        )
        u = clamp(p_d + integral, lower=lower, upper=upper)
        return torch.stack([integral, err], dim=-1), {"inputSignal": u}


