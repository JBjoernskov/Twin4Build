# Standard library imports
import datetime
from typing import List

# Third party imports
import numpy as np
import torch
import torch.nn as nn

# Local application imports
import twin4build.core as core
import twin4build.utils.types as tps
from twin4build.systems.utils.smooth_saturation import (
    clamp,
    hardclamp_smooth_grad,
    smooth_saturation,
)
from twin4build.translator.translator import (
    StepRule,
    Node,
    SignaturePattern,
)

# Define @profile decorator for line_profiler (no-op if not available)
# This allows the code to work both with kernprof and programmatic LineProfiler
try:
    # Check if profile is defined in builtins (injected by kernprof)
    if isinstance(__builtins__, dict):
        profile = __builtins__.get("profile")
    else:
        profile = getattr(__builtins__, "profile", None)
    if profile is None:
        raise AttributeError
except (KeyError, AttributeError, TypeError):
    # If not available, define as no-op
    def profile(func):
        """No-op decorator when line_profiler is not active."""
        return func


class PIDControllerSystem(core.System, nn.Module):
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
        isReverse: Boolean flag to indicate if the controller is reverse
    """

    def __init__(
        self,
        kp=0.001,
        Ti=10,
        Td=0.0,
        output_min=0.0,
        output_max=1.0,
        is_reverse=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        nn.Module.__init__(self)
        self.is_reverse = is_reverse

        kp = abs(kp)
        Ti = abs(Ti)
        Td = abs(Td)

        self.kp = tps.Parameter(
            torch.tensor(kp, dtype=tps.float_dtype()),
            min_value=0.001,
            max_value=10.0,
            requires_grad=False,
            scaling="log",
        )
        self.Ti = tps.Parameter(
            torch.tensor(Ti, dtype=tps.float_dtype()),
            min_value=0.1,
            max_value=10000.0,
            requires_grad=False,
            scaling="log",
        )
        self.Td = tps.Parameter(
            torch.tensor(Td, dtype=tps.float_dtype()), requires_grad=False
        )

        self.output_min = tps.Parameter(
            torch.tensor(output_min, dtype=tps.float_dtype()),
            min_value=0.0,
            max_value=1.0,
            requires_grad=False,
        )
        self.output_max = tps.Parameter(
            torch.tensor(output_max, dtype=tps.float_dtype()),
            min_value=0.0,
            max_value=1.0,
            requires_grad=False,
        )

        self.input = {"actualValue": tps.Scalar(), "setpointValue": tps.Scalar()}
        self.output = {"inputSignal": tps.Scalar(0)}
        # The PID's memory as a first-class state (width 2): [integral,
        # err_prev].  Zero initial condition.
        self._state = tps.State(
            n_v=2, init_value=0.0,
            names=[f"{self.id}.integral", f"{self.id}.err_prev"],
        )
        self._config = {
            "parameters": ["kp", "Ti", "Td", "output_min", "output_max", "is_reverse"]
        }

    @property
    def config(self):
        return self._config

    @property
    def is_reverse(self):
        """The direction of action: ``True`` acts on ``setpoint - feedback``,
        ``False`` on ``feedback - setpoint``.  A bool, or a bool tensor with
        one direction per instance on a batched controller
        (``Model.batch_components``); a list or array of one value reads as a
        bool."""
        return self.__dict__["_is_reverse"]

    @is_reverse.setter
    def is_reverse(self, value) -> None:
        if not isinstance(value, (torch.Tensor, list, tuple, np.ndarray)):
            self.__dict__["_is_reverse"] = bool(value)
            return
        value = torch.as_tensor(value).detach().reshape(-1).to(torch.bool)
        self.__dict__["_is_reverse"] = bool(value[0]) if value.numel() == 1 else value

    @property
    def isReverse(self) -> bool:
        """Deprecated alias of ``is_reverse`` (until 2.1).  It reads and
        writes the same flag, so the action written by a rewire or restored
        from a serialized model is the one every reader sees."""
        return self.is_reverse

    @isReverse.setter
    def isReverse(self, value: bool) -> None:
        self.is_reverse = bool(value)

    def initialize(
        self,
        start_time: List[datetime.datetime],
        end_time: List[datetime.datetime],
        step_size: int,
    ) -> None:
        _, _, max_timesteps, _ = core.Simulator.get_simulation_timesteps(
            start_time, end_time, step_size
        )
        batch_size = len(start_time)
        self.input["actualValue"].initialize(
            n_t=max_timesteps,
            n_s=batch_size,
            n_c=self.n_c,
        )
        self.input["setpointValue"].initialize(
            n_t=max_timesteps,
            n_s=batch_size,
            n_c=self.n_c,
        )
        self.output["inputSignal"].initialize(
            n_t=max_timesteps,
            n_s=batch_size,
            n_c=self.n_c,
        )

        # Expand parameters to n_c dimension for vectorization
        self.kp = self.kp.expand_to_n_c(self.n_c)
        self.Ti = self.Ti.expand_to_n_c(self.n_c)
        self.Td = self.Td.expand_to_n_c(self.n_c)
        self.output_min = self.output_min.expand_to_n_c(self.n_c)
        self.output_max = self.output_max.expand_to_n_c(self.n_c)

        # Allocate the PID state (n_s, n_c, 2), zero initial value.
        self._state.initialize(n_s=batch_size, n_c=self.n_c, n_v=2, force=True)

        # Cache step_size as tensor to avoid creating it every step
        # step_size may be a list with one value per batch element, so unsqueeze(1) gives shape (batch, 1)
        self._step_size_tensor = torch.tensor(
            step_size, dtype=tps.float_dtype(), requires_grad=False
        ).unsqueeze(1)

        # Drop per-params forward caches: a fresh simulation must not reuse
        # coefficients (or their autograd graph) from a previous run.
        self._fwd_coef_cache = None
        self._forward_params_cache = None

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

    def do_step(
        self,
        second_time: float,
        date_time: datetime.datetime,
        step_size: int,
        step_index: int,
    ) -> None:
        """Thin port-I/O wrapper around :meth:`forward` (the single source of
        truth for the PID math).  ``forward``'s identity-keyed
        coefficient cache replaces the old per-attribute caching: the params
        dict from ``_forward_params`` and ``self._step_size_tensor`` are both
        identity-stable across steps, so the coefficients are recomputed only
        when a parameter actually changes."""
        inputs = {
            "setpointValue": self.input["setpointValue"].get(),
            "actualValue": self.input["actualValue"].get(),
        }
        x_next, outs = self.forward(
            self._state.get(),  # (n_s, n_c, 2) = [integral, err_prev]
            inputs,
            self._forward_params(),
            self._step_size_tensor,
        )
        self._state.set(x_next)
        self.output["inputSignal"]._set(outs["inputSignal"], i_t=step_index)

    # Continuous state (the memory [integral, err_prev]) is
    # the ``tps.State`` ``self._state``; get/set/enumeration come from the System
    # base class generically.

    #: Physical parameters, in a fixed order (the ``forward`` theta contract).
    SUPPORTS_TRANSFORM_MODE = True
    PARAM_NAMES = ("kp", "Ti", "Td", "output_min", "output_max")

    def forward(self, x, inputs, params, sample_time, transform_mode=None):
        """Pure one-step PID ``(state, inputs, params) -> (new_state, outputs)``.

        Functorch-compatible re-expression of :meth:`do_step`.  ``x`` is the memory
        ``(n_c, 2)`` = ``[integral, err_prev]``; ``inputs`` provides
        ``setpointValue`` / ``actualValue``; ``params`` a dict for
        :attr:`PARAM_NAMES`.  Returns ``(x_next, {"inputSignal"})``.
        """
        # Params-only coefficients, cached per params-dict identity (computed
        # once per theta in a sequential rollout, not once per step).  The
        # sample-time check is by identity too: ``do_step`` passes the stable
        # ``_step_size_tensor`` (a ``!=`` on a batched tensor would be
        # ambiguous), the composer a stable float.
        if transform_mode:
            coefficients = self._compute_pid_coefficients(
                params["kp"], params["Ti"], params["Td"], sample_time
            )
        else:
            cache = getattr(self, "_fwd_coef_cache", None)
            if cache is None or cache[0] is not params or cache[1] is not sample_time:
                cache = (
                    params,
                    sample_time,
                    self._compute_pid_coefficients(
                        params["kp"], params["Ti"], params["Td"], sample_time
                    ),
                )
                self._fwd_coef_cache = cache
            coefficients = cache[2]
        kp, ki, kd = coefficients
        err = inputs["setpointValue"] - inputs["actualValue"]
        reverse = self.is_reverse
        if isinstance(reverse, torch.Tensor):
            # a batched controller: one direction per instance
            err = torch.where(reverse.to(device=err.device), err, -err)
        elif reverse is False:
            err = -err
        integral, err_prev = x[..., 0], x[..., 1]
        lower, upper = params["output_min"], params["output_max"]
        p_d = kp * err + kd * (err - err_prev)
        # conditional integration: toward saturating the output and no
        # further; held while the output saturates on the error's side
        integral = torch.minimum(
            torch.maximum(integral + ki * err, torch.minimum(integral, lower - p_d)),
            torch.maximum(integral, upper - p_d),
        )
        u = clamp(p_d + integral, lower=lower, upper=upper)
        return torch.stack([integral, err], dim=-1), {"inputSignal": u}


