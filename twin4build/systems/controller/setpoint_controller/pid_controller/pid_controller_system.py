# Standard library imports
import datetime
from typing import List

# Third party imports
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

    This class implements a PID controller with a differentiable saturation function.

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
        isReverse=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        nn.Module.__init__(self)
        self.isReverse = isReverse

        kp = abs(kp)
        Ti = abs(Ti)
        Td = abs(Td)

        self.kp = tps.Parameter(
            torch.tensor(kp, dtype=torch.float64),
            min_value=0.001,
            max_value=10.0,
            requires_grad=False,
            scaling="log",
        )
        self.Ti = tps.Parameter(
            torch.tensor(Ti, dtype=torch.float64),
            min_value=0.1,
            max_value=10000.0,
            requires_grad=False,
            scaling="log",
        )
        self.Td = tps.Parameter(
            torch.tensor(Td, dtype=torch.float64), requires_grad=False
        )

        self.output_min = tps.Parameter(
            torch.tensor(output_min, dtype=torch.float64),
            min_value=0.0,
            max_value=1.0,
            requires_grad=False,
        )
        self.output_max = tps.Parameter(
            torch.tensor(output_max, dtype=torch.float64),
            min_value=0.0,
            max_value=1.0,
            requires_grad=False,
        )

        self.input = {"actualValue": tps.Scalar(), "setpointValue": tps.Scalar()}
        self.output = {"inputSignal": tps.Scalar(0)}
        # Velocity-form PID memory as a first-class state (width 3):
        # [u_prev, err_prev, err_prev_m1].  Zero initial condition.
        self._state = tps.State(
            n_v=3, init_value=0.0,
            names=[f"{self.id}.u_prev", f"{self.id}.err_prev", f"{self.id}.err_prev_m1"],
        )
        self._config = {
            "parameters": ["kp", "Ti", "Td", "output_min", "output_max", "isReverse"]
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
        self.kp = self.kp.expand_to_n_c(self.n_c)
        self.Ti = self.Ti.expand_to_n_c(self.n_c)
        self.Td = self.Td.expand_to_n_c(self.n_c)
        self.output_min = self.output_min.expand_to_n_c(self.n_c)
        self.output_max = self.output_max.expand_to_n_c(self.n_c)

        # Allocate the velocity-form PID state (n_s, n_c, 3), zero initial value.
        self._state.initialize(n_s=batch_size, n_c=self.n_c, n_v=3, force=True)

        # Cache step_size as tensor to avoid creating it every step
        # step_size may be a list with one value per batch element, so unsqueeze(1) gives shape (batch, 1)
        self._step_size_tensor = torch.tensor(
            step_size, dtype=torch.float64, requires_grad=False
        ).unsqueeze(1)

        # Cache for PID coefficients (recomputed when parameters change)
        self._cached_coeffs = None
        self._cached_kp = None
        self._cached_Ti = None
        self._cached_Td = None

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
        """
        Pre-compute PID coefficients to reduce per-step tensor operations.

        The incremental PID formula is:
            du = kp * (c0 * err + c1 * err_prev + c2 * err_prev_m1)
        where:
            c0 = 1 + dt/Ti + Td/dt
            c1 = -1 - 2*Td/dt
            c2 = Td/dt
        """
        Td_over_step = Td / step_size
        c0 = kp * (1 + step_size / Ti + Td_over_step)  # coefficient for err
        c1 = kp * (-1 - 2 * Td_over_step)  # coefficient for err_prev
        c2 = kp * Td_over_step  # coefficient for err_prev_m1
        return c0, c1, c2

    def do_step(
        self,
        second_time: float,
        date_time: datetime.datetime,
        step_size: int,
        step_index: int,
    ) -> None:
        # Get current parameter values
        kp = self.kp.get()
        Ti = self.Ti.get()
        Td = self.Td.get()

        # Recompute coefficients only if parameters changed (common during estimation)
        # Using 'is not' for tensor identity check - fast when unchanged
        params_changed = (
            self._cached_coeffs is None
            or self._cached_kp is not kp
            or self._cached_Ti is not Ti
            or self._cached_Td is not Td
        )

        if params_changed:
            self._cached_coeffs = self._compute_pid_coefficients(
                kp, Ti, Td, self._step_size_tensor
            )
            self._cached_kp = kp
            self._cached_Ti = Ti
            self._cached_Td = Td

        c0, c1, c2 = self._cached_coeffs

        # Read the velocity-form memory from state: [u_prev, err_prev, err_prev_m1].
        _st = self._state.get()  # (n_s, n_c, 3)
        u_prev = _st[..., 0]
        err_prev = _st[..., 1]
        err_prev_m1 = _st[..., 2]

        # Compute error
        err = self.input["setpointValue"].get() - self.input["actualValue"].get()
        if self.isReverse == False:
            err = -err

        # Compute control increment using pre-computed coefficients
        # This reduces multiple tensor operations to just 3 multiplications + 2 additions
        du = c0 * err + c1 * err_prev + c2 * err_prev_m1

        u = u_prev + du
        # Differentiable clamp into [output_min, output_max].  The mode
        # (``"smooth"`` vs ``"hard"``) is read from the process-global
        # toggle in ``smooth_saturation``: smooth for cold-start
        # exploration (gradient flows through deep windup), hard for
        # bias-correction refinement (forward exact at bounds, no
        # faithfulness floor).  See ``estimate_with_refinement`` for
        # the recommended two-stage workflow.
        u = clamp(
            u,
            lower=self.output_min.get(),
            upper=self.output_max.get(),
        )

        # if date_time[0].hour<5 or (date_time[0].hour==5 and date_time[0].minute==0) or date_time[0].hour>17:
        #     u = u*0

        # print("DEBUG: u after saturation: ", u)
        # Write new memory: u_prev<-u, err_prev<-err, err_prev_m1<-old err_prev.
        # All of u/err/err_prev are (n_s, n_c), so the stack is (n_s, n_c, 3).
        self._state.set(torch.stack([u, err, err_prev], dim=-1))

        self.output["inputSignal"]._set(u, i_t=step_index)

    # Continuous state (velocity-form memory [u_prev, err_prev, err_prev_m1]) is
    # the ``tps.State`` ``self._state``; get/set/enumeration come from the System
    # base class generically.

    #: Physical parameters, in a fixed order (the ``forward`` theta contract).
    PARAM_NAMES = ("kp", "Ti", "Td", "output_min", "output_max")

    def forward(self, x, inputs, params, sample_time):
        """Pure one-step velocity-form PID ``(state, inputs, params) -> (new_state, outputs)``.

        Functorch-compatible re-expression of :meth:`do_step`.  ``x`` is the memory
        ``(n_c, 3)`` = ``[u_prev, err_prev, err_prev_m1]``; ``inputs`` provides
        ``setpointValue`` / ``actualValue``; ``params`` a dict for
        :attr:`PARAM_NAMES`.  Returns ``(x_next, {"inputSignal"})``.
        """
        c0, c1, c2 = self._compute_pid_coefficients(
            params["kp"], params["Ti"], params["Td"], sample_time
        )
        err = inputs["setpointValue"] - inputs["actualValue"]
        if self.isReverse is False:
            err = -err
        u_prev, err_prev, err_prev_m1 = x[..., 0], x[..., 1], x[..., 2]
        u = u_prev + (c0 * err + c1 * err_prev + c2 * err_prev_m1)
        u = clamp(u, lower=params["output_min"], upper=params["output_max"])
        return torch.stack([u, err, err_prev], dim=-1), {"inputSignal": u}


def saref_signature_pattern():
    node0 = Node(cls=core.namespace.S4BLDG.SetpointController)
    node1 = Node(cls=core.namespace.SAREF.Sensor)
    node2 = Node(cls=core.namespace.SAREF.Property)
    node3 = Node(cls=core.namespace.S4BLDG.Schedule)
    node4 = Node(cls=core.namespace.XSD.boolean)
    sp = SignaturePattern(id="pid_controller_signature_pattern")
    sp.add_rule(
        StepRule(subject=node0, object=node2, predicate=core.namespace.SAREF.observes)
    )
    sp.add_rule(
        StepRule(subject=node1, object=node2, predicate=core.namespace.SAREF.observes)
    )
    sp.add_rule(
        StepRule(subject=node0, object=node3, predicate=core.namespace.SAREF.hasProfile)
    )
    sp.add_rule(
        StepRule(subject=node0, object=node4, predicate=core.namespace.S4BLDG.isReverse)
    )

    sp.add_input("actualValue", node1, "measuredValue")
    sp.add_input("setpointValue", node3, "scheduleValue")
    sp.add_parameter("isReverse", node4)
    sp.add_modeled_node(node0)
    return sp


PIDControllerSystem.add_signature_pattern(saref_signature_pattern())
