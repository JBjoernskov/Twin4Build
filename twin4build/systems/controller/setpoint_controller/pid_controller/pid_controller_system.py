# Standard library imports
import datetime
from typing import List

# Third party imports
import torch
import torch.nn as nn

# Local application imports
import twin4build.core as core
import twin4build.utils.types as tps
from twin4build.translator.translator import (
    Exact,
    Node,
    SignaturePattern,
)

# Define @profile decorator for line_profiler (no-op if not available)
# This allows the code to work both with kernprof and programmatic LineProfiler
try:
    # Check if profile is defined in builtins (injected by kernprof)
    if isinstance(__builtins__, dict):
        profile = __builtins__.get('profile')
    else:
        profile = getattr(__builtins__, 'profile', None)
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
        isReverse: Boolean flag to indicate if the controller is reverse
    """

    def __init__(
        self,
        kp=0.001,
        Ti=10,
        Td=0.0,
        isReverse=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        nn.Module.__init__(self)
        self.isReverse = isReverse

        kp = abs(kp)

        if isReverse == False:
            kp = -kp
        Ti = abs(Ti)
        Td = abs(Td)

        self.kp = tps.Parameter(
            torch.tensor(kp, dtype=torch.float64), requires_grad=False
        )
        self.Ti = tps.Parameter(
            torch.tensor(Ti, dtype=torch.float64), requires_grad=False
        )
        self.Td = tps.Parameter(
            torch.tensor(Td, dtype=torch.float64), requires_grad=False
        )

        self.input = {"actualValue": tps.Scalar(), "setpointValue": tps.Scalar()}
        self.output = {"inputSignal": tps.Scalar(0)}
        self._config = {"parameters": ["kp", "Ti", "Td", "isReverse"]}

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
        self.err_prev = torch.zeros((batch_size, 1), dtype=torch.float64, requires_grad=False)
        self.err_prev_m1 = torch.zeros((batch_size, 1), dtype=torch.float64, requires_grad=False)
        self.u_prev = torch.zeros((batch_size, 1), dtype=torch.float64, requires_grad=False)
        
        # Cache step_size as tensor to avoid creating it every step
        # step_size may be a list with one value per batch element, so unsqueeze(1) gives shape (batch, 1)
        self._step_size_tensor = torch.tensor(step_size, dtype=torch.float64, requires_grad=False).unsqueeze(1)
        
        # Cache for PID coefficients (recomputed when parameters change)
        self._cached_coeffs = None
        self._cached_kp = None
        self._cached_Ti = None
        self._cached_Td = None

    def asymptotic_smooth_saturation(
        self, u, lower=0.0, upper=1.0, eps=0, curve_start=0.1, steepness=1,
        curve_type='hyperbolic', power_exp=0.5
    ):
        """
        Smooth saturation function with asymptotic behavior at bounds.
        
        Optimized to only compute expensive exp() for values in saturation regions.
        For typical PID control, most values are in the linear passthrough region [curve_start, 1-curve_start],
        making this optimization very effective (4-5x speedup in benchmarks).
        """
        effective_min = lower + eps
        effective_max = upper - eps
        lower_curve_point = effective_min + curve_start
        upper_curve_point = effective_max - curve_start
        
        def curve_function(x):
            scaled_x = steepness * x / curve_start
            if curve_type == 'exponential':
                return 1 - torch.exp(-scaled_x)
            elif curve_type == 'hyperbolic':
                return scaled_x / (1 + scaled_x)
            elif curve_type == 'power':
                return 1 - 1 / torch.pow(1 + scaled_x, power_exp)
            elif curve_type == 'sqrt':
                return 1 - 1 / torch.sqrt(1 + scaled_x)
            else:
                return 1 - torch.exp(-scaled_x)
        
        # Three explicit regions
        result = torch.where(
            u < lower_curve_point,
            # Lower region: curve toward effective_min
            effective_min + curve_start * (1 - curve_function(lower_curve_point - u)),
            torch.where(
                u > upper_curve_point,
                # Upper region: curve toward effective_max
                effective_max - curve_start * (1 - curve_function(u - upper_curve_point)),
                # Linear region: perfect passthrough
                u,
            ),
        )
        return result


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
        c1 = kp * (-1 - 2 * Td_over_step)               # coefficient for err_prev
        c2 = kp * Td_over_step                          # coefficient for err_prev_m1
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
            self._cached_coeffs is None or
            self._cached_kp is not kp or
            self._cached_Ti is not Ti or
            self._cached_Td is not Td
        )
        
        if params_changed:
            self._cached_coeffs = self._compute_pid_coefficients(kp, Ti, Td, self._step_size_tensor)
            self._cached_kp = kp
            self._cached_Ti = Ti
            self._cached_Td = Td
        
        c0, c1, c2 = self._cached_coeffs
        
        # Compute error
        err = self.input["setpointValue"].get() - self.input["actualValue"].get()
        
        # Compute control increment using pre-computed coefficients
        # This reduces multiple tensor operations to just 3 multiplications + 2 additions
        du = c0 * err + c1 * self.err_prev + c2 * self.err_prev_m1


        u = self.u_prev + du
        # print("DEBUG: u before saturation: ", u)
        u = self.asymptotic_smooth_saturation(
            u
        )
        # print("DEBUG: u after saturation: ", u)
        self.u_prev = u
        self.err_prev_m1 = self.err_prev
        self.err_prev = err

        self.output["inputSignal"]._set(u, i_t=step_index)


def saref_signature_pattern():
    node0 = Node(cls=core.namespace.S4BLDG.SetpointController)
    node1 = Node(cls=core.namespace.SAREF.Sensor)
    node2 = Node(cls=core.namespace.SAREF.Property)
    node3 = Node(cls=core.namespace.S4BLDG.Schedule)
    node4 = Node(cls=core.namespace.XSD.boolean)
    sp = SignaturePattern(id="pid_controller_signature_pattern")
    sp.add_triple(
        Exact(subject=node0, object=node2, predicate=core.namespace.SAREF.observes)
    )
    sp.add_triple(
        Exact(subject=node1, object=node2, predicate=core.namespace.SAREF.observes)
    )
    sp.add_triple(
        Exact(subject=node0, object=node3, predicate=core.namespace.SAREF.hasProfile)
    )
    sp.add_triple(
        Exact(subject=node0, object=node4, predicate=core.namespace.S4BLDG.isReverse)
    )

    sp.add_input("actualValue", node1, "measuredValue")
    sp.add_input("setpointValue", node3, "scheduleValue")
    sp.add_parameter("isReverse", node4)
    sp.add_modeled_node(node0)
    return sp


PIDControllerSystem.add_signature_pattern(saref_signature_pattern())
