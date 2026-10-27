# Standard library imports
import datetime
from math import comb
from typing import Optional, Sequence

# Third party imports
import numpy as np
import torch
import torch.nn as nn

# Local application imports
import twin4build.core as core
import twin4build.utils.constants as constants
import twin4build.utils.types as tps
from twin4build.systems.utils.smooth_saturation import clamp


def polynomial_to_bernstein(
    nominalPowerRate, nominalAirFlowRate, c: Sequence, lower, upper
) -> list:
    r"""The Bernstein coefficients :math:`\beta_0 \dots \beta_3` [W] over the
    flow range ``[lower, upper]`` [kg/s] of the polynomial fan curve

    .. math::

        P = P_{nom}\,(c_1 + c_2 x + c_3 x^2 + c_4 x^3), \qquad x = \dot m / \dot m_{nom}

    (``c`` = ``(c1, c2, c3, c4)``).  The two forms are the same cubic, so the
    curves agree at every flow, inside the range or not.  Plain arithmetic:
    floats, arrays and tensors alike.
    """
    x0 = lower / nominalAirFlowRate
    h = (upper - lower) / nominalAirFlowRate
    # the monomials in t = (m - lower) / (upper - lower), from x = x0 + h t
    a = [nominalPowerRate * sum(c[j] * comb(j, k) * x0 ** (j - k) * h**k for j in range(k, 4)) for k in range(4)]
    return [sum(comb(i, k) / comb(3, k) * a[k] for k in range(i + 1)) for i in range(4)]


def bernstein_to_polynomial(
    beta: Sequence, lower, upper, nominalPowerRate=1.0, nominalAirFlowRate=1.0
) -> list:
    r"""The coefficients ``(c1, c2, c3, c4)`` of the polynomial fan curve
    (:func:`polynomial_to_bernstein`) with the given ``nominalPowerRate`` [W]
    and ``nominalAirFlowRate`` [kg/s] that is the same cubic as the Bernstein
    coefficients ``beta`` [W] over ``[lower, upper]`` [kg/s].  The polynomial
    form's nominal values only scale its coefficients: any choice gives the
    same curve.  The inverse of :func:`polynomial_to_bernstein`.
    """
    b0, b1, b2, b3 = beta
    a = [b0, 3 * (b1 - b0), 3 * (b2 - 2 * b1 + b0), b3 - 3 * b2 + 3 * b1 - b0]
    # t = u0 + s x for x = m / m_nom
    s = nominalAirFlowRate / (upper - lower)
    u0 = -lower / (upper - lower)
    return [sum(a[k] * comb(k, j) * u0 ** (k - j) * s**j for k in range(j, 4)) / nominalPowerRate for j in range(4)]


class FanSystem(core.System, nn.Module):
    r"""
    A fan system model implemented with PyTorch for gradient-based optimization.

    This model represents a fan that controls air flow rate and temperature, considering
    both the power consumption and the heat added to the air stream.

    Args:
        beta0 : Power at the bottom of the flow range [W]
        beta1 : First inner control point of the power curve [W]
        beta2 : Second inner control point of the power curve [W]
        beta3 : Power at the top of the flow range [W]
        flowRangeLower : Bottom of the operating flow range [kg/s] (fixed)
        flowRangeUpper : Top of the operating flow range [kg/s] (fixed)
        f_total : Total fan efficiency factor (0-1)

    Mathematical Formulation
    ------------------------

    The fan power is a cubic in the air flow, written in Bernstein form over
    the fan's operating flow range :math:`[\dot{m}_{lo}, \dot{m}_{hi}]`:

        .. math::

            t = \frac{\dot{m} - \dot{m}_{lo}}{\dot{m}_{hi} - \dot{m}_{lo}}, \qquad
            P = \beta_0 (1-t)^3 + 3\beta_1 t(1-t)^2 + 3\beta_2 t^2(1-t) + \beta_3 t^3

    where:
       - :math:`P` is the fan power [W]
       - :math:`\dot{m}` is the air mass flow rate [kg/s]
       - :math:`\beta_0` and :math:`\beta_3` are the powers at the bottom and the top
         of the range, :math:`\beta_1` and :math:`\beta_2` shape the curve between them

    On the range the four weights are non-negative and sum to one, so the
    power lies between the smallest and the largest :math:`\beta`: the class
    bound :math:`\beta_k \ge 0` keeps the curve non-negative over the whole
    operating range, a constraint on the curve rather than on its output.
    The polynomial form :math:`P_{nom}(c_1 + c_2 x + c_3 x^2 + c_4 x^3)`,
    :math:`x = \dot m / \dot m_{nom}`, is the same cubic in other coordinates:
    :func:`polynomial_to_bernstein` and :func:`bernstein_to_polynomial` map
    between the two exactly.  The range is fixed (not estimated): a site
    sets it to the flows its fans run at (:meth:`set_flow_range`, which keeps
    the curve).

    The outlet air temperature is calculated considering the heat added by the fan:

        .. math::

            T_{out} = T_{in} + \frac{P \cdot f_{total}}{\dot{m} \cdot c_p}

    where:
       - :math:`T_{out}` is the outlet temperature [°C]
       - :math:`T_{in}` is the inlet temperature [°C]
       - :math:`f_{total}` is the fraction of power that is converted to heat and added to the air stream
       - :math:`c_p` is the specific heat capacity of air [J/(kg·K)]

    Notes
    -----
    Model Assumptions:
       - Fan power follows a cubic in the flow; outside the operating range the
         cubic extrapolates, floored at zero there: no fan gives power back
         (exact in hard saturation, a smooth curve of :attr:`POWER_CURVE_START`
         watts in the smooth mode)
       - A fraction :math:`f_{total}` of the fan power is added as heat to the air stream
       - Constant air density and specific heat capacity
       - No mechanical losses considered separately

    Implementation Details:
       - Uses PyTorch for gradient-based optimization
       - Parameters are stored as ``tps.Parameter`` objects (``requires_grad=False``
         by default); they are not trained directly but can be calibrated via the
         Estimator
       - All calculations performed in SI units
    """

    def __init__(
        self,
        beta0: float = 0.0,
        beta1: float = 800.0 / 3.0,
        beta2: float = 600.0,
        beta3: float = 1000.0,
        flowRangeLower: float = 0.0,
        flowRangeUpper: float = 1.0,
        f_total: float = 0.9,
        **kwargs,
    ):
        """
        Initialize the fan system model.

        The default curve is 1 kW at 1 kg/s, 0.8 linear plus 0.2 square in
        the flow, over the range 0-1 kg/s.

        Args:
            beta0-beta3: Bernstein coefficients of the power curve [W]
            flowRangeLower, flowRangeUpper: Operating flow range [kg/s]
            f_total: Total fan efficiency factor
        """
        super().__init__(**kwargs)
        nn.Module.__init__(self)
        if not flowRangeUpper > flowRangeLower:
            raise ValueError(f"{kwargs.get('id')}: flow range [{flowRangeLower}, {flowRangeUpper}] is empty")

        # Store parameters as tps.Parameters for gradient tracking
        for name, value in (
            ("beta0", beta0),
            ("beta1", beta1),
            ("beta2", beta2),
            ("beta3", beta3),
            ("flowRangeLower", flowRangeLower),
            ("flowRangeUpper", flowRangeUpper),
            ("f_total", f_total),
        ):
            setattr(self, name, tps.Parameter(torch.tensor(value, dtype=tps.float_dtype()), requires_grad=False))

        # Define inputs and outputs as private variables
        self._input = {"airFlowRate": tps.Scalar(), "inletAirTemperature": tps.Scalar()}
        self._output = {"outletAirTemperature": tps.Scalar(), "Power": tps.Scalar()}

        # Define parameters for calibration.  The flow range is not one: it
        # gives the betas their meaning (a moved range is another curve for
        # the same betas), and the four betas already span every cubic.
        self.parameter = {
            # Power at the curve's control points [W].  Non-negative: with
            # every beta >= 0 the power is >= 0 over the whole range (no fan
            # gives power back), the constraint the solver sees as bounds.
            # 100 kW is beyond any single AHU's fans.  Size quantities: once
            # the fan is sized (``size_to``), each bound is 0 to 4 times its
            # sized value.
            "beta0": {"lb": 0.0, "ub": 1e5, "relative": (0.0, 4.0)},
            "beta1": {"lb": 0.0, "ub": 1e5, "relative": (0.0, 4.0)},
            "beta2": {"lb": 0.0, "ub": 1e5, "relative": (0.0, 4.0)},
            "beta3": {"lb": 0.0, "ub": 1e5, "relative": (0.0, 4.0)},
            # Total fan efficiency.  Real centrifugal AHU fans land
            # between 0.4 and 0.85; lower bound 0.3 keeps very small
            # / poorly maintained units in scope without admitting
            # the unphysical 0.
            "f_total": {"lb": 0.3, "ub": 0.9},
        }

        self._config = {"parameters": list(self.parameter.keys())}
        self.INITIALIZED = False

    @property
    def config(self):
        """Get the configuration of the fan system."""
        return self._config

    @property
    def input(self) -> dict:
        """
        Get the input ports of the fan system.

        Returns:
            dict: Dictionary containing input ports:
                - "airFlowRate": Air mass flow rate [kg/s]
                - "inletAirTemperature": Inlet air temperature [°C]
        """
        return self._input

    @property
    def output(self) -> dict:
        """
        Get the output ports of the fan system.

        Returns:
            dict: Dictionary containing output ports:
                - "outletAirTemperature": Outlet air temperature [°C]
                - "Power": Fan power consumption [W]
        """
        return self._output

    def set_flow_range(self, lower: float, upper: float) -> None:
        """Move the operating flow range to ``[lower, upper]`` [kg/s], keeping
        the curve: the betas are mapped onto the new range."""
        if not upper > lower:
            raise ValueError(f"{self.id}: flow range [{lower}, {upper}] is empty")
        with torch.no_grad():
            beta = [getattr(self, f"beta{k}").get() for k in range(4)]
            old_lo, old_hi = self.flowRangeLower.get(), self.flowRangeUpper.get()
            # watts and kg/s as the polynomial form's units: the curve itself
            c = bernstein_to_polynomial(beta, old_lo, old_hi)
            for k, value in enumerate(polynomial_to_bernstein(1.0, 1.0, c, lower, upper)):
                getattr(self, f"beta{k}").set(value, normalized=False)
            self.flowRangeLower.set(float(lower), normalized=False)
            self.flowRangeUpper.set(float(upper), normalized=False)

    def curve_power(self, m_dot: float) -> float:
        """The power curve at the flow ``m_dot`` [kg/s] [W] (the first instance's betas; no floor)."""
        beta = [float(getattr(self, f"beta{k}").get().detach().reshape(-1)[0]) for k in range(4)]
        lo = float(self.flowRangeLower.get().detach().reshape(-1)[0])
        hi = float(self.flowRangeUpper.get().detach().reshape(-1)[0])
        t = (float(m_dot) - lo) / (hi - lo)
        s = 1.0 - t
        return beta[0] * s**3 + 3.0 * beta[1] * t * s**2 + 3.0 * beta[2] * t**2 * s + beta[3] * t**3

    def size_to(self, m_dot: float, power: float) -> bool:
        """Size the fan: its curve scaled, shape kept, so it draws ``power``
        [W] at the flow ``m_dot`` [kg/s] (a fan's measured peak), and each
        beta a size quantity anchored there (``System.size_parameter``: its
        bounds 0 to 4 times the sized value; the bounds live in the fan's
        spec, so they hold when a unit that carries the fan lists them).
        Returns whether every beta was sized (a beta at zero keeps its
        absolute bounds)."""
        at = self.curve_power(m_dot)
        if not (at > 0 and power > 0):
            return False
        factor = float(power) / at
        sized = True
        for k in range(4):
            value = getattr(self, f"beta{k}").get().detach() * factor
            sized = self.size_parameter(f"beta{k}", value) and sized
        return sized

    def initialize(
        self,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
        step_size: int,
    ) -> None:
        """Initialize the fan system."""
        # Initialize I/O
        _, _, max_timesteps, _ = core.Simulator.get_simulation_timesteps(
            start_time, end_time, step_size
        )
        batch_size = len(start_time)

        if hasattr(self, "_n_c_batched") and self._n_c_batched > 1:
            self.n_c = self._n_c_batched
        else:
            self.n_c = 1

        for input in self.input.values():
            input.initialize(
                n_t=max_timesteps,
                n_s=batch_size,
                n_c=self.n_c,
            )
        for output in self.output.values():
            output.initialize(
                n_t=max_timesteps,
                n_s=batch_size,
                n_c=self.n_c,
            )

        # Expand parameters to n_c dimension for vectorization
        for name in self.PARAM_NAMES:
            setattr(self, name, getattr(self, name).expand_to_n_c(self.n_c))

        self.INITIALIZED = True

    PARAM_NAMES = (
        "beta0",
        "beta1",
        "beta2",
        "beta3",
        "flowRangeLower",
        "flowRangeUpper",
        "f_total",
    )

    #: Bound on the fan's air-temperature rise [K], see ``forward``.
    MAX_DELTA_T = 30.0
    #: Width [W] of the smooth mode's curve where the power floors at zero.
    POWER_CURVE_START = 10.0

    def forward(self, x, inputs, params, sample_time):
        """Pure one-step fan model (functorch-safe, stateless).

        Fan power from the Bernstein power curve over the flow range,
        ``t = (m - lo)/(hi - lo)``,
        ``P = b0 (1-t)^3 + 3 b1 t (1-t)^2 + 3 b2 t^2 (1-t) + b3 t^3``,
        outlet temperature from the heat added to the air stream:
        ``T_out = T_in + (P * f_total) / (m * c_p)``.
        """
        m_dot = inputs["airFlowRate"]
        inlet_temp = inputs["inletAirTemperature"]

        # Position in the operating range (outside it the cubic extrapolates)
        lo = params["flowRangeLower"]
        t = (m_dot - lo) / (params["flowRangeUpper"] - lo)
        s = 1.0 - t

        # Calculate fan power from the Bernstein form
        power = (
            params["beta0"] * s**3
            + 3.0 * params["beta1"] * t * s**2
            + 3.0 * params["beta2"] * t**2 * s
            + params["beta3"] * t**3
        )
        # No fan gives power back.  Non-negative betas keep the curve >= 0 on
        # the range; this floor guards the extrapolation outside it.
        power = clamp(
            power, lower=0.0, upper=1e12, curve_start=self.POWER_CURVE_START,
            steepness=2.0, curve_type="power", power_exp=0.5,
        )

        # Calculate temperature rise.  Zero flow (dampers closed, fan off)
        # is a normal state: guard the division so the outlet temperature
        # stays finite (no air moves, so no heating of the air).
        tol = 1e-9
        has_flow = m_dot > tol
        delta_T = torch.where(
            has_flow,
            (power * params["f_total"])
            / (torch.clamp(m_dot, min=tol) * constants.CP_AIR),
            torch.zeros_like(m_dot),
        )
        # A trickle of air (the fan running against closed dampers) would
        # otherwise take the whole fan power and leave at thousands of
        # degrees; no fan heats its air stream by more than a few tens of K.
        delta_T = torch.clamp(delta_T, -self.MAX_DELTA_T, self.MAX_DELTA_T)
        outlet_temp = inlet_temp + delta_T
        return x, {"outletAirTemperature": outlet_temp, "Power": power}

    def do_step(
        self,
        second_time: float,
        date_time: datetime.datetime,
        step_size: int,
        step_index: int,
    ) -> None:
        """
        Perform one step of the fan system simulation.

        Thin port-I/O wrapper delegating the math to :meth:`forward`.
        """
        inputs = {
            "airFlowRate": self.input["airFlowRate"].get(),
            "inletAirTemperature": self.input["inletAirTemperature"].get(),
        }
        _, outs = self.forward(
            None, inputs, self._forward_params(), self._scalar_sample_time(step_size)
        )
        self.output["outletAirTemperature"]._set(
            outs["outletAirTemperature"], i_t=step_index
        )
        self.output["Power"]._set(outs["Power"], i_t=step_index)


# Deprecated aliases (removed in twin4build 2.1)
FanTorchSystem = FanSystem
