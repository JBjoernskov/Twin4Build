# Third party imports
import torch

# Local application imports
import twin4build.core as core
from twin4build.systems.saref4syst.declarations import Input, Output, Parameter
from twin4build.translator.translator import (
    StepRule,
    AnyPathRule,
    Node,
    OptionalRule,
    SignaturePattern,
    PathRule,
)


class ValveSystem(core.System):
    r"""
    A valve system model implemented with PyTorch for gradient-based optimization.

    This model represents a valve that controls water flow rate based on valve position.
    The valve characteristic is modeled using the valve authority equation, which provides
    a more accurate representation of the valve's behavior compared to a simple linear
    relationship.

    Args:
        waterFlowRateMax: Maximum water flow rate [kg/s]
        valveAuthority: Valve authority (0-1)
        **kwargs: Additional keyword arguments

    Mathematical Formulation
    ------------------------

    The valve characteristic is calculated using the valve authority equation:

        .. math::

            u_{norm} = \frac{u}{\sqrt{u^2 (1-a) + a}}

    where:
       - :math:`u` is the valve position (0-1)
       - :math:`a` is the valve authority (0-1)
       - :math:`u_{norm}` is the normalized valve position

    The water flow rate is then calculated as:

        .. math::

            \dot{m}_w = u_{norm} \cdot \dot{m}_{w,max}

    where:
       - :math:`\dot{m}_w` is the water flow rate [kg/s]
       - :math:`\dot{m}_{w,max}` is the maximum water flow rate [kg/s]

    Notes
    -----
    Valve Authority Characteristics:
       This is the installed characteristic of an inherently linear valve with
       authority :math:`a`:

       - :math:`a = 1` (default): :math:`u_{norm} = u`, i.e. an ideal linear
         characteristic where flow rate is directly proportional to valve position
       - :math:`a \rightarrow 0`: the flow saturates rapidly — even small openings
         yield close to full flow (quick-opening distortion), so the valve loses
         control over most of its travel
       - :math:`0 < a < 1`: intermediate distortion of the linear characteristic

    Implementation Details:
       - The model assumes ideal valve behavior (no hysteresis or deadband)
    """

    valvePosition = Input(doc="Valve position (0-1)")
    valvePosition = Output(doc="Valve position (0-1)")
    waterFlowRate = Output(doc="Water flow rate [kg/s]")
    # 1000 W of heating when the water cools from 60 to 45 degrees.
    waterFlowRateMax = Parameter(
        1000 / ((60 - 45) * 4180), lb=5e-3, ub=1.0, relative=(0.25, 4.0), scaling="log",
        doc="Maximum water flow rate [kg/s]",
    )
    valveAuthority = Parameter(1.0, lb=0.3, ub=1.0, doc="Valve authority (0-1); 1 is linear")

    def forward(self, x, inputs, p, step_size):
        """Stateless: ``(inputs, p) -> outputs``."""
        vp = inputs["valvePosition"]
        a = p["valveAuthority"]
        u_norm = vp / torch.sqrt(vp**2 * (1 - a) + a)
        return x, {"valvePosition": vp, "waterFlowRate": u_norm * p["waterFlowRateMax"]}


# Deprecated aliases (removed in twin4build 2.1)
ValveTorchSystem = ValveSystem
