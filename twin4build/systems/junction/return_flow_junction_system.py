# Third party imports
import torch

# Local application imports
import twin4build.core as core
from twin4build.systems.saref4syst.declarations import Input, Output, Setting


class ReturnFlowJunctionSystem(core.System):
    r"""
    A return flow junction system model for combining air flow rates and temperatures.

    This model represents a junction that combines multiple return air flows and their temperatures
    into a single output flow and temperature. The total output flow is the sum of all input flows
    (plus an optional bias), and the output temperature is the flow-weighted average of the input temperatures.

    Args:
        airFlowRateBias: Bias to be added to the total flow rate [kg/s].

    Mathematical Formulation
    ========================

    The total output flow rate is:

        .. math::

            \dot{m}_{out} = \sum_{i=1}^{n} \dot{m}_i + b

    where:
       - :math:`\dot{m}_{out}` is the total output flow rate [kg/s]
       - :math:`\dot{m}_i` are the individual input flow rates [kg/s]
       - :math:`n` is the number of input flows
       - :math:`b` is the optional flow rate bias [kg/s]

    The output temperature is the flow-weighted average:

        .. math::

            T_{out} = \frac{\sum_{i=1}^{n} T_i \dot{m}_i}{\dot{m}_{out}}

    where:
       - :math:`T_{out}` is the output temperature [°C]
       - :math:`T_i` are the input temperatures [°C]
       - :math:`\dot{m}_i` are the input flow rates [kg/s]
       - :math:`\dot{m}_{out}` is the total output flow rate [kg/s]
    """

    airFlowRateBias = Setting(0.0, doc="Added to the total flow [kg/s]")
    airFlowRateIn = Input(size="wired", doc="The branches' flows [kg/s]")
    airTemperatureIn = Input(size=airFlowRateIn, doc="The branches' temperatures [degC], a slot per branch")
    airFlowRateOut = Output(doc="The total flow [kg/s]")
    airTemperatureOut = Output(doc="The flow-weighted temperature [degC]")

    #: The streams' sum and their flow-weighted temperature (:mod:`twin4build.model.signal_dependencies`): every slot.
    OUTPUT_DEPENDENCIES = {
        "airFlowRateOut": {"inputs": ("airFlowRateIn",)},
        "airTemperatureOut": {"inputs": ("airFlowRateIn", "airTemperatureIn")},
    }

    @property
    def n_input_ports(self) -> int:
        """The number of branches."""
        return self.input["airFlowRateIn"]._n_v

    @n_input_ports.setter
    def n_input_ports(self, n: int) -> None:
        # A junction owned by a composite has no connections: its owner sets
        # the width.
        self.input["airFlowRateIn"]._n_v = int(n)

    def forward(self, x, inputs, p, step_size):
        """Stateless: the total flow (plus the bias) and the flow-weighted
        temperature (20 degC without flow)."""
        flows = inputs["airFlowRateIn"]
        m_dot_in = flows.sum(dim=-1)
        Q_dot_in = (inputs["airTemperatureIn"] * flows).sum(dim=-1)
        tol = 1e-5
        has_flow = m_dot_in > tol
        flow_rate_out = m_dot_in + p["airFlowRateBias"]
        temp_out_flow = Q_dot_in / torch.clamp(flow_rate_out, min=tol)
        return x, {
            "airFlowRateOut": torch.where(has_flow, flow_rate_out, torch.zeros_like(m_dot_in)),
            "airTemperatureOut": torch.where(has_flow, temp_out_flow, torch.full_like(m_dot_in, 20.0)),
        }

