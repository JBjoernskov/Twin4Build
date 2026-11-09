# Local application imports
import twin4build.core as core
from twin4build.systems.saref4syst.declarations import Input, Output, Setting


class SupplyFlowJunctionSystem(core.System):
    r"""
    A supply flow junction system model for combining air flow rates.

    This model represents a junction that combines multiple air flow rates into
    a single flow rate. It sums all input flow rates and can apply an optional
    bias to the total flow rate. This is typically used in air handling units
    to combine flows from different branches.

    Args:
        airFlowRateBias: Bias to be added to the total flow rate [kg/s].

    Mathematical Formulation
    ========================

    The total flow rate is calculated as the sum of all input flow rates plus an optional bias:

        .. math::

            \dot{m}_{total} = \sum_{i=1}^{n} \dot{m}_i + b

    where:
       - :math:`\dot{m}_{total}` is the total flow rate [kg/s]
       - :math:`\dot{m}_i` are the individual input flow rates [kg/s]
       - :math:`n` is the number of input flows
       - :math:`b` is the optional flow rate bias [kg/s]

    The bias term can be used to account for:
       - Measurement errors
       - Leakage
       - System losses
       - Calibration offsets


    """

    airFlowRateBias = Setting(0.0, doc="Added to the total flow [kg/s]")
    airFlowRateOut = Input(size="wired", doc="The branches' flows [kg/s]")
    airFlowRateIn = Output(doc="The total flow [kg/s]")

    #: The sum of the branches (:mod:`twin4build.model.signal_dependencies`): every slot.
    OUTPUT_DEPENDENCIES = {"airFlowRateIn": {"inputs": ("airFlowRateOut",)}}

    @property
    def n_input_ports(self) -> int:
        """The number of branches."""
        return self.input["airFlowRateOut"]._n_v

    @n_input_ports.setter
    def n_input_ports(self, n: int) -> None:
        # A junction owned by a composite has no connections: its owner sets
        # the width.
        self.input["airFlowRateOut"]._n_v = int(n)

    def forward(self, x, inputs, p, step_size):
        """Stateless: the total flow plus the bias."""
        return x, {"airFlowRateIn": inputs["airFlowRateOut"].sum(dim=-1) + p["airFlowRateBias"]}

