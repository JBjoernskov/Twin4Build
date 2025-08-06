# Standard library imports
import datetime
from typing import Optional

# Local application imports
import twin4build.core as core
import twin4build.utils.types as tps
from twin4build.translator.translator import (
    Exact,
    MultiPath,
    Node,
    Optional_,
    SignaturePattern,
    SinglePath,
)


def get_signature_pattern():
    """Get the signature pattern for the supply flow junction system.

    Returns:
        SignaturePattern: The signature pattern defining the system's connections.
    """
    node0 = Node(cls=core.namespace.S4BLDG.FlowJunction)  # flow junction
    node1 = Node(cls=core.namespace.S4BLDG.Damper)  # damper
    node2 = Node(
        cls=(
            core.namespace.S4BLDG.Coil,
            core.namespace.S4BLDG.AirToAirHeatRecovery,
            core.namespace.S4BLDG.Fan,
        )
    )  # building space
    sp = SignaturePattern(
        semantic_model_=core.ontologies,
        ownedBy="SupplyFlowJunctionSystem",
        priority=160,
    )
    sp.add_triple(
        MultiPath(
            subject=node0, object=node1, predicate=core.namespace.FSO.suppliesFluidTo
        )
    )
    sp.add_triple(
        SinglePath(
            subject=node0, object=node2, predicate=core.namespace.FSO.hasFluidSuppliedBy
        )
    )
    sp.add_input("airFlowRateOut", node1, "airFlowRate")
    sp.add_modeled_node(node0)
    return sp


class SupplyFlowJunctionSystem(core.System):
    r"""
    A supply flow junction system model for combining air flow rates.

    This model represents a junction that combines multiple air flow rates into
    a single flow rate. It sums all input flow rates and can apply an optional
    bias to the total flow rate. This is typically used in air handling units
    to combine flows from different branches.

    Mathematical Formulation
    -----------------------

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

    Args:
        airFlowRateBias (float, optional): Bias to be added to the total flow rate [kg/s].
            Defaults to 0.
        **kwargs: Additional keyword arguments passed to the parent System class.
    """

    sp = [get_signature_pattern()]

    def __init__(self, airFlowRateBias=None, **kwargs):
        super().__init__(**kwargs)
        if airFlowRateBias is not None:
            self.airFlowRateBias = airFlowRateBias
        else:
            self.airFlowRateBias = 0

        self.input = {"airFlowRateOut": tps.Vector()}
        self.output = {"airFlowRateIn": tps.Scalar()}
        self._config = {"parameters": ["airFlowRateBias"]}

    @property
    def config(self):
        """Get the configuration parameters.

        Returns:
            dict: Dictionary containing configuration parameters.
        """
        return self._config

    def initialize(
        self,
        startTime: datetime.datetime,
        endTime: datetime.datetime,
        stepSize: int,
        simulator: core.Simulator,
    ) -> None:
        """Initialize the supply flow junction system.

        This method is a no-op as the supply flow junction system does not require initialization.
        The system has no internal state to initialize and performs a simple summation
        of input flow rates with an optional bias.

        Args:
            startTime (datetime.datetime): Start time of the simulation period.
            endTime (datetime.datetime): End time of the simulation period.
            stepSize (int): Time step size in seconds.
            simulator (core.Simulator): Simulation model object.
        """
        pass

    def do_step(
        self,
        secondTime: float,
        dateTime: datetime.datetime,
        stepSize: int,
        stepIndex: int,
    ) -> None:
        """Perform one simulation step.

        This method sums all input flow rates and adds the bias to calculate
        the total flow rate. The input flow rates are provided as a vector,
        and the output is a scalar representing the total flow rate.

        Args:
            secondTime (float, optional): Current simulation time in seconds.
            dateTime (datetime, optional): Current simulation date and time.
            stepSize (float, optional): Time step size in seconds.
            stepIndex (int, optional): Current simulation step index.
        """
        self.output["airFlowRateIn"].set(
            (self.input["airFlowRateOut"].get().sum()) + self.airFlowRateBias, stepIndex
        )
