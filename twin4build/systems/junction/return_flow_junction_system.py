# Standard library imports
import datetime
from typing import Optional

# Third party imports
import numpy as np

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
    node0 = Node(cls=core.namespace.S4BLDG.FlowJunction)  # flow junction
    node1 = Node(cls=core.namespace.S4BLDG.Damper)  # damper
    node2 = Node(cls=core.namespace.S4BLDG.BuildingSpace)  # building space
    sp = SignaturePattern(
        semantic_model_=core.ontologies,
        ownedBy="ReturnFlowJunctionSystem",
        priority=160,
    )
    sp.add_triple(
        MultiPath(
            subject=node0, object=node1, predicate=core.namespace.FSO.hasFluidReturnedBy
        )
    )
    sp.add_triple(
        Exact(
            subject=node1, object=node2, predicate=core.namespace.FSO.hasFluidReturnedBy
        )
    )

    sp.add_input("airFlowRateIn", node1, "airFlowRate")
    sp.add_input("airTemperatureIn", node2, "indoorTemperature")
    # sp.add_input("inletAirTemperature", node15, ("outletAirTemperature", "primaryTemperatureOut", "outletAirTemperature"))
    sp.add_modeled_node(node0)
    # cs.add_parameter("globalIrradiation", node2, "globalIrradiation")
    return sp


class ReturnFlowJunctionSystem(core.System):
    r"""
    A return flow junction system model for combining air flow rates and temperatures.

    This model represents a junction that combines multiple return air flows and their temperatures
    into a single output flow and temperature. The total output flow is the sum of all input flows
    (plus an optional bias), and the output temperature is the flow-weighted average of the input temperatures.

    Mathematical Formulation
    -----------------------

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

        self.input = {
            "airFlowRateIn": tps.Vector(),
            "airTemperatureIn": tps.Vector(),
        }
        self.output = {
            "airFlowRateOut": tps.Scalar(),
            "airTemperatureOut": tps.Scalar(),
        }
        self._config = {"parameters": ["airFlowRateBias"]}

    @property
    def config(self):
        return self._config

    def initialize(
        self,
        startTime: datetime.datetime,
        endTime: datetime.datetime,
        stepSize: int,
        simulator: core.Simulator,
    ) -> None:
        pass

    def do_step(
        self,
        secondTime: float,
        dateTime: datetime.datetime,
        stepSize: int,
        stepIndex: int,
    ) -> None:
        with np.errstate(invalid="raise"):
            m_dot_in = self.input["airFlowRateIn"].get().sum()
            Q_dot_in = (
                self.input["airTemperatureIn"].get() * self.input["airFlowRateIn"].get()
            )
            tol = 1e-5
            if m_dot_in > tol:
                self.output["airFlowRateOut"].set(
                    m_dot_in + self.airFlowRateBias, stepIndex
                )
                self.output["airTemperatureOut"].set(
                    Q_dot_in.sum() / self.output["airFlowRateOut"].get(), stepIndex
                )
            else:
                self.output["airFlowRateOut"].set(0, stepIndex)
                self.output["airTemperatureOut"].set(20, stepIndex)
