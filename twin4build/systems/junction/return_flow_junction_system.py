# Standard library imports
import datetime
from typing import List, Optional, Union

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

    def __init__(self, airFlowRateBias=None, **kwargs):
        super().__init__(**kwargs)
        if airFlowRateBias is not None:
            self.airFlowRateBias = airFlowRateBias
        else:
            self.airFlowRateBias = 0
        self.n_input_ports = (
            2  
        )
        self._manual_setup_n_input_ports = False

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
    def n_input_ports(self):
        return self._n_input_ports

    @n_input_ports.setter
    def n_input_ports(self, n_input_ports: int):
        self._manual_setup_n_input_ports = True
        self._n_input_ports = n_input_ports

    @property
    def config(self):
        return self._config

    def setup_variable_inputs(self):

        #Check that the number of airFlowRateIn and airTemperatureIn are the same
        connection_point_airFlowRateIn = [cp for cp in self.connects_at if cp.inputPort == "airFlowRateIn"]
        connection_point_airTemperatureIn = [cp for cp in self.connects_at if cp.inputPort == "airTemperatureIn"]
        if len(connection_point_airFlowRateIn) != len(connection_point_airTemperatureIn):
            raise ValueError("The number of airFlowRateIn and airTemperatureIn must be the same")
        
        if self._manual_setup_n_input_ports == False:
            #Assert that the number of input ports is at least 1
            connection_point = [cp for cp in self.connects_at if cp.inputPort == "airFlowRateIn"]
            if len(connection_point) == 0:
                raise ValueError("No input port found for airFlowRateIn")
            n_input_ports = len(connection_point[0].connects_system_through)
            self.n_input_ports = n_input_ports
   
   
   
    def initialize(
        self,
        start_time: Union[List[datetime.datetime], datetime.datetime],
        end_time: Union[List[datetime.datetime], datetime.datetime],
        step_size: Union[List[int], int],
    ) -> None:
        _, _, max_timesteps, _ = core.Simulator.get_simulation_timesteps(
            start_time, end_time, step_size
        )
        batch_size = len(start_time)
        self.setup_variable_inputs()
        
        
        for input in self.input.values():
            input.initialize(
                n_timesteps=max_timesteps,
                batch_size=batch_size,
                size=self.n_input_ports, #both inputs must have the same number of input ports
            )
        for output in self.output.values():
            output.initialize(
                n_timesteps=max_timesteps,
                batch_size=batch_size,
            )

    def do_step(
        self,
        second_time: float,
        date_time: datetime.datetime,
        step_size: int,
        step_index: int,
    ) -> None:
        with np.errstate(invalid="raise"):
            m_dot_in = self.input["airFlowRateIn"].get().sum()
            Q_dot_in = (
                self.input["airTemperatureIn"].get() * self.input["airFlowRateIn"].get()
            )
            tol = 1e-5
            if m_dot_in > tol:
                self.output["airFlowRateOut"].set(
                    m_dot_in + self.airFlowRateBias, step_index
                )
                self.output["airTemperatureOut"].set(
                    Q_dot_in.sum() / self.output["airFlowRateOut"].get(), step_index
                )
            else:
                self.output["airFlowRateOut"].set(0, step_index)
                self.output["airTemperatureOut"].set(20, step_index)


def saref_signature_pattern():
    """
    Get the SAREF signature pattern of the return flow junction component.

    Returns:
        SignaturePattern: The SAREF signature pattern of the return flow junction component.
    """
    node0 = Node(cls=core.namespace.S4BLDG.FlowJunction)  # flow junction
    node1 = Node(cls=core.namespace.S4BLDG.Damper)  # damper
    node2 = Node(cls=core.namespace.S4BLDG.BuildingSpace)  # building space
    sp = SignaturePattern(
        id="return_flow_junction_signature_pattern",
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


def brick_signature_pattern():
    """
    Get the BRICK signature pattern of the return flow junction component.

    Returns:
        SignaturePattern: The BRICK signature pattern of the return flow junction component.
    """
    node0 = Node(cls=core.namespace.BRICK.Air_Flow_Junction)  # flow junction
    node1 = Node(cls=core.namespace.BRICK.Damper)  # damper
    node2 = Node(cls=core.namespace.BRICK.HVAC_Zone)  # building space/zone

    sp = SignaturePattern(
        id="return_flow_junction_signature_pattern_brick",
    )
    sp.add_triple(
        Exact(subject=node1, object=node0, predicate=core.namespace.BRICK.feeds)
    )
    sp.add_triple(
        Exact(subject=node2, object=node1, predicate=core.namespace.BRICK.feeds)
    )

    sp.add_input("airFlowRateIn", node1, "airFlowRate")
    sp.add_input("airTemperatureIn", node2, "indoorTemperature")
    sp.add_modeled_node(node0)
    return sp


ReturnFlowJunctionSystem.add_signature_pattern(brick_signature_pattern())
ReturnFlowJunctionSystem.add_signature_pattern(saref_signature_pattern())
