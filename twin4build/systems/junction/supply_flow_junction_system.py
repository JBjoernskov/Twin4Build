# Standard library imports
import datetime
from typing import List, Optional, Union

# Local application imports
import twin4build.core as core
import twin4build.utils.types as tps
from twin4build.translator.translator import (
    StepRule,
    AnyPathRule,
    Node,
    OptionalRule,
    SignaturePattern,
    PathRule,
)


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

    def __init__(self, airFlowRateBias=None, **kwargs):
        super().__init__(**kwargs)
        if airFlowRateBias is not None:
            self.airFlowRateBias = airFlowRateBias
        else:
            self.airFlowRateBias = 0

        self._manual_setup_n_input_ports = False
        self._n_input_ports = 0  # At least one input port is required

        self.input = {"airFlowRateOut": tps.Vector()}
        self.output = {"airFlowRateIn": tps.Scalar()}
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
        """Get the configuration parameters.

        Returns:
            dict: Dictionary containing configuration parameters.
        """
        return self._config

    def initialize(
        self,
        start_time: Union[List[datetime.datetime], datetime.datetime],
        end_time: Union[List[datetime.datetime], datetime.datetime],
        step_size: Union[List[int], int],
    ) -> None:
        """Initialize the supply flow junction system.

        This method is a no-op as the supply flow junction system does not require initialization.
        The system has no internal state to initialize and performs a simple summation
        of input flow rates with an optional bias.

        Args:
            start_time (datetime.datetime): Start time of the simulation period.
            end_time (datetime.datetime): End time of the simulation period.
            step_size (int): Time step size in seconds.
            simulator (core.Simulator): Simulation model object.
        """

        _, _, max_timesteps, _ = core.Simulator.get_simulation_timesteps(
            start_time, end_time, step_size
        )
        batch_size = len(start_time)
        self.setup_variable_inputs()
        self.input["airFlowRateOut"].initialize(
            n_t=max_timesteps, n_s=batch_size, n_v=self.n_input_ports
        )

        for output in self.output.values():
            output.initialize(
                n_t=max_timesteps,
                n_s=batch_size,
            )

    def setup_variable_inputs(self):
        if self._manual_setup_n_input_ports == False:
            # Assert that the number of input ports is at least 1
            connection_point = [
                cp for cp in self.connects_at if cp.input_port == "airFlowRateOut"
            ]
            if len(connection_point) == 0:
                raise ValueError("No input port found for airFlowRateOut")
            n_input_ports = len(connection_point[0].connects_system_through)
            self.n_input_ports = n_input_ports

    def do_step(
        self,
        second_time: float,
        date_time: datetime.datetime,
        step_size: int,
        step_index: int,
    ) -> None:
        """Perform one simulation step.

        This method sums all input flow rates and adds the bias to calculate
        the total flow rate. The input flow rates are provided as a vector,
        and the output is a scalar representing the total flow rate.

        Args:
            second_time (float, optional): Current simulation time in seconds.
            date_time (date_time, optional): Current simulation date and time.
            step_size (float, optional): Time step size in seconds.
            step_index (int, optional): Current simulation step index.
        """
        self.output["airFlowRateIn"]._set(
            (self.input["airFlowRateOut"].get().sum(dim=-1)) + self.airFlowRateBias,
            step_index,
        )


def saref_signature_pattern():
    """
    Get the SAREF signature pattern of the supply flow junction component.

    Returns:
        SignaturePattern: The SAREF signature pattern of the supply flow junction component.
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
        id="supply_flow_junction_signature_pattern",
    )
    sp.add_rule(
        AnyPathRule(
            subject=node0, object=node1, predicate=core.namespace.FSO.suppliesFluidTo
        )
    )
    sp.add_rule(
        PathRule(
            subject=node0, object=node2, predicate=core.namespace.FSO.hasFluidSuppliedBy
        )
    )
    sp.add_input("airFlowRateOut", node1, "airFlowRate")
    sp.add_modeled_node(node0)
    return sp


def brick_signature_pattern():
    """
    Get the BRICK signature pattern of the supply flow junction component.

    Returns:
        SignaturePattern: The BRICK signature pattern of the supply flow junction component.
    """
    node0 = Node(cls=core.namespace.BRICK.Air_Flow_Junction)  # flow junction
    node1 = Node(cls=core.namespace.BRICK.Damper)  # damper
    node2 = Node(cls=core.namespace.BRICK.AHU)  # air handling unit

    sp = SignaturePattern(
        id="supply_flow_junction_signature_pattern_brick",
    )
    sp.add_rule(
        StepRule(subject=node0, object=node1, predicate=core.namespace.BRICK.feeds)
    )
    sp.add_rule(
        StepRule(subject=node2, object=node0, predicate=core.namespace.BRICK.feeds)
    )
    sp.add_input("airFlowRateOut", node1, "airFlowRate")
    sp.add_modeled_node(node0)
    return sp


SupplyFlowJunctionSystem.add_signature_pattern(brick_signature_pattern())
SupplyFlowJunctionSystem.add_signature_pattern(saref_signature_pattern())
