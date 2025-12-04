# Standard library imports
import datetime
from typing import Optional
import torch

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
from twin4build.utils.constants import Constants

class AirToAirHeatRecoverySystem(core.System):
    r"""
    An air-to-air heat recovery system model.

    This model represents a heat exchanger that recovers heat between supply and
    exhaust air streams. The effectiveness varies based on flow rates and operation
    mode (heating or cooling). The model includes temperature setpoint control and
    handles cases where heat recovery is not feasible.

    The model is implemented with the following features:
       - Flow-dependent effectiveness interpolation
       - Separate effectiveness values for heating and cooling modes
       - Temperature setpoint control
       - Energy conservation between air streams

    Args:
       eps_75_h: Effectiveness at 75% flow in heating mode
       eps_100_h: Effectiveness at 100% flow in heating mode
       eps_75_c: Effectiveness at 75% flow in cooling mode
       eps_100_c: Effectiveness at 100% flow in cooling mode
       primaryAirFlowRateMax: Maximum primary (supply) air flow rate [kg/s]
       secondaryAirFlowRateMax: Maximum secondary (exhaust) air flow rate [kg/s]

    Mathematical Formulation:

       The effectiveness :math:`\varepsilon` is interpolated based on flow rate fraction :math:`f`:

       .. math::

          f = \frac{1}{2} \frac{\dot{m}_{a,sup} + \dot{m}_{a,exh}}{\operatorname{max}(\dot{m}_{a,sup}, \dot{m}_{a,exh})}

       .. math::

          \varepsilon = \varepsilon_{75} + (\varepsilon_{100} - \varepsilon_{75}) \cdot \frac{f - 0.75}{1 - 0.75}

       where:
          - :math:`\varepsilon_{75}`: Effectiveness at 75% flow
          - :math:`\varepsilon_{100}`: Effectiveness at 100% flow
          - :math:`f`: Normalized flow rate

       The outlet temperature of the supply air stream is:

       .. math::

          T_\text{out,sup} = T_\text{in,sup} + \varepsilon(f) \cdot (T_\text{in,exh} - T_\text{in,sup}) \cdot \frac{C_\min}{C_\text{sup}}

       where:
          - :math:`T_\text{in,sup}`: Inlet temperature of the supply air
          - :math:`T_\text{in,exh}`: Inlet temperature of the exhaust air
          - :math:`C_\text{sup}=\dot{m}_{a,primary} \cdot c_p`: Heat capacity rate of the supply (primary) air
          - :math:`C_\text{exh}=\dot{m}_{a,secondary} \cdot c_p`: Heat capacity rate of the exhaust (secondary) air
          - :math:`C_\min = \min(C_\text{sup}, C_\text{exh})`: Minimum heat capacity rate


       The outlet temperature of the exhaust air stream is:

       .. math::

          T_\text{out,exh} = T_\text{in,exh} - \Delta T \cdot \frac{C_\text{sup}}{C_\text{exh}}

       where:
          - :math:`\Delta T = T_\text{out,sup} - T_\text{in,sup}`: Temperature change in supply air stream

    """

    def __init__(
        self,
        eps_75_h=None,
        eps_100_h=None,
        eps_75_c=None,
        eps_100_c=None,
        primaryAirFlowRateMax=None,
        secondaryAirFlowRateMax=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Store attributes as private variables
        self.eps_75_h = tps.Parameter(
            torch.tensor(eps_75_h, dtype=torch.float64), requires_grad=False
        )
        self.eps_100_h = tps.Parameter(
            torch.tensor(eps_100_h, dtype=torch.float64), requires_grad=False
        )
        self.eps_75_c = tps.Parameter(
            torch.tensor(eps_75_c, dtype=torch.float64), requires_grad=False
        )
        self.eps_100_c = tps.Parameter(
            torch.tensor(eps_100_c, dtype=torch.float64), requires_grad=False
        )
        self.primaryAirFlowRateMax = tps.Parameter(
            torch.tensor(primaryAirFlowRateMax, dtype=torch.float64), requires_grad=False
        )
        self.secondaryAirFlowRateMax = tps.Parameter(
            torch.tensor(secondaryAirFlowRateMax, dtype=torch.float64), requires_grad=False
        )

        # Define inputs and outputs as private variables
        self._input = {
            "primaryAirFlowRate": tps.Scalar(),
            "secondaryAirFlowRate": tps.Scalar(),
            "primaryTemperatureIn": tps.Scalar(),
            "secondaryTemperatureIn": tps.Scalar(),
            "primaryTemperatureOutSetpoint": tps.Scalar(),
        }
        self._output = {
            "primaryTemperatureOut": tps.Scalar(),
            "secondaryTemperatureOut": tps.Scalar(),
        }
        self._config = {
            "parameters": [
                "eps_75_h",
                "eps_100_h",
                "eps_75_c",
                "eps_100_c",
                "primaryAirFlowRateMax",
                "secondaryAirFlowRateMax",
            ]
        }

    @property
    def config(self):
        """Get the configuration parameters.

        Returns:
            dict: Configuration parameters.
        """
        return self._config

    @property
    def input(self) -> dict:
        """
        Get the input ports of the air-to-air heat recovery system.

        Returns:
            dict: Dictionary containing input ports:
                - "primaryAirFlowRate": Primary (supply) air flow rate [kg/s]
                - "secondaryAirFlowRate": Secondary (exhaust) air flow rate [kg/s]
                - "primaryTemperatureIn": Primary air inlet temperature [°C]
                - "secondaryTemperatureIn": Secondary air inlet temperature [°C]
                - "primaryTemperatureOutSetpoint": Primary air outlet temperature setpoint [°C]
        """
        return self._input

    @property
    def output(self) -> dict:
        """
        Get the output ports of the air-to-air heat recovery system.

        Returns:
            dict: Dictionary containing output ports:
                - "primaryTemperatureOut": Primary air outlet temperature [°C]
                - "secondaryTemperatureOut": Secondary air outlet temperature [°C]
        """
        return self._output

    def initialize(
        self,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
        step_size: int,
    ) -> None:
        """Initialize the system for simulation.

        This method is currently not implemented as the system does not require initialization.

        Args:
            start_time: Start time of the simulation period.
            end_time: End time of the simulation period.
            step_size: Time step size in seconds.
            simulator: Simulation model object.
        """
        _, _, max_timesteps, _ = core.Simulator.get_simulation_timesteps(
            start_time, end_time, step_size
        )
        batch_size = len(start_time)
        for input in self.input.values():
            input.initialize(
                n_timesteps=max_timesteps,
                batch_size=batch_size,
            )
        for output in self.output.values():
            output.initialize(
                n_timesteps=max_timesteps,
                batch_size=batch_size,
            )
        self.INITIALIZED = True

    def do_step(
        self,
        second_time: float,
        date_time: datetime.datetime,
        step_size: int,
        step_index: int,
    ) -> None:
        """Perform one simulation step.

        This method calculates the heat recovery between supply and exhaust air streams
        based on the current flow rates and temperatures. The effectiveness is interpolated
        based on the flow rates, and the operation mode (heating/cooling) is determined
        by comparing inlet temperatures and setpoints.

        The method handles the following cases:
        1. No flow in either stream: Pass-through temperatures
        2. Heat recovery feasible: Calculate effectiveness and heat transfer
        3. Heat recovery not feasible: Pass-through temperatures

        Args:
            second_time: Current simulation time in seconds.
            date_time: Current simulation date and time.
            step_size: Time step size in seconds.
            step_index: Current simulation step index.
        """
        tol = 1e-5
        
        # Get input values
        primary_flow = self.input["primaryAirFlowRate"].get()
        secondary_flow = self.input["secondaryAirFlowRate"].get()
        primary_temp_in = self.input["primaryTemperatureIn"].get()
        secondary_temp_in = self.input["secondaryTemperatureIn"].get()
        primary_temp_setpoint = self.input["primaryTemperatureOutSetpoint"].get()
        
        # Condition: both flow rates above tolerance
        has_flow = (primary_flow > tol) & (secondary_flow > tol)
        
        # Determine feasible mode (heating if primary < secondary, else cooling)
        is_heating_feasible = primary_temp_in < secondary_temp_in
        
        # Select effectiveness values based on feasible mode
        eps_75 = torch.where(is_heating_feasible, self.eps_75_h.get(), self.eps_75_c.get())
        eps_100 = torch.where(is_heating_feasible, self.eps_100_h.get(), self.eps_100_c.get())
        
        # Determine operation mode (heating if primary < setpoint, else cooling)
        is_heating_mode = primary_temp_in < primary_temp_setpoint
        
        # Feasible matches operation mode when both are heating or both are cooling
        feasible_matches_mode = is_heating_feasible == is_heating_mode
        
        # Condition for heat recovery: has flow AND feasible matches mode
        do_heat_recovery = has_flow & feasible_matches_mode
        
        # Calculate heat recovery values (computed for all, selected later)
        m_a_max = torch.max(self.primaryAirFlowRateMax.get(), self.secondaryAirFlowRateMax.get())
        
        f_flow = 0.5 * (primary_flow + secondary_flow) / m_a_max
        eps_op = eps_75 + (eps_100 - eps_75) * (f_flow - 0.75) / (1 - 0.75)
        
        C_sup = primary_flow * Constants.specificHeatCapacity["air"]
        C_exh = secondary_flow * Constants.specificHeatCapacity["air"]
        C_min = torch.min(C_sup, C_exh)
        
        # Calculate primary temperature out with heat recovery
        primary_temp_out_hr = (
            primary_temp_in
            + eps_op * (secondary_temp_in - primary_temp_in) * (C_min / C_sup)
        )
        
        # Clamp to setpoint based on operation mode
        # In heating mode: clamp if output > setpoint
        # In cooling mode: clamp if output < setpoint
        heating_exceeds = is_heating_mode & (primary_temp_out_hr > primary_temp_setpoint)
        cooling_exceeds = (~is_heating_mode) & (primary_temp_out_hr < primary_temp_setpoint)
        needs_clamping = heating_exceeds | cooling_exceeds
        
        primary_temp_out_hr = torch.where(
            needs_clamping, primary_temp_setpoint, primary_temp_out_hr
        )
        
        # Calculate secondary temperature out using energy conservation
        primary_delta_T = primary_temp_out_hr - primary_temp_in
        secondary_delta_T = primary_delta_T * (C_sup / C_exh)
        secondary_temp_out_hr = secondary_temp_in - secondary_delta_T
        
        # Select final outputs: heat recovery values if conditions met, else pass-through
        primary_temp_out = torch.where(do_heat_recovery, primary_temp_out_hr, primary_temp_in)
        secondary_temp_out = torch.where(do_heat_recovery, secondary_temp_out_hr, secondary_temp_in)
        
        # Set outputs
        self.output["primaryTemperatureOut"].set(primary_temp_out, step_index)
        self.output["secondaryTemperatureOut"].set(secondary_temp_out, step_index)


def saref_signature_pattern():
    node0 = Node(cls=core.namespace.S4BLDG.AirToAirHeatRecovery)
    node1 = Node(cls=core.namespace.S4BLDG.OutdoorEnvironment)
    node2 = Node(cls=core.namespace.S4BLDG.FlowJunction)
    node3 = Node(cls=core.namespace.S4BLDG.FlowJunction)
    node4 = Node(cls=core.namespace.S4BLDG.PrimaryAirFlowRateMax)
    node5 = Node(cls=core.namespace.SAREF.PropertyValue)
    node6 = Node(cls=core.namespace.XSD.float)
    node7 = Node(cls=core.namespace.S4BLDG.SecondaryAirFlowRateMax)
    node8 = Node(cls=core.namespace.SAREF.PropertyValue)
    node9 = Node(cls=core.namespace.XSD.float)
    node10 = Node(cls=core.namespace.S4BLDG.AirToAirHeatRecovery)  # primary
    node11 = Node(cls=core.namespace.S4BLDG.AirToAirHeatRecovery)  # secondary
    node12 = Node(cls=core.namespace.S4BLDG.Controller)
    node13 = Node(cls=core.namespace.SAREF.Motion)
    node14 = Node(cls=core.namespace.S4BLDG.Schedule)
    sp = SignaturePattern(id="air_to_air_heat_recovery_signature_pattern")

    # buildingTemperature (SecondaryTemperatureIn)
    sp.add_triple(
        SinglePath(
            subject=node10,
            object=node1,
            predicate=core.namespace.FSO.hasFluidSuppliedBy,
        )
    )
    sp.add_triple(
        SinglePath(
            subject=node10, object=node2, predicate=core.namespace.FSO.suppliesFluidTo
        )
    )
    sp.add_triple(
        SinglePath(
            subject=node11,
            object=node3,
            predicate=core.namespace.FSO.hasFluidReturnedBy,
        )
    )

    sp.add_triple(
        Optional_(subject=node5, object=node6, predicate=core.namespace.SAREF.hasValue)
    )
    sp.add_triple(
        Optional_(
            subject=node5,
            object=node4,
            predicate=core.namespace.SAREF.isValueOfProperty,
        )
    )
    sp.add_triple(
        Optional_(
            subject=node0, object=node5, predicate=core.namespace.SAREF.hasPropertyValue
        )
    )

    # airFlowRateMax
    sp.add_triple(
        Optional_(subject=node8, object=node9, predicate=core.namespace.SAREF.hasValue)
    )
    sp.add_triple(
        Optional_(
            subject=node8,
            object=node7,
            predicate=core.namespace.SAREF.isValueOfProperty,
        )
    )
    sp.add_triple(
        Optional_(
            subject=node0, object=node8, predicate=core.namespace.SAREF.hasPropertyValue
        )
    )

    sp.add_triple(
        Exact(subject=node10, object=node0, predicate=core.namespace.S4SYST.subSystemOf)
    )
    sp.add_triple(
        Exact(subject=node11, object=node0, predicate=core.namespace.S4SYST.subSystemOf)
    )

    sp.add_triple(
        Exact(subject=node12, object=node13, predicate=core.namespace.SAREF.controls)
    )
    sp.add_triple(
        Exact(subject=node13, object=node0, predicate=core.namespace.SAREF.isPropertyOf)
    )
    sp.add_triple(
        Exact(subject=node12, object=node14, predicate=core.namespace.SAREF.hasProfile)
    )

    sp.add_parameter("primaryAirFlowRateMax", node6)
    sp.add_parameter("secondaryAirFlowRateMax", node9)

    sp.add_input("primaryTemperatureIn", node1, "outdoorTemperature")
    sp.add_input("secondaryTemperatureIn", node3, "airTemperatureOut")
    sp.add_input("primaryAirFlowRate", node2, "airFlowRateIn")
    sp.add_input("secondaryAirFlowRate", node3, "airFlowRateOut")
    sp.add_input("primaryTemperatureOutSetpoint", node14, "scheduleValue")

    sp.add_modeled_node(node0)
    sp.add_modeled_node(node10)
    sp.add_modeled_node(node11)

    return sp


AirToAirHeatRecoverySystem.add_signature_pattern(saref_signature_pattern())
