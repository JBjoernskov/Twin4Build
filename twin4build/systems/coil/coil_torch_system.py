# Standard library imports
import datetime
from typing import Optional

# Third party imports
import numpy as np
import torch
import torch.nn as nn

# Local application imports
import twin4build.core as core
import twin4build.utils.constants as constants
import twin4build.utils.types as tps


class CoilTorchSystem(core.System, nn.Module):
    r"""
    A coil system model implemented with PyTorch for gradient-based optimization.

    This model represents a heating/cooling coil that transfers heat between air and water,
    calculating the required heating or cooling power based on air flow rate and temperature
    differences.

    Mathematical Formulation
    ------------------------

    The heating/cooling power is calculated using the following equations:

    For heating mode (when :math:`T_{in} < T_{out,set}`):

        .. math::

            P_{heat} = \dot{m}_{air} \cdot c_{p,air} \cdot (T_{out,set} - T_{in})

        .. math::

            P_{cool} = 0

    For cooling mode (when :math:`T_{in} \geq T_{out,set}`):

        .. math::

            P_{heat} = 0

        .. math::

            P_{cool} = \dot{m}_{air} \cdot c_{p,air} \cdot (T_{in} - T_{out,set})

    where:
       - :math:`P_{heat}` is the heating power [W]
       - :math:`P_{cool}` is the cooling power [W]
       - :math:`\dot{m}_{air}` is the air flow rate [kg/s]
       - :math:`c_{p,air}` is the specific heat capacity of air [J/(kg·K)]
       - :math:`T_{in}` is the inlet air temperature [°C]
       - :math:`T_{out,set}` is the outlet air temperature setpoint [°C]

    Notes
    -----
    Model Assumptions:
       - Perfect heat transfer (outlet temperature equals setpoint)
       - Constant specific heat capacity of air
       - No heat losses to the environment
       - No water-side calculations (focus on air-side performance)

    Implementation Details:
       - If air flow rate is below threshold (1e-5 kg/s), both heating and cooling
         powers are set to zero
       - The model uses PyTorch tensors for gradient-based optimization
       - All calculations are performed in SI units
       - The specific heat capacity of air is the constant ``CP_AIR`` from
         ``twin4build.utils.constants`` (not a stored parameter)
    """

    def __init__(self, **kwargs):
        """
        Initialize the coil system model.
        """
        super().__init__(**kwargs)
        nn.Module.__init__(self)

        # Define inputs and outputs as private variables
        self._input = {
            "inletAirTemperature": tps.Scalar(),
            "outletAirTemperatureSetpoint": tps.Scalar(),
            "airFlowRate": tps.Scalar(),
        }
        self._output = {
            "heatingPower": tps.Scalar(),
            "coolingPower": tps.Scalar(),
            "outletAirTemperature": tps.Scalar(),
        }

        # Define parameters for calibration
        self.parameter = {}

        self._config = {"parameters": list(self.parameter.keys())}
        self.INITIALIZED = False

    @property
    def config(self):
        """Get the configuration of the coil system."""
        return self._config

    @property
    def input(self) -> dict:
        """
        Get the input ports of the coil system.

        Returns:
            dict: Dictionary containing input ports:
                - "inletAirTemperature": Inlet air temperature [°C]
                - "outletAirTemperatureSetpoint": Outlet air temperature setpoint [°C]
                - "airFlowRate": Air flow rate [kg/s]
        """
        return self._input

    @property
    def output(self) -> dict:
        """
        Get the output ports of the coil system.

        Returns:
            dict: Dictionary containing output ports:
                - "heatingPower": Heating power [W]
                - "coolingPower": Cooling power [W]
                - "outletAirTemperature": Outlet air temperature [°C]
        """
        return self._output

    @property
    def specificHeatCapacityAir(self) -> tps.Parameter:
        """
        Accessor for an externally set specific heat capacity of air.

        Note:
            This property is not used by the model: ``do_step`` uses the
            constant ``CP_AIR`` from ``twin4build.utils.constants``, and the
            backing attribute is never set by this class. Accessing it without
            first assigning a value raises ``AttributeError``.

        Returns:
            tps.Parameter: Specific heat capacity of air [J/(kg·K)], if previously set.
        """
        return self._specificHeatCapacityAir

    @specificHeatCapacityAir.setter
    def specificHeatCapacityAir(self, value: tps.Parameter) -> None:
        """
        Set the specific heat capacity of air.

        Note:
            The stored value is not used in calculations; ``do_step`` always
            uses ``constants.CP_AIR``.

        Args:
            value (tps.Parameter): Specific heat capacity of air [J/(kg·K)].
        """
        self._specificHeatCapacityAir = value

    def initialize(
        self,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
        step_size: int,
    ) -> None:
        """Initialize the coil system."""
        # Initialize I/O
        _, _, max_timesteps, _ = core.Simulator.get_simulation_timesteps(
            start_time, end_time, step_size
        )
        batch_size = len(start_time)

        if hasattr(self, "_n_c_compiled") and self._n_c_compiled > 1:
            self.n_c = self._n_c_compiled
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
        self.INITIALIZED = True

    def do_step(
        self,
        second_time: float,
        date_time: datetime.datetime,
        step_size: int,
        step_index: int,
    ) -> None:
        """
        Perform one step of the coil system simulation.

        The model calculates heating/cooling power based on:
        - Air flow rate
        - Inlet air temperature
        - Outlet air temperature setpoint

        If the air flow rate is zero, the output power is set to 0.
        """
        # Get inputs (assumed to be tensors)
        inlet_air_temp = self.input["inletAirTemperature"].get()
        outlet_air_temp_setpoint = self.input["outletAirTemperatureSetpoint"].get()
        air_flow_rate = self.input["airFlowRate"].get()

        # Calculate heating/cooling power based on temperature difference
        tol = 1e-5
        zero = torch.zeros_like(air_flow_rate)

        # Condition: flow rate above tolerance
        has_flow = air_flow_rate > tol

        # Condition: heating mode (inlet < setpoint)
        is_heating_mode = inlet_air_temp < outlet_air_temp_setpoint

        # Calculate power magnitude (same formula, different sign interpretation)
        power = (
            air_flow_rate
            * constants.CP_AIR
            * torch.abs(outlet_air_temp_setpoint - inlet_air_temp)
        )

        # Select heating/cooling power based on mode and flow
        heating_power = torch.where(has_flow & is_heating_mode, power, zero)
        cooling_power = torch.where(has_flow & (~is_heating_mode), power, zero)

        # Update outputs
        self.output["heatingPower"]._set(heating_power, i_t=step_index)
        self.output["coolingPower"]._set(cooling_power, i_t=step_index)
        self.output["outletAirTemperature"]._set(
            outlet_air_temp_setpoint, i_t=step_index
        )
