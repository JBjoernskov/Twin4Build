# Standard library imports
import datetime
from typing import Optional

# Third party imports
import numpy as np
import torch
import torch.nn as nn

# Local application imports
import twin4build.core as core
import twin4build.utils.types as tps
from twin4build.utils.constants import Constants


class CoilTorchSystem(core.System, nn.Module):
    r"""
    A coil system model implemented with PyTorch for gradient-based optimization.

    This model represents a heating/cooling coil that transfers heat between air and water,
    calculating the required heating or cooling power based on air flow rate and temperature
    differences.

    Mathematical Formulation
    -----------------------

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

    Parameters
    ----------
    None
        All parameters are set via constants or inputs.

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
       - The specific heat capacity is stored as a non-trainable PyTorch parameter
    """

    def __init__(self, **kwargs):
        """
        Initialize the coil system model.
        """
        super().__init__(**kwargs)
        nn.Module.__init__(self)

        # Store specific heat capacity as tps.Parameter with private variable
        self._specificHeatCapacityAir = tps.Parameter(
            torch.tensor(Constants.specificHeatCapacity["air"], dtype=torch.float64),
            requires_grad=False,
        )

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
        Get the specific heat capacity of air.

        Returns:
            tps.Parameter: Specific heat capacity of air [J/(kg·K)].
        """
        return self._specificHeatCapacityAir

    @specificHeatCapacityAir.setter
    def specificHeatCapacityAir(self, value: tps.Parameter) -> None:
        """
        Set the specific heat capacity of air.

        Args:
            value (tps.Parameter): Specific heat capacity of air [J/(kg·K)].
        """
        self._specificHeatCapacityAir = value

    def initialize(
        self,
        startTime: datetime.datetime,
        endTime: datetime.datetime,
        stepSize: int,
        simulator: core.Simulator,
    ) -> None:
        """Initialize the coil system."""
        # Initialize I/O
        for input in self.input.values():
            input.initialize(
                startTime=startTime,
                endTime=endTime,
                stepSize=stepSize,
                simulator=simulator,
            )
        for output in self.output.values():
            output.initialize(
                startTime=startTime,
                endTime=endTime,
                stepSize=stepSize,
                simulator=simulator,
            )

        self.INITIALIZED = True

    def do_step(
        self,
        secondTime: float,
        dateTime: datetime.datetime,
        stepSize: int,
        stepIndex: int,
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
        tol = torch.tensor(1e-5, dtype=torch.float64)
        if air_flow_rate > tol:
            if inlet_air_temp < outlet_air_temp_setpoint:
                # Heating mode
                heating_power = (
                    air_flow_rate
                    * self.specificHeatCapacityAir.get()
                    * (outlet_air_temp_setpoint - inlet_air_temp)
                )
                cooling_power = torch.tensor(0.0, dtype=torch.float64)
            else:
                # Cooling mode
                heating_power = torch.tensor(0.0, dtype=torch.float64)
                cooling_power = (
                    air_flow_rate
                    * self.specificHeatCapacityAir.get()
                    * (inlet_air_temp - outlet_air_temp_setpoint)
                )
        else:
            # No flow
            heating_power = torch.tensor(0.0, dtype=torch.float64)
            cooling_power = torch.tensor(0.0, dtype=torch.float64)

        # Update outputs
        self.output["heatingPower"].set(heating_power, stepIndex)
        self.output["coolingPower"].set(cooling_power, stepIndex)
        self.output["outletAirTemperature"].set(outlet_air_temp_setpoint, stepIndex)
