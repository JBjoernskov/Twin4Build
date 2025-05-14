import numpy as np
import torch
import torch.nn as nn
import twin4build.utils.input_output_types as tps
import twin4build.core as core
import datetime
from typing import Optional
from twin4build.utils.constants import Constants

class CoilTorchSystem(core.System, nn.Module):
    """
    A coil system model implemented with PyTorch for gradient-based optimization.
    
    This model represents a heating/cooling coil that transfers heat between air and water.
    The model calculates heating/cooling power based on air flow rate, inlet temperature,
    and outlet temperature setpoint.
    """
    
    def __init__(self,
                **kwargs):
        """
        Initialize the coil system model.
        """
        super().__init__(**kwargs)
        nn.Module.__init__(self)
        
        # Store specific heat capacity as nn.Parameter
        self.specificHeatCapacityAir = nn.Parameter(torch.tensor(Constants.specificHeatCapacity["air"], dtype=torch.float32), requires_grad=False)
        
        # Define inputs and outputs
        self.input = {"inletAirTemperature": tps.Scalar(),
                     "outletAirTemperatureSetpoint": tps.Scalar(),
                     "airFlowRate": tps.Scalar()}
        self.output = {"heatingPower": tps.Scalar(),
                      "coolingPower": tps.Scalar(),
                      "outletAirTemperature": tps.Scalar()}
        
        # Define parameters for calibration
        self.parameter = {}
        
        self._config = {"parameters": list(self.parameter.keys())}
        self.INITIALIZED = False
    
    @property
    def config(self):
        """Get the configuration of the coil system."""
        return self._config
    
    def cache(self, startTime=None, endTime=None, stepSize=None):
        """Cache method placeholder."""
        pass
    
    def initialize(self, 
                   startTime=None, 
                   endTime=None, 
                   stepSize=None, 
                   simulator=None):
        """Initialize the coil system."""
        # Initialize I/O
        for input in self.input.values():
            input.initialize(startTime=startTime,
                             endTime=endTime,
                             stepSize=stepSize,
                             simulator=simulator)
        for output in self.output.values():
            output.initialize(startTime=startTime,
                             endTime=endTime,
                             stepSize=stepSize,
                             simulator=simulator)
        
        self.INITIALIZED = True
    
    def do_step(self, 
                secondTime: Optional[float] = None, 
                dateTime: Optional[datetime.datetime] = None, 
                stepSize: Optional[float] = None, 
                stepIndex: Optional[int] = None) -> None:
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
        tol = torch.tensor(1e-5, dtype=torch.float32)
        if air_flow_rate > tol:
            if inlet_air_temp < outlet_air_temp_setpoint:
                # Heating mode
                heating_power = air_flow_rate * self.specificHeatCapacityAir * (outlet_air_temp_setpoint - inlet_air_temp)
                cooling_power = torch.tensor(0.0, dtype=torch.float32)
            else:
                # Cooling mode
                heating_power = torch.tensor(0.0, dtype=torch.float32)
                cooling_power = air_flow_rate * self.specificHeatCapacityAir * (inlet_air_temp - outlet_air_temp_setpoint)
        else:
            # No flow
            heating_power = torch.tensor(0.0, dtype=torch.float32)
            cooling_power = torch.tensor(0.0, dtype=torch.float32)
        
        # Update outputs
        self.output["heatingPower"].set(heating_power, stepIndex)
        self.output["coolingPower"].set(cooling_power, stepIndex)
        self.output["outletAirTemperature"].set(outlet_air_temp_setpoint, stepIndex) 