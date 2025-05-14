import numpy as np
import torch
import torch.nn as nn
import twin4build.utils.input_output_types as tps
import twin4build.core as core
import datetime
from typing import Optional

class FanTorchSystem(core.System, nn.Module):
    """
    A fan system model implemented with PyTorch for gradient-based optimization.
    
    This model represents a fan that controls air flow rate and temperature.
    The fan power is calculated using a polynomial equation with coefficients c1-c4.
    """
    
    def __init__(self,
                nominalPowerRate: float = None,
                nominalAirFlowRate: float = None,
                c1: float = None,
                c2: float = None,
                c3: float = None,
                c4: float = None,
                f_total: float = None,
                **kwargs):
        """
        Initialize the fan system model.
        
        Args:
            nominalPowerRate: Nominal power rate [W]
            nominalAirFlowRate: Nominal air flow rate [m³/s]
            c1-c4: Polynomial coefficients for power calculation
            f_total: Total fan efficiency factor
        """
        super().__init__(**kwargs)
        nn.Module.__init__(self)
        
        # Store parameters as nn.Parameters for gradient tracking
        self.nominalPowerRate = nn.Parameter(torch.tensor(nominalPowerRate, dtype=torch.float32), requires_grad=False)
        self.nominalAirFlowRate = nn.Parameter(torch.tensor(nominalAirFlowRate, dtype=torch.float32), requires_grad=False)
        self.c1 = nn.Parameter(torch.tensor(c1, dtype=torch.float32), requires_grad=False)
        self.c2 = nn.Parameter(torch.tensor(c2, dtype=torch.float32), requires_grad=False)
        self.c3 = nn.Parameter(torch.tensor(c3, dtype=torch.float32), requires_grad=False)
        self.c4 = nn.Parameter(torch.tensor(c4, dtype=torch.float32), requires_grad=False)
        self.f_total = nn.Parameter(torch.tensor(f_total, dtype=torch.float32), requires_grad=False)
        
        # Define inputs and outputs
        self.input = {"airFlowRate": tps.Scalar(),
                     "inletAirTemperature": tps.Scalar()}
        self.output = {"outletAirTemperature": tps.Scalar(),
                      "Power": tps.Scalar()}
        
        # Define parameters for calibration
        self.parameter = {
            "nominalPowerRate": {"lb": 0.0, "ub": 10000.0},
            "nominalAirFlowRate": {"lb": 0.0, "ub": 10.0},
            "c1": {"lb": -10.0, "ub": 10.0},
            "c2": {"lb": -10.0, "ub": 10.0},
            "c3": {"lb": -10.0, "ub": 10.0},
            "c4": {"lb": -10.0, "ub": 10.0},
            "f_total": {"lb": 0.0, "ub": 1.0}
        }
        
        self._config = {"parameters": list(self.parameter.keys())}
        self.INITIALIZED = False
    
    @property
    def config(self):
        """Get the configuration of the fan system."""
        return self._config
    
    def cache(self, startTime=None, endTime=None, stepSize=None):
        """Cache method placeholder."""
        pass
    
    def initialize(self, 
                   startTime=None, 
                   endTime=None, 
                   stepSize=None, 
                   simulator=None):
        """Initialize the fan system."""
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
        Perform one step of the fan system simulation.
        
        The fan power is calculated using a polynomial equation:
        P = P_nom * (c1 + c2*(m/m_nom) + c3*(m/m_nom)^2 + c4*(m/m_nom)^3)
        where:
        - P is the fan power
        - P_nom is the nominal power
        - m is the air flow rate
        - m_nom is the nominal air flow rate
        - c1-c4 are polynomial coefficients
        
        The outlet air temperature is calculated considering the heat added by the fan:
        T_out = T_in + (P * f_total) / (m * c_p)
        where:
        - T_out is the outlet temperature
        - T_in is the inlet temperature
        - f_total is the total fan efficiency
        - c_p is the specific heat capacity of air
        """
        # Get inputs
        air_flow_rate = self.input["airFlowRate"].get()
        inlet_temp = self.input["inletAirTemperature"].get()
        
        # Convert to torch tensors if not already
        if not isinstance(air_flow_rate, torch.Tensor):
            air_flow_rate = torch.tensor(air_flow_rate, dtype=torch.float32)
        if not isinstance(inlet_temp, torch.Tensor):
            inlet_temp = torch.tensor(inlet_temp, dtype=torch.float32)
        
        # Calculate normalized flow rate
        m_norm = air_flow_rate / self.nominalAirFlowRate
        
        # Calculate fan power using polynomial equation
        power = self.nominalPowerRate * (self.c1 + 
                                       self.c2 * m_norm + 
                                       self.c3 * m_norm**2 + 
                                       self.c4 * m_norm**3)
        
        # Calculate outlet temperature
        # Using air properties at standard conditions
        c_p = 1005.0  # J/(kg·K)
        rho = 1.2     # kg/m³
        
        # Convert volume flow rate to mass flow rate
        m_dot = air_flow_rate * rho
        
        # Calculate temperature rise
        delta_T = (power * self.f_total) / (m_dot * c_p)
        outlet_temp = inlet_temp + delta_T
        
        # Update outputs
        self.output["outletAirTemperature"].set(outlet_temp, stepIndex)
        self.output["Power"].set(power, stepIndex) 