import numpy as np
import torch
import torch.nn as nn
import twin4build.utils.input_output_types as tps
import twin4build.core as core
import datetime
from typing import Optional

class DamperTorchSystem(core.System, nn.Module):
    """
    A damper system model implemented with PyTorch for gradient-based optimization.
    
    This model represents a damper that controls air flow rate based on damper position.
    The damper characteristic is modeled using an exponential equation:
    m = a * exp(b * u) + c
    where:
    - m is the air flow rate
    - a is the shape parameter
    - b is calculated to ensure m=nominalAirFlowRate at u=1
    - c is calculated to ensure m=0 at u=0
    - u is the damper position (0-1)
    """
    
    def __init__(self,
                a: float = None,
                nominalAirFlowRate: float = None,
                **kwargs):
        """
        Initialize the damper system model.
        
        Args:
            a: Shape parameter for the air flow curve
            nominalAirFlowRate: Nominal air flow rate [m³/s]
        """
        super().__init__(**kwargs)
        nn.Module.__init__(self)
        
        # Store parameters as nn.Parameters for gradient tracking
        self.a = nn.Parameter(torch.tensor(a, dtype=torch.float32), requires_grad=False)
        self.nominalAirFlowRate = nn.Parameter(torch.tensor(nominalAirFlowRate, dtype=torch.float32), requires_grad=False)
        
        # Define inputs and outputs
        self.input = {"damperPosition": tps.Scalar()}
        self.output = {"damperPosition": tps.Scalar(),
                      "airFlowRate": tps.Scalar()}
        
        # Define parameters for calibration
        self.parameter = {
            "a": {"lb": 0.0001, "ub": 5},
            "nominalAirFlowRate": {"lb": 0.0001, "ub": 5}
        }
        
        self._config = {"parameters": list(self.parameter.keys())}
        self.INITIALIZED = False
    
    @property
    def config(self):
        """Get the configuration of the damper system."""
        return self._config
    
    def cache(self, startTime=None, endTime=None, stepSize=None):
        """Cache method placeholder."""
        pass
    
    def initialize(self, 
                   startTime=None, 
                   endTime=None, 
                   stepSize=None, 
                   simulator=None):
        """Initialize the damper system."""
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
        
        # Calculate b and c parameters
        self.c = -self.a  # Ensures that m=0 at u=0
        self.b = torch.log((self.nominalAirFlowRate-self.c)/self.a)  # Ensures that m=nominalAirFlowRate at u=1
        
        self.INITIALIZED = True
    
    def do_step(self, 
                secondTime: Optional[float] = None, 
                dateTime: Optional[datetime.datetime] = None, 
                stepSize: Optional[float] = None, 
                stepIndex: Optional[int] = None) -> None:
        """
        Perform one step of the damper system simulation.
        
        The damper characteristic is calculated using an exponential equation:
        m = a * exp(b * u) + c
        where:
        - m is the air flow rate
        - a is the shape parameter
        - b is calculated to ensure m=nominalAirFlowRate at u=1
        - c is calculated to ensure m=0 at u=0
        - u is the damper position (0-1)
        """
        # Get input damper position (assumed to be a tensor)
        damper_position = self.input["damperPosition"].get()
        
        # Calculate air flow rate using exponential equation
        air_flow_rate = self.a * torch.exp(self.b * damper_position) + self.c
        
        # Update outputs
        self.output["damperPosition"].set(damper_position, stepIndex)
        self.output["airFlowRate"].set(air_flow_rate, stepIndex) 