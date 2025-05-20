import numpy as np
import torch
import torch.nn as nn
import twin4build.utils.input_output_types as tps
import twin4build.core as core
import datetime
from typing import Optional

class DamperTorchSystem(core.System, nn.Module):
    r"""
    A damper system model implemented with PyTorch for gradient-based optimization.

    This model represents a damper that controls air flow rate based on damper position,
    using an exponential equation for accurate flow control representation.

    Mathematical Formulation
    -----------------------

    The damper characteristic is calculated using an exponential equation:

        .. math::

            \dot{m} = a \cdot e^{b \cdot u} + c

    where:
       - :math:`\dot{m}` is the air flow rate [m³/s]
       - :math:`a` is the shape parameter
       - :math:`b` is calculated to ensure :math:`\dot{m} = \dot{m}_{nom}` at :math:`u = 1`
       - :math:`c` is calculated to ensure :math:`\dot{m} = 0` at :math:`u = 0`
       - :math:`u` is the damper position (0-1)
       - :math:`\dot{m}_{nom}` is the nominal air flow rate [m³/s]

    The parameters :math:`b` and :math:`c` are calculated during initialization:

        .. math::

            c = -a
            b = \ln(\frac{\dot{m}_{nom} - c}{a})

    where:
       - :math:`c = -a` ensures zero flow at closed position
       - :math:`b` is calculated to ensure nominal flow at fully open position

    Parameters
    ----------
    a : float
        Shape parameter for the air flow curve. Controls the non-linearity
        of the damper characteristic. Higher values result in more non-linear behavior.
    nominalAirFlowRate : float
        Nominal air flow rate [m³/s] at fully open position

    Attributes
    ----------
    input : Dict[str, Scalar]
        Dictionary containing input ports:
        - "damperPosition": Damper position (0-1)
    output : Dict[str, Scalar]
        Dictionary containing output ports:
        - "damperPosition": Damper position (0-1)
        - "airFlowRate": Air flow rate [m³/s]
    parameter : Dict[str, Dict[str, float]]
        Dictionary containing parameter bounds for calibration:
        - "a": {"lb": 0.0001, "ub": 5}
        - "nominalAirFlowRate": {"lb": 0.0001, "ub": 5}
    a : torch.nn.Parameter
        Shape parameter, stored as a PyTorch parameter
    nominalAirFlowRate : torch.nn.Parameter
        Nominal air flow rate [m³/s], stored as a PyTorch parameter
    b : torch.Tensor
        Exponential coefficient calculated during initialization
    c : torch.Tensor
        Offset coefficient calculated during initialization

    Notes
    -----
    Damper Characteristics:
       - The exponential characteristic provides a more realistic representation
         of damper behavior compared to a linear relationship
       - The shape parameter 'a' controls the non-linearity of the flow curve
       - Higher values of 'a' result in more non-linear behavior
       - The model ensures zero flow at closed position and nominal flow at
         fully open position

    Implementation Details:
       - The model uses PyTorch tensors for gradient-based optimization
       - Parameters 'a' and 'nominalAirFlowRate' are stored as non-trainable
         PyTorch parameters
       - Parameters 'b' and 'c' are calculated during initialization
       - The model assumes ideal damper behavior (no hysteresis or deadband)
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