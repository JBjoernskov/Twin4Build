import numpy as np
import torch
import torch.nn as nn
import twin4build.utils.input_output_types as tps
import twin4build.core as core
import datetime
from typing import Optional
from twin4build.translator.translator import SignaturePattern, Node, Exact, SinglePath, MultiPath, Optional_

def get_signature_pattern():
    node0 = Node(cls=core.S4BLDG.Valve) #supply valve
    node1 = Node(cls=core.S4BLDG.Controller)
    node2 = Node(cls=core.SAREF.OpeningPosition)
    sp = SignaturePattern(semantic_model_=core.ontologies, ownedBy="ValveFMUSystem")

    sp.add_triple(Exact(subject=node1, object=node2, predicate=core.SAREF.controls))
    sp.add_triple(Exact(subject=node2, object=node0, predicate=core.SAREF.isPropertyOf))

    sp.add_input("valvePosition", node1, "inputSignal")
    sp.add_modeled_node(node0)

    return sp

class ValveTorchSystem(core.System, nn.Module):
    r"""
    A valve system model implemented with PyTorch for gradient-based optimization.
    
    This model represents a valve that controls water flow rate based on valve position.
    The valve characteristic is modeled using the valve authority equation, which provides
    a more accurate representation of the valve's behavior compared to a simple linear
    relationship.

    Mathematical Formulation
    -----------------------

    The valve characteristic is calculated using the valve authority equation:

        .. math::

            u_{norm} = \frac{u}{\sqrt{u^2 (1-a) + a}}

    where:
       - :math:`u` is the valve position (0-1)
       - :math:`a` is the valve authority (0-1)
       - :math:`u_{norm}` is the normalized valve position

    The water flow rate is then calculated as:

        .. math::

            \dot{m}_w = u_{norm} \cdot \dot{m}_{w,max}

    where:
       - :math:`\dot{m}_w` is the water flow rate [kg/s]
       - :math:`\dot{m}_{w,max}` is the maximum water flow rate [kg/s]

    Parameters
    ----------
    waterFlowRateMax : float
        Maximum water flow rate [kg/s]
    valveAuthority : float
        Valve authority (0-1), where:
        - 0: Linear characteristic
        - 1: Equal percentage characteristic
        - Values in between: Mixed characteristic

    Attributes
    ----------
    input : Dict[str, Scalar]
        Dictionary containing input ports:
        - "valvePosition": Valve position (0-1)
    output : Dict[str, Scalar]
        Dictionary containing output ports:
        - "valvePosition": Valve position (0-1)
        - "waterFlowRate": Water flow rate [kg/s]
    parameter : Dict[str, Dict[str, float]]
        Dictionary containing parameter bounds for calibration:
        - "waterFlowRateMax": {"lb": 0.0, "ub": 10.0}
        - "valveAuthority": {"lb": 0.0, "ub": 1.0}
    waterFlowRateMax : torch.nn.Parameter
        Maximum water flow rate [kg/s], stored as a PyTorch parameter
    valveAuthority : torch.nn.Parameter
        Valve authority (0-1), stored as a PyTorch parameter

    Notes
    -----
    Valve Authority Characteristics:
       - Linear (a = 0): Flow rate is directly proportional to valve position
       - Equal Percentage (a = 1): Flow rate changes exponentially with valve position
       - Mixed (0 < a < 1): Combination of linear and equal percentage characteristics

    Implementation Details:
       - The model uses PyTorch tensors for gradient-based optimization
       - All parameters are stored as non-trainable PyTorch parameters
       - The valve authority equation provides better control at low flow rates
       - The model assumes ideal valve behavior (no hysteresis or deadband)
    """
    sp = [get_signature_pattern()]
    def __init__(self, 
                waterFlowRateMax: float = 1000/((60-45)*4180), #Provide 1000 W of heating power when cooling from 60 to 45 degrees
                valveAuthority: float = 1, # Linear relation by default
                **kwargs):
        """
        Initialize the valve system model.
        
        Args:
            waterFlowRateMax: Maximum water flow rate [kg/s]
            valveAuthority: Valve authority (0-1)
        """
        super().__init__(**kwargs)
        nn.Module.__init__(self)
        
        # Store parameters as nn.Parameters for gradient tracking
        self.waterFlowRateMax = nn.Parameter(torch.tensor(waterFlowRateMax, dtype=torch.float32), requires_grad=False)
        self.valveAuthority = nn.Parameter(torch.tensor(valveAuthority, dtype=torch.float32), requires_grad=False)
        
        # Define inputs and outputs
        self.input = {"valvePosition": tps.Scalar()}
        self.output = {"valvePosition": tps.Scalar(),
                      "waterFlowRate": tps.Scalar()}
        
        # Define parameters for calibration
        self.parameter = {
            "waterFlowRateMax": {"lb": 0.0, "ub": 10.0},
            "valveAuthority": {"lb": 0.0, "ub": 1.0}
        }
        
        self._config = {"parameters": list(self.parameter.keys())}
        self.INITIALIZED = False
    
    @property
    def config(self):
        """Get the configuration of the valve system."""
        return self._config
    
    def cache(self, startTime=None, endTime=None, stepSize=None):
        """Cache method placeholder."""
        pass
    
    def initialize(self, 
                   startTime=None, 
                   endTime=None, 
                   stepSize=None, 
                   simulator=None):
        """Initialize the valve system."""
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
        Perform one step of the valve system simulation.
        
        The valve characteristic is calculated using the valve authority equation:
        u_norm = u / sqrt(u^2 * (1-a) + a)
        where:
        - u is the valve position (0-1)
        - a is the valve authority (0-1)
        - u_norm is the normalized valve position
        
        The water flow rate is then calculated as:
        m_w = u_norm * waterFlowRateMax
        """
        # Get input valve position (assumed to be a tensor)
        valve_position = self.input["valvePosition"].get()
        
        # Calculate normalized valve position using valve authority equation
        u_norm = valve_position / torch.sqrt(valve_position**2 * (1 - self.valveAuthority) + self.valveAuthority)
        
        # Calculate water flow rate
        m_w = u_norm * self.waterFlowRateMax
        
        # Update outputs
        self.output["valvePosition"].set(valve_position, stepIndex)
        self.output["waterFlowRate"].set(m_w, stepIndex)