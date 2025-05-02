import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Union, Tuple

from twin4build import core
import twin4build.utils.input_output_types as tps

class DiscreteStatespaceSystem(core.System):
    """
    A general-purpose discrete state space system for modeling dynamical systems.
    
    This model implements a discrete state space representation:
    
    Continuous form:
        x'(t) = A*x(t) + B*u(t)
        y(t) = C*x(t) + D*u(t)
    
    Discretized form:
        x[k+1] = Ad*x[k] + Bd*u[k]
        y[k] = Cd*x[k] + Dd*u[k]
    
    This implementation is fully generalized and can be configured for any linear system.
    """
    
    def __init__(self,
                 A: torch.Tensor = None,           # Continuous state matrix
                 B: torch.Tensor = None,           # Continuous input matrix
                 C: torch.Tensor = None,           # Continuous output matrix
                 D: torch.Tensor = None,           # Continuous feedthrough matrix
                 sample_time: float = 1.0,         # Sampling time for discretization
                 x0: torch.Tensor = None,          # Initial state vector
                 state_names: List[str] = None,    # Names of states
                 **kwargs):
        """
        Initialize a DiscreteStatespaceSystem object.
        
        Args:
            A (torch.Tensor): System dynamics matrix
            B (torch.Tensor): Control input matrix
            C (torch.Tensor): Output matrix
            D (torch.Tensor): Feedthrough matrix (defaults to zeros if None)
            sample_time (float): Sampling time for discretization
            x0 (torch.Tensor): Initial state vector
            state_names (List[str]): Names for system states
            **kwargs: Additional keyword arguments
        """
        super().__init__(**kwargs)
        
        # Verify and store continuous system matrices
        if A is not None and B is not None and C is not None:
            self.A = A
            self.B = B
            self.C = C
            self.D = D if D is not None else torch.zeros((C.shape[0], B.shape[1]), dtype=torch.float32)
            
            # State and I/O dimensions
            self.n_states = self.A.shape[0]
            self.n_inputs = self.B.shape[1]
            self.n_outputs = self.C.shape[0]
        else:
            raise ValueError("System matrices A, B, and C must be provided")
        
        # Store sample time as a regular attribute
        self.sample_time = sample_time
        

        self.x0 = x0 if x0 is not None else torch.zeros(self.n_states, dtype=torch.float32)

        # Current state
        self.x = x0 
        
        # Names for states
        self.state_names = state_names if state_names is not None else [f"x{i}" for i in range(self.n_states)]
        
        # Ensure state names list has correct length
        if len(self.state_names) != self.n_states:
            raise ValueError(f"state_names should have length {self.n_states}, got {len(self.state_names)}")
        
        # Set up inputs and outputs
        self.input = {"u": tps.Vector(self.n_inputs)}  # Single input vector
        self.output = {"y": tps.Vector(self.n_outputs)}  # Single output vector
        
        # Define parameters for potential calibration
        self.parameter = {
            # Additional parameters could be added for matrix entries
        }
        
        self._config = {"parameters": list(self.parameter.keys())}
        
        # Initialize discretized matrices
        self.discretize_system()
        self.INITIALIZED = True
    
    @property
    def config(self):
        """Get the configuration of the DiscreteStatespaceSystem."""
        return self._config
    
    def discretize_system(self) -> None:
        """
        Discretize the continuous-time state space model while maintaining the computational graph.
        
        Uses the matrix exponential method to compute Ad and Bd efficiently.
        The implementation ensures that gradients can flow back through the discretization process.
        """
        T = self.sample_time
        n = self.n_states

        # Compute Ad using matrix exponential
        self.Ad = torch.matrix_exp(self.A * T)

        # Compute Bd using the analytical formula
        # (A must be invertible; for thermal systems, this is almost always true)
        I = torch.eye(n, dtype=self.A.dtype, device=self.A.device)
        A_inv = torch.linalg.inv(self.A)
        self.Bd = A_inv @ (self.Ad - I) @ self.B

        self.Cd = self.C
        self.Dd = self.D
    
    def cache(self, startTime=None, endTime=None, stepSize=None) -> None:
        """Cache method placeholder."""
        pass
    
    def initialize(self, startTime=None, endTime=None, stepSize=None, model=None) -> None:
        """
        Initialize the discrete state space model by computing discretized matrices.
        
        Args:
            startTime: Simulation start time.
            endTime: Simulation end time.
            stepSize: Simulation step size.
            model: Reference to the simulation model.
        """
        print("========== INITIALIZING STATE SPACE SYSTEM ==========")
        self.input["u"].initialize()
        self.output["y"].initialize()
        if self.INITIALIZED:
            # Reset the state if already initialized
            self.x = self.x0
        else:
            # Discretize the system
            self.discretize_system()
            self.INITIALIZED = True
    
    def do_step(self, secondTime=None, dateTime=None, stepSize=None) -> None:
        """
        Perform one step of the state space model simulation.
        
        Args:
            secondTime: Current simulation time in seconds.
            dateTime: Current simulation date/time.
            stepSize: Current simulation step size.
        """
        if stepSize != self.sample_time:
            # Update sample time and recompute discretized matrices
            self.sample_time = stepSize
            self.discretize_system()

        # Extract inputs from input dictionary
        u = self.input["u"].get()
        
        # Propagate the state according to the discrete state space model
        # x[k+1] = Ad*x[k] + Bd*u[k]
        self.x = self.Ad @ self.x + self.Bd @ u

        # Calculate outputs
        # y[k] = Cd*x[k] + Dd*u[k]
        y = self.Cd @ self.x + self.Dd @ u
        
        # Update output dictionary
        self.output["y"].set(y)
    
    @classmethod
    def from_matrices(cls, A, B, C, D=None, sample_time=1.0, **kwargs):
        """
        Create a DiscreteStatespaceSystem from continuous-time matrices.
        
        Args:
            A: System dynamics matrix
            B: Control input matrix
            C: Output matrix
            D: Feedthrough matrix (optional)
            sample_time: Sampling time for discretization
            **kwargs: Additional arguments to pass to constructor
            
        Returns:
            DiscreteStatespaceSystem: Initialized system
        """
        return cls(A=A, B=B, C=C, D=D, sample_time=sample_time, **kwargs)
    
    @classmethod
    def from_transfer_function(cls, num, den, sample_time=1.0, **kwargs):
        """
        Create a DiscreteStatespaceSystem from a transfer function.
        
        Args:
            num: Transfer function numerator polynomial coefficients
            den: Transfer function denominator polynomial coefficients
            sample_time: Sampling time for discretization
            **kwargs: Additional arguments to pass to constructor
            
        Returns:
            DiscreteStatespaceSystem: Initialized system
        """
        # Convert transfer function to state space
        from scipy import signal
        A, B, C, D = signal.tf2ss(num, den)
        return cls(A=A, B=B, C=C, D=D, sample_time=sample_time, **kwargs)
    
    def get_state(self) -> torch.Tensor:
        """Get the current state vector."""
        return self.x.clone()
    
    def set_state(self, x: torch.Tensor) -> None:
        """
        Set the current state vector.
        
        Args:
            x: New state vector
        """
        if x.shape != self.x.shape:
            raise ValueError(f"State vector should have shape {self.x.shape}, got {x.shape}")
        self.x = x.clone()




