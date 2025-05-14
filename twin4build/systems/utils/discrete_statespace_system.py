import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Union, Tuple

from twin4build import core
import twin4build.utils.input_output_types as tps

class DiscreteStatespaceSystem(core.System):
    """
    A general-purpose discrete state space system for modeling dynamical systems.
    Now supports bilinear (state-input coupled) terms via an E tensor,
    and input-input coupled terms via an F tensor.
    
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
                 E: torch.Tensor = None,           # State-input coupling (M,N,N)
                 F: torch.Tensor = None,           # Input-input coupling (M,M,N)
                 **kwargs):
        """
        Initialize a DiscreteStatespaceSystem object.
        
        Args:
            A (torch.Tensor): System dynamics matrix of shape (N, N)
            B (torch.Tensor): Control input matrix of shape (N, M)
            C (torch.Tensor): Output matrix of shape (P, N)
            D (torch.Tensor): Feedthrough matrix of shape (P, M). Optional.
            sample_time (float): Sampling time for discretization
            x0 (torch.Tensor): Initial state vector of shape (N,)
            state_names (List[str]): Names for system states
            E (torch.Tensor): Bilinear state-input tensor of shape (M, N, N). Optional.
            F (torch.Tensor): Input-input coupling tensor of shape (M, M, N). Optional.
            **kwargs: Additional keyword arguments
        """
        super().__init__(**kwargs)
        
        # Verify and store continuous system matrices
        if A is not None and B is not None and C is not None:
            _A = A
            _B = B
            _C = C
            _D = D if D is not None else torch.zeros((C.shape[0], B.shape[1]), dtype=torch.float32)

            self._A_base = _A.clone() # We need a base matrix because we change them dynamically in do_step
            self._B_base = _B.clone() # We need a base matrix because we change them dynamically in do_step
            self._C = _C.clone()
            self._D = _D.clone()
            
            # State and I/O dimensions
            self.n_states = self._A_base.shape[0]
            self.n_inputs = self._B_base.shape[1]
            self.n_outputs = self._C.shape[0]
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
        self.input = {"u": tps.Vector(size=self.n_inputs)}  # Single input vector
        self.output = {"y": tps.Vector(size=self.n_outputs)}  # Single output vector
        
        # Define parameters for potential calibration
        self.parameter = {
            # Additional parameters could be added for matrix entries
        }
        
        self._config = {"parameters": list(self.parameter.keys())}
        
        # Initialize discretized matrices
        # self.discretize_system()
        self.INITIALIZED = True
        
        self._E = E  # shape (M, N, N) or None
        self._F = F  # shape (M, M, N) or None
        
        # Store base matrices for each step
        # self._A_base = self._A.clone()
        # self._B_base = self._B.clone()
        self._prev_u = None  # For input change detection
    
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
        self.Ad = torch.matrix_exp(self._A * T)

        # Compute Bd using the analytical formula
        # (A must be invertible; for thermal systems, this is almost always true)
        I = torch.eye(n, dtype=self._A.dtype, device=self._A.device)
        A_inv = torch.linalg.inv(self._A)
        self.Bd = A_inv @ (self.Ad - I) @ self._B

        self.Cd = self._C
        self.Dd = self._D
    
    def cache(self, startTime=None, endTime=None, stepSize=None) -> None:
        """Cache method placeholder."""
        pass
    
    def initialize(self, 
                   startTime=None, 
                   endTime=None, 
                   stepSize=None, 
                   simulator=None) -> None:
        """
        Initialize the discrete state space model by computing discretized matrices.
        
        Args:
            startTime: Simulation start time.
            endTime: Simulation end time.
            stepSize: Simulation step size.
            model: Reference to the simulation model.
        """
        # Reset and initialize I/O
        self.input["u"].reset()
        self.input["u"].initialize()
        self.output["y"].reset()
        self.output["y"].initialize()

        # Refresh all base matrices with fresh computational graphs
        # self._A_base = self._A_base.detach()
        # self._B_base = self._B_base.detach()
        # self._C_base = self._C.detach().clone()
        # self._D_base = self._D.detach().clone()
        
        # Reset state to initial state and detach from old graph
        self.x = self.x0.detach().clone()
        
        # Clear any previous computational graph
        self._prev_u = None
            
        # if not self.INITIALIZED:
        #     # Discretize the system
        #     self.discretize_system()
        #     self.INITIALIZED = True
        # else:
            # Re-discretize with fresh matrices
            # self.discretize_system()
    
    def do_step(self, 
                secondTime=None, 
                dateTime=None, 
                stepSize=None,
                stepIndex: Optional[int] = None) -> None:
        """
        Perform one step of the state space model simulation.
        Now supports bilinear (state-input coupled) terms using the trapezoidal (average of old and new states) method for the E and F terms.
        Ad and Bd are only computed in discretize_system for efficiency.
        """
        if stepSize != self.sample_time:
            self.sample_time = stepSize
        u = self.input["u"].get()
        x = self.x

        # Note: If we have a model with tiny inputs, we may need to use a different tolerance for the comparison
        # Warning: In this case, the simulation may silently fail to update the state space matrices
        if self._prev_u is None or torch.allclose(u, self._prev_u)==False:
            # Compute effective A
            A_eff = self._A_base.clone()
            if self._E is not None:
                A_eff += torch.einsum('mij,m->ij', self._E, u)
            # Compute effective B
            B_eff = self._B_base.clone()
            if self._F is not None:
                B_eff += torch.einsum('mij,m->ij', self._F, u)
            # Discretize with new A and B
            self._A = A_eff
            self._B = B_eff
            self.discretize_system()
            self._prev_u = u.clone()
        # State update
        x_new = self.Ad @ x + self.Bd @ u
        self.x = x_new
        # Output
        y = self.Cd @ self.x + self.Dd @ u
        self.output["y"].set(y, stepIndex)
    
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




