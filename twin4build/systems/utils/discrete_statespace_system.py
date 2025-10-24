# Standard library imports
import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

# Third party imports
import torch
import torch.nn as nn

# Local application imports
import twin4build.utils.types as tps
from twin4build import core


class DiscreteStatespaceSystem(core.System):
    r"""
    A general-purpose discrete state space system for modeling dynamical systems with batch support.

    This class implements a discrete state-space system that supports both linear and bilinear
    dynamics through state-input and input-input coupling terms. The system serves as the
    computational core for various physical models in the Twin4Build framework, including
    thermal RC networks and mass balance systems.
    
    **NESTED BATCH DIMENSION SUMMARY:**
    ===================================
    
    This system supports nested batch operations with two batch dimensions:
    
    1. **System Batch Dimension**: Different system configurations (A, B, C, D matrices)
    2. **Simulation Batch Dimension**: Parallel simulations of each system configuration
    
    **Total Batch Size = sim_batch_size × system_batch_size**
    
    Core matrices (after expansion):
        - A: (sim_batch_size × system_batch_size, n_states, n_states) - System dynamics matrix
        - B: (sim_batch_size × system_batch_size, n_states, n_inputs) - Input matrix  
        - C: (sim_batch_size × system_batch_size, n_outputs, n_states) - Output matrix
        - D: (sim_batch_size × system_batch_size, n_outputs, n_inputs) - Feedthrough matrix
        
    Bilinear matrices (optional, after expansion):
        - E: (sim_batch_size × system_batch_size, n_inputs, n_states, n_states) - State-input coupling
        - F: (sim_batch_size × system_batch_size, n_inputs, n_states, n_inputs) - Input-input coupling
        
    State and I/O vectors (after expansion):
        - x: (sim_batch_size × system_batch_size, n_states) - State vector
        - u: (sim_batch_size × system_batch_size, n_inputs) - Input vector
        - y: (sim_batch_size × system_batch_size, n_outputs) - Output vector
        
    **Expansion Pattern (sim_batch_size first):**
    Simulation batches cycle through all systems:
    [sim0_sys0, sim0_sys1, ..., sim0_sysN, sim1_sys0, sim1_sys1, ..., sim1_sysN, ...]
     |------- all systems for sim0 -------|  |------- all systems for sim1 -------|

    
    Args:
        A: System dynamics matrix of shape (batch_size, N, N) or (N, N)
        B: Control input matrix of shape (batch_size, N, M) or (N, M)
        C: Output matrix of shape (batch_size, P, N) or (P, N)
        D: Feedthrough matrix of shape (batch_size, P, M) or (P, M). Optional.
        sample_time: Sampling time for discretization
        x0: Initial state vector of shape (batch_size, N) or (N,)
        state_names: Names for system states
        E: Bilinear state-input tensor of shape (batch_size, M, N, N) or (M, N, N). Optional.
        F: Input-input coupling tensor of shape (batch_size, M, M, N) or (M, M, N). Optional.
        **kwargs: Additional keyword arguments
        
    Note:
        This class supports batch operations for parallel simulation of multiple instances.
        When batch_size > 1, all matrices and vectors are automatically expanded to include
        the batch dimension. Input matrices can be provided either with or without the batch
        dimension - if provided without, they will be automatically broadcasted.
        
        **Dynamic Batch Expansion:**
        The system can dynamically expand its batch size during the `initialize()` method
        to match the simulation batch size. This allows creating a system with batch_size=1
        and then expanding it to simulate multiple instances in parallel.

    Mathematical Formulation:
    =========================

    **Continuous-Time State-Space Representation:**

    The general continuous-time state-space system with bilinear terms is formulated as:

    .. math::

       \frac{d\mathbf{x}}{dt} = \mathbf{A}\mathbf{x} + \mathbf{B}\mathbf{u} + \sum_{i=1}^{m} \mathbf{E}_i\mathbf{x}u_i + \sum_{i=1}^{m} \mathbf{F}_{i}\mathbf{u} u_i

    .. math::

       \mathbf{y} = \mathbf{C}\mathbf{x} + \mathbf{D}\mathbf{u}

    where:

       - :math:`\mathbf{x} \in \mathbb{R}^n`: State vector (internal system variables)
       - :math:`\mathbf{u} \in \mathbb{R}^m`: Input vector (external driving signals)
       - :math:`\mathbf{y} \in \mathbb{R}^p`: Output vector (observable quantities)
       - :math:`\mathbf{A} \in \mathbb{R}^{n \times n}`: State transition matrix
       - :math:`\mathbf{B} \in \mathbb{R}^{n \times m}`: Input matrix
       - :math:`\mathbf{C} \in \mathbb{R}^{p \times n}`: Output matrix
       - :math:`\mathbf{D} \in \mathbb{R}^{p \times m}`: Feedthrough matrix
       - :math:`\mathbf{E} \in \mathbb{R}^{m \times n \times n}`: State-input coupling tensor, with :math:`\mathbf{E}_i \in \mathbb{R}^{n \times n}` being the :math:`i`-th slice of the tensor
       - :math:`\mathbf{F} \in \mathbb{R}^{m \times n \times m}`: Input-input coupling tensor, with :math:`\mathbf{F}_i \in \mathbb{R}^{n \times m}` being the :math:`i`-th slice of the tensor

    **Bilinear Extensions:**

    The bilinear terms extend the basic linear state-space model to handle:

    *State-Input Coupling (E matrices):*
       - Models where inputs affect the dynamics matrix
       - Example: :math:`\dot{m}_{exh} \times T_{air}` in thermal systems
       - Formulation: :math:`\sum_{i=1}^{m} \mathbf{E}_i\mathbf{x}u_i`

    *Input-Input Coupling (F matrices):*
       - Models where the product of two inputs affects the state derivative
       - Example: :math:`\dot{m}_{sup} \times T_{sup}` in thermal systems
       - Formulation: :math:`\sum_{i=1}^{m} \mathbf{F}_{i}\mathbf{u} u_i`

    **Discretization Method:**

    For numerical simulation, the continuous system is discretized using zero-order hold (ZOH).
    For a linear system, this would be a one-time operation.
    However, when bilinear terms (E and F matrices) are present, the effective A and B matrices
    must be recomputed every time inputs change significantly.

    *Step 1: Compute Equivalent Matrices*

    
    We can calculate the \textit{equivalent} A and B matrices by factoring out the state and input vectors :math:`\mathbf{x}` and :math:`\mathbf{u}`:

    .. math::

       \mathbf{A}^*[k] = \mathbf{A} + \sum_{i=1}^{m} \mathbf{E}_i u_i[k]

    .. math::

       \mathbf{B}^*[k] = \mathbf{B} + \sum_{i=1}^{m} \mathbf{F}_i u_i[k]

    where the effective matrices depend on the current input vector :math:`\mathbf{u}[k]`.

    *Step 2: Discretize Effective Matrices*

    The effective matrices are then discretized using the matrix exponential method:

    .. math::

       \mathbf{A}_d[k] = e^{\mathbf{A}_{eff}[k] T_s}

       \mathbf{B}_d[k] = \int_0^{T_s} e^{\mathbf{A}_{eff}[k]\tau}d\tau \mathbf{B}_{eff}[k]

    *Step 3: State Update*

    The discrete-time state update becomes:

    .. math::

       \mathbf{x}[k+1] = \mathbf{A}_d[k]\mathbf{x}[k] + \mathbf{B}_d[k]\mathbf{u}[k]

       \mathbf{y}[k] = \mathbf{C}\mathbf{x}[k] + \mathbf{D}\mathbf{u}[k]

    where :math:`T_s` is the sampling time. This approach ensures that the bilinear coupling
    effects are properly incorporated into the discrete-time dynamics while preserving
    numerical accuracy through the matrix exponential method.

    **Computational Efficiency:**

    The effective matrices and their discretization are recomputed only when the input
    vector changes significantly, providing computational efficiency while maintaining
    accuracy for time-varying bilinear systems.

    **Practical Implementation:**

    In practice, the matrix exponential computation is performed using a block matrix approach
    for numerical stability:

    .. math::

       \mathbf{M} = \begin{bmatrix}
           \mathbf{A}_{eff}[k] T_s & \mathbf{B}_{eff}[k] T_s \\
           \mathbf{0} & \mathbf{0}
       \end{bmatrix}

       e^{\mathbf{M}} = \begin{bmatrix}
           \mathbf{A}_d[k] & \mathbf{B}_d[k] \\
           \mathbf{0} & \mathbf{I}
       \end{bmatrix}

    Physical Interpretation:
    =======================

    **In Thermal Systems:**
       - States: Temperatures of thermal nodes (air, walls, etc.)
       - Inputs: Weather conditions, HVAC flows, heat gains
       - A matrix: Thermal coupling between nodes via resistances
       - B matrix: External heat inputs and boundary conditions
       - E/F matrices: Flow-dependent heat transfer

    **In Mass Balance Systems:**
       - States: Concentration levels (CO2, humidity, etc.)
       - Inputs: Ventilation flows, generation rates, outdoor conditions
       - A matrix: Dilution and mixing effects
       - B matrix: Source terms and boundary inflows
       - E/F matrices: Flow-dependent transport

    Computational Features:
    ======================

       - **Automatic Differentiation:** PyTorch tensors enable gradient computation
       - **Adaptive Discretization:** Matrices updated when inputs change significantly
       - **Numerical Stability:** Matrix exponential method for accurate discretization
       - **Efficient Simulation:** Optimized for repeated time-stepping


    Examples
    --------
    Basic linear state-space system (single instance):

    >>> import torch
    >>> import twin4build as tb
    >>>
    >>> # Define system matrices (no batch dimension)
    >>> A = torch.tensor([[-0.1, 0.05], [0.02, -0.08]], dtype=torch.float64)  # (2, 2)
    >>> B = torch.tensor([[1.0], [0.5]], dtype=torch.float64)  # (2, 1)
    >>> C = torch.tensor([[1.0, 0.0]], dtype=torch.float64)  # (1, 2)
    >>> x0 = torch.tensor([20.0, 18.0], dtype=torch.float64)  # (2,)
    >>>
    >>> # Create system (automatically adds batch dimension)
    >>> system = tb.DiscreteStatespaceSystem(
    ...     A=A, B=B, C=C, x0=x0, sample_time=3600.0,
    ...     state_names=["T_air", "T_wall"]
    ... )
    >>> # Resulting tensors: A(1,2,2), B(1,2,1), C(1,1,2), x0(1,2)

    Batch system for parallel simulation:

    >>> # Define batch system matrices
    >>> batch_size = 3
    >>> A_batch = torch.randn(batch_size, 2, 2, dtype=torch.float64)  # (3, 2, 2)
    >>> B_batch = torch.randn(batch_size, 2, 1, dtype=torch.float64)  # (3, 2, 1)
    >>> C_batch = torch.randn(batch_size, 1, 2, dtype=torch.float64)  # (3, 1, 2)
    >>> x0_batch = torch.randn(batch_size, 2, dtype=torch.float64)    # (3, 2)
    >>>
    >>> # Create batch system
    >>> batch_system = tb.DiscreteStatespaceSystem(
    ...     A=A_batch, B=B_batch, C=C_batch, x0=x0_batch, sample_time=3600.0,
    ...     state_names=["T_air", "T_wall"]
    ... )

    Nested batch expansion during simulation:

    >>> # Create system with 2 different configurations
    >>> A_batch = torch.randn(2, 2, 2, dtype=torch.float64)  # 2 different A matrices
    >>> B_batch = torch.randn(2, 2, 1, dtype=torch.float64)  # 2 different B matrices  
    >>> C_batch = torch.randn(2, 1, 2, dtype=torch.float64)  # 2 different C matrices
    >>> system = tb.DiscreteStatespaceSystem(A=A_batch, B=B_batch, C=C_batch, sample_time=3600.0)
    >>> print(f"System batch size: {system.system_batch_size}")  # 2
    >>> print(f"Sim batch size: {system.sim_batch_size}")      # 1
    >>> print(f"Total batch size: {system.batch_size}")        # 2
    >>>
    >>> # Initialize with 3 parallel simulations - expands to 3×2=6 total
    >>> start_times = [datetime(2024,1,1)] * 3  # 3 parallel simulations
    >>> system.initialize(start_times, end_times, step_sizes)
    >>> print(f"After expansion - Total batch size: {system.batch_size}")  # 6
    >>> # Result: [sim0_sys0, sim0_sys1, sim1_sys0, sim1_sys1, sim2_sys0, sim2_sys1]

    Bilinear system with state-input coupling:

    >>> # Define bilinear coupling matrices
    >>> E = torch.zeros((1, 2, 2), dtype=torch.float64)
    >>> E[0, 0, 1] = 0.001  # Input 0 affects coupling between states 0 and 1
    >>>
    >>> # Create bilinear system
    >>> bilinear_system = tb.DiscreteStatespaceSystem(
    ...     A=A, B=B, C=C, E=E, x0=x0, sample_time=3600.0,
    ...     state_names=["T_air", "T_wall"]
    ... )

    System with input-input coupling:

    >>> # Define input-input coupling
    >>> F = torch.zeros((2, 2, 2), dtype=torch.float64)
    >>> F[0, 1, 0] = 0.1  # Product of inputs 0 and 1 affects state 0
    >>>
    >>> # Create system with F matrices
    >>> coupled_system = tb.DiscreteStatespaceSystem(
    ...     A=A, B=B, C=C, F=F, x0=x0, sample_time=3600.0,
    ...     state_names=["T_air", "T_wall"]
    ... )
    """

    @staticmethod
    def _expand_to_batch(tensor, target_shape, batch_size):
        """Expand tensor to include batch dimension if needed."""
        if tensor is None:
            return None
        
        # If tensor already has batch dimension, verify it matches
        if len(tensor.shape) == len(target_shape):
            if tensor.shape[0] != batch_size and tensor.shape[0] != 1:
                raise ValueError(f"Batch dimension mismatch: expected {batch_size} or 1, got {tensor.shape[0]}")
            # Expand if batch_size is 1 but we need larger batch
            if tensor.shape[0] == 1 and batch_size > 1:
                expand_dims = [batch_size] + [-1] * (len(tensor.shape) - 1)
                return tensor.expand(*expand_dims).contiguous()  # Keep contiguous here - expand may not be contiguous
            return tensor
        
        # If tensor doesn't have batch dimension, add it
        elif len(tensor.shape) == len(target_shape) - 1:
            return tensor.unsqueeze(0).expand(batch_size, *tensor.shape).contiguous()
        else:
            raise ValueError(f"Tensor shape {tensor.shape} incompatible with target shape {target_shape}")
    
    @staticmethod
    def _expand_to_nested_batch(tensor, system_batch_size, sim_batch_size):
        """
        Expand tensor to nested batch structure: (sim_batch_size, system_batch_size, ...).
        
        The simulation batch dimension comes first, then system batch dimension.
        This allows for easier indexing where sim_batch_size varies dynamically.
        
        Args:
            tensor: Input tensor with shape (system_batch_size, ...)
            system_batch_size: Number of different system configurations (preserved)
            sim_batch_size: Number of parallel simulations per system
            
        Returns:
            Expanded tensor with shape (sim_batch_size * system_batch_size, ...)
            where the pattern is [sim0_sys0, sim0_sys1, ..., sim1_sys0, sim1_sys1, ...]
        """
        if tensor is None:
            return None
        
        current_system_batch = tensor.shape[0]
        
        # Verify system batch size matches
        if current_system_batch != system_batch_size:
            raise ValueError(f"System batch size mismatch: expected {system_batch_size}, got {current_system_batch}")
        
        # If sim_batch_size is 1, no expansion needed
        if sim_batch_size == 1:
            return tensor
        
        # Expand to nested structure: (sim_batch_size, system_batch_size, ...)
        # Original shape: (system_batch_size, ...)
        # Target shape: (sim_batch_size * system_batch_size, ...)
        
        # Method: Create tensor with sim_batch_size first, system_batch_size second
        # Step 1: Add simulation batch dimension at the front
        # (system_batch_size, ...) -> (1, system_batch_size, ...)
        tensor_expanded = tensor.unsqueeze(0)
        
        # Step 2: Expand along simulation dimension (first dimension)
        # (1, system_batch_size, ...) -> (sim_batch_size, system_batch_size, ...)
        remaining_dims = [-1] * (len(tensor.shape))
        expand_shape = [sim_batch_size] + remaining_dims
        tensor_replicated = tensor_expanded.expand(*expand_shape)
        
        # Step 3: Flatten to target shape
        # (sim_batch_size, system_batch_size, ...) -> (sim_batch_size * system_batch_size, ...)
        target_shape = (sim_batch_size * system_batch_size,) + tensor.shape[1:]
        return tensor_replicated.reshape(target_shape)

    def __init__(
        self,
        A: torch.Tensor = None,  # Continuous state matrix
        B: torch.Tensor = None,  # Continuous input matrix
        C: torch.Tensor = None,  # Continuous output matrix
        D: torch.Tensor = None,  # Continuous feedthrough matrix
        sample_time: float = 1.0,  # Sampling time for discretization
        x0: torch.Tensor = None,  # Initial state vector
        state_names: List[str] = None,  # Names of states
        E: torch.Tensor = None,  # State-input coupling (M,N,N)
        F: torch.Tensor = None,  # Input-input coupling (M,M,N)
        **kwargs,
    ):
        """
        Initialize a DiscreteStatespaceSystem object.

        Args:
            A: System dynamics matrix of shape (batch_size, N, N) or (N, N)
            B: Control input matrix of shape (batch_size, N, M) or (N, M)
            C: Output matrix of shape (batch_size, P, N) or (P, N)
            D: Feedthrough matrix of shape (batch_size, P, M) or (P, M). Optional.
            sample_time: Sampling time for discretization
            x0: Initial state vector of shape (batch_size, N) or (N,)
            state_names: Names for system states
            E: Bilinear state-input tensor of shape (batch_size, M, N, N) or (M, N, N). Optional.
            F: Input-input coupling tensor of shape (batch_size, M, M, N) or (M, M, N). Optional.
            **kwargs: Additional keyword arguments
        """
        super().__init__(**kwargs)

        # Verify and store continuous system matrices
        if A is not None and B is not None and C is not None:
            # Determine batch size from input matrices
            # Priority: A matrix batch size > B matrix batch size > 1
            batch_size = 1
            if len(A.shape) == 3:  # A has batch dimension
                batch_size = A.shape[0]
            elif len(B.shape) == 3:  # B has batch dimension
                batch_size = B.shape[0]
            elif len(C.shape) == 3:  # C has batch dimension
                batch_size = C.shape[0]
            
            # Determine base dimensions (without batch)
            n_states = A.shape[-2]  # Last two dimensions are (n_states, n_states)
            n_inputs = B.shape[-1]  # Last dimension is n_inputs
            n_outputs = C.shape[-2]  # Second to last dimension is n_outputs
            
            # Expand all matrices to batch dimensions
            _A = self._expand_to_batch(A, (batch_size, n_states, n_states), batch_size)
            _B = self._expand_to_batch(B, (batch_size, n_states, n_inputs), batch_size)
            _C = self._expand_to_batch(C, (batch_size, n_outputs, n_states), batch_size)
            
            # Handle D matrix
            if D is not None:
                _D = self._expand_to_batch(D, (batch_size, n_outputs, n_inputs), batch_size)
            else:
                _D = torch.zeros((batch_size, n_outputs, n_inputs), dtype=torch.float64)

            # Store matrices (we need base matrices because we change them dynamically in do_step)
            self._A = _A.clone()
            self._B = _B.clone()
            self._C = _C.clone()
            self._D = _D.clone()

            self._A_base = _A.clone()
            self._B_base = _B.clone()

            # Store dimensions
            self.system_batch_size = batch_size  # Number of different system configurations
            self.batch_size = batch_size         # Total batch size (will be system_batch_size * sim_batch_size)
            self.sim_batch_size = 1              # Number of parallel simulations per system (default 1)
            self.n_states = n_states
            self.n_inputs = n_inputs
            self.n_outputs = n_outputs
        else:
            raise ValueError("System matrices A, B, and C must be provided")

        # Store sample time as a regular attribute
        self.sample_time = sample_time

        # Handle initial state with batch dimension
        if x0 is not None:
            self.x0 = self._expand_to_batch(x0, (self.system_batch_size, self.n_states), self.system_batch_size)
        else:
            self.x0 = torch.zeros((self.system_batch_size, self.n_states), dtype=torch.float64)

        # Current state (system_batch_size, n_states) - will be expanded during initialize()
        self.x = self.x0.clone()

        # Names for states
        self.state_names = (
            state_names
            if state_names is not None
            else [f"x{i}" for i in range(self.n_states)]
        )

        # Ensure state names list has correct length
        if len(self.state_names) != self.n_states:
            raise ValueError(
                f"state_names should have length {self.n_states}, got {len(self.state_names)}"
            )

        # Set up inputs and outputs as private variables with batch support
        self._input = {"u": tps.Vector(size=self.n_inputs)}  # Input vector (batch_size, n_inputs)
        self._output = {"y": tps.Vector(size=self.n_outputs)}  # Output vector (batch_size, n_outputs)

        # Define parameters for potential calibration
        self.parameter = {
            # Additional parameters could be added for matrix entries
        }

        self._config = {"parameters": list(self.parameter.keys())}

        # Initialize discretized matrices
        # self.discretize_system()
        self.INITIALIZED = True

        # Handle bilinear matrices with batch dimensions
        if E is not None:
            # E: (system_batch_size, M, N, N) or (M, N, N)
            self._E = self._expand_to_batch(E, (self.system_batch_size, self.n_inputs, self.n_states, self.n_states), self.system_batch_size)
            # Check which input indices have non-zero E matrices (across all system batch elements)
            self.non_zero_E = torch.zeros(self.n_inputs, dtype=torch.bool)
            for i in range(self.n_inputs):
                self.non_zero_E[i] = torch.any(self._E[:, i, :, :])
        else:
            self._E = None
            self.non_zero_E = torch.zeros(0, dtype=torch.bool)

        if F is not None:
            # F: (system_batch_size, M, N, M) or (M, N, M) 
            self._F = self._expand_to_batch(F, (self.system_batch_size, self.n_inputs, self.n_states, self.n_inputs), self.system_batch_size)
            # Check which input indices have non-zero F matrices (across all system batch elements)
            self.non_zero_F = torch.zeros(self.n_inputs, dtype=torch.bool)
            for i in range(self.n_inputs):
                self.non_zero_F[i] = torch.any(self._F[:, i, :, :])
        else:
            self._F = None
            self.non_zero_F = torch.zeros(0, dtype=torch.bool)
            
        self._prev_u = None  # For input change detection (total_batch_size, n_inputs)

    @property
    def config(self):
        """
        Get the configuration parameters of the discrete state-space system.

        Returns:
            dict: Configuration parameters including all system matrices.
        """
        return self._config

    @property
    def input(self) -> dict:
        """
        Get the input ports of the discrete state-space system.

        Returns:
            dict: Dictionary containing input ports:
                - "u": Input vector of size n_inputs
        """
        return self._input

    @property
    def output(self) -> dict:
        """
        Get the output ports of the discrete state-space system.

        Returns:
            dict: Dictionary containing output ports:
                - "y": Output vector of size n_outputs
        """
        return self._output

    def discretize_system(self) -> None:
        """
        Discretize the continuous-time state space model while maintaining the computational graph.

        Uses the matrix exponential method to compute Ad and Bd efficiently for batch operations.
        The implementation ensures that gradients can flow back through the discretization process.
        
        Tensor shapes:
            self._A: (batch_size, n_states, n_states)
            self._B: (batch_size, n_states, n_inputs)
            M: (batch_size, n_states + n_inputs, n_states + n_inputs)
            Ad: (batch_size, n_states, n_states)
            Bd: (batch_size, n_states, n_inputs)
        """
        T = self.sample_time
        batch_size = self.batch_size
        n = self.n_states
        m = self.n_inputs

        # Create block matrix for batch matrix exponential
        # M shape: (batch_size, n + m, n + m)
        ###
        M = torch.zeros((batch_size, n + m, n + m), dtype=self._A.dtype, device=self._A.device)
        M[:, :n, :n] = self._A * T  # A block: (batch_size, n, n)
        M[:, :n, n:] = self._B * T  # B block: (batch_size, n, m)
        ###

        ###
        # Use torch.cat to avoid in-place operations that break gradients
        # A_block = self._A * T  # (batch_size, n, n)
        # B_block = self._B * T  # (batch_size, n, m)
        # zeros_bottom = torch.zeros((batch_size, m, n + m), dtype=self._A.dtype, device=self._A.device)
        
        # # Build M using concatenation (no in-place operations)
        # top_row = torch.cat([A_block, B_block], dim=2)  # (batch_size, n, n+m)
        # M = torch.cat([top_row, zeros_bottom], dim=1)   # (batch_size, n+m, n+m)
        ###
        
        # Compute matrix exponential for each batch element
        expM = torch.matrix_exp(M)  # (batch_size, n + m, n + m)
        
        # Extract discretized matrices
        self.Ad = expM[:, :n, :n]  # (batch_size, n_states, n_states)
        self.Bd = expM[:, :n, n:]  # (batch_size, n_states, n_inputs)

        # Output matrices (no discretization needed)
        self.Cd = self._C  # (batch_size, n_outputs, n_states)
        self.Dd = self._D  # (batch_size, n_outputs, n_inputs)

    def initialize(
        self,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
        step_size: int,
    ) -> None:
        """
        Initialize the discrete state space model by computing discretized matrices.
        
        This method automatically expands the system's batch dimensions to match the
        simulation batch size if they differ. This enables dynamic scaling from a
        single-instance system to multi-instance parallel simulation.

        Args:
            start_time: Simulation start time (can be list for batch simulation).
            end_time: Simulation end time (can be list for batch simulation).
            step_size: Simulation step size.
            
        Note:
            If len(start_time) != self.batch_size, all system matrices will be
            dynamically expanded to match len(start_time) using intelligent
            broadcasting rules.
        """
        # Reset and initialize I/O
        _, _, n_timesteps = core.Simulator.get_simulation_timesteps(start_time, end_time, step_size)
        sim_batch_size = len(start_time)
        
        # Calculate total batch size: sim_batch_size * system_batch_size (sim dimension first)
        total_batch_size = sim_batch_size * self.system_batch_size
        
        # Update simulation batch size and total batch size
        if sim_batch_size != self.sim_batch_size:
            # print(f"Expanding from {self.sim_batch_size} sims × {self.system_batch_size} systems = {self.batch_size} total")
            # print(f"            to {sim_batch_size} sims × {self.system_batch_size} systems = {total_batch_size} total")
            
            # Expand all system matrices using nested batch expansion
            self._A = self._expand_to_nested_batch(self._A, self.system_batch_size, sim_batch_size)
            self._B = self._expand_to_nested_batch(self._B, self.system_batch_size, sim_batch_size)
            self._C = self._expand_to_nested_batch(self._C, self.system_batch_size, sim_batch_size)
            self._D = self._expand_to_nested_batch(self._D, self.system_batch_size, sim_batch_size)
            
            # Expand base matrices (used for bilinear terms)
            self._A_base = self._expand_to_nested_batch(self._A_base, self.system_batch_size, sim_batch_size)
            self._B_base = self._expand_to_nested_batch(self._B_base, self.system_batch_size, sim_batch_size)
            
            # Expand bilinear matrices if they exist
            if self._E is not None:
                self._E = self._expand_to_nested_batch(self._E, self.system_batch_size, sim_batch_size)
            if self._F is not None:
                self._F = self._expand_to_nested_batch(self._F, self.system_batch_size, sim_batch_size)
            
            # Expand initial state
            self.x0 = self._expand_to_nested_batch(self.x0, self.system_batch_size, sim_batch_size)
            
            # Update stored batch sizes
            self.sim_batch_size = sim_batch_size
            self.batch_size = total_batch_size
        
        self.input["u"].initialize(n_timesteps, batch_size=self.batch_size)
        self.output["y"].initialize(n_timesteps, batch_size=self.batch_size)

        # Reset state to initial state and detach from old graph
        # self.x shape: (batch_size, n_states)
        self.x = self.x0.detach().clone()

        # Clear any previous computational graph
        self._prev_u = None  # Will be (batch_size, n_inputs) when set

    def do_step(
        self,
        second_time: float,
        date_time: datetime.datetime,
        step_size: int,
        step_index: int,
    ) -> None:
        """
        Perform one step of the state space model simulation with batch support.
        
        Now supports bilinear (state-input coupled) terms using batch operations.
        Ad and Bd are recomputed when inputs change significantly.
        
        Tensor shapes:
            u: (batch_size, n_inputs)
            x: (batch_size, n_states)  
            Ad: (batch_size, n_states, n_states)
            Bd: (batch_size, n_states, n_inputs)
            y: (batch_size, n_outputs)
        """
        assert all(step_size_==step_size[0] for step_size_ in step_size), "DiscreteStatespaceSystem only supports a single step size for batched simulations"
        step_size = step_size[0]
        if step_index == 0:
            first_step = True
        else:
            first_step = False

        if step_size != self.sample_time:
            self.sample_time = step_size
            
        # Get current input: (batch_size, n_inputs)
        u = self.input["u"].get()
        # Current state: (batch_size, n_states)
        x = self.x
        
        non_zero_E = False
        non_zero_F = False

        # Check if inputs have changed significantly (for any batch element)
        if (
            self._prev_u is not None
            and self._E is not None
            and len(self.non_zero_E) > 0
        ):
            # Check across all batch elements for non-zero E inputs
            u_relevant = u[:, self.non_zero_E]  # (batch_size, num_nonzero_E)
            prev_u_relevant = self._prev_u[:, self.non_zero_E]
            non_zero_E = not torch.allclose(u_relevant, prev_u_relevant)

        if (
            self._prev_u is not None
            and self._F is not None
            and len(self.non_zero_F) > 0
        ):
            # Check across all batch elements for non-zero F inputs
            u_relevant = u[:, self.non_zero_F]  # (batch_size, num_nonzero_F)
            prev_u_relevant = self._prev_u[:, self.non_zero_F]
            non_zero_F = not torch.allclose(u_relevant, prev_u_relevant)

        # Recompute effective matrices if needed
        if first_step or self._prev_u is None or non_zero_E or non_zero_F:

            if (self._prev_u is None or non_zero_E) and self._E is not None:
                # Compute effective A matrix for each batch element
                # A_eff shape: (batch_size, n_states, n_states)
                # Use non-in-place operation to preserve gradients
                # Einstein summation for batch operations: 
                # E: (batch_size, n_inputs, n_states, n_states)
                # u: (batch_size, n_inputs)
                # Result: (batch_size, n_states, n_states)
                A_eff = self._A_base + torch.einsum("bmij,bm->bij", self._E, u)
                self._A = A_eff

            if (self._prev_u is None or non_zero_F) and self._F is not None:
                # Compute effective B matrix for each batch element
                # B_eff shape: (batch_size, n_states, n_inputs)
                # Use non-in-place operation to preserve gradients
                # Einstein summation for batch operations:
                # F: (batch_size, n_inputs, n_states, n_inputs)  
                # u: (batch_size, n_inputs)
                # Result: (batch_size, n_states, n_inputs)
                B_eff = self._B_base + torch.einsum("bmij,bm->bij", self._F, u)
                self._B = B_eff

            self.discretize_system()
            self._prev_u = u.clone()

        # State update with batch matrix multiplication
        # Ad: (batch_size, n_states, n_states) @ x: (batch_size, n_states) -> (batch_size, n_states)
        # Bd: (batch_size, n_states, n_inputs) @ u: (batch_size, n_inputs) -> (batch_size, n_states)
        x_new = torch.bmm(self.Ad, x.unsqueeze(-1)).squeeze(-1) + torch.bmm(self.Bd, u.unsqueeze(-1)).squeeze(-1)
        self.x = x_new
        
        # Output computation with batch matrix multiplication
        # Cd: (batch_size, n_outputs, n_states) @ x: (batch_size, n_states) -> (batch_size, n_outputs)
        # Dd: (batch_size, n_outputs, n_inputs) @ u: (batch_size, n_inputs) -> (batch_size, n_outputs)
        y = torch.bmm(self.Cd, self.x.unsqueeze(-1)).squeeze(-1) + torch.bmm(self.Dd, u.unsqueeze(-1)).squeeze(-1)
        self.output["y"].set(y, step_index)

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
        # Third party imports
        from scipy import signal

        A, B, C, D = signal.tf2ss(num, den)
        return cls(A=A, B=B, C=C, D=D, sample_time=sample_time, **kwargs)

    def get_state(self) -> torch.Tensor:
        """
        Get the current state vector.
        
        Returns:
            torch.Tensor: Current state vector of shape (batch_size, n_states)
        """
        return self.x.clone()

    def set_state(self, x: torch.Tensor) -> None:
        """
        Set the current state vector.

        Args:
            x: New state vector of shape (batch_size, n_states) or (n_states,)
               If (n_states,), it will be broadcasted to all batch elements
        """
        if len(x.shape) == 1 and x.shape[0] == self.n_states:
            # Broadcast single state to all batch elements
            x = x.unsqueeze(0).expand(self.batch_size, -1).contiguous()
        elif x.shape != self.x.shape:
            raise ValueError(
                f"State vector should have shape {self.x.shape} or ({self.n_states},), got {x.shape}"
            )
        self.x = x.clone()
