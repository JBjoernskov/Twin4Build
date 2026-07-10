# Standard library imports
import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

# Third party imports
import torch
import torch.nn as nn

# Local application imports
import twin4build.utils.types as tps
from twin4build import core


def bilinear_onestep(A, B, C, D, E, F, x, u, sample_time):
    """Pure one ZOH step of a (bilinear) state-space system.

    A functorch-compatible re-expression of :meth:`DiscreteStatespaceSystem.do_step`
    with **no ports, no history, no state mutation** -- the building block of the
    ``forward`` fast path.  Given the continuous matrices and the current
    ``(x, u)`` it forms the input-dependent effective matrices, discretizes them
    via the matrix-exponential block trick, and returns ``(x_next, y)``.

    Shapes (``n_c`` = parallel components, ``n`` = states, ``m`` = inputs):
        A ``(n_c, n, n)``   B ``(n_c, n, m)``   C ``(n_c, p, n)``   D ``(n_c, p, m)``
        E ``(n_c, m, n, n)`` or None            F ``(n_c, m, n, m)`` or None
        x ``(n_c, n)``      u ``(n_c, m)``
    Returns ``x_next (n_c, n)``, ``y (n_c, p)``.  ``vmap`` maps this over segments.
    """
    A_eff = A if E is None else A + torch.einsum("cmij,cm->cij", E, u)
    B_eff = B if F is None else B + torch.einsum("cmij,cm->cij", F, u)

    n = A.shape[-1]
    m = B.shape[-1]
    T = sample_time
    # Block matrix M = [[A*T, B*T], [0, 0]]; expm(M) = [[Ad, Bd], [0, I]].
    # Built out-of-place (cat + zero-row pad) so it is ``vmap``-safe -- an
    # in-place ``M[..., :n, :n] = ...`` fails when A_eff/B_eff carry a vmap batch
    # dim that the freshly-allocated M does not.
    top = torch.cat([A_eff * T, B_eff * T], dim=-1)  # (..., n, n+m)
    M = torch.nn.functional.pad(top, (0, 0, 0, m))  # add m zero rows -> (..., n+m, n+m)
    expM = torch.matrix_exp(M)
    Ad = expM[..., :n, :n]
    Bd = expM[..., :n, n:]

    x_next = (Ad @ x.unsqueeze(-1)).squeeze(-1) + (Bd @ u.unsqueeze(-1)).squeeze(-1)
    y = (C @ x_next.unsqueeze(-1)).squeeze(-1) + (D @ u.unsqueeze(-1)).squeeze(-1)
    return x_next, y


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
                raise ValueError(
                    f"Batch dimension mismatch: expected {batch_size} or 1, got {tensor.shape[0]}"
                )
            # Expand if batch_size is 1 but we need larger batch
            if tensor.shape[0] == 1 and batch_size > 1:
                expand_dims = [batch_size] + [-1] * (len(tensor.shape) - 1)
                return tensor.expand(
                    *expand_dims
                ).contiguous()  # Keep contiguous here - expand may not be contiguous
            return tensor

        # If tensor doesn't have batch dimension, add it
        elif len(tensor.shape) == len(target_shape) - 1:
            return tensor.unsqueeze(0).expand(batch_size, *tensor.shape).contiguous()
        else:
            raise ValueError(
                f"Tensor shape {tensor.shape} incompatible with target shape {target_shape}"
            )

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
            raise ValueError(
                f"System batch size mismatch: expected {system_batch_size}, got {current_system_batch}"
            )

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
        E: torch.Tensor = None,  # State-input coupling (n_c, M, N, N)
        F: torch.Tensor = None,  # Input-input coupling (n_c, M, N, M)
        **kwargs,
    ):
        """
        Initialize a DiscreteStatespaceSystem object.

        All matrices use the convention:
        - n_c: number of parallel components (different system configurations)
        - n_s: number of parallel simulations (set during initialize)

        Args:
            A: System dynamics matrix of shape (n_c, N, N) or (N, N)
            B: Control input matrix of shape (n_c, N, M) or (N, M)
            C: Output matrix of shape (n_c, P, N) or (P, N)
            D: Feedthrough matrix of shape (n_c, P, M) or (P, M). Optional.
            sample_time: Sampling time for discretization
            x0: Initial state vector of shape (n_c, N) or (N,)
            state_names: Names for system states
            E: Bilinear state-input tensor of shape (n_c, M, N, N) or (M, N, N). Optional.
            F: Input-input coupling tensor of shape (n_c, M, N, M) or (M, N, M). Optional.
            **kwargs: Additional keyword arguments
        """
        super().__init__(**kwargs)

        # Verify and store continuous system matrices
        if A is not None and B is not None and C is not None:
            # Determine n_c from input matrices (number of component configurations)
            n_c = 1
            if len(A.shape) == 3:
                n_c = A.shape[0]
            elif len(B.shape) == 3:
                n_c = B.shape[0]
            elif len(C.shape) == 3:
                n_c = C.shape[0]

            # Determine base dimensions (without batch)
            n_states = A.shape[-2]
            n_inputs = B.shape[-1]
            n_outputs = C.shape[-2]

            # Expand all matrices to (n_c, ...) shape
            _A = self._expand_to_batch(A, (n_c, n_states, n_states), n_c)
            _B = self._expand_to_batch(B, (n_c, n_states, n_inputs), n_c)
            _C = self._expand_to_batch(C, (n_c, n_outputs, n_states), n_c)

            # Handle D matrix
            if D is not None:
                _D = self._expand_to_batch(D, (n_c, n_outputs, n_inputs), n_c)
            else:
                _D = torch.zeros((n_c, n_outputs, n_inputs), dtype=torch.float64)

            # Store base matrices (n_c, ...) - these don't change
            self._A_base = _A.clone()
            self._B_base = _B.clone()
            self._C = _C.clone()
            self._D = _D.clone()

            # Pre-expand C and D for efficient batched matmul in do_step
            # Shape: (1, n_c, n_outputs, n_states) and (1, n_c, n_outputs, n_inputs)
            # This avoids implicit broadcasting overhead at runtime (~1.7x speedup)
            self._C_expanded = self._C.unsqueeze(0)
            self._D_expanded = self._D.unsqueeze(0)

            # Store dimensions
            self.n_c = n_c  # Number of different system configurations
            self.n_s = 1  # Number of parallel simulations (set during initialize)
            self.n_states = n_states
            self.n_inputs = n_inputs
            self.n_outputs = n_outputs
            # State as a first-class ``tps.State`` (shape (n_s, n_c, n_states)).
            # The ``x`` property below delegates to it, so all existing
            # ``self.x`` reads/writes keep working while state is now a declared,
            # enumerable member (see twin4build.utils.types.State).
            self._x_state = tps.State(n_v=n_states)
        else:
            raise ValueError("System matrices A, B, and C must be provided")

        # Store sample time
        self.sample_time = sample_time

        # Handle initial state - can be (n_states,), (n_c, n_states), or (n_s, n_c, n_states)
        # We store the original x0 and expand it during initialize() when n_s is known
        if x0 is not None:
            if x0.dim() == 1:
                # (n_states,) -> (n_c, n_states)
                self.x0 = x0.unsqueeze(0).expand(self.n_c, -1).clone()
            elif x0.dim() == 2:
                # (n_c, n_states)
                self.x0 = x0.clone()
            elif x0.dim() == 3:
                # (n_s, n_c, n_states) - store directly, will validate in initialize()
                self.x0 = x0.clone()
            else:
                raise ValueError(f"x0 must have 1, 2, or 3 dimensions, got {x0.dim()}")
        else:
            self.x0 = torch.zeros((self.n_c, self.n_states), dtype=torch.float64)

        # Current state (n_s, n_c, n_states) is held in ``self._x_state`` and
        # exposed via the ``x`` property; it is populated in initialize().

        # Names for states
        self.state_names = (
            state_names
            if state_names is not None
            else [f"x{i}" for i in range(self.n_states)]
        )

        if len(self.state_names) != self.n_states:
            raise ValueError(
                f"state_names should have length {self.n_states}, got {len(self.state_names)}"
            )

        # Set up inputs and outputs - will be initialized with (n_s, n_c, n_v) shape
        self._input = {"u": tps.Vector(n_v=self.n_inputs)}
        self._output = {"y": tps.Vector(n_v=self.n_outputs)}

        self.parameter = {}
        self._config = {"parameters": list(self.parameter.keys())}
        self.INITIALIZED = True

        # Handle bilinear matrices - shape (n_c, n_inputs, n_states, n_states/n_inputs)
        if E is not None:
            self._E = self._expand_to_batch(
                E, (self.n_c, self.n_inputs, self.n_states, self.n_states), self.n_c
            )
            self.non_zero_E = torch.zeros(self.n_inputs, dtype=torch.bool)
            for i in range(self.n_inputs):
                self.non_zero_E[i] = torch.any(self._E[:, i, :, :])
        else:
            self._E = None
            self.non_zero_E = torch.zeros(0, dtype=torch.bool)

        if F is not None:
            self._F = self._expand_to_batch(
                F, (self.n_c, self.n_inputs, self.n_states, self.n_inputs), self.n_c
            )
            self.non_zero_F = torch.zeros(self.n_inputs, dtype=torch.bool)
            for i in range(self.n_inputs):
                self.non_zero_F[i] = torch.any(self._F[:, i, :, :])
        else:
            self._F = None
            self.non_zero_F = torch.zeros(0, dtype=torch.bool)

        # For input change detection - shape (n_s, n_c, n_inputs)
        self._prev_u = None

        # Discretized matrices - computed in discretize_system()
        # Shape: (n_s, n_c, ...) when bilinear, (n_c, ...) when linear
        self.Ad = None
        self.Bd = None

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

    # Backward compatibility properties
    @property
    def batch_size(self) -> int:
        """Total batch size (n_s * n_c). Provided for backward compatibility."""
        return self.n_s * self.n_c

    @property
    def sim_batch_size(self) -> int:
        """Alias for n_s. Provided for backward compatibility."""
        return self.n_s

    @property
    def system_batch_size(self) -> int:
        """Alias for n_c. Provided for backward compatibility."""
        return self.n_c

    def discretize_system(
        self, A_eff: torch.Tensor, B_eff: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Discretize the continuous-time state space model using the matrix exponential method.

        Args:
            A_eff: Effective A matrix, shape (n_s, n_c, n_states, n_states)
            B_eff: Effective B matrix, shape (n_s, n_c, n_states, n_inputs)

        Returns:
            Ad: Discretized A matrix, shape (n_s, n_c, n_states, n_states)
            Bd: Discretized B matrix, shape (n_s, n_c, n_states, n_inputs)
        """
        T = self.sample_time
        n_s, n_c, n, _ = A_eff.shape
        m = self.n_inputs

        # Flatten (n_s, n_c) -> (n_s*n_c) for batched matrix_exp
        A_flat = A_eff.reshape(-1, n, n)
        B_flat = B_eff.reshape(-1, n, m)
        batch_size = A_flat.shape[0]

        # Build block matrix: M = | A*T  B*T |
        #                         |  0    0  |
        M = torch.zeros(
            (batch_size, n + m, n + m), dtype=A_flat.dtype, device=A_flat.device
        )
        M[:, :n, :n] = A_flat * T
        M[:, :n, n:] = B_flat * T

        # Compute matrix exponential: e^M = | Ad  Bd |
        #                                   |  0   I |
        expM = torch.matrix_exp(M)

        # Extract and reshape back to (n_s, n_c, ...)
        Ad = expM[:, :n, :n].reshape(n_s, n_c, n, n)
        Bd = expM[:, :n, n:].reshape(n_s, n_c, n, m)

        return Ad, Bd

    def initialize(
        self,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
        step_size: int,
    ) -> None:
        """
        Initialize the discrete state space model.

        Sets up the state vector and I/O with proper (n_s, n_c, ...) dimensions.

        Args:
            start_time: Simulation start time (list for batch simulation).
            end_time: Simulation end time (list for batch simulation).
            step_size: Simulation step size.
        """
        _, _, max_timesteps, _ = core.Simulator.get_simulation_timesteps(
            start_time, end_time, step_size
        )
        n_s = len(start_time)
        self.n_s = n_s

        # Initialize I/O with (n_s, n_c, n_v) shape
        self.input["u"].initialize(n_t=max_timesteps, n_s=n_s, n_c=self.n_c)
        self.output["y"].initialize(n_t=max_timesteps, n_s=n_s, n_c=self.n_c)

        # Initialize state with shape (n_s, n_c, n_states)
        if self.x0.dim() == 2:
            # x0 is (n_c, n_states) -> expand to (n_s, n_c, n_states)
            self.x = self.x0.unsqueeze(0).expand(n_s, -1, -1).clone()
        else:
            # x0 is already (n_s, n_c, n_states) - validate and use directly
            assert (
                self.x0.shape[0] == n_s
            ), f"x0 has n_s={self.x0.shape[0]} but simulation requires n_s={n_s}"
            assert (
                self.x0.shape[1] == self.n_c
            ), f"x0 has n_c={self.x0.shape[1]} but system has n_c={self.n_c}"
            self.x = self.x0.clone()

        # Clear previous input for bilinear term detection
        self._prev_u = None

    def do_step(
        self,
        second_time: float,
        date_time: datetime.datetime,
        step_size: int,
        step_index: int,
    ) -> None:
        """
        Perform one step of the state space model simulation.

        Supports bilinear (state-input coupled) terms using proper (n_s, n_c, ...) dimensions.
        Ad and Bd are recomputed when inputs change significantly.

        Tensor shapes:
            u: (n_s, n_c, n_inputs)
            x: (n_s, n_c, n_states)
            Ad: (n_s, n_c, n_states, n_states) or (n_c, n_states, n_states) for linear systems
            Bd: (n_s, n_c, n_states, n_inputs) or (n_c, n_states, n_inputs) for linear systems
            y: (n_s, n_c, n_outputs)
        """
        assert all(
            [step_size_ == step_size[0] for step_size_ in step_size]
        ), "DiscreteStatespaceSystem currently only supports a single step size."
        step_size = step_size[0]
        first_step = step_index == 0

        if step_size != self.sample_time:
            self.sample_time = step_size

        # Get current input: (n_s, n_c, n_inputs)
        u = self.input["u"].get().clone()
        x = self.x  # (n_s, n_c, n_states)

        # Check if we need to recompute discretized matrices
        need_rediscretize = first_step or self._prev_u is None

        if not need_rediscretize and self._E is not None and len(self.non_zero_E) > 0:
            u_relevant = u[:, :, self.non_zero_E]
            prev_u_relevant = self._prev_u[:, :, self.non_zero_E]
            need_rediscretize = not torch.allclose(u_relevant, prev_u_relevant)

        if not need_rediscretize and self._F is not None and len(self.non_zero_F) > 0:
            u_relevant = u[:, :, self.non_zero_F]
            prev_u_relevant = self._prev_u[:, :, self.non_zero_F]
            need_rediscretize = not torch.allclose(u_relevant, prev_u_relevant)

        # Compute effective matrices and discretize if needed
        if need_rediscretize:
            # Expand base matrices to (n_s, n_c, ...) shape
            A_eff = self._A_base.unsqueeze(0).expand(
                self.n_s, -1, -1, -1
            )  # (n_s, n_c, n_states, n_states)
            B_eff = self._B_base.unsqueeze(0).expand(
                self.n_s, -1, -1, -1
            )  # (n_s, n_c, n_states, n_inputs)

            if self._E is not None:
                # Add bilinear E term: E (n_c, n_inputs, n_states, n_states), u (n_s, n_c, n_inputs)
                A_eff = A_eff + torch.einsum("cmij,scm->scij", self._E, u)

            if self._F is not None:
                # Add bilinear F term: F (n_c, n_inputs, n_states, n_inputs), u (n_s, n_c, n_inputs)
                B_eff = B_eff + torch.einsum("cmij,scm->scij", self._F, u)

            self.Ad, self.Bd = self.discretize_system(A_eff, B_eff)
            self._prev_u = u.clone()

        # State update: x_new = Ad @ x + Bd @ u
        # Using batched matmul instead of einsum for ~1.6-1.9x speedup
        # Ad: (n_s, n_c, n_states, n_states), x: (n_s, n_c, n_states) -> x.unsqueeze(-1): (n_s, n_c, n_states, 1)
        x_new = (self.Ad @ x.unsqueeze(-1)).squeeze(-1) + (
            self.Bd @ u.unsqueeze(-1)
        ).squeeze(-1)
        self.x = x_new

        # Output: y = C @ x + D @ u
        # Using pre-expanded C and D with batched matmul for ~1.7x speedup
        # _C_expanded: (1, n_c, n_outputs, n_states), x: (n_s, n_c, n_states)
        y = (self._C_expanded @ x_new.unsqueeze(-1)).squeeze(-1) + (
            self._D_expanded @ u.unsqueeze(-1)
        ).squeeze(-1)

        self.output["y"]._set(y, i_t=step_index)

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

    @property
    def x(self):
        """Current state tensor ``(n_s, n_c, n_states)``, or ``None`` before
        initialize().  Backed by the ``tps.State`` in ``self._x_state``."""
        if self._x_state is None or self._x_state.tensor is None:
            return None
        return self._x_state.get()

    @x.setter
    def x(self, value):
        if value is None:
            return
        # Adopt the assigned tensor as the whole state (syncs n_s/n_c/n_v from its
        # shape), so ``self.x = <tensor>`` works everywhere as before -- including
        # a re-simulation whose batch size n_s differs from the previous run.
        self._x_state.reset(value)

    def get_state(self) -> torch.Tensor:
        """
        Get the current state vector.

        Returns:
            torch.Tensor: Current state vector of shape (n_s, n_c, n_states)
        """
        return self._x_state.get()

    def set_state(self, x: torch.Tensor) -> None:
        """
        Set the current state vector.

        Args:
            x: New state vector of shape (n_s, n_c, n_states), (n_c, n_states), or (n_states,)
        """
        if x.dim() == 1 and x.shape[0] == self.n_states:
            # Broadcast single state to all elements
            x = x.unsqueeze(0).unsqueeze(0).expand(self.n_s, self.n_c, -1).clone()
        elif x.dim() == 2 and x.shape == (self.n_c, self.n_states):
            # Broadcast (n_c, n_states) to (n_s, n_c, n_states)
            x = x.unsqueeze(0).expand(self.n_s, -1, -1).clone()
        elif x.dim() == 3 and x.shape == (self.n_s, self.n_c, self.n_states):
            x = x.clone()
        else:
            raise ValueError(
                f"State vector should have shape ({self.n_s}, {self.n_c}, {self.n_states}), "
                f"({self.n_c}, {self.n_states}), or ({self.n_states},), got {x.shape}"
            )
        self.x = x
