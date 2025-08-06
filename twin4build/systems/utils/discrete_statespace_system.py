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
    A general-purpose discrete state space system for modeling dynamical systems.

    This class implements a discrete state-space system that supports both linear and bilinear
    dynamics through state-input and input-input coupling terms. The system serves as the
    computational core for various physical models in the Twin4Build framework, including
    thermal RC networks and mass balance systems.

    Mathematical Formulation:
    =========================

    **Continuous-Time State-Space Representation:**

    The general continuous-time state-space system with bilinear terms is formulated as:

    .. math::

       \frac{d\mathbf{x}}{dt} = \mathbf{A}\mathbf{x} + \mathbf{B}\mathbf{u} + \sum_{i=1}^{m} \mathbf{E}_i\mathbf{x}u_i + \sum_{i=1}^{m}\sum_{j=1}^{m} \mathbf{F}_{ij}u_i u_j

       \mathbf{y} = \mathbf{C}\mathbf{x} + \mathbf{D}\mathbf{u}

    where:

       - :math:`\mathbf{x} \in \mathbb{R}^n`: State vector (internal system variables)
       - :math:`\mathbf{u} \in \mathbb{R}^m`: Input vector (external driving signals)
       - :math:`\mathbf{y} \in \mathbb{R}^p`: Output vector (observable quantities)
       - :math:`\mathbf{A} \in \mathbb{R}^{n \times n}`: State transition matrix
       - :math:`\mathbf{B} \in \mathbb{R}^{n \times m}`: Input matrix
       - :math:`\mathbf{C} \in \mathbb{R}^{p \times n}`: Output matrix
       - :math:`\mathbf{D} \in \mathbb{R}^{p \times m}`: Feedthrough matrix
       - :math:`\mathbf{E}_i \in \mathbb{R}^{n \times n}`: State-input coupling matrices
       - :math:`\mathbf{F}_{ij} \in \mathbb{R}^{n}`: Input-input coupling terms

    **Bilinear Extensions:**

    The bilinear terms extend the basic linear state-space model to handle:

    *State-Input Coupling (E matrices):*
       - Models where inputs affect the dynamics matrix
       - Example: :math:`\dot{m}_{exh} \times T_{air}` in thermal systems
       - Formulation: :math:`\sum_{i=1}^{m} \mathbf{E}_i\mathbf{x}u_i`

    *Input-Input Coupling (F matrices):*
       - Models where the product of two inputs affects the state derivative
       - Example: :math:`\dot{m}_{sup} \times T_{sup}` in thermal systems
       - Formulation: :math:`\sum_{i=1}^{m}\sum_{j=1}^{m} \mathbf{F}_{ij}u_i u_j`

    **Discretization Method:**

    For numerical simulation, the continuous system is discretized using zero-order hold (ZOH).
    However, when bilinear terms (E and F matrices) are present, the effective A and B matrices
    are first computed by incorporating the current input values before discretization:

    *Step 1: Compute Effective Matrices*

    .. math::

       \mathbf{A}_{eff}[k] = \mathbf{A} + \sum_{i=1}^{m} \mathbf{E}_i u_i[k]

       \mathbf{B}_{eff}[k] = \mathbf{B} + \sum_{i=1}^{m} \mathbf{F}_i u_i[k]

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

    Parameters
    ----------
    A : torch.Tensor
        Continuous state transition matrix (n×n)
    B : torch.Tensor
        Continuous input matrix (n×m)
    C : torch.Tensor
        Output matrix (p×n)
    D : torch.Tensor, optional
        Feedthrough matrix (p×m)
    sample_time : float, default=1.0
        Sampling time for discretization [s]
    x0 : torch.Tensor, optional
        Initial state vector (n,)
    state_names : List[str], optional
        Names for system states
    E : torch.Tensor, optional
        State-input coupling tensor (m×n×n)
    F : torch.Tensor, optional
        Input-input coupling tensor (m×n×m)
    **kwargs
        Additional keyword arguments

    Examples
    --------
    Basic linear state-space system:

    >>> import torch
    >>> import twin4build as tb
    >>>
    >>> # Define system matrices
    >>> A = torch.tensor([[-0.1, 0.05], [0.02, -0.08]], dtype=torch.float64)
    >>> B = torch.tensor([[1.0], [0.5]], dtype=torch.float64)
    >>> C = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    >>> x0 = torch.tensor([20.0, 18.0], dtype=torch.float64)
    >>>
    >>> # Create system
    >>> system = tb.DiscreteStatespaceSystem(
    ...     A=A, B=B, C=C, x0=x0, sample_time=3600.0,
    ...     state_names=["T_air", "T_wall"]
    ... )

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
            _D = (
                D
                if D is not None
                else torch.zeros((C.shape[0], B.shape[1]), dtype=torch.float64)
            )

            self._A = (
                _A.clone()
            )  # We need a base matrix because we change them dynamically in do_step
            self._B = (
                _B.clone()
            )  # We need a base matrix because we change them dynamically in do_step
            self._C = _C.clone()
            self._D = _D.clone()

            self._A_base = _A.clone()
            self._B_base = _B.clone()

            # State and I/O dimensions
            self.n_states = self._A_base.shape[0]
            self.n_inputs = self._B_base.shape[1]
            self.n_outputs = self._C.shape[0]
        else:
            raise ValueError("System matrices A, B, and C must be provided")

        # Store sample time as a regular attribute
        self.sample_time = sample_time

        self.x0 = (
            x0 if x0 is not None else torch.zeros(self.n_states, dtype=torch.float64)
        )

        # Current state
        self.x = x0

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

        # Set up inputs and outputs as private variables
        self._input = {"u": tps.Vector(size=self.n_inputs)}  # Single input vector
        self._output = {"y": tps.Vector(size=self.n_outputs)}  # Single output vector

        # Define parameters for potential calibration
        self.parameter = {
            # Additional parameters could be added for matrix entries
        }

        self._config = {"parameters": list(self.parameter.keys())}

        # Initialize discretized matrices
        # self.discretize_system()
        self.INITIALIZED = True

        self._E = E  # shape (M, N, N) or None
        self._F = F  # shape (M, N, M) or None
        self._prev_u = None  # For input change detection

        if E is not None:
            self.non_zero_E = torch.zeros(E.shape[0], dtype=torch.bool)
            for i in range(E.shape[0]):
                self.non_zero_E[i] = torch.any(E[i, :, :])
        else:
            self.non_zero_E = torch.zeros(0, dtype=torch.bool)

        if F is not None:
            self.non_zero_F = torch.zeros(F.shape[0], dtype=torch.bool)
            for i in range(F.shape[0]):
                self.non_zero_F[i] = torch.any(F[i, :, :])
        else:
            self.non_zero_F = torch.zeros(0, dtype=torch.bool)

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

        Uses the matrix exponential method to compute Ad and Bd efficiently.
        The implementation ensures that gradients can flow back through the discretization process.
        """
        T = self.sample_time
        n = self.n_states

        # # Compute Ad using matrix exponential
        # self.Ad = torch.matrix_exp(self._A * T)

        # # Compute Bd using the analytical formula
        # # (A must be invertible; for thermal systems, this is almost always true)
        # I = torch.eye(n, dtype=self._A.dtype, device=self._A.device)
        # A_inv = torch.linalg.inv(self._A)
        # self.Bd = A_inv @ (self.Ad - I) @ self._B

        n = self._A.shape[0]
        m = self._B.shape[1]
        M = torch.zeros((n + m, n + m), dtype=self._A.dtype, device=self._A.device)
        M[:n, :n] = self._A * T
        M[:n, n:] = self._B * T
        expM = torch.matrix_exp(M)
        self.Ad = expM[:n, :n]
        self.Bd = expM[:n, n:]

        self.Cd = self._C
        self.Dd = self._D

    def initialize(
        self,
        startTime: datetime.datetime,
        endTime: datetime.datetime,
        stepSize: int,
        simulator: core.Simulator,
    ) -> None:
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

    def do_step(
        self,
        secondTime: float,
        dateTime: datetime.datetime,
        stepSize: int,
        stepIndex: int,
    ) -> None:
        """
        Perform one step of the state space model simulation.
        Now supports bilinear (state-input coupled) terms using the trapezoidal (average of old and new states) method for the E and F terms.
        Ad and Bd are only computed in discretize_system for efficiency.
        """
        if stepIndex == 0:
            first_step = True
        else:
            first_step = False

        if stepSize != self.sample_time:
            self.sample_time = stepSize
        u = self.input["u"].get()
        x = self.x
        non_zero_E = False
        non_zero_F = False

        if (
            self._prev_u is not None
            and self._E is not None
            and torch.allclose(u[self.non_zero_E], self._prev_u[self.non_zero_E])
            == False
        ):
            non_zero_E = True

        if (
            self._prev_u is not None
            and self._F is not None
            and torch.allclose(u[self.non_zero_F], self._prev_u[self.non_zero_F])
            == False
        ):
            non_zero_F = True

        # Note: If we have a model with tiny inputs, we may need to use a different tolerance for the comparison
        # Warning: In this case, the simulation may silently fail to update the state space matrices
        if first_step or self._prev_u is None or non_zero_E or non_zero_F:

            if (self._prev_u is None or non_zero_E) and self._E is not None:
                # Compute effective A
                A_eff = self._A_base.clone()
                A_eff += torch.einsum("mij,m->ij", self._E, u)
                self._A = A_eff

            if (self._prev_u is None or non_zero_F) and self._F is not None:
                # Compute effective B
                B_eff = self._B_base.clone()
                B_eff += torch.einsum("mij,m->ij", self._F, u)
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
        # Third party imports
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
            raise ValueError(
                f"State vector should have shape {self.x.shape}, got {x.shape}"
            )
        self.x = x.clone()
