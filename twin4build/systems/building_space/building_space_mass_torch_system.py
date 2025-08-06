# Standard library imports
import datetime
from typing import Any, Dict, List, Optional

# Third party imports
import numpy as np
import torch
import torch.nn as nn

# Local application imports
import twin4build.core as core
import twin4build.utils.types as tps
from twin4build.systems.utils.discrete_statespace_system import DiscreteStatespaceSystem
from twin4build.utils.constants import Constants


class BuildingSpaceMassTorchSystem(core.System, nn.Module):
    r"""
    Building Space CO2 Concentration Model using Mass Balance Dynamics.

    This model represents the CO2 concentration dynamics in a building space considering
    supply and exhaust air flows, occupant CO2 generation, infiltration, and outdoor 
    CO2 concentration using bilinear state-space dynamics.

    Mathematical Formulation:
    =========================

    **Continuous-Time Differential Equation:**

    The CO2 concentration dynamics are governed by the mass balance equation:

    .. math::

       V\frac{dC}{dt} = \dot{m}_{sup}C_{out} - \dot{m}_{exh}C + \dot{m}_{inf}(C_{out} - C) + G_{occ} N_{occ}

    where:

       - :math:`V`: Volume of the space [m³]
       - :math:`C`: Indoor CO2 concentration [ppm] (state variable)
       - :math:`\dot{m}_{sup}`: Supply air flow rate [kg/s] (input)
       - :math:`\dot{m}_{exh}`: Exhaust air flow rate [kg/s] (input)
       - :math:`\dot{m}_{inf}`: Infiltration rate [kg/s] (parameter)
       - :math:`C_{out}`: Outdoor CO2 concentration [ppm] (input)
       - :math:`G_{occ}`: CO2 generation rate per occupant [ppm·kg/s] (parameter)
       - :math:`N_{occ}`: Number of occupants (input)

    Note: Supply air CO2 concentration is assumed equal to outdoor CO2 concentration.

    **State-Space Representation:**

    The system is implemented using the DiscreteStatespaceSystem with matrices:

    *State vector:* :math:`\mathbf{x} = [C]`

    *Input vector:* :math:`\mathbf{u} = [\dot{m}_{sup}, \dot{m}_{exh}, C_{out}, N_{occ}]^T`

    *Base System Matrices:*

    .. math::

       \mathbf{A} = \left[ -\frac{\dot{m}_{inf}}{V} \right]

       \mathbf{B} = \left[ \frac{1}{V} \quad -\frac{1}{V} \quad \frac{\dot{m}_{inf}}{V} \quad \frac{G_{occ}}{V} \right]

       \mathbf{C} = \left[ 1 \right]

       \mathbf{D} = \left[ 0 \quad 0 \quad 0 \quad 0 \right]

    **Bilinear Coupling Matrices:**

    *State-Input Coupling (E matrices):*

    .. math::

       \mathbf{E} \in \mathbb{R}^{4 \times 1 \times 1} = \left[\begin{array}{l}
       \left[ 0 \right] \text{ (supply flow)} \\
       \left[ -\frac{1}{V} \right] \text{ (exhaust flow)} \\
       \left[ 0 \right] \text{ (outdoor CO2)} \\
       \left[ 0 \right] \text{ (occupants)}
       \end{array}\right]

    *Input-Input Coupling (F matrices):*

    .. math::

       \mathbf{F} \in \mathbb{R}^{4 \times 1 \times 4} = \left[\begin{array}{l}
       \left[ 0 \quad 0 \quad \frac{1}{V} \quad 0 \right] \text{ (supply flow)} \\
       \left[ 0 \quad 0 \quad 0 \quad 0 \right] \text{ (exhaust flow)} \\
       \left[ 0 \quad 0 \quad 0 \quad 0 \right] \text{ (outdoor CO2)} \\
       \left[ 0 \quad 0 \quad 0 \quad 0 \right] \text{ (occupants)}
       \end{array}\right]

    **Discretization with Bilinear Terms:**

    The bilinear terms are incorporated into the base matrices before discretization:

    *Step 1: Compute Effective Matrices*

    .. math::

       \mathbf{A}_{eff}[k] = \mathbf{A} + \sum_{i=1}^{4} \mathbf{E}_i u_i[k]

       \mathbf{B}_{eff}[k] = \mathbf{B} + \sum_{i=1}^{4} \mathbf{F}_i u_i[k]

    where the effective matrices depend on the current input vector :math:`\mathbf{u}[k]`.

    *Step 2: Discretize Effective Matrices*

    The effective matrices are then discretized using the matrix exponential method implemented
    in the DiscreteStatespaceSystem base class.

    *Step 3: Bilinear Effects*

    The bilinear terms handle specific flow-dependent mass transfer effects:
       - :math:`\mathbf{E}[1,0,0] \cdot u_1 \cdot x_0 = -\frac{1}{V} \dot{m}_{exh} C`: Exhaust flow removing CO2
       - :math:`\mathbf{F}[0,0,2] \cdot u_0 \cdot u_2 = \frac{1}{V} \dot{m}_{sup} C_{out}`: Supply flow bringing outdoor air

    Physical Interpretation:
    ======================

    **Mass Balance System:**
       - Single state represents indoor CO2 concentration
       - Inputs represent ventilation flows, outdoor conditions, and occupancy
       - Bilinear terms model flow-dependent mass transfer accurately

    **Flow-Dependent Effects:**
       - Supply air flow brings outdoor CO2 at outdoor concentration (F matrix coupling)
       - Exhaust air flow removes CO2 at indoor concentration (E matrix coupling)
       - These effects are critical for accurate IAQ modeling

    Computational Features:
    ======================

       - **Automatic Differentiation:** PyTorch tensors enable gradient computation
       - **Adaptive Discretization:** Matrices updated when flows change significantly
       - **Efficient Simulation:** Optimized for building IAQ simulation
       - **Parameter Estimation:** All mass balance parameters available for calibration

    Parameters
    ----------
    V : float, default=100
        Volume of the space [m³]
    G_occ : float, default=5e-6
        CO2 generation rate per occupant [ppm·kg/s]
    m_inf : float, default=0.001
        Infiltration rate [kg/s]
    **kwargs
        Additional keyword arguments

    Examples
    --------
    Basic CO2 model:

    >>> import twin4build as tb
    >>>
    >>> # Create CO2 model with default parameters
    >>> co2_model = tb.BuildingSpaceMassTorchSystem(
    ...     V=150,          # Room volume [m³]
    ...     G_occ=6e-6,     # Higher CO2 generation per person
    ...     m_inf=0.002,    # Higher infiltration rate
    ...     id="zone_1_co2"
    ... )

    Large space CO2 model:

    >>> # Model for large space with higher occupancy
    >>> co2_model = tb.BuildingSpaceMassTorchSystem(
    ...     V=500,          # Large space volume
    ...     G_occ=4e-6,     # Lower per-person generation
    ...     m_inf=0.005,    # Higher infiltration for large space
    ...     id="large_space_co2"
    ... )
    """

    def __init__(
        self, V: float = 100, G_occ: float = 5e-6, m_inf: float = 0.001, **kwargs
    ):
        super().__init__(**kwargs)
        nn.Module.__init__(self)

        # Store parameters as tps.Parameters
        self.V = tps.Parameter(
            torch.tensor(V, dtype=torch.float64), requires_grad=False
        )
        self.G_occ = tps.Parameter(
            torch.tensor(G_occ, dtype=torch.float64), requires_grad=False
        )
        self.m_inf = tps.Parameter(
            torch.tensor(m_inf, dtype=torch.float64), requires_grad=False
        )

        # Define inputs and outputs
        self.input = {
            "supplyAirFlowRate": tps.Scalar(),  # Supply air flow rate [kg/s]
            "exhaustAirFlowRate": tps.Scalar(),  # Exhaust air flow rate [kg/s]
            "outdoorCO2": tps.Scalar(),  # Supply air CO2 concentration [ppm]
            "numberOfPeople": tps.Scalar(),  # Number of occupants
        }

        # Define outputs
        self.output = {
            "indoorCO2": tps.Scalar(400),  # Indoor CO2 concentration [ppm]
        }

        # Define parameters for calibration
        self.parameter = {
            "V": {"lb": 10.0, "ub": 1000.0},
            "G_occ": {"lb": 0.000001, "ub": 0.00001},
            "m_inf": {"lb": 0.0001, "ub": 0.01},
        }

        self._config = {"parameters": list(self.parameter.keys())}
        self.INITIALIZED = False

    def initialize(
        self,
        startTime: datetime.datetime,
        endTime: datetime.datetime,
        stepSize: int,
        simulator: core.Simulator,
    ) -> None:
        """Initialize the mass balance model by setting up the state-space representation."""
        # Initialize I/O
        for input in self.input.values():
            input.initialize(
                startTime=startTime,
                endTime=endTime,
                stepSize=stepSize,
                simulator=simulator,
            )
        for output in self.output.values():
            output.initialize(
                startTime=startTime,
                endTime=endTime,
                stepSize=stepSize,
                simulator=simulator,
            )

        if not self.INITIALIZED:
            # First initialization
            self._create_state_space_model()
            self.ss_model.initialize(startTime, endTime, stepSize, simulator)
            self.INITIALIZED = True
        else:
            # Re-initialize the state space
            self._create_state_space_model()  # We need to re-create the model because the parameters have changed to create a new computation graph
            self.ss_model.initialize(startTime, endTime, stepSize, simulator)

    def _create_state_space_model(self):
        """Create the state space model matrices using PyTorch tensors."""
        # Single state for CO2 concentration
        n_states = 1
        n_inputs = len(self.input)

        # Initialize A and B matrices with zeros
        A = torch.zeros((n_states, n_states), dtype=torch.float64)
        B = torch.zeros((n_states, n_inputs), dtype=torch.float64)

        # State matrix A: -sum of all flow rates / volume
        A[0, 0] = -(
            self.m_inf.get() / self.V.get()
        )  # Base coefficient from infiltration

        # Input matrix B coefficients
        # Supply air flow rate * supply air CO2
        B[0, 0] = 1 / self.V.get()  # supplyAirFlowRate coefficient
        B[0, 2] = 1 / self.V.get()  # outdoorCO2 coefficient

        # Exhaust air flow rate
        B[0, 1] = -1 / self.V.get()  # exhaustAirFlowRate coefficient

        # Outdoor CO2
        B[0, 2] = self.m_inf.get() / self.V.get()  # outdoorCO2 coefficient

        # Number of people
        B[0, 3] = self.G_occ.get() / self.V.get()  # numberOfPeople coefficient

        # Output matrix C - Identity matrix for direct observation
        C = torch.eye(n_states, dtype=torch.float64)

        # Feedthrough matrix D (no direct feedthrough)
        D = torch.zeros((n_states, n_inputs), dtype=torch.float64)

        # Initial state
        x0 = torch.tensor([self.output["indoorCO2"].get()], dtype=torch.float64)

        # E matrix for input-state coupling: shape (n_inputs, n_states, n_states)
        E = torch.zeros((n_inputs, n_states, n_states), dtype=torch.float64)
        # -m_ex*C (input 1, state 0)
        E[1, 0, 0] = -1 / self.V.get()  # exhaustAirFlowRate * C

        # F matrix for input-input coupling: shape (n_inputs, n_states, n_inputs)
        F = torch.zeros((n_inputs, n_states, n_inputs), dtype=torch.float64)
        # m_sup*C_sup (inputs 0 and 2)
        F[0, 0, 2] = 1 / self.V.get()  # supplyAirFlowRate * supplyAirCO2

        self.ss_model = DiscreteStatespaceSystem(
            A=A,
            B=B,
            C=C,
            D=D,
            x0=x0,
            state_names=None,
            add_noise=False,
            id=f"ss_mass_model_{self.id}",
            E=E,
            F=F,
        )

    @property
    def config(self):
        """Get the system configuration."""
        return self._config

    def do_step(
        self,
        secondTime: Optional[float] = None,
        dateTime: Optional[datetime.datetime] = None,
        stepSize: Optional[float] = None,
        stepIndex: Optional[int] = None,
    ) -> None:
        """Execute a single simulation step using the state-space model."""
        # Build input vector u
        u = torch.stack(
            [
                self.input["supplyAirFlowRate"].get(),
                self.input["exhaustAirFlowRate"].get(),
                self.input["outdoorCO2"].get(),
                self.input["numberOfPeople"].get(),
            ]
        ).squeeze()

        self.ss_model.input["u"].set(u, stepIndex)
        self.ss_model.do_step(secondTime, dateTime, stepSize, stepIndex=stepIndex)
        y = self.ss_model.output["y"].get()
        self.output["indoorCO2"].set(y[0], stepIndex)
