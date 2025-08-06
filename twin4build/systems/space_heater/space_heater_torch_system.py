# Standard library imports
import datetime
from typing import Dict, List, Optional

# Third party imports
import torch
import torch.nn as nn
from scipy.optimize import fsolve

# Local application imports
import twin4build.utils.types as tps
from twin4build import core
from twin4build.systems.utils.discrete_statespace_system import DiscreteStatespaceSystem
from twin4build.translator.translator import (
    Exact,
    MultiPath,
    Node,
    Optional_,
    SignaturePattern,
    SinglePath,
)
from twin4build.utils.constants import Constants


def get_signature_pattern():
    """
    Get the signature pattern of the FMU component.

    Returns:
        SignaturePattern: The signature pattern for the building space 0 adjacent boundary outdoor FMU system.
    """

    node2 = Node(cls=core.namespace.S4BLDG.BuildingSpace)
    node3 = Node(cls=core.namespace.S4BLDG.Valve)  # supply valve
    node4 = Node(cls=core.namespace.S4BLDG.SpaceHeater)
    sp = SignaturePattern(
        semantic_model_=core.ontologies,
        ownedBy="BuildingSpace0AdjBoundaryOutdoorFMUSystem",
        priority=60,
    )

    sp.add_triple(
        Exact(
            subject=node3, object=node2, predicate=core.namespace.S4BLDG.isContainedIn
        )
    )
    sp.add_triple(
        Exact(
            subject=node4, object=node2, predicate=core.namespace.S4BLDG.isContainedIn
        )
    )
    sp.add_triple(
        Exact(subject=node3, object=node4, predicate=core.namespace.FSO.suppliesFluidTo)
    )

    sp.add_input("waterFlowRate", node3)
    sp.add_input("indoorTemperature", node2, "indoorTemperature")
    sp.add_modeled_node(node4)

    return sp


class SpaceHeaterTorchSystem(core.System, nn.Module):
    r"""
    Space Heater (Radiator) Model with Finite Element Discretization.

    This model represents a radiator that transfers heat from hot water to a room using
    multiple finite elements and bilinear state-space dynamics. The radiator is 
    discretized to capture temperature distribution along its length with flow-dependent
    heat transfer characteristics.

    Mathematical Formulation:
    =========================

    **Continuous-Time Differential Equations:**

    The model uses a state-space representation with n finite elements. For each element i,
    the temperature dynamics are described by:

    .. math::

        C_i \frac{dT_i}{dt} = \dot{m} \cdot c_p \cdot (T_{i-1} - T_i) - \frac{UA}{n} \cdot (T_i - T_z)

    where:
       - :math:`C_i` is the thermal capacitance of element i [J/K]
       - :math:`T_i` is the temperature of element i [°C]
       - :math:`\dot{m}` is the water flow rate [kg/s]
       - :math:`c_p` is the specific heat capacity of water [J/(kg·K)]
       - :math:`UA` is the overall heat transfer coefficient [W/K]
       - :math:`n` is the number of elements
       - :math:`T_z` is the zone (room) temperature [°C]

    **Heat Output Calculation:**

    The total heat output is calculated as:

    .. math::

        Q = \frac{UA}{n} \sum_{i=1}^n (T_i - T_z)

    **State-Space Representation:**

    The system is implemented using the DiscreteStatespaceSystem with matrices:

    *State vector:* :math:`\mathbf{x} = [T_1, T_2, ..., T_n]^T`

    *Input vector:* :math:`\mathbf{u} = [T_{sup}, \dot{m}, T_z]^T`

    *Base System Matrices:*

    For n finite elements:

    .. math::

       \mathbf{A} = \left[\begin{array}{cccc}
       -\frac{UA/n}{C_1} & 0 & 0 & 0 \\
       0 & -\frac{UA/n}{C_2} & 0 & 0 \\
       0 & 0 & \ddots & 0 \\
       0 & 0 & 0 & -\frac{UA/n}{C_n}
       \end{array}\right]

       \mathbf{B} = \left[\begin{array}{ccc}
       0 & 0 & \frac{UA/n}{C_1} \\
       0 & 0 & \frac{UA/n}{C_2} \\
       \vdots & \vdots & \vdots \\
       0 & 0 & \frac{UA/n}{C_n}
       \end{array}\right]

       \mathbf{C} = \left[ 0 \quad 0 \quad \cdots \quad 1 \right]

       \mathbf{D} = \left[ 0 \quad 0 \quad 0 \right]

    **Bilinear Coupling Matrices:**

    *State-Input Coupling (E matrices):*

    .. math::

       \mathbf{E} \in \mathbb{R}^{3 \times n \times n} = \left[\begin{array}{l}
       \mathbf{0}_{n \times n} \text{ (supply temperature)} \\
       \left[\begin{array}{cccc}
       -\frac{c_p}{C_1} & 0 & 0 & 0 \\
       \frac{c_p}{C_2} & -\frac{c_p}{C_2} & 0 & 0 \\
       0 & \frac{c_p}{C_3} & \ddots & 0 \\
       0 & 0 & \cdots & -\frac{c_p}{C_n}
       \end{array}\right] \text{ (water flow rate)} \\
       \mathbf{0}_{n \times n} \text{ (zone temperature)}
       \end{array}\right]

    *Input-Input Coupling (F matrices):*

    .. math::

       \mathbf{F} \in \mathbb{R}^{3 \times n \times 3} = \left[\begin{array}{l}
       \left[\begin{array}{ccc}
       0 & \frac{c_p}{C_1} & 0 \\
       0 & 0 & 0 \\
       \vdots & \vdots & \vdots \\
       0 & 0 & 0
       \end{array}\right] \text{ (supply temperature)} \\
       \mathbf{0}_{n \times 3} \text{ (water flow rate)} \\
       \mathbf{0}_{n \times 3} \text{ (zone temperature)}
       \end{array}\right]

    **Discretization with Bilinear Terms:**

    The bilinear terms are incorporated into the base matrices before discretization:

    *Step 1: Compute Effective Matrices*

    .. math::

       \mathbf{A}_{eff}[k] = \mathbf{A} + \sum_{i=1}^{3} \mathbf{E}_i u_i[k]

       \mathbf{B}_{eff}[k] = \mathbf{B} + \sum_{i=1}^{3} \mathbf{F}_i u_i[k]

    where the effective matrices depend on the current input vector :math:`\mathbf{u}[k]`.

    *Step 2: Discretize Effective Matrices*

    The effective matrices are then discretized using the matrix exponential method implemented
    in the DiscreteStatespaceSystem base class.

    *Step 3: Bilinear Effects*

    The bilinear terms handle specific flow-dependent heat transfer effects:
       - :math:`\mathbf{E}[1,i,i] \cdot u_1 \cdot x_i = -\frac{c_p}{C_i} \dot{m} T_i`: Water flow removing heat from element i
       - :math:`\mathbf{E}[1,i,i-1] \cdot u_1 \cdot x_{i-1} = \frac{c_p}{C_i} \dot{m} T_{i-1}`: Water flow bringing heat from previous element
       - :math:`\mathbf{F}[0,0,1] \cdot u_0 \cdot u_1 = \frac{c_p}{C_1} T_{sup} \dot{m}`: Supply temperature bringing heat to first element

    **Model Initialization:**

    The model is initialized by solving for UA to match the nominal heat output:

    .. math::

        Q_{nom} = UA \cdot (T_{b,nom} - T_{z,nom})

    where:
       - :math:`Q_{nom}` is the nominal heat output [W]
       - :math:`T_{b,nom}` is the nominal return water temperature [°C]
       - :math:`T_{z,nom}` is the nominal zone temperature [°C]

    Physical Interpretation:
    ======================

    **Finite Element Discretization:**
       - Each element represents a section of the radiator with its own thermal mass
       - States capture temperature distribution along radiator length
       - Flow-dependent heat transfer between elements modeled with bilinear terms

    **Flow-Dependent Effects:**
       - Water flow brings heat at supply temperature to first element (F matrix coupling)
       - Water flow transfers heat between consecutive elements (E matrix coupling)
       - These effects are critical for accurate radiator performance modeling

    Computational Features:
    ======================

       - **Automatic Differentiation:** PyTorch tensors enable gradient computation
       - **Adaptive Discretization:** Matrices updated when flow conditions change significantly
       - **Efficient Simulation:** Optimized for HVAC component simulation
       - **Parameter Estimation:** UA coefficient and thermal mass available for calibration

    Parameters
    ----------
    Q_flow_nominal_sh : float
        Nominal heat output [W]
    T_a_nominal_sh : float
        Nominal supply water temperature [°C]
    T_b_nominal_sh : float
        Nominal return water temperature [°C]
    TAir_nominal_sh : float
        Nominal room air temperature [°C]
    thermalMassHeatCapacity : float
        Total thermal mass heat capacity [J/K]
    nelements : int, default=1
        Number of finite elements

    Attributes
    ----------
    input : Dict[str, Scalar]
        Dictionary containing input ports:
        - "supplyWaterTemperature": Supply water temperature [°C]
        - "waterFlowRate": Water flow rate [kg/s]
        - "indoorTemperature": Indoor air temperature [°C]
    output : Dict[str, Union[Scalar, Vector]]
        Dictionary containing output ports:
        - "outletWaterTemperature": Vector of element temperatures [°C]
        - "Power": Heat output power [W]
    parameter : Dict[str, Dict]
        Dictionary containing parameter bounds for calibration
    UA : torch.tps.Parameter
        Overall heat transfer coefficient [W/K], calculated during initialization
    thermalMassHeatCapacity : torch.tps.Parameter
        Total thermal mass heat capacity [J/K], stored as a PyTorch parameter

    Examples
    --------
    Basic space heater model:

    >>> import twin4build as tb
    >>>
    >>> # Create single-element radiator
    >>> heater = tb.SpaceHeaterTorchSystem(
    ...     Q_flow_nominal_sh=2000,        # 2 kW nominal output
    ...     T_a_nominal_sh=70,             # 70°C supply temperature
    ...     T_b_nominal_sh=50,             # 50°C return temperature
    ...     TAir_nominal_sh=20,            # 20°C room temperature
    ...     thermalMassHeatCapacity=50000, # 50 kJ/K thermal mass
    ...     nelements=1,
    ...     id="radiator_single"
    ... )

    Multi-element radiator for detailed modeling:

    >>> # Create multi-element radiator for better temperature distribution
    >>> heater = tb.SpaceHeaterTorchSystem(
    ...     Q_flow_nominal_sh=3500,        # 3.5 kW nominal output
    ...     T_a_nominal_sh=75,             # Higher supply temperature
    ...     T_b_nominal_sh=55,             # Higher return temperature
    ...     TAir_nominal_sh=22,            # Slightly warmer room
    ...     thermalMassHeatCapacity=80000, # Higher thermal mass
    ...     nelements=5,                   # 5 elements for detail
    ...     id="radiator_multi_element"
    ... )

    Notes
    -----
    Model Characteristics:
       - The radiator is discretized into multiple elements for accurate
         temperature distribution modeling
       - Each element has its own thermal mass and heat transfer characteristics
       - The UA value is calculated numerically to match nominal conditions
       - The model accounts for both convective and radiative heat transfer

    Implementation Details:
       - The model uses a state-space representation for efficient computation
       - All calculations are performed using PyTorch tensors for gradient tracking
       - The UA value is optimized using numerical methods during initialization
       - The model supports both steady-state and dynamic simulations
    """

    sp = [get_signature_pattern()]

    def __init__(
        self,
        Q_flow_nominal_sh: float = 1000,
        T_a_nominal_sh: float = 60,
        T_b_nominal_sh: float = 45,
        TAir_nominal_sh: float = 21,
        thermalMassHeatCapacity: float = 500000,
        nelements: int = 3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        nn.Module.__init__(self)
        self.Q_flow_nominal_sh = Q_flow_nominal_sh
        self.T_a_nominal_sh = T_a_nominal_sh
        self.T_b_nominal_sh = T_b_nominal_sh
        self.TAir_nominal_sh = TAir_nominal_sh
        self.nelements = nelements
        self.UA = tps.Parameter(
            torch.tensor(10.0, dtype=torch.float64), requires_grad=False
        )  # Placeholder, will be set in initialize
        self.thermalMassHeatCapacity = tps.Parameter(
            torch.tensor(thermalMassHeatCapacity, dtype=torch.float64),
            requires_grad=False,
        )

        # Define inputs and outputs as private variables
        self._input = {
            "supplyWaterTemperature": tps.Scalar(),
            "waterFlowRate": tps.Scalar(),
            "indoorTemperature": tps.Scalar(),
        }
        self._output = {
            # "outletWaterTemperature": tps.Vector(tensor=torch.ones(nelements)*21, size=nelements),
            "outletWaterTemperature": tps.Scalar(21),
            "Power": tps.Scalar(0),
        }
        self.parameter = {
            "Q_flow_nominal_sh": {},
            "T_a_nominal_sh": {},
            "T_b_nominal_sh": {},
            "TAir_nominal_sh": {},
            "thermalMassHeatCapacity": {},
            "UA": {},
        }
        self._config = {"parameters": list(self.parameter.keys())}
        self.INITIALIZED = False

    @property
    def config(self) -> Dict[str, List[str]]:
        """Get the configuration parameters.

        Returns:
            Dict[str, List[str]]: Dictionary containing configuration parameter names.
        """
        return self._config

    @property
    def input(self) -> dict:
        """
        Get the input ports of the space heater system.

        Returns:
            dict: Dictionary containing input ports:
                - "supplyWaterTemperature": Supply water temperature [°C]
                - "waterFlowRate": Water flow rate [kg/s]
                - "indoorTemperature": Indoor air temperature [°C]
        """
        return self._input

    @property
    def output(self) -> dict:
        """
        Get the output ports of the space heater system.

        Returns:
            dict: Dictionary containing output ports:
                - "outletWaterTemperature": Outlet water temperature [°C]
                - "Power": Heating power [W]
        """
        return self._output

    def initialize(
        self,
        startTime: datetime.datetime,
        endTime: datetime.datetime,
        stepSize: int,
        simulator: core.Simulator,
    ) -> None:
        """Initialize the space heater system for simulation.

        This method performs the following initialization steps:
        1. Numerically solves for the UA value that matches the nominal heat output
        2. Initializes input/output data structures
        3. Creates or reinitializes the state-space model

        Args:
            startTime (datetime.datetime): Start time of the simulation period.
            endTime (datetime.datetime): End time of the simulation period.
            stepSize (int): Time step size in seconds.
            simulator (core.Simulator): Simulation model object.
        """

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
            # Numerically solve for UA using fsolve so that steady-state output matches Q_flow_nominal_sh
            UA0 = float(
                self.Q_flow_nominal_sh / (self.T_b_nominal_sh - self.TAir_nominal_sh)
            )
            root = fsolve(self._ua_residual, UA0, full_output=True)
            UA_val = root[0][0]
            self.UA.data = torch.tensor(UA_val, dtype=torch.float64)
            # First initialization
            self._create_state_space_model()
            self.ss_model.initialize(startTime, endTime, stepSize, simulator)
            self.INITIALIZED = True
        else:
            # Re-initialize the state space model
            self._create_state_space_model()
            self.ss_model.initialize(startTime, endTime, stepSize, simulator)

    def _ua_residual(self, UA_candidate):
        """Calculate the residual for UA optimization.

        This method is used by fsolve to find the UA value that matches the nominal
        heat output. It calculates the steady-state temperatures and heat output for
        a given UA value and returns the difference from the nominal heat output.

        Args:
            UA_candidate (float): Candidate UA value to evaluate.

        Returns:
            float: Difference between calculated and nominal heat output.
        """
        n = self.nelements
        C_elem = float(self.thermalMassHeatCapacity.get().item()) / n
        UA_elem = float(UA_candidate) / n
        m_dot = float(
            self.Q_flow_nominal_sh
            / (
                Constants.specificHeatCapacity["water"]
                * (self.T_a_nominal_sh - self.T_b_nominal_sh)
            )
        )
        c_p = float(Constants.specificHeatCapacity["water"])
        # Build A, B
        A = torch.zeros((n, n), dtype=torch.float64)
        B = torch.zeros((n, 3), dtype=torch.float64)
        for i in range(n):
            A[i, i] = -(m_dot * c_p + UA_elem) / C_elem
            if i > 0:
                A[i, i - 1] = (m_dot * c_p) / C_elem
        B[0, 0] = (m_dot * c_p) / C_elem
        for i in range(n):
            B[i, 2] = UA_elem / C_elem
        u = torch.tensor(
            [self.T_a_nominal_sh, m_dot, self.TAir_nominal_sh], dtype=torch.float64
        )
        try:
            x_ss = -torch.linalg.solve(A, B @ u)
        except Exception:
            return 1e6
        Power = UA_elem * torch.sum(x_ss - self.TAir_nominal_sh)
        return Power - self.Q_flow_nominal_sh

    def _create_state_space_model(self):
        """Create the state-space model for the space heater.

        This method creates a discrete state-space model representing the thermal
        dynamics of the space heater. The model includes:
        - State matrix A for thermal dynamics
        - Input matrix B for external inputs
        - Output matrix C for temperature output
        - Feedthrough matrix D
        - State-input coupling matrix E for flow effects
        - Input-input coupling matrix F for supply temperature effects
        """
        n = self.nelements
        n_inputs = 3  # [supplyWaterTemperature, waterFlowRate, indoorTemperature]
        C_elem = self.thermalMassHeatCapacity.get() / n
        UA_elem = self.UA.get() / n
        c_p = Constants.specificHeatCapacity["water"]

        # LTI part: Only UA/C on diagonal
        A = torch.zeros((n, n), dtype=torch.float64)
        for i in range(n):
            A[i, i] = -UA_elem / C_elem

        # B matrix: Only UA/C for indoor temperature input
        B = torch.zeros((n, n_inputs), dtype=torch.float64)
        for i in range(n):
            B[i, 2] = UA_elem / C_elem

        # State-input coupling (E): waterFlowRate
        E = torch.zeros((n_inputs, n, n), dtype=torch.float64)
        for i in range(n):
            E[1, i, i] = -c_p / C_elem
            if i > 0:
                E[1, i, i - 1] = c_p / C_elem

        # Input-input coupling (F): T_supply * m_dot for first state
        F = torch.zeros((n_inputs, n, n_inputs), dtype=torch.float64)
        F[0, 0, 1] = c_p / C_elem  # Only first state, T_supply * m_dot

        C = torch.zeros((1, n), dtype=torch.float64)
        C[0, n - 1] = 1.0
        D = torch.zeros((1, n_inputs), dtype=torch.float64)
        x0 = torch.zeros(n, dtype=torch.float64)
        x0[:] = self.output["outletWaterTemperature"].get()

        self.ss_model = DiscreteStatespaceSystem(
            A=A,
            B=B,
            C=C,
            D=D,
            x0=x0,
            state_names=[f"T_{i+1}" for i in range(n)],
            E=E,  # State-input coupling
            F=F,  # Input-input coupling
            add_noise=False,
            id=f"ss_model_{self.id}",
        )

    def do_step(
        self,
        secondTime=None,
        dateTime=None,
        stepSize=None,
        stepIndex: Optional[int] = None,
    ):
        """Perform one simulation step.

        This method advances the state-space model by one time step and calculates
        the outlet water temperature and heat output. The method:
        1. Collects current input values
        2. Updates the state-space model
        3. Calculates the heat output based on element temperatures
        4. Updates output values

        Args:
            secondTime (float, optional): Current simulation time in seconds.
            dateTime (datetime, optional): Current simulation date and time.
            stepSize (float, optional): Time step size in seconds.
            stepIndex (int, optional): Current simulation step index.
        """
        u = torch.stack(
            [
                self.input["supplyWaterTemperature"].get(),
                self.input["waterFlowRate"].get(),
                self.input["indoorTemperature"].get(),
            ]
        ).squeeze()
        self.ss_model.input["u"].set(u, stepIndex)
        self.ss_model.do_step(secondTime, dateTime, stepSize, stepIndex)
        y = self.ss_model.output["y"].get()
        outletWaterTemperature = y[0]
        UA_elem = self.UA.get() / self.nelements
        temps = self.ss_model.get_state()
        # print("----")
        # print("temps: ", temps)
        # print("u[2]: ", u[2])
        Power = UA_elem * torch.sum(temps - u[2])
        # print("Power: ", Power)
        self.output["outletWaterTemperature"].set(outletWaterTemperature, stepIndex)
        self.output["Power"].set(Power, stepIndex)
