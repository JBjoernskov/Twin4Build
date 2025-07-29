# Standard library imports
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
    A state-space model for a space heater (radiator) with multiple finite elements.

    This model represents a radiator that transfers heat from hot water to a room.
    The radiator is discretized into multiple elements to capture the temperature
    distribution along its length. Each element has its own thermal mass and heat
    transfer characteristics.

    Mathematical Formulation
    -----------------------

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

    The total heat output is calculated as:

        .. math::

            Q = \frac{UA}{n} \sum_{i=1}^n (T_i - T_z)

    The model is initialized by solving for UA to match the nominal heat output:

        .. math::

            Q_{nom} = UA \cdot (T_{b,nom} - T_{z,nom})

    where:
       - :math:`Q_{nom}` is the nominal heat output [W]
       - :math:`T_{b,nom}` is the nominal return water temperature [°C]
       - :math:`T_{z,nom}` is the nominal zone temperature [°C]

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
    nelements : int, optional
        Number of finite elements. Defaults to 1.

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
        Dictionary containing parameter bounds for calibration:
        - "Q_flow_nominal_sh": Nominal heat output parameters
        - "T_a_nominal_sh": Nominal supply temperature parameters
        - "T_b_nominal_sh": Nominal return temperature parameters
        - "TAir_nominal_sh": Nominal air temperature parameters
        - "thermalMassHeatCapacity": Thermal mass parameters
        - "UA": Heat transfer coefficient parameters
    UA : torch.tps.Parameter
        Overall heat transfer coefficient [W/K], calculated during initialization
    thermalMassHeatCapacity : torch.tps.Parameter
        Total thermal mass heat capacity [J/K], stored as a PyTorch parameter

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

        self.input = {
            "supplyWaterTemperature": tps.Scalar(),
            "waterFlowRate": tps.Scalar(),
            "indoorTemperature": tps.Scalar(),
        }
        self.output = {
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

    def initialize(self, startTime=None, endTime=None, stepSize=None, simulator=None):
        """Initialize the space heater system for simulation.

        This method performs the following initialization steps:
        1. Numerically solves for the UA value that matches the nominal heat output
        2. Initializes input/output data structures
        3. Creates or reinitializes the state-space model

        Args:
            startTime (datetime, optional): Start time of the simulation period.
            endTime (datetime, optional): End time of the simulation period.
            stepSize (float, optional): Time step size in seconds.
            simulator (object, optional): Simulation model object.
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
