import torch
import torch.nn as nn
from twin4build import core
import twin4build.utils.input_output_types as tps
from twin4build.systems.utils.discrete_statespace_system import DiscreteStatespaceSystem
from twin4build.utils.constants import Constants
from scipy.optimize import fsolve
from typing import Dict, List, Optional

class SpaceHeaterTorchSystem(core.System, nn.Module):
    """
    State-space model for a space heater (radiator) with nelements finite elements.
    ODE for each element:
    C * dT_i/dt = m_dot * c_p * (T_{i-1} - T_i) - UA/nelements * (T_i - T_z)
    """
    def __init__(self,
                 Q_flow_nominal_sh: float,
                 T_a_nominal_sh: float,
                 T_b_nominal_sh: float,
                 TAir_nominal_sh: float,
                 thermalMassHeatCapacity: float,
                 nelements: int = 1,
                 **kwargs):
        super().__init__(**kwargs)
        nn.Module.__init__(self)
        self.Q_flow_nominal_sh = Q_flow_nominal_sh
        self.T_a_nominal_sh = T_a_nominal_sh
        self.T_b_nominal_sh = T_b_nominal_sh
        self.TAir_nominal_sh = TAir_nominal_sh
        self.thermalMassHeatCapacity = nn.Parameter(torch.tensor(thermalMassHeatCapacity, dtype=torch.float32), requires_grad=False)
        self.nelements = nelements
        self.UA = nn.Parameter(torch.tensor(1.0, dtype=torch.float32), requires_grad=False)  # Placeholder, will be set in initialize

        self.input = {
            "supplyWaterTemperature": tps.Scalar(),
            "waterFlowRate": tps.Scalar(),
            "indoorTemperature": tps.Scalar(),
        }
        self.output = {
            "outletWaterTemperature": tps.Vector(tensor=torch.ones(nelements)*21, size=nelements),
            "Power": tps.Scalar(),
            "Energy": tps.Scalar(),
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

    def initialize(self, 
                   startTime=None, 
                   endTime=None, 
                   stepSize=None, 
                   simulator=None):
        # Numerically solve for UA using fsolve so that steady-state output matches Q_flow_nominal_sh
        UA0 = float(self.Q_flow_nominal_sh / (self.T_b_nominal_sh - self.TAir_nominal_sh))
        root = fsolve(self._ua_residual, UA0, full_output=True)
        UA_val = root[0][0]
        self.UA.data = torch.tensor(UA_val, dtype=torch.float32)

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

        if not self.INITIALIZED:
            # First initialization
            self._create_state_space_model()
            self.ss_model.initialize(startTime, endTime, stepSize, simulator)
            self.INITIALIZED = True
        else:
            # Re-initialize the state space model
            self.ss_model.initialize(startTime, endTime, stepSize, simulator)

    def _ua_residual(self, UA_candidate):
        n = self.nelements
        C_elem = float(self.thermalMassHeatCapacity.item()) / n
        UA_elem = float(UA_candidate) / n
        m_dot = float(self.Q_flow_nominal_sh / (Constants.specificHeatCapacity["water"] * (self.T_a_nominal_sh - self.T_b_nominal_sh)))
        c_p = float(Constants.specificHeatCapacity["water"])
        # Build A, B
        A = torch.zeros((n, n), dtype=torch.float64)
        B = torch.zeros((n, 3), dtype=torch.float64)
        for i in range(n):
            A[i, i] = - (m_dot * c_p + UA_elem) / C_elem
            if i > 0:
                A[i, i-1] = (m_dot * c_p) / C_elem
        B[0, 0] = (m_dot * c_p) / C_elem
        for i in range(n):
            B[i, 2] = UA_elem / C_elem
        u = torch.tensor([self.T_a_nominal_sh, m_dot, self.TAir_nominal_sh], dtype=torch.float64)
        try:
            x_ss = -torch.linalg.solve(A, B @ u)
        except Exception:
            return 1e6
        Power = UA_elem * torch.sum(x_ss - self.TAir_nominal_sh)
        return Power - self.Q_flow_nominal_sh

    def _create_state_space_model(self):
        n = self.nelements
        n_inputs = 3  # [supplyWaterTemperature, waterFlowRate, indoorTemperature]
        C_elem = self.thermalMassHeatCapacity / n
        UA_elem = self.UA / n
        c_p = Constants.specificHeatCapacity["water"]

        # LTI part: Only UA/C on diagonal
        A = torch.zeros((n, n), dtype=torch.float32)
        for i in range(n):
            A[i, i] = -UA_elem / C_elem

        # B matrix: Only UA/C for indoor temperature input
        B = torch.zeros((n, n_inputs), dtype=torch.float32)
        for i in range(n):
            B[i, 2] = UA_elem / C_elem

        # State-input coupling (E): waterFlowRate
        E = torch.zeros((n_inputs, n, n), dtype=torch.float32)
        for i in range(n):
            E[1, i, i] = -c_p / C_elem
            if i > 0:
                E[1, i, i-1] = c_p / C_elem

        # Input-input coupling (F): T_supply * m_dot for first state
        F = torch.zeros((n_inputs, n, n_inputs), dtype=torch.float32)
        F[0, 0, 1] = c_p / C_elem  # Only first state, T_supply * m_dot

        C = torch.zeros((1, n), dtype=torch.float32)
        C[0, n-1] = 1.0
        D = torch.zeros((1, n_inputs), dtype=torch.float32)
        x0 = torch.zeros(n, dtype=torch.float32)
        x0[:] = self.output["outletWaterTemperature"].get()

        self.ss_model = DiscreteStatespaceSystem(
            A=A, B=B, C=C, D=D,
            x0=x0,
            state_names=[f"T_{i+1}" for i in range(n)],
            E=E,  # State-input coupling
            F=F,  # Input-input coupling
            add_noise=False,
            id=f"ss_model_{self.id}"
        )

    def do_step(self, 
                secondTime=None, 
                dateTime=None, 
                stepSize=None, 
                stepIndex: Optional[int] = None):
        u = torch.stack([
            self.input["supplyWaterTemperature"].get(),
            self.input["waterFlowRate"].get(),
            self.input["indoorTemperature"].get()
        ]).squeeze()
        self.ss_model.input["u"].set(u, stepIndex)
        self.ss_model.do_step(secondTime, dateTime, stepSize, stepIndex)
        y = self.ss_model.output["y"].get()
        outletWaterTemperature = y[0]
        UA_elem = self.UA / self.nelements
        temps = self.ss_model.get_state()
        Power = UA_elem * torch.sum(temps - u[2])
        self.output["outletWaterTemperature"].set(outletWaterTemperature, stepIndex)
        self.output["Power"].set(Power, stepIndex)