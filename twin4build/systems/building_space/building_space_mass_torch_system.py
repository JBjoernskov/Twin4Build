import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
import sympy as sp
from sympy import symbols, linear_eq_to_matrix
from twin4build import core
import twin4build.utils.input_output_types as tps
from twin4build.systems.utils.discrete_statespace_system import DiscreteStatespaceSystem
import datetime
from typing import Optional
from twin4build.utils.constants import Constants

class BuildingSpaceMassTorchSystem(core.System, nn.Module):
    """
    A building space model for CO2 concentration dynamics using a state space representation.
    Implements the mass balance equation for CO2 as described in the provided reference.
    
    Equation:
        m_z * dCz/dt = C_sup * m_dot_a,sup - Cz * m_dot_a,exh + K_occ * N_occ
    """
    def __init__(self,
                 airVolume: float,                # Mass of air in the room [kg]
                 CO2_occ_gain: float = 0.004, # CO2 generation per person [kg/s]
                 infiltrationRate: float = 0.0, # Infiltration rate [kg/s]
                 **kwargs):
        """
        Initialize the CO2 mass balance model.
        Args:
            m_air: Mass of air in the room [kg]
            CO2_occ_gain: CO2 generation per person [kg/s]
            **kwargs: Additional keyword arguments passed to parent
        """
        super().__init__(**kwargs)
        nn.Module.__init__(self)
        self.airVolume = nn.Parameter(torch.tensor(airVolume, dtype=torch.float32), requires_grad=False)
        self.CO2_occ_gain = nn.Parameter(torch.tensor(CO2_occ_gain, dtype=torch.float32), requires_grad=False)
        self.infiltrationRate = nn.Parameter(torch.tensor(infiltrationRate, dtype=torch.float32), requires_grad=False)
        self.airMass = self.airVolume*Constants.density["air"] ###


        MM_CO2 = 44.01 # g/mol
        MM_AIR = 28.97 # g/mol
        self.coeff = MM_AIR/MM_CO2*1e+6  # coefficient for converting CO2 mass to CO2 concentration [ppm]

        # Define inputs
        self.input = {
            "supplyAirCo2Concentration": tps.Scalar(),  # CO2 concentration in supply air [ppm]
            "supplyAirFlowRate": tps.Scalar(),          # Supply airflow rate [kg/s]
            "exhaustAirFlowRate": tps.Scalar(),         # Exhaust airflow rate [kg/s]
            "numberOfPeople": tps.Scalar(),             # Number of occupants
        }

        # Define outputs
        self.output = {
            "indoorCo2Concentration": tps.Scalar(400),   # Indoor CO2 concentration [ppm]
        }

        # Parameters for calibration
        self.parameter = {
            "airVolume": {"lb": 10.0, "ub": 1000.0},
            "CO2_occ_gain": {"lb": 0.001, "ub": 0.01},
        }
        self._config = {"parameters": list(self.parameter.keys())}
        self.INITIALIZED = False

    def initialize(self, startTime=None, endTime=None, stepSize=None, simulator=None):
        for input in self.input.values():
            input.initialize(startTime=startTime, endTime=endTime, stepSize=stepSize, simulator=simulator)
        for output in self.output.values():
            output.initialize(startTime=startTime, endTime=endTime, stepSize=stepSize, simulator=simulator)
        if not self.INITIALIZED:
            self._create_state_space_model()
            self.ss_model.initialize(startTime, endTime, stepSize, simulator)
            self.INITIALIZED = True
        else:
            self.ss_model.initialize(startTime, endTime, stepSize, simulator)

    def _create_state_space_model(self):
        # State: indoor CO2 concentration [ppm]
        n_states = 1
        n_inputs = 4  # [Csup, m_dot_sup, m_dot_exh, N_occ]
        # -Cz * infiltrationRate / m_air (input 0)
        A = torch.zeros((n_states, n_states), dtype=torch.float32)
        A[0, 0] = -self.infiltrationRate / self.airMass

        B = torch.zeros((n_states, n_inputs), dtype=torch.float32)
        # Csup * infiltrationRate / m_air (input 0)
        B[0, 0] = self.infiltrationRate / self.airMass

        # Kocc * N_occ / m_air (input 3)
        B[0, 3] = self.CO2_occ_gain / self.airMass

        # E matrix for state-input coupling: shape (n_inputs, n_states, n_states)
        E = torch.zeros((n_inputs, n_states, n_states), dtype=torch.float32)
        # -Cz * m_dot_exh / m_air (input 2)
        E[2, 0, 0] = -1.0 / self.airMass

        # F matrix for input-input coupling: shape (n_inputs, n_states, n_inputs)
        F = torch.zeros((n_inputs, n_states, n_inputs), dtype=torch.float32)
        # Csup * m_dot_sup / m_air (inputs 0 and 1)
        F[0, 0, 1] = 1.0 / self.airMass
        # Output matrix
        C = torch.eye(n_states, dtype=torch.float32)
        D = torch.zeros((n_states, n_inputs), dtype=torch.float32)
        x0 = torch.zeros(n_states, dtype=torch.float32)
        x0[0] = self.output["indoorCo2Concentration"].get()/self.coeff
        self.ss_model = DiscreteStatespaceSystem(
            A=A, B=B, C=C, D=D,
            x0=x0,
            state_names=None,
            add_noise=False,
            id=f"ss_mass_model_{self.id}",
            E=E,
            F=F
        )

    @property
    def config(self):
        return self._config

    def cache(self, startTime=None, endTime=None, stepSize=None):
        pass

    def do_step(self, secondTime: Optional[float] = None, dateTime: Optional[datetime.datetime] = None, stepSize: Optional[float] = None, stepIndex: Optional[int] = None) -> None:
        # Build input vector u: [Csup, m_dot_sup, m_dot_exh, N_occ]
        u = torch.stack([
            self.input["supplyAirCo2Concentration"].get()/self.coeff,
            self.input["supplyAirFlowRate"].get(),
            self.input["exhaustAirFlowRate"].get(),
            self.input["numberOfPeople"].get()
        ]).squeeze()
        self.ss_model.input["u"].set(u, stepIndex)
        self.ss_model.do_step(secondTime, dateTime, stepSize, stepIndex=stepIndex)
        y = self.ss_model.output["y"].get()
        self.output["indoorCo2Concentration"].set(y[0]*self.coeff, stepIndex)