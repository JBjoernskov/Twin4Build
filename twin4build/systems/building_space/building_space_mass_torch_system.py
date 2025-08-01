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
    A building space mass balance model for CO2 concentration.

    This model represents the CO2 concentration dynamics in a building space considering:
    - Supply and exhaust air flows
    - Occupant CO2 generation
    - Infiltration
    - Outdoor CO2 concentration

    Mathematical Formulation:

       The CO2 concentration dynamics are represented using a mass balance equation:

       .. math::

          V\frac{dC}{dt} = \dot{m}_{sup}(C_{sup} - C) - \dot{m}_{exh}C + \dot{m}_{inf}(C_{out} - C) + G_{occ} N_{occ}

       where:

          - :math:`V`: Volume of the space [m³]
          - :math:`C`: CO2 concentration [ppm]
          - :math:`\dot{m}_{sup}`: Supply air flow rate [kg/s]
          - :math:`C_{sup}`: Supply air CO2 concentration [ppm]
          - :math:`\dot{m}_{exh}`: Exhaust air flow rate [kg/s]
          - :math:`\dot{m}_{inf}`: Infiltration rate [kg/s]
          - :math:`C_{out}`: Outdoor CO2 concentration [ppm]
          - :math:`G_{occ}`: CO2 generation rate per occupant [ppm·kg/s]
          - :math:`N_{occ}`: Number of occupants

       The model is implemented using a state-space representation for efficient computation
       and gradient-based optimization.

    Args:
        V (float): Volume of the space [m³]
        G_occ (float, optional): CO2 generation rate per occupant [ppm·kg/s]. Defaults to 5e-6
        m_inf (float, optional): Infiltration rate [kg/s]. Defaults to 0.001
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

    def initialize(self, startTime=None, endTime=None, stepSize=None, simulator=None):
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
