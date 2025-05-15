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

class BuildingSpaceThermalTorchSystem(core.System, nn.Module):
    """
    A building space model using an RC (Resistance-Capacitance) thermal model 
    implemented with a discrete state space representation.
    
    This model represents the thermal dynamics of a building space considering:
    - Air temperature in multiple zones
    - Exterior wall temperature
    - Interior wall temperatures (for adjacent zones)
    - Heat exchange between zones and outdoor environment
    - Internal heat gains
    - HVAC inputs
    - Solar radiation gains
    - Space heater heat input
    
    The underlying state space model is derived from symbolic equations:
    x'(t) = A*x(t) + B*u(t)
    y(t) = C*x(t) + D*u(t)
    """

    def __init__(self,
                 # Thermal parameters
                 C_air: float,                # Thermal capacitance of indoor air [J/K]
                 C_wall: float,               # Thermal capacitance of exterior wall [J/K]
                 C_int: float,                # Thermal capacitance of internal structure [J/K]
                 C_boundary: float,           # Thermal capacitance of boundary wall [J/K]
                 R_out: float,                # Thermal resistance between wall and outdoor [K/W]
                 R_in: float,                 # Thermal resistance between wall and indoor [K/W]
                 R_int: float,                # Thermal resistance between internal structure and indoor air [K/W]
                 R_boundary: float,           # Thermal resistance of boundary [K/W]
                 # Heat gain parameters
                 f_wall: float = 0.3,         # Radiation factor for exterior wall
                 f_air: float = 0.1,          # Radiation factor for air
                 Q_occ_gain: float = 100.0,   # Heat gain per occupant [W]
                 **kwargs):
        """
        Initialize the RC building space model with interior wall dynamics.
        
        Args:
            C_air: Thermal capacitance of indoor air [J/K]
            C_wall: Thermal capacitance of exterior walls [J/K]
            C_int: Thermal capacitance of internal mass [J/K]
            C_boundary: Thermal capacitance of boundary wall [J/K]
            R_out: Thermal resistance between exterior wall and outdoor [K/W]
            R_in: Thermal resistance between exterior wall and indoor [K/W]
            R_int: Thermal resistance between internal mass and indoor air [K/W]
            R_boundary: Thermal resistance of boundary [K/W]
            f_wall: Radiation factor for exterior wall
            f_air: Radiation factor for air/internal mass
            Q_occ_gain: Heat gain per occupant [W]
            add_noise: Whether to add noise to the model
            **kwargs: Additional keyword arguments passed to parent
        """
        super().__init__(**kwargs)
        nn.Module.__init__(self)
        
        # Store thermal parameters as nn.Parameters
        self.C_air = nn.Parameter(torch.tensor(C_air, dtype=torch.float32), requires_grad=False)
        self.C_wall = nn.Parameter(torch.tensor(C_wall, dtype=torch.float32), requires_grad=False)
        self.C_int = nn.Parameter(torch.tensor(C_int, dtype=torch.float32), requires_grad=False)
        self.C_boundary = nn.Parameter(torch.tensor(C_boundary, dtype=torch.float32), requires_grad=False)
        self.R_out = nn.Parameter(torch.tensor(R_out, dtype=torch.float32), requires_grad=False)
        self.R_in = nn.Parameter(torch.tensor(R_in, dtype=torch.float32), requires_grad=False)
        self.R_int = nn.Parameter(torch.tensor(R_int, dtype=torch.float32), requires_grad=False)
        self.R_boundary = nn.Parameter(torch.tensor(R_boundary, dtype=torch.float32), requires_grad=False)
        
        # Store other parameters as nn.Parameters
        self.f_wall = nn.Parameter(torch.tensor(f_wall, dtype=torch.float32), requires_grad=False)
        self.f_air = nn.Parameter(torch.tensor(f_air, dtype=torch.float32), requires_grad=False)
        self.Q_occ_gain = nn.Parameter(torch.tensor(Q_occ_gain, dtype=torch.float32), requires_grad=False)
        
        # Define inputs and outputs
        self.input = {
            "outdoorTemperature": tps.Scalar(),   # Outdoor temperature [°C]
            "supplyAirFlowRate": tps.Scalar(),    # Supply air flow rate [kg/s]
            "exhaustAirFlowRate": tps.Scalar(),   # Exhaust air flow rate [kg/s]
            "supplyAirTemperature": tps.Scalar(), # Supply air temperature [°C]
            "globalIrradiation": tps.Scalar(),    # Solar radiation [W/m²]
            "numberOfPeople": tps.Scalar(),       # Number of occupants
            "Q_sh": tps.Scalar(),                 # Space heater heat input [W]
            "T_boundary": tps.Scalar(),           # Boundary temperature [°C]
            "indoorTemperature_adj": tps.Vector(),# Adjacent zone temperature [°C]
        }
        
        # Define outputs
        self.output = {
            "indoorTemperature": tps.Scalar(20),     # Indoor air temperature [°C]
            "wallTemperature": tps.Scalar(20),       # Exterior wall temperature [°C]
        }
        
        # Define parameters for calibration
        self.parameter = {
            "C_air": {"lb": 1000.0, "ub": 1000000.0},
            "C_wall": {"lb": 10000.0, "ub": 10000000.0},
            "C_int": {"lb": 10000.0, "ub": 10000000.0},
            "C_boundary": {"lb": 10000.0, "ub": 10000000.0},
            "R_out": {"lb": 0.001, "ub": 1.0},
            "R_in": {"lb": 0.001, "ub": 1.0},
            "R_int": {"lb": 0.001, "ub": 1.0},
            "R_boundary": {"lb": 0.001, "ub": 1.0},
            "f_wall": {"lb": 0.0, "ub": 1.0},
            "f_air": {"lb": 0.0, "ub": 1.0},
            "Q_occ_gain": {"lb": 50.0, "ub": 200.0},
        }
        
        self._config = {"parameters": list(self.parameter.keys())}
        self.INITIALIZED = False
    
    def initialize(self, 
                   startTime=None, 
                   endTime=None, 
                   stepSize=None, 
                   simulator=None):
        """
        Initialize the RC model by initializing the state space model.
        
        Args:
            startTime: Simulation start time.
            endTime: Simulation end time.
            stepSize: Simulation step size.
            model: Reference to the simulation model.
        """
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
    
    def _create_state_space_model(self):
        """
        Create the state space model using PyTorch tensors.
        
        This formulation directly constructs the state space matrices A and B
        using PyTorch tensors for gradient tracking.
        """
        # Find number of adjacent zones
        connection_point = [cp for cp in self.connectsAt if cp.inputPort == "indoorTemperature_adj"]
        n_adjacent_zones = len(connection_point[0].connectsSystemThrough) if connection_point else 0
        
        # Calculate number of states
        n_states = 2  # Base states: air and wall temperature
        n_states += 1  # Add boundary wall state
        n_states += n_adjacent_zones  # Add one state for each adjacent zone's interior wall
        
        # Calculate number of inputs based on input dictionary
        n_inputs = len(self.input)-1  # Base inputs from input dictionary
        n_inputs += n_adjacent_zones  # Add one input for each adjacent zone temperature
        
        # Initialize A and B matrices with zeros
        A = torch.zeros((n_states, n_states), dtype=torch.float32)
        B = torch.zeros((n_states, n_inputs), dtype=torch.float32)
        
        # Air temperature equation coefficients
        A[0, 0] = -1/(self.R_in * self.C_air) - 1/(self.R_boundary * self.C_air)
        A[0, 1] = 1/(self.R_in * self.C_air)  # T_wall coefficient
        A[0, 2] = 1/(self.R_boundary * self.C_air)  # T_bound_wall coefficient

        # Exterior wall temperature equation coefficients
        A[1, 0] = 1/(self.R_in * self.C_wall)  # T_air coefficient
        A[1, 1] = -1/(self.R_in * self.C_wall) - 1/(self.R_out * self.C_wall)  # T_wall coefficient

        # Add heat exchange with boundary wall
        A[2, 0] = 1/(self.R_boundary * self.C_boundary)  # T_air coefficient for boundary wall
        A[2, 2] = -1/(self.R_boundary * self.C_boundary)  # T_bound_wall coefficient
        
        # Add heat exchange with interior walls of adjacent zones
        for i in range(n_adjacent_zones):
            adj_wall_idx = 3 + i  # Interior walls are after boundary wall
            A[0, adj_wall_idx] = 1/(self.R_int * self.C_air)  # T_int_wall coefficient for each adjacent zone
            A[adj_wall_idx, 0] = 1/(self.R_int * self.C_int)  # T_air coefficient for each interior wall
            A[adj_wall_idx, adj_wall_idx] = -1/(self.R_int * self.C_int)  # T_int_wall coefficient for each interior wall
        
        # Input matrix B coefficients - match the order in do_step
        # Outdoor temperature
        B[1, 0] = 1/(self.R_out * self.C_wall)  # T_out coefficient for wall
        
        # Solar radiation
        B[0, 4] = self.f_air/self.C_air  # Radiation coefficient for air
        B[1, 4] = self.f_wall/self.C_wall  # Radiation coefficient for wall
        
        # Number of people
        B[0, 5] = self.Q_occ_gain/self.C_air  # N_people coefficient
        
        # Space heater heat input
        B[0, 6] = 1/self.C_air  # Q_sh coefficient
        
        # Boundary temperature
        B[2, 7] = 1/(self.R_boundary * self.C_boundary)  # T_bound coefficient
        
        # Adjacent zone temperatures (at the end of the input vector)
        for i in range(n_adjacent_zones):
            adj_wall_idx = 3 + i  # Interior walls are after boundary wall
            B[adj_wall_idx, 8+i] = 1/(self.R_int * self.C_int)  # T_adj coefficient for each adjacent zone
        
        # Output matrix C - Identity matrix for direct observation of all states
        C = torch.eye(n_states, dtype=torch.float32)
        
        # Feedthrough matrix D (no direct feedthrough)
        D = torch.zeros((n_states, n_inputs), dtype=torch.float32)
        
        # Initial state
        x0 = torch.zeros(n_states, dtype=torch.float32)
        x0[0] = self.output["indoorTemperature"].get()
        x0[1] = self.output["wallTemperature"].get()
        
        # Initialize boundary wall temperature with indoor temperature
        x0[2] = self.output["indoorTemperature"].get()
        
        # Initialize interior wall temperatures with indoor temperature
        for i in range(n_adjacent_zones):
            adj_wall_idx = 3 + i  # Interior walls are after boundary wall
            x0[adj_wall_idx] = self.output["indoorTemperature"].get()

        # E matrix for input-state coupling: shape (n_inputs, n_states, n_states)
        E = torch.zeros((n_inputs, n_states, n_states), dtype=torch.float32)
        # -m_ex*cp*T_air (input 2, state 0)
        E[2, 0, 0] = -Constants.specificHeatCapacity["air"]/self.C_air  # exhaustAirFlowRate * T_air

        # Use E and F matrices for correct couplings
        # F matrix for input-input coupling: shape (n_inputs, n_states, n_inputs)
        F = torch.zeros((n_inputs, n_states, n_inputs), dtype=torch.float32)
        # m_sup*cp*T_sup (inputs 1 and 3)
        F[1, 0, 3] = Constants.specificHeatCapacity["air"]/self.C_air  # supplyAirFlowRate * supplyAirTemperature

        # Pass E and F to DiscreteStatespaceSystem
        self.ss_model = DiscreteStatespaceSystem(
            A=A, B=B, C=C, D=D,
            x0=x0,
            state_names=None,
            add_noise=False,
            id=f"ss_model_{self.id}",
            E=E,
            F=F
        )
    
    @property
    def config(self):
        """Get the configuration of the RC model."""
        return self._config
    
    def cache(self, startTime=None, endTime=None, stepSize=None):
        """Cache method placeholder."""
        pass
    
    def do_step(self, 
                secondTime: Optional[float] = None, 
                dateTime: Optional[datetime.datetime] = None, 
                stepSize: Optional[float] = None, 
                stepIndex: Optional[int] = None) -> None:
        """
        Perform one step of the RC model simulation.
        
        Args:
            secondTime: Current simulation time in seconds.
            dateTime: Current simulation date/time.
            stepSize: Current simulation step size.
        """
        # Build input vector u with fixed inputs first
        u = torch.stack([
            self.input["outdoorTemperature"].get(),
            self.input["supplyAirFlowRate"].get(),
            self.input["exhaustAirFlowRate"].get(),
            self.input["supplyAirTemperature"].get(),
            self.input["globalIrradiation"].get(),
            self.input["numberOfPeople"].get(),
            self.input["Q_sh"].get(),
            self.input["T_boundary"].get()
        ]).squeeze()
        # Add adjacent zone temperatures at the end
        if self.input["indoorTemperature_adj"].get() is not None:
            u = torch.cat([u, self.input["indoorTemperature_adj"].get()])
        
        
        # Set the input vector
        self.ss_model.input["u"].set(u, stepIndex)
        
        # Execute state space model step
        self.ss_model.do_step(secondTime, dateTime, stepSize, stepIndex=stepIndex)
        
        # Get the output vector
        y = self.ss_model.output["y"].get()
        
        # Update individual outputs from the output vector
        self.output["indoorTemperature"].set(y[0], stepIndex)
        self.output["wallTemperature"].set(y[1], stepIndex)


    

    
