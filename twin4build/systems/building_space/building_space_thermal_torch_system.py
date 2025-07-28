import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
import twin4build.core as core
import twin4build.utils.types as tps
from twin4build.systems.utils.discrete_statespace_system import DiscreteStatespaceSystem
import datetime
from typing import Optional
from twin4build.utils.constants import Constants

class BuildingSpaceThermalTorchSystem(core.System, nn.Module):
    r"""
    Building Space Thermal System Model (PyTorch-based)

    This class implements a state-space thermal model for building spaces using a PyTorch-based
    implementation. The model represents thermal dynamics using a network of thermal resistances
    and capacitances (RC network).

    Mathematical Formulation:

       The thermal dynamics are represented using a state-space model:

       .. math::

          \frac{d\mathbf{x}}{dt} = \mathbf{A}\mathbf{x} + \mathbf{B}\mathbf{u} + \mathbf{E}\mathbf{x}\mathbf{u} + \mathbf{F}\mathbf{u}\mathbf{u}
          \mathbf{y} = \mathbf{C}\mathbf{x} + \mathbf{D}\mathbf{u}

       where:

          - :math:`\mathbf{x}`: State vector containing temperatures
          - :math:`\mathbf{u}`: Input vector containing environmental and control inputs
          - :math:`\mathbf{A}, \mathbf{B}, \mathbf{C}, \mathbf{D}`: State-space matrices
          - :math:`\mathbf{E}, \mathbf{F}`: Nonlinear coupling matrices

       The state vector :math:`\mathbf{x}` contains:

          - :math:`T_i`: Indoor air temperature
          - :math:`T_w`: Exterior wall temperature
          - :math:`T_{bw}`: Boundary wall temperature
          - :math:`T_{iw}`: Interior wall temperatures for adjacent zones

       The input vector :math:`\mathbf{u}` contains:

          - :math:`T_o`: Outdoor temperature
          - :math:`\dot{m}_{sup}`: Supply air flow rate
          - :math:`\dot{m}_{exh}`: Exhaust air flow rate
          - :math:`T_{sup}`: Supply air temperature
          - :math:`\Phi_{sol}`: Solar radiation
          - :math:`N_{occ}`: Number of occupants
          - :math:`Q_{sh}`: Space heater heat input
          - :math:`T_{bound}`: Boundary temperature
          - :math:`T_{adj}`: Adjacent zone temperatures

       The system matrices are constructed based on the following physical relationships:

       1. Indoor Air Temperature:

          .. math::

             C_{air}\frac{dT_i}{dt} = \frac{T_w - T_i}{R_{in}} + \frac{T_{bw} - T_i}{R_{boundary}} + \sum_{j}\frac{T_{iw,j} - T_i}{R_{int}} + Q_{occ} + Q_{sh} + f_{air}\Phi_{sol} + c_p\dot{m}_{sup}(T_{sup} - T_i)

       2. Exterior Wall Temperature:

          .. math::

             C_{wall}\frac{dT_w}{dt} = \frac{T_o - T_w}{R_{out}} + \frac{T_i - T_w}{R_{in}} + f_{wall}\Phi_{sol}

       3. Boundary Wall Temperature:

          .. math::

             C_{boundary}\frac{dT_{bw}}{dt} = \frac{T_i - T_{bw}}{R_{boundary}} + \frac{T_{bound} - T_{bw}}{R_{boundary}}

       4. Interior Wall Temperature (for each adjacent zone):

          .. math::

             C_{int}\frac{dT_{iw,j}}{dt} = \frac{T_i - T_{iw,j}}{R_{int}} + \frac{T_{adj,j} - T_{iw,j}}{R_{int}}

       where:

          - :math:`C_{air}, C_{wall}, C_{boundary}, C_{int}`: Thermal capacitances
          - :math:`R_{in}, R_{out}, R_{boundary}, R_{int}`: Thermal resistances
          - :math:`f_{air}, f_{wall}`: Radiation factors
          - :math:`Q_{occ}`: Heat gain from occupants
          - :math:`Q_{sh}`: Space heater heat input
          - :math:`c_p`: Specific heat capacity of air

       This model represents the thermal dynamics of a building space considering:
          - Air temperature in multiple zones
          - Exterior wall temperature
          - Interior wall temperatures (for adjacent zones)
          - Heat exchange between zones and outdoor environment
          - Internal heat gains
          - HVAC inputs
          - Solar radiation gains
          - Space heater heat input

       The model is implemented using a state-space representation for efficient computation
       and gradient-based optimization.

    Args:
       C_air (float): Thermal capacitance of indoor air [J/K]
       C_wall (float): Thermal capacitance of exterior wall [J/K]
       C_int (float): Thermal capacitance of internal structure [J/K]
       C_boundary (float): Thermal capacitance of boundary wall [J/K]
       R_out (float): Thermal resistance between wall and outdoor [K/W]
       R_in (float): Thermal resistance between wall and indoor [K/W]
       R_int (float): Thermal resistance between internal structure and indoor air [K/W]
       R_boundary (float): Thermal resistance of boundary [K/W]
       f_wall (float, optional): Radiation factor for exterior wall. Defaults to 0.3
       f_air (float, optional): Radiation factor for air. Defaults to 0.1
       Q_occ_gain (float, optional): Heat gain per occupant [W]. Defaults to 100.0
    """

    def __init__(self,
                 # Thermal parameters
                 C_air: float = 1e+6,                # Thermal capacitance of indoor air [J/K]
                 C_wall: float = 1e+6,               # Thermal capacitance of exterior wall [J/K]
                 C_int: float = 1e+5,                # Thermal capacitance of internal structure [J/K]
                 C_boundary: float = 1e+6,           # Thermal capacitance of boundary wall [J/K]
                 R_out: float = 0.05,                # Thermal resistance between wall and outdoor [K/W]
                 R_in: float = 0.05,                 # Thermal resistance between wall and indoor [K/W]
                 R_int: float = 0.01,                # Thermal resistance between internal structure and indoor air [K/W]
                 R_boundary: float = 0.01,           # Thermal resistance of boundary [K/W]
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
            **kwargs: Additional keyword arguments passed to parent
        """
        super().__init__(**kwargs)
        nn.Module.__init__(self)
        
        # Store thermal parameters as tps.Parameters
        self.C_air = tps.Parameter(torch.tensor(C_air, dtype=torch.float64), requires_grad=False)
        self.C_wall = tps.Parameter(torch.tensor(C_wall, dtype=torch.float64), requires_grad=False)
        self.C_int = tps.Parameter(torch.tensor(C_int, dtype=torch.float64), requires_grad=False)
        self.C_boundary = tps.Parameter(torch.tensor(C_boundary, dtype=torch.float64), requires_grad=False)
        self.R_out = tps.Parameter(torch.tensor(R_out, dtype=torch.float64), requires_grad=False)
        self.R_in = tps.Parameter(torch.tensor(R_in, dtype=torch.float64), requires_grad=False)
        self.R_int = tps.Parameter(torch.tensor(R_int, dtype=torch.float64), requires_grad=False)
        self.R_boundary = tps.Parameter(torch.tensor(R_boundary, dtype=torch.float64), requires_grad=False)
        
        # Store other parameters as tps.Parameters
        self.f_wall = tps.Parameter(torch.tensor(f_wall, dtype=torch.float64), requires_grad=False)
        self.f_air = tps.Parameter(torch.tensor(f_air, dtype=torch.float64), requires_grad=False)
        self.Q_occ_gain = tps.Parameter(torch.tensor(Q_occ_gain, dtype=torch.float64), requires_grad=False)
        
        # Define inputs and outputs
        self.input = {
            "outdoorTemperature": tps.Scalar(),   # Outdoor temperature [°C]
            "supplyAirFlowRate": tps.Scalar(),    # Supply air flow rate [kg/s]
            "exhaustAirFlowRate": tps.Scalar(),   # Exhaust air flow rate [kg/s]
            "supplyAirTemperature": tps.Scalar(), # Supply air temperature [°C]
            "globalIrradiation": tps.Scalar(),    # Solar radiation [W/m²]
            "numberOfPeople": tps.Scalar(),       # Number of occupants
            "heatGain": tps.Scalar(),              # Space heater heat input [W]
            "boundaryTemperature": tps.Scalar(21),   # Boundary temperature [°C], optional
            "adjacentZoneTemperature": tps.Vector(), # Adjacent zone temperature [°C], optional
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
        self._n_adjacent_zones = 0
        self._n_boundary_temperature = 0
        self._manual_setup_n_adjacent_zones = False
        self._manual_setup_n_boundary_temperature = False

    @property
    def n_adjacent_zones(self):
        return self._n_adjacent_zones

    @n_adjacent_zones.setter
    def n_adjacent_zones(self, n_adjacent_zones: int):
        self._manual_setup_n_adjacent_zones = True
        self._n_adjacent_zones = n_adjacent_zones

    @property
    def n_boundary_temperature(self):
        return self._n_boundary_temperature

    @n_boundary_temperature.setter
    def n_boundary_temperature(self, n_boundary_temperature: int):
        self._manual_setup_n_boundary_temperature = True
        self._n_boundary_temperature = n_boundary_temperature

    @property
    def manual_setup_n_adjacent_zones(self):
        return self._manual_setup_n_adjacent_zones

    @property
    def manual_setup_n_boundary_temperature(self):
        return self._manual_setup_n_boundary_temperature


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
            # print("CREATED STATE SPACE MODEL 1")
            # print("C_air: ", self.C_air.get().detach())
            self.ss_model.initialize(startTime, endTime, stepSize, simulator)
            self.INITIALIZED = True
        else:
            # Re-initialize the state space model
            self._create_state_space_model() # We need to re-create the model because the parameters have changed to create a new computation graph
            # print("CREATED STATE SPACE MODEL 2")
            # print("C_air: ", self.C_air.get().detach())
            self.ss_model.initialize(startTime, endTime, stepSize, simulator)

        self._manual_setup_n_adjacent_zones = False
        self._manual_setup_n_boundary_temperature = False
        
    
    def _create_state_space_model(self):
        """
        Create the state space model using PyTorch tensors.
        
        This formulation directly constructs the state space matrices A and B
        using PyTorch tensors for gradient tracking.
        """
        if self.manual_setup_n_boundary_temperature==False:
            # Find if boundary temperature is set as input
            connection_point = [cp for cp in self.connectsAt if cp.inputPort == "boundaryTemperature"]
            n_boundary_temperature = len(connection_point[0].connectsSystemThrough) if connection_point else 0
            self.n_boundary_temperature = n_boundary_temperature
        assert self.n_boundary_temperature==0 or self.n_boundary_temperature==1, "Maximum one boundary temperature input is allowed"

        if self.manual_setup_n_adjacent_zones==False:
            # Find number of adjacent zones
            connection_point = [cp for cp in self.connectsAt if cp.inputPort == "adjacentZoneTemperature"]
            n_adjacent_zones = len(connection_point[0].connectsSystemThrough) if connection_point else 0
            self.n_adjacent_zones = n_adjacent_zones

        # Calculate number of states
        n_states = 2  # Base states: air and wall temperature
        n_states += self.n_boundary_temperature  # Add boundary wall state
        n_states += self.n_adjacent_zones  # Add one state for each adjacent zone's interior wall
        self.n_states = n_states
        
        # Calculate number of inputs based on input dictionary
        n_inputs = len(self.input)-2  # Base inputs from input dictionary
        n_inputs += self.n_adjacent_zones  # Add one input for each adjacent zone temperature
        n_inputs += self.n_boundary_temperature  # Add one input for boundary temperature
        self.n_inputs = n_inputs

        # Initialize A and B matrices with zeros
        A = torch.zeros((n_states, n_states), dtype=torch.float64)
        B = torch.zeros((n_states, n_inputs), dtype=torch.float64)
        
        # Air temperature equation coefficients
        A[0, 0] = -1/(self.R_in.get() * self.C_air.get())
        A[0, 1] = 1/(self.R_in.get() * self.C_air.get())  # T_wall coefficient

        if self.n_boundary_temperature == 1:
            # Add heat exchange with boundary wall
            A[0, 0] -= 1/(self.R_boundary.get() * self.C_air.get())  # T_bound_wall coefficient
            A[0, 2] = 1/(self.R_boundary.get() * self.C_air.get())  # T_bound_wall coefficient
            A[2, 0] = 1/(self.R_boundary.get() * self.C_boundary.get())  # T_air coefficient for boundary wall
            A[2, 2] = -2/(self.R_boundary.get() * self.C_boundary.get())  # T_bound_wall coefficient (heat exchange with both air and boundary)

        # Add heat loss to interior walls of adjacent zones
        A[0, 0] -= self.n_adjacent_zones * 1/(self.R_int.get() * self.C_air.get())  # Heat loss to interior walls

        # Exterior wall temperature equation coefficients
        A[1, 0] = 1/(self.R_in.get() * self.C_wall.get())  # T_air coefficient
        A[1, 1] = -1/(self.R_in.get() * self.C_wall.get()) - 1/(self.R_out.get() * self.C_wall.get())  # T_wall coefficient

        
        # Add heat exchange with interior walls of adjacent zones
        for i in range(self.n_adjacent_zones):
            adj_wall_idx = (n_states - self.n_adjacent_zones - self.n_boundary_temperature) + i  # Interior walls are after boundary wall
            A[0, adj_wall_idx] = 1/(self.R_int.get() * self.C_air.get())  # T_int_wall coefficient for each adjacent zone
            A[adj_wall_idx, 0] = 1/(self.R_int.get() * self.C_int.get())  # T_air coefficient for each interior wall
            A[adj_wall_idx, adj_wall_idx] = -2/(self.R_int.get() * self.C_int.get())  # T_int_wall coefficient for each interior wall (heat exchange with both air and adjacent zone)
        
        # Input matrix B coefficients - match the order in do_step
        # Outdoor temperature
        B[1, 0] = 1/(self.R_out.get() * self.C_wall.get())  # T_out coefficient for wall
        
        # Solar radiation
        B[0, 4] = self.f_air.get()/self.C_air.get()  # Radiation coefficient for air
        B[1, 4] = self.f_wall.get()/self.C_wall.get()  # Radiation coefficient for wall
        
        # Number of people
        B[0, 5] = self.Q_occ_gain.get()/self.C_air.get()  # N_people coefficient
        
        # Space heater heat input
        B[0, 6] = 1/self.C_air.get()  # Q_sh coefficient
        
        if self.n_boundary_temperature == 1:
            # Boundary temperature
            B[2, 7] = 1/(self.R_boundary.get() * self.C_boundary.get())  # T_bound coefficient
        
        # Adjacent zone temperatures (at the end of the input vector)
        for i in range(self.n_adjacent_zones):
            adj_wall_state_idx = (n_states - self.n_adjacent_zones - self.n_boundary_temperature) + i  # Interior walls are after boundary wall
            adj_wall_input_idx = (n_inputs - self.n_adjacent_zones - self.n_boundary_temperature) + i  # Adjacent zone temperatures are after boundary temperature
            B[adj_wall_state_idx, adj_wall_input_idx] = 1/(self.R_int.get() * self.C_int.get())  # T_adj coefficient for each adjacent zone
        
        # Output matrix C - Identity matrix for direct observation of all states
        C = torch.eye(n_states, dtype=torch.float64)
        
        # Feedthrough matrix D (no direct feedthrough)
        D = torch.zeros((n_states, n_inputs), dtype=torch.float64)
        
        # Initial state
        x0 = torch.zeros(n_states, dtype=torch.float64)
        x0[0] = self.output["indoorTemperature"].get()
        x0[1] = self.output["wallTemperature"].get()
        

        if self.n_boundary_temperature == 1:
            # Initialize boundary wall temperature with indoor temperature
            x0[2] = self.output["indoorTemperature"].get()
            
        # Initialize interior wall temperatures with indoor temperature
        for i in range(self.n_adjacent_zones):
            adj_wall_idx = (n_states - self.n_adjacent_zones - self.n_boundary_temperature) + i  # Interior walls are after boundary wall
            x0[adj_wall_idx] = self.output["indoorTemperature"].get()

        # E matrix for input-state coupling: shape (n_inputs, n_states, n_states)
        E = torch.zeros((n_inputs, n_states, n_states), dtype=torch.float64)
        # -m_ex*cp*T_air (input 2, state 0)
        E[2, 0, 0] = -Constants.specificHeatCapacity["air"]/self.C_air.get()  # exhaustAirFlowRate * T_air

        # Use E and F matrices for correct couplings
        # F matrix for input-input coupling: shape (n_inputs, n_states, n_inputs)
        F = torch.zeros((n_inputs, n_states, n_inputs), dtype=torch.float64)
        # m_sup*cp*T_sup (inputs 1 and 3)
        F[1, 0, 3] = Constants.specificHeatCapacity["air"]/self.C_air.get()  # supplyAirFlowRate * supplyAirTemperature

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
        
        # # Debug output for parameter validation
        # if torch.any(torch.isnan(A)) or torch.any(torch.isinf(A)):
        #     print("WARNING: A matrix contains NaN or Inf values!")
        #     print("Parameters:")
        #     print(f"C_air: {self.C_air.get().item()}")
        #     print(f"C_wall: {self.C_wall.get().item()}")  
        #     print(f"C_boundary: {self.C_boundary.get().item()}")
        #     print(f"R_out: {self.R_out.get().item()}")
        #     print(f"R_in: {self.R_in.get().item()}")
        #     print(f"R_boundary: {self.R_boundary.get().item()}")
        #     print("A matrix:", A)
            
        # # Check for very small resistances that could cause numerical instability
        # if self.R_boundary.get() < 1e-4:
        #     print(f"WARNING: R_boundary is very small ({self.R_boundary.get().item():.6f}), this may cause numerical instability!")
        # if self.R_in.get() < 1e-4:
        #     print(f"WARNING: R_in is very small ({self.R_in.get().item():.6f}), this may cause numerical instability!")
        # if self.R_out.get() < 1e-4:
        #     print(f"WARNING: R_out is very small ({self.R_out.get().item():.6f}), this may cause numerical instability!")
    
    @property
    def config(self):
        """Get the configuration of the RC model."""
        return self._config
    
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
            self.input["heatGain"].get(),
        ]).squeeze()
        
        if self.n_boundary_temperature == 1:
            u = torch.cat([u, self.input["boundaryTemperature"].get()])
        # Add adjacent zone temperatures at the end
        if self.n_adjacent_zones > 0:
            u = torch.cat([u, self.input["adjacentZoneTemperature"].get()])
        
        # Set the input vector
        self.ss_model.input["u"].set(u, stepIndex)
        
        # Execute state space model step
        self.ss_model.do_step(secondTime, dateTime, stepSize, stepIndex=stepIndex)
        
        # Get the output vector
        y = self.ss_model.output["y"].get()

        
        # Update individual outputs from the output vector
        self.output["indoorTemperature"].set(y[0], stepIndex)
        self.output["wallTemperature"].set(y[1], stepIndex)

        # if torch.any(torch.isnan(self.output["indoorTemperature"].get())):
        #     print("Parameters:")
        #     print(f"C_air: {self.C_air.get().item()}")
        #     print(f"C_wall: {self.C_wall.get().item()}")
        #     print(f"C_int: {self.C_int.get().item()}")
        #     print(f"C_boundary: {self.C_boundary.get().item()}")
        #     print(f"R_out: {self.R_out.get().item()}")
        #     print(f"R_in: {self.R_in.get().item()}")
        #     print(f"R_int: {self.R_int.get().item()}")
        #     print(f"R_boundary: {self.R_boundary.get().item()}")
        #     print(f"f_wall: {self.f_wall.get().item()}")
        #     print(f"f_air: {self.f_air.get().item()}")
        #     print(f"Q_occ_gain: {self.Q_occ_gain.get().item()}")

        #     print(f"Indoor temperature is NaN at step {stepIndex}")
        #     print(f"Input vector: {u}")
        #     print(f"Output vector: {y}")
        #     print(f"State vector: {self.ss_model.x}")


        #     raise ValueError("Indoor temperature is NaN")


    

    
