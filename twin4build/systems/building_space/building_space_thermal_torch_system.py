# Standard library imports
import datetime
import warnings
from typing import Any, Dict, List, Optional

# Third party imports
import numpy as np
import torch
import torch.nn as nn

# Local application imports
import twin4build.core as core
import twin4build.utils.constants as constants
import twin4build.utils.types as tps
from twin4build.systems.utils.discrete_statespace_system import (
    DiscreteStatespaceSystem,
    bilinear_onestep,
)
from twin4build.translator.translator import (
    StepRule,
    AnyPathRule,
    Node,
    SignaturePattern,
    PathRule,
)


class BuildingSpaceThermalTorchSystem(core.System, nn.Module):
    r"""
    Building Space Thermal Model using RC Network Dynamics.

    This class implements a thermal model for building spaces using a network of thermal 
    resistances and capacitances (RC network). The model represents heat transfer between 
    indoor air, exterior walls and (optionally) a boundary wall using bilinear 
    state-space dynamics. Heat exchange with neighbouring zones or other boundary
    temperatures is modeled by connecting one or more
    :class:`~twin4build.systems.wall.wall_torch_system.WallTorchSystem` components
    to the ``wallHeatGain`` vector input port.

    Args:
        C_air: Thermal capacitance of indoor air [J/K]
        C_wall: Thermal capacitance of exterior wall [J/K]
        C_boundary: Thermal capacitance of boundary wall [J/K] (deprecated, use WallTorchSystem)
        R_out: Thermal resistance between wall and outdoor [K/W]
        R_in: Thermal resistance between wall and indoor [K/W]
        R_boundary: Thermal resistance of boundary [K/W] (deprecated, use WallTorchSystem)
        f_wall: Radiation factor for exterior wall
        f_air: Radiation factor for air
        Q_occ_gain: Heat gain per occupant [W]

    Mathematical Formulation
    ------------------------

    **Continuous-Time Differential Equations:**

    The thermal dynamics are governed by energy balance equations for each thermal node:

    *1. Indoor Air Temperature:*

    .. math::

       C_{air}\frac{dT_i}{dt} = \frac{T_w - T_i}{R_{in}} + \frac{T_{bw} - T_i}{R_{boundary}} + \sum_{j}\dot{Q}_{wall,j} + Q_{occ} N_{occ} + Q_{sh} + f_{air}\Phi_{sol} + c_p\dot{m}_{sup}(T_{sup} - T_i) - c_p\dot{m}_{exh}T_i

    *2. Exterior Wall Temperature:*

       .. math::

          C_{wall}\frac{dT_w}{dt} = \frac{T_o - T_w}{R_{out}} + \frac{T_i - T_w}{R_{in}} + f_{wall}\Phi_{sol}

    *3. Boundary Wall Temperature (if present; deprecated):*

       .. math::

          C_{boundary}\frac{dT_{bw}}{dt} = \frac{T_i - T_{bw}}{R_{boundary}} + \frac{T_{bound} - T_{bw}}{R_{boundary}}

    where:

       - :math:`T_i`: Indoor air temperature [°C] (state)
       - :math:`T_w`: Exterior wall temperature [°C] (state)  
       - :math:`T_{bw}`: Boundary wall temperature [°C] (state, optional, deprecated)
       - :math:`T_o`: Outdoor temperature [°C] (input)
       - :math:`T_{sup}`: Supply air temperature [°C] (input)
       - :math:`T_{bound}`: Boundary temperature [°C] (input, optional, deprecated)
       - :math:`\dot{Q}_{wall,j}`: Heat flow from connected wall j [W] (input,
         optional; produced by a ``WallTorchSystem``, which owns the wall state
         so the interzonal energy balance holds by construction)
       - :math:`\dot{m}_{sup}`: Supply air flow rate [kg/s] (input)
       - :math:`\dot{m}_{exh}`: Exhaust air flow rate [kg/s] (input)
       - :math:`\Phi_{sol}`: Solar radiation [W/m²] (input)
       - :math:`N_{occ}`: Number of occupants (input)
       - :math:`Q_{sh}`: Space heater heat input [W] (input)

    **State-Space Representation:**

    The system is implemented using the DiscreteStatespaceSystem with matrices:

    *State vector:* :math:`\mathbf{x} = \begin{bmatrix}T_i \\ T_w \\ T_{bw}\end{bmatrix}`

    *Input vector:* :math:`\mathbf{u} = \begin{bmatrix}T_o \\ \dot{m}_{sup} \\ \dot{m}_{exh} \\ T_{sup} \\ \Phi_{sol} \\ N_{occ} \\ Q_{sh} \\ T_{bound} \\ \dot{Q}_{wall,1} \\ \vdots \\ \dot{Q}_{wall,n}\end{bmatrix}`

    *Base System Matrices:*

    For a system with base thermal states (air, wall) + 1 boundary and 1 connected wall:

    .. math::

       \mathbf{A} = \begin{bmatrix}
       -\frac{1}{R_{in}C_{air}} - \frac{1}{R_{boundary}C_{air}} & \frac{1}{R_{in}C_{air}} & \frac{1}{R_{boundary}C_{air}} \\
       \frac{1}{R_{in}C_{wall}} & -\frac{1}{R_{in}C_{wall}} - \frac{1}{R_{out}C_{wall}} & 0 \\
       \frac{1}{R_{boundary}C_{boundary}} & 0 & -\frac{2}{R_{boundary}C_{boundary}}
       \end{bmatrix}

       \mathbf{B} = \begin{bmatrix}
       0 & 0 & 0 & 0 & \frac{f_{air}}{C_{air}} & \frac{Q_{occ}}{C_{air}} & \frac{1}{C_{air}} & 0 & \frac{1}{C_{air}} \\
       \frac{1}{R_{out}C_{wall}} & 0 & 0 & 0 & \frac{f_{wall}}{C_{wall}} & 0 & 0 & 0 & 0 \\
       0 & 0 & 0 & 0 & 0 & 0 & 0 & \frac{1}{R_{boundary}C_{boundary}} & 0
       \end{bmatrix}

       \mathbf{C} = \begin{bmatrix}
       1 & 0 & 0 \\
       0 & 1 & 0
       \end{bmatrix}

       \mathbf{D} = \begin{bmatrix}
       0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
       0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0
       \end{bmatrix}

    **Bilinear Coupling Matrices:**

    *State-Input Coupling (E matrices):*

    The only state-input coupling is the exhaust flow removing heat at the
    indoor air temperature:

    .. math::

       \mathbf{E}[2, 0, 0] = -\frac{c_p}{C_{air}} \quad \text{(exhaust flow} \cdot T_i\text{)}

    *Input-Input Coupling (F matrices):*

    The only input-input coupling is the supply flow bringing heat at the
    supply air temperature:

    .. math::

       \mathbf{F}[1, 0, 3] = \frac{c_p}{C_{air}} \quad \text{(supply flow} \cdot T_{sup}\text{)}

    Input vector mapping: :math:`[T_o, \dot{m}_{sup}, \dot{m}_{exh}, T_{sup}, \Phi_{sol}, N_{occ}, Q_{sh}, T_{bound}, \dot{Q}_{wall,1}]^T`

    *Bilinear Effects*

    The bilinear terms handle specific flow-dependent heat transfer effects:
       - :math:`\mathbf{E}[2,0,0] \cdot u_2 \cdot x_0 = -\frac{c_p}{C_{air}} \dot{m}_{exh} T_i`: Exhaust air removing heat
       - :math:`\mathbf{F}[1,0,3] \cdot u_1 \cdot u_3 = \frac{c_p}{C_{air}} \dot{m}_{sup} T_{sup}`: Supply air bringing heat

    Physical Interpretation
    -----------------------

    **Thermal Network:**
       - RC network represents building thermal mass and resistances
       - States capture temperature of air, walls, and structural elements
       - Inputs represent weather, HVAC, occupancy, and heat sources
       - Bilinear terms model flow-dependent heat transfer accurately

    **Interzonal Heat Transfer:**
       - Partition walls between zones are modeled by a separate
         ``WallTorchSystem``: the zone sends its ``indoorTemperature`` to the
         wall and receives the wall's heat flow on ``wallHeatGain``
       - Because a single wall component owns the wall state, the heat leaving
         one zone equals the heat stored in the wall plus the heat entering the
         other zone (energy-consistent by construction)

    **Flow-Dependent Effects:**
       - Supply air flow brings heat at supply temperature (F matrix coupling)
       - Exhaust air flow removes heat at indoor temperature (E matrix coupling)
       - These effects are critical for accurate HVAC modeling

    Computational Features
    ----------------------

       - **Automatic Differentiation:** PyTorch tensors enable gradient computation
       - **Adaptive Discretization:** Matrices updated when flows change significantly
       - **Parameter Estimation:** All RC parameters available for calibration


    Examples
    --------
    Basic thermal model:

    >>> import twin4build as tb
    >>>
    >>> # Create thermal model with default RC parameters
    >>> thermal_model = tb.BuildingSpaceThermalTorchSystem(
    ...     C_air=2e6,      # Higher air thermal mass
    ...     C_wall=5e6,     # Wall thermal mass
    ...     R_out=0.1,      # Outdoor thermal resistance
    ...     R_in=0.05,      # Indoor thermal resistance
    ...     f_air=0.15,     # Air radiation factor
    ...     id="zone_1_thermal"
    ... )

    Zone coupled to a neighbour zone through a wall component:

    >>> wall = tb.WallTorchSystem(C=2e5, R_a=0.05, R_b=0.05, id="wall_AB")
    >>> # zone_a.indoorTemperature -> wall.temperatureA
    >>> # wall.heatFlowRateA -> zone_a.wallHeatGain (and mirrored for zone_b)
    """

    def __init__(
        self,
        # Thermal parameters
        C_air: float = 1e6,  # Thermal capacitance of indoor air [J/K]
        C_wall: float = 1e6,  # Thermal capacitance of exterior wall [J/K]
        C_boundary: float = 1e6,  # Thermal capacitance of boundary wall [J/K] (deprecated)
        R_out: float = 0.05,  # Thermal resistance between wall and outdoor [K/W]
        R_in: float = 0.05,  # Thermal resistance between wall and indoor [K/W]
        R_boundary: float = 0.01,  # Thermal resistance of boundary [K/W] (deprecated)
        # Heat gain parameters
        f_wall: float = 0.3,  # Radiation factor for exterior wall
        f_air: float = 0.1,  # Radiation factor for air
        Q_occ_gain: float = 100.0,  # Heat gain per occupant [W]
        **kwargs,
    ):
        """
        Initialize the RC building space model.

        Args:
            C_air: Thermal capacitance of indoor air [J/K]
            C_wall: Thermal capacitance of exterior walls [J/K]
            C_boundary: Thermal capacitance of boundary wall [J/K]
                (deprecated -- connect a ``WallTorchSystem`` instead)
            R_out: Thermal resistance between exterior wall and outdoor [K/W]
            R_in: Thermal resistance between exterior wall and indoor [K/W]
            R_boundary: Thermal resistance of boundary [K/W]
                (deprecated -- connect a ``WallTorchSystem`` instead)
            f_wall: Radiation factor for exterior wall
            f_air: Radiation factor for air/internal mass
            Q_occ_gain: Heat gain per occupant [W]
            **kwargs: Additional keyword arguments passed to parent
        """
        super().__init__(**kwargs)
        nn.Module.__init__(self)

        # Store thermal parameters as tps.Parameters
        self.C_air = tps.Parameter(
            torch.tensor(C_air, dtype=tps.float_dtype()), requires_grad=False, scaling="log"
        )
        self.C_wall = tps.Parameter(
            torch.tensor(C_wall, dtype=tps.float_dtype()),
            requires_grad=False,
            scaling="log",
        )
        self.C_boundary = tps.Parameter(
            torch.tensor(C_boundary, dtype=tps.float_dtype()),
            requires_grad=False,
            scaling="log",
        )
        self.R_out = tps.Parameter(
            torch.tensor(R_out, dtype=tps.float_dtype()), requires_grad=False, scaling="log"
        )
        self.R_in = tps.Parameter(
            torch.tensor(R_in, dtype=tps.float_dtype()), requires_grad=False, scaling="log"
        )
        self.R_boundary = tps.Parameter(
            torch.tensor(R_boundary, dtype=tps.float_dtype()),
            requires_grad=False,
            scaling="log",
        )

        # Store other parameters as tps.Parameters
        self.f_wall = tps.Parameter(
            torch.tensor(f_wall, dtype=tps.float_dtype()), requires_grad=False
        )
        self.f_air = tps.Parameter(
            torch.tensor(f_air, dtype=tps.float_dtype()), requires_grad=False
        )
        self.Q_occ_gain = tps.Parameter(
            torch.tensor(Q_occ_gain, dtype=tps.float_dtype()), requires_grad=False
        )

        # Define inputs and outputs
        self.input = {
            "outdoorTemperature": tps.Scalar(),  # Outdoor temperature [°C]
            "supplyAirFlowRate": tps.Scalar(),  # Supply air flow rate [kg/s]
            "exhaustAirFlowRate": tps.Scalar(),  # Exhaust air flow rate [kg/s]
            "supplyAirTemperature": tps.Scalar(),  # Supply air temperature [°C]
            "globalIrradiation": tps.Scalar(),  # Solar radiation [W/m²]
            "numberOfPeople": tps.Scalar(),  # Number of occupants
            "heatGain": tps.Scalar(),  # Space heater heat input [W]
            "boundaryTemperature": tps.Scalar(
                21, optional=True
            ),  # Boundary temperature [°C], optional (deprecated: use WallTorchSystem)
            "wallHeatGain": tps.Vector(
                optional=True
            ),  # Heat flow from connected WallTorchSystem components [W], optional
        }

        # Define outputs
        self.output = {
            "indoorTemperature": tps.Scalar(20),  # Indoor air temperature [°C]
            "wallTemperature": tps.Scalar(20),  # Exterior wall temperature [°C]
        }

        # Define parameters for calibration
        self.parameter = {
            "C_air": {"lb": 1000.0, "ub": 1000000.0},
            "C_wall": {"lb": 10000.0, "ub": 10000000.0},
            "C_boundary": {"lb": 10000.0, "ub": 10000000.0},
            "R_out": {"lb": 0.001, "ub": 1.0},
            "R_in": {"lb": 0.001, "ub": 1.0},
            "R_boundary": {"lb": 0.001, "ub": 1.0},
            "f_wall": {"lb": 0.0, "ub": 1.0},
            "f_air": {"lb": 0.0, "ub": 1.0},
            "Q_occ_gain": {"lb": 50.0, "ub": 200.0},
        }

        self._config = {"parameters": list(self.parameter.keys())}
        self.INITIALIZED = False
        self._n_walls = 0
        self._n_boundary_temperature = 0
        self._manual_setup_n_walls = False
        self._manual_setup_n_boundary_temperature = False

    @property
    def n_walls(self):
        return self._n_walls

    @n_walls.setter
    def n_walls(self, n_walls: int):
        self._manual_setup_n_walls = True
        self._n_walls = n_walls

    @property
    def n_boundary_temperature(self):
        return self._n_boundary_temperature

    @n_boundary_temperature.setter
    def n_boundary_temperature(self, n_boundary_temperature: int):
        self._manual_setup_n_boundary_temperature = True
        self._n_boundary_temperature = n_boundary_temperature

    @property
    def manual_setup_n_walls(self):
        return self._manual_setup_n_walls

    @property
    def manual_setup_n_boundary_temperature(self):
        return self._manual_setup_n_boundary_temperature

    def initialize(
        self,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
        step_size: int,
    ) -> None:
        """
        Initialize the RC model by initializing the state space model.

        Args:
            start_time (datetime.datetime): Simulation start time.
            end_time (datetime.datetime): Simulation end time.
            step_size (int): Simulation step size.
            simulator (core.Simulator): Reference to the simulation model.
        """
        _, _, max_timesteps, _ = core.Simulator.get_simulation_timesteps(
            start_time, end_time, step_size
        )
        batch_size = len(start_time)

        if hasattr(self, "_n_c_compiled") and self._n_c_compiled > 1:
            self.n_c = self._n_c_compiled
        else:
            self.n_c = 1

        self.setup_variable_inputs()
        self.input["wallHeatGain"].initialize(
            n_t=max_timesteps, n_s=batch_size, n_c=self.n_c, n_v=self.n_walls
        )
        # Initialize I/O
        for input in self.input.values():
            input.initialize(
                n_t=max_timesteps,
                n_s=batch_size,
                n_c=self.n_c,
            )
        for output in self.output.values():
            output.initialize(
                n_t=max_timesteps,
                n_s=batch_size,
                n_c=self.n_c,
            )

        # Expand parameters to n_c dimension for vectorization
        self.C_air = self.C_air.expand_to_n_c(self.n_c)
        self.C_wall = self.C_wall.expand_to_n_c(self.n_c)
        self.C_boundary = self.C_boundary.expand_to_n_c(self.n_c)
        self.R_out = self.R_out.expand_to_n_c(self.n_c)
        self.R_in = self.R_in.expand_to_n_c(self.n_c)
        self.R_boundary = self.R_boundary.expand_to_n_c(self.n_c)
        self.f_wall = self.f_wall.expand_to_n_c(self.n_c)
        self.f_air = self.f_air.expand_to_n_c(self.n_c)
        self.Q_occ_gain = self.Q_occ_gain.expand_to_n_c(self.n_c)

        if not self.INITIALIZED:
            # First initialization
            self._create_state_space_model()
            # print("CREATED STATE SPACE MODEL 1")
            # print("C_air: ", self.C_air.get().detach())
            self.ss_model.initialize(start_time, end_time, step_size)

            # FIX: Set correct initial state for batch
            x0_tensor = self._get_initial_state_tensor()
            self.ss_model.set_state(x0_tensor)

            self.INITIALIZED = True
        else:
            # Re-initialize the state space model
            self._create_state_space_model()  # We need to re-create the model because the parameters might have changed to create a new computation graph
            # print("CREATED STATE SPACE MODEL 2")
            # print("C_air: ", self.C_air.get().detach())
            self.ss_model.initialize(start_time, end_time, step_size)

            # FIX: Set correct initial state for batch
            x0_tensor = self._get_initial_state_tensor()
            self.ss_model.set_state(x0_tensor)

        self._manual_setup_n_walls = False
        self._manual_setup_n_boundary_temperature = False

        # Drop per-params forward caches: a fresh simulation must not reuse
        # matrices (or their autograd graph) from a previous run.
        self._fwd_mat_cache = None
        self._forward_params_cache = None

    def setup_variable_inputs(self):
        if self.manual_setup_n_boundary_temperature == False:
            # Find if boundary temperature is set as input
            connection_point = [
                cp for cp in self.connects_at if cp.input_port == "boundaryTemperature"
            ]
            n_boundary_temperature = (
                len(connection_point[0].connects_system_through)
                if connection_point
                else 0
            )
            self.n_boundary_temperature = n_boundary_temperature
        assert (
            self.n_boundary_temperature == 0 or self.n_boundary_temperature == 1
        ), "Maximum one boundary temperature input is allowed"
        if self.n_boundary_temperature == 1:
            warnings.warn(
                "The in-zone boundary-wall path (boundaryTemperature / R_boundary / "
                "C_boundary) is deprecated. Connect a WallTorchSystem to the "
                "wallHeatGain port instead.",
                DeprecationWarning,
                stacklevel=2,
            )

        if self.manual_setup_n_walls == False:
            # Find number of connected walls
            connection_point = [
                cp for cp in self.connects_at if cp.input_port == "wallHeatGain"
            ]
            n_walls = (
                len(connection_point[0].connects_system_through)
                if connection_point
                else 0
            )
            self.n_walls = n_walls

    def _get_initial_state_tensor(self):
        # Get dimensions from indoorTemperature
        # Scalar.get() returns shape (n_s, n_c)
        t_indoor = self.output["indoorTemperature"].get()
        n_s = t_indoor.shape[0]
        n_c = t_indoor.shape[1]

        # x0 shape: (n_s, n_c, n_states)
        x0 = torch.zeros(
            (n_s, n_c, self.n_states),
            dtype=t_indoor.dtype,
            device=t_indoor.device,
        )

        t_wall = self.output["wallTemperature"].get()  # (n_s, n_c)

        x0[:, :, 0] = t_indoor
        x0[:, :, 1] = t_wall

        if self.n_boundary_temperature == 1:
            # Initialize boundary wall temperature with indoor temperature
            x0[:, :, 2] = t_indoor

        return x0

    #: Physical RC parameters, in a fixed order (the ``forward`` theta contract).
    PARAM_NAMES = (
        "C_air", "C_wall", "C_boundary",
        "R_in", "R_out", "R_boundary",
        "f_air", "f_wall", "Q_occ_gain",
    )

    #: Fusable coupling ports (see FusedStateSpaceSystem): connected
    #: WallTorchSystem heat flows enter the linear B matrix, and
    #: ``indoorTemperature`` is a pure state observation.
    FUSABLE_INPUT_PORTS = frozenset({"wallHeatGain"})
    FUSABLE_OUTPUT_PORTS = frozenset({"indoorTemperature"})

    def _ss_layout(self):
        """Port <-> matrix index map, mirroring :meth:`forward` exactly.

        ``u = [outdoorTemperature, supplyAirFlowRate, exhaustAirFlowRate,
        supplyAirTemperature, globalIrradiation, numberOfPeople, heatGain,
        (boundaryTemperature,) wallHeatGain x n_walls]``; output rows are the
        observed states.  Valid after :meth:`initialize` (needs ``n_walls`` /
        ``n_boundary_temperature``).
        """
        u = [
            ("outdoorTemperature", 1), ("supplyAirFlowRate", 1),
            ("exhaustAirFlowRate", 1), ("supplyAirTemperature", 1),
            ("globalIrradiation", 1), ("numberOfPeople", 1), ("heatGain", 1),
        ]
        if self.n_boundary_temperature == 1:
            u.append(("boundaryTemperature", 1))
        if self.n_walls > 0:
            u.append(("wallHeatGain", self.n_walls))
        return {"u": u, "y": {"indoorTemperature": 0, "wallTemperature": 1}}

    def _build_matrices(self, p=None):
        """Build the RC state-space matrices ``(A, B, C, D, E, F)`` from the
        physical parameters -- a **pure** function of ``p`` (no side effects
        beyond caching ``n_states`` / ``n_inputs``).

        ``p`` is a dict ``{name: value}`` of *physical* parameter tensors
        (:attr:`PARAM_NAMES`).  When ``None`` it defaults to the component's own
        values (``self.<name>.get()`` -- the ``do_step`` path).  Passing ``p``
        explicitly is the functorch fast path: because the parameters are plain
        tensor *arguments* (not ``tps.Parameter`` methods), ``jacrev`` w.r.t. ``p``
        is clean under ``vmap`` -- avoiding the Tensor-subclass fragility that
        ``functional_call`` on ``tps.Parameter`` would hit.  Shapes:
        ``A (n_c, n, n)``, ``B (n_c, n, m)``, ``C (n_c, n, n)``,
        ``D (n_c, n, m)``, ``E (n_c, m, n, n)``, ``F (n_c, m, n, m)``.
        """
        if p is None:
            p = {name: getattr(self, name).get() for name in self.PARAM_NAMES}

        # Calculate number of states
        n_states = 2  # Base states: air and wall temperature
        n_states += self.n_boundary_temperature  # Add boundary wall state
        self.n_states = n_states

        # Calculate number of inputs based on input dictionary
        n_inputs = len(self.input) - 2  # Base inputs from input dictionary
        n_inputs += self.n_walls  # Add one input for each connected wall
        n_inputs += (
            self.n_boundary_temperature
        )  # Add one input for boundary temperature
        self.n_inputs = n_inputs

        # Get parameter values - shape (n_c_param,); may be 1 even when
        # self.n_c > 1 (compiled/batched components share identical params).
        C_air = p["C_air"]
        C_wall = p["C_wall"]
        C_boundary = p["C_boundary"]
        R_in = p["R_in"]
        R_out = p["R_out"]
        R_boundary = p["R_boundary"]
        f_air = p["f_air"]
        f_wall = p["f_wall"]
        Q_occ_gain = p["Q_occ_gain"]
        n_c = self.n_c
        # Allocate on the parameters' device/dtype: _build_matrices re-runs on
        # cache miss during stepping, outside initialize()'s device context.
        dev, dt = C_air.device, C_air.dtype

        # Initialize A and B matrices with zeros - shape (n_c, n_states, n_states/n_inputs)
        A = torch.zeros((n_c, n_states, n_states), dtype=dt, device=dev)
        B = torch.zeros((n_c, n_states, n_inputs), dtype=dt, device=dev)

        # Air temperature equation coefficients
        A[:, 0, 0] = -1 / (R_in * C_air)
        A[:, 0, 1] = 1 / (R_in * C_air)  # T_wall coefficient

        if self.n_boundary_temperature == 1:
            # Add heat exchange with boundary wall
            A[:, 0, 0] -= 1 / (R_boundary * C_air)  # T_bound_wall coefficient
            A[:, 0, 2] = 1 / (R_boundary * C_air)  # T_bound_wall coefficient
            A[:, 2, 0] = 1 / (
                R_boundary * C_boundary
            )  # T_air coefficient for boundary wall
            A[:, 2, 2] = -2 / (R_boundary * C_boundary)  # T_bound_wall coefficient

        # Exterior wall temperature equation coefficients
        A[:, 1, 0] = 1 / (R_in * C_wall)  # T_air coefficient
        A[:, 1, 1] = -1 / (R_in * C_wall) - 1 / (R_out * C_wall)  # T_wall coefficient

        # Input matrix B coefficients - match the order in do_step
        # Outdoor temperature
        B[:, 1, 0] = 1 / (R_out * C_wall)  # T_out coefficient for wall

        # Solar radiation
        B[:, 0, 4] = f_air / C_air  # Radiation coefficient for air
        B[:, 1, 4] = f_wall / C_wall  # Radiation coefficient for wall

        # Number of people
        B[:, 0, 5] = Q_occ_gain / C_air  # N_people coefficient

        # Space heater heat input
        B[:, 0, 6] = 1 / C_air  # Q_sh coefficient

        if self.n_boundary_temperature == 1:
            # Boundary temperature
            B[:, 2, 7] = 1 / (R_boundary * C_boundary)  # T_bound coefficient

        # Wall heat gains (at the end of the input vector): heat flow [W]
        # produced by connected WallTorchSystem components enters the air node.
        for i in range(self.n_walls):
            wall_input_idx = (n_inputs - self.n_walls) + i
            B[:, 0, wall_input_idx] = 1 / C_air  # Q_wall coefficient

        # Output matrix C - Identity matrix for direct observation of all states
        # Shape: (n_c, n_states, n_states)
        C_out = (
            torch.eye(n_states, dtype=dt, device=dev)
            .unsqueeze(0)
            .expand(n_c, -1, -1)
            .clone()
        )

        # Feedthrough matrix D (no direct feedthrough) - Shape: (n_c, n_states, n_inputs)
        D = torch.zeros((n_c, n_states, n_inputs), dtype=dt, device=dev)

        # E matrix for input-state coupling: shape (n_c, n_inputs, n_states, n_states)
        E = torch.zeros((n_c, n_inputs, n_states, n_states), dtype=dt, device=dev)
        # -m_ex*cp*T_air (input 2, state 0)
        E[:, 2, 0, 0] = -constants.CP_AIR / C_air  # exhaustAirFlowRate * T_air

        # F matrix for input-input coupling: shape (n_c, n_inputs, n_states, n_inputs)
        F = torch.zeros((n_c, n_inputs, n_states, n_inputs), dtype=dt, device=dev)
        # m_sup*cp*T_sup (inputs 1 and 3)
        F[:, 1, 0, 3] = (
            constants.CP_AIR / C_air
        )  # supplyAirFlowRate * supplyAirTemperature

        return A, B, C_out, D, E, F

    def _create_state_space_model(self):
        """Create the internal :class:`DiscreteStatespaceSystem` used by
        ``do_step`` from the matrices built by :meth:`_build_matrices`."""
        A, B, C_out, D, E, F = self._build_matrices()

        # Initial state - shape (n_c, n_states)
        x0_tensor = self._get_initial_state_tensor()  # (n_s, n_c, n_states)
        x0 = x0_tensor[
            0, :, :
        ]  # Take first simulation, all components: (n_c, n_states)

        # Pass E and F to DiscreteStatespaceSystem
        self.ss_model = DiscreteStatespaceSystem(
            A=A,
            B=B,
            C=C_out,
            D=D,
            x0=x0,
            state_names=None,
            add_noise=False,
            id=f"ss_model_{self.id}",
            E=E,
            F=F,
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

    def do_step(
        self,
        second_time: Optional[float] = None,
        date_time: Optional[datetime.datetime] = None,
        step_size: Optional[float] = None,
        step_index: Optional[int] = None,
    ) -> None:
        """
        Perform one step of the RC model simulation.

        Args:
            second_time: Current simulation time in seconds.
            date_time: Current simulation date/time.
            step_size: Current simulation step size.

        Thin port-I/O wrapper around :meth:`forward` (the single source of
        truth for the dynamics); the inner ``ss_model`` only carries the
        state between steps.
        """
        inputs = {
            port: self.input[port].get()
            for port in (
                "outdoorTemperature", "supplyAirFlowRate", "exhaustAirFlowRate",
                "supplyAirTemperature", "globalIrradiation", "numberOfPeople",
                "heatGain",
            )
        }
        if self.n_boundary_temperature == 1:
            inputs["boundaryTemperature"] = self.input["boundaryTemperature"].get()
        if self.n_walls > 0:
            # Vector port: get() returns (n_s, n_c, n_v)
            inputs["wallHeatGain"] = self.input["wallHeatGain"].get()

        x = self.ss_model.get_state()  # (n_s, n_c, n_states)
        x_next, outs = self.forward(
            x, inputs, self._forward_params(), self._scalar_sample_time(step_size)
        )
        self.ss_model.set_state(x_next)
        self.output["indoorTemperature"]._set(outs["indoorTemperature"], i_t=step_index)
        self.output["wallTemperature"]._set(outs["wallTemperature"], i_t=step_index)

    def forward(self, x, inputs, params, sample_time):
        """Pure one-step dynamics: ``(state, inputs, params) -> (new_state, outputs)``.

        The functorch-compatible re-expression of :meth:`do_step` -- it rebuilds
        the RC matrices from ``params`` (:meth:`_build_matrices`) and takes one
        bilinear ZOH step, with **no ports/history/state mutation**.  ``params``
        being plain-tensor arguments (not ``tps.Parameter`` methods) is what makes
        ``vmap(jacrev(...))`` clean: it yields the per-segment collocation Jacobian
        blocks (``dx'/dx`` and ``dx'/dtheta``) in one shot.

        Args:
            x: state ``(n_c, n_states)`` = ``[T_indoor, T_wall,
                (T_boundary)]``.
            inputs: dict of resolved input-port values (each ``(n_c,)`` scalar, or
                ``(n_c, n_v)`` for ``wallHeatGain``).  Assembled here
                into the ``do_step`` input order.
            params: dict of *physical* parameter values (:attr:`PARAM_NAMES`).
            sample_time: step size in seconds.

        Returns:
            ``(x_next (n_c, n_states), {"indoorTemperature", "wallTemperature"})``.
        """
        # The matrices depend only on params, not on (x, u): cache them per
        # params-dict identity so a sequential rollout builds them once per
        # theta instead of once per step (see OneStepComposer._params_for /
        # System._forward_params).  sample_time is part of the key because the
        # attached disc_cache holds (Ad, Bd) discretized at a specific T.
        cache = getattr(self, "_fwd_mat_cache", None)
        if cache is None or cache[0] is not params or cache[2] != sample_time:
            cache = (params, self._build_matrices(params), sample_time, {})
            self._fwd_mat_cache = cache
        A, B, C, D, E, F = cache[1]
        cols = [
            inputs["outdoorTemperature"], inputs["supplyAirFlowRate"],
            inputs["exhaustAirFlowRate"], inputs["supplyAirTemperature"],
            inputs["globalIrradiation"], inputs["numberOfPeople"], inputs["heatGain"],
        ]
        if self.n_boundary_temperature == 1:
            cols.append(inputs["boundaryTemperature"])
        u = torch.stack(cols, dim=-1)  # (n_c, n_base_inputs)
        if self.n_walls > 0:
            u = torch.cat([u, inputs["wallHeatGain"]], dim=-1)
        x_next, y = bilinear_onestep(
            A, B, C, D, E, F, x, u, sample_time, disc_cache=cache[3]
        )
        return x_next, {"indoorTemperature": y[..., 0], "wallTemperature": y[..., 1]}


def brick_signature_pattern():
    """
    Get the BRICK-only signature pattern of the building space component.

    Returns:
        SignaturePattern: The BRICK-only signature pattern of the building space component.
    """

    node0 = Node(cls=core.namespace.BRICK.AHU)
    node2 = Node(cls=core.namespace.BRICK.HVAC_Zone)  # building space/room
    node3 = Node(cls=core.namespace.BRICK.Room)
    node4 = Node(cls=core.namespace.BRICK.Air_Temperature_Sensor)
    node6 = Node(
        cls=core.namespace.BRICK.Outside_Air_Temperature_Sensor
    )  # outdoor temperature sensor

    sp = SignaturePattern(
        id="building_space_signature_pattern_brick",
    )

    sp.add_rule(
        StepRule(subject=node0, object=node2, predicate=core.namespace.BRICK.feeds)
    )
    # sp.add_rule(StepRule(subject=node1, object=node2, predicate=core.namespace.BRICK.isFedBy))
    sp.add_rule(
        StepRule(subject=node2, object=node3, predicate=core.namespace.BRICK.hasPart)
    )
    sp.add_rule(
        StepRule(subject=node4, object=node3, predicate=core.namespace.BRICK.isPointOf)
    )
    # sp.add_rule(AnyPathRule(subject=node9, object=node2, predicate=core.namespace.BRICK.isAdjacentTo)) # TODO: Makes _prune_recursive fail, infinite recursion

    # Optional
    # heatGain
    # numberOfPeople

    sp.add_input("supplyAirFlowRate", node0, "airFlowRate")
    sp.add_input("exhaustAirFlowRate", node0, "airsFlowRate")
    # sp.add_input("numberOfPeople", node5, "measuredValue")
    sp.add_input("outdoorTemperature", node6, "measuredValue")
    # sp.add_input("outdoorCO2", node6, "outdoorCo2Concentration")
    # sp.add_input("globalIrradiation", node6, "globalIrradiation")
    sp.add_input("supplyAirTemperature", node0)

    sp.add_modeled_node(node3)
    return sp


BuildingSpaceThermalTorchSystem.add_signature_pattern(brick_signature_pattern())
