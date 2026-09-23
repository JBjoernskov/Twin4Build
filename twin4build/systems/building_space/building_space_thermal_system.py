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
from twin4build.systems.building_space import air_balance
import twin4build.utils.types as tps
from twin4build.utils.slots import wired_width
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


class BuildingSpaceThermalSystem(core.System, nn.Module):
    r"""
    Building Space Thermal Model using RC Network Dynamics.

    This class implements a thermal model for building spaces using a network of thermal 
    resistances and capacitances (RC network). The model represents heat transfer between 
    indoor air, exterior walls and (optionally) a boundary wall using bilinear 
    state-space dynamics. Heat exchange with neighbouring zones or other boundary
    temperatures is modeled by connecting one or more
    :class:`~twin4build.systems.wall.wall_system.WallSystem` components
    to the ``wallHeatGain`` vector input port.

    Args:
        C_air: Thermal capacitance of indoor air [J/K]
        C_wall: Thermal capacitance of exterior wall [J/K]
        C_boundary: Thermal capacitance of boundary wall [J/K] (deprecated, use WallSystem)
        R_out: Thermal resistance between wall and outdoor [K/W]
        R_in: Thermal resistance between wall and indoor [K/W]
        R_boundary: Thermal resistance of boundary [K/W] (deprecated, use WallSystem)
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
         optional; produced by a ``WallSystem``, which owns the wall state
         so the interzonal energy balance holds by construction)
       - :math:`\dot{m}_{sup}`: Supply air flow rate [kg/s] (input)
       - :math:`\dot{m}_{exh}`: Exhaust air flow rate [kg/s] (input); enters
         the dynamics only through the outdoor **make-up flow**
         :math:`\dot{m}_{mu} = \max(\dot{m}_{exh} - \dot{m}_{sup}, 0)`,
         see below
       - :math:`\Phi_{sol}`: Solar radiation [W/m²] (input)
       - :math:`N_{occ}`: Number of occupants (input)
       - :math:`Q_{sh}`: Space heater heat input [W] (input)

    **State-Space Representation:**

    The system is implemented using the DiscreteStatespaceSystem with matrices:

    *State vector:* :math:`\mathbf{x} = \begin{bmatrix}T_i \\ T_w \\ T_{bw}\end{bmatrix}`

    *Input vector:* :math:`\mathbf{u} = \begin{bmatrix}T_o \\ \dot{m}_{sup} \\ \dot{m}_{mu} \\ T_{sup} \\ \Phi_{sol} \\ N_{occ} \\ Q_{sh} \\ T_{bound} \\ \dot{Q}_{wall,1} \\ \vdots \\ \dot{Q}_{wall,n}\end{bmatrix}`

    The third slot carries the make-up flow
    :math:`\dot{m}_{mu} = \max(\dot{m}_{exh} - \dot{m}_{sup}, 0)`, not the
    raw exhaust flow: the ``exhaustAirFlowRate`` port value is transformed at
    input assembly (:func:`~twin4build.systems.building_space.air_balance.balanced_flow_inputs`).

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

    Every air stream entering the room is balanced by room air leaving at
    :math:`T_i` (the room air mass is constant), so each flow input removes
    heat at the indoor temperature:

    .. math::

       \mathbf{E}[1, 0, 0] = -\frac{c_p}{C_{air}} \quad \text{(supply flow} \cdot T_i\text{)}, \qquad
       \mathbf{E}[2, 0, 0] = -\frac{c_p}{C_{air}} \quad \text{(make-up flow} \cdot T_i\text{)}

    *Input-Input Coupling (F matrices):*

    The supply flow brings heat at the supply air temperature, the make-up
    flow at the outdoor temperature:

    .. math::

       \mathbf{F}[1, 0, 3] = \frac{c_p}{C_{air}} \quad \text{(supply flow} \cdot T_{sup}\text{)}, \qquad
       \mathbf{F}[2, 0, 0] = \frac{c_p}{C_{air}} \quad \text{(make-up flow} \cdot T_o\text{)}

    Input vector mapping: :math:`[T_o, \dot{m}_{sup}, \dot{m}_{mu}, T_{sup}, \Phi_{sol}, N_{occ}, Q_{sh}, T_{bound}, \dot{Q}_{wall,1}]^T`

    *Bilinear Effects*

    Together the bilinear terms give the balanced ventilation heat flow

    .. math::

       \dot{Q}_{vent} = \dot{m}_{sup} c_p (T_{sup} - T_i) + \dot{m}_{mu} c_p (T_o - T_i)

    With :math:`\dot{m}_{sup} \ge \dot{m}_{exh}` the surplus supply leaves
    through the envelope at room state and the exhaust flow drops out; with
    :math:`\dot{m}_{exh} > \dot{m}_{sup}` the deficit is made up by outdoor
    air drawn through the envelope.  No fictitious
    :math:`(\dot{m}_{sup} - \dot{m}_{exh}) c_p T_i` storage term survives an
    imbalance between the two (mis)measured flows.  See
    :mod:`twin4build.systems.building_space.air_balance`.

    Physical Interpretation
    -----------------------

    **Thermal Network:**
       - RC network represents building thermal mass and resistances
       - States capture temperature of air, walls, and structural elements
       - Inputs represent weather, HVAC, occupancy, and heat sources
       - Bilinear terms model flow-dependent heat transfer accurately

    **Interzonal Heat Transfer:**
       - Partition walls between zones are modeled by a separate
         ``WallSystem``: the zone sends its ``indoorTemperature`` to the
         wall and receives the wall's heat flow on ``wallHeatGain``
       - Because a single wall component owns the wall state, the heat leaving
         one zone equals the heat stored in the wall plus the heat entering the
         other zone (energy-consistent by construction)

    **Flow-Dependent Effects:**
       - Supply air flow brings heat at supply temperature and displaces room
         air at indoor temperature (F and E matrix coupling)
       - Exhaust in excess of the supply draws outdoor air through the
         envelope (make-up flow, F and E coupling on the third input slot)
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
    >>> thermal_model = tb.BuildingSpaceThermalSystem(
    ...     C_air=2e6,      # Higher air thermal mass
    ...     C_wall=5e6,     # Wall thermal mass
    ...     R_out=0.1,      # Outdoor thermal resistance
    ...     R_in=0.05,      # Indoor thermal resistance
    ...     f_air=0.15,     # Air radiation factor
    ...     id="zone_1_thermal"
    ... )

    Zone coupled to a neighbour zone through a wall component:

    >>> wall = tb.WallSystem(C=2e5, R_a=0.05, R_b=0.05, id="wall_AB")
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
                (deprecated -- connect a ``WallSystem`` instead)
            R_out: Thermal resistance between exterior wall and outdoor [K/W]
            R_in: Thermal resistance between exterior wall and indoor [K/W]
            R_boundary: Thermal resistance of boundary [K/W]
                (deprecated -- connect a ``WallSystem`` instead)
            f_wall: Radiation factor for exterior wall
            f_air: Radiation factor for air/internal mass
            Q_occ_gain: Heat gain per occupant [W]
            **kwargs: Additional keyword arguments passed to parent
        """
        super().__init__(**kwargs)
        nn.Module.__init__(self)

        # Store thermal parameters as tps.Parameters
        self.C_air = tps.Parameter(
            torch.tensor(C_air, dtype=tps.float_dtype()),
            requires_grad=False,
            scaling="log",
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
            torch.tensor(R_out, dtype=tps.float_dtype()),
            requires_grad=False,
            scaling="log",
        )
        self.R_in = tps.Parameter(
            torch.tensor(R_in, dtype=tps.float_dtype()),
            requires_grad=False,
            scaling="log",
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
            # Temperature of the make-up air the exhaust deficit draws in.
            # Unwired: outdoor air (the classic balance).  Wired from a
            # transfer node: the corridor's temperature.
            "makeUpAirTemperature": tps.Scalar(0.0, optional=True),
            "boundaryTemperature": tps.Scalar(
                21, optional=True
            ),  # Boundary temperature [°C], optional (deprecated: use WallSystem)
            "wallHeatGain": tps.Vector(
                optional=True
            ),  # Heat flow from connected WallSystem components [W], optional
        }

        # Define outputs
        self.output = {
            "indoorTemperature": tps.Scalar(20),  # Indoor air temperature [°C]
            "wallTemperature": tps.Scalar(20),  # Exterior wall temperature [°C]
        }

        # Define parameters for calibration
        self.parameter = {
            # Effective air-node capacity: air plus furniture / light
            # internal mass, which is why 1e6 J/K binds on classrooms.
            "C_air": {"lb": 1000.0, "ub": 3000000.0},
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
        self._make_up_wired = False
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

        if hasattr(self, "_n_c_batched") and self._n_c_batched > 1:
            self.n_c = self._n_c_batched
        else:
            self.n_c = 1
        # Structural: does a producer feed makeUpAirTemperature?  Fixed per model.
        self._make_up_wired = any(
            cp.input_port == "makeUpAirTemperature" and len(cp.connects_system_through) > 0
            for cp in self.connects_at
        )

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

    def _has_boundary_temperature(self) -> bool:
        """Whether the (deprecated) in-zone boundary-wall path is in use:
        set up manually, or a connection on ``boundaryTemperature``."""
        if self.manual_setup_n_boundary_temperature:
            return self.n_boundary_temperature == 1
        return any(
            cp.input_port == "boundaryTemperature" and cp.connects_system_through
            for cp in self.connects_at
        )

    def _inactive_parameters(self):
        """Parameters the wiring leaves without effect (skipped by
        :meth:`get_estimable_parameters`): ``C_boundary`` / ``R_boundary``
        unless a boundary temperature is connected."""
        if self._has_boundary_temperature():
            return ()
        return ("C_boundary", "R_boundary")

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
                "C_boundary) is deprecated. Connect a WallSystem to the "
                "wallHeatGain port instead.",
                DeprecationWarning,
                stacklevel=2,
            )

        if self.manual_setup_n_walls == False:
            # Count logical vector slots, not connection objects. A compiled
            # meta-component can have several connections targeting the same
            # slot, each covering a different subset of its n_c branches.
            self.n_walls = wired_width(self, "wallHeatGain")

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
    SUPPORTS_TRANSFORM_MODE = True
    PARAM_NAMES = (
        "C_air",
        "C_wall",
        "C_boundary",
        "R_in",
        "R_out",
        "R_boundary",
        "f_air",
        "f_wall",
        "Q_occ_gain",
    )

    #: Fusable coupling ports (see FusedStateSpaceSystem): connected
    #: WallSystem heat flows enter the linear B matrix, and
    #: ``indoorTemperature`` is a pure state observation.
    FUSABLE_INPUT_PORTS = frozenset({"wallHeatGain", "heatGain"})

    FUSABLE_OUTPUT_PORTS = frozenset({"indoorTemperature"})

    #: Base slot of ``makeUpAirTemperature`` in ``u`` (after ``heatGain``).
    MAKE_UP_TEMPERATURE_SLOT = 7
    #: Slot of ``boundaryTemperature`` when present (after the base slots).
    BOUNDARY_SLOT = 8

    def _ss_layout(self):
        """Port <-> matrix index map, mirroring :meth:`forward` exactly.

        ``u = [outdoorTemperature, supplyAirFlowRate, exhaustAirFlowRate,
        supplyAirTemperature, globalIrradiation, numberOfPeople, heatGain,
        (boundaryTemperature,) wallHeatGain x n_walls]`` (the exhaust slot
        holds the make-up flow after :meth:`_ss_transform_inputs`); output rows are the
        observed states.  Valid after :meth:`initialize` (needs ``n_walls`` /
        ``n_boundary_temperature``).
        """
        u = [
            ("outdoorTemperature", 1),
            ("supplyAirFlowRate", 1),
            ("exhaustAirFlowRate", 1),
            ("supplyAirTemperature", 1),
            ("globalIrradiation", 1),
            ("numberOfPeople", 1),
            ("heatGain", 1),
            ("makeUpAirTemperature", 1),
        ]
        if self.n_boundary_temperature == 1:
            u.append(("boundaryTemperature", 1))
        if self.n_walls > 0:
            u.append(("wallHeatGain", self.n_walls))
        return {"u": u, "y": {"indoorTemperature": 0, "wallTemperature": 1}}

    def _ss_support(self):
        """Conservative structural support of the ``D``, ``E`` and ``F`` matrices.

        Entries are matrix-index tuples that may be nonzero for any admissible
        parameter value.  Fusion uses this static contract for control flow; it
        must therefore remain a superset even when a coefficient is zero at a
        particular parameter iterate.
        """
        return {
            "D": frozenset(),
            "E": frozenset({(1, 0, 0), (2, 0, 0)}),
            # Superset: the make-up stream brings heat at slot 0 (outdoor) or
            # at the makeUpAirTemperature slot (a wired transfer node).
            "F": frozenset({(1, 0, 3), (2, 0, 0), (2, 0, self.MAKE_UP_TEMPERATURE_SLOT)}),
        }

    #: Input ports whose values :meth:`_ss_transform_inputs` reads.
    SS_TRANSFORM_PORTS = air_balance.TRANSFORM_PORTS

    @staticmethod
    def _ss_transform_inputs(inputs):
        """Balanced-ventilation input transform (see
        :mod:`~twin4build.systems.building_space.air_balance`): the
        ``exhaustAirFlowRate`` slot carries the outdoor make-up flow
        ``max(m_exh - m_sup, 0)``.  Pure function of the original inputs;
        applied by :meth:`forward` and by the fused block."""
        return air_balance.balanced_flow_inputs(inputs)

    def _build_matrices(self, p=None):
        """Build the RC state-space matrices ``(A, B, C, D, E, F)`` from the
        physical parameters -- a **pure** function of ``p``.

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

        # Calculate number of inputs based on input dictionary
        n_inputs = len(self.input) - 2  # Base inputs from input dictionary
        n_inputs += self.n_walls  # Add one input for each connected wall
        n_inputs += (
            self.n_boundary_temperature
        )  # Add one input for boundary temperature

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

        zero = torch.zeros_like(C_air)
        air_wall = 1 / (R_in * C_air)
        wall_air = 1 / (R_in * C_wall)
        wall_outdoor = 1 / (R_out * C_wall)
        if self.n_boundary_temperature == 1:
            air_boundary = 1 / (R_boundary * C_air)
            boundary_air = 1 / (R_boundary * C_boundary)
            A = torch.stack(
                [
                    torch.stack(
                        [-air_wall - air_boundary, air_wall, air_boundary],
                        dim=-1,
                    ),
                    torch.stack([wall_air, -wall_air - wall_outdoor, zero], dim=-1),
                    torch.stack([boundary_air, zero, -2 * boundary_air], dim=-1),
                ],
                dim=1,
            )
        else:
            A = torch.stack(
                [
                    torch.stack([-air_wall, air_wall], dim=-1),
                    torch.stack([wall_air, -wall_air - wall_outdoor], dim=-1),
                ],
                dim=1,
            )

        # Base slots: [T_out, m_sup, m_mu, T_sup, irradiation, N, Q_heat, T_mu]
        air_inputs = [
            zero,
            zero,
            zero,
            zero,
            f_air / C_air,
            Q_occ_gain / C_air,
            1 / C_air,
            zero,
        ]
        wall_inputs = [
            wall_outdoor,
            zero,
            zero,
            zero,
            f_wall / C_wall,
            zero,
            zero,
            zero,
        ]
        if self.n_boundary_temperature == 1:
            air_inputs.append(zero)
            wall_inputs.append(zero)
        air_inputs.extend([1 / C_air] * self.n_walls)
        wall_inputs.extend([zero] * self.n_walls)
        b_rows = [
            torch.stack(air_inputs, dim=-1),
            torch.stack(wall_inputs, dim=-1),
        ]
        if self.n_boundary_temperature == 1:
            boundary_inputs = [zero] * n_inputs
            boundary_inputs[self.BOUNDARY_SLOT] = boundary_air
            b_rows.append(torch.stack(boundary_inputs, dim=-1))
        B = torch.stack(b_rows, dim=1)

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

        # Balanced ventilation (see air_balance.py): slot 1 is the supply
        # flow, slot 2 the outdoor make-up flow max(m_exh - m_sup, 0).  Each
        # entering stream displaces room air at T_i (E) and brings heat at
        # its own temperature (F): supply at T_sup (slot 3), make-up at T_o
        # (slot 0).
        input_basis = torch.eye(n_inputs, dtype=dt, device=dev)
        state_basis = torch.eye(n_states, dtype=dt, device=dev)
        u_outdoor_temperature = input_basis[0]
        u_supply_flow = input_basis[1]
        u_make_up_flow = input_basis[2]
        u_supply_temperature = input_basis[3]
        # The make-up stream's temperature: outdoor unless a transfer node
        # feeds makeUpAirTemperature (structural, fixed at initialize).
        u_make_up_temperature = (
            input_basis[self.MAKE_UP_TEMPERATURE_SLOT] if self._make_up_wired else u_outdoor_temperature
        )
        state_air = state_basis[0]
        gain = (constants.CP_AIR / C_air).reshape(n_c, 1, 1, 1)

        # E matrix for input-state coupling: shape (n_c, n_inputs, n_states, n_states)
        E = (
            -gain
            * (u_supply_flow + u_make_up_flow).reshape(1, n_inputs, 1, 1)
            * state_air.reshape(1, 1, n_states, 1)
            * state_air.reshape(1, 1, 1, n_states)
        )

        # F matrix for input-input coupling: shape (n_c, n_inputs, n_states, n_inputs)
        F = gain * state_air.reshape(1, 1, n_states, 1) * (
            u_supply_flow.reshape(1, n_inputs, 1, 1)
            * u_supply_temperature.reshape(1, 1, 1, n_inputs)
            + u_make_up_flow.reshape(1, n_inputs, 1, 1)
            * u_make_up_temperature.reshape(1, 1, 1, n_inputs)
        )

        return A, B, C_out, D, E, F

    def _create_state_space_model(self):
        """Create the internal :class:`DiscreteStatespaceSystem` used by
        ``do_step`` from the matrices built by :meth:`_build_matrices`."""
        A, B, C_out, D, E, F = self._build_matrices()
        self.n_states = A.shape[-1]
        self.n_inputs = B.shape[-1]

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
                "outdoorTemperature",
                "supplyAirFlowRate",
                "exhaustAirFlowRate",
                "supplyAirTemperature",
                "globalIrradiation",
                "numberOfPeople",
                "heatGain",
                "makeUpAirTemperature",
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

    def forward(self, x, inputs, params, sample_time, transform_mode=None):
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
        if transform_mode:
            matrices = self._build_matrices(params)
            disc_cache = None
        else:
            cache = getattr(self, "_fwd_mat_cache", None)
            if cache is None or cache[0] is not params or cache[2] != sample_time:
                cache = (params, self._build_matrices(params), sample_time, {})
                self._fwd_mat_cache = cache
            matrices = cache[1]
            disc_cache = cache[3]
        A, B, C, D, E, F = matrices
        inputs = {**inputs, **self._ss_transform_inputs(inputs)}
        cols = [
            inputs["outdoorTemperature"],
            inputs["supplyAirFlowRate"],
            inputs["exhaustAirFlowRate"],
            inputs["supplyAirTemperature"],
            inputs["globalIrradiation"],
            inputs["numberOfPeople"],
            inputs["heatGain"],
            inputs.get("makeUpAirTemperature", inputs["outdoorTemperature"]),
        ]
        if self.n_boundary_temperature == 1:
            cols.append(inputs["boundaryTemperature"])
        u = torch.stack(cols, dim=-1)  # (n_c, n_base_inputs)
        if self.n_walls > 0:
            u = torch.cat([u, inputs["wallHeatGain"]], dim=-1)
        x_next, y = bilinear_onestep(
            A,
            B,
            C,
            D,
            E,
            F,
            x,
            u,
            sample_time,
            disc_cache=disc_cache,
            transform_mode=transform_mode,
        )
        return x_next, {"indoorTemperature": y[..., 0], "wallTemperature": y[..., 1]}


# Deprecated aliases (removed in twin4build 2.1)
BuildingSpaceThermalTorchSystem = BuildingSpaceThermalSystem
