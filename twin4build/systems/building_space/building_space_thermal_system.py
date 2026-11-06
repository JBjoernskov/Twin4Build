# Third party imports
import torch

# Local application imports
import twin4build.core as core
import twin4build.utils.constants as constants
from twin4build.systems.building_space import air_balance
from twin4build.systems.saref4syst.declarations import Input, Output, Parameter, State
from twin4build.systems.utils.discrete_statespace_system import bilinear_onestep

#: What the in-zone boundary wall's names say now (removed in 2.0).
BOUNDARY_WALL_GONE = "the in-zone boundary wall is gone: connect a WallSystem to wallHeatGain"


class BuildingSpaceThermalSystem(core.System):
    r"""
    Building Space Thermal Model using RC Network Dynamics.

    This class implements a thermal model for building spaces using a network of thermal
    resistances and capacitances (RC network). The model represents heat transfer between
    indoor air and exterior walls using bilinear state-space dynamics. Heat
    exchange with neighbouring zones or other boundary temperatures is
    modeled by connecting one or more
    :class:`~twin4build.systems.wall.wall_system.WallSystem` components
    to the ``wallHeatGain`` vector input port.

    Args:
        C_air: Thermal capacitance of indoor air [J/K]
        C_wall: Thermal capacitance of exterior wall [J/K]
        R_out: Thermal resistance between wall and outdoor [K/W]
        R_in: Thermal resistance between wall and indoor [K/W]
        f_wall: Radiation factor for exterior wall
        f_air: Radiation factor for air
        Q_occ_gain: Heat gain per occupant [W]
        m_inf: Infiltration air flow [kg/s] (in a :class:`BuildingSpaceSystem`
            the CO2 model's ``mass.m_inf``: one flow for heat and CO2)

    Mathematical Formulation
    ------------------------

    **Continuous-Time Differential Equations:**

    The thermal dynamics are governed by energy balance equations for each thermal node:

    *1. Indoor Air Temperature:*

    .. math::

       C_{air}\frac{dT_i}{dt} = \frac{T_w - T_i}{R_{in}} + \sum_{j}\dot{Q}_{wall,j} + Q_{occ} N_{occ} + Q_{sh} + f_{air}\Phi_{sol} + c_p\dot{m}_{sup}(T_{sup} - T_i) + c_p\dot{m}_{mu}(T_{mu} - T_i) + c_p\dot{m}_{inf}(T_o - T_i)

    The last term is the infiltration :math:`\dot{m}_{inf}`: outdoor air
    through the envelope, balanced by room air leaving, the same flow the CO2
    model carries.  :math:`T_{mu}` is the make-up air's temperature: outdoor,
    unless ``makeUpAirTemperature`` is wired (a neighbouring zone's).

    *2. Exterior Wall Temperature:*

       .. math::

          C_{wall}\frac{dT_w}{dt} = \frac{T_o - T_w}{R_{out}} + \frac{T_i - T_w}{R_{in}} + f_{wall}\Phi_{sol}

    where:

       - :math:`T_i`: Indoor air temperature [°C] (state)
       - :math:`T_w`: Exterior wall temperature [°C] (state)
       - :math:`T_o`: Outdoor temperature [°C] (input)
       - :math:`T_{sup}`: Supply air temperature [°C] (input)
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

    *State vector:* :math:`\mathbf{x} = \begin{bmatrix}T_i \\ T_w\end{bmatrix}`

    *Input vector:* :math:`\mathbf{u} = \begin{bmatrix}T_o \\ \dot{m}_{sup} \\ \dot{m}_{mu} \\ T_{sup} \\ \Phi_{sol} \\ N_{occ} \\ Q_{sh} \\ T_{mu} \\ \dot{Q}_{wall,1} \\ \vdots \\ \dot{Q}_{wall,n} \\ \dot{m}_{inf}\end{bmatrix}`

    The third slot carries the make-up flow
    :math:`\dot{m}_{mu} = \max(\dot{m}_{exh} - \dot{m}_{sup}, 0)`, not the
    raw exhaust flow: the ``exhaustAirFlowRate`` port value is transformed at
    input assembly (:func:`~twin4build.systems.building_space.air_balance.balanced_flow_inputs`).

    *Base System Matrices:*

    For one connected wall (the infiltration conductance :math:`G_{inf} = c_p
    \dot{m}_{inf} / C_{air}` adds :math:`-G_{inf}` to :math:`A_{00}` and
    :math:`G_{inf}` to :math:`B_{00}`, left out below):

    .. math::

       \mathbf{A} = \begin{bmatrix}
       -\frac{1}{R_{in}C_{air}} & \frac{1}{R_{in}C_{air}} \\
       \frac{1}{R_{in}C_{wall}} & -\frac{1}{R_{in}C_{wall}} - \frac{1}{R_{out}C_{wall}}
       \end{bmatrix}

       \mathbf{B} = \begin{bmatrix}
       0 & 0 & 0 & 0 & \frac{f_{air}}{C_{air}} & \frac{Q_{occ}}{C_{air}} & \frac{1}{C_{air}} & 0 & \frac{1}{C_{air}} & 0 \\
       \frac{1}{R_{out}C_{wall}} & 0 & 0 & 0 & \frac{f_{wall}}{C_{wall}} & 0 & 0 & 0 & 0 & 0
       \end{bmatrix}

       \mathbf{C} = \begin{bmatrix}
       1 & 0 \\
       0 & 1
       \end{bmatrix}

       \mathbf{D} = \begin{bmatrix}
       0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
       0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0
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
    flow at the make-up air temperature (outdoor unless wired):

    .. math::

       \mathbf{F}[1, 0, 3] = \frac{c_p}{C_{air}} \quad \text{(supply flow} \cdot T_{sup}\text{)}, \qquad
       \mathbf{F}[2, 0, 7] = \frac{c_p}{C_{air}} \quad \text{(make-up flow} \cdot T_{mu}\text{)}

    Input vector mapping: :math:`[T_o, \dot{m}_{sup}, \dot{m}_{mu}, T_{sup}, \Phi_{sol}, N_{occ}, Q_{sh}, T_{mu}, \dot{Q}_{wall,1}, \dot{m}_{inf}]^T`

    *Bilinear Effects*

    Together the bilinear terms give the balanced ventilation heat flow

    .. math::

       \dot{Q}_{vent} = \dot{m}_{sup} c_p (T_{sup} - T_i) + \dot{m}_{mu} c_p (T_{mu} - T_i)

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

    REMOVED_ATTRIBUTES = {
        name: BOUNDARY_WALL_GONE
        for name in ("C_boundary", "R_boundary", "boundaryTemperature", "n_boundary_temperature",
                     "manual_setup_n_boundary_temperature")
    }

    outdoorTemperature = Input(doc="Outdoor temperature [degC]")
    supplyAirFlowRate = Input(doc="Supply air flow rate [kg/s]")
    exhaustAirFlowRate = Input(doc="Exhaust air flow rate [kg/s]")
    supplyAirTemperature = Input(doc="Supply air temperature [degC]")
    globalIrradiation = Input(doc="Solar irradiation [W/m2]")
    numberOfPeople = Input(doc="Number of occupants")
    heatGain = Input(doc="Space heater heat input [W]")
    makeUpAirTemperature = Input(
        default=outdoorTemperature, doc="Temperature of the make-up air [degC]; unwired: outdoor air"
    )
    wallHeatGain = Input(size="wired", doc="Heat flows from connected WallSystem components [W], one slot per wall")
    infiltrationAirFlowRate = Input(default=0.0, doc="Envelope inflow on top of the make-up flow [kg/s]")
    indoorTemperature = Output(init=20.0, doc="Indoor air temperature [degC]")
    wallTemperature = Output(init=20.0, doc="Exterior wall temperature [degC]")
    surplusAirFlowRate = Output(doc="Supply in excess of the exhaust [kg/s]")
    makeUpAirFlowRate = Output(doc="Exhaust in excess of the supply, made up through the envelope [kg/s]")
    temperatures = State(size=2, init=20.0, names=["T_air", "T_wall"], doc="Indoor air and wall temperatures [degC]")
    C_air = Parameter(1e6, lb=1000.0, ub=3e6, relative=(0.1, 10.0), scaling="log", doc="Indoor air capacitance [J/K]")
    C_wall = Parameter(1e6, lb=1e4, ub=1e7, relative=(0.1, 10.0), scaling="log", doc="Exterior wall capacitance [J/K]")
    R_in = Parameter(0.05, lb=0.001, ub=1.0, relative=(0.1, 10.0), scaling="log",
                     doc="Resistance between the wall and the indoor air [K/W]")
    R_out = Parameter(0.05, lb=0.001, ub=1.0, relative=(0.1, 10.0), scaling="log",
                      doc="Resistance between the wall and the outdoor air [K/W]")
    f_air = Parameter(0.1, lb=0.0, ub=1.0, doc="Share of the solar gain on the air [-]")
    f_wall = Parameter(0.3, lb=0.0, ub=1.0, doc="Share of the solar gain on the wall [-]")
    Q_occ_gain = Parameter(100.0, lb=50.0, ub=200.0, doc="Heat gain per occupant [W]")
    m_inf = Parameter(0.0, lb=0.0, ub=0.01, doc="Infiltration air flow [kg/s] (in a room: mass.m_inf)")

    @property
    def n_walls(self) -> int:
        """The connected walls: the slots of ``wallHeatGain``."""
        return int(self.input["wallHeatGain"]._n_v or 0)

    @n_walls.setter
    def n_walls(self, n: int) -> None:
        # The width of a model stepped on its own (wiring gives it otherwise).
        self.input["wallHeatGain"]._n_v = int(n)

    def _state_heat_capacities(self):
        """``[T_air, T_wall]`` hold ``C_air`` and ``C_wall`` [J/K]."""
        caps = [self.C_air.get(), self.C_wall.get()]
        caps = [c.reshape(-1) for c in caps]
        n = max(int(c.numel()) for c in caps)
        return torch.stack([c.expand(n) for c in caps], dim=-1)

    SS_DERIVED_OUTPUT_PORTS = ("surplusAirFlowRate", "makeUpAirFlowRate")
    SS_DERIVED_INPUT_PORTS = ("supplyAirFlowRate", "exhaustAirFlowRate")
    SS_DERIVED_ROW_PORTS = ()

    #: The imbalance flows depend on the flow ports alone, not on the state or a parameter
    #: (:mod:`twin4build.model.signal_dependencies`); the temperatures are undeclared (everything).
    OUTPUT_DEPENDENCIES = {
        "surplusAirFlowRate": {"inputs": ("supplyAirFlowRate", "exhaustAirFlowRate")},
        "makeUpAirFlowRate": {"inputs": ("supplyAirFlowRate", "exhaustAirFlowRate")},
    }
    #: The class that computes those state-free outputs on its own, from the same ports: a world that needs them
    #: without the room gets it in the room's place (:mod:`twin4build.model.signal_dependencies`).
    OUTPUT_PROJECTIONS = {"surplusAirFlowRate": "AirImbalanceSystem", "makeUpAirFlowRate": "AirImbalanceSystem"}

    @staticmethod
    def _ss_derived_outputs(inputs, rows):
        """The air-flow imbalance of the step from the supply and exhaust
        flows (the ports' values, before the make-up transform): the surplus
        and the make-up flow
        (:mod:`~twin4build.systems.building_space.air_balance`)."""
        surplus, make_up = air_balance.imbalance_flows(inputs["supplyAirFlowRate"], inputs["exhaustAirFlowRate"])
        return {"surplusAirFlowRate": surplus, "makeUpAirFlowRate": make_up}

    #: Fusable coupling ports (see FusedStateSpaceSystem): connected
    #: WallSystem heat flows enter the linear B matrix, and
    #: ``indoorTemperature`` is a pure state observation.
    FUSABLE_INPUT_PORTS = frozenset({"wallHeatGain", "heatGain"})

    FUSABLE_OUTPUT_PORTS = frozenset({"indoorTemperature"})
    #: ``indoorTemperature`` publishes the air state as it is (see
    #: :meth:`System.state_output_ports`).
    STATE_OUTPUT_PORTS = {"indoorTemperature": 0}

    #: Base slot of ``makeUpAirTemperature`` in ``u`` (after ``heatGain``).
    MAKE_UP_TEMPERATURE_SLOT = 7

    def _ss_layout(self):
        """Port <-> matrix index map, mirroring :meth:`forward` exactly.

        ``u = [outdoorTemperature, supplyAirFlowRate, exhaustAirFlowRate,
        supplyAirTemperature, globalIrradiation, numberOfPeople, heatGain,
        makeUpAirTemperature, wallHeatGain x n_walls, infiltrationAirFlowRate]``
        (the exhaust slot holds the make-up flow after
        :meth:`_ss_transform_inputs`); output rows are the observed states.
        Valid once the ports are sized (needs ``n_walls``).
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
        if self.n_walls > 0:
            u.append(("wallHeatGain", self.n_walls))
        u.append((air_balance.INFILTRATION_PORT, 1))  # last: it acts through the transform only
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
    SS_TRANSFORM_PORTS = air_balance.TRANSFORM_PORTS + (
        air_balance.INFILTRATION_PORT, "outdoorTemperature", "makeUpAirTemperature"
    )

    @staticmethod
    def _ss_transform_inputs(inputs):
        """Balanced-ventilation input transform (see
        :mod:`~twin4build.systems.building_space.air_balance`): the
        ``exhaustAirFlowRate`` slot carries the outdoor make-up flow
        ``max(m_exh - m_sup, 0)``, with an envelope inflow the deficit it
        leaves plus the inflow, at their mixed temperature on the
        ``makeUpAirTemperature`` slot.  Pure function of the original inputs;
        applied by :meth:`forward` and by the fused block."""
        return air_balance.balanced_flow_inputs(inputs, "makeUpAirTemperature", "outdoorTemperature")

    def _build_matrices(self, p=None):
        """Build the RC state-space matrices ``(A, B, C, D, E, F)`` from the
        physical parameters -- a **pure** function of ``p``.

        ``p`` is a dict ``{name: value}`` of *physical* parameter tensors
        (:attr:`PARAM_NAMES`); ``None``: the component's own values.  Passing
        ``p`` explicitly is the functorch fast path: the parameters are plain
        tensor *arguments*, so ``jacrev`` w.r.t. ``p`` is clean under ``vmap``.
        Shapes: ``A (n_c, 2, 2)``, ``B (n_c, 2, m)``, ``C (n_c, 2, 2)``,
        ``D (n_c, 2, m)``, ``E (n_c, m, 2, 2)``, ``F (n_c, m, 2, m)``.
        """
        if p is None:
            p = {name: getattr(self, name).get() for name in self.PARAM_NAMES}
        n_states = 2  # air and wall temperature
        n_inputs = 9 + self.n_walls  # the eight base inputs, the walls, the infiltration

        # Parameter values, shape (n_c_param,); may be 1 even when self.n_c > 1
        # (batched components share identical params).
        C_air = p["C_air"]
        C_wall = p["C_wall"]
        R_in = p["R_in"]
        R_out = p["R_out"]
        f_air = p["f_air"]
        f_wall = p["f_wall"]
        Q_occ_gain = p["Q_occ_gain"]
        m_inf = p["m_inf"]
        n_c = self.n_c
        dev, dt = C_air.device, C_air.dtype

        zero = torch.zeros_like(C_air)
        air_wall = 1 / (R_in * C_air)
        wall_air = 1 / (R_in * C_wall)
        wall_outdoor = 1 / (R_out * C_wall)
        # Infiltration: outdoor air in, room air out, at a constant flow.
        air_outdoor = constants.CP_AIR * m_inf / C_air
        A = torch.stack(
            [
                torch.stack([-air_wall - air_outdoor, air_wall], dim=-1),
                torch.stack([wall_air, -wall_air - wall_outdoor], dim=-1),
            ],
            dim=1,
        )

        # Base slots: [T_out, m_sup, m_mu, T_sup, irradiation, N, Q_heat, T_mu]
        air_inputs = [air_outdoor, zero, zero, zero, f_air / C_air, Q_occ_gain / C_air, 1 / C_air, zero]
        wall_inputs = [wall_outdoor, zero, zero, zero, f_wall / C_wall, zero, zero, zero]
        air_inputs.extend([1 / C_air] * self.n_walls)
        wall_inputs.extend([zero] * self.n_walls)
        # the envelope inflow's column (last): it acts through the input
        # transform, on the make-up slot, not through B
        air_inputs.append(zero)
        wall_inputs.append(zero)
        B = torch.stack([torch.stack(air_inputs, dim=-1), torch.stack(wall_inputs, dim=-1)], dim=1)

        # Both states observed; no direct feedthrough.
        C_out = torch.eye(n_states, dtype=dt, device=dev).unsqueeze(0).expand(n_c, -1, -1).clone()
        D = torch.zeros((n_c, n_states, n_inputs), dtype=dt, device=dev)

        # Balanced ventilation (see air_balance.py): slot 1 is the supply
        # flow, slot 2 the make-up flow max(m_exh - m_sup, 0).  Each entering
        # stream displaces room air at T_i (E) and brings heat at its own
        # temperature (F): supply at T_sup (slot 3), make-up at T_mu (slot 7,
        # the outdoor temperature unless makeUpAirTemperature is wired).
        input_basis = torch.eye(n_inputs, dtype=dt, device=dev)
        state_basis = torch.eye(n_states, dtype=dt, device=dev)
        u_supply_flow = input_basis[1]
        u_make_up_flow = input_basis[2]
        u_supply_temperature = input_basis[3]
        u_make_up_temperature = input_basis[self.MAKE_UP_TEMPERATURE_SLOT]
        state_air = state_basis[0]
        gain = (constants.CP_AIR / C_air).reshape(n_c, 1, 1, 1)
        E = (
            -gain
            * (u_supply_flow + u_make_up_flow).reshape(1, n_inputs, 1, 1)
            * state_air.reshape(1, 1, n_states, 1)
            * state_air.reshape(1, 1, 1, n_states)
        )
        F = gain * state_air.reshape(1, 1, n_states, 1) * (
            u_supply_flow.reshape(1, n_inputs, 1, 1) * u_supply_temperature.reshape(1, 1, 1, n_inputs)
            + u_make_up_flow.reshape(1, n_inputs, 1, 1) * u_make_up_temperature.reshape(1, 1, 1, n_inputs)
        )
        return A, B, C_out, D, E, F

    def derive(self, p, step_size):
        """The state-space matrices, which depend on the parameters alone."""
        return {"matrices": self._build_matrices(p)}

    def forward(self, x, inputs, params, step_size):
        """One step of the room air and wall: it takes the RC matrices from
        ``params`` (:meth:`derive`) and one bilinear ZOH step.

        Args:
            x: state ``(n_c, 2)`` = ``[T_indoor, T_wall]``.
            inputs: the input-port values (each ``(n_c,)``, or ``(n_c, n_v)``
                for ``wallHeatGain``).
            params: the physical parameter values and the matrices.
            step_size: step size in seconds.

        Returns:
            ``(x_next (n_c, 2), {"indoorTemperature", "wallTemperature",
            "surplusAirFlowRate", "makeUpAirFlowRate"})``.
        """
        A, B, C, D, E, F = params["matrices"]
        # the imbalance from the ports' values, before the make-up transform
        derived = self._ss_derived_outputs(inputs, {})
        inputs = {**inputs, **self._ss_transform_inputs(inputs)}
        cols = [
            inputs["outdoorTemperature"],
            inputs["supplyAirFlowRate"],
            inputs["exhaustAirFlowRate"],
            inputs["supplyAirTemperature"],
            inputs["globalIrradiation"],
            inputs["numberOfPeople"],
            inputs["heatGain"],
            inputs["makeUpAirTemperature"],
        ]
        u = torch.stack(cols, dim=-1)  # (n_c, 8)
        if self.n_walls > 0:
            u = torch.cat([u, inputs["wallHeatGain"]], dim=-1)
        u = torch.cat([u, inputs[air_balance.INFILTRATION_PORT].unsqueeze(-1)], dim=-1)
        x_next, y = bilinear_onestep(A, B, C, D, E, F, x, u, step_size)
        return x_next, {"indoorTemperature": y[..., 0], "wallTemperature": y[..., 1], **derived}


# Deprecated aliases (removed in twin4build 2.1)
BuildingSpaceThermalTorchSystem = BuildingSpaceThermalSystem
