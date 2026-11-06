# Third party imports
import torch

# Local application imports
import twin4build.core as core
import twin4build.utils.constants as constants
from twin4build.systems.building_space import air_balance
from twin4build.systems.saref4syst.declarations import Input, Output, Parameter, State
from twin4build.systems.utils.discrete_statespace_system import bilinear_onestep


#: Matrix contract of the CO2 mass balance, shared with every model that has
#: to speak it (see :func:`mass_matrices`).
N_STATES = 1
#: ``u = [supplyAirFlowRate, makeUpAirFlow, outdoorCo2Concentration, numberOfPeople, makeUpAirCO2]``
N_INPUTS = 5
#: Index of the outdoor-CO2 slot: the concentration the supply stream and,
#: when ``makeUpAirCO2`` is unwired, the make-up stream bring in.
OUTDOOR_CO2_SLOT = 2
#: Index of the occupancy slot in ``u`` -- the column of ``Bd`` that the
#: occupancy inversion divides by.
OCCUPANCY_SLOT = 3
#: Index of the make-up-air CO2 slot: the concentration the make-up stream
#: brings in when the port is wired (a transfer node), else ignored.
MAKE_UP_CO2_SLOT = 4
#: Index of the supply-air CO2 slot, a column only when ``supplyAirCO2`` is
#: wired (the supply stream then brings that concentration instead of the
#: outdoor one); the ``exchangeCO2Gain`` slots follow it.  Unwired zones keep
#: the :data:`N_INPUTS` base columns, bit for bit.
SUPPLY_CO2_SLOT = N_INPUTS
#: Input slots whose ``B_eff`` column is structurally nonzero: the base ``B``
#: is nonzero only in 2 (outdoor CO2) and 3 (occupancy), and ``F`` writes only
#: slot 2.  ``B_d = phi(A_eff dt) dt B_eff`` is linear in ``B_eff`` column by
#: column, so discretizing with only these columns yields BIT-IDENTICAL
#: ``A_d`` and the same ``B_d`` entries -- the dropped columns enter the
#: Taylor/squaring recursion only as exact-zero addends.  The occupancy
#: inversion uses this to exponentiate a 3x3 block instead of a 5x5, which
#: matters: the captured reverse-mode rollout is the run's memory ceiling.
#: ``test_occupancy_inversion_round_trip`` pins both the bit-identity and the
#: claim that no other column is ever nonzero.
ACTIVE_INPUT_SLOTS = (2, 3, 4)
#: Position of the occupancy slot within :data:`ACTIVE_INPUT_SLOTS`.
OCCUPANCY_SLOT_ACTIVE = 1
#: The same selection as a ``slice``.  The active slots are contiguous, and
#: they MUST be selected this way: ``t[..., [2, 3]]`` is advanced indexing,
#: which builds a CPU index tensor and copies it to the device -- illegal
#: during CUDA-graph capture ("Cannot copy between CPU and CUDA tensors
#: during CUDA graph capture").  ``torch.compile`` folds the list away at
#: trace time, so a list index only fails on the UNCOMPILED captured
#: rollout (the post-fit ``Simulator.simulate``), which is why it survives
#: the estimation and then aborts.  A slice is a view: no index tensor, no
#: copy, nothing to fold.
ACTIVE_INPUT_SLICE = slice(ACTIVE_INPUT_SLOTS[0], ACTIVE_INPUT_SLOTS[-1] + 1)
assert tuple(range(*ACTIVE_INPUT_SLICE.indices(N_INPUTS))) == ACTIVE_INPUT_SLOTS, (
    "ACTIVE_INPUT_SLOTS must stay contiguous for ACTIVE_INPUT_SLICE to match"
)


def mass_matrices(V, G_occ, m_inf, n_c, make_up_co2_slot=OUTDOOR_CO2_SLOT, n_exchanges=0, supply_co2=False,
                  infiltration_column=False):
    r"""``(A, B, C, D, E, F)`` of the room CO2 mass balance -- the single
    source of the matrix contract.

    A pure function of the physical parameters (each shaped ``(n_c,)``) and
    the parallel-component count.  :meth:`BuildingSpaceMassSystem._build_matrices`
    is a thin wrapper, and
    :meth:`~twin4build.systems.utils.occupancy_system.OccupancySystem.invert_zoh_occupancy`
    calls it directly: the inversion is the inverse of *these* matrices
    discretized by *the same* :func:`~twin4build.systems.utils.discrete_statespace_system._discretize_onestep`,
    not of a separately derived reduced system.  Any second derivation
    drifts from the forward model (and puts an untested matrix
    exponential on the captured CUDA graph).

    ``make_up_co2_slot`` is the ``u`` column whose concentration the make-up
    stream brings in: :data:`OUTDOOR_CO2_SLOT` (default; the outdoor level)
    or :data:`MAKE_UP_CO2_SLOT` when a transfer node feeds ``makeUpAirCO2``.

    ``supply_co2`` appends the :data:`SUPPLY_CO2_SLOT` column, the
    concentration the supply stream brings in (``supplyAirCO2`` wired, e.g. a
    storey interior supplied by its rooms' surplus air); without it the supply
    stream brings the outdoor level.

    ``n_exchanges`` appends one ``exchangeCO2Gain`` column per connected
    :class:`~twin4build.systems.utils.opening_system.OpeningSystem`, a linear
    source ``1/m_air`` [ppm kg/s -> ppm/s] exactly as ``wallHeatGain`` enters
    the thermal zone with ``1/C_air``.

    Shapes with ``m = 5 + supply_co2 + n_exchanges``: ``A (n_c, 1, 1)``, ``B (n_c, 1, m)``,
    ``C (n_c, 1, 1)``, ``D (n_c, 1, m)``, ``E (n_c, m, 1, 1)``, ``F (n_c, m, 1, m)``.
    """
    n_supply = int(bool(supply_co2))
    # infiltration_column: the envelope inflow's column, last (it acts through the input transform, on the
    # make-up slot, not through B); the room model has it, the occupancy inversion does not
    n_inflow = int(bool(infiltration_column))
    n_states, n_inputs = N_STATES, N_INPUTS + n_supply + int(n_exchanges) + n_inflow
    # Parameters' device/dtype: _build_matrices re-runs on cache miss
    # during stepping, outside initialize()'s device context.
    dev, dt = V.device, V.dtype

    # Calculate air mass from volume and density
    density_air = constants.RHO_AIR
    air_mass = V * density_air  # (n_c,)

    zero = torch.zeros_like(air_mass)
    infiltration = m_inf / air_mass
    people_gain = (G_occ / air_mass) * (constants.M_AIR / constants.M_CO2) * 1e6
    A = (-infiltration).reshape(n_c, n_states, n_states)
    exchange_gain = [1 / air_mass] * int(n_exchanges)  # exchangeCO2Gain slots
    supply_slot = [zero] * n_supply  # supplyAirCO2: enters through F only
    B = torch.stack(
        [zero, zero, infiltration, people_gain, zero] + supply_slot + exchange_gain + [zero] * n_inflow, dim=-1
    ).unsqueeze(1)

    # Output matrix C - Identity matrix for direct observation
    # Shape: (n_c, n_states, n_states)
    C = (
        torch.eye(n_states, dtype=dt, device=dev)
        .unsqueeze(0)
        .expand(n_c, -1, -1)
        .clone()
    )

    # Feedthrough matrix D (no direct feedthrough) - Shape: (n_c, n_states, n_inputs)
    D = torch.zeros((n_c, n_states, n_inputs), dtype=dt, device=dev)

    # Balanced ventilation (see air_balance.py): slot 0 is the supply
    # flow, slot 1 the outdoor make-up flow max(m_exh - m_sup, 0).  Each
    # entering stream displaces room air at C (E) and brings outdoor air
    # at C_out (F, slot 2).
    # E matrix for input-state coupling: shape (n_c, n_inputs, n_states, n_states)
    E = torch.stack(
        [-1 / air_mass, -1 / air_mass, zero, zero, zero] + [zero] * (n_supply + int(n_exchanges) + n_inflow), dim=1
    ).reshape(n_c, n_inputs, n_states, n_states)

    # F matrix for input-input coupling: shape (n_c, n_inputs, n_states, n_inputs)
    # The supply stream brings air at the supply slot -- outdoor (slot 2)
    # unless supplyAirCO2 is wired; the make-up stream brings air at
    # ``make_up_co2_slot`` -- outdoor unless a transfer node is wired.
    input_basis = torch.eye(n_inputs, dtype=dt, device=dev)
    supply_co2_slot = SUPPLY_CO2_SLOT if n_supply else OUTDOOR_CO2_SLOT
    F = (1 / air_mass).reshape(n_c, 1, 1, 1) * (
        input_basis[0].reshape(1, n_inputs, 1, 1)
        * input_basis[supply_co2_slot].reshape(1, 1, 1, n_inputs)
        + input_basis[1].reshape(1, n_inputs, 1, 1)
        * input_basis[make_up_co2_slot].reshape(1, 1, 1, n_inputs)
    )

    return A, B, C, D, E, F


class BuildingSpaceMassSystem(core.System):
    r"""
    Building Space CO2 Concentration Model using Mass Balance Dynamics.

    This model represents the CO2 concentration dynamics in a building space considering
    supply and exhaust air flows, occupant CO2 generation, infiltration, and outdoor
    CO2 concentration using bilinear state-space dynamics.

    Args:
        V: Volume of the space [m³]
        G_occ: CO2 generation rate per occupant [kg_CO2/s]
        m_inf: Infiltration rate [kg/s]

    Mathematical Formulation
    ------------------------

    **Continuous-Time Differential Equation:**

    The CO2 concentration dynamics are governed by a mass balance on the room air.
    With the room air mass :math:`m_{air} = \rho_{air} V` [kg], the implemented
    equation is:

    .. math::

       m_{air}\frac{dC}{dt} = \dot{m}_{sup}(C_{out} - C) + \dot{m}_{mu}(C_{out} - C) + \dot{m}_{inf}(C_{out} - C) + G_{occ} N_{occ} \frac{M_{air}}{M_{CO2}} \cdot 10^6

    with the outdoor **make-up flow** :math:`\dot{m}_{mu} = \max(\dot{m}_{exh}
    - \dot{m}_{sup}, 0)`: the room air mass is constant, so every entering
    stream (supply, make-up, infiltration) is balanced by room air leaving at
    :math:`C`.  Supply in excess of the exhaust leaves through the envelope and
    the exhaust flow drops out; exhaust in excess of the supply draws outdoor
    air in through the envelope.  See
    :mod:`twin4build.systems.building_space.air_balance`.

    where:

       - :math:`m_{air} = \rho_{air} V`: Mass of air in the space [kg]
         (:math:`\rho_{air}` is the constant air density from
         ``twin4build.utils.constants``)
       - :math:`V`: Volume of the space [m³] (parameter)
       - :math:`C`: Indoor CO2 concentration [ppmv] (state variable)
       - :math:`\dot{m}_{sup}`: Supply air mass flow rate [kg/s] (input)
       - :math:`\dot{m}_{exh}`: Exhaust air mass flow rate [kg/s] (input; enters
         only through :math:`\dot{m}_{mu}`)
       - :math:`\dot{m}_{inf}`: Infiltration mass flow rate [kg/s] (parameter)
       - :math:`C_{out}`: Outdoor CO2 concentration [ppmv] (input)
       - :math:`G_{occ}`: CO2 generation rate per occupant [kg_CO2/s] (parameter)
       - :math:`N_{occ}`: Number of occupants (input)
       - :math:`M_{air}`, :math:`M_{CO2}`: Molar masses of air and CO2

    The factor :math:`\frac{M_{air}}{M_{CO2}} \cdot 10^6` converts the occupant
    CO2 mass generation [kg_CO2/s] per kg of room air into a rate of change of
    the volumetric (molar) concentration [ppmv/s].

    .. note::
       Concentrations are expressed in **ppmv** (parts per million by volume),
       which is equivalent to **ppm-moles** (molar fraction × 10⁶) for ideal gases.

    Note: the supply air's CO2 concentration is the outdoor one unless
    ``supplyAirCO2`` is wired (a zone supplied by other zones' air, e.g. a
    storey interior taking its rooms' surplus): then :math:`C_{out}` in the
    supply term is that port's value.

    **State-Space Representation:**

    The system is implemented using the DiscreteStatespaceSystem (the continuous
    dynamics above are discretized each step by ``bilinear_onestep`` using the
    bilinear/Tustin one-step map) with matrices:

    *State vector:* :math:`\mathbf{x} = \begin{bmatrix}C\end{bmatrix}`

    *Input vector:* :math:`\mathbf{u} = \begin{bmatrix}\dot{m}_{sup} \\ \dot{m}_{mu} \\ C_{out} \\ N_{occ}\end{bmatrix}`
    (the ``exhaustAirFlowRate`` port is transformed to :math:`\dot{m}_{mu}` at
    input assembly)

    *Base System Matrices:*

    .. math::

       \mathbf{A} = \begin{bmatrix} -\frac{\dot{m}_{inf}}{m_{air}} \end{bmatrix}

       \mathbf{B} = \begin{bmatrix} 0 & 0 & \frac{\dot{m}_{inf}}{m_{air}} & \frac{G_{occ}}{m_{air}}\frac{M_{air}}{M_{CO2}} \cdot 10^6 \end{bmatrix}

       \mathbf{C} = \begin{bmatrix} 1 \end{bmatrix}

       \mathbf{D} = \begin{bmatrix} 0 & 0 & 0 & 0 \end{bmatrix}

    **Bilinear Coupling Matrices:**

    *State-Input Coupling (E matrices):*

    .. math::

       \mathbf{E} \in \mathbb{R}^{4 \times 1 \times 1} = \begin{bmatrix}
       \begin{bmatrix} -\frac{1}{m_{air}} \end{bmatrix} & \text{(supply flow)} \\
       \begin{bmatrix} -\frac{1}{m_{air}} \end{bmatrix} & \text{(make-up flow)} \\
       \begin{bmatrix} 0 \end{bmatrix} & \text{(outdoor CO2)} \\
       \begin{bmatrix} 0 \end{bmatrix} & \text{(occupants)}
       \end{bmatrix}

    *Input-Input Coupling (F matrices):*

    .. math::

       \mathbf{F} \in \mathbb{R}^{4 \times 1 \times 4} = \begin{bmatrix}
       \begin{bmatrix} 0 & 0 & \frac{1}{m_{air}} & 0 \end{bmatrix} & \text{(supply flow)} \\
       \begin{bmatrix} 0 & 0 & \frac{1}{m_{air}} & 0 \end{bmatrix} & \text{(make-up flow)} \\
       \begin{bmatrix} 0 & 0 & 0 & 0 \end{bmatrix} & \text{(outdoor CO2)} \\
       \begin{bmatrix} 0 & 0 & 0 & 0 \end{bmatrix} & \text{(occupants)}
       \end{bmatrix}


    *Bilinear Effects*

    The bilinear terms handle specific flow-dependent mass transfer effects:
       - :math:`\mathbf{F}[0,0,2] \cdot u_0 \cdot u_2 + \mathbf{E}[0,0,0] \cdot u_0 \cdot x_0 = \frac{1}{m_{air}} \dot{m}_{sup} (C_{out} - C)`: Supply flow replacing room air by outdoor air
       - :math:`\mathbf{F}[1,0,2] \cdot u_1 \cdot u_2 + \mathbf{E}[1,0,0] \cdot u_1 \cdot x_0 = \frac{1}{m_{air}} \dot{m}_{mu} (C_{out} - C)`: Make-up flow doing the same for the exhaust deficit

    Physical Interpretation
    -----------------------

    **Mass Balance System:**
       - Single state represents indoor CO2 concentration
       - Inputs represent ventilation flows, outdoor conditions, and occupancy
       - Bilinear terms model flow-dependent mass transfer accurately

    **Flow-Dependent Effects:**
       - Supply air flow brings outdoor CO2 at outdoor concentration and
         displaces room air at indoor concentration (F and E matrix coupling)
       - Exhaust in excess of the supply draws outdoor air through the
         envelope (make-up flow, same coupling on the second slot); the indoor
         concentration can therefore never be driven below the outdoor one
         by ventilation alone

    Computational Features
    ----------------------

       - **Automatic Differentiation:** PyTorch tensors enable gradient computation
       - **Adaptive Discretization:** Matrices updated when flows change significantly
       - **Parameter Estimation:** All mass balance parameters available for calibration

    Examples
    --------
    Basic CO2 model:

    >>> import twin4build as tb
    >>>
    >>> # Create CO2 model with default parameters
    >>> co2_model = tb.BuildingSpaceMassSystem(
    ...     V=150,          # Room volume [m³]
    ...     G_occ=6e-6,     # Higher CO2 generation per person
    ...     m_inf=0.002,    # Higher infiltration rate
    ...     id="zone_1_co2"
    ... )

    Large space CO2 model:

    >>> # Model for large space with higher occupancy
    >>> co2_model = tb.BuildingSpaceMassSystem(
    ...     V=500,          # Large space volume
    ...     G_occ=4e-6,     # Lower per-person generation
    ...     m_inf=0.005,    # Higher infiltration for large space
    ...     id="large_space_co2"
    ... )
    """

    supplyAirFlowRate = Input(doc="Supply air flow rate [kg/s]")
    exhaustAirFlowRate = Input(doc="Exhaust air flow rate [kg/s]")
    outdoorCo2Concentration = Input(default=400.0, alias="outdoorCO2", doc="Outdoor CO2 concentration [ppmv]")
    numberOfPeople = Input(doc="Number of occupants")
    makeUpAirCO2 = Input(default=outdoorCo2Concentration, doc="CO2 of the make-up air [ppmv]; unwired: outdoor air")
    supplyAirCO2 = Input(default=outdoorCo2Concentration, doc="CO2 of the supply air [ppmv]; unwired: outdoor air")
    exchangeCO2Gain = Input(size="wired", doc="CO2 exchanged with other zones [kg/s], one slot per opening")
    infiltrationAirFlowRate = Input(default=0.0, doc="Envelope inflow on top of the make-up flow [kg/s]")
    indoorCO2 = Output(init=400.0, doc="Indoor CO2 concentration [ppmv]")
    co2 = State(size=1, init=400.0, names=["CO2"], doc="Indoor CO2 concentration [ppmv]")
    V = Parameter(100.0, lb=10.0, ub=1000.0, relative=(0.5, 2.0), doc="Air volume [m3]")
    G_occ = Parameter(5e-6, lb=1e-6, ub=1e-5, doc="CO2 generation per occupant [kg/s]")
    m_inf = Parameter(0.001, lb=1e-4, ub=0.01, relative=(0.2, 5.0), doc="Infiltration air flow [kg/s]")

    #: ``indoorCO2`` publishes the CO2 state as it is (see
    #: :meth:`System.state_output_ports`).
    STATE_OUTPUT_PORTS = {"indoorCO2": 0}

    @property
    def n_exchanges(self) -> int:
        """Connected ``exchangeCO2Gain`` slots (openings to other zones)."""
        return int(self.input["exchangeCO2Gain"]._n_v or 0)

    @n_exchanges.setter
    def n_exchanges(self, value: int) -> None:
        # The width of a model stepped on its own (wiring gives it otherwise).
        self.input["exchangeCO2Gain"]._n_v = int(value)

    def _ss_layout(self):
        """Port <-> matrix index map, mirroring :meth:`forward` exactly:
        ``u = [supplyAirFlowRate, exhaustAirFlowRate, outdoorCo2Concentration,
        numberOfPeople, makeUpAirCO2, supplyAirCO2, exchangeCO2Gain x
        n_exchanges, infiltrationAirFlowRate]`` (the exhaust slot holds the
        make-up flow after :meth:`_ss_transform_inputs`); single output row
        ``indoorCO2``."""
        u = [
            ("supplyAirFlowRate", 1),
            ("exhaustAirFlowRate", 1),
            ("outdoorCo2Concentration", 1),
            ("numberOfPeople", 1),
            ("makeUpAirCO2", 1),
            ("supplyAirCO2", 1),
        ]
        if self.n_exchanges > 0:
            u.append(("exchangeCO2Gain", self.n_exchanges))
        u.append((air_balance.INFILTRATION_PORT, 1))  # last: it acts through the transform only
        return {"u": u, "y": {"indoorCO2": 0}}

    def _ss_support(self):
        """Conservative structural support of the ``D``, ``E`` and ``F`` matrices."""
        return {
            "D": frozenset(),
            "E": frozenset({(0, 0, 0), (1, 0, 0)}),
            "F": frozenset({(0, 0, 2), (1, 0, 2), (1, 0, MAKE_UP_CO2_SLOT), (0, 0, SUPPLY_CO2_SLOT)}),
        }

    @staticmethod
    def _ss_transform_inputs(inputs):
        """Balanced-ventilation input transform (see
        :mod:`~twin4build.systems.building_space.air_balance`): the
        ``exhaustAirFlowRate`` slot carries the outdoor make-up flow
        ``max(m_exh - m_sup, 0)``.  Pure function of the original inputs;
        applied by :meth:`forward` and by the fused block."""
        return air_balance.balanced_flow_inputs(inputs, "makeUpAirCO2", "outdoorCo2Concentration")

    def _build_matrices(self, p=None):
        """Build the CO2 mass-balance matrices ``(A, B, C, D, E, F)`` from the
        physical parameters -- a pure function of ``p`` (a dict of physical values
        for :attr:`PARAM_NAMES`; defaults to ``self.<name>.get()``).  The make-up
        and the supply stream bring the CO2 of their ports, which read the
        outdoor CO2 while unwired."""
        if p is None:
            p = {name: getattr(self, name).get() for name in self.PARAM_NAMES}
        return mass_matrices(
            p["V"], p["G_occ"], p["m_inf"], self.n_c,
            make_up_co2_slot=MAKE_UP_CO2_SLOT, n_exchanges=self.n_exchanges, supply_co2=True,
            infiltration_column=True,
        )

    def derive(self, p, step_size):
        """The state-space matrices, which depend on the parameters alone."""
        return {"matrices": self._build_matrices(p)}

    def forward(self, x, inputs, params, step_size):
        """One step of the room's CO2: ``(state, inputs, params) -> (new_state,
        {"indoorCO2"})``."""
        A, B, C, D, E, F = params["matrices"]
        inputs = {**inputs, **self._ss_transform_inputs(inputs)}
        u = torch.stack(
            [
                inputs["supplyAirFlowRate"],
                inputs["exhaustAirFlowRate"],
                inputs["outdoorCo2Concentration"],
                inputs["numberOfPeople"],
                inputs["makeUpAirCO2"],
                inputs["supplyAirCO2"],
            ],
            dim=-1,
        )
        if self.n_exchanges > 0:
            u = torch.cat([u, inputs["exchangeCO2Gain"]], dim=-1)
        u = torch.cat([u, inputs[air_balance.INFILTRATION_PORT].unsqueeze(-1)], dim=-1)
        x_next, y = bilinear_onestep(A, B, C, D, E, F, x, u, step_size)
        return x_next, {"indoorCO2": y[..., 0]}


# Deprecated aliases (removed in twin4build 2.1)
BuildingSpaceMassTorchSystem = BuildingSpaceMassSystem
