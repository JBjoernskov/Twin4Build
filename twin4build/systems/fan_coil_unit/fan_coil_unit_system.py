# Standard library imports
import datetime
from typing import Dict, List, Optional

# Third party imports
import torch
import torch.nn as nn
from scipy.optimize import fsolve

# Local application imports
import twin4build.utils.constants as constants
import twin4build.utils.types as tps
from twin4build import core
from twin4build.systems.utils.discrete_statespace_system import (
    DiscreteStatespaceSystem,
    bilinear_onestep,
)
from twin4build.systems.valve.valve_system import ValveSystem
from twin4build.translator.translator import (
    StepRule,
    Node,
    OptionalRule,
    SignaturePattern,
)


class FanCoilUnitSystem(core.System, nn.Module):
    r"""
    Fan Coil Unit (FCU) Model with Finite Element Discretization.

    This model represents a fan coil unit that transfers heat between a hot/chilled water
    stream and a forced air stream using multiple finite elements and bilinear state-space
    dynamics. The water side of the coil is discretized to capture temperature distribution
    along its length, while the air side is treated as quasi-steady-state (valid when
    simulation timesteps are much larger than air transit time through the coil).

    The model reuses the same bilinear state-space framework as ``SpaceHeaterSystem``,
    with the key difference that heat is exchanged with a forced air stream rather than
    quasi-static room air, and an air outlet temperature is computed from an energy balance.

    Args:
        Q_flow_nominal: Nominal heat output [W]
        T_w_supply_nominal: Nominal supply water temperature [°C]
        T_w_return_nominal: Nominal return water temperature [°C]
        T_air_in_nominal: Nominal inlet air temperature [°C]
        thermalMassHeatCapacity: Total water-side thermal mass heat capacity [J/K]
        nelements: Number of finite elements along the water path
        initialize_UA: If True (default), UA is computed via fsolve to match nominal
            conditions on first initialization. If False, the UA value is used as-is,
            which is useful when UA is being estimated/calibrated.

    Mathematical Formulation
    ------------------------

    **Continuous-Time Differential Equations:**

    The model uses a state-space representation with n finite elements along the water
    flow path. For each water element i, the temperature dynamics are:

    .. math::

        C_1 \frac{dT_1}{dt} = \dot{m}_w \cdot c_{p,w} \cdot (T_{w,sup} - T_1)
            - \frac{UA}{n} \cdot (T_1 - T_{air,in}) \quad \forall i = 1

    .. math::

        C_i \frac{dT_i}{dt} = \dot{m}_w \cdot c_{p,w} \cdot (T_{i-1} - T_i)
            - \frac{UA}{n} \cdot (T_i - T_{air,in}) \quad \forall i \in \{2..n\}

    where:
       - :math:`C_i` is the thermal capacitance of water element i [J/K]
       - :math:`T_i` is the water temperature of element i [°C]
       - :math:`\dot{m}_w` is the water mass flow rate [kg/s]
       - :math:`c_{p,w}` is the specific heat capacity of water [J/(kg·K)]
       - :math:`UA` is the overall heat transfer coefficient [W/K]
       - :math:`n` is the number of elements
       - :math:`T_{air,in}` is the inlet air temperature [°C]
       - :math:`T_{w,sup}` is the supply water temperature [°C]

    The air side is treated as quasi-steady-state, which is valid because the air transit
    time through the coil (~seconds) is much smaller than typical simulation timesteps
    (~minutes). This is consistent with standard building simulation tools (EnergyPlus,
    Modelica Buildings library).

    **Heat Output and Air Outlet Temperature (Effectiveness-NTU):**

    The air-side outlet temperature and heat transfer use the effectiveness-NTU
    method, which treats the air stream as the heat-capacity-limited fluid
    (the water-side capacity rate is effectively infinite over a single timestep
    thanks to the thermal mass per element):

    .. math::

        \mathrm{NTU} = \frac{UA}{\dot{m}_a \cdot c_{p,a}}

    .. math::

        \varepsilon = 1 - e^{-\mathrm{NTU}}

    .. math::

        T_{air,out} = T_{air,in} + \varepsilon \cdot (\bar{T}_w - T_{air,in})

    .. math::

        Q = \varepsilon \cdot \dot{m}_a \cdot c_{p,a} \cdot (\bar{T}_w - T_{air,in})

    where :math:`\bar{T}_w = \tfrac{1}{n}\sum_i T_i` is the average water
    temperature across elements, :math:`\dot{m}_a` is the air mass flow rate
    [kg/s] and :math:`c_{p,a}` is the specific heat capacity of air [J/(kg·K)].

    Limiting behavior:

    * As :math:`\dot{m}_a \to 0`: :math:`\mathrm{NTU} \to \infty`,
      :math:`\varepsilon \to 1`, the air outlet asymptotes to
      :math:`\bar{T}_w` and the actual heat transfer :math:`Q \to 0`
      (no air to carry heat away).  This avoids the unbounded
      :math:`Q / \dot{m}_a` blowup of a naive energy balance.
    * As :math:`\dot{m}_a \to \infty` (large NTU is small):
      :math:`\varepsilon \to \mathrm{NTU}`, so
      :math:`Q \to UA \cdot (\bar{T}_w - T_{air,in})`, recovering the
      constant-UA limit used by the water-side state equations.

    Positive Q indicates heating (water hotter than air), negative Q indicates cooling
    (water colder than air).  ``Power`` carries the sign of :math:`\dot{m}_a`
    to remain consistent with backflow conventions.

    .. note::
       The water-side state equations still use the constant ``UA/n``
       coefficient in their ``A`` / ``B`` matrices.  At very low air flow,
       this means the water-side cools (or heats) slightly faster than the
       air actually carries the heat -- a small energy-balance
       inconsistency.  This simplification keeps the state-space framework
       bilinear; ``outletAirTemperature`` and ``Power`` are guaranteed to
       stay physically bounded.

    **State-Space Representation:**

    The state-space matrices are identical in structure to ``SpaceHeaterSystem``:

    *State vector:* :math:`\mathbf{x} = \begin{bmatrix}T_1 \\ T_2 \\ \vdots \\ T_n\end{bmatrix}`

    *Input vector:* :math:`\mathbf{u} = \begin{bmatrix}T_{w,sup} \\ \dot{m}_w \\ T_{air,in}\end{bmatrix}`

    The bilinear state-space formulation handles the flow-dependent heat transfer:

    .. math::

        \dot{\mathbf{x}} = \mathbf{A}\mathbf{x} + \mathbf{B}\mathbf{u}
            + \sum_k \mathbf{E}_k \mathbf{x} u_k
            + \sum_k \mathbf{F}_k \mathbf{u} u_k

    .. math::

        \mathbf{y} = \mathbf{C}\mathbf{x} + \mathbf{D}\mathbf{u}

    Physical Interpretation
    -----------------------

    **Cross-Flow Heat Exchanger:**
       - Water flows through tubes in a serpentine pattern
       - Fan forces room air across the coil perpendicular to the water flow
       - Each water element exchanges heat with fresh inlet air (cross-flow assumption)
       - The model captures both heating and cooling modes

    **Flow-Dependent Effects:**
       - Water flow brings heat at supply temperature to first element (F matrix)
       - Water flow transfers heat between consecutive elements (E matrix)
       - Air outlet temperature depends on both total heat transfer and air flow rate

    Examples
    --------
    Basic fan coil unit for heating:

    >>> import twin4build as tb
    >>>
    >>> fcu = tb.FanCoilUnitSystem(
    ...     Q_flow_nominal=2000,            # 2 kW nominal output
    ...     T_w_supply_nominal=60,          # 60°C hot water supply
    ...     T_w_return_nominal=45,          # 45°C water return
    ...     T_air_in_nominal=21,            # 21°C room air entering coil
    ...     thermalMassHeatCapacity=50000,  # 50 kJ/K thermal mass
    ...     nelements=3,
    ...     id="heating_fcu"
    ... )

    Fan coil unit for cooling:

    >>> fcu_cool = tb.FanCoilUnitSystem(
    ...     Q_flow_nominal=-3000,           # 3 kW cooling (negative = cooling)
    ...     T_w_supply_nominal=7,           # 7°C chilled water supply
    ...     T_w_return_nominal=12,          # 12°C water return
    ...     T_air_in_nominal=26,            # 26°C room air entering coil
    ...     thermalMassHeatCapacity=30000,  # 30 kJ/K thermal mass
    ...     nelements=3,
    ...     id="cooling_fcu"
    ... )

    Notes
    -----
    Model Characteristics:
       - The water side is discretized into multiple elements for accurate temperature
         distribution modeling along the coil
       - The air side uses a quasi-steady-state assumption (valid for typical HVAC timesteps)
       - UA is the overall heat transfer coefficient including both water-side and air-side
         resistances
       - The model supports both heating and cooling operation depending on water and air
         temperatures

    Implementation Details:
       - The bilinear state-space matrices are structurally identical to SpaceHeaterSystem
       - All calculations use PyTorch tensors for gradient tracking
       - The UA value is optimized using numerical methods during initialization
       - Air outlet temperature is computed post-state-update via energy balance
    """

    def __init__(
        self,
        Q_flow_nominal: float = 2000,
        T_w_supply_nominal: float = 60,
        T_w_return_nominal: float = 45,
        T_air_in_nominal: float = 21,
        thermalMassHeatCapacity: float = 100000,
        nelements: int = 3,
        initialize_UA: bool = True,
        waterFlowRateMax: float = 1000 / ((60 - 45) * 4180),
        valveAuthority: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        nn.Module.__init__(self)
        self.Q_flow_nominal = Q_flow_nominal
        self.T_w_supply_nominal = T_w_supply_nominal
        self.T_w_return_nominal = T_w_return_nominal
        self.T_air_in_nominal = T_air_in_nominal
        self.nelements = nelements
        self.initialize_UA = initialize_UA
        # Placeholder ``UA`` value -- recomputed by ``fsolve`` at the
        # first ``initialize`` call from ``Q_flow_nominal``,
        # ``T_w_return_nominal`` and ``T_air_in_nominal`` (giving
        # ~ 80 W/K for the default 2 kW / 24 K nominal).  Set inside
        # the auto-estimator bound range so ``get_estimable_parameters``
        # captures a valid ``x0`` even when called before the parent
        # ``SimulationModel.initialize`` has run fsolve on this FCU.
        self.UA = tps.Parameter(
            torch.tensor(100.0, dtype=torch.float64), requires_grad=False
        )
        self.thermalMassHeatCapacity = tps.Parameter(
            torch.tensor(thermalMassHeatCapacity, dtype=torch.float64),
            requires_grad=False,
            scaling="log",
        )
        self._valve = ValveSystem(
            waterFlowRateMax=waterFlowRateMax,
            valveAuthority=valveAuthority,
            id=f"_valve_{self.id}",
        )

        self._input = {
            "supplyWaterTemperature": tps.Scalar(),
            "valvePosition": self._valve.input["valvePosition"],  # shared with internal valve
            "airFlowRate": tps.Scalar(),
            "inletAirTemperature": tps.Scalar(),
        }
        self._output = {
            "outletWaterTemperature": tps.Scalar(21),
            "outletAirTemperature": tps.Scalar(21),
            "waterFlowRate": tps.Scalar(0),
            "Power": tps.Scalar(0),
        }
        # Per-leaf bounds for the auto-estimator (see
        # :meth:`twin4build.core.System.get_estimable_parameters`).
        # Empty dicts mark attributes that are stored as plain Python
        # floats on the FCU (construction nominals) -- they are NOT
        # ``tps.Parameter`` instances and so are skipped by the
        # auto-estimator regardless of any bounds entry.  ``UA`` and
        # ``thermalMassHeatCapacity`` are tunable ``tps.Parameter`` s,
        # so they get real bounds here; the ``_valve.*`` paths
        # delegate via attribute walk to the valve sub-system's own
        # ``parameter`` map.
        # Physically-realistic bounds for the auto-estimator.  An FCU
        # / VAV reheat-coil sits roughly inside these ranges; widening
        # them just hands the solver useless feasible space and yields
        # estimates pinned to the limit (UA = 0.1 W/K is essentially
        # "no coil", UA = 1000 W/K an industrial-grade unit).  See
        # ``self._valve.parameter`` for the source-of-truth on the
        # ``_valve.*`` paths -- the duplicates below MUST stay in sync.
        self.parameter = {
            "Q_flow_nominal": {},
            "T_w_supply_nominal": {},
            "T_w_return_nominal": {},
            "T_air_in_nominal": {},
            # log-scaled (lb > 0 mandatory); spans copper coil + water
            # mass of small fan-coil to large reheat unit (~kg of metal
            # times c_p ~ 385 J/kgK plus ~kg of water times c_p ~ 4180).
            "thermalMassHeatCapacity": {"lb": 5e3, "ub": 5e5},
            # linear; VAV reheat coil ~ 50 - 500 W/K, dedicated FCU
            # coil up to ~ 2 kW/K.  Lower bound at 20 W/K still allows
            # very small zones; below that the coil cannot heat at all.
            "UA": {"lb": 20.0, "ub": 2000.0},
            "initialize_UA": {},
            # Duplicates the bounds on the inner valve sub-system; see
            # comment above ``self.parameter`` for the rule and check
            # ``self._valve.parameter`` for the rationale.
            "_valve.waterFlowRateMax": {"lb": 5e-5, "ub": 1.0},
            "_valve.valveAuthority": {"lb": 0.3, "ub": 1.0},
        }
        self._config = {"parameters": list(self.parameter.keys())}
        self.INITIALIZED = False

    @property
    def config(self) -> Dict[str, List[str]]:
        return self._config

    @property
    def input(self) -> dict:
        """
        Get the input ports of the fan coil unit system.

        Returns:
            dict: Dictionary containing input ports:
                - "supplyWaterTemperature": Supply water temperature [°C]
                - "valvePosition": Water-side valve opening in [0, 1] (drives the
                  internal valve, which converts it to a water mass flow rate).
                - "airFlowRate": Air mass flow rate [kg/s]
                - "inletAirTemperature": Inlet air temperature [°C]
        """
        return self._input

    @property
    def output(self) -> dict:
        """
        Get the output ports of the fan coil unit system.

        Returns:
            dict: Dictionary containing output ports:
                - "outletWaterTemperature": Outlet water temperature [°C]
                - "outletAirTemperature": Outlet air temperature [°C]
                - "Power": Heat transfer rate to air [W] (positive=heating, negative=cooling)
        """
        return self._output

    def initialize(
        self,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
        step_size: int,
    ) -> None:
        """Initialize the fan coil unit system for simulation.

        Performs the following steps:

        1. If ``initialize_UA`` is True and this is the first call, numerically solves
           for the UA value that matches the nominal heat output
        2. Initializes input/output data structures
        3. Creates or reinitializes the bilinear state-space model

        Args:
            start_time: Start time of the simulation period.
            end_time: End time of the simulation period.
            step_size: Time step size in seconds.
        """
        _, _, max_timesteps, _ = core.Simulator.get_simulation_timesteps(
            start_time, end_time, step_size
        )
        batch_size = len(start_time)
        for input in self.input.values():
            input.initialize(n_t=max_timesteps, n_s=batch_size)
        for output in self.output.values():
            output.initialize(n_t=max_timesteps, n_s=batch_size)

        self.UA = self.UA.expand_to_n_c(self.n_c)
        self.thermalMassHeatCapacity = self.thermalMassHeatCapacity.expand_to_n_c(
            self.n_c
        )
        self._valve.n_c = self.n_c
        self._valve.initialize(start_time, end_time, step_size)

        if not self.INITIALIZED and self.initialize_UA:
            UA0 = float(
                abs(self.Q_flow_nominal)
                / abs(self.T_w_return_nominal - self.T_air_in_nominal)
            )
            root = fsolve(self._ua_residual, UA0, full_output=True)
            UA_val = root[0][0]
            self.UA.data.fill_(UA_val)

        self._create_state_space_model()
        self.ss_model.initialize(start_time, end_time, step_size)

        x0_tensor = self._get_initial_state_tensor()
        self.ss_model.set_state(x0_tensor)

        self.INITIALIZED = True

    def _ua_residual(self, UA_candidate):
        """Calculate the residual for UA optimization.

        Builds a steady-state linear system (with bilinear terms collapsed at nominal
        flow) and checks that the resulting heat output matches ``Q_flow_nominal``.

        Args:
            UA_candidate: Candidate UA value to evaluate.

        Returns:
            Difference between calculated and nominal heat output.
        """
        n = self.nelements
        C_elem = float(self.thermalMassHeatCapacity.get().item()) / n
        UA_elem = float(UA_candidate.item()) / n
        m_dot_w = float(
            abs(self.Q_flow_nominal)
            / (constants.CP_WATER * abs(self.T_w_supply_nominal - self.T_w_return_nominal))
        )
        c_p_w = float(constants.CP_WATER)

        # Build steady-state A, B (bilinear terms collapsed at nominal flow)
        # Input vector: [T_w_supply, m_dot_w, T_air_in]
        A = torch.zeros((n, n), dtype=torch.float64)
        B = torch.zeros((n, 3), dtype=torch.float64)
        for i in range(n):
            A[i, i] = -(m_dot_w * c_p_w + UA_elem) / C_elem
            if i > 0:
                A[i, i - 1] = (m_dot_w * c_p_w) / C_elem
        B[0, 0] = (m_dot_w * c_p_w) / C_elem
        for i in range(n):
            B[i, 2] = UA_elem / C_elem

        u = torch.tensor(
            [self.T_w_supply_nominal, m_dot_w, self.T_air_in_nominal],
            dtype=torch.float64,
        )
        try:
            x_ss = -torch.linalg.solve(A, B @ u)
        except Exception:
            return 1e6
        Power = UA_elem * torch.sum(x_ss - self.T_air_in_nominal)
        return Power - self.Q_flow_nominal

    def _get_initial_state_tensor(self):
        t_outlet = self.output["outletWaterTemperature"].get()
        n_s = t_outlet.shape[0]
        n_c = t_outlet.shape[1]
        x0 = torch.zeros((n_s, n_c, self.nelements), dtype=torch.float64)
        for i in range(self.nelements):
            x0[:, :, i] = t_outlet
        return x0

    PARAM_NAMES = (
        "thermalMassHeatCapacity",
        "UA",
        "_valve.waterFlowRateMax",
        "_valve.valveAuthority",
    )

    def _build_matrices(self, p=None):
        """Build the coil state-space matrices ``(A, B, C, D, E, F)`` from the
        physical parameters -- a pure function of ``p`` (a dict of physical
        values for :attr:`PARAM_NAMES`; defaults to the component's own
        values).  Passing ``p`` is the functorch fast path; the structure is
        identical to ``SpaceHeaterSystem`` with the air inlet
        temperature in the zone-temperature role.
        """
        if p is None:
            p = {
                "thermalMassHeatCapacity": self.thermalMassHeatCapacity.get(),
                "UA": self.UA.get(),
            }

        n = self.nelements
        n_inputs = 3  # [supplyWaterTemperature, waterFlowRate, inletAirTemperature]

        C_elem = p["thermalMassHeatCapacity"] / n  # (n_c,)
        UA_elem = p["UA"] / n  # (n_c,)
        n_c = C_elem.shape[0]
        c_p_w = constants.CP_WATER

        # A matrix: UA/C on diagonal (heat exchange with air) - shape (n_c, n, n)
        A = torch.zeros((n_c, n, n), dtype=torch.float64)
        for i in range(n):
            A[:, i, i] = -UA_elem / C_elem

        # B matrix: UA/C for air temperature input - shape (n_c, n, n_inputs)
        B = torch.zeros((n_c, n, n_inputs), dtype=torch.float64)
        for i in range(n):
            B[:, i, 2] = UA_elem / C_elem

        # E matrix: water flow rate coupling - shape (n_c, n_inputs, n, n)
        E = torch.zeros((n_c, n_inputs, n, n), dtype=torch.float64)
        for i in range(n):
            E[:, 1, i, i] = -c_p_w / C_elem
            if i > 0:
                E[:, 1, i, i - 1] = c_p_w / C_elem

        # F matrix: supply temperature * flow for first element - shape (n_c, n_inputs, n, n_inputs)
        F = torch.zeros((n_c, n_inputs, n, n_inputs), dtype=torch.float64)
        F[:, 0, 0, 1] = c_p_w / C_elem

        # Output: last element temperature
        C_out = torch.zeros((n_c, 1, n), dtype=torch.float64)
        C_out[:, 0, n - 1] = 1.0
        D = torch.zeros((n_c, 1, n_inputs), dtype=torch.float64)

        return A, B, C_out, D, E, F

    def _create_state_space_model(self):
        """Create the bilinear state-space model for the fan coil unit from
        the matrices built by :meth:`_build_matrices`."""
        n = self.nelements
        A, B, C_out, D, E, F = self._build_matrices()

        x0_tensor = self._get_initial_state_tensor()  # (n_s, n_c, n_states)
        x0 = x0_tensor[0, :, :]  # (n_c, n_states)

        self.ss_model = DiscreteStatespaceSystem(
            A=A,
            B=B,
            C=C_out,
            D=D,
            x0=x0,
            state_names=[f"T_w_{i+1}" for i in range(n)],
            E=E,
            F=F,
            add_noise=False,
            id=f"ss_model_{self.id}",
        )

    def forward(self, x, inputs, params, sample_time):
        """Pure one-step FCU dynamics ``(state, inputs, params) -> (new_state, outputs)``.

        Functorch-compatible single source of truth for :meth:`do_step`.
        The internal valve's pure ``forward`` converts ``valvePosition`` to a
        water flow (its params come from the dotted ``_valve.*`` entries),
        the bilinear state-space advances the water elements, and the
        air side uses the effectiveness-NTU formulation (see the class
        docstring for the physics and limiting behavior).
        """
        # Valve: position -> water flow (delegates to the valve's own pure
        # forward -- single source of truth for the valve characteristic).
        _, v_out = self._valve.forward(
            None,
            {"valvePosition": inputs["valvePosition"]},
            {
                "waterFlowRateMax": params["_valve.waterFlowRateMax"],
                "valveAuthority": params["_valve.valveAuthority"],
            },
            sample_time,
        )
        waterFlowRate = v_out["waterFlowRate"]

        # Params-only matrices, cached per params-dict identity (rebuilt once
        # per theta in a sequential rollout, not once per step).
        cache = getattr(self, "_fwd_mat_cache", None)
        if cache is None or cache[0] is not params or cache[2] != sample_time:
            cache = (params, self._build_matrices(params), sample_time, {})
            self._fwd_mat_cache = cache
        A, B, C_out, D, E, F = cache[1]

        u = torch.stack(
            [
                inputs["supplyWaterTemperature"],
                waterFlowRate,
                inputs["inletAirTemperature"],
            ],
            dim=-1,
        )
        x_next, y = bilinear_onestep(
            A, B, C_out, D, E, F, x, u, sample_time, disc_cache=cache[3]
        )
        outletWaterTemperature = y[..., 0]

        # Air-side heat transfer using the effectiveness-NTU method (see the
        # class docstring / do_step history for the physical rationale).
        UA = params["UA"]
        T_air_in = inputs["inletAirTemperature"]
        m_dot_a = inputs["airFlowRate"]
        T_water_avg = torch.mean(x_next, dim=-1)

        # Use |m_dot_a| for the heat-capacity rate; clamp to avoid div-by-0
        # in NTU. ``tol`` is small enough to give eff ~ 1 (full equilibration)
        # while keeping ``C_air`` numerically safe.
        tol = 1e-8
        abs_m_dot_a = torch.clamp(m_dot_a.abs(), min=tol)
        C_air = abs_m_dot_a * constants.CP_AIR
        NTU = UA / C_air
        effectiveness = 1.0 - torch.exp(-NTU)  # in [0, 1]

        outletAirTemperature = T_air_in + effectiveness * (T_water_avg - T_air_in)
        # Power is the heat actually delivered to the air stream (bounded).
        # Using the original m_dot_a sign so backflow (m_dot_a < 0) keeps a
        # signed Power consistent with the airflow direction.
        sign_m_dot_a = torch.where(
            m_dot_a >= 0,
            torch.ones_like(m_dot_a),
            -torch.ones_like(m_dot_a),
        )
        Power = sign_m_dot_a * C_air * effectiveness * (T_water_avg - T_air_in)

        return x_next, {
            "outletWaterTemperature": outletWaterTemperature,
            "outletAirTemperature": outletAirTemperature,
            "waterFlowRate": waterFlowRate,
            "Power": Power,
        }

    def do_step(
        self,
        second_time=None,
        date_time=None,
        step_size=None,
        step_index: Optional[int] = None,
    ):
        """Perform one simulation step.

        Advances the water-side state-space model by one timestep, then computes:
        - Outlet water temperature (last element state)
        - Total heat transfer rate (Power)
        - Outlet air temperature from energy balance on the air side

        Args:
            second_time: Current simulation time in seconds.
            date_time: Current simulation date and time.
            step_size: Time step size in seconds.
            step_index: Current simulation step index.
        """
        # Thin port-I/O wrapper around :meth:`forward` (the single source of
        # truth for the valve characteristic, water-side dynamics and
        # effectiveness-NTU air side); the inner ``ss_model`` only carries
        # the state between steps.
        inputs = {
            "supplyWaterTemperature": self.input["supplyWaterTemperature"].get(),
            "valvePosition": self.input["valvePosition"].get(),
            "airFlowRate": self.input["airFlowRate"].get(),
            "inletAirTemperature": self.input["inletAirTemperature"].get(),
        }
        x = self.ss_model.get_state()  # (n_s, n_c, n_states)
        x_next, outs = self.forward(
            x, inputs, self._forward_params(), self._scalar_sample_time(step_size)
        )
        self.ss_model.set_state(x_next)

        self.output["outletWaterTemperature"]._set(
            outs["outletWaterTemperature"], i_t=step_index
        )
        self.output["outletAirTemperature"]._set(
            outs["outletAirTemperature"], i_t=step_index
        )
        self.output["waterFlowRate"]._set(outs["waterFlowRate"], i_t=step_index)
        self.output["Power"]._set(outs["Power"], i_t=step_index)


def brick_signature_pattern():
    """
    BRICK pattern for a Fan_Coil_Unit with an associated Space, water-flow sensor,
    and zone-air-temperature sensor.

    Topology::

        Fan_Coil_Unit  hasPart       Space
        Flow_Sensor    isPointOf     Fan_Coil_Unit   → waterFlowRate
        Temp_Sensor    isPointOf     Space           → inletAirTemperature
    """
    node0 = Node(cls=core.namespace.BRICK.Fan_Coil_Unit)
    node1 = Node(cls=core.namespace.BRICK.Space)
    node2 = Node(cls=core.namespace.BRICK.Heating_Water_Flow_Sensor)
    node3 = Node(cls=core.namespace.BRICK.Zone_Air_Temperature_Sensor)

    sp = SignaturePattern(id="fan_coil_unit_signature_pattern_brick")

    sp.add_rule(
        StepRule(subject=node0, object=node1, predicate=core.namespace.BRICK.hasPart)
    )
    sp.add_rule(
        StepRule(subject=node2, object=node0, predicate=core.namespace.BRICK.isPointOf)
    )
    sp.add_rule(
        StepRule(subject=node3, object=node1, predicate=core.namespace.BRICK.isPointOf)
    )

    sp.add_connection(node2, "measuredValue", "valvePosition")
    sp.add_connection(node3, "measuredValue", "inletAirTemperature")
    sp.add_modeled_node(node0)

    return sp


def brick_signature_pattern_vav_ahu():
    """
    BRICK pattern for a VAV-with-reheat-coil modelled as a FanCoilUnit.

    Both inlet air temperature and air flow rate come from the upstream AHU,
    which distributes supply air per branch (indexed by VAV).

    Topology::

        AHU   feeds   VAV   (requires reheat command to distinguish from plain VAV)

    Connections::

        AHU.supplyAirTemperature[vav] → FCU.inletAirTemperature
        AHU.supplyAirFlowRate[vav]    → FCU.airFlowRate
    """
    ahu = Node(cls=core.namespace.BRICK.AHU)
    vav = Node(cls=core.namespace.BRICK.VAV)
    reheat_cmd = Node(cls=core.namespace.BRICK.Command)

    sp = SignaturePattern(id="fan_coil_unit_signature_pattern_brick_vav_ahu")

    sp.add_rule(
        StepRule(subject=ahu, object=vav, predicate=core.namespace.BRICK.feeds)
    )
    sp.add_rule(
        StepRule(subject=vav, object=reheat_cmd, predicate=core.namespace.BRICK.hasPoint)
    )

    sp.add_connection(ahu, "supplyAirTemperature", "inletAirTemperature")
    sp.add_connection(ahu, "supplyAirFlowRate", "airFlowRate", output_port_index=vav)
    # Source-side port is ``inputSignal``: when ``ControllerIdentificationPI
    # TorchSystem`` is in Stage-2's ``systems_``, the controller component
    # is matched at the same ``reheat_cmd`` URI as the historised
    # ``SensorSystem`` and provides this ``inputSignal`` output, closing
    # the reheat-valve control loop natively during translation -- no
    # separate extract/wire post-process is needed.  ``output_port_index=
    # reheat_cmd`` picks the CITS actuator slot for this command (CITS.input
    # Signal is a Vector indexed by actuator).
    sp.add_connection(
        reheat_cmd,
        "inputSignal",
        "valvePosition",
        output_port_index=reheat_cmd,
    )
    sp.add_modeled_node(vav)

    return sp




# NOTE: A SAREF signature pattern was deliberately *not* registered for
# ``FanCoilUnitSystem``.  SAREF4BLDG has no ``FanCoilUnit`` class --
# the closest concept is ``S4BLDG.SpaceHeater`` -- and any pattern keyed
# on ``SpaceHeater`` + ``Valve`` + ``BuildingSpace`` would be
# structurally identical to ``SpaceHeaterSystem.saref_signature_pattern``.
# Both classes would then claim the same entity in every SAREF model, the
# MILP would have no way to tell them apart, and the alphabetical
# tie-breaker would silently route plain radiator/space-heater models
# through the FCU physics (forced-air bilinear coil, ``Q_flow_nominal``,
# ``T_w_supply_nominal``, ``inletAirTemperature``, ...) instead of the
# quasi-static space-heater physics that ``S4BLDG.SpaceHeater`` is
# intended to represent.  FCU therefore only registers BRICK patterns:
# BRICK has a dedicated ``Fan_Coil_Unit`` class and the VAV/AHU
# topology, both of which carry the air-side wiring that genuinely
# distinguishes an FCU from a radiator.
FanCoilUnitSystem.add_signature_pattern(brick_signature_pattern())
FanCoilUnitSystem.add_signature_pattern(brick_signature_pattern_vav_ahu())

# Deprecated aliases (removed in twin4build 2.1)
FanCoilUnitTorchSystem = FanCoilUnitSystem
