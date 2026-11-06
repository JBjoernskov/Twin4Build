# Third party imports
import torch
from scipy.optimize import fsolve

# Local application imports
import twin4build.utils.constants as constants
import twin4build.utils.types as tps
from twin4build import core
from twin4build.systems.saref4syst.declarations import Child, Input, Output, Parameter, Setting, State
from twin4build.systems.utils.discrete_statespace_system import bilinear_onestep
from twin4build.systems.valve.valve_system import ValveSystem


def _ua_at_nominal(unit) -> float:
    """``UA``'s default: the UA at which the coil meets its nominal sizing
    (:meth:`FanCoilUnitSystem.solve_UA`)."""
    return unit.solve_UA()


class FanCoilUnitSystem(core.System):
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
        UA: Overall heat transfer coefficient [W/K]; by default the UA at which
            the coil delivers its nominal power at its nominal temperatures
            (:meth:`solve_UA`), computed when the unit is built
        valve: The water valve's values, ``valve=dict(waterFlowRateMax=...,
            valveAuthority=...)``

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

    #: The valve's values went to the valve (removed in 2.1).
    DEPRECATED_KEYWORDS = {
        "waterFlowRateMax": "valve.waterFlowRateMax",
        "valveAuthority": "valve.valveAuthority",
    }
    REMOVED_ATTRIBUTES = {
        "initialize_UA": "UA starts at the nominal solve; give UA= to start elsewhere, "
        "or unit.UA = unit.solve_UA() to solve again after a nominal changed",
        "_valve": "the valve is the child unit.valve",
    }

    Q_flow_nominal = Setting(2000.0, doc="Nominal heat output [W]")
    T_w_supply_nominal = Setting(60.0, doc="Nominal supply water temperature [degC]")
    T_w_return_nominal = Setting(45.0, doc="Nominal return water temperature [degC]")
    T_air_in_nominal = Setting(21.0, doc="Nominal inlet air temperature [degC]")
    nelements = Setting(3, doc="Number of elements along the water path")
    supplyWaterTemperature = Input(doc="Supply water temperature [degC]")
    valvePosition = Input(doc="Water valve opening (0-1)")
    airFlowRate = Input(doc="Air flow rate [kg/s]")
    inletAirTemperature = Input(doc="Inlet air temperature [degC]")
    elements = State(size=nelements, init=21.0, doc="Element water temperatures [degC], along the flow")
    outletWaterTemperature = Output(init=elements, doc="Outlet water temperature [degC]")
    outletAirTemperature = Output(init=21.0, doc="Outlet air temperature [degC]")
    waterFlowRate = Output(doc="Water flow rate [kg/s]")
    Power = Output(doc="Heat transfer rate to the air [W] (positive heats)")
    # A copper coil and its water, small fan-coil to large reheat unit.
    thermalMassHeatCapacity = Parameter(
        100000.0, lb=5e3, ub=5e5, scaling="log", doc="Water-side thermal mass heat capacity [J/K]"
    )
    # VAV reheat coil ~ 50 - 500 W/K, dedicated FCU coil up to ~ 2 kW/K; below
    # 20 W/K the coil cannot heat at all.
    UA = Parameter(_ua_at_nominal, lb=20.0, ub=2000.0, doc="Overall heat transfer coefficient [W/K]; default: the nominal solve")
    valve = Child(ValveSystem)

    def _ua_residual(self, UA_candidate):
        """Calculate the residual for UA optimization.

        Builds a steady-state linear system (with bilinear terms collapsed at nominal
        flow) and checks that the resulting heat output matches ``Q_flow_nominal``.

        Args:
            UA_candidate: Candidate UA value to evaluate.

        Returns:
            Difference between calculated and nominal heat output.
        """
        # CPU + plain floats ON PURPOSE -- see the note in
        # ``SpaceHeaterSystem._ua_residual``: SciPy's ``fsolve`` runs this
        # through ``numpy.asanyarray``, and under ``Model.to("cuda")``
        # component initialization happens inside ``with
        # torch.device(self.device)``, so tensors built here would land on the
        # GPU and fail to convert.
        n = self.nelements
        cpu = torch.device("cpu")
        C_elem = float(self.thermalMassHeatCapacity.get().item()) / n
        UA_elem = float(UA_candidate.item()) / n
        q_nom = float(self.Q_flow_nominal)
        t_w_sup = float(self.T_w_supply_nominal)
        t_w_ret = float(self.T_w_return_nominal)
        t_air_in = float(self.T_air_in_nominal)
        c_p_w = float(constants.CP_WATER)
        m_dot_w = abs(q_nom) / (c_p_w * abs(t_w_sup - t_w_ret))

        # Build steady-state A, B (bilinear terms collapsed at nominal flow)
        # Input vector: [T_w_supply, m_dot_w, T_air_in]
        A = torch.zeros((n, n), dtype=tps.float_dtype(), device=cpu)
        B = torch.zeros((n, 3), dtype=tps.float_dtype(), device=cpu)
        for i in range(n):
            A[i, i] = -(m_dot_w * c_p_w + UA_elem) / C_elem
            if i > 0:
                A[i, i - 1] = (m_dot_w * c_p_w) / C_elem
        B[0, 0] = (m_dot_w * c_p_w) / C_elem
        for i in range(n):
            B[i, 2] = UA_elem / C_elem

        u = torch.tensor(
            [t_w_sup, m_dot_w, t_air_in], dtype=tps.float_dtype(), device=cpu
        )
        try:
            x_ss = -torch.linalg.solve(A, B @ u)
        except Exception:
            return 1e6
        # Return a plain float so the residual is device-agnostic to fsolve.
        power = UA_elem * float(torch.sum(x_ss - t_air_in).item())
        return power - q_nom

    def _build_matrices(self, p=None):
        """Build the coil state-space matrices ``(A, B, C, D, E, F)`` from the
        physical parameters -- a pure function of ``p`` (a dict of physical
        values for :attr:`PARAM_NAMES`; defaults to the component's own
        values).  Passing ``p`` is the functorch fast path; the structure is
        identical to ``SpaceHeaterSystem`` with the air inlet
        temperature in the zone-temperature role.
        """
        if p is None:
            p = {name: getattr(self, name).get() for name in ("thermalMassHeatCapacity", "UA")}

        n = self.nelements
        n_inputs = 3  # [supplyWaterTemperature, waterFlowRate, inletAirTemperature]

        C_elem = p["thermalMassHeatCapacity"] / n  # (n_c,)
        UA_elem = p["UA"] / n  # (n_c,)
        n_c = C_elem.shape[0]
        c_p_w = constants.CP_WATER
        # Parameters' device/dtype: _build_matrices re-runs on cache miss
        # during stepping, outside initialize()'s device context.
        dev, dt = C_elem.device, C_elem.dtype

        ua_over_c = UA_elem / C_elem
        ones_n = torch.ones((n,), dtype=dt, device=dev)
        input_basis = torch.eye(n_inputs, dtype=dt, device=dev)
        state_basis = torch.eye(n, dtype=dt, device=dev)
        input_supply = input_basis[0]
        input_flow = input_basis[1]
        input_air = input_basis[2]
        first_state = state_basis[0]
        last_state = state_basis[-1]

        A = torch.diag_embed(-ua_over_c.unsqueeze(-1) * ones_n)
        B = (
            ua_over_c.reshape(n_c, 1, 1)
            * ones_n.reshape(1, n, 1)
            * input_air.reshape(1, 1, n_inputs)
        )

        transport = -torch.eye(n, dtype=dt, device=dev)
        if n > 1:
            transport = transport + torch.diag(
                torch.ones(n - 1, dtype=dt, device=dev), diagonal=-1
            )
        E = (
            (c_p_w / C_elem).reshape(n_c, 1, 1, 1)
            * input_flow.reshape(1, n_inputs, 1, 1)
            * transport.reshape(1, 1, n, n)
        )
        F = (
            (c_p_w / C_elem).reshape(n_c, 1, 1, 1)
            * input_supply.reshape(1, n_inputs, 1, 1)
            * first_state.reshape(1, 1, n, 1)
            * input_flow.reshape(1, 1, 1, n_inputs)
        )

        C_out = last_state.reshape(1, 1, n).expand(n_c, -1, -1)
        D = torch.zeros((n_c, 1, n_inputs), dtype=dt, device=dev)

        return A, B, C_out, D, E, F

    def derive(self, p, step_size):
        """The state-space matrices, which depend on the parameters alone, and
        the valve's parameters."""
        valve = {k.partition(".")[2]: v for k, v in p.items() if k.startswith("valve.")}
        return {"matrices": self._build_matrices(p), "valve": self.valve.full_params(valve, step_size)}

    def forward(self, x, inputs, params, step_size):
        """Pure one-step FCU dynamics ``(state, inputs, params) -> (new_state, outputs)``.

        The internal valve's pure ``forward`` converts ``valvePosition`` to a
        water flow (its params come from the dotted ``valve.*`` entries),
        the bilinear state-space advances the water elements, and the
        air side uses the effectiveness-NTU formulation (see the class
        docstring for the physics and limiting behavior).
        """
        # Valve: position -> water flow (delegates to the valve's own pure
        # forward -- single source of truth for the valve characteristic).
        _, v_out = self.valve.forward(None, {"valvePosition": inputs["valvePosition"]}, params["valve"], step_size)
        waterFlowRate = v_out["waterFlowRate"]

        A, B, C_out, D, E, F = params["matrices"]

        u = torch.stack(
            [
                inputs["supplyWaterTemperature"],
                waterFlowRate,
                inputs["inletAirTemperature"],
            ],
            dim=-1,
        )
        x_next, y = bilinear_onestep(
            A,
            B,
            C_out,
            D,
            E,
            F,
            x,
            u,
            step_size,
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

    def solve_UA(self) -> float:
        """The ``UA`` that meets the coil's nominal sizing: in steady state at
        its nominal temperatures it delivers ``Q_flow_nominal``.  It is
        ``UA``'s default; the unit is not changed.  After a nominal changed,
        solve again: ``unit.UA = unit.solve_UA()``.

        Returns:
            The heat transfer coefficient [W/K].
        """
        UA0 = float(abs(self.Q_flow_nominal) / abs(self.T_w_return_nominal - self.T_air_in_nominal))
        root = fsolve(self._ua_residual, UA0, full_output=True)
        return float(root[0][0])


# Deprecated aliases (removed in twin4build 2.1)
FanCoilUnitTorchSystem = FanCoilUnitSystem
