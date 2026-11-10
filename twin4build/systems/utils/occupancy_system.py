"""The people a zone's measured CO2 implies: the inverse of its CO2 balance
(ADR 0001: a declared System)."""

# Third party imports
import torch

# Local application imports
import twin4build.core as core
from twin4build.systems.air_handling_unit.air_handling_unit_core_system import resolve_sub_params
from twin4build.systems.building_space import air_balance
from twin4build.systems.building_space.building_space_mass_system import (
    ACTIVE_INPUT_SLICE,
    OCCUPANCY_SLOT_ACTIVE,
    mass_matrices,
)
from twin4build.systems.saref4syst.declarations import Child, Input, Output, Parameter, Setting, State
from twin4build.systems.utils.discrete_statespace_system import (
    _discretize_onestep,
    effective_matrices,
)
from twin4build.systems.utils.smooth_saturation import clamp

_FILE_MODE = (
    "wire indoorCo2Measured from the room's CO2 sensor (its measuredData) and damperPositionMeasured "
    "from the damper position sensors (their measuredData, one slot each)"
)


class _OccupancyMass(core.System):
    """The zone's CO2-balance parameters the inversion reads, at the paths of
    the room's (``mass.V``, ``mass.G_occ``, ``mass.m_inf``), so an estimation
    can share them with the room."""

    V = Parameter(100.0, doc="Air volume [m3]")
    G_occ = Parameter(5e-6, doc="CO2 generation per occupant [kg/s]")
    m_inf = Parameter(0.001, doc="Infiltration air flow [kg/s]")


class _OccupancyDamper(core.System):
    """A damper's characteristic, with the names and the scaling of
    ``DamperSystem``'s, so an estimation can share them with a model damper:
    ``([damper, occupancy.supply_damper], "a", 1, 1, 10, "shared")``."""

    REMOVED_ATTRIBUTES = {
        "c_tied": "give c a value to untie it (occupancy.supply_damper.c = value)",
        "set_c": "give c a value to untie it (occupancy.supply_damper.c = value)",
    }

    a = Parameter(1.0, scaling="log", doc="Shape of the characteristic [-]")
    nominalAirFlowRate = Parameter(0.001, doc="Nominal air flow rate [kg/s] (fully open)")
    c = Parameter(None, doc="Offset of the characteristic [kg/s]; absent: tied to -a")


class OccupancySystem(core.System):
    r"""The number of people that explains a zone's measured CO2.

    The exact inverse of :class:`BuildingSpaceMassSystem`'s ZOH step, through
    the zone's *own* matrices (:meth:`invert_zoh_occupancy`): an Euler
    rearrangement, an analytic ``expm1`` inverse, or a separately derived
    reduced system are all different maps than the zone step.  The measured
    CO2 arrives on ``indoorCo2Measured`` (the room's CO2 sensor's
    ``measuredData``) and the damper positions on ``damperPositionMeasured``
    (the damper position sensors of the terminals serving the room, one slot
    each); the previous CO2 sample is a one-step lag state.  It starts
    empty, so the first step of a run takes no change of CO2 and the initial
    occupancy comes from the static balance.  Inputs arrive through
    connections only: the System reads no file (ADR 0001, decision 6).

    .. note::
       Because the inverse is exact, measured-vs-predicted CO2 is
       tautological up to sensor noise, the occupancy clamp and the initial
       state: it cannot identify ``V``, ``G_occ`` or ``m_inf``.  Those act on
       the fit only through the occupancy the zone's heat balance then sees,
       and there only as the product ``Q_occ V / G_occ`` unless one factor is
       pinned.

    The children ``mass`` (``V``, ``G_occ``, ``m_inf``), ``supply_damper`` and
    ``exhaust_damper`` (``a``, ``nominalAirFlowRate``, ``c``; one per damper
    slot) are estimable and share their paths with the room's and the model
    dampers' parameters::

        ([space, occupancy], "mass.V", 65, 50, 80, "shared"),
        ([supply_damper, occupancy.supply_damper], "a", 1, 1, 10, "shared"),

    Args:
        mass: The values of the zone's CO2 balance, ``dict(V=..., G_occ=...,
            m_inf=...)``.
        supply_damper: The supply damper's characteristic,
            ``dict(a=..., nominalAirFlowRate=..., c=...)``.
        exhaust_damper: The exhaust damper's characteristic.
        exhaust_follows_supply: Mirror of ``AirHandlingUnitSystem``'s option:
            the exhaust flow is ``exhaustFlowRatio`` times the supply flow
            (share the ratio with the AHU's so inverse and forward agree).
        exhaustFlowRatio: Exhaust-to-supply flow ratio [-] (estimable).
        smoothing: First-order smoothing of the occupancy before the clamp
            at zero, ``N_f = s * N_f_prev + (1 - s) * N``: the inversion
            amplifies CO2 sensor noise by ``V / (dt * alpha)``.  ``0``
            (default): none, and no state for it.
    """

    #: The 1.x flat keywords and where their values go now.
    DEPRECATED_KEYWORDS = {
        "V": "mass.V",
        "G_occ": "mass.G_occ",
        "m_inf": "mass.m_inf",
        "supply_damper_a": "supply_damper.a",
        "supply_damper_nominalAirFlowRate": "supply_damper.nominalAirFlowRate",
        "exhaust_damper_a": "exhaust_damper.a",
        "exhaust_damper_nominalAirFlowRate": "exhaust_damper.nominalAirFlowRate",
    }

    #: The file mode of 1.x (ADR 0001, decision 6).
    REMOVED_ATTRIBUTES = {
        name: _FILE_MODE
        for name in (
            "co2_filename", "co2_date_column", "co2_value_column",
            "damper_filename", "damper_date_column", "damper_value_column",
            "co2_datecolumn", "co2_valuecolumn", "damper_datecolumn", "damper_valuecolumn",
        )
    }

    #: Fan speed (0-1) above which the fan is fully "on" (as the AHU's).
    FAN_ON_SPEED = 0.1

    exhaust_follows_supply = Setting(False, structural=True, doc="The exhaust flow is the supply flow times the ratio")
    smoothing = Setting(0.0, structural=True, doc="Smoothing factor of the occupancy (0: none)")
    outdoorCo2Concentration = Input(default=400.0, doc="Outdoor CO2 concentration [ppmv]")
    indoorCo2Measured = Input(doc="The room's measured CO2 [ppmv]")
    damperPositionMeasured = Input(size="wired", doc="Measured damper positions (0-1), one slot per terminal")
    fanSpeedMeasured = Input(default=1.0, doc="Fan state of the air handler (0-1); unwired: running")
    scheduleValue = Output(doc="The number of people")
    co2_prev = State(init=0.0, doc="The measured CO2 of the step before [ppmv]; 0: none yet")
    occupancy_prev = State(size=smoothing, init=0.0, doc="The smoothed occupancy of the step before")
    exhaustFlowRatio = Parameter(1.0, lb=0.3, ub=1.5, doc="Exhaust-to-supply flow ratio [-]")
    mass = Child(_OccupancyMass)
    supply_damper = Child(_OccupancyDamper, per_slot=damperPositionMeasured)
    exhaust_damper = Child(_OccupancyDamper, per_slot=damperPositionMeasured)

    PARAM_NAMES = (
        "mass.V",
        "mass.G_occ",
        "mass.m_inf",
        "supply_damper.a",
        "supply_damper.nominalAirFlowRate",
        "supply_damper.c",
        "exhaust_damper.a",
        "exhaust_damper.nominalAirFlowRate",
        "exhaust_damper.c",
        "exhaustFlowRatio",
    )

    _CHILDREN = ("mass", "supply_damper", "exhaust_damper")

    def _inactive_parameters(self):
        return () if self.exhaust_follows_supply else ("exhaustFlowRatio",)

    def get_estimable_parameters(self):
        entries = super().get_estimable_parameters()
        if self.exhaust_follows_supply:
            entries = [e for e in entries if not str(e[1]).startswith("exhaust_damper.")]
        else:
            entries = [e for e in entries if str(e[1]) != "exhaustFlowRatio"]
        return entries

    @staticmethod
    def _airflow(
        a: torch.Tensor, nominal: torch.Tensor, position: torch.Tensor, c=None
    ) -> torch.Tensor:
        """Damper position (0-1) -> airflow [kg/s]; the characteristic of
        ``DamperSystem`` (``c = -a`` unless given: ``a + c`` is then the
        closed-damper flow) on explicit parameter tensors so ``forward``
        stays pure."""
        from twin4build.systems.damper.damper_system import DamperSystem

        return DamperSystem.characteristic(a, nominal, position, c)

    @staticmethod
    def invert_zoh_occupancy(
        mass_params,
        inputs,
        C_prev,
        C_now,
        step_size,
        n_c=1,
        transform_mode=None,
    ):
        r"""People count that takes ``C_prev`` to ``C_now`` under the zone's own ZOH step.

        This is the algebraic inverse of
        :meth:`BuildingSpaceMassSystem.forward` -- not of a separately
        derived reduced system.  It calls the *same*
        :func:`~twin4build.systems.building_space.building_space_mass_system.mass_matrices`
        and the *same*
        :func:`~twin4build.systems.utils.discrete_statespace_system._discretize_onestep`
        on the *same* four-slot input vector, so there is one matrix
        contract, one matrix exponential and one set of shapes on the
        captured graph.  A second, hand-rolled 1-state/2-input system
        agrees numerically but drifts from the zone whenever the zone's
        matrices change, and adds an untested ``expm`` to the compiled /
        CUDA-graph rollout.

        Occupant count is slot :data:`OCCUPANCY_SLOT` of ``u`` and enters
        neither ``A_eff`` nor ``B_eff`` (``E`` and ``F`` are zero there),
        so with ``u_occ = 0``

        .. math::

           n = \frac{C_k - A_d C_{k-1} - B_d u\big|_{n=0}}{B_d[\,0, 3\,]}

        Only :data:`ACTIVE_INPUT_SLOTS` of ``B_eff`` are ever nonzero, and
        ``B_d`` is linear in ``B_eff`` column by column, so the dropped
        columns contribute exact zeros.  Discretizing the reduced system
        therefore gives bit-identical ``A_d`` and ``B_d`` entries while
        exponentiating a 3x3 block rather than a 5x5 -- nine saved
        activations per product instead of twenty-five, on a captured
        reverse-mode rollout that runs the device out of memory.

        Args:
            mass_params: ``{"V", "G_occ", "m_inf"}`` -- the zone's
                :attr:`BuildingSpaceMassSystem.PARAM_NAMES`, each ``(n_c,)``.
            inputs: the zone's ``supplyAirFlowRate`` / ``exhaustAirFlowRate``
                / ``outdoorCo2Concentration`` ports, *untransformed*: the balanced-flow
                transform is applied here exactly as the zone applies it.
            C_prev: CO2 at the start of the step [ppmv].
            C_now: measured CO2 at the end of the step [ppmv].
            step_size: step length [s].
            n_c: parallel-component count of the matrices.
            transform_mode: passed through to the discretization (``True``
                selects ``_expm_ss`` / ``_expm_ss_fused``, as under vmap and
                ``torch.compile``).

        Tensor construction follows
        ``docs/source/manual/differentiable_system_models.rst``: no
        tensor-to-Python (no ``result_type``, no ``.to(dtype=...)``), and
        ``transform_mode`` is passed explicitly rather than queried.
        """
        A, B, _, _, E, F = mass_matrices(
            mass_params["V"], mass_params["G_occ"], mass_params["m_inf"], n_c
        )
        # Same transform, same slot order as BuildingSpaceMassSystem.forward.
        inputs = {**inputs, **air_balance.balanced_flow_inputs(inputs)}
        C_out = inputs["outdoorCo2Concentration"]
        # Slot order of BuildingSpaceMassSystem.forward; the make-up stream is
        # outdoor air here (the inverse has no transfer-node port).
        u = torch.stack(
            [
                inputs["supplyAirFlowRate"],
                inputs["exhaustAirFlowRate"],
                C_out,
                torch.zeros_like(C_out),
                inputs.get("makeUpAirCO2", C_out),
            ],
            dim=-1,
        )
        # The zone's own effective matrices (the bilinear flow terms read the
        # FULL u), then only the columns that are ever nonzero.
        A_eff, B_eff = effective_matrices(A, B, E, F, u)
        # A slice, never a list of indices: see ACTIVE_INPUT_SLICE.
        u_r = u[..., ACTIVE_INPUT_SLICE]
        Ad, Bd = _discretize_onestep(
            A_eff,
            B_eff[..., ACTIVE_INPUT_SLICE],
            None,
            None,
            u_r,
            step_size,
            transform_mode=transform_mode,
        )
        # bilinear_onestep's x_next, with the occupancy slot zeroed out.
        x = C_prev.unsqueeze(-1)
        x_no_occ = (Ad @ x.unsqueeze(-1)).squeeze(-1) + (
            Bd @ u_r.unsqueeze(-1)
        ).squeeze(-1)
        return (C_now.unsqueeze(-1) - x_no_occ).squeeze(-1) / Bd[
            ..., 0, OCCUPANCY_SLOT_ACTIVE
        ]

    def derive(self, p, step_size):
        """The children's parameters (estimated values from ``p``, the rest the
        children's own), resolved once per run or rollout."""
        return {"sub_params": {n: resolve_sub_params(getattr(self, n), n, p) for n in self._CHILDREN}}

    @staticmethod
    def _on_slots(value, positions: torch.Tensor) -> torch.Tensor:
        """A damper value laid out on the slots of ``positions`` ``(rows,
        n_v)``: one value as it is, one per slot of every copy on the rows and
        the slots."""
        value = torch.as_tensor(value, dtype=positions.dtype, device=positions.device)
        if value.numel() == 1:
            return value.reshape(())
        value = value.reshape(-1, positions.shape[-1])
        if value.shape[0] != positions.shape[0]:
            value = value.repeat(positions.shape[0] // value.shape[0], 1)
        return value

    def _flow(self, P, positions: torch.Tensor) -> torch.Tensor:
        """The total flow of a damper on every slot of ``positions``: a lumped
        zone only sees the total."""
        P = {k: self._on_slots(v, positions) for k, v in P.items()}
        return self._airflow(P["a"], P["nominalAirFlowRate"], positions, P.get("c")).sum(dim=-1)

    def forward(self, x, inputs, p, step_size):
        """One step: the people that take the last measured CO2 to this one,
        through the zone's own ZOH step; the measured CO2 becomes the state."""
        P = p["sub_params"]
        C_indoor = inputs["indoorCo2Measured"]
        # The lag starts empty (0 ppm): the first step takes no change of CO2,
        # so the people come from the static balance.
        C_prev = torch.where(x[..., 0] == 0.0, C_indoor, x[..., 0])
        positions = inputs["damperPositionMeasured"]
        m_sup = self._flow(P["supply_damper"], positions)
        if self.exhaust_follows_supply:
            m_exh = m_sup * p["exhaustFlowRatio"]
        else:
            m_exh = self._flow(P["exhaust_damper"], positions)
        # Same gate as DamperSystem.fan_gate: nothing moves with the fan
        # stopped, the damper sets the flow once it runs.  A hard clamp: with
        # the port unwired (value 1) the gate is exactly 1.
        gate = torch.clamp(inputs["fanSpeedMeasured"] / self.FAN_ON_SPEED, 0.0, 1.0)
        mass = P["mass"]
        N_occ = self.invert_zoh_occupancy(
            {"V": mass["V"], "G_occ": mass["G_occ"], "m_inf": mass["m_inf"]},
            {
                "supplyAirFlowRate": m_sup * gate,
                "exhaustAirFlowRate": m_exh * gate,
                "outdoorCo2Concentration": inputs["outdoorCo2Concentration"],
            },
            C_prev,
            C_indoor,
            step_size,
            n_c=torch.as_tensor(mass["V"]).reshape(-1).shape[0],
        )
        parts = [C_indoor.unsqueeze(-1)]
        if self.smoothing:
            N_occ = self.smoothing * x[..., 1] + (1.0 - self.smoothing) * N_occ
            parts.append(N_occ.unsqueeze(-1))
        N_occ = clamp(N_occ, lower=0.0, upper=1e6)
        return torch.cat(parts, dim=-1), {"scheduleValue": N_occ}
