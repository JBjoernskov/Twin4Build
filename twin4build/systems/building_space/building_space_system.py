# Standard library imports
import datetime
from typing import Optional

# Third party imports
import torch
import torch.nn as nn

# Local application imports
import twin4build.core as core
import twin4build.utils.types as tps
from twin4build.systems.building_space.building_space_mass_system import (
    BuildingSpaceMassSystem,
)
from twin4build.systems.building_space.building_space_thermal_system import (
    BuildingSpaceThermalSystem,
)
from twin4build.translator.translator import (
    SetStepRule,
    StepRule,
    AnyPathRule,
    Node,
    NoStepRule,
    OptionalRule,
    Predicate,
    SignaturePattern,
    PathRule,
)


class BuildingSpaceSystem(core.System, nn.Module):
    r"""
    Combined building space model for both thermal (RC) and CO2 (mass balance) dynamics.

    This class composes BuildingSpaceThermalSystem and BuildingSpaceMassSystem
    to provide a unified building space model that captures both thermal and air quality
    dynamics in a building zone.

    Args:
       thermal_kwargs: Keyword arguments for BuildingSpaceThermalSystem
       mass_kwargs: Keyword arguments for BuildingSpaceMassSystem
       kwargs: Additional keyword arguments (must include 'id')

    Mathematical Formulation
    ------------------------

       See individual component documentation:
          - BuildingSpaceThermalSystem: RC network thermal dynamics
          - BuildingSpaceMassSystem: CO2 mass balance dynamics

       Both models use DiscreteStatespaceSystem for efficient computation and
       automatic differentiation support.

    System Composition:

       The combined model consists of two parallel subsystems:

       **Thermal Subsystem (BuildingSpaceThermalSystem):**
          - Models temperature dynamics using RC network
          - Handles heat transfer between indoor air, walls, and adjacent zones
          - Includes HVAC thermal effects, solar gains, and occupant heat gains

       **Mass Balance Subsystem (BuildingSpaceMassSystem):**
          - Models CO2 concentration dynamics using mass balance equations
          - Handles ventilation, infiltration, and occupant CO2 generation
          - Tracks indoor air quality changes

    Implementation Details:

       - Both subsystems run in parallel during each simulation step
       - Input signals are shared between both models where applicable
       - Each subsystem maintains its own state variables and outputs
       - The combined model provides unified input/output interfaces
       - All parameters from both subsystems are available for calibration

    Combined Input/Output Interface:

       **Shared Inputs:**
          - supplyAirFlowRate: Used by both thermal (heating/cooling) and mass (ventilation)
          - exhaustAirFlowRate: Used by both thermal (heat removal) and mass (CO2 removal)
          - numberOfPeople: Used by both thermal (heat gain) and mass (CO2 generation)
          - outdoorTemperature: Used by thermal model
          - outdoorCO2: Used by mass balance model

       **Thermal-Only Inputs:**
          - supplyAirTemperature, globalIrradiation, heatGain
          - wallHeatGain (heat flows from connected WallSystem components)
          - boundaryTemperature (deprecated -- use WallSystem)

       **Combined Outputs:**
          - indoorTemperature: From thermal subsystem
          - wallTemperature: From thermal subsystem
          - indoorCO2: From mass balance subsystem
    """

    SUPPORTS_TRANSFORM_MODE = True

    def __init__(self, thermal_kwargs: dict = None, mass_kwargs: dict = None, **kwargs):
        """Initialize the combined building space system."""
        if thermal_kwargs is None:
            thermal_kwargs = {}
        if mass_kwargs is None:
            mass_kwargs = {}
        super().__init__(**kwargs)
        nn.Module.__init__(self)

        if "id" not in thermal_kwargs:
            assert "id" in kwargs, "id is required for thermal model"
            thermal_kwargs["id"] = kwargs["id"] + "_thermal"
        if "id" not in mass_kwargs:
            assert "id" in kwargs, "id is required for mass model"
            mass_kwargs["id"] = kwargs["id"] + "_mass"

        assert "id" in kwargs, "id is required for thermal model"
        self.thermal = BuildingSpaceThermalSystem(**thermal_kwargs)
        self.mass = BuildingSpaceMassSystem(**mass_kwargs)

        # Merge input and output dictionaries as private variables.
        #
        # ``{**a, **b}`` keeps ``b``'s entry on key collision, so for any
        # input port declared by BOTH ``thermal`` and ``mass`` (today:
        # ``supplyAirFlowRate``, ``exhaustAirFlowRate``, ``numberOfPeople``)
        # the merge silently shadows the ``thermal`` port object with the
        # ``mass`` port.  The simulator writes to ``self._input`` (the
        # parent's view) -- i.e. the ``mass`` port -- and the parallel
        # ``thermal.input[k]`` Scalar stays at its construction default
        # (0).  ``thermal.do_step`` then reads ``self.input[k].get()`` to
        # build its state-space input vector ``u``, gets back 0 for
        # ``m_sup`` and ``m_exh``, and the bilinear F-matrix term
        # ``m_sup * cp * T_sup / C_air`` that feeds supply-air enthalpy
        # into ``T_air`` contributes zero -- so the air state is never
        # convectively heated.  Rooms drift to a low equilibrium driven
        # only by solar / wall conduction.
        #
        # Fix: snap every shared key to a single port object across all
        # three dicts (parent, thermal, mass) so a single write
        # propagates to every consumer.  The earlier "forward inputs
        # in do_step" code (still preserved below as a comment) became
        # unnecessary once the merge was assumed to be alias-preserving;
        # this restores that invariant for collision keys as well.
        self._input = {**self.thermal.input, **self.mass.input}
        for k in set(self.thermal.input) & set(self.mass.input):
            shared = self._input[k]
            self.thermal.input[k] = shared
            self.mass.input[k] = shared
        # The air-flow ports are Vectors on the composite -- one slot per
        # branch (VAV) serving the zone -- and are summed into the submodels'
        # shared scalar ports in ``do_step`` / ``forward``.  A lumped zone
        # only sees the total, but every branch keeps its own damper and its
        # own calibratable nominal flow (issue #179).
        for k in self._FLOW_PORTS:
            self._input[k] = tps.Vector()
        self._output = {**self.thermal.output, **self.mass.output}
        for k in set(self.thermal.output) & set(self.mass.output):
            shared = self._output[k]
            self.thermal.output[k] = shared
            self.mass.output[k] = shared
        thermal_parameters = [
            "thermal." + s for s in self.thermal._config["parameters"]
        ]
        mass_parameters = ["mass." + s for s in self.mass._config["parameters"]]
        all_parameters = thermal_parameters + mass_parameters
        self._config = {"parameters": all_parameters}
        self.parameter = {k: {} for k in all_parameters}
        self.INITIALIZED = False

    @property
    def input(self) -> dict:
        """
        Get the input ports of the building space system.

        Returns:
            dict: Dictionary containing combined input ports from thermal and mass models
        """
        return self._input

    @property
    def output(self) -> dict:
        """
        Get the output ports of the building space system.

        Returns:
            dict: Dictionary containing combined output ports from thermal and mass models
        """
        return self._output

    def initialize(
        self,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
        step_size: int,
    ) -> None:
        """Initialize the system and its submodels."""
        is_batched = hasattr(self, "_n_c_batched") and self._n_c_batched > 1

        # Propagate batched n_c to sub-models so they allocate
        # I/O tensors with the correct parallel-component dimension.
        if is_batched:
            self.thermal._n_c_batched = self._n_c_batched
            self.mass._n_c_batched = self._n_c_batched

        if is_batched and self.thermal.manual_setup_n_walls:
            # Batched meta component: topology values were pre-set by
            # _copy_init_attrs during model compilation.  The meta
            # component's connects_at may have a different connection
            # count than the per-component topology, so skip discovery.
            pass
        else:
            # Find if boundary temperature is set as input
            connection_point = [
                cp for cp in self.connects_at if cp.input_port == "boundaryTemperature"
            ]
            n_boundary_temperature = (
                len(connection_point[0].connects_system_through)
                if connection_point
                else 0
            )
            assert (
                n_boundary_temperature == 0 or n_boundary_temperature == 1
            ), "Maximum one boundary temperature input is allowed"

            # Find number of connected walls
            connection_point = [
                cp for cp in self.connects_at if cp.input_port == "wallHeatGain"
            ]
            n_walls = (
                len(connection_point[0].connects_system_through)
                if connection_point
                else 0
            )

            self.thermal.n_walls = n_walls
            self.thermal.n_boundary_temperature = n_boundary_temperature
            # Openings to other zones (the mass zone's ``wallHeatGain``).
            cps = [cp for cp in self.connects_at if cp.input_port == "exchangeCO2Gain"]
            self.mass.n_exchanges = len(cps[0].connects_system_through) if cps else 0

        _, _, max_timesteps, _ = core.Simulator.get_simulation_timesteps(
            start_time, end_time, step_size
        )
        for k in self._FLOW_PORTS:
            self.input[k].initialize(
                n_t=max_timesteps,
                n_s=len(start_time),
                n_c=getattr(self, "n_c", 1) or 1,
                n_v=self.get_n_v_from_connections(k) or 1,
            )
        self.thermal.initialize(start_time, end_time, step_size)
        self.mass.initialize(start_time, end_time, step_size)
        # Drop the per-params routing cache (fresh graph per run, like the
        # submodels' matrix caches).
        self._fwd_param_cache = None
        self.INITIALIZED = True

    @property
    def config(self):
        """Get the system configuration."""
        return self._config

    #: Vector input ports of the composite, summed into the submodels.
    _FLOW_PORTS = ("supplyAirFlowRate", "exhaustAirFlowRate")

    def do_step(
        self,
        second_time: float,
        date_time: datetime.datetime,
        step_size: int,
        step_index: int,
    ) -> None:
        """Execute a single simulation step for both submodels.

        ``self.input`` / ``self.output`` share port objects with
        ``self.thermal.{input,output}`` and ``self.mass.{input,output}``
        for every key, including the ones that exist in both submodels
        (``supplyAirFlowRate``, ``exhaustAirFlowRate``, ``numberOfPeople``).
        The aliasing is set up in ``__init__`` after the dict-merge:
        for collision keys the merge alone would keep only the ``mass``
        port, leaving ``thermal.input[k]`` pointing at an orphan Scalar
        that the simulator never writes -- so we explicitly snap all
        three dicts to a single shared port per name there.

        Consequence here: ``Simulator._assign_component_inputs`` writes
        once to ``self.input[k]`` and both submodels read the same value
        via ``self.{thermal,mass}.input[k].get()``.  No per-step
        forwarding loop is needed; the older code that copied
        ``self.input -> thermal.input -> mass.input`` step-by-step was
        only correct *because* it bypassed the aliasing problem, and is
        redundant once the aliases are guaranteed.
        """
        for k in self._FLOW_PORTS:
            # The submodels share one scalar port per flow: the total.
            self.thermal.input[k].set(self.input[k].get().sum(dim=-1), step_index)
        self.thermal.do_step(second_time, date_time, step_size, step_index=step_index)
        self.mass.do_step(second_time, date_time, step_size, step_index=step_index)

    # State (thermal | mass) is discovered generically by System.get_state /
    # set_state via the owned submodels' ``tps.State`` -- no per-component code.

    #: Fusable coupling ports (see FusedStateSpaceSystem): delegated to the
    #: thermal submodel, which owns the wall coupling.
    FUSABLE_INPUT_PORTS = frozenset({"wallHeatGain", "exchangeCO2Gain"})
    FUSABLE_OUTPUT_PORTS = frozenset({"indoorTemperature", "indoorCO2"})

    def _ss_units(self):
        """State-space leaf units in state order (``thermal`` then ``mass`` --
        the order :meth:`System.get_state` concatenates)."""
        return [("thermal", self.thermal), ("mass", self.mass)]

    @staticmethod
    def _resolve_sub_params(sub, prefix, params):
        """Full physical-parameter dict for a submodel: estimated values from
        ``params`` (keyed ``"<prefix>.<name>"``), the rest from the submodel's own
        ``tps.Parameter`` defaults."""
        out = {}
        for name in sub.PARAM_NAMES:
            key = f"{prefix}.{name}"
            out[name] = params[key] if key in params else getattr(sub, name).get()
        return out

    def forward(self, x, inputs, params, sample_time, transform_mode=None):
        """Pure one-step of the composite = thermal ++ mass.

        State is ``[thermal_state | mass_state]`` (the order
        :meth:`System.get_state` produces).  ``params`` is keyed by the composite
        attr path (``"thermal.C_air"``, ``"mass.V"``, ...); it is routed to the two
        submodels, filling non-estimated entries from their defaults.  Both
        submodels read the shared ``inputs`` dict (they pick the ports they need).

        Returns ``(x_next, {**thermal_outputs, **mass_outputs})`` -- i.e.
        ``indoorTemperature``, ``wallTemperature``, ``indoorCO2``.
        """
        n_th = self.thermal.state_size()
        x_th, x_ma = x[..., :n_th], x[..., n_th:]
        # Vector flow ports carry one trailing branch axis more than the
        # scalar ports; sum it away for the submodels.
        ref_dim = inputs["outdoorTemperature"].dim()
        inputs = dict(inputs)
        for k in self._FLOW_PORTS:
            v = inputs.get(k)
            if v is not None and v.dim() > ref_dim:
                inputs[k] = v.sum(dim=-1)
        # Identity-keyed cache: a sequential rollout re-calls forward with the
        # SAME params dict every step (see OneStepComposer._params_for), so
        # the sub-param routing -- and, downstream, the submodels' state-space
        # matrix builds -- are theta-only work that can be done once per theta.
        if transform_mode:
            p_th = self._resolve_sub_params(self.thermal, "thermal", params)
            p_ma = self._resolve_sub_params(self.mass, "mass", params)
        else:
            cache = getattr(self, "_fwd_param_cache", None)
            if cache is None or cache[0] is not params:
                cache = (
                    params,
                    self._resolve_sub_params(self.thermal, "thermal", params),
                    self._resolve_sub_params(self.mass, "mass", params),
                )
                self._fwd_param_cache = cache
            _, p_th, p_ma = cache
        x_th_n, out_th = self.thermal.forward(
            x_th, inputs, p_th, sample_time, transform_mode=transform_mode
        )
        x_ma_n, out_ma = self.mass.forward(
            x_ma, inputs, p_ma, sample_time, transform_mode=transform_mode
        )
        return torch.cat([x_th_n, x_ma_n], dim=-1), {**out_th, **out_ma}


# Deprecated aliases (removed in twin4build 2.1)
BuildingSpaceTorchSystem = BuildingSpaceSystem
