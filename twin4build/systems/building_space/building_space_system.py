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
        is_compiled = hasattr(self, "_n_c_compiled") and self._n_c_compiled > 1

        # Propagate compiled n_c to sub-models so they allocate
        # I/O tensors with the correct parallel-component dimension.
        if is_compiled:
            self.thermal._n_c_compiled = self._n_c_compiled
            self.mass._n_c_compiled = self._n_c_compiled

        if is_compiled and self.thermal.manual_setup_n_walls:
            # Compiled meta component: topology values were pre-set by
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
                len(connection_point[0].connects_system_through) if connection_point else 0
            )
            assert (
                n_boundary_temperature == 0 or n_boundary_temperature == 1
            ), "Maximum one boundary temperature input is allowed"

            # Find number of connected walls
            connection_point = [
                cp for cp in self.connects_at if cp.input_port == "wallHeatGain"
            ]
            n_walls = (
                len(connection_point[0].connects_system_through) if connection_point else 0
            )

            self.thermal.n_walls = n_walls
            self.thermal.n_boundary_temperature = n_boundary_temperature

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
        self.thermal.do_step(second_time, date_time, step_size, step_index=step_index)
        self.mass.do_step(second_time, date_time, step_size, step_index=step_index)

    # State (thermal | mass) is discovered generically by System.get_state /
    # set_state via the owned submodels' ``tps.State`` -- no per-component code.

    #: Fusable coupling ports (see FusedStateSpaceSystem): delegated to the
    #: thermal submodel, which owns the wall coupling.
    FUSABLE_INPUT_PORTS = frozenset({"wallHeatGain"})
    FUSABLE_OUTPUT_PORTS = frozenset({"indoorTemperature"})

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


def saref_signature_pattern_sensor():
    """
    Get the SAREF signature pattern (with supply-air temperature sensor) of the
    building space component.

    Returns:
        SignaturePattern: The signature pattern of the building space component.
    """

    node0 = Node(cls=core.namespace.S4BLDG.Damper)  # supply damper
    node1 = Node(cls=core.namespace.S4BLDG.Damper)  # return damper
    node2 = Node(cls=core.namespace.S4BLDG.BuildingSpace)
    node4 = Node(cls=core.namespace.S4BLDG.SpaceHeater)
    node5 = Node(cls=core.namespace.S4BLDG.Schedule)
    node6 = Node(cls=core.namespace.S4BLDG.OutdoorEnvironment)
    node7 = Node(cls=core.namespace.SAREF.Sensor)
    node8 = Node(cls=core.namespace.SAREF.Temperature)
    sp = SignaturePattern(
        id="building_space_signature_pattern_sensor",
    )

    sp.add_rule(
        StepRule(subject=node0, object=node2, predicate=core.namespace.FSO.suppliesFluidTo)
    )
    sp.add_rule(
        StepRule(
            subject=node1, object=node2, predicate=core.namespace.FSO.hasFluidReturnedBy
        )
    )
    sp.add_rule(
        StepRule(
            subject=node4, object=node2, predicate=core.namespace.S4BLDG.isContainedIn
        )
    )
    sp.add_rule(
        StepRule(subject=node2, object=node5, predicate=core.namespace.SAREF.hasProfile)
    )
    sp.add_rule(
        StepRule(subject=node2, object=node6, predicate=core.namespace.S4SYST.connectedTo)
    )
    sp.add_rule(
        PathRule(
            subject=node0, object=node7, predicate=core.namespace.FSO.hasFluidSuppliedBy
        )
    )
    sp.add_rule(
        StepRule(subject=node7, object=node8, predicate=core.namespace.SAREF.observes)
    )

    sp.add_input("supplyAirFlowRate", node0, "airFlowRate")
    sp.add_input("exhaustAirFlowRate", node1, "airFlowRate")
    sp.add_input("heatGain", node4, "Power")
    sp.add_input("numberOfPeople", node5, "scheduleValue")
    sp.add_input("outdoorTemperature", node6, "outdoorTemperature")
    sp.add_input("outdoorCO2", node6, "outdoorCo2Concentration")
    sp.add_input("globalIrradiation", node6, "globalIrradiation")
    sp.add_input("supplyAirTemperature", node7, "measuredValue")
    # Interzonal/boundary coupling is modeled by a separate WallSystem
    # (wired manually, or via a future wall/adjacency signature pattern).

    sp.add_modeled_node(node2)
    return sp


def saref_signature_pattern():
    """
    Get the SAREF signature pattern of the building space component.

    Returns:
        SignaturePattern: The signature pattern of the building space component.
    """

    node0 = Node(cls=core.namespace.S4BLDG.Damper)  # supply damper
    node1 = Node(cls=core.namespace.S4BLDG.Damper)  # return damper
    node2 = Node(cls=core.namespace.S4BLDG.BuildingSpace)
    node4 = Node(cls=core.namespace.S4BLDG.SpaceHeater)
    node5 = Node(cls=core.namespace.S4BLDG.Schedule)
    node6 = Node(cls=core.namespace.S4BLDG.OutdoorEnvironment)
    node7 = Node(
        cls=(
            core.namespace.S4BLDG.Coil,
            core.namespace.S4BLDG.AirToAirHeatRecovery,
            core.namespace.S4BLDG.Fan,
        )
    )

    sp = SignaturePattern(
        id="building_space_signature_pattern",
    )

    sp.add_rule(
        StepRule(subject=node0, object=node2, predicate=core.namespace.FSO.suppliesFluidTo)
    )
    sp.add_rule(
        StepRule(
            subject=node1, object=node2, predicate=core.namespace.FSO.hasFluidReturnedBy
        )
    )
    sp.add_rule(
        StepRule(
            subject=node4, object=node2, predicate=core.namespace.S4BLDG.isContainedIn
        )
    )
    sp.add_rule(
        StepRule(subject=node2, object=node5, predicate=core.namespace.SAREF.hasProfile)
    )
    sp.add_rule(
        StepRule(subject=node2, object=node6, predicate=core.namespace.S4SYST.connectedTo)
    )
    sp.add_rule(
        PathRule(
            subject=node0, object=node7, predicate=core.namespace.FSO.hasFluidSuppliedBy
        )
    )

    sp.add_input("supplyAirFlowRate", node0, "airFlowRate")
    sp.add_input("exhaustAirFlowRate", node1, "airFlowRate")
    sp.add_input("heatGain", node4, "Power")
    sp.add_input("numberOfPeople", node5, "scheduleValue")
    sp.add_input("outdoorTemperature", node6, "outdoorTemperature")
    sp.add_input("outdoorCO2", node6, "outdoorCo2Concentration")
    sp.add_input("globalIrradiation", node6, "globalIrradiation")
    sp.add_input(
        "supplyAirTemperature",
        node7,
        ("outletAirTemperature", "primaryTemperatureOut", "outletAirTemperature"),
    )
    # Interzonal/boundary coupling is modeled by a separate WallSystem
    # (wired manually, or via a future wall/adjacency signature pattern).

    sp.add_modeled_node(node2)
    return sp


def brick_signature_pattern():  # Fits to site A
    """
    Get the BRICK-only signature pattern of the building space component.

    Returns:
        SignaturePattern: The BRICK-only signature pattern of the building space component.
    """

    ahu = Node(cls=core.namespace.BRICK.AHU)
    # node1 = Node(cls=core.namespace.BRICK.Damper)
    # node2 = Node(cls=core.namespace.BRICK.Zone)  # Compatibility with both site A and B (A uses Zone and B uses HVAC_Zone)
    space = Node(
        cls=(
            core.namespace.BRICK.Room,
            core.namespace.BRICK.Enclosed_space,
            core.namespace.BRICK.Open_space,
            core.namespace.BRICK.HVAC_Zone,
            core.namespace.BOT.Space,
        )
    )  # TODO: '_space' should be '_Office', but the site b ttl file has a bug
    solar_radiance_sensor = Node(cls=core.namespace.BRICK.Global_Solar_Irradiation_Sensor)
    outside_air_temperature_sensor = Node(
        cls=core.namespace.BRICK.Outside_Air_Temperature_Sensor
    )

    vav = Node(cls=core.namespace.BRICK.VAV)

    feeds = Predicate((core.namespace.BRICK.feeds, core.namespace.FSO.feedsFluidTo))

    sp = SignaturePattern(
        id="building_space_signature_pattern_brick",
    )

    sp.add_node(solar_radiance_sensor, optional=True) # Optional because it is not always present
    sp.add_node(outside_air_temperature_sensor, optional=True) # Optional because it is not always present

    sp.add_rule(
        AnyPathRule(
            subject=ahu, object=space, predicate=feeds, endpoints_only=True
        ) & NoStepRule(subject=ahu, object=vav, predicate=feeds)
    )



    sp.add_connection(
        ahu, "supplyAirFlowRate", "supplyAirFlowRate", output_port_index=space
    )
    sp.add_connection(
        ahu, "exhaustAirFlowRate", "exhaustAirFlowRate", output_port_index=space
    )
    # # sp.add_input("numberOfPeople", node5, "measuredValue")
    sp.add_connection(
        outside_air_temperature_sensor, "outdoorTemperature", "outdoorTemperature"
    )
    sp.add_connection(
        solar_radiance_sensor, "globalIrradiation", "globalIrradiation"
    )
    sp.add_connection(ahu, "supplyAirTemperature", "supplyAirTemperature")

    # Interzonal/boundary coupling is modeled by a separate WallSystem
    # (wired manually, or via a future wall/adjacency signature pattern).
    sp.add_modeled_node(space)

    # sp_eq = SignaturePattern(
    #     id="building_space_signature_pattern_brick_eq",
    # )

    #######################

    # sp_eq.add_rule(
    #     StepRule(subject=node0, object=node2, predicate=core.namespace.BRICK.feeds)
    # )
    # sp_eq.add_rule(
    #     StepRule(subject=node2, object=node3, predicate=core.namespace.BRICK.hasPart)
    # )

    # # TODO: How to handle inverse predicates?
    # diff = core.Diff()
    # diff.remove(node0, core.namespace.BRICK.feeds, node2)
    # diff.remove(node2, core.namespace.BRICK.isFedBy, node0)
    # diff.add(node2, core.namespace.BRICK.hasPart, node3)

    # sp.add_equivalent(sp_eq, diff)

    return sp


def brick_signature_pattern_vav():
    """
    BRICK signature pattern for a building space served by an AHU via a VAV/FCU.

    Mirrors physical reality: AHU → VAV/FCU (adds reheat) → Room.
    The supply air temperature entering the room comes from the FCU outlet,
    not directly from the AHU.

    Topology::

        AHU  feeds  VAV  feeds  Room

    Connections:
        AHU.supplyAirFlowRate  → BuildingSpace.supplyAirFlowRate
        AHU.exhaustAirFlowRate → BuildingSpace.exhaustAirFlowRate
        VAV.outletAirTemperature → BuildingSpace.supplyAirTemperature

    The MILP solver will prefer this pattern over the direct AHU pattern when
    a VAV is present between the AHU and the room, because it covers more nodes.
    """
    ahu = Node(cls=core.namespace.BRICK.AHU)
    vav = Node(cls=core.namespace.BRICK.VAV)
    space = Node(
        cls=(
            core.namespace.BRICK.Room,
            core.namespace.BRICK.Enclosed_space,
            core.namespace.BRICK.Open_space,
            core.namespace.BRICK.HVAC_Zone,
            core.namespace.BOT.Space,
        )
    )
    solar_radiance_sensor = Node(cls=core.namespace.BRICK.Global_Solar_Irradiation_Sensor )
    outside_air_temperature_sensor = Node(
        cls=core.namespace.BRICK.Outside_Air_Temperature_Sensor
    )

    feeds = Predicate((core.namespace.BRICK.feeds, core.namespace.FSO.feedsFluidTo))

    sp = SignaturePattern(id="building_space_signature_pattern_brick_vav")

    sp.add_node(solar_radiance_sensor, optional=True) # Optional because it is not always present
    sp.add_node(outside_air_temperature_sensor, optional=True) # Optional because it is not always present

    sp.add_rule(StepRule(subject=ahu, object=vav, predicate=feeds))
    sp.add_rule(StepRule(subject=vav, object=space, predicate=feeds))

    sp.add_connection(
        ahu, "supplyAirFlowRate", "supplyAirFlowRate", output_port_index=space
    )
    sp.add_connection(
        ahu, "exhaustAirFlowRate", "exhaustAirFlowRate", output_port_index=space
    )
    sp.add_connection(vav, "outletAirTemperature", "supplyAirTemperature")
    sp.add_connection(solar_radiance_sensor, "globalIrradiation", "globalIrradiation")
    sp.add_connection(outside_air_temperature_sensor, "outdoorTemperature", "outdoorTemperature")
    sp.add_modeled_node(space)

    return sp


BuildingSpaceSystem.add_signature_pattern(brick_signature_pattern_vav())
BuildingSpaceSystem.add_signature_pattern(brick_signature_pattern())
BuildingSpaceSystem.add_signature_pattern(saref_signature_pattern())
BuildingSpaceSystem.add_signature_pattern(saref_signature_pattern_sensor())

# Deprecated aliases (removed in twin4build 2.1)
BuildingSpaceTorchSystem = BuildingSpaceSystem
