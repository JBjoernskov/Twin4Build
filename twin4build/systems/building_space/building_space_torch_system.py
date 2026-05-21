# Standard library imports
import datetime
from typing import Optional

# Third party imports
import torch
import torch.nn as nn

# Local application imports
import twin4build.core as core
import twin4build.utils.types as tps
from twin4build.systems.building_space.building_space_mass_torch_system import (
    BuildingSpaceMassTorchSystem,
)
from twin4build.systems.building_space.building_space_thermal_torch_system import (
    BuildingSpaceThermalTorchSystem,
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


class BuildingSpaceTorchSystem(core.System, nn.Module):
    r"""
    Combined building space model for both thermal (RC) and CO2 (mass balance) dynamics.

    This class composes BuildingSpaceThermalTorchSystem and BuildingSpaceMassTorchSystem
    to provide a unified building space model that captures both thermal and air quality
    dynamics in a building zone.

    Args:
       thermal_kwargs: Keyword arguments for BuildingSpaceThermalTorchSystem
       mass_kwargs: Keyword arguments for BuildingSpaceMassTorchSystem
       kwargs: Additional keyword arguments (must include 'id')

    Mathematical Formulation:
    =========================

       See individual component documentation:
          - BuildingSpaceThermalTorchSystem: RC network thermal dynamics
          - BuildingSpaceMassTorchSystem: CO2 mass balance dynamics

       Both models use DiscreteStatespaceSystem for efficient computation and
       automatic differentiation support.

    System Composition:

       The combined model consists of two parallel subsystems:

       **Thermal Subsystem (BuildingSpaceThermalTorchSystem):**
          - Models temperature dynamics using RC network
          - Handles heat transfer between indoor air, walls, and adjacent zones
          - Includes HVAC thermal effects, solar gains, and occupant heat gains

       **Mass Balance Subsystem (BuildingSpaceMassTorchSystem):**
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
          - boundaryTemperature, adjacentZoneTemperature

       **Combined Outputs:**
          - indoorTemperature: From thermal subsystem
          - wallTemperature: From thermal subsystem
          - indoorCO2: From mass balance subsystem
    """

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
        self.thermal = BuildingSpaceThermalTorchSystem(**thermal_kwargs)
        self.mass = BuildingSpaceMassTorchSystem(**mass_kwargs)

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
        self._config = {"parameters": thermal_parameters + mass_parameters}
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
        # _, _, n_timesteps = core.Simulator.get_simulation_timesteps(start_time, end_time, step_size)
        # batch_size = len(start_time)

        # Find if boundary temperature is set as input
        connection_point = [
            cp for cp in self.connects_at if cp.input_port == "boundaryTemperature"
        ]
        n_boundary_temperature = (
            len(connection_point[0].connects_system_through) if connection_point else 0
        )
        n_boundary_temperature = n_boundary_temperature
        assert (
            n_boundary_temperature == 0 or n_boundary_temperature == 1
        ), "Maximum one boundary temperature input is allowed"

        # Find number of adjacent zones
        connection_point = [
            cp for cp in self.connects_at if cp.input_port == "adjacentZoneTemperature"
        ]
        n_adjacent_zones = (
            len(connection_point[0].connects_system_through) if connection_point else 0
        )

        # We dont have to initialize the input and output of the combined system, because the thermal and mass systems will initialize them (copied in __init__)
        # # Initialize I/O for the combined system
        # for input in self.input.values():
        #     input.initialize(
        #         n_timesteps=n_timesteps,
        #         batch_size=batch_size,
        #     )
        # for output in self.output.values():
        #     output.initialize(
        #         n_timesteps=n_timesteps,
        #         batch_size=batch_size,
        #     )

        # self.input["adjacentZoneTemperature"].initialize(n_timesteps=n_timesteps, batch_size=batch_size, size=n_adjacent_zones)

        self.thermal.n_adjacent_zones = n_adjacent_zones
        self.thermal.n_boundary_temperature = n_boundary_temperature
        self.thermal.initialize(start_time, end_time, step_size)
        self.mass.initialize(start_time, end_time, step_size)
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


def saref_signature_pattern_sensor():
    """
    Get the signature pattern of the FMU component.

    Returns:
        SignaturePattern: The signature pattern of the FMU component.
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
    # sp.add_rule(AnyPathRule(subject=node9, object=node2, predicate=core.namespace.S4SYST.connectedTo)) # TODO: Makes _prune_recursive fail, infinite recursion

    sp.add_input("supplyAirFlowRate", node0, "airFlowRate")
    sp.add_input("exhaustAirFlowRate", node1, "airFlowRate")
    sp.add_input("heatGain", node4, "Power")
    sp.add_input("numberOfPeople", node5, "scheduleValue")
    sp.add_input("outdoorTemperature", node6, "outdoorTemperature")
    sp.add_input("outdoorCO2", node6, "outdoorCo2Concentration")
    sp.add_input("globalIrradiation", node6, "globalIrradiation")
    sp.add_input("supplyAirTemperature", node7, "measuredValue")
    # sp.add_input("adjacentZoneTemperature", node9, "indoorTemperature")

    sp.add_modeled_node(node2)
    return sp


def saref_signature_pattern():
    """
    Get the signature pattern of the FMU component.

    Returns:
        SignaturePattern: The signature pattern of the FMU component.
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
    # node9 = Node(cls=core.namespace.S4BLDG.BuildingSpace)

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
    # sp.add_rule(AnyPathRule(subject=node9, object=node2, predicate=core.namespace.S4SYST.connectedTo)) # TODO: Makes _prune_recursive fail, infinite recursion

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
    # sp.add_input("adjacentZoneTemperature", node9, "indoorTemperature")

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

    # sp.add_input("adjacentZoneTemperature", node9, "indoorTemperature")
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


BuildingSpaceTorchSystem.add_signature_pattern(brick_signature_pattern_vav())
BuildingSpaceTorchSystem.add_signature_pattern(brick_signature_pattern())
BuildingSpaceTorchSystem.add_signature_pattern(saref_signature_pattern())
BuildingSpaceTorchSystem.add_signature_pattern(saref_signature_pattern_sensor())
