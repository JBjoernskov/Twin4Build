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
    Exact,
    MultiPath,
    Node,
    SignaturePattern,
    SinglePath,
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

        # Merge input and output dictionaries as private variables
        self._input = {**self.thermal.input, **self.mass.input}
        self._output = {**self.thermal.output, **self.mass.output}
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
        simulator: core.Simulator,
    ) -> None:
        """Initialize the system and its submodels."""
        # Initialize I/O for the combined system
        for input in self.input.values():
            input.initialize(
                start_time=start_time,
                end_time=end_time,
                step_size=step_size,
                simulator=simulator,
            )
        for output in self.output.values():
            output.initialize(
                start_time=start_time,
                end_time=end_time,
                step_size=step_size,
                simulator=simulator,
            )

        # Find if boundary temperature is set as input
        connection_point = [
            cp for cp in self.connects_at if cp.inputPort == "boundaryTemperature"
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
            cp for cp in self.connects_at if cp.inputPort == "adjacentZoneTemperature"
        ]
        n_adjacent_zones = (
            len(connection_point[0].connects_system_through) if connection_point else 0
        )
        n_adjacent_zones = n_adjacent_zones

        self.thermal.n_adjacent_zones = n_adjacent_zones
        self.thermal.n_boundary_temperature = n_boundary_temperature
        self.thermal.initialize(start_time, end_time, step_size, simulator)
        self.mass.initialize(start_time, end_time, step_size, simulator)
        self.INITIALIZED = True

    @property
    def config(self):
        """Get the system configuration."""
        return self._config

    def do_step(
        self,
        secondTime: float,
        dateTime: datetime.datetime,
        step_size: int,
        step_index: int,
    ) -> None:
        """Execute a single simulation step for both submodels."""
        # Set inputs for thermal submodel
        for k in self.thermal.input:
            self.thermal.input[k].set(self.input[k].get(), step_index)
        # Set inputs for mass submodel
        for k in self.mass.input:
            self.mass.input[k].set(self.input[k].get(), step_index)
        self.thermal.do_step(secondTime, dateTime, step_size, step_index)
        self.mass.do_step(secondTime, dateTime, step_size, step_index)
        # Update outputs from both submodels
        for k in self.thermal.output:
            self.output[k].set(self.thermal.output[k].get(), step_index)
        for k in self.mass.output:
            self.output[k].set(self.mass.output[k].get(), step_index)




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
    node9 = Node(cls=core.namespace.S4BLDG.BuildingSpace)

    sp = SignaturePattern(
        semantic_model_=core.ontologies,
        id="building_space_signature_pattern",
    )

    sp.add_triple(
        Exact(subject=node0, object=node2, predicate=core.namespace.FSO.suppliesFluidTo)
    )
    sp.add_triple(
        Exact(
            subject=node1, object=node2, predicate=core.namespace.FSO.hasFluidReturnedBy
        )
    )
    sp.add_triple(
        Exact(
            subject=node4, object=node2, predicate=core.namespace.S4BLDG.isContainedIn
        )
    )
    sp.add_triple(
        Exact(subject=node2, object=node5, predicate=core.namespace.SAREF.hasProfile)
    )
    sp.add_triple(
        Exact(subject=node2, object=node6, predicate=core.namespace.S4SYST.connectedTo)
    )
    sp.add_triple(
        SinglePath(
            subject=node0, object=node7, predicate=core.namespace.FSO.hasFluidSuppliedBy
        )
    )
    # sp.add_triple(MultiPath(subject=node9, object=node2, predicate=core.namespace.S4SYST.connectedTo)) # TODO: Makes _prune_recursive fail, infinite recursion

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
    sp.add_input("adjacentZoneTemperature", node9, "indoorTemperature")

    sp.add_modeled_node(node2)
    return sp


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
    node5 = Node(cls=core.namespace.BRICK.CO2_Sensor)
    node6 = Node(
        cls=core.namespace.BRICK.Outside_Air_Temperature_Sensor
    )  # outdoor temperature sensor

    sp = SignaturePattern(
        semantic_model_=core.ontologies,
        id="building_space_signature_pattern_brick",
    )

    sp.add_triple(
        Exact(subject=node0, object=node2, predicate=core.namespace.BRICK.feeds)
    )
    # sp.add_triple(Exact(subject=node1, object=node2, predicate=core.namespace.BRICK.isFedBy))
    sp.add_triple(
        Exact(subject=node2, object=node3, predicate=core.namespace.BRICK.hasPart)
    )
    sp.add_triple(
        Exact(subject=node4, object=node3, predicate=core.namespace.BRICK.isPointOf)
    )
    sp.add_triple(
        Exact(subject=node5, object=node3, predicate=core.namespace.BRICK.isPointOf)
    )

    # sp.add_triple(MultiPath(subject=node9, object=node2, predicate=core.namespace.BRICK.isAdjacentTo)) # TODO: Makes _prune_recursive fail, infinite recursion

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

    # sp.add_input("adjacentZoneTemperature", node9, "indoorTemperature")

    sp.add_modeled_node(node3)
    return sp


BuildingSpaceTorchSystem.add_signature_pattern(brick_signature_pattern())
BuildingSpaceTorchSystem.add_signature_pattern(saref_signature_pattern())
