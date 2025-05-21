"""Building Space System Module.

This module implements a combined building space model that handles both thermal dynamics
and CO2 mass balance calculations through composition of specialized submodels.

The system is composed of:
    - :class:`BuildingSpaceThermalTorchSystem`: Handles thermal dynamics using a state-space RC network model
    - :class:`BuildingSpaceMassTorchSystem`: Handles CO2 concentration using a mass balance model

See the respective submodel classes for detailed mathematical formulations.
"""

import torch
import torch.nn as nn
import twin4build.core as core
import twin4build.utils.input_output_types as tps
from twin4build.systems.building_space.building_space_thermal_torch_system import BuildingSpaceThermalTorchSystem
from twin4build.systems.building_space.building_space_mass_torch_system import BuildingSpaceMassTorchSystem
import datetime
from typing import Optional
from twin4build.translator.translator import SignaturePattern, Node, Exact, SinglePath



def get_signature_pattern():
    """
    Get the signature pattern of the FMU component.

    Returns:
        SignaturePattern: The signature pattern of the FMU component.
    """

    node0 = Node(cls=core.S4BLDG.Damper) #supply damper
    node1 = Node(cls=core.S4BLDG.Damper) #return damper
    node2 = Node(cls=core.S4BLDG.BuildingSpace)
    node3 = Node(cls=core.S4BLDG.Valve) #supply valve
    node4 = Node(cls=core.S4BLDG.SpaceHeater)
    node5 = Node(cls=core.S4BLDG.Schedule)
    node6 = Node(cls=core.S4BLDG.OutdoorEnvironment)
    node7 = Node(cls=(core.S4BLDG.Coil, core.S4BLDG.AirToAirHeatRecovery, core.S4BLDG.Fan))
    node9 = Node(cls=core.S4BLDG.BuildingSpace)
    node10 = Node(cls=core.S4BLDG.BuildingSpace)
    node11 = Node(cls=core.S4BLDG.BuildingSpace)
    node12 = Node(cls=core.S4BLDG.BuildingSpace)
    node13 = Node(cls=core.S4BLDG.BuildingSpace)
    node14 = Node(cls=core.S4BLDG.BuildingSpace)
    node15 = Node(cls=core.S4BLDG.BuildingSpace)
    node16 = Node(cls=core.S4BLDG.BuildingSpace)
    node17 = Node(cls=core.S4BLDG.BuildingSpace)
    node18 = Node(cls=core.S4BLDG.BuildingSpace)
    node19 = Node(cls=core.S4BLDG.BuildingSpace)
    sp = SignaturePattern(semantic_model_=core.ontologies, ownedBy="BuildingSpace11AdjBoundaryOutdoorFMUSystem", priority=510)

    sp.add_triple(Exact(subject=node0, object=node2, predicate=core.FSO.suppliesFluidTo))
    sp.add_triple(Exact(subject=node1, object=node2, predicate=core.FSO.hasFluidReturnedBy))
    sp.add_triple(Exact(subject=node3, object=node2, predicate=core.S4BLDG.isContainedIn))
    sp.add_triple(Exact(subject=node4, object=node2, predicate=core.S4BLDG.isContainedIn))
    sp.add_triple(Exact(subject=node3, object=node4, predicate=core.FSO.suppliesFluidTo))
    sp.add_triple(Exact(subject=node2, object=node5, predicate=core.SAREF.hasProfile))
    sp.add_triple(Exact(subject=node2, object=node6, predicate=core.S4SYST.connectedTo))
    sp.add_triple(SinglePath(subject=node0, object=node7, predicate=core.FSO.hasFluidSuppliedBy))
    sp.add_triple(Exact(subject=node9, object=node2, predicate=core.S4SYST.connectedTo))
    sp.add_triple(Exact(subject=node10, object=node2, predicate=core.S4SYST.connectedTo))
    sp.add_triple(Exact(subject=node11, object=node2, predicate=core.S4SYST.connectedTo))
    sp.add_triple(Exact(subject=node12, object=node2, predicate=core.S4SYST.connectedTo))
    sp.add_triple(Exact(subject=node13, object=node2, predicate=core.S4SYST.connectedTo))
    sp.add_triple(Exact(subject=node14, object=node2, predicate=core.S4SYST.connectedTo))
    sp.add_triple(Exact(subject=node15, object=node2, predicate=core.S4SYST.connectedTo))
    sp.add_triple(Exact(subject=node16, object=node2, predicate=core.S4SYST.connectedTo))
    sp.add_triple(Exact(subject=node17, object=node2, predicate=core.S4SYST.connectedTo))
    sp.add_triple(Exact(subject=node18, object=node2, predicate=core.S4SYST.connectedTo))
    sp.add_triple(Exact(subject=node19, object=node2, predicate=core.S4SYST.connectedTo))

    sp.add_input("airFlowRate", node0)
    sp.add_input("waterFlowRate", node3)
    sp.add_input("numberOfPeople", node5, "scheduleValue")
    sp.add_input("outdoorTemperature", node6, "outdoorTemperature")
    sp.add_input("outdoorCo2Concentration", node6, "outdoorCo2Concentration")
    sp.add_input("globalIrradiation", node6, "globalIrradiation")
    sp.add_input("supplyAirTemperature", node7, ("outletAirTemperature", "primaryTemperatureOut", "outletAirTemperature"))
    sp.add_input("indoorTemperature_adj1", node9, "indoorTemperature")
    sp.add_input("indoorTemperature_adj2", node10, "indoorTemperature")
    sp.add_input("indoorTemperature_adj3", node11, "indoorTemperature")
    sp.add_input("indoorTemperature_adj4", node12, "indoorTemperature")
    sp.add_input("indoorTemperature_adj5", node13, "indoorTemperature")
    sp.add_input("indoorTemperature_adj6", node14, "indoorTemperature")
    sp.add_input("indoorTemperature_adj7", node15, "indoorTemperature")
    sp.add_input("indoorTemperature_adj8", node16, "indoorTemperature")
    sp.add_input("indoorTemperature_adj9", node17, "indoorTemperature")
    sp.add_input("indoorTemperature_adj10", node18, "indoorTemperature")
    sp.add_input("indoorTemperature_adj11", node19, "indoorTemperature")

    sp.add_modeled_node(node4)
    sp.add_modeled_node(node2)
    return sp

def get_signature_pattern_sensor():
    """
    Get the signature pattern of the FMU component.

    Returns:
        SignaturePattern: The signature pattern of the FMU component.
    """

    node0 = Node(cls=core.S4BLDG.Damper) #supply damper
    node1 = Node(cls=core.S4BLDG.Damper) #return damper
    node2 = Node(cls=core.S4BLDG.BuildingSpace)
    node3 = Node(cls=core.S4BLDG.Valve) #supply valve
    node4 = Node(cls=core.S4BLDG.SpaceHeater)
    node5 = Node(cls=core.S4BLDG.Schedule) #return valve
    node6 = Node(cls=core.S4BLDG.OutdoorEnvironment)
    node7 = Node(cls=core.SAREF.Sensor)
    node8 = Node(cls=core.SAREF.Temperature)
    node9 = Node(cls=core.S4BLDG.BuildingSpace)
    node10 = Node(cls=core.S4BLDG.BuildingSpace)
    node11 = Node(cls=core.S4BLDG.BuildingSpace)
    node12 = Node(cls=core.S4BLDG.BuildingSpace)
    node13 = Node(cls=core.S4BLDG.BuildingSpace)
    node14 = Node(cls=core.S4BLDG.BuildingSpace)
    node15 = Node(cls=core.S4BLDG.BuildingSpace)
    node16 = Node(cls=core.S4BLDG.BuildingSpace)
    node17 = Node(cls=core.S4BLDG.BuildingSpace)
    node18 = Node(cls=core.S4BLDG.BuildingSpace)
    node19 = Node(cls=core.S4BLDG.BuildingSpace)
    sp = SignaturePattern(semantic_model_=core.ontologies, ownedBy="BuildingSpace11AdjBoundaryOutdoorFMUSystem", priority=509)
    

    sp.add_triple(Exact(subject=node0, object=node2, predicate=core.FSO.suppliesFluidTo))
    sp.add_triple(Exact(subject=node1, object=node2, predicate=core.FSO.hasFluidReturnedBy))
    sp.add_triple(Exact(subject=node3, object=node2, predicate=core.S4BLDG.isContainedIn))
    sp.add_triple(Exact(subject=node4, object=node2, predicate=core.S4BLDG.isContainedIn))
    sp.add_triple(Exact(subject=node3, object=node4, predicate=core.FSO.suppliesFluidTo))
    sp.add_triple(Exact(subject=node2, object=node5, predicate=core.SAREF.hasProfile))
    sp.add_triple(Exact(subject=node2, object=node6, predicate=core.S4SYST.connectedTo))
    sp.add_triple(SinglePath(subject=node0, object=node7, predicate=core.FSO.hasFluidSuppliedBy))
    sp.add_triple(Exact(subject=node7, object=node8, predicate=core.SAREF.observes))
    sp.add_triple(Exact(subject=node9, object=node2, predicate=core.S4SYST.connectedTo))
    sp.add_triple(Exact(subject=node10, object=node2, predicate=core.S4SYST.connectedTo))
    sp.add_triple(Exact(subject=node11, object=node2, predicate=core.S4SYST.connectedTo))
    sp.add_triple(Exact(subject=node12, object=node2, predicate=core.S4SYST.connectedTo))
    sp.add_triple(Exact(subject=node13, object=node2, predicate=core.S4SYST.connectedTo))
    sp.add_triple(Exact(subject=node14, object=node2, predicate=core.S4SYST.connectedTo))
    sp.add_triple(Exact(subject=node15, object=node2, predicate=core.S4SYST.connectedTo))
    sp.add_triple(Exact(subject=node16, object=node2, predicate=core.S4SYST.connectedTo))
    sp.add_triple(Exact(subject=node17, object=node2, predicate=core.S4SYST.connectedTo))
    sp.add_triple(Exact(subject=node18, object=node2, predicate=core.S4SYST.connectedTo))
    sp.add_triple(Exact(subject=node19, object=node2, predicate=core.S4SYST.connectedTo))

    sp.add_input("supplyAirFlowRate", node0)
    sp.add_input("exhaustAirFlowRate", node1)
    sp.add_input("waterFlowRate", node3)
    sp.add_input("numberOfPeople", node5, "scheduleValue")
    sp.add_input("outdoorTemperature", node6, "outdoorTemperature")
    sp.add_input("outdoorCO2", node6, "outdoorCo2Concentration")
    sp.add_input("globalIrradiation", node6, "globalIrradiation")
    sp.add_input("supplyAirTemperature", node7, "measuredValue")
    sp.add_input("adjacentZoneTemperature", node9, "indoorTemperature")


    sp.add_modeled_node(node4)
    sp.add_modeled_node(node2)
    return sp


class BuildingSpaceTorchSystem(core.System, nn.Module):
    """Combined building space model for thermal and CO2 dynamics.
    
    This class composes BuildingSpaceThermalTorchSystem and BuildingSpaceMassTorchSystem
    to provide a unified interface for building space simulation. The system combines
    the inputs and outputs of both submodels while maintaining their separate functionality.
    
    For detailed mathematical formulations, refer to:
        - :class:`BuildingSpaceThermalTorchSystem` for thermal dynamics
        - :class:`BuildingSpaceMassTorchSystem` for CO2 mass balance
    
    Args:
        thermal_kwargs (dict): Configuration parameters for the thermal model
        mass_kwargs (dict): Configuration parameters for the mass balance model
        **kwargs: Additional arguments passed to the parent System class
    """
    
    def __init__(self, thermal_kwargs: dict, mass_kwargs: dict, **kwargs):
        """Initialize the combined building space system."""
        super().__init__(**kwargs)
        nn.Module.__init__(self)
        self.thermal = BuildingSpaceThermalTorchSystem(**thermal_kwargs)
        self.mass = BuildingSpaceMassTorchSystem(**mass_kwargs)
        # Merge input and output dictionaries
        self.input = {**self.thermal.input, **self.mass.input}
        self.output = {**self.thermal.output, **self.mass.output}
        thermal_parameters = ["thermal."+s for s in self.thermal._config["parameters"]]
        mass_parameters = ["mass."+s for s in self.mass._config["parameters"]]
        self._config = {"parameters": thermal_parameters+mass_parameters}
        self.INITIALIZED = False

    def initialize(self, startTime=None, endTime=None, stepSize=None, simulator=None):
        """Initialize the system and its submodels."""
        # Initialize I/O for the combined system
        for input in self.input.values():
            input.initialize(startTime=startTime, endTime=endTime, stepSize=stepSize, simulator=simulator)
        for output in self.output.values():
            output.initialize(startTime=startTime, endTime=endTime, stepSize=stepSize, simulator=simulator)
        self.thermal.initialize(startTime, endTime, stepSize, simulator)
        self.mass.initialize(startTime, endTime, stepSize, simulator)
        self.INITIALIZED = True

    @property
    def config(self):
        """Get the system configuration."""
        return self._config

    def cache(self, startTime=None, endTime=None, stepSize=None):
        """Cache simulation data for both submodels."""
        self.thermal.cache(startTime, endTime, stepSize)
        self.mass.cache(startTime, endTime, stepSize)

    def do_step(self, secondTime: Optional[float] = None, dateTime: Optional[datetime.datetime] = None, 
                stepSize: Optional[float] = None, stepIndex: Optional[int] = None) -> None:
        """Execute a single simulation step for both submodels."""
        # Set inputs for thermal submodel
        for k in self.thermal.input:
            self.thermal.input[k].set(self.input[k].get(), stepIndex)
        # Set inputs for mass submodel
        for k in self.mass.input:
            self.mass.input[k].set(self.input[k].get(), stepIndex)
        self.thermal.do_step(secondTime, dateTime, stepSize, stepIndex)
        self.mass.do_step(secondTime, dateTime, stepSize, stepIndex)
        # Update outputs from both submodels
        for k in self.thermal.output:
            self.output[k].set(self.thermal.output[k].get(), stepIndex)
        for k in self.mass.output:
            self.output[k].set(self.mass.output[k].get(), stepIndex)