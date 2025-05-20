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