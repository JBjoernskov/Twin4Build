"""
This API documentation focuses on describing the **behavior and concepts** of each module rather than implementation details.
You'll find explanations of what each component does, how it interacts with other parts of the system,
and the conceptual framework behind the functionality - not the internal code structure.
"""

# Test flag must be defined FIRST to avoid circular imports
_IS_TESTING = False
_IMPORT_COMPLETE = False

# Local application imports
from twin4build.systems.saref4syst.system import System
from twin4build.systems.saref4syst.connection import Connection
from twin4build.systems.saref4syst.connection_point import ConnectionPoint
from twin4build.model.model import Model
from twin4build.model.semantic_model.semantic_model import SemanticModel
from twin4build.model.simulation_model.simulation_model import SimulationModel
from twin4build.simulator.simulator import Simulator
from twin4build.estimator.estimator import Estimator
from twin4build.translator.translator import Translator
from twin4build.optimizer.optimizer import Optimizer, OptimizationResult
from twin4build.estimator.estimator import EstimationResult

# from twin4build.core import ontologies
import twin4build.utils.plot as plot
from twin4build.systems import *  # Note that only names in the __all__ list are imported. It is VERY important to have this import last

import twin4build.utils.types as types
from twin4build.utils.deprecation import deprecate_name

# Preferred custom-component types
Vector = types.Vector
Scalar = types.Scalar
Parameter = types.Parameter
State = types.State

_DEPRECATED_TOP_LEVEL = {
    "RewireReport": "twin4build.systems.controller.controller_identification",
    "LoopScore": "twin4build.systems.controller.controller_identification",
    "ActuatorSeeds": "twin4build.systems.controller.controller_identification",
    "score_pair": "twin4build.systems.controller.controller_identification",
    "derive_actuator_seeds": "twin4build.systems.controller.controller_identification",
    "confidence_label": "twin4build.systems.controller.controller_identification",
}

_TORCH_ALIASES = {
    "BuildingSpaceTorchSystem": "BuildingSpaceSystem",
    "BuildingSpaceMassTorchSystem": "BuildingSpaceMassSystem",
    "BuildingSpaceThermalTorchSystem": "BuildingSpaceThermalSystem",
    "WallTorchSystem": "WallSystem",
    "DamperTorchSystem": "DamperSystem",
    "ValveTorchSystem": "ValveSystem",
    "CoilTorchSystem": "CoilSystem",
    "FanTorchSystem": "FanSystem",
    "SpaceHeaterTorchSystem": "SpaceHeaterSystem",
    "FanCoilUnitTorchSystem": "FanCoilUnitSystem",
    "AirHandlingUnitTorchSystem": "AirHandlingUnitSystem",
    "OnOffControllerTorchSystem": "SmoothOnOffControllerSystem",
    "ScheduleSwitchControllerTorchSystem": "ScheduleSwitchControllerSystem",
    "SATCompensatedControllerTorchSystem": "SATCompensatedControllerSystem",
    "ControllerIdentificationTorchSystem": "ControllerIdentificationSystem",
    "ControllerIdentificationPITorchSystem": "ControllerIdentificationPISystem",
    "fmuSystem": "FmuSystem",
}

__all__ = [
    "System",
    "Connection",
    "ConnectionPoint",
    "Model",
    "SemanticModel",
    "SimulationModel",
    "Simulator",
    "Estimator",
    "EstimationResult",
    "Translator",
    "Optimizer",
    "OptimizationResult",
    "plot",
    "types",
    "Vector",
    "Scalar",
    "Parameter",
    "State",
]


def __getattr__(name: str):
    if name in _TORCH_ALIASES:
        deprecate_name(name, _TORCH_ALIASES[name])
        import twin4build.systems as _systems

        return getattr(_systems, name)
    if name in _DEPRECATED_TOP_LEVEL:
        deprecate_name(name, _DEPRECATED_TOP_LEVEL[name])
        import twin4build.systems as _systems

        return getattr(_systems, name)
    raise AttributeError(f"module 'twin4build' has no attribute {name!r}")



_IMPORT_COMPLETE = True
