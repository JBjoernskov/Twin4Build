"""Twin4Build's public API.

The primary workflow is intentionally small:

1. construct a :class:`Model` directly, or translate a :class:`SemanticModel`;
2. call ``model.load()`` and optionally ``model.to(device, dtype)``;
3. create a :class:`Simulator` and call ``simulate``;
4. pass that simulator to :class:`Estimator` or :class:`Optimizer`.

System classes and the ``Scalar``, ``Vector``, ``Parameter``, and ``State``
types are also available from this namespace. Backend-specific solver controls
belong in the ``options`` argument of ``estimate`` or ``optimize``.
"""

# Standard library imports
import importlib
from importlib.metadata import PackageNotFoundError, version

# Test flag must be defined FIRST to avoid circular imports
_IS_TESTING = False
_IMPORT_COMPLETE = False

try:
    __version__ = version("twin4build")
except PackageNotFoundError:
    __version__ = "0+unknown"

# Local application imports
import twin4build.systems as _systems
from twin4build.utils.deprecation import deprecate_name

_PUBLIC_MODULES = {
    "System": "twin4build.systems.saref4syst.system",
    "Connection": "twin4build.systems.saref4syst.connection",
    "ConnectionPoint": "twin4build.systems.saref4syst.connection_point",
    "Model": "twin4build.model.model",
    "SemanticModel": "twin4build.model.semantic_model.semantic_model",
    "SimulationModel": "twin4build.model.simulation_model.simulation_model",
    "Simulator": "twin4build.simulator.simulator",
    "Estimator": "twin4build.estimator.estimator",
    "EstimationResult": "twin4build.estimator.estimator",
    "Translator": "twin4build.translator.translator",
    "Optimizer": "twin4build.optimizer.optimizer",
    "OptimizationResult": "twin4build.optimizer.optimizer",
}

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
    "__version__",
]


def __getattr__(name: str):
    if name == "types":
        value = importlib.import_module("twin4build.utils.types")
    elif name == "plot":
        value = importlib.import_module("twin4build.utils.plot")
    elif name in {"Vector", "Scalar", "Parameter", "State"}:
        value = getattr(importlib.import_module("twin4build.utils.types"), name)
    elif name in _PUBLIC_MODULES:
        value = getattr(importlib.import_module(_PUBLIC_MODULES[name]), name)
    elif name in _systems.__all__:
        value = getattr(_systems, name)
    else:
        value = None
    if value is not None:
        globals()[name] = value
        return value
    if name in _TORCH_ALIASES:
        deprecate_name(name, _TORCH_ALIASES[name])

        return getattr(_systems, name)
    if name in _DEPRECATED_TOP_LEVEL:
        deprecate_name(name, _DEPRECATED_TOP_LEVEL[name])

        return getattr(_systems, name)
    raise AttributeError(f"module 'twin4build' has no attribute {name!r}")


_IMPORT_COMPLETE = True
