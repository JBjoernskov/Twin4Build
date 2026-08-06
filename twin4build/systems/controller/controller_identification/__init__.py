"""Controller identification subpackage.

Public symbols:

- :class:`ControllerIdentificationSystem`: the generic, multi-candidate
  controller-identification system (kept for programmatic use; its BRICK
  signature patterns are currently disabled in favour of the PI variant).
- :class:`ControllerIdentificationPISystem`: the PI-only variant used
  by the data-driven rewire pipeline.
- :func:`rewire_pi_loops`: idempotent model-graph operation that prunes
  candidates and seeds PI parameters from observed signals.
- :class:`LoopScore`, :class:`RewireReport`, :func:`score_pair`,
  :func:`derive_actuator_seeds`, :func:`confidence_label`: building blocks
  exposed for diagnostics, testing, and custom pipelines.
"""

from twin4build.systems.controller.controller_identification.controller_identification_system import (
    ControllerIdentificationSystem,
)
from twin4build.systems.controller.controller_identification.controller_identification_pi_system import (
    ControllerIdentificationPISystem,
)
from twin4build.systems.controller.controller_identification.loop_classifier import (
    ActuatorSeeds,
    LoopScore,
    confidence_label,
    derive_actuator_seeds,
    score_pair,
)
from twin4build.systems.controller.controller_identification.pi_loop_rewire import (
    RewireReport,
    rewire_pi_loops,
)

__all__ = [
    "ControllerIdentificationSystem",
    "ControllerIdentificationPISystem",
    "rewire_pi_loops",
    "RewireReport",
    "score_pair",
    "derive_actuator_seeds",
    "confidence_label",
    "LoopScore",
    "ActuatorSeeds",
]
