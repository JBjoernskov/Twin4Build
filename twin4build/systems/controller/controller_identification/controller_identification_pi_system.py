"""PI-only CITS variant for data-driven loop classification and parameter seeding.

This subclass of :class:`ControllerIdentificationSystem` constrains the
candidate controller pool to a single
:class:`~twin4build.systems.controller.setpoint_controller.pid_controller.pid_controller_system.PIDControllerSystem`
with ``Td`` frozen at zero.  It exists to make the identification problem clean
for the joint regression machinery in
:mod:`twin4build.systems.controller.controller_identification.loop_classifier`:
the residual bias of the ``cov(du, de)/var(de)`` slope estimator is exactly
``kp * (1 + h/(2*Ti))`` for a discrete PI controller, and is unbounded once
``Td > 0`` is admitted.

The two BRICK signature patterns (VAV with point-modeled actuator command,
VAV with damper-equipment-modeled actuator command) that previously lived on
the generic CITS class are re-registered on this PI subclass so the translator
builds PI-CITS instances directly during translation.  The matching patterns
on the generic CITS class are disabled while this subclass is the default
(see ``controller_identification_system.py``).
"""

from __future__ import annotations

# Standard library imports
from typing import Optional

# Third party imports
import torch

# Local application imports
import twin4build.core as core
from twin4build.systems.controller.controller_identification.controller_identification_system import (
    ControllerIdentificationSystem,
)
from twin4build.systems.controller.setpoint_controller.pid_controller.pid_controller_system import (
    PIDControllerSystem,
)
from twin4build.translator.translator import (
    ModeledNode,
    Node,
    SetStepRule,
    Predicate,
    SignaturePattern,
    StepRule,
)


class ControllerIdentificationPISystem(ControllerIdentificationSystem):
    """PI-only CITS.  Single PI candidate with ``Td`` frozen at zero.

    All other CITS infrastructure -- BandGate, multi-candidate gating via
    alpha/beta/gamma weights, signal routing -- is inherited unchanged.  The
    classifier-driven rewire pipeline (see
    :mod:`twin4build.systems.controller.controller_identification.pi_loop_rewire`)
    collapses ``n_sensors``/``n_setpoints`` to 1 and writes data-driven seeds
    onto the surviving PI candidate.

    Args:
        n_sensors: Number of candidate feedback sensors.  May be left ``None``
            to be inferred from connections during translation.
        n_setpoints: Number of candidate setpoint signals.  May be ``None``.
        n_actuators: Number of actuator outputs (default 1).
        **kwargs: Forwarded to :class:`ControllerIdentificationSystem`.

    Example:
        >>> cits = ControllerIdentificationPISystem(id="pi_cits")
        >>> # The classifier-driven rewire writes kp/Ti/output_min/etc.
        >>> # onto cits.candidate_0_0 once data is available.
    """

    def __init__(
        self,
        n_sensors: Optional[int] = None,
        n_setpoints: Optional[int] = None,
        n_on_off_signals: Optional[int] = None,
        n_actuators: int = 1,
        **kwargs,
    ):
        # Force a single PI candidate unless caller explicitly overrides.
        # Defaults are physically reasonable for HVAC zone loops at h ~ 600 s
        # (kp = 1 / unit-K error, Ti = 30 min); the rewire step replaces
        # these with data-driven values.
        kwargs.setdefault("setpoint_controllers", [PIDControllerSystem])
        kwargs.setdefault(
            "setpoint_controller_kwargs",
            [{"kp": 1.0, "Ti": 1800.0, "Td": 0.0, "is_reverse": False}],
        )

        super().__init__(
            n_sensors=n_sensors,
            n_setpoints=n_setpoints,
            n_on_off_signals=n_on_off_signals,
            n_actuators=n_actuators,
            **kwargs,
        )

    def _build_components(self) -> None:
        """Build candidates, freeze ``Td = 0`` and mark ``kp`` / ``Ti`` estimable.

        The base class already constructs the candidate with ``Td = 0`` from
        ``_candidate_entries[*][1]``, but we also flip ``requires_grad`` off
        defensively so callers that wire ``Td`` into their estimable-parameter
        list cannot accidentally re-introduce a derivative term.

        ``kp`` and ``Ti`` are switched *on*: :class:`PIDControllerSystem`
        creates them with ``requires_grad=False`` and
        :meth:`ControllerIdentificationSystem.get_estimable_parameters` skips
        frozen parameters, so without this ``Estimator.estimate(parameters=
        "auto")`` silently fitted only the gate parameters and left the
        gains at their rewire seeds.
        """
        super()._build_components()
        for a in range(self.n_actuators):
            for c in range(self.n_candidates):
                cand = getattr(self, f"candidate_{a}_{c}")
                if hasattr(cand, "Td"):
                    cand.Td.set(
                        torch.tensor(0.0, dtype=torch.float64), normalized=False
                    )
                    cand.Td.requires_grad = False
                for attr in ("kp", "Ti"):
                    p = getattr(cand, attr, None)
                    if p is not None and hasattr(p, "requires_grad"):
                        p.requires_grad = True


# ---------------------------------------------------------------------------
# BRICK signature patterns (moved from the generic CITS class).
# ---------------------------------------------------------------------------
#
# These are verbatim copies of ``brick_signature_pattern_vav`` and
# ``brick_signature_pattern_vav_damper`` from
# ``controller_identification_system.py``.  They are registered on the
# PI subclass so the translator produces ``ControllerIdentificationPITorch
# System`` instances when matching BRICK VAV topologies.  See the original
# functions for the topology rationale; the only change here is the class on
# which the patterns are registered.


# Deprecated aliases (removed in twin4build 2.1)
ControllerIdentificationPITorchSystem = ControllerIdentificationPISystem
