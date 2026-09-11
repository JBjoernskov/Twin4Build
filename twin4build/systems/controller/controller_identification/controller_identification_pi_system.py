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
    NoStepRule,
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


def brick_signature_pattern_vav():
    """BRICK VAV pattern: hasPoint sensors/setpoints/actuator-commands directly.

    See :func:`brick_signature_pattern_vav` in
    ``controller_identification_system.py`` for the topology rationale.
    """
    vav = Node(cls=core.namespace.BRICK.VAV)
    sensors = Node(
        cls=(
            core.namespace.BRICK.Zone_Air_Temperature_Sensor,
            core.namespace.BRICK.Supply_Air_Temperature_Sensor,
            core.namespace.BRICK.Air_Flow_Sensor,
            core.namespace.BRICK.Supply_Air_Flow_Sensor,
        )
    )
    setpoints = Node(cls=core.namespace.BRICK.Zone_Air_Temperature_Setpoint)
    actuators = Node(cls=core.namespace.BRICK.Command)
    externalref = Node(cls=(core.namespace.BRICKREF.ExternalReference, core.BlankNode))
    timeseries_id = Node(cls=core.namespace.XSD.string)

    sp = SignaturePattern(id="controller_identification_pi_vav_brick")
    sp.add_rule(SetStepRule(subject=vav, object=sensors, predicate=core.namespace.BRICK.hasPoint))
    sp.add_rule(SetStepRule(subject=vav, object=setpoints, predicate=core.namespace.BRICK.hasPoint))
    sp.add_rule(SetStepRule(subject=vav, object=actuators, predicate=core.namespace.BRICK.hasPoint))
    sp.add_rule(StepRule(subject=actuators, object=externalref, predicate=core.namespace.BRICKREF.hasExternalReference))
    sp.add_rule(StepRule(subject=externalref, object=timeseries_id, predicate=core.namespace.BRICKREF.hasTimeseriesId))

    sp.add_connection(sensors, "measuredValue", "sensorValue", input_port_index=sensors)
    sp.add_connection(setpoints, "measuredValue", "setpointValue", input_port_index=setpoints)
    # Auto-mirror every setpoint into the gate-input bus.  The gate
    # selects schedule-like signals via ``gamma_gate``; mirroring all
    # setpoints by default makes the schedule available without
    # requiring extra signature-pattern rules per Brick class.  The
    # rewire pipeline never prunes ``onOffSignal`` connections, so the
    # gate's input space is preserved across rewire.
    sp.add_connection(setpoints, "measuredValue", "onOffSignal", input_port_index=setpoints)
    ModeledNode([vav, sensors, setpoints, actuators])
    return sp


ControllerIdentificationPISystem.add_signature_pattern(brick_signature_pattern_vav())


def brick_signature_pattern_vav_room():
    """BRICK VAV pattern with the loop variables on the *room* the VAV serves.

    BMS-derived graphs (Hoeje-Taastrup Raadhus) keep the zone temperature
    sensor and setpoints on the room and only the flow points and the
    damper command on the VAV::

        VAV  feeds     Room
        Room hasPoint  Zone_Air_Temperature_Sensor              -> sensorValue
        Room hasPoint  Zone_Air_Temperature_Setpoint (+ heating /
                       cooling subclasses)                      -> setpointValue
        VAV  hasPoint  Command (damper)                         -> actuator
        VAV  hasPoint  Supply_Air_Flow_Setpoint                 -> onOffSignal

    The loop identified is *damper = PI(zone temperature setpoint - zone
    temperature)*, as in :func:`brick_signature_pattern_vav`.  The VAV's
    supply-air-flow setpoint is the output of the room controller; it is
    zero whenever the zone is off and positive otherwise, so it is offered
    to the gate bus (``onOffSignal``) as the schedule-like signal, never as
    a tracked setpoint (a pattern can feed one node into a port, so the
    zone setpoints are not mirrored onto the gate bus here).
    """
    # The VAV is declared first on purpose: the matcher seeds each walk at
    # the first node of the pattern graph (``ModeledNode`` members are not
    # recognised as seeds), and seeding at the shared AHU would collapse
    # the VAV branches of one room into a single match.
    vav = Node(cls=core.namespace.BRICK.VAV)
    room = Node(
        cls=(
            core.namespace.BRICK.Room,
            core.namespace.BRICK.HVAC_Zone,
            core.namespace.BRICK.Enclosed_space,
            core.namespace.BRICK.Open_space,
            core.namespace.REC.Room,
            core.namespace.REC.Zone,
        )
    )
    sensors = Node(cls=core.namespace.BRICK.Zone_Air_Temperature_Sensor)
    setpoints = Node(cls=core.namespace.BRICK.Zone_Air_Temperature_Setpoint)
    actuators = Node(cls=core.namespace.BRICK.Command)
    flow_setpoints = Node(cls=core.namespace.BRICK.Supply_Air_Flow_Setpoint)
    externalref = Node(cls=(core.namespace.BRICKREF.ExternalReference, core.BlankNode))
    timeseries_id = Node(cls=core.namespace.XSD.string)
    feeds = Predicate((core.namespace.BRICK.feeds, core.namespace.FSO.feedsFluidTo))

    sp = SignaturePattern(id="controller_identification_pi_vav_room_brick")
    sp.add_rule(StepRule(subject=vav, object=room, predicate=feeds))
    sp.add_rule(SetStepRule(subject=room, object=sensors, predicate=core.namespace.BRICK.hasPoint))
    sp.add_rule(SetStepRule(subject=room, object=setpoints, predicate=core.namespace.BRICK.hasPoint))
    sp.add_rule(SetStepRule(subject=vav, object=actuators, predicate=core.namespace.BRICK.hasPoint))
    sp.add_rule(SetStepRule(subject=vav, object=flow_setpoints, predicate=core.namespace.BRICK.hasPoint))
    sp.add_rule(StepRule(subject=actuators, object=externalref, predicate=core.namespace.BRICKREF.hasExternalReference))
    sp.add_rule(StepRule(subject=externalref, object=timeseries_id, predicate=core.namespace.BRICKREF.hasTimeseriesId))

    sp.add_connection(sensors, "measuredValue", "sensorValue", input_port_index=sensors)
    sp.add_connection(setpoints, "measuredValue", "setpointValue", input_port_index=setpoints)
    sp.add_connection(flow_setpoints, "measuredValue", "onOffSignal", input_port_index=flow_setpoints)
    # Only the VAV and its command form the modeled identity: the room's
    # sensor / setpoint points are shared by every VAV serving that room
    # (four in some HTR rooms), and putting them in the group would make
    # those VAV controllers mutually exclusive.
    ModeledNode([vav, actuators])
    return sp


def brick_signature_pattern_space_heater_room(explicit_equipment: bool = True):
    """BRICK space-heater pattern: the thermostatic radiator valve loop.

    The heating mirror image of :func:`brick_signature_pattern_vav_room`.
    Where the VAV loop identifies *damper = PI(zone temperature setpoint -
    zone temperature)*, this one identifies *valve = PI(...)* on the same
    zone signals::

        Space_Heater  feeds         Room
        Room          hasPoint      Zone_Air_Temperature_Sensor   -> sensorValue
        Room          hasPoint      Zone_Air_Temperature_Setpoint
                                    (+ heating / cooling subclasses) -> setpointValue
        Space_Heater  hasPoint      Heating_Command                -> actuator
        Room          isFedBy       VAV
        VAV           hasPoint      Supply_Air_Flow_Setpoint       -> onOffSignal

    The gate bus carries the supply-air-flow setpoints of the VAVs serving
    the same room -- the very signal the damper loops are gated on, and the
    only schedule-like one a BMS room reliably carries as a number (the
    ``Operating_Mode_Status`` point that would read more naturally is
    text-valued in practice, so it has no numeric series to gate on).  It is
    offered to the gate bus, never as a tracked setpoint.

    Gating heating on the ventilation schedule is a hypothesis, not an
    assumption: ``alpha_gate`` is estimated, and drives the gate factor to a
    constant 1 when the signal explains nothing -- which is what a radiator
    running through an unoccupied night should produce.

    Direct vs reverse action is not fixed by the pattern: the PI subclass
    offers a reverse-acting and a direct-acting candidate and the alpha
    weights select between them, so a radiator valve (opens when the room
    is *below* setpoint) and a damper (opens when the room is *above* its
    cooling setpoint) are both reachable.

    The modeled identity is ``[space_heater, heating_cmd]``: the room's
    sensor / setpoint / mode points are shared with the VAV loops serving
    the same room, so putting them in the group would make those
    controllers mutually exclusive.

    ``explicit_equipment=False`` is the shape without a radiator node: the
    command hangs directly off the room, guarded against the equipment
    shape by a ``NoStepRule``.  Its modeled identity is ``[heating_cmd,
    timeseries_id]``, distinct from the radiator (``[room, heating_cmd]``),
    the valve (``[room, heating_cmd, externalref]``) and the command sensor
    (``[heating_cmd, externalref]``) that all bind the same command.
    """
    # Declared first on purpose: the matcher seeds each walk at the first
    # node of the pattern graph (see the VAV pattern's note).
    space_heater_classes = (
            core.namespace.BRICK.Space_Heater,
            core.namespace.BRICK.Radiator,
            core.namespace.BRICK.Radiant_Panel,
            core.namespace.BRICK.Baseboard_Radiator,
        )
    space_heater = Node(cls=space_heater_classes)
    room = Node(cls=(
            core.namespace.BRICK.Room,
            core.namespace.BRICK.HVAC_Zone,
            core.namespace.BRICK.Enclosed_space,
            core.namespace.BRICK.Open_space,
            core.namespace.REC.Room,
            core.namespace.REC.Zone,
        ))
    sensors = Node(cls=core.namespace.BRICK.Zone_Air_Temperature_Sensor)
    setpoints = Node(cls=core.namespace.BRICK.Zone_Air_Temperature_Setpoint)
    actuators = Node(cls=core.namespace.BRICK.Heating_Command)
    vavs = Node(cls=core.namespace.BRICK.VAV)
    gates = Node(cls=core.namespace.BRICK.Supply_Air_Flow_Setpoint)
    externalref = Node(cls=(core.namespace.BRICKREF.ExternalReference, core.BlankNode))
    timeseries_id = Node(cls=core.namespace.XSD.string)
    located_in = Predicate(
        (core.namespace.BRICK.feeds, core.namespace.FSO.feedsFluidTo)
    )

    suffix = "" if explicit_equipment else "_room_command"
    sp = SignaturePattern(
        id=f"controller_identification_pi_space_heater_room_brick{suffix}"
    )
    if explicit_equipment:
        sp.add_rule(StepRule(subject=space_heater, object=room, predicate=located_in))
    else:
        # No radiator node: the command hangs off the room.  ``space_heater``
        # then *is* the room for the rules below.
        space_heater = room
        sp.add_rule(
            NoStepRule(
                subject=Node(cls=space_heater_classes),
                object=actuators,
                predicate=core.namespace.BRICK.hasPoint,
            )
        )
    sp.add_rule(
        SetStepRule(subject=room, object=sensors, predicate=core.namespace.BRICK.hasPoint)
    )
    sp.add_rule(
        SetStepRule(subject=room, object=setpoints, predicate=core.namespace.BRICK.hasPoint)
    )
    sp.add_rule(
        SetStepRule(
            subject=space_heater, object=actuators, predicate=core.namespace.BRICK.hasPoint
        )
    )
    # The VAVs serving this room, through the materialised inverse of
    # ``feeds``, and their flow setpoints.
    sp.add_rule(
        SetStepRule(subject=room, object=vavs, predicate=core.namespace.BRICK.isFedBy)
    )
    sp.add_rule(
        SetStepRule(subject=vavs, object=gates, predicate=core.namespace.BRICK.hasPoint)
    )
    sp.add_rule(
        StepRule(
            subject=actuators,
            object=externalref,
            predicate=core.namespace.BRICKREF.hasExternalReference,
        )
    )
    sp.add_rule(
        StepRule(
            subject=externalref,
            object=timeseries_id,
            predicate=core.namespace.BRICKREF.hasTimeseriesId,
        )
    )
    sp.add_connection(sensors, "measuredValue", "sensorValue", input_port_index=sensors)
    sp.add_connection(
        setpoints, "measuredValue", "setpointValue", input_port_index=setpoints
    )
    sp.add_connection(gates, "measuredValue", "onOffSignal", input_port_index=gates)
    if explicit_equipment:
        ModeledNode([space_heater, actuators])
    else:
        # Distinct from the radiator's ``[room, cmd]``, the valve's ``[room,
        # cmd, externalref]`` and the command sensor's ``[cmd, externalref]``
        # buckets on the same command.
        ModeledNode([actuators, timeseries_id])
    return sp


ControllerIdentificationPISystem.add_signature_pattern(brick_signature_pattern_vav_room())
ControllerIdentificationPISystem.add_signature_pattern(
    brick_signature_pattern_space_heater_room(explicit_equipment=True)
)
ControllerIdentificationPISystem.add_signature_pattern(
    brick_signature_pattern_space_heater_room(explicit_equipment=False)
)


def brick_signature_pattern_vav_damper():
    """BRICK VAV pattern with damper-equipment-modeled actuator command.

    See :func:`brick_signature_pattern_vav_damper` in
    ``controller_identification_system.py`` for the topology rationale.
    """
    vav = Node(cls=core.namespace.BRICK.VAV)
    sensors = Node(
        cls=(
            core.namespace.BRICK.Zone_Air_Temperature_Sensor,
            core.namespace.BRICK.Supply_Air_Temperature_Sensor,
            core.namespace.BRICK.Air_Flow_Sensor,
            core.namespace.BRICK.Supply_Air_Flow_Sensor,
        )
    )
    setpoints = Node(cls=core.namespace.BRICK.Zone_Air_Temperature_Setpoint)
    damper_equip = Node(cls=core.namespace.BRICK.Damper)
    damper_cmd = Node(cls=core.namespace.BRICK.Damper_Position_Setpoint)
    externalref = Node(cls=(core.namespace.BRICKREF.ExternalReference, core.BlankNode))
    timeseries_id = Node(cls=core.namespace.XSD.string)

    sp = SignaturePattern(id="controller_identification_pi_vav_damper_brick")
    sp.add_rule(SetStepRule(subject=vav, object=sensors, predicate=core.namespace.BRICK.hasPoint))
    sp.add_rule(SetStepRule(subject=vav, object=setpoints, predicate=core.namespace.BRICK.hasPoint))
    sp.add_rule(StepRule(subject=damper_equip, object=vav, predicate=core.namespace.BRICK.isPartOf))
    sp.add_rule(SetStepRule(subject=damper_equip, object=damper_cmd, predicate=core.namespace.BRICK.hasPoint))
    sp.add_rule(StepRule(subject=damper_cmd, object=externalref, predicate=core.namespace.BRICKREF.hasExternalReference))
    sp.add_rule(StepRule(subject=externalref, object=timeseries_id, predicate=core.namespace.BRICKREF.hasTimeseriesId))

    sp.add_connection(sensors, "measuredValue", "sensorValue", input_port_index=sensors)
    sp.add_connection(setpoints, "measuredValue", "setpointValue", input_port_index=setpoints)
    # Auto-mirror every setpoint into the gate-input bus (see the
    # non-damper sibling pattern for rationale).
    sp.add_connection(setpoints, "measuredValue", "onOffSignal", input_port_index=setpoints)
    ModeledNode([vav, sensors, setpoints, damper_equip, damper_cmd])
    return sp


ControllerIdentificationPISystem.add_signature_pattern(
    brick_signature_pattern_vav_damper()
)

# Deprecated aliases (removed in twin4build 2.1)
ControllerIdentificationPITorchSystem = ControllerIdentificationPISystem
