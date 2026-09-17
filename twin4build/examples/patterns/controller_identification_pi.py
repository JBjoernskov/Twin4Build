"""Example signature patterns for :mod:`twin4build.systems.controller.controller_identification.controller_identification_pi_system`.

Moved out of the system module (#200): patterns describe how one kind
of graph maps onto the component, and are examples of that, not a
standard.  Bound to their classes by :mod:`twin4build.examples.patterns`.
"""

from twin4build.translator.translator import (
    ModeledNode,
    Node,
    SetStepRule,
    Predicate,
    SignaturePattern,
    StepRule,
)
import twin4build.core as core


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


def brick_signature_pattern_space_heater_room():
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
        Room          hasPoint      Operating_Mode_Status          -> onOffSignal

    The gate bus carries the room's operating-mode status, the heating
    counterpart of the VAV's supply-air-flow setpoint: a schedule-like
    signal that says whether the loop is enabled, never a tracked setpoint.
    A building whose graph has no such point simply gets no heating loop
    here; and where the valve is in fact ungated the CITS can switch the
    gate off on its own, because ``alpha_gate`` is estimated and drives the
    gate factor to 1 when the signal carries no information.

    Direct vs reverse action is not fixed by the pattern: the PI subclass
    offers a reverse-acting and a direct-acting candidate and the alpha
    weights select between them, so a radiator valve (opens when the room
    is *below* setpoint) and a damper (opens when the room is *above* its
    cooling setpoint) are both reachable.

    The modeled identity is ``[space_heater, heating_cmd]``: the room's
    sensor / setpoint / mode points are shared with the VAV loops serving
    the same room, so putting them in the group would make those
    controllers mutually exclusive.
    """
    # Declared first on purpose: the matcher seeds each walk at the first
    # node of the pattern graph (see the VAV pattern's note).
    space_heater = Node(cls=(
            core.namespace.BRICK.Space_Heater,
            core.namespace.BRICK.Radiator,
            core.namespace.BRICK.Radiant_Panel,
            core.namespace.BRICK.Baseboard_Radiator,
        ))
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
    gates = Node(cls=core.namespace.BRICK.Operating_Mode_Status)
    externalref = Node(cls=(core.namespace.BRICKREF.ExternalReference, core.BlankNode))
    timeseries_id = Node(cls=core.namespace.XSD.string)
    located_in = Predicate(
        (core.namespace.BRICK.feeds, core.namespace.FSO.feedsFluidTo)
    )

    sp = SignaturePattern(id="controller_identification_pi_space_heater_room_brick")
    sp.add_rule(StepRule(subject=space_heater, object=room, predicate=located_in))
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
    sp.add_rule(
        SetStepRule(subject=room, object=gates, predicate=core.namespace.BRICK.hasPoint)
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
    ModeledNode([space_heater, actuators])
    return sp


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
