"""Example signature patterns for :mod:`twin4build.systems.valve.valve_system`.

Moved out of the system module (#200); examples of a graph shape, not a
standard.  Bound to their classes by :mod:`twin4build.examples.patterns`.
"""

from twin4build.translator.translator import (
    StepRule,
    AnyPathRule,
    Node,
    Predicate,
    NoStepRule,
    ModeledNode,
    OptionalRule,
    SignaturePattern,
    PathRule,
)
import twin4build.core as core


def saref_signature_pattern():
    """
    Get the SAREF signature pattern of the valve component.

    Returns:
        SignaturePattern: The SAREF signature pattern of the valve component.
    """
    node0 = Node(cls=core.namespace.S4BLDG.Valve)  # supply valve
    node1 = Node(cls=core.namespace.S4BLDG.Controller)
    node2 = Node(cls=core.namespace.SAREF.OpeningPosition)
    sp = SignaturePattern()

    sp.add_rule(
        StepRule(subject=node1, object=node2, predicate=core.namespace.SAREF.controls)
    )
    sp.add_rule(
        StepRule(
            subject=node2, object=node0, predicate=core.namespace.SAREF.isPropertyOf
        )
    )

    sp.add_input("valvePosition", node1, "inputSignal")
    sp.add_modeled_node(node0)

    return sp


def brick_signature_pattern():
    """
    Get the BRICK signature pattern of the valve component.

    Returns:
        SignaturePattern: The BRICK signature pattern of the valve component.
    """
    node0 = Node(cls=core.namespace.BRICK.Valve)
    node1 = Node(cls=core.namespace.BRICK.Valve_Position_Setpoint)
    node2 = Node(cls=core.namespace.BRICK.Water_Flow_Sensor)

    sp = SignaturePattern(id="valve_signature_pattern_brick")

    sp.add_rule(
        StepRule(subject=node1, object=node0, predicate=core.namespace.BRICK.isPointOf)
    )
    sp.add_rule(
        StepRule(subject=node2, object=node0, predicate=core.namespace.BRICK.isPointOf)
    )

    sp.add_input("valvePosition", node1, "setpoint")
    sp.add_modeled_node(node0)

    return sp


_SPACE_HEATER_CLASSES = (
    core.namespace.BRICK.Space_Heater,
    core.namespace.BRICK.Radiator,
    core.namespace.BRICK.Radiant_Panel,
    core.namespace.BRICK.Baseboard_Radiator,
)


_ROOM_CLASSES = (
    core.namespace.BRICK.Room,
    core.namespace.BRICK.HVAC_Zone,
    core.namespace.BRICK.Enclosed_space,
    core.namespace.BRICK.Open_space,
    core.namespace.REC.Room,
    core.namespace.REC.Zone,
    core.namespace.BRICK.Space,
)


def _heating_command_valve_pattern(sp_id: str, explicit_equipment: bool):
    """A radiator valve modelled on its ``brick:Heating_Command``.

    BMS graphs carry the radiator valve as a single command point in
    percent, with no ``brick:Valve`` equipment, no water-flow sensor and no
    position feedback (Hoeje-Taastrup Raadhus: 113 ``Heating_Command``
    points, no valve position sensor anywhere), so
    :func:`brick_signature_pattern` cannot match.  Two shapes occur::

        Space_Heater  feeds     Room                (explicit equipment)
        Space_Heater  hasPoint  Heating_Command  -> valvePosition

        Room          hasPoint  Heating_Command  -> valvePosition
                                                    (no equipment node)

    ``valvePosition`` is read from the command's ``inputSignal`` -- the port
    a controller identified at that URI produces
    (``ControllerIdentificationPISystem.brick_signature_pattern_space_heater
    _room``) -- so the loop closes during translation, exactly as the AHU
    pattern reads its damper commands.  The command is a normalised opening
    (percent data is scaled by the caller's ``Model.set_transformations``,
    and the controller is identified against that same 0-1 series);
    ``waterFlowRateMax`` turns it into kg/s and is estimated.  The radiator
    then reads ``waterFlowRate`` from this valve (see
    ``SpaceHeaterSystem``'s BRICK patterns): the water-side mirror of
    *controller -> damper -> AHU branch*.

    Modeled identity is the group ``[room, heating_cmd]``.  It must be a
    multi-member group: the historised command sensor
    (``get_brick_sensor_leaf_pattern``) is a singleton on the command node,
    and a singleton valve on the same node would be mutex-ed against it --
    the MILP then drops one, and whichever it drops unwires the radiator and
    the room behind it.  The group is distinct from the controller's
    (``[space_heater, heating_cmd]``) and the command sensor's
    (``[heating_cmd, externalref]``), so all three coexist on the command.
    """
    room = Node(cls=_ROOM_CLASSES)
    heating_cmd = Node(cls=core.namespace.BRICK.Heating_Command)
    sp = SignaturePattern(id=sp_id)
    if explicit_equipment:
        space_heater = Node(cls=_SPACE_HEATER_CLASSES)
        feeds = Predicate((core.namespace.BRICK.feeds, core.namespace.FSO.feedsFluidTo))
        sp.add_rule(StepRule(subject=space_heater, object=room, predicate=feeds))
        sp.add_rule(
            StepRule(
                subject=space_heater,
                object=heating_cmd,
                predicate=core.namespace.BRICK.hasPoint,
            )
        )
    else:
        sp.add_rule(
            StepRule(subject=room, object=heating_cmd, predicate=core.namespace.BRICK.hasPoint)
        )
        # Only when the radiator is *not* explicit equipment (see the
        # SpaceHeaterSystem room-command pattern for the same guard).
        sp.add_rule(
            NoStepRule(
                subject=Node(cls=_SPACE_HEATER_CLASSES),
                object=heating_cmd,
                predicate=core.namespace.BRICK.hasPoint,
            )
        )
    sp.add_connection(
        heating_cmd, "inputSignal", "valvePosition", output_port_index=heating_cmd
    )
    if explicit_equipment:
        ModeledNode([room, heating_cmd])
    else:
        # The radiator of this shape is already modelled on ``[room,
        # heating_cmd]`` (SpaceHeaterSystem's room-command pattern); the same
        # members would share its mutex bucket.  Adding the command's external
        # reference keeps the bucket distinct from that and from the command
        # sensor's ``[heating_cmd, externalref]``.
        externalref = Node(cls=(core.namespace.BRICKREF.ExternalReference, core.BlankNode))
        sp.add_rule(
            StepRule(
                subject=heating_cmd,
                object=externalref,
                predicate=core.namespace.BRICKREF.hasExternalReference,
            )
        )
        ModeledNode([room, heating_cmd, externalref])
    return sp


def brick_signature_pattern_space_heater_command():
    """See :func:`_heating_command_valve_pattern` (explicit equipment)."""
    return _heating_command_valve_pattern(
        "valve_signature_pattern_brick_space_heater_command", explicit_equipment=True
    )


def brick_signature_pattern_room_heating_command():
    """See :func:`_heating_command_valve_pattern` (command on the room)."""
    return _heating_command_valve_pattern(
        "valve_signature_pattern_brick_room_heating_command", explicit_equipment=False
    )
