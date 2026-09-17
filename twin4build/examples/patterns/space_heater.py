"""Example signature patterns for :mod:`twin4build.systems.space_heater.space_heater_system`.

Moved out of the system module (#200): patterns describe how one kind
of graph maps onto the component, and are examples of that, not a
standard.  Bound to their classes by :mod:`twin4build.examples.patterns`.
"""

from twin4build import core
from twin4build.translator.translator import (
    StepRule,
    AnyPathRule,
    Node,
    NoStepRule,
    Predicate,
    ModeledNode,
    OptionalRule,
    SignaturePattern,
    PathRule,
)


def saref_signature_pattern():
    """
    Get the SAREF signature pattern of the space heater component.

    Returns:
        SignaturePattern: The SAREF signature pattern of the space heater component.
    """

    node2 = Node(cls=core.namespace.S4BLDG.BuildingSpace)
    node3 = Node(cls=core.namespace.S4BLDG.Valve)  # supply valve
    node4 = Node(cls=core.namespace.S4BLDG.SpaceHeater)
    sp = SignaturePattern(
        id="space_heater_signature_pattern",
    )

    sp.add_rule(
        StepRule(
            subject=node3, object=node2, predicate=core.namespace.S4BLDG.isContainedIn
        )
    )
    sp.add_rule(
        StepRule(
            subject=node4, object=node2, predicate=core.namespace.S4BLDG.isContainedIn
        )
    )
    sp.add_rule(
        StepRule(
            subject=node3, object=node4, predicate=core.namespace.FSO.suppliesFluidTo
        )
    )

    sp.add_input("waterFlowRate", node3)
    sp.add_input("indoorTemperature", node2, "indoorTemperature")
    sp.add_modeled_node(node4)

    return sp


def brick_signature_pattern():
    """
    Get the BRICK signature pattern of the space heater component.

    Returns:
        SignaturePattern: The BRICK signature pattern of the space heater component.
    """
    node0 = Node(cls=core.namespace.BRICK.Radiator)  # space heater
    node1 = Node(cls=core.namespace.BRICK.Space)  # building space
    node2 = Node(cls=core.namespace.BRICK.Heating_Water_Flow_Sensor)
    node3 = Node(cls=core.namespace.BRICK.Zone_Air_Temperature_Sensor)

    sp = SignaturePattern(
        id="space_heater_signature_pattern_brick",
    )

    sp.add_rule(
        StepRule(
            subject=node0, object=node1, predicate=core.namespace.BRICK.isLocationOf
        )
    )
    sp.add_rule(
        StepRule(subject=node2, object=node0, predicate=core.namespace.BRICK.isPointOf)
    )
    sp.add_rule(
        StepRule(subject=node3, object=node1, predicate=core.namespace.BRICK.isPointOf)
    )

    sp.add_input("waterFlowRate", node2, "measuredValue")
    sp.add_input("indoorTemperature", node3, "measuredValue")
    sp.add_modeled_node(node0)

    return sp


def brick_signature_pattern_room_heating_command():
    """BRICK pattern for a radiator driven by a room-level heating command.

    BMS-derived graphs often carry no radiator equipment, no water-flow
    sensor and no supply-temperature point: the only heating information on
    a room is a valve command (Hoeje-Taastrup Raadhus: ``R08_01_MVV01``, a
    ``brick:Heating_Command`` in percent).  This pattern models one space
    heater per such command::

        Room  hasPoint  Heating_Command   -> waterFlowRate  (valve command)
        Room                              -> indoorTemperature

    ``waterFlowRate`` is fed from the command's historised
    :class:`SensorSystem` (the leaf pattern matches the same point), so the
    caller supplies the percent -> kg/s conversion through
    :meth:`Model.set_transformations` (a ``brick:Heating_Command`` rule)
    and the nominal water temperature through
    :meth:`Model.fill_missing_inputs` (``supplyWaterTemperature``); both are
    site data the ontology does not carry.  ``UA`` and
    ``thermalMassHeatCapacity`` are then estimable per room.

    The delivered ``Power`` is consumed by the room: the BRICK building-space
    patterns bind the same ``Heating_Command`` node and connect
    ``Power -> heatGain`` (see
    :func:`twin4build.systems.building_space.building_space_system._brick_space_pattern`).

    The modeled identity is the multi-member group ``[space, heating_cmd]``:
    multi-member groups are mutex-ed per fingerprint, so this component can
    coexist with the :class:`BuildingSpaceSystem` modeled on ``space`` and
    with the leaf :class:`SensorSystem` modeled on the command.
    """
    space = Node(
        cls=(
            core.namespace.BRICK.Room,
            core.namespace.BRICK.HVAC_Zone,
            core.namespace.BRICK.Enclosed_space,
            core.namespace.BRICK.Open_space,
            core.namespace.REC.Room,
            core.namespace.REC.Zone,
            core.namespace.BRICK.Space,
        )
    )
    heating_cmd = Node(cls=core.namespace.BRICK.Heating_Command)

    sp = SignaturePattern(id="space_heater_signature_pattern_brick_room_command")
    sp.add_rule(
        StepRule(
            subject=space, object=heating_cmd, predicate=core.namespace.BRICK.hasPoint
        )
    )
    # Only when the radiator is *not* modelled as equipment.  Graphs that
    # carry both (the room and an explicit space heater both point at the
    # command) would otherwise match this pattern and
    # ``brick_signature_pattern_space_heater_valve`` at once, and the two
    # modeled identities (``[room, command]`` vs the equipment) do not
    # exclude each other -- giving two radiators per real one.
    sp.add_rule(
        NoStepRule(
            subject=Node(
                cls=(
                    core.namespace.BRICK.Space_Heater,
                    core.namespace.BRICK.Radiator,
                    core.namespace.BRICK.Radiant_Panel,
                    core.namespace.BRICK.Baseboard_Radiator,
                )
            ),
            object=heating_cmd,
            predicate=core.namespace.BRICK.hasPoint,
        )
    )
    sp.add_connection(heating_cmd, "measuredValue", "waterFlowRate")
    sp.add_connection(space, "indoorTemperature", "indoorTemperature")
    ModeledNode([space, heating_cmd])
    return sp


def brick_signature_pattern_space_heater_valve():
    """BRICK pattern for an explicit space heater driven by a valve command.

    The shape a BMS graph takes once the radiators are modelled as
    equipment rather than as bare points on the room::

        Space_Heater  feeds         Room
        Space_Heater  hasPoint      Heating_Command

    The water side comes from the command point::

        Heating_Command.inputSignal -> SpaceHeaterSystem.waterFlowRate
        Room.indoorTemperature      -> SpaceHeaterSystem.indoorTemperature
        SpaceHeaterSystem.Power     -> BuildingSpaceSystem.heatGain

    ``inputSignal`` is the port a controller identified at that URI
    produces (see ``ControllerIdentificationPISystem``'s
    ``brick_signature_pattern_space_heater_room``), so the loop closes
    during translation -- the same convention the AHU pattern uses for
    ``Damper_Position_Command``.  Naming the controller's port explicitly
    also makes the wiring deterministic: reading ``measuredValue`` would
    resolve to either the historised command sensor or the
    controller-driven one, whichever the MILP happened to pick.

    The command carries a valve opening, so the caller's
    ``Model.set_transformations`` converts it to kg/s (a
    ``brick:Heating_Command`` rule); the controller is identified against
    that same transformed series, so it produces the same units.  Without a
    controller in the translation the water side stays unwired -- use
    :func:`brick_signature_pattern_room_heating_command` (or supply the
    flow through ``fill_missing_inputs``) for replay-only models.

    ``supplyWaterTemperature`` stays unwired for
    :meth:`Model.fill_missing_inputs`; ``UA`` and
    ``thermalMassHeatCapacity`` are estimated per radiator.  The modeled
    identity is the space heater itself.
    """
    space_heater = Node(
        cls=(
            core.namespace.BRICK.Space_Heater,
            core.namespace.BRICK.Radiator,
            core.namespace.BRICK.Radiant_Panel,
            core.namespace.BRICK.Baseboard_Radiator,
        )
    )
    room = Node(
        cls=(
            core.namespace.BRICK.Room,
            core.namespace.BRICK.HVAC_Zone,
            core.namespace.BRICK.Enclosed_space,
            core.namespace.BRICK.Open_space,
            core.namespace.REC.Room,
            core.namespace.REC.Zone,
            core.namespace.BRICK.Space,
        )
    )
    heating_cmd = Node(cls=core.namespace.BRICK.Heating_Command)
    located_in = Predicate(
        (core.namespace.BRICK.feeds, core.namespace.FSO.feedsFluidTo)
    )
    sp = SignaturePattern(id="space_heater_signature_pattern_brick_space_heater_valve")
    sp.add_rule(StepRule(subject=space_heater, object=room, predicate=located_in))
    sp.add_rule(
        StepRule(
            subject=space_heater,
            object=heating_cmd,
            predicate=core.namespace.BRICK.hasPoint,
        )
    )
    sp.add_connection(
        heating_cmd, "inputSignal", "waterFlowRate", output_port_index=heating_cmd
    )
    sp.add_connection(room, "indoorTemperature", "indoorTemperature")
    sp.add_modeled_node(space_heater)
    return sp
