"""Example signature patterns for :mod:`twin4build.systems.utils.occupancy_system`.

Moved out of the system module (#200); examples of a graph shape, not a
standard.  Bound to their classes by :mod:`twin4build.examples.patterns`.
"""

import twin4build.core as core


def brick_signature_pattern_room_co2():
    """BRICK pattern: occupancy inferred from a room's CO2 balance.

    The graph-driven form of the CSV-driven construction in
    ``full_workflow_example``::

        Room  hasPoint  Zone_CO2_Level_Sensor       -> indoorCo2Measured
        Room  isFedBy   VAV
        VAV   hasPoint  Damper_Position_Command |
                        Damper_Position_Setpoint    -> damperPositionMeasured
                                                       (one slot per VAV)
        OccupancySystem.scheduleValue -> BuildingSpaceSystem.numberOfPeople

    The occupancy derives the air flow from the historised damper
    COMMAND through its own damper model -- the same point, and the same
    characteristic, that drive the branch flows of the AHU pattern
    (``Damper_Position_Command`` -> ``supplyDamperPosition``), so the
    inverse balance (occupancy from CO2) and the forward one (CO2 from
    occupancy) see one and the same air flow.  A damper position SENSOR
    is not that signal on a pressure-independent VAV: its local flow loop
    moves the blade to hold the commanded flow against the duct pressure,
    and the blade parks open when the fan stops.

    Both inputs are historised sensors, so the fast paths capture them as
    exogenous signals and no gradient feedback loop runs through the
    inferred occupancy.  ``outdoorCo2Concentration`` stays unwired for
    ``fill_missing_inputs`` unless the building has an outdoor CO2 sensor.

    Modeled identity is ``[room, co2_sensor]``: the leaf sensor pattern is a
    singleton on the CO2 point and the zone a singleton on the room, so a
    multi-member group is the only identity that coexists with both.
    """
    from twin4build.translator.translator import (
        ModeledNode,
        Node,
        OptionalRule,
        SetStepRule,
        SignaturePattern,
        StepRule,
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
    co2_sensor = Node(cls=core.namespace.BRICK.Zone_CO2_Level_Sensor)
    vavs = Node(cls=core.namespace.BRICK.VAV)
    damper_positions = Node(
        cls=(
            core.namespace.BRICK.Damper_Position_Command,
            core.namespace.BRICK.Damper_Position_Setpoint,
        )
    )
    sp = SignaturePattern(id="occupancy_signature_pattern_brick_room_co2")
    sp.add_rule(
        StepRule(subject=room, object=co2_sensor, predicate=core.namespace.BRICK.hasPoint)
    )
    sp.add_rule(
        SetStepRule(subject=room, object=vavs, predicate=core.namespace.BRICK.isFedBy)
    )
    sp.add_rule(
        SetStepRule(
            subject=vavs, object=damper_positions, predicate=core.namespace.BRICK.hasPoint
        )
    )
    # The fan state of the air handler feeding those VAVs, optional.
    ahu = Node(cls=core.namespace.BRICK.AHU)
    supply_fan = Node(cls=core.namespace.BRICK.Supply_Fan)
    fan_speed = Node(cls=core.namespace.BRICK.Fan_Speed_Command)
    sp.add_rule(SetStepRule(subject=vavs, object=ahu, predicate=core.namespace.BRICK.isFedBy))
    sp.add_rule(OptionalRule(subject=ahu, object=supply_fan, predicate=core.namespace.BRICK.hasPart))
    sp.add_rule(OptionalRule(subject=supply_fan, object=fan_speed, predicate=core.namespace.BRICK.hasPoint))
    sp.add_connection(fan_speed, "measuredData", "fanSpeedMeasured")
    sp.add_connection(co2_sensor, "measuredData", "indoorCo2Measured")
    sp.add_connection(
        damper_positions,
        "measuredData",
        "damperPositionMeasured",
        input_port_index=damper_positions,
    )
    ModeledNode([room, co2_sensor])
    return sp
