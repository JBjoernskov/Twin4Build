"""Example signature patterns for :mod:`twin4build.systems.building_space.building_space_thermal_system`.

Moved out of the system module (#200): patterns describe how one kind
of graph maps onto the component, and are examples of that, not a
standard.  Bound to their classes by :mod:`twin4build.examples.patterns`.
"""

from twin4build.translator.translator import (
    StepRule,
    AnyPathRule,
    Node,
    SignaturePattern,
    PathRule,
)
import twin4build.core as core


def brick_signature_pattern():
    """
    Get the BRICK-only signature pattern of the building space component.

    Returns:
        SignaturePattern: The BRICK-only signature pattern of the building space component.
    """

    node0 = Node(cls=core.namespace.BRICK.AHU)
    node2 = Node(cls=core.namespace.BRICK.HVAC_Zone)  # building space/room
    node3 = Node(cls=core.namespace.BRICK.Room)
    node4 = Node(cls=core.namespace.BRICK.Air_Temperature_Sensor)
    node6 = Node(
        cls=core.namespace.BRICK.Outside_Air_Temperature_Sensor
    )  # outdoor temperature sensor

    sp = SignaturePattern(
        id="building_space_signature_pattern_brick",
    )

    sp.add_rule(
        StepRule(subject=node0, object=node2, predicate=core.namespace.BRICK.feeds)
    )
    # sp.add_rule(StepRule(subject=node1, object=node2, predicate=core.namespace.BRICK.isFedBy))
    sp.add_rule(
        StepRule(subject=node2, object=node3, predicate=core.namespace.BRICK.hasPart)
    )
    sp.add_rule(
        StepRule(subject=node4, object=node3, predicate=core.namespace.BRICK.isPointOf)
    )
    # sp.add_rule(AnyPathRule(subject=node9, object=node2, predicate=core.namespace.BRICK.isAdjacentTo)) # TODO: Makes _prune_recursive fail, infinite recursion

    # Optional
    # heatGain
    # numberOfPeople

    sp.add_input("supplyAirFlowRate", node0, "airFlowRate")
    sp.add_input("exhaustAirFlowRate", node0, "airsFlowRate")
    # sp.add_input("numberOfPeople", node5, "measuredValue")
    sp.add_input("outdoorTemperature", node6, "measuredValue")
    # sp.add_input("outdoorCO2", node6, "outdoorCo2Concentration")
    # sp.add_input("globalIrradiation", node6, "globalIrradiation")
    sp.add_input("supplyAirTemperature", node0)

    sp.add_modeled_node(node3)
    return sp
