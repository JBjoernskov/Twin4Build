"""Example signature patterns for :mod:`twin4build.systems.junction.return_flow_junction_system`.

Moved out of the system module (#200): patterns describe how one kind
of graph maps onto the component, and are examples of that, not a
standard.  Bound to their classes by :mod:`twin4build.examples.patterns`.
"""

from twin4build.translator.translator import (
    StepRule,
    AnyPathRule,
    Node,
    OptionalRule,
    SignaturePattern,
    PathRule,
)
import twin4build.core as core


def saref_signature_pattern():
    """
    Get the SAREF signature pattern of the return flow junction component.

    Returns:
        SignaturePattern: The SAREF signature pattern of the return flow junction component.
    """
    node0 = Node(cls=core.namespace.S4BLDG.FlowJunction)  # flow junction
    node1 = Node(cls=core.namespace.S4BLDG.Damper)  # damper
    node2 = Node(cls=core.namespace.S4BLDG.BuildingSpace)  # building space
    sp = SignaturePattern(
        id="return_flow_junction_signature_pattern",
    )
    sp.add_rule(
        AnyPathRule(
            subject=node0, object=node1, predicate=core.namespace.FSO.hasFluidReturnedBy
        )
    )
    sp.add_rule(
        StepRule(
            subject=node1, object=node2, predicate=core.namespace.FSO.hasFluidReturnedBy
        )
    )

    sp.add_input("airFlowRateIn", node1, "airFlowRate")
    sp.add_input("airTemperatureIn", node2, "indoorTemperature")
    # sp.add_input("inletAirTemperature", node15, ("outletAirTemperature", "primaryTemperatureOut", "outletAirTemperature"))
    sp.add_modeled_node(node0)
    # cs.add_parameter("globalIrradiation", node2, "globalIrradiation")
    return sp


def brick_signature_pattern():
    """
    Get the BRICK signature pattern of the return flow junction component.

    Returns:
        SignaturePattern: The BRICK signature pattern of the return flow junction component.
    """
    node0 = Node(cls=core.namespace.BRICK.Air_Flow_Junction)  # flow junction
    node1 = Node(cls=core.namespace.BRICK.Damper)  # damper
    node2 = Node(cls=core.namespace.BRICK.HVAC_Zone)  # building space/zone

    sp = SignaturePattern(
        id="return_flow_junction_signature_pattern_brick",
    )
    sp.add_rule(
        StepRule(subject=node1, object=node0, predicate=core.namespace.BRICK.feeds)
    )
    sp.add_rule(
        StepRule(subject=node2, object=node1, predicate=core.namespace.BRICK.feeds)
    )

    sp.add_input("airFlowRateIn", node1, "airFlowRate")
    sp.add_input("airTemperatureIn", node2, "indoorTemperature")
    sp.add_modeled_node(node0)
    return sp
