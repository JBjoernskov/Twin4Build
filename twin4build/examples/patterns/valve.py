"""Example signature patterns for :mod:`twin4build.systems.valve.valve_system`.

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
