"""Example signature patterns for :mod:`twin4build.systems.damper.damper_system`.

Moved out of the system module (#200): patterns describe how one kind
of graph maps onto the component, and are examples of that, not a
standard.  Bound to their classes by :mod:`twin4build.examples.patterns`.
"""

from twin4build.translator.translator import (
    StepRule,
    Node,
    OptionalRule,
    SignaturePattern,
)
import twin4build.core as core


def saref_signature_pattern():
    """
    Get the SAREF signature pattern of the damper component.

    Returns:
        SignaturePattern: The SAREF signature pattern of the damper component.
    """
    node0 = Node(cls=core.namespace.S4BLDG.Damper)
    node1 = Node(cls=core.namespace.S4BLDG.Controller)
    node2 = Node(cls=core.namespace.SAREF.OpeningPosition)
    node3 = Node(cls=core.namespace.SAREF.Property)
    node4 = Node(cls=core.namespace.SAREF.PropertyValue)
    node5 = Node(cls=core.namespace.XSD.float)
    node6 = Node(cls=core.namespace.S4BLDG.NominalAirFlowRate)
    sp = SignaturePattern(id="damper_signature_pattern")

    # Add edges to the signature pattern
    sp.add_rule(
        StepRule(subject=node1, object=node2, predicate=core.namespace.SAREF.controls)
    )
    sp.add_rule(
        StepRule(
            subject=node2, object=node0, predicate=core.namespace.SAREF.isPropertyOf
        )
    )
    sp.add_rule(
        StepRule(subject=node1, object=node3, predicate=core.namespace.SAREF.observes)
    )
    sp.add_rule(
        OptionalRule(
            subject=node4, object=node5, predicate=core.namespace.SAREF.hasValue
        )
    )
    sp.add_rule(
        OptionalRule(
            subject=node4,
            object=node6,
            predicate=core.namespace.SAREF.isValueOfProperty,
        )
    )
    sp.add_rule(
        OptionalRule(
            subject=node0, object=node4, predicate=core.namespace.SAREF.hasPropertyValue
        )
    )

    # Configure inputs, parameters, and modeled nodes
    sp.add_input("damperPosition", node1, "inputSignal")
    sp.add_parameter("nominalAirFlowRate", node5)
    sp.add_modeled_node(node0)

    return sp
