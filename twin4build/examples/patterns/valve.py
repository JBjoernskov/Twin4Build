"""Example signature patterns for :mod:`twin4build.systems.valve.valve_system`.

Moved out of the system module (#200); examples of a graph shape, not a
standard.  Bound to their classes by :mod:`twin4build.examples.patterns`.
"""

from twin4build.translator.translator import (
    StepRule,
    Node,
    SignaturePattern,
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
    sp = SignaturePattern(id="valve_signature_pattern")

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
