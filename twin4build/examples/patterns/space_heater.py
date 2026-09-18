"""Example signature patterns for :mod:`twin4build.systems.space_heater.space_heater_system`.

Moved out of the system module (#200); examples of a graph shape, not a
standard.  Bound to their classes by :mod:`twin4build.examples.patterns`.
"""

from twin4build import core
from twin4build.translator.translator import (
    StepRule,
    Node,
    SignaturePattern,
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
