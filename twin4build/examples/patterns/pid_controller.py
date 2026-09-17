"""Example signature patterns for :mod:`twin4build.systems.controller.setpoint_controller.pid_controller.pid_controller_system`.

Moved out of the system module (#200): patterns describe how one kind
of graph maps onto the component, and are examples of that, not a
standard.  Bound to their classes by :mod:`twin4build.examples.patterns`.
"""

from twin4build.translator.translator import (
    StepRule,
    Node,
    SignaturePattern,
)
import twin4build.core as core


def saref_signature_pattern():
    node0 = Node(cls=core.namespace.S4BLDG.SetpointController)
    node1 = Node(cls=core.namespace.SAREF.Sensor)
    node2 = Node(cls=core.namespace.SAREF.Property)
    node3 = Node(cls=core.namespace.S4BLDG.Schedule)
    node4 = Node(cls=core.namespace.XSD.boolean)
    sp = SignaturePattern(id="pid_controller_signature_pattern")
    sp.add_rule(
        StepRule(subject=node0, object=node2, predicate=core.namespace.SAREF.observes)
    )
    sp.add_rule(
        StepRule(subject=node1, object=node2, predicate=core.namespace.SAREF.observes)
    )
    sp.add_rule(
        StepRule(subject=node0, object=node3, predicate=core.namespace.SAREF.hasProfile)
    )
    sp.add_rule(
        StepRule(subject=node0, object=node4, predicate=core.namespace.S4BLDG.isReverse)
    )

    sp.add_input("actualValue", node1, "measuredValue")
    sp.add_input("setpointValue", node3, "scheduleValue")
    sp.add_parameter("is_reverse", node4)
    sp.add_modeled_node(node0)
    return sp
