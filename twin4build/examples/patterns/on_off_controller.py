"""Example signature patterns for :mod:`twin4build.systems.controller.rulebased_controller.on_off_controller.on_off_controller_system`.

Moved out of the system module (#200): patterns describe how one kind
of graph maps onto the component, and are examples of that, not a
standard.  Bound to their classes by :mod:`twin4build.examples.patterns`.
"""

from twin4build.translator.translator import StepRule, Node, SignaturePattern
import twin4build.core as core


def saref_signature_pattern():
    """Get the SAREF signature pattern of the on-off controller component."""
    node0 = Node(cls=(core.namespace.S4BLDG.RulebasedController))
    node1 = Node(cls=(core.namespace.SAREF.Sensor))
    node2 = Node(cls=(core.namespace.SAREF.Property))
    node3 = Node(cls=(core.namespace.S4BLDG.Schedule))
    sp = SignaturePattern(id="on_off_controller_signature_pattern")
    sp.add_rule(
        StepRule(subject=node0, object=node2, predicate=core.namespace.SAREF.observes)
    )
    sp.add_rule(
        StepRule(subject=node1, object=node2, predicate=core.namespace.SAREF.observes)
    )
    sp.add_rule(
        StepRule(subject=node0, object=node3, predicate=core.namespace.SAREF.hasProfile)
    )
    sp.add_input("actualValue", node1, "measuredValue")
    sp.add_input("setpointValue", node3, "scheduleValue")
    sp.add_modeled_node(node0)
    return sp


def brick_signature_pattern():
    """Get the BRICK signature pattern of the on-off controller component."""
    node0 = Node(cls=core.namespace.BRICK.On_Off_Controller)
    node1 = Node(cls=core.namespace.BRICK.Sensor)
    node2 = Node(cls=core.namespace.BRICK.Setpoint)

    sp = SignaturePattern(id="on_off_controller_signature_pattern_brick")
    sp.add_rule(
        StepRule(subject=node1, object=node0, predicate=core.namespace.BRICK.isPointOf)
    )
    sp.add_rule(
        StepRule(subject=node2, object=node0, predicate=core.namespace.BRICK.isPointOf)
    )

    sp.add_input("actualValue", node1, "measuredValue")
    sp.add_input("setpointValue", node2, "setpoint")
    sp.add_modeled_node(node0)
    return sp
