"""Example signature patterns for :mod:`twin4build.systems.schedule.schedule_system`.

Moved out of the system module (#200): patterns describe how one kind
of graph maps onto the component, and are examples of that, not a
standard.  Bound to their classes by :mod:`twin4build.examples.patterns`.
"""

from twin4build.translator.translator import (
    Node,
    SignaturePattern,
)
import twin4build.core as core


def saref_signature_pattern():
    """
    Get the SAREF signature pattern of the schedule component.

    Returns:
        SignaturePattern: The SAREF signature pattern of the schedule component.
    """
    node0 = Node(cls=(core.namespace.S4BLDG.Schedule))
    sp = SignaturePattern(id="schedule_signature_pattern")
    sp.add_modeled_node(node0)
    return sp
