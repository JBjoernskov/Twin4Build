"""Example signature patterns for :mod:`twin4build.systems.air_to_air_heat_recovery.air_to_air_heat_recovery_system`.

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
    node0 = Node(cls=core.namespace.S4BLDG.AirToAirHeatRecovery)
    node1 = Node(cls=core.namespace.S4BLDG.OutdoorEnvironment)
    node2 = Node(cls=core.namespace.S4BLDG.FlowJunction)
    node3 = Node(cls=core.namespace.S4BLDG.FlowJunction)
    node4 = Node(cls=core.namespace.S4BLDG.PrimaryAirFlowRateMax)
    node5 = Node(cls=core.namespace.SAREF.PropertyValue)
    node6 = Node(cls=core.namespace.XSD.float)
    node7 = Node(cls=core.namespace.S4BLDG.SecondaryAirFlowRateMax)
    node8 = Node(cls=core.namespace.SAREF.PropertyValue)
    node9 = Node(cls=core.namespace.XSD.float)
    node10 = Node(cls=core.namespace.S4BLDG.AirToAirHeatRecovery)  # primary
    node11 = Node(cls=core.namespace.S4BLDG.AirToAirHeatRecovery)  # secondary
    node12 = Node(cls=core.namespace.S4BLDG.Controller)
    node13 = Node(cls=core.namespace.SAREF.Motion)
    node14 = Node(cls=core.namespace.S4BLDG.Schedule)
    sp = SignaturePattern(id="air_to_air_heat_recovery_signature_pattern")

    # buildingTemperature (SecondaryTemperatureIn)
    sp.add_rule(
        PathRule(
            subject=node10,
            object=node1,
            predicate=core.namespace.FSO.hasFluidSuppliedBy,
        )
    )
    sp.add_rule(
        PathRule(
            subject=node10, object=node2, predicate=core.namespace.FSO.suppliesFluidTo
        )
    )
    sp.add_rule(
        PathRule(
            subject=node11,
            object=node3,
            predicate=core.namespace.FSO.hasFluidReturnedBy,
        )
    )

    sp.add_rule(
        OptionalRule(subject=node5, object=node6, predicate=core.namespace.SAREF.hasValue)
    )
    sp.add_rule(
        OptionalRule(
            subject=node5,
            object=node4,
            predicate=core.namespace.SAREF.isValueOfProperty,
        )
    )
    sp.add_rule(
        OptionalRule(
            subject=node0, object=node5, predicate=core.namespace.SAREF.hasPropertyValue
        )
    )

    # airFlowRateMax
    sp.add_rule(
        OptionalRule(subject=node8, object=node9, predicate=core.namespace.SAREF.hasValue)
    )
    sp.add_rule(
        OptionalRule(
            subject=node8,
            object=node7,
            predicate=core.namespace.SAREF.isValueOfProperty,
        )
    )
    sp.add_rule(
        OptionalRule(
            subject=node0, object=node8, predicate=core.namespace.SAREF.hasPropertyValue
        )
    )

    sp.add_rule(
        StepRule(subject=node10, object=node0, predicate=core.namespace.S4SYST.subSystemOf)
    )
    sp.add_rule(
        StepRule(subject=node11, object=node0, predicate=core.namespace.S4SYST.subSystemOf)
    )

    sp.add_rule(
        StepRule(subject=node12, object=node13, predicate=core.namespace.SAREF.controls)
    )
    sp.add_rule(
        StepRule(subject=node13, object=node0, predicate=core.namespace.SAREF.isPropertyOf)
    )
    sp.add_rule(
        StepRule(subject=node12, object=node14, predicate=core.namespace.SAREF.hasProfile)
    )

    sp.add_parameter("primaryAirFlowRateMax", node6)
    sp.add_parameter("secondaryAirFlowRateMax", node9)

    sp.add_input("primaryTemperatureIn", node1, "outdoorTemperature")
    sp.add_input("secondaryTemperatureIn", node3, "airTemperatureOut")
    sp.add_input("primaryAirFlowRate", node2, "airFlowRateIn")
    sp.add_input("secondaryAirFlowRate", node3, "airFlowRateOut")
    sp.add_input("primaryTemperatureOutSetpoint", node14, "scheduleValue")

    sp.add_modeled_node(node0)
    sp.add_modeled_node(node10)
    sp.add_modeled_node(node11)

    return sp
