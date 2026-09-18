"""Example signature patterns for :mod:`twin4build.systems.building_space.building_space_system`.

Moved out of the system module (#200); examples of a graph shape, not a
standard.  Bound to their classes by :mod:`twin4build.examples.patterns`.
"""

from twin4build.translator.translator import (
    StepRule,
    Node,
    SignaturePattern,
    PathRule,
)
import twin4build.core as core


def saref_signature_pattern_sensor():
    """
    Get the SAREF signature pattern (with supply-air temperature sensor) of the
    building space component.

    Returns:
        SignaturePattern: The signature pattern of the building space component.
    """

    node0 = Node(cls=core.namespace.S4BLDG.Damper)  # supply damper
    node1 = Node(cls=core.namespace.S4BLDG.Damper)  # return damper
    node2 = Node(cls=core.namespace.S4BLDG.BuildingSpace)
    node4 = Node(cls=core.namespace.S4BLDG.SpaceHeater)
    node5 = Node(cls=core.namespace.S4BLDG.Schedule)
    node6 = Node(cls=core.namespace.S4BLDG.OutdoorEnvironment)
    node7 = Node(cls=core.namespace.SAREF.Sensor)
    node8 = Node(cls=core.namespace.SAREF.Temperature)
    sp = SignaturePattern(
        id="building_space_signature_pattern_sensor",
    )

    sp.add_rule(
        StepRule(
            subject=node0, object=node2, predicate=core.namespace.FSO.suppliesFluidTo
        )
    )
    sp.add_rule(
        StepRule(
            subject=node1, object=node2, predicate=core.namespace.FSO.hasFluidReturnedBy
        )
    )
    sp.add_rule(
        StepRule(
            subject=node4, object=node2, predicate=core.namespace.S4BLDG.isContainedIn
        )
    )
    sp.add_rule(
        StepRule(subject=node2, object=node5, predicate=core.namespace.SAREF.hasProfile)
    )
    sp.add_rule(
        StepRule(
            subject=node2, object=node6, predicate=core.namespace.S4SYST.connectedTo
        )
    )
    sp.add_rule(
        PathRule(
            subject=node0, object=node7, predicate=core.namespace.FSO.hasFluidSuppliedBy
        )
    )
    sp.add_rule(
        StepRule(subject=node7, object=node8, predicate=core.namespace.SAREF.observes)
    )

    sp.add_input("supplyAirFlowRate", node0, "airFlowRate")
    sp.add_input("exhaustAirFlowRate", node1, "airFlowRate")
    sp.add_input("heatGain", node4, "Power")
    sp.add_input("numberOfPeople", node5, "scheduleValue")
    sp.add_input("outdoorTemperature", node6, "outdoorTemperature")
    sp.add_input("outdoorCO2", node6, "outdoorCo2Concentration")
    sp.add_input("globalIrradiation", node6, "globalIrradiation")
    sp.add_input("supplyAirTemperature", node7, "measuredValue")
    # Interzonal/boundary coupling is modeled by a separate WallSystem
    # (wired manually, or via a future wall/adjacency signature pattern).

    sp.add_modeled_node(node2)
    return sp


def saref_signature_pattern():
    """
    Get the SAREF signature pattern of the building space component.

    Returns:
        SignaturePattern: The signature pattern of the building space component.
    """

    node0 = Node(cls=core.namespace.S4BLDG.Damper)  # supply damper
    node1 = Node(cls=core.namespace.S4BLDG.Damper)  # return damper
    node2 = Node(cls=core.namespace.S4BLDG.BuildingSpace)
    node4 = Node(cls=core.namespace.S4BLDG.SpaceHeater)
    node5 = Node(cls=core.namespace.S4BLDG.Schedule)
    node6 = Node(cls=core.namespace.S4BLDG.OutdoorEnvironment)
    node7 = Node(
        cls=(
            core.namespace.S4BLDG.Coil,
            core.namespace.S4BLDG.AirToAirHeatRecovery,
            core.namespace.S4BLDG.Fan,
        )
    )

    sp = SignaturePattern(
        id="building_space_signature_pattern",
    )

    sp.add_rule(
        StepRule(
            subject=node0, object=node2, predicate=core.namespace.FSO.suppliesFluidTo
        )
    )
    sp.add_rule(
        StepRule(
            subject=node1, object=node2, predicate=core.namespace.FSO.hasFluidReturnedBy
        )
    )
    sp.add_rule(
        StepRule(
            subject=node4, object=node2, predicate=core.namespace.S4BLDG.isContainedIn
        )
    )
    sp.add_rule(
        StepRule(subject=node2, object=node5, predicate=core.namespace.SAREF.hasProfile)
    )
    sp.add_rule(
        StepRule(
            subject=node2, object=node6, predicate=core.namespace.S4SYST.connectedTo
        )
    )
    sp.add_rule(
        PathRule(
            subject=node0, object=node7, predicate=core.namespace.FSO.hasFluidSuppliedBy
        )
    )

    sp.add_input("supplyAirFlowRate", node0, "airFlowRate")
    sp.add_input("exhaustAirFlowRate", node1, "airFlowRate")
    sp.add_input("heatGain", node4, "Power")
    sp.add_input("numberOfPeople", node5, "scheduleValue")
    sp.add_input("outdoorTemperature", node6, "outdoorTemperature")
    sp.add_input("outdoorCO2", node6, "outdoorCo2Concentration")
    sp.add_input("globalIrradiation", node6, "globalIrradiation")
    sp.add_input(
        "supplyAirTemperature",
        node7,
        ("outletAirTemperature", "primaryTemperatureOut", "outletAirTemperature"),
    )
    # Interzonal/boundary coupling is modeled by a separate WallSystem
    # (wired manually, or via a future wall/adjacency signature pattern).

    sp.add_modeled_node(node2)
    return sp
