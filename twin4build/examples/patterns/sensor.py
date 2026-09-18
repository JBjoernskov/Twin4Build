"""Example signature patterns for :mod:`twin4build.systems.sensor.sensor_system`.

Moved out of the system module (#200); examples of a graph shape, not a
standard.  Bound to their classes by :mod:`twin4build.examples.patterns`.
"""

from twin4build.translator.translator import (
    Node,
    PathRule,
    SignaturePattern,
    StepRule,
)
import twin4build.core as core


def get_signature_pattern_input():
    node0 = Node(cls=(core.namespace.SAREF.Sensor,))
    sp = SignaturePattern(
        id="signature_pattern_input",
    )
    sp.add_modeled_node(node0)
    return sp


def get_space_temperature_signature_pattern():
    node0 = Node(cls=(core.namespace.SAREF.Sensor))
    node1 = Node(cls=(core.namespace.SAREF.Temperature))
    node2 = Node(cls=(core.namespace.S4BLDG.BuildingSpace))
    sp = SignaturePattern(id="space_temperature_signature_pattern")
    sp.add_rule(
        StepRule(subject=node0, object=node1, predicate=core.namespace.SAREF.observes)
    )
    sp.add_rule(
        StepRule(subject=node1, object=node2, predicate=core.namespace.SAREF.isPropertyOf)
    )
    sp.add_input("measuredValue", node2, ("indoorTemperature"))
    sp.add_modeled_node(node0)
    return sp


def get_space_co2_signature_pattern():
    node0 = Node(cls=(core.namespace.SAREF.Sensor,))
    node1 = Node(cls=(core.namespace.SAREF.Co2,))
    node2 = Node(cls=(core.namespace.S4BLDG.BuildingSpace,))
    sp = SignaturePattern(id="space_co2_signature_pattern")
    sp.add_rule(
        StepRule(subject=node0, object=node1, predicate=core.namespace.SAREF.observes)
    )
    sp.add_rule(
        StepRule(subject=node1, object=node2, predicate=core.namespace.SAREF.isPropertyOf)
    )
    sp.add_input("measuredValue", node2, ("indoorCO2"))
    sp.add_modeled_node(node0)
    return sp


def get_position_signature_pattern():
    node0 = Node(cls=(core.namespace.SAREF.Sensor,))
    node1 = Node(cls=(core.namespace.SAREF.OpeningPosition,))
    node2 = Node(
        cls=(
            core.namespace.S4BLDG.Valve,
            core.namespace.S4BLDG.Damper,
        )
    )
    node3 = Node(cls=(core.namespace.S4BLDG.Controller))
    sp = SignaturePattern(id="position_signature_pattern")
    sp.add_rule(
        StepRule(subject=node0, object=node1, predicate=core.namespace.SAREF.observes)
    )
    sp.add_rule(
        StepRule(subject=node1, object=node2, predicate=core.namespace.SAREF.isPropertyOf)
    )
    sp.add_rule(
        StepRule(subject=node3, object=node1, predicate=core.namespace.SAREF.controls)
    )
    sp.add_input("measuredValue", node3, ("inputSignal", "inputSignal"))
    sp.add_modeled_node(node0)
    return sp


def get_temperature_before_air_to_air_supply_side():
    node0 = Node(cls=(core.namespace.SAREF.Sensor,))
    node1 = Node(cls=(core.namespace.SAREF.Temperature,))
    node2 = Node(cls=(core.namespace.S4BLDG.AirToAirHeatRecovery,))  # AirToAirPrimary
    node9 = Node(cls=(core.namespace.S4BLDG.AirToAirHeatRecovery))  # AirToAirSuper
    sp = SignaturePattern(id="temperature_before_air_to_air_supply_side")

    sp.add_rule(
        StepRule(subject=node0, object=node1, predicate=core.namespace.SAREF.observes)
    )
    sp.add_rule(
        PathRule(
            subject=node2, object=node0, predicate=core.namespace.FSO.hasFluidSuppliedBy
        )
    )
    sp.add_rule(
        StepRule(subject=node2, object=node9, predicate=core.namespace.S4SYST.subSystemOf)
    )

    sp.add_input("measuredValue", node2, ("primaryTemperatureIn"))
    sp.add_modeled_node(node0)

    return sp


def get_temperature_before_air_to_air_exhaust_side():
    node0 = Node(cls=(core.namespace.SAREF.Sensor,))
    node1 = Node(cls=(core.namespace.SAREF.Temperature,))
    node2 = Node(cls=(core.namespace.S4BLDG.AirToAirHeatRecovery))  # AirToAirPrimary

    node9 = Node(cls=(core.namespace.S4BLDG.AirToAirHeatRecovery))  # AirToAirSuper

    sp = SignaturePattern(id="temperature_before_air_to_air_exhaust_side")
    sp.add_rule(
        StepRule(subject=node0, object=node1, predicate=core.namespace.SAREF.observes)
    )
    sp.add_rule(
        PathRule(
            subject=node0, object=node2, predicate=core.namespace.FSO.returnsFluidTo
        )
    )
    sp.add_rule(
        StepRule(subject=node2, object=node9, predicate=core.namespace.S4SYST.subSystemOf)
    )

    sp.add_input("measuredValue", node2, ("secondaryTemperatureIn"))
    sp.add_modeled_node(node0)

    return sp


def get_temperature_after_air_to_air_supply_side():
    node0 = Node(cls=(core.namespace.SAREF.Sensor,))
    node1 = Node(cls=(core.namespace.SAREF.Temperature,))
    node2 = Node(cls=(core.namespace.S4BLDG.AirToAirHeatRecovery))  # AirToAirPrimary
    node9 = Node(cls=(core.namespace.S4BLDG.AirToAirHeatRecovery))  # AirToAirSuper

    sp = SignaturePattern(id="temperature_after_air_to_air_supply_side")
    sp.add_rule(
        StepRule(subject=node0, object=node1, predicate=core.namespace.SAREF.observes)
    )
    sp.add_rule(
        StepRule(
            subject=node0, object=node2, predicate=core.namespace.FSO.hasFluidSuppliedBy
        )
    )
    sp.add_rule(
        StepRule(subject=node2, object=node9, predicate=core.namespace.S4SYST.subSystemOf)
    )

    sp.add_input("measuredValue", node2, ("primaryTemperatureOut"))
    sp.add_modeled_node(node0)

    return sp


def get_temperature_after_air_to_air_exhaust_side():
    node0 = Node(cls=(core.namespace.SAREF.Sensor,))
    node1 = Node(cls=(core.namespace.SAREF.Temperature,))
    node2 = Node(cls=(core.namespace.S4BLDG.AirToAirHeatRecovery))  # AirToAirPrimary

    node9 = Node(cls=(core.namespace.S4BLDG.AirToAirHeatRecovery))  # AirToAirSuper

    sp = SignaturePattern(id="temperature_after_air_to_air_exhaust_side")
    sp.add_rule(
        StepRule(subject=node0, object=node1, predicate=core.namespace.SAREF.observes)
    )
    sp.add_rule(
        StepRule(subject=node2, object=node0, predicate=core.namespace.FSO.returnsFluidTo)
    )
    sp.add_rule(
        StepRule(subject=node2, object=node9, predicate=core.namespace.S4SYST.subSystemOf)
    )

    sp.add_input("measuredValue", node2, ("secondaryTemperatureOut"))
    sp.add_modeled_node(node0)

    return sp
