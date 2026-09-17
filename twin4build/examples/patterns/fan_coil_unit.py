"""Example signature patterns for :mod:`twin4build.systems.fan_coil_unit.fan_coil_unit_system`.

Moved out of the system module (#200): patterns describe how one kind
of graph maps onto the component, and are examples of that, not a
standard.  Bound to their classes by :mod:`twin4build.examples.patterns`.
"""

from twin4build import core
from twin4build.translator.translator import (
    StepRule,
    Node,
    OptionalRule,
    SignaturePattern,
)


def brick_signature_pattern():
    """
    BRICK pattern for a Fan_Coil_Unit with an associated Space, water-flow sensor,
    and zone-air-temperature sensor.

    Topology::

        Fan_Coil_Unit  hasPart       Space
        Flow_Sensor    isPointOf     Fan_Coil_Unit   → waterFlowRate
        Temp_Sensor    isPointOf     Space           → inletAirTemperature
    """
    node0 = Node(cls=core.namespace.BRICK.Fan_Coil_Unit)
    node1 = Node(cls=core.namespace.BRICK.Space)
    node2 = Node(cls=core.namespace.BRICK.Heating_Water_Flow_Sensor)
    node3 = Node(cls=core.namespace.BRICK.Zone_Air_Temperature_Sensor)

    sp = SignaturePattern(id="fan_coil_unit_signature_pattern_brick")

    sp.add_rule(
        StepRule(subject=node0, object=node1, predicate=core.namespace.BRICK.hasPart)
    )
    sp.add_rule(
        StepRule(subject=node2, object=node0, predicate=core.namespace.BRICK.isPointOf)
    )
    sp.add_rule(
        StepRule(subject=node3, object=node1, predicate=core.namespace.BRICK.isPointOf)
    )

    sp.add_connection(node2, "measuredValue", "valvePosition")
    sp.add_connection(node3, "measuredValue", "inletAirTemperature")
    sp.add_modeled_node(node0)

    return sp


def brick_signature_pattern_vav_ahu():
    """
    BRICK pattern for a VAV-with-reheat-coil modelled as a FanCoilUnit.

    Both inlet air temperature and air flow rate come from the upstream AHU,
    which distributes supply air per branch (indexed by VAV).

    Topology::

        AHU   feeds   VAV   (requires reheat command to distinguish from plain VAV)

    Connections::

        AHU.supplyAirTemperature[vav] → FCU.inletAirTemperature
        AHU.supplyAirFlowRate[vav]    → FCU.airFlowRate
    """
    ahu = Node(cls=core.namespace.BRICK.AHU)
    vav = Node(cls=core.namespace.BRICK.VAV)
    reheat_cmd = Node(cls=core.namespace.BRICK.Command)

    sp = SignaturePattern(id="fan_coil_unit_signature_pattern_brick_vav_ahu")

    sp.add_rule(
        StepRule(subject=ahu, object=vav, predicate=core.namespace.BRICK.feeds)
    )
    sp.add_rule(
        StepRule(subject=vav, object=reheat_cmd, predicate=core.namespace.BRICK.hasPoint)
    )

    sp.add_connection(ahu, "supplyAirTemperature", "inletAirTemperature")
    sp.add_connection(ahu, "supplyAirFlowRate", "airFlowRate", output_port_index=vav)
    # Source-side port is ``inputSignal``: when ``ControllerIdentificationPI
    # TorchSystem`` is in Stage-2's ``systems_``, the controller component
    # is matched at the same ``reheat_cmd`` URI as the historised
    # ``SensorSystem`` and provides this ``inputSignal`` output, closing
    # the reheat-valve control loop natively during translation -- no
    # separate extract/wire post-process is needed.  ``output_port_index=
    # reheat_cmd`` picks the CITS actuator slot for this command (CITS.input
    # Signal is a Vector indexed by actuator).
    sp.add_connection(
        reheat_cmd,
        "inputSignal",
        "valvePosition",
        output_port_index=reheat_cmd,
    )
    sp.add_modeled_node(vav)

    return sp
