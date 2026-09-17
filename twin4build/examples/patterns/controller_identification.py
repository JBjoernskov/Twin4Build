"""Example signature patterns for :mod:`twin4build.systems.controller.controller_identification.controller_identification_system`.

Moved out of the system module (#200): patterns describe how one kind
of graph maps onto the component, and are examples of that, not a
standard.  Bound to their classes by :mod:`twin4build.examples.patterns`.
"""

from twin4build.translator.translator import (
    StepRule,
    SetStepRule,
    ModeledNode,
    AnyPathRule,
    Node,
    Predicate,
    SignaturePattern,
)
import twin4build.core as core


def brick_signature_pattern_vav():
    """
    BRICK signature pattern for VAV zone controller identification.

    A single pattern matching a VAV box with all directly connected signals:
    feedback sensors, setpoints, and actuator commands (with timeseries IDs).

    Groups are formed as the cross-product of (sensor × setpoint × actuator)
    per VAV.  resolve_port_indices uses unique-value ordinals (not raw group
    indices) so each signal type is correctly indexed independent of the
    cross-product size:

      sensors   → sensorValue[0..n_sensors-1]
      setpoints → setpointValue[0..n_setpoints-1]
      actuators → CITS groups (for SensorSystem command pattern index resolution)

    Only commands with ref:hasTimeseriesId are included (purely logical flags
    like Heating_Mode which lack timeseries data are excluded).

    Topology::

        VAV  hasPoint  <Zone_Air_Temperature_Sensor>    → sensorValue[0]
        VAV  hasPoint  <Supply_Air_Temperature_Sensor>  → sensorValue[1]
        VAV  hasPoint  <Air_Flow_Sensor>                → sensorValue[2]
        VAV  hasPoint  <Supply_Air_Flow_Sensor>         → sensorValue[3]
        VAV  hasPoint  <Zone_Air_Temperature_Setpoint>  → setpointValue[0]
        VAV  hasPoint  <Zone_Air_Temperature_Setpoint>  → setpointValue[1]
        VAV  hasPoint  <Command (with ts-id)>           → (actuator slot for index resolution)
    """
    vav = Node(cls=core.namespace.BRICK.VAV)
    sensors = Node(
        cls=(
            core.namespace.BRICK.Zone_Air_Temperature_Sensor,
            core.namespace.BRICK.Supply_Air_Temperature_Sensor,
            core.namespace.BRICK.Air_Flow_Sensor,
            core.namespace.BRICK.Supply_Air_Flow_Sensor,
        )
    )
    setpoints = Node(cls=core.namespace.BRICK.Zone_Air_Temperature_Setpoint)
    actuators = Node(cls=core.namespace.BRICK.Command)
    externalref = Node(cls=(core.namespace.BRICKREF.ExternalReference, core.BlankNode))
    timeseries_id = Node(cls=core.namespace.XSD.string)

    sp = SignaturePattern(id="controller_identification_vav_brick")
    # The three VAV ``hasPoint`` rules use ``SetStepRule`` to collapse
    # the per-point cross-product into one group per VAV: ``sensors``,
    # ``setpoints`` and ``actuators`` each bind to the *tuple of all*
    # matching points. Downstream ``StepRule`` hops (the
    # actuator → externalref → timeseries_id chain) are auto-broadcast
    # per element by the matcher so they remain scalar rules here.
    sp.add_rule(
        SetStepRule(
            subject=vav, object=sensors, predicate=core.namespace.BRICK.hasPoint
        )
    )
    sp.add_rule(
        SetStepRule(
            subject=vav, object=setpoints, predicate=core.namespace.BRICK.hasPoint
        )
    )
    sp.add_rule(
        SetStepRule(
            subject=vav, object=actuators, predicate=core.namespace.BRICK.hasPoint
        )
    )
    sp.add_rule(
        StepRule(
            subject=actuators,
            object=externalref,
            predicate=core.namespace.BRICKREF.hasExternalReference,
        )
    )
    sp.add_rule(
        StepRule(
            subject=externalref,
            object=timeseries_id,
            predicate=core.namespace.BRICKREF.hasTimeseriesId,
        )
    )

    sp.add_connection(sensors, "measuredValue", "sensorValue", input_port_index=sensors)
    sp.add_connection(
        setpoints, "measuredValue", "setpointValue", input_port_index=setpoints
    )
    # Auto-mirror every setpoint into the gate-input bus.  See the
    # corresponding pattern on ``ControllerIdentificationPISystem``
    # for the full rationale -- in short, the ``onOffSignal`` port is
    # never pruned by the rewire and gives the gate access to the
    # schedule even when the rewire winner picks a different setpoint
    # for the PI error term.
    sp.add_connection(
        setpoints, "measuredValue", "onOffSignal", input_port_index=setpoints
    )
    # The VAV controller entity is not a first-class node in BRICK; it is
    # identified jointly by the VAV, its sensor/setpoint points and its
    # command actuators. Expressing this as a ``ModeledNode`` group makes
    # the composite identity explicit and leaves the member SM nodes
    # (notably the ``BRICK.Command`` actuators) available for other
    # systems (e.g. ``SensorSystem``) to model on their own via the
    # non-exclusive mutex semantics.
    ModeledNode([vav, sensors, setpoints, actuators])
    return sp


def brick_signature_pattern_vav_damper():
    """
    BRICK signature pattern for VAV damper controller identification.

    Damper commands are modeled indirectly in BRICK via a Damper equipment
    entity rather than as a direct hasPoint of the VAV:

        Damper  isPartOf   VAV
        Damper  hasPoint   <Damper_Position_Setpoint (with ts-id)>

    Sensors and setpoints are still direct hasPoint of the VAV.
    Each matched damper command becomes its own CITS instance
    (one CITS per actuator, n_actuators=1).
    """
    vav = Node(cls=core.namespace.BRICK.VAV)
    sensors = Node(
        cls=(
            core.namespace.BRICK.Zone_Air_Temperature_Sensor,
            core.namespace.BRICK.Supply_Air_Temperature_Sensor,
            core.namespace.BRICK.Air_Flow_Sensor,
            core.namespace.BRICK.Supply_Air_Flow_Sensor,
        )
    )
    setpoints = Node(cls=core.namespace.BRICK.Zone_Air_Temperature_Setpoint)
    damper_equip = Node(cls=core.namespace.BRICK.Damper)
    damper_cmd = Node(cls=core.namespace.BRICK.Damper_Position_Setpoint)
    externalref = Node(cls=(core.namespace.BRICKREF.ExternalReference, core.BlankNode))
    timeseries_id = Node(cls=core.namespace.XSD.string)

    sp = SignaturePattern(id="controller_identification_vav_damper_brick")
    # ``sensors`` and ``setpoints`` collect all points per VAV as sets;
    # ``damper_cmd`` collects all damper commands per damper equipment.
    # The scalar edges (isPartOf, the actuator → externalref →
    # timeseries_id chain) stay as plain ``StepRule`` and are
    # auto-broadcast over the set-bound endpoints.
    sp.add_rule(
        SetStepRule(
            subject=vav, object=sensors, predicate=core.namespace.BRICK.hasPoint
        )
    )
    sp.add_rule(
        SetStepRule(
            subject=vav, object=setpoints, predicate=core.namespace.BRICK.hasPoint
        )
    )
    sp.add_rule(
        StepRule(
            subject=damper_equip, object=vav, predicate=core.namespace.BRICK.isPartOf
        )
    )
    sp.add_rule(
        SetStepRule(
            subject=damper_equip,
            object=damper_cmd,
            predicate=core.namespace.BRICK.hasPoint,
        )
    )
    sp.add_rule(
        StepRule(
            subject=damper_cmd,
            object=externalref,
            predicate=core.namespace.BRICKREF.hasExternalReference,
        )
    )
    sp.add_rule(
        StepRule(
            subject=externalref,
            object=timeseries_id,
            predicate=core.namespace.BRICKREF.hasTimeseriesId,
        )
    )

    sp.add_connection(sensors, "measuredValue", "sensorValue", input_port_index=sensors)
    sp.add_connection(
        setpoints, "measuredValue", "setpointValue", input_port_index=setpoints
    )
    # Auto-mirror setpoints into the gate-input bus (see sibling
    # pattern for rationale).
    sp.add_connection(
        setpoints, "measuredValue", "onOffSignal", input_port_index=setpoints
    )
    # See ``brick_signature_pattern_vav``: the damper controller is an
    # implicit entity, identified by the (VAV, sensors, setpoints,
    # damper equipment, damper command) tuple.
    ModeledNode([vav, sensors, setpoints, damper_equip, damper_cmd])
    return sp
