"""Example signature patterns for :mod:`twin4build.systems.sensor.sensor_system`.

Moved out of the system module (#200); examples of a graph shape, not a
standard.  Bound to their classes by :mod:`twin4build.examples.patterns`.
"""

from twin4build.translator.translator import (
    ModeledNode,
    Node,
    OptionalRule,
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


def get_flow_signature_pattern_after_coil_air_side():
    node0 = Node(cls=(core.namespace.SAREF.Sensor,))
    node1 = Node(cls=(core.namespace.SAREF.Temperature,))
    node2 = Node(cls=(core.namespace.S4BLDG.Coil))  # waterside
    node3 = Node(cls=(core.namespace.S4BLDG.Coil))  # airside
    node4 = Node(cls=(core.namespace.S4BLDG.Coil))  # supersystem
    node5 = Node(cls=core.namespace.S4SYST.System)  # before waterside
    node6 = Node(cls=core.namespace.S4SYST.System)  # after waterside
    node7 = Node(cls=core.namespace.S4SYST.System)  # before airside
    node8 = Node(cls=core.namespace.S4SYST.System)  # after airside
    sp = SignaturePattern(id="flow_signature_pattern_after_coil_air_side")
    sp.add_rule(
        StepRule(subject=node0, object=node1, predicate=core.namespace.SAREF.observes)
    )
    sp.add_rule(
        StepRule(subject=node5, object=node2, predicate=core.namespace.FSO.suppliesFluidTo)
    )
    sp.add_rule(
        StepRule(subject=node2, object=node6, predicate=core.namespace.FSO.returnsFluidTo)
    )
    sp.add_rule(
        StepRule(subject=node7, object=node3, predicate=core.namespace.FSO.suppliesFluidTo)
    )
    sp.add_rule(
        StepRule(subject=node3, object=node8, predicate=core.namespace.FSO.suppliesFluidTo)
    )
    sp.add_rule(
        StepRule(subject=node2, object=node4, predicate=core.namespace.S4SYST.subSystemOf)
    )
    sp.add_rule(
        StepRule(subject=node3, object=node4, predicate=core.namespace.S4SYST.subSystemOf)
    )
    sp.add_rule(
        PathRule(
            subject=node3, object=node0, predicate=core.namespace.FSO.suppliesFluidTo
        )
    )
    sp.add_input("measuredValue", node4, ("outletAirTemperature"))
    sp.add_modeled_node(node0)
    return sp


def get_flow_signature_pattern_after_coil_air_side_simple():
    node0 = Node(cls=(core.namespace.SAREF.Sensor,))
    node1 = Node(cls=(core.namespace.SAREF.Temperature,))
    node3 = Node(cls=(core.namespace.S4BLDG.Coil))  # airside
    node4 = Node(cls=(core.namespace.S4BLDG.Coil))  # supersystem
    sp = SignaturePattern(
        id="flow_signature_pattern_after_coil_air_side_simple",
    )
    sp.add_rule(
        StepRule(subject=node0, object=node1, predicate=core.namespace.SAREF.observes)
    )
    sp.add_rule(
        StepRule(subject=node3, object=node4, predicate=core.namespace.S4SYST.subSystemOf)
    )
    sp.add_rule(
        PathRule(
            subject=node3, object=node0, predicate=core.namespace.FSO.suppliesFluidTo
        )
    )
    sp.add_input("measuredValue", node4, ("outletAirTemperature"))
    sp.add_modeled_node(node0)
    return sp


def get_flow_signature_pattern_after_coil_water_side():
    node0 = Node(cls=(core.namespace.SAREF.Sensor,))
    node1 = Node(cls=(core.namespace.SAREF.Temperature,))
    node2 = Node(cls=(core.namespace.S4BLDG.Coil))  # waterside
    node3 = Node(cls=(core.namespace.S4BLDG.Coil))  # airside
    node4 = Node(cls=(core.namespace.S4BLDG.Coil))  # supersystem
    node5 = Node(cls=core.namespace.S4SYST.System)  # before waterside
    node6 = Node(cls=core.namespace.S4SYST.System)  # after waterside
    node7 = Node(cls=core.namespace.S4SYST.System)  # before airside
    node8 = Node(cls=core.namespace.S4SYST.System)  # after airside
    sp = SignaturePattern(
        id="flow_signature_pattern_after_coil_water_side",
    )
    sp.add_rule(
        StepRule(subject=node0, object=node1, predicate=core.namespace.SAREF.observes)
    )
    sp.add_rule(
        StepRule(subject=node5, object=node2, predicate=core.namespace.FSO.suppliesFluidTo)
    )
    sp.add_rule(
        StepRule(subject=node2, object=node6, predicate=core.namespace.FSO.returnsFluidTo)
    )
    sp.add_rule(
        StepRule(subject=node7, object=node3, predicate=core.namespace.FSO.suppliesFluidTo)
    )
    sp.add_rule(
        StepRule(subject=node3, object=node8, predicate=core.namespace.FSO.suppliesFluidTo)
    )
    sp.add_rule(
        StepRule(subject=node2, object=node4, predicate=core.namespace.S4SYST.subSystemOf)
    )
    sp.add_rule(
        StepRule(subject=node3, object=node4, predicate=core.namespace.S4SYST.subSystemOf)
    )
    sp.add_rule(
        PathRule(
            subject=node2, object=node0, predicate=core.namespace.FSO.returnsFluidTo
        )
    )
    sp.add_input("measuredValue", node4, ("outletWaterTemperature"))
    sp.add_modeled_node(node0)
    return sp


def get_flow_signature_pattern_before_coil_water_side():
    node0 = Node(cls=(core.namespace.SAREF.Sensor,))
    node1 = Node(cls=(core.namespace.SAREF.Temperature,))
    node2 = Node(cls=(core.namespace.S4BLDG.Coil))  # waterside
    node3 = Node(cls=(core.namespace.S4BLDG.Coil))  # airside
    node4 = Node(cls=(core.namespace.S4BLDG.Coil))  # supersystem
    node6 = Node(cls=core.namespace.S4SYST.System)  # after waterside
    node7 = Node(cls=core.namespace.S4SYST.System)  # before airside
    node8 = Node(cls=core.namespace.S4SYST.System)  # after airside
    sp = SignaturePattern(
        id="flow_signature_pattern_before_coil_water_side",
    )
    sp.add_rule(
        StepRule(subject=node0, object=node1, predicate=core.namespace.SAREF.observes)
    )
    # sp.add_rule(StepRule(subject=node5, object=node2, predicate="suppliesFluidTo"))
    sp.add_rule(
        StepRule(subject=node2, object=node6, predicate=core.namespace.FSO.returnsFluidTo)
    )
    sp.add_rule(
        StepRule(subject=node7, object=node3, predicate=core.namespace.FSO.suppliesFluidTo)
    )
    sp.add_rule(
        StepRule(subject=node3, object=node8, predicate=core.namespace.FSO.suppliesFluidTo)
    )
    sp.add_rule(
        StepRule(subject=node2, object=node4, predicate=core.namespace.S4SYST.subSystemOf)
    )
    sp.add_rule(
        StepRule(subject=node3, object=node4, predicate=core.namespace.S4SYST.subSystemOf)
    )
    sp.add_rule(
        PathRule(
            subject=node2, object=node0, predicate=core.namespace.FSO.hasFluidSuppliedBy
        )
    )
    sp.add_input("measuredValue", node4, ("inletWaterTemperature"))
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


def get_brick_sensor_leaf_pattern():
    """
    Generic BRICK leaf sensor pattern.

    Matches any BRICK Point that has a Brick reference timeseries ID
    (ref:hasExternalReference → ref:hasTimeseriesId). The UUID is extracted and
    assigned to the SensorSystem so it can read from the database.

    This is the fallback pattern for all BRICK sensors that are not matched by a
    more specific virtual-sensor pattern.
    """
    sensor = Node(cls=core.namespace.BRICK.Point)
    externalref = Node(cls=(core.namespace.BRICKREF.ExternalReference, core.BlankNode))
    timeseries_id = Node(cls=core.namespace.XSD.string)

    sp = SignaturePattern(id="brick_sensor_leaf_pattern")
    sp.add_rule(
        StepRule(
            subject=sensor,
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
    sp.add_parameter("uuid", timeseries_id)
    sp.add_modeled_node(sensor)
    return sp


def get_brick_command_sensor_pattern():
    """
    BRICK actuator command sensor pattern.

    Matches any BRICK Command that is a hasPoint of a terminal unit -- a VAV
    on the air side, a space heater on the water side -- and has a timeseries
    UUID.  The SensorSystem holds the measured actuator command (ground truth for
    estimation) and receives the CITS predicted command via inputSignal so that
    the estimator can minimise the error.

    Topology::

        VAV | Space_Heater  hasPoint  <Command>
                          └─ hasExternalReference → <ExternalRef/BNode>
                                                        └─ hasTimeseriesId → <uuid>

    Connection: CITS.inputSignal[i] -> SensorSystem.measuredValue
    where i is the slot index of this command within the CITS actuator groups.

    The sender_node is ``command`` (not ``vav``) so that _sem2sim_map lookup
    finds the CITS (which is modeled on BRICK.Command).  The sensor is
    modeled on ``externalref`` (unique per command timeseries) to avoid
    the MILP mutual-exclusion constraint that would prevent both the CITS
    and this sensor from being active on the same Command entity.
    """
    command = Node(cls=core.namespace.BRICK.Command)
    # The equipment the command actuates.  Space heaters are here so that a
    # radiator valve command (``brick:Heating_Command`` on a
    # ``brick:Space_Heater``) becomes controller-driven exactly like a VAV's
    # damper command; without it the heating loop stays open in Stage 2,
    # because the space heater reads the historised series instead of its
    # controller's output.
    vav = Node(
        cls=(
            core.namespace.BRICK.VAV,
            core.namespace.BRICK.Space_Heater,
            core.namespace.BRICK.Radiator,
            core.namespace.BRICK.Radiant_Panel,
            core.namespace.BRICK.Baseboard_Radiator,
        )
    )
    externalref = Node(cls=(core.namespace.BRICKREF.ExternalReference, core.BlankNode))
    timeseries_id = Node(cls=core.namespace.XSD.string)

    sp = SignaturePattern(id="brick_command_sensor_pattern")
    sp.add_rule(
        StepRule(
            subject=vav,
            object=command,
            predicate=core.namespace.BRICK.hasPoint,
        )
    )
    sp.add_rule(
        StepRule(
            subject=command,
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
    sp.add_parameter("uuid", timeseries_id)
    sp.add_connection(
        command,
        "inputSignal",
        "measuredValue",
        output_port_index=command,
    )
    # Multi-member modeled identity: ``command`` is added alongside
    # ``externalref`` so that ``Translator.sim2sem_map`` carries the
    # ``BRICK.Command`` URI as a key for this stub SensorSystem.  Without
    # ``command`` in the group, ``Model.set_transformations`` cannot see
    # the ``BRICK.Command`` rdf:type on this sensor and silently skips
    # any unit conversion the user mapped for ``BRICK.Command`` (e.g.
    # the 0-100% -> 0-1 lambda used by every Mortar valve command),
    # which leaves the ground truth in 0-100% while the CITS predicts
    # against rewire-seeded output saturation -- producing the
    # characteristic ``rmse ~ 25`` Stage-1 signature.
    #
    # Mirrors the damper-command pattern, where ``ModeledNode(
    # [damper_cmd, externalref])`` already does the same for
    # ``BRICK.Damper_Position_Setpoint``.
    ModeledNode([command, externalref])
    return sp


def get_brick_damper_command_sensor_pattern():
    """
    BRICK damper command sensor — via Damper equipment.

    Damper commands are modeled indirectly through a Damper equipment entity::

        Damper  isPartOf   VAV
        Damper  hasPoint   <Damper_Position_Setpoint>
                              └─ hasExternalReference → <ExternalRef/BNode>
                                                           └─ hasTimeseriesId → <uuid>

    Connection: CITS_damper.inputSignal[0] -> SensorSystem.measuredValue

    The sender_node is ``damper_cmd`` (not ``vav``) so that ``_sem2sim_map``
    lookup finds the damper CITS (modeled on ``BRICK.Damper_Position_Setpoint``).

    Modeled identity is the multi-member group
    ``ModeledNode([damper_cmd, externalref])``.  ``externalref`` keeps the
    original "unique per timeseries" identity so two damper commands with
    different external references do not collide.  ``damper_cmd`` is added
    so Stage-2 ``_sem2sim_map`` carries the ``Damper_Position_Setpoint``
    URI as a key for *this* SensorSystem -- without that key the
    Stage-1 -> Stage-2 controller-extraction merge cannot locate the
    historised damper-command sensor when rewiring an extracted PI
    controller's output to ``AHU.supplyDamperPosition``: the merge looks
    components up by the actuator BRICK URI, which for damper-equipment
    topologies is the ``Damper_Position_Setpoint`` URI.

    Multi-member ``ModeledNode`` groups are mutex-ed per-fingerprint,
    not per-member (see :class:`twin4build.translator.translator.ModeledNode`'s
    "Mutex semantics" section), so the damper CITS (whose own
    ``ModeledNode`` group also contains ``damper_cmd``) and this
    SensorSystem can both bind the same ``Damper_Position_Setpoint`` SM
    node simultaneously.
    """
    damper_cmd = Node(cls=core.namespace.BRICK.Damper_Position_Setpoint)
    damper_equip = Node(cls=core.namespace.BRICK.Damper)
    vav = Node(cls=core.namespace.BRICK.VAV)
    externalref = Node(cls=(core.namespace.BRICKREF.ExternalReference, core.BlankNode))
    timeseries_id = Node(cls=core.namespace.XSD.string)

    sp = SignaturePattern(id="brick_damper_command_sensor_pattern")
    sp.add_rule(
        StepRule(
            subject=damper_equip,
            object=vav,
            predicate=core.namespace.BRICK.isPartOf,
        )
    )
    sp.add_rule(
        StepRule(
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
    sp.add_parameter("uuid", timeseries_id)
    sp.add_connection(
        damper_cmd,
        "inputSignal",
        "measuredValue",
        output_port_index=damper_cmd,
    )
    ModeledNode([damper_cmd, externalref])
    return sp


def get_brick_zone_air_temp_sensor_with_ref_pattern():
    """BRICK Zone_Air_Temperature_Sensor with an external Brick timeseries reference.

    Topology::

        Zone_Air_Temperature_Sensor  isPointOf             VAV
        VAV                          feeds                 Room / HVAC_Zone
        Zone_Air_Temperature_Sensor  hasExternalReference  <ExternalRef/BNode>
                                                                └─ hasTimeseriesId → <uuid>

    The SensorSystem is connected to the room's ``indoorTemperature`` so that
    the CITS / other downstream systems can read the *modelled* zone
    temperature, and the UUID is extracted so the sensor can additionally load
    physical measurements from the database.

    Paired with :func:`get_brick_zone_air_temp_sensor_virtual_pattern`; see the
    module-level note at the top of this section for why the two-pattern split
    is necessary.
    """
    sensor = Node(cls=core.namespace.BRICK.Zone_Air_Temperature_Sensor)
    vav = Node(cls=core.namespace.BRICK.VAV)
    room = Node(
        cls=(
            core.namespace.BRICK.Room,
            core.namespace.BRICK.HVAC_Zone,
            # Brick 1.4 deprecates its location classes in favour of
            # RealEstateCore (``brick:Room brick:isReplacedBy rec:Room``,
            # ``brick:HVAC_Zone`` -> ``rec:HVACZone`` < ``rec:Zone``).
            core.namespace.REC.Room,
            core.namespace.REC.Zone,
            core.namespace.BRICK.Enclosed_space,
            core.namespace.BRICK.Open_space,
        )
    )
    externalref = Node(cls=(core.namespace.BRICKREF.ExternalReference, core.BlankNode))
    timeseries_id = Node(cls=core.namespace.XSD.string)

    sp = SignaturePattern(id="brick_zone_air_temp_sensor_with_ref_pattern")
    sp.add_rule(
        StepRule(
            subject=sensor,
            object=vav,
            predicate=core.namespace.BRICK.isPointOf,
        )
    )
    sp.add_rule(
        StepRule(
            subject=vav,
            object=room,
            predicate=core.namespace.BRICK.feeds,
        )
    )
    sp.add_rule(
        StepRule(
            subject=sensor,
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
    sp.add_parameter("uuid", timeseries_id)
    sp.add_connection(room, "indoorTemperature", "measuredValue")
    sp.add_modeled_node(sensor)
    sp.add_modeled_node(externalref)
    return sp


def get_brick_zone_air_temp_sensor_virtual_pattern():
    """BRICK Zone_Air_Temperature_Sensor without a Brick timeseries reference.

    Topology::

        Zone_Air_Temperature_Sensor  isPointOf  VAV
        VAV                          feeds      Room / HVAC_Zone

    The SensorSystem is connected to the room's ``indoorTemperature`` so the
    modelled zone temperature is still available to downstream systems (CITS,
    controllers, …) even when no physical Brick timeseries is attached.  No
    ``uuid`` parameter is extracted — see the module-level note above for the
    two-pattern design.  Mutually exclusive with
    :func:`get_brick_zone_air_temp_sensor_with_ref_pattern` via the shared
    ``sensor`` modeled node; the with-ref variant wins whenever both match.
    """
    sensor = Node(cls=core.namespace.BRICK.Zone_Air_Temperature_Sensor)
    vav = Node(cls=core.namespace.BRICK.VAV)
    room = Node(
        cls=(
            core.namespace.BRICK.Room,
            core.namespace.BRICK.HVAC_Zone,
            # Brick 1.4 deprecates its location classes in favour of
            # RealEstateCore (``brick:Room brick:isReplacedBy rec:Room``,
            # ``brick:HVAC_Zone`` -> ``rec:HVACZone`` < ``rec:Zone``).
            core.namespace.REC.Room,
            core.namespace.REC.Zone,
            core.namespace.BRICK.Enclosed_space,
            core.namespace.BRICK.Open_space,
        )
    )

    sp = SignaturePattern(id="brick_zone_air_temp_sensor_virtual_pattern")
    sp.add_rule(
        StepRule(
            subject=sensor,
            object=vav,
            predicate=core.namespace.BRICK.isPointOf,
        )
    )
    sp.add_rule(
        StepRule(
            subject=vav,
            object=room,
            predicate=core.namespace.BRICK.feeds,
        )
    )
    sp.add_connection(room, "indoorTemperature", "measuredValue")
    sp.add_modeled_node(sensor)
    return sp


def _brick_room_classes():
    return (
        core.namespace.BRICK.Room,
        core.namespace.BRICK.HVAC_Zone,
        core.namespace.BRICK.Enclosed_space,
        core.namespace.BRICK.Open_space,
        core.namespace.REC.Room,
        core.namespace.REC.Zone,
    )


def get_brick_room_zone_air_temp_sensor_with_ref_pattern():
    """BRICK Zone_Air_Temperature_Sensor attached to the *room*, with a timeseries reference.

    BMS-derived graphs (e.g. Hoeje-Taastrup Raadhus) hang the zone
    temperature sensor off the room rather than off the VAV serving it::

        Zone_Air_Temperature_Sensor  isPointOf             Room / Zone
        Zone_Air_Temperature_Sensor  hasExternalReference  <ExternalRef/BNode>
                                                                +- hasTimeseriesId -> <uuid>

    Same wiring as :func:`get_brick_zone_air_temp_sensor_with_ref_pattern`
    (``room.indoorTemperature -> sensor.measuredValue``); paired with
    :func:`get_brick_room_zone_air_temp_sensor_virtual_pattern` following
    the two-pattern split explained above.
    """
    sensor = Node(cls=core.namespace.BRICK.Zone_Air_Temperature_Sensor)
    room = Node(cls=_brick_room_classes())
    externalref = Node(cls=(core.namespace.BRICKREF.ExternalReference, core.BlankNode))
    timeseries_id = Node(cls=core.namespace.XSD.string)

    sp = SignaturePattern(id="brick_room_zone_air_temp_sensor_with_ref_pattern")
    sp.add_rule(
        StepRule(subject=sensor, object=room, predicate=core.namespace.BRICK.isPointOf)
    )
    sp.add_rule(
        StepRule(
            subject=sensor,
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
    sp.add_parameter("uuid", timeseries_id)
    sp.add_connection(room, "indoorTemperature", "measuredValue")
    sp.add_modeled_node(sensor)
    sp.add_modeled_node(externalref)
    return sp


def get_brick_room_zone_air_temp_sensor_virtual_pattern():
    """BRICK Zone_Air_Temperature_Sensor attached to the *room*, no timeseries reference.

    Topology::

        Zone_Air_Temperature_Sensor  isPointOf  Room / Zone

    Mutually exclusive with
    :func:`get_brick_room_zone_air_temp_sensor_with_ref_pattern` via the
    shared ``sensor`` modeled node; the with-ref variant wins when both match.
    """
    sensor = Node(cls=core.namespace.BRICK.Zone_Air_Temperature_Sensor)
    room = Node(cls=_brick_room_classes())

    sp = SignaturePattern(id="brick_room_zone_air_temp_sensor_virtual_pattern")
    sp.add_rule(
        StepRule(subject=sensor, object=room, predicate=core.namespace.BRICK.isPointOf)
    )
    sp.add_connection(room, "indoorTemperature", "measuredValue")
    sp.add_modeled_node(sensor)
    return sp


def get_brick_room_zone_co2_sensor_with_ref_pattern():
    """BRICK Zone_CO2_Level_Sensor attached to the room, with a timeseries reference.

    Topology::

        Zone_CO2_Level_Sensor  isPointOf             Room / Zone
        Zone_CO2_Level_Sensor  hasExternalReference  <ExternalRef/BNode>
                                                         +- hasTimeseriesId -> <uuid>

    Wires the BuildingSpace mass-balance output ``indoorCO2`` into the
    sensor so simulated and measured CO2 can be compared.
    """
    sensor = Node(cls=core.namespace.BRICK.Zone_CO2_Level_Sensor)
    room = Node(cls=_brick_room_classes())
    externalref = Node(cls=(core.namespace.BRICKREF.ExternalReference, core.BlankNode))
    timeseries_id = Node(cls=core.namespace.XSD.string)

    sp = SignaturePattern(id="brick_room_zone_co2_sensor_with_ref_pattern")
    sp.add_rule(
        StepRule(subject=sensor, object=room, predicate=core.namespace.BRICK.isPointOf)
    )
    sp.add_rule(
        StepRule(
            subject=sensor,
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
    sp.add_parameter("uuid", timeseries_id)
    sp.add_connection(room, "indoorCO2", "measuredValue")
    sp.add_modeled_node(sensor)
    sp.add_modeled_node(externalref)
    return sp


def get_brick_ahu_supply_air_temp_sensor_with_ref_pattern():
    """BRICK Supply_Air_Temperature_Sensor on an AHU, with a Brick timeseries reference.

    Topology::

        Supply_Air_Temperature_Sensor  isPointOf             AHU
        Supply_Air_Temperature_Sensor  hasExternalReference  <ExternalRef/BNode>
                                                                 └─ hasTimeseriesId → <uuid>

    Paired with :func:`get_brick_ahu_supply_air_temp_sensor_virtual_pattern`;
    see the module-level note above.
    """
    # The Preheat_ subclass (heat-recovery outlet) has its own pattern.
    sensor = Node(
        cls=core.namespace.BRICK.Supply_Air_Temperature_Sensor,
        exclude=core.namespace.BRICK.Preheat_Supply_Air_Temperature_Sensor,
    )
    ahu = Node(cls=core.namespace.BRICK.AHU)
    externalref = Node(cls=(core.namespace.BRICKREF.ExternalReference, core.BlankNode))
    timeseries_id = Node(cls=core.namespace.XSD.string)

    sp = SignaturePattern(id="brick_ahu_supply_air_temp_sensor_with_ref_pattern")
    sp.add_rule(
        StepRule(
            subject=sensor,
            object=ahu,
            predicate=core.namespace.BRICK.isPointOf,
        )
    )
    sp.add_rule(
        StepRule(
            subject=sensor,
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
    sp.add_parameter("uuid", timeseries_id)
    sp.add_connection(ahu, "supplyAirTemperature", "measuredValue")
    sp.add_modeled_node(sensor)
    sp.add_modeled_node(externalref)
    return sp


def get_brick_supply_air_flow_sensor_with_ref_pattern():
    """BRICK Supply_Air_Flow_Sensor at a VAV branch with timeseries reference.

    Topology (e.g. Mortar bldg1)::

        Supply_Air_Flow_Sensor  isPointOf             VAV
        VAV                     feeds                 Room / HVAC_Zone
        AHU                     feeds                 VAV
        Supply_Air_Flow_Sensor  hasExternalReference  <ExternalRef/BNode>
                                                          └─ hasTimeseriesId → <uuid>

    Wires the AHU's per-branch ``supplyAirFlowRate`` Vector output at this
    space's slot into the SensorSystem's ``measuredValue`` input.  The
    Vector slot key is the matched ``room`` URI -- the same key the AHU
    pattern uses for its ``supplyAirFlowRate`` / ``supplyDamperPosition``
    Vectors (``input_port_index=spaces``) and the BuildingSpace pattern
    uses for its ``output_port_index=space`` consumption, so all three
    end up aligned on the same per-zone slot.

    Result: ``SensorSystem.output["measuredValue"]`` carries the
    *simulated* branch flow each step, while ``time_series_input.values``
    (loaded via the extracted ``uuid`` + ``dbconfig`` from
    ``_prepare_stage1_model``) carries the DB-recorded *measured* flow.
    The downstream plot block can then compare sim vs measured branch
    flow per zone -- the same convention as zone temperature.

    Paired with :func:`get_brick_supply_air_flow_sensor_virtual_pattern`
    via the shared ``sensor`` modeled node; with-ref wins when both match.
    """
    sensor = Node(cls=core.namespace.BRICK.Supply_Air_Flow_Sensor)
    vav = Node(cls=core.namespace.BRICK.VAV)
    ahu = Node(cls=core.namespace.BRICK.AHU)
    room = Node(
        cls=(
            core.namespace.BRICK.Room,
            core.namespace.BRICK.HVAC_Zone,
            # Brick 1.4 deprecates its location classes in favour of
            # RealEstateCore (``brick:Room brick:isReplacedBy rec:Room``,
            # ``brick:HVAC_Zone`` -> ``rec:HVACZone`` < ``rec:Zone``).
            core.namespace.REC.Room,
            core.namespace.REC.Zone,
            core.namespace.BRICK.Enclosed_space,
            core.namespace.BRICK.Open_space,
        )
    )
    externalref = Node(cls=(core.namespace.BRICKREF.ExternalReference, core.BlankNode))
    timeseries_id = Node(cls=core.namespace.XSD.string)

    sp = SignaturePattern(id="brick_supply_air_flow_sensor_with_ref_pattern")
    sp.add_rule(
        StepRule(
            subject=sensor, object=vav, predicate=core.namespace.BRICK.isPointOf
        )
    )
    sp.add_rule(
        StepRule(
            subject=vav, object=room, predicate=core.namespace.BRICK.feeds
        )
    )
    sp.add_rule(
        StepRule(
            subject=ahu, object=vav, predicate=core.namespace.BRICK.feeds
        )
    )
    sp.add_rule(
        StepRule(
            subject=sensor,
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
    sp.add_parameter("uuid", timeseries_id)
    sp.add_connection(
        ahu,
        "supplyAirFlowRate",
        "measuredValue",
        output_port_index=vav,
    )
    sp.add_modeled_node(sensor)
    sp.add_modeled_node(externalref)
    return sp


def get_brick_supply_air_flow_sensor_virtual_pattern():
    """BRICK Supply_Air_Flow_Sensor at a VAV branch without timeseries reference.

    Topology::

        Supply_Air_Flow_Sensor  isPointOf  VAV
        VAV                     feeds      Room / HVAC_Zone
        AHU                     feeds      VAV

    Wires ``AHU.supplyAirFlowRate[space_slot] -> SensorSystem.measuredValue``
    so the simulated flow is still observable as a SensorSystem output
    even when no Brick timeseries reference is attached.  Mutually
    exclusive with :func:`get_brick_supply_air_flow_sensor_with_ref_pattern`
    via the shared ``sensor`` modeled node; with-ref wins when both match.
    """
    sensor = Node(cls=core.namespace.BRICK.Supply_Air_Flow_Sensor)
    vav = Node(cls=core.namespace.BRICK.VAV)
    ahu = Node(cls=core.namespace.BRICK.AHU)
    room = Node(
        cls=(
            core.namespace.BRICK.Room,
            core.namespace.BRICK.HVAC_Zone,
            # Brick 1.4 deprecates its location classes in favour of
            # RealEstateCore (``brick:Room brick:isReplacedBy rec:Room``,
            # ``brick:HVAC_Zone`` -> ``rec:HVACZone`` < ``rec:Zone``).
            core.namespace.REC.Room,
            core.namespace.REC.Zone,
            core.namespace.BRICK.Enclosed_space,
            core.namespace.BRICK.Open_space,
        )
    )

    sp = SignaturePattern(id="brick_supply_air_flow_sensor_virtual_pattern")
    sp.add_rule(
        StepRule(
            subject=sensor, object=vav, predicate=core.namespace.BRICK.isPointOf
        )
    )
    sp.add_rule(
        StepRule(
            subject=vav, object=room, predicate=core.namespace.BRICK.feeds
        )
    )
    sp.add_rule(
        StepRule(
            subject=ahu, object=vav, predicate=core.namespace.BRICK.feeds
        )
    )
    sp.add_connection(
        ahu,
        "supplyAirFlowRate",
        "measuredValue",
        output_port_index=vav,
    )
    sp.add_modeled_node(sensor)
    return sp


def get_brick_ahu_supply_air_temp_sensor_virtual_pattern():
    """BRICK Supply_Air_Temperature_Sensor on an AHU, without a Brick timeseries reference.

    Topology::

        Supply_Air_Temperature_Sensor  isPointOf  AHU

    Mutually exclusive with
    :func:`get_brick_ahu_supply_air_temp_sensor_with_ref_pattern` via the
    shared ``sensor`` modeled node; the with-ref variant wins whenever both
    match.
    """
    # The Preheat_ subclass (heat-recovery outlet) has its own pattern.
    sensor = Node(
        cls=core.namespace.BRICK.Supply_Air_Temperature_Sensor,
        exclude=core.namespace.BRICK.Preheat_Supply_Air_Temperature_Sensor,
    )
    ahu = Node(cls=core.namespace.BRICK.AHU)

    sp = SignaturePattern(id="brick_ahu_supply_air_temp_sensor_virtual_pattern")
    sp.add_rule(
        StepRule(
            subject=sensor,
            object=ahu,
            predicate=core.namespace.BRICK.isPointOf,
        )
    )
    sp.add_connection(ahu, "supplyAirTemperature", "measuredValue")
    sp.add_modeled_node(sensor)
    return sp


def _brick_ahu_flow_sensor_pattern(sensor_cls, output_port: str, sp_id: str, with_ref: bool):
    """An AHU-level air-flow sensor: the total over the AHU's branches.

    Topology::

        <Supply|Return>_Air_Flow_Sensor  isPointOf             AHU
        <Supply|Return>_Air_Flow_Sensor  hasExternalReference  <ExternalRef/BNode>   (with_ref)
                                                                   └─ hasTimeseriesId → <uuid>

    Wires ``AHU.totalSupplyAirFlowRate`` (supply) or
    ``AHU.totalExhaustAirFlowRate`` (return) to ``measuredValue``, so the
    AHU's own flow meters calibrate the branch flows in sum, next to the
    per-VAV sensors (:func:`get_brick_supply_air_flow_sensor_with_ref_pattern`)
    that calibrate them one by one.  The with-ref / virtual pair follows
    the module-level note above.
    """
    sensor = Node(cls=sensor_cls)
    ahu = Node(cls=core.namespace.BRICK.AHU)
    sp = SignaturePattern(id=sp_id)
    sp.add_rule(StepRule(subject=sensor, object=ahu, predicate=core.namespace.BRICK.isPointOf))
    if with_ref:
        externalref = Node(cls=(core.namespace.BRICKREF.ExternalReference, core.BlankNode))
        timeseries_id = Node(cls=core.namespace.XSD.string)
        sp.add_rule(
            StepRule(subject=sensor, object=externalref, predicate=core.namespace.BRICKREF.hasExternalReference)
        )
        sp.add_rule(
            StepRule(subject=externalref, object=timeseries_id, predicate=core.namespace.BRICKREF.hasTimeseriesId)
        )
        sp.add_parameter("uuid", timeseries_id)
        sp.add_modeled_node(externalref)
    sp.add_connection(ahu, output_port, "measuredValue")
    sp.add_modeled_node(sensor)
    return sp


def get_brick_ahu_supply_air_flow_sensor_with_ref_pattern():
    return _brick_ahu_flow_sensor_pattern(
        core.namespace.BRICK.Supply_Air_Flow_Sensor, "totalSupplyAirFlowRate",
        "brick_ahu_supply_air_flow_sensor_with_ref_pattern", with_ref=True,
    )


def get_brick_ahu_supply_air_flow_sensor_virtual_pattern():
    return _brick_ahu_flow_sensor_pattern(
        core.namespace.BRICK.Supply_Air_Flow_Sensor, "totalSupplyAirFlowRate",
        "brick_ahu_supply_air_flow_sensor_virtual_pattern", with_ref=False,
    )


def get_brick_ahu_return_air_flow_sensor_with_ref_pattern():
    return _brick_ahu_flow_sensor_pattern(
        core.namespace.BRICK.Return_Air_Flow_Sensor, "totalExhaustAirFlowRate",
        "brick_ahu_return_air_flow_sensor_with_ref_pattern", with_ref=True,
    )


def get_brick_ahu_return_air_flow_sensor_virtual_pattern():
    return _brick_ahu_flow_sensor_pattern(
        core.namespace.BRICK.Return_Air_Flow_Sensor, "totalExhaustAirFlowRate",
        "brick_ahu_return_air_flow_sensor_virtual_pattern", with_ref=False,
    )
