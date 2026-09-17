"""Example signature patterns for :mod:`twin4build.systems.building_space.building_space_system`.

Moved out of the system module (#200): patterns describe how one kind
of graph maps onto the component, and are examples of that, not a
standard.  Bound to their classes by :mod:`twin4build.examples.patterns`.
"""

from twin4build.translator.translator import (
    SetStepRule,
    StepRule,
    AnyPathRule,
    Node,
    NoStepRule,
    OptionalRule,
    Predicate,
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


_BRICK_SPACE_CLASSES = (
    core.namespace.BRICK.Room,
    core.namespace.BRICK.Enclosed_space,
    core.namespace.BRICK.Open_space,
    core.namespace.BRICK.HVAC_Zone,
    # Brick 1.4 deprecates its location classes in favour of
    # RealEstateCore (``brick:Room brick:isReplacedBy rec:Room``,
    # ``brick:HVAC_Zone`` -> ``rec:HVACZone`` < ``rec:Zone``).
    core.namespace.REC.Room,
    core.namespace.REC.Zone,
    core.namespace.BOT.Space,
)


_SOLAR_SENSOR_CLASSES = (
    # ``Global_Solar_Irradiation_Sensor`` is not a Brick class (it survives
    # for graphs that extend Brick with it); ``Solar_Irradiance_Sensor`` is
    # the Brick 1.4 class (W/m2).
    core.namespace.BRICK.Global_Solar_Irradiation_Sensor,
    core.namespace.BRICK.Solar_Irradiance_Sensor,
)


_REHEAT_PART_CLASSES = (
    core.namespace.BRICK.Heating_Coil,
    core.namespace.BRICK.Cooling_Coil,
    core.namespace.BRICK.Reheat_Valve,
)


_REHEAT_POINT_CLASSES = (
    core.namespace.BRICK.Reheat_Command,
    core.namespace.BRICK.Heating_Command,
    core.namespace.BRICK.Valve_Command,
)


def _add_brick_volume_parameter(sp, space):
    """Bind the room volume ``space brick:volume [ brick:value x ]`` to ``mass.V``.

    The chain is *required* and the value holder is a modeled node: with an
    ``OptionalRule`` chain the translator's disconnected-merge step treats
    the free literal as a shared resource and copies one room's volume into
    every other room (the same failure mode documented for the sensor
    ``externalref`` chains).  Every Brick space pattern is therefore
    registered twice -- with and without this chain -- and the MILP prefers
    the with-volume variant (one more modeled node) whenever the graph
    carries the geometry.
    """
    volume_node = Node(cls=(core.BlankNode,))
    volume_value = Node(
        cls=(
            core.namespace.XSD.float,
            core.namespace.XSD.double,
            core.namespace.XSD.decimal,
            core.namespace.XSD.integer,
        )
    )
    sp.add_rule(StepRule(subject=space, object=volume_node, predicate=core.namespace.BRICK.volume))
    sp.add_rule(StepRule(subject=volume_node, object=volume_value, predicate=core.namespace.BRICK.value))
    sp.add_parameter("mass.V", volume_value)
    sp.add_modeled_node(volume_node)


def _brick_space_pattern(topology: str, with_volume: bool, heat_source: str = "command"):
    """Factory for the Brick building-space patterns.

    ``topology``:

    * ``"vav_no_reheat"`` -- ``AHU feeds VAV feeds Room`` where the VAV is a
      plain damper box (no coil part, no reheat / heating / valve command
      point): the room receives ``AHU.supplyAirTemperature``.
    * ``"vav"`` -- ``AHU feeds VAV feeds Room`` with reheat: the room receives
      ``VAV.outletAirTemperature`` from a :class:`FanCoilUnitSystem`.
    * ``"direct"`` -- AHU feeds the room through any path *not* via a VAV
      (Mortar site A style): the room receives ``AHU.supplyAirTemperature``.

    ``heat_source`` says where the optional ``heatGain`` comes from -- a
    pattern can feed only one node into a port, so the two radiator
    topologies are separate patterns:

    * ``"command"`` -- the room carries a bare ``brick:Heating_Command``
      point and :class:`SpaceHeaterSystem` is modelled on
      ``[room, command]``;
    * ``"equipment"`` -- an explicit space heater is located in the room and
      carries the command; the radiator is modelled on the equipment.

    Both are optional rules, so a room with no radiator at all matches
    either pattern and ``fill_missing_inputs`` supplies a constant.

    All variants share the ``space`` modeled node, so the MILP keeps one per
    room; ``with_volume`` adds the ``brick:volume`` parameter chain (see
    :func:`_add_brick_volume_parameter`).
    """
    ahu = Node(cls=core.namespace.BRICK.AHU)
    vav = Node(cls=core.namespace.BRICK.VAV)
    space = Node(cls=_BRICK_SPACE_CLASSES)
    solar_radiance_sensor = Node(cls=_SOLAR_SENSOR_CLASSES)
    outside_air_temperature_sensor = Node(
        cls=core.namespace.BRICK.Outside_Air_Temperature_Sensor
    )
    # Room-level radiator: BMS graphs carry only a valve command on the room
    # (no radiator equipment).  When a :class:`SpaceHeaterSystem` is matched
    # there (see its ``brick_signature_pattern_room_heating_command``), its
    # delivered ``Power`` is the room's ``heatGain``.  Optional, so rooms
    # without heating still match.
    heating_cmd = Node(cls=core.namespace.BRICK.Heating_Command)
    space_heater = Node(
        cls=(
            core.namespace.BRICK.Space_Heater,
            core.namespace.BRICK.Radiator,
            core.namespace.BRICK.Radiant_Panel,
            core.namespace.BRICK.Baseboard_Radiator,
        )
    )
    located_in = Predicate(
        (core.namespace.BRICK.feeds, core.namespace.FSO.feedsFluidTo)
    )
    feeds = Predicate((core.namespace.BRICK.feeds, core.namespace.FSO.feedsFluidTo))

    suffix = "_with_volume" if with_volume else ""
    heat_suffix = "" if heat_source == "command" else f"_{heat_source}_heat"
    sp = SignaturePattern(
        id=f"building_space_signature_pattern_brick_{topology}{suffix}{heat_suffix}"
    )
    sp.add_node(solar_radiance_sensor, optional=True)  # not always present
    sp.add_node(outside_air_temperature_sensor, optional=True)  # not always present

    if topology == "direct":
        sp.add_rule(
            AnyPathRule(subject=ahu, object=space, predicate=feeds, endpoints_only=True)
            & NoStepRule(subject=ahu, object=vav, predicate=feeds)
        )
        sp.add_connection(ahu, "supplyAirTemperature", "supplyAirTemperature")
    else:
        sp.add_rule(StepRule(subject=ahu, object=vav, predicate=feeds))
        sp.add_rule(StepRule(subject=vav, object=space, predicate=feeds))
        if topology == "vav_no_reheat":
            sp.add_rule(
                NoStepRule(
                    subject=vav,
                    object=Node(cls=_REHEAT_PART_CLASSES),
                    predicate=core.namespace.BRICK.hasPart,
                )
            )
            sp.add_rule(
                NoStepRule(
                    subject=vav,
                    object=Node(cls=_REHEAT_POINT_CLASSES),
                    predicate=core.namespace.BRICK.hasPoint,
                )
            )
            sp.add_connection(ahu, "supplyAirTemperature", "supplyAirTemperature")
        elif topology == "vav":
            sp.add_connection(vav, "outletAirTemperature", "supplyAirTemperature")
        else:
            raise ValueError(topology)

    # One AHU branch per VAV: the zone's Vector flow ports get one slot per
    # VAV serving it (``input_port_index=vav``), read from that VAV's branch
    # of the AHU (``output_port_index=vav``, the key the AHU damper pattern
    # indexes its branches by).  The direct topology has no VAV: the AHU's
    # single branch for the space lands in slot 0.
    if topology == "direct":
        sp.add_connection(ahu, "supplyAirFlowRate", "supplyAirFlowRate", output_port_index=space)
        sp.add_connection(ahu, "exhaustAirFlowRate", "exhaustAirFlowRate", output_port_index=space)
    else:
        # The room's VAVs as a set (one group per VAV, in lockstep with the
        # AHU pattern's ``vavs``): each VAV is a distinct slot of the zone's
        # Vector flow ports and a distinct branch of the AHU.
        vavs = Node(cls=core.namespace.BRICK.VAV)
        sp.add_rule(SetStepRule(subject=space, object=vavs, predicate=core.namespace.BRICK.isFedBy))
        sp.add_connection(
            ahu, "supplyAirFlowRate", "supplyAirFlowRate",
            output_port_index=vavs, input_port_index=vavs,
        )
        sp.add_connection(
            ahu, "exhaustAirFlowRate", "exhaustAirFlowRate",
            output_port_index=vavs, input_port_index=vavs,
        )
    sp.add_connection(solar_radiance_sensor, "globalIrradiation", "globalIrradiation")
    sp.add_connection(
        outside_air_temperature_sensor, "outdoorTemperature", "outdoorTemperature"
    )
    if heat_source == "command":
        sp.add_rule(
            OptionalRule(
                subject=space,
                object=heating_cmd,
                predicate=core.namespace.BRICK.hasPoint,
            )
        )
        sp.add_connection(heating_cmd, "Power", "heatGain")
    elif heat_source == "equipment":
        sp.add_rule(
            OptionalRule(subject=space_heater, object=space, predicate=located_in)
        )
        sp.add_connection(space_heater, "Power", "heatGain")
    else:
        raise ValueError(heat_source)
    if with_volume:
        _add_brick_volume_parameter(sp, space)
    # Interzonal/boundary coupling is modeled by a separate WallSystem
    # (wired manually, or via a future wall/adjacency signature pattern).
    sp.add_modeled_node(space)
    return sp


def brick_signature_pattern_vav_no_reheat(with_volume: bool = False, heat_source: str = "command"):
    """See :func:`_brick_space_pattern` (``"vav_no_reheat"``)."""
    return _brick_space_pattern("vav_no_reheat", with_volume, heat_source)


def brick_signature_pattern_vav(with_volume: bool = False, heat_source: str = "command"):
    """See :func:`_brick_space_pattern` (``"vav"``); kept for site B / Mortar graphs."""
    return _brick_space_pattern("vav", with_volume, heat_source)


def brick_signature_pattern(with_volume: bool = False, heat_source: str = "command"):  # Fits to site A
    """See :func:`_brick_space_pattern` (``"direct"``)."""
    return _brick_space_pattern("direct", with_volume, heat_source)
