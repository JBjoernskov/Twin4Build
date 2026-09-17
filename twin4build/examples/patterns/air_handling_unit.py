"""Example signature patterns for :mod:`twin4build.systems.air_handling_unit.air_handling_unit_system`.

Moved out of the system module (#200): patterns describe how one kind
of graph maps onto the component, and are examples of that, not a
standard.  Bound to their classes by :mod:`twin4build.examples.patterns`.
"""

from twin4build.translator.translator import (
    AnyPathRule,
    ModeledNode,
    Node,
    OptionalRule,
    Predicate,
    SetAnyPathRule,
    SetStepRule,
    SignaturePattern,
    StepRule,
)
import twin4build.core as core


def brick_signature_pattern():
    """
    Signature pattern for an AHU composed of damper, coil, and heat recovery.
    """
    sp = SignaturePattern(id="air_handling_unit_signature_pattern_brick")

    ahu = Node(cls=core.namespace.BRICK.AHU)
    spaces = Node(
        cls=(
            core.namespace.BRICK.Room,
            core.namespace.BRICK.Enclosed_space,
            core.namespace.BRICK.Open_space,
            core.namespace.BRICK.HVAC_Zone,
            # Brick 1.4 deprecates its location classes in favour of
            # RealEstateCore (``brick:Room brick:isReplacedBy rec:Room``,
            # ``brick:HVAC_Zone`` -> ``rec:HVACZone`` < ``rec:Zone``).
            core.namespace.REC.Room,
            core.namespace.REC.Zone,
        )
    )

    feeds = Predicate((core.namespace.BRICK.feeds, core.namespace.FSO.feedsFluidTo))

    # ``SetAnyPathRule`` collapses every Room/HVAC_Zone reachable from
    # the AHU via any feeds-path (e.g. AHU -> VAV -> Space) into a single
    # tuple binding on ``spaces``.  The earlier ``AnyPathRule`` produced
    # one scalar branch per reachable endpoint, yielding ``N_zones``
    # separate AHU components per AHU; the set-bound binding instead
    # gives one AHU component whose ``exhaustTemperature`` Vector input
    # is indexed by the served zones.
    sp.add_rule(SetAnyPathRule(subject=ahu, object=spaces, predicate=feeds))

    sp.add_connection(
        spaces, "indoorTemperature", "exhaustTemperature", input_port_index=spaces
    )

    sp.add_modeled_node(ahu)

    return sp


def brick_signature_pattern_vav_dampers():
    """AHU pattern for VAV systems with per-zone dampers + AHU-level points.

    Mirrors the building's actual topology:

    - AHU ``feeds`` N VAVs (each VAV serves one space).
    - Each VAV has a ``Damper`` (``isPartOf``) with a
      ``Damper_Position_Setpoint`` command point.  That command drives
      the AHU's per-branch ``supplyDamperPosition`` / ``exhaustDamperPosition``
      ``Vector`` inputs (the AHU torch model owns supply/exhaust damper
      sub-systems internally and reshapes the per-branch positions
      across the vectorised :class:`DamperSystem`).
    - The AHU's ``hasPoint`` ``Outside_Air_Temperature_Sensor`` matches
      :class:`OutdoorEnvironmentSystem` (modelled at the same URI), which
      provides the ``outdoorTemperature`` source.
    - The AHU's ``hasPoint`` ``Supply_Air_Temperature_Setpoint`` (when
      present -- not every AHU has one in real BMS data; e.g. Mortar
      bldg1 AHU02 only has ``Supply_Air_Temp``) provides the supply-air
      temperature setpoint.

    The ``Damper`` and ``Damper_Position_Setpoint`` nodes are absorbed
    into the AHU's ``ModeledNode`` group rather than being matched by a
    separate t4b component: the supply/exhaust damper torch sub-systems
    are owned by the AHU and BRICK's per-VAV ``Damper`` equipment does
    not deserve its own component.  Including ``damper_cmds`` in the
    group also means Stage-2 ``_sim2sem_map[ahu]`` carries every per-VAV
    damper-command URI, so the Stage-1 -> Stage-2 controller-extraction
    merge (see
    :func:`twin4build.systems.controller.controller_identification.extractor.wire_extracted_controllers`)
    can locate the historised damper-command ``SensorSystem`` (matched
    by ``brick_sensor_leaf_pattern`` at the same URI) and rewire its
    consumers to the extracted PI controller.
    """
    sp = SignaturePattern(id="air_handling_unit_signature_pattern_brick_vav_dampers")

    ahu = Node(cls=core.namespace.BRICK.AHU)
    spaces = Node(
        cls=(
            core.namespace.BRICK.Room,
            core.namespace.BRICK.Enclosed_space,
            core.namespace.BRICK.Open_space,
            core.namespace.BRICK.HVAC_Zone,
            # Brick 1.4 deprecates its location classes in favour of
            # RealEstateCore (``brick:Room brick:isReplacedBy rec:Room``,
            # ``brick:HVAC_Zone`` -> ``rec:HVACZone`` < ``rec:Zone``).
            core.namespace.REC.Room,
            core.namespace.REC.Zone,
        )
    )
    vavs = Node(cls=core.namespace.BRICK.VAV)
    dampers = Node(cls=core.namespace.BRICK.Damper)
    damper_cmds = Node(cls=core.namespace.BRICK.Damper_Position_Setpoint)
    # ``sat_setpoint`` deliberately matches ONLY ``Supply_Air_Temperature_Setpoint``,
    # never ``Supply_Air_Temperature_Sensor``.  Allowing the sensor class here
    # used to close a positive-feedback loop in buildings that only carry a
    # ``Supply_Air_Temp`` measurement (e.g. Mortar bldg1 AHU02):
    #
    #   * ``brick_signature_pattern_ahu_supply_air_temp`` (sensor_system.py)
    #     wires ``ahu.supplyAirTemperature -> sensor.measuredValue``, i.e. the
    #     sensor is a virtual pass-through of the simulated AHU output.
    #   * If the same sensor URI also bound ``sat_setpoint`` here, the AHU
    #     pattern would wire ``sensor.measuredValue -> ahu.supplyAirTemperature
    #     Setpoint``, and the coil + supply-fan chain would feed
    #     ``setpoint[k+1] = AHU.supplyAirTemperature[k]`` back into itself.
    #     Each step the supply fan adds ``delta_T = P_fan * f_total /
    #     (m_dot * c_p)``, so the loop becomes a pure accumulator
    #     ``AHU.supplyAirTemperature[k+1] = AHU.supplyAirTemperature[k] +
    #     delta_T``.  Over a 10-day run at 600 s step that drifts to ~1000 K.
    #
    # A real BMS setpoint is an input (commanded value), not a measurement of
    # the AHU's own output, so this match is structurally correct.  For AHUs
    # without a setpoint URI the ``OptionalRule`` below simply leaves
    # ``supplyAirTemperatureSetpoint`` unwired; ``fill_missing_inputs`` (or
    # the user) is expected to supply a default.
    sat_setpoint = Node(cls=core.namespace.BRICK.Supply_Air_Temperature_Setpoint)
    oat_sensor = Node(cls=core.namespace.BRICK.Outside_Air_Temperature_Sensor)

    feeds = Predicate((core.namespace.BRICK.feeds, core.namespace.FSO.feedsFluidTo))

    # AHU -> VAVs (multi-hop, set-bound).  ``SetAnyPathRule`` BFS-walks
    # the ``feeds`` predicate from the AHU and bundles every reachable
    # VAV into a single tuple binding on ``vavs``; downstream
    # ``StepRule``\ s on the set-bound ``vavs`` SP node auto-broadcast
    # per element (see SetStepRule docs), so we get parallel set-bound
    # bindings for ``spaces``, ``dampers`` and ``damper_cmds`` -- one
    # per VAV -- without the unsupported nested-set chaining.
    #
    # Using ``SetAnyPathRule`` instead of ``SetStepRule`` matters when
    # the BMS graph models AHU -> air-handler-segment -> VAV (i.e. the
    # AHU does not feed VAVs in a single hop): the multi-hop traversal
    # still discovers them, while remaining set-bound at the AHU level
    # so each AHU yields one match instead of one per VAV.
    #
    # We walk:
    #   vav  --feeds--> space          (each VAV serves one space)
    #   vav  --hasPart--> damper       (inverse of ``damper isPartOf vav``;
    #                                    BRICK declares ``hasPart owl:inverseOf
    #                                    isPartOf`` and the SemanticModel
    #                                    reasoner materialises both directions)
    #   damper --hasPoint--> damper_cmd
    #
    # All four set-bound nodes (``vavs``, ``spaces``, ``dampers``,
    # ``damper_cmds``) end up indexed in lockstep, so the translator can
    # align AHU's ``supplyAirFlowRate`` Vector output (indexed by
    # ``spaces`` from the BuildingSpace pattern that consumes it) with
    # ``supplyDamperPosition`` Vector input (indexed by ``spaces`` here).
    sp.add_rule(SetAnyPathRule(subject=ahu, object=vavs, predicate=feeds))
    sp.add_rule(StepRule(subject=vavs, object=spaces, predicate=feeds))
    sp.add_rule(
        StepRule(subject=vavs, object=dampers, predicate=core.namespace.BRICK.hasPart)
    )
    sp.add_rule(
        StepRule(subject=dampers, object=damper_cmds, predicate=core.namespace.BRICK.hasPoint)
    )
    # Outside-air temperature sensor: free-floating optional, NOT tied
    # to ``ahu hasPoint oat_sensor``.  Every AHU in the dataset has its
    # own per-unit ``Outside_Air_Temp`` ``hasPoint`` (e.g.
    # ``bldg1.AHU.AHU01.Outside_Air_Temp`` and ``bldg1.AHU.AHU02.Outside_Air_Temp``),
    # but only one ``OutdoorEnvironmentSystem`` instance typically
    # materialises per building -- its ``brick_signature_pattern_standalone``
    # pairs an ``Outside_Air_Temperature_Sensor`` with a
    # ``Global_Solar_Irradiation_Sensor`` (both required) and the MILP
    # gets to claim each modeled URI at most once.  If we anchored
    # ``oat_sensor`` to the AHU via ``hasPoint``, AHU02 would bind to
    # its own ``Outside_Air_Temp`` URI -- at which no ``OutdoorEnvironment
    # System`` is modelled -- and ``outdoorAirTemperature`` would stay
    # dangling.  By declaring ``oat_sensor`` as a structurally-free
    # optional node we let the matcher bind it to whichever
    # ``Outside_Air_Temperature_Sensor`` URI an ``OutdoorEnvironmentSystem``
    # actually claims, so both AHUs end up reading from the same shared
    # outdoor-environment instance (and so do all BuildingSpaces -- the
    # ``BuildingSpace`` pattern uses the same free-floating optional
    # convention for outdoor temperature and solar irradiation).
    sp.add_node(oat_sensor, optional=True)
    # Supply-air temperature setpoint: optional -- some AHUs only carry
    # the Supply_Air_Temp measurement without a separate setpoint point
    # (e.g. Mortar bldg1 AHU02).  Wrap in OptionalRule so the pattern
    # still matches; the AHU's supplyAirTemperatureSetpoint port simply
    # stays unwired in that case (validation will warn but not fail).
    sp.add_rule(
        OptionalRule(
            subject=ahu,
            object=sat_setpoint,
            predicate=core.namespace.BRICK.hasPoint,
        )
    )

    # Branches are indexed by ``vavs`` -- one damper, one branch, one
    # calibratable nominal flow per VAV (issue #179) -- and the zone reads
    # its branches back with the same key (``output_port_index=vav`` in the
    # BuildingSpace pattern, one slot per VAV of its Vector flow ports).
    # ``exhaustTemperature`` stays indexed by ``spaces``: a room has one
    # exhaust temperature however many branches serve it, and the AHU maps
    # branch -> room itself (``_exhaust_branch_map``).
    sp.add_connection(
        spaces, "indoorTemperature", "exhaustTemperature", input_port_index=spaces
    )
    # Per-VAV damper command -> AHU supply / exhaust damper vectors.
    # The source-side port name is ``inputSignal``, which is the output
    # port a CITS / extracted PI controller produces at the
    # ``Damper_Position_Setpoint`` URI.  When ``ControllerIdentificationPI
    # TorchSystem`` is in Stage-2's ``systems_``, the controller component
    # is matched at the same command URI as the historised
    # ``SensorSystem`` and provides this ``inputSignal`` output, closing
    # the control loop natively during translation -- no separate
    # extract/wire post-process is needed.
    #
    # Both vectors are driven by the same per-VAV command because the
    # bldg1 BRICK graph models a single damper per VAV; for VAV systems
    # without dedicated exhaust modulation the return-air branch tracks
    # supply 1:1.  If a building genuinely models separate exhaust
    # dampers, register a sibling pattern that binds them separately.
    # ``output_port_index=damper_cmds`` picks the CITS actuator slot
    # corresponding to this damper command -- CITS.inputSignal is a
    # Vector (one slot per actuator) so we need a key on the sender side
    # too, matching the convention in ``brick_damper_command_sensor_pattern``.
    sp.add_connection(
        damper_cmds,
        "inputSignal",
        "supplyDamperPosition",
        output_port_index=damper_cmds,
        input_port_index=vavs,
    )
    sp.add_connection(
        damper_cmds,
        "inputSignal",
        "exhaustDamperPosition",
        output_port_index=damper_cmds,
        input_port_index=vavs,
    )
    sp.add_connection(sat_setpoint, "measuredValue", "supplyAirTemperatureSetpoint")
    sp.add_connection(oat_sensor, "outdoorTemperature", "outdoorAirTemperature")

    ModeledNode([ahu, vavs, dampers, damper_cmds])
    return sp


def brick_signature_pattern_vav_damper_commands():
    """AHU pattern for VAV systems whose damper command is a direct VAV point.

    Sibling of :func:`brick_signature_pattern_vav_dampers` for BMS-derived
    BRICK graphs that do not model a ``brick:Damper`` equipment under each
    VAV but attach the damper command point to the VAV itself (e.g. the
    Hoeje-Taastrup Raadhus graph, where every VAV carries a
    ``Damper_Position_Command`` next to its flow sensor / setpoint)::

        AHU  feeds     VAV
        VAV  feeds     Room / Zone
        VAV  hasPoint  Damper_Position_Command | Damper_Position_Setpoint

    Everything else (connections, Vector indexing by ``spaces``, the
    free-floating optional outside-air sensor and the optional supply-air
    temperature setpoint) mirrors the damper-equipment variant, so the
    downstream BuildingSpace / sensor patterns line up on the same
    ``spaces`` slots.

    The two AHU patterns are mutually exclusive in practice: a VAV either
    has a ``Damper`` part (damper-equipment variant) or a direct damper
    command point (this variant).  A graph that models *both* on the same
    VAV would produce two AirHandlingUnitSystem components per AHU (see the
    note below on the non-exclusive multi-member ``ModeledNode`` mutex).
    """
    sp = SignaturePattern(
        id="air_handling_unit_signature_pattern_brick_vav_damper_commands"
    )

    ahu = Node(cls=core.namespace.BRICK.AHU)
    spaces = Node(
        cls=(
            core.namespace.BRICK.Room,
            core.namespace.BRICK.Enclosed_space,
            core.namespace.BRICK.Open_space,
            core.namespace.BRICK.HVAC_Zone,
            core.namespace.REC.Room,
            core.namespace.REC.Zone,
        )
    )
    vavs = Node(cls=core.namespace.BRICK.VAV)
    damper_cmds = Node(
        cls=(
            core.namespace.BRICK.Damper_Position_Command,
            core.namespace.BRICK.Damper_Position_Setpoint,
        )
    )
    sat_setpoint = Node(cls=core.namespace.BRICK.Supply_Air_Temperature_Setpoint)
    oat_sensor = Node(cls=core.namespace.BRICK.Outside_Air_Temperature_Sensor)

    feeds = Predicate((core.namespace.BRICK.feeds, core.namespace.FSO.feedsFluidTo))

    sp.add_rule(SetAnyPathRule(subject=ahu, object=vavs, predicate=feeds))
    sp.add_rule(StepRule(subject=vavs, object=spaces, predicate=feeds))
    sp.add_rule(
        StepRule(
            subject=vavs, object=damper_cmds, predicate=core.namespace.BRICK.hasPoint
        )
    )
    sp.add_node(oat_sensor, optional=True)
    sp.add_rule(
        OptionalRule(
            subject=ahu,
            object=sat_setpoint,
            predicate=core.namespace.BRICK.hasPoint,
        )
    )

    sp.add_connection(
        spaces, "indoorTemperature", "exhaustTemperature", input_port_index=spaces
    )
    sp.add_connection(
        damper_cmds,
        "inputSignal",
        "supplyDamperPosition",
        output_port_index=damper_cmds,
        input_port_index=vavs,
    )
    sp.add_connection(
        damper_cmds,
        "inputSignal",
        "exhaustDamperPosition",
        output_port_index=damper_cmds,
        input_port_index=vavs,
    )
    sp.add_connection(sat_setpoint, "measuredValue", "supplyAirTemperatureSetpoint")
    sp.add_connection(oat_sensor, "outdoorTemperature", "outdoorAirTemperature")

    ModeledNode([ahu, vavs, damper_cmds])
    return sp
