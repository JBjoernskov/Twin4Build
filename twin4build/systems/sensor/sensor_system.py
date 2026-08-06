# Standard library imports
import datetime
from typing import Any, Callable, Dict, List, Optional, Union

# Third party imports
import pandas as pd

# Local application imports
import twin4build.core as core
import twin4build.utils.types as tps
from twin4build.systems.utils.pass_input_to_output import PassInputToOutput
from twin4build.systems.utils.time_series_input_system import TimeSeriesInputSystem
from twin4build.translator.translator import (
    ModeledNode,
    Node,
    OptionalRule,
    PathRule,
    SignaturePattern,
    StepRule,
)
from twin4build.utils.logger import LOGGER, autoreset_print


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


# Properties of spaces
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


# Properties of spaces
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

    Matches any BRICK Command that is a hasPoint of a VAV and has a timeseries
    UUID.  The SensorSystem holds the measured actuator command (ground truth for
    estimation) and receives the CITS predicted command via inputSignal so that
    the estimator can minimise the error.

    Topology::

        VAV  hasPoint  <Command>
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
    vav = Node(cls=core.namespace.BRICK.VAV)
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


# -----------------------------------------------------------------------------
# Why the zone / AHU air-temperature sensors are expressed as *two* patterns each
# -----------------------------------------------------------------------------
#
# We want the virtual sensor (connected to ``BuildingSpace.indoorTemperature`` /
# ``AHU.supplyAirTemperature``) to appear in the simulation model whether or not
# the BRICK graph actually carries a Brick-reference timeseries UUID.  The first,
# natural encoding was a single pattern with the external-ref / timeseries-id
# triples wrapped in :class:`OptionalRule`.  That encoding is **broken** for a
# very specific reason:
#
# 1. OptionalRule triples are eligible to be matched on a *disconnected* subgraph,
#    separate from the (sensor, vav, room) subgraph.
# 2. In the translator's ``_try_merge_with_incomplete`` disconnected-merge
#    branch, any sub-group whose modeled_node slots are *not* filled is
#    classified as a "shared resource" and placed in ``groups_to_preserve``
#    — i.e. kept for reuse across every subsequent match of the pattern.
# 3. Because the original pattern only declared ``sensor`` as a modeled node,
#    the ``(externalref, timeseries_id)`` sub-group is *always* a shared
#    resource.  The translator therefore picks *one* arbitrary
#    ``(blank_node, uuid_literal)`` pair (the first one it enumerates) and
#    rebinds *every* Zone_Air_Temperature_Sensor in the building to it.
#
# The fix is to split the single pattern into two mutually-exclusive variants:
#
# * ``*_with_ref_pattern`` — requires the external-ref chain via :class:`StepRule`
#   triples and additionally declares ``externalref`` as a modeled node so the
#   (externalref, timeseries_id) sub-group can never be re-used.  Models
#   ``{sensor, externalref}`` (two modeled nodes).
# * ``*_virtual_pattern``  — omits the external-ref chain entirely.  Models
#   ``{sensor}`` (one modeled node).
#
# The translator's MILP objective is
# ``component_selection_cost - semantic_instance_benefit * n_modeled_nodes``,
# and the mutual-exclusion constraint is keyed on each ``modeled_node`` /
# ``sm_node`` pair.  Consequently:
#
# * If a sensor has a Brick timeseries reference, *both* patterns match on the
#   same ``sensor`` modeled node, but the with-ref variant has two modeled
#   nodes (cheaper in the minimisation) and wins.
# * If a sensor has *no* Brick timeseries reference, only the virtual variant
#   matches and is selected.
#
# Net effect: the UUID is preserved when available and the virtual sensor is
# preserved when no UUID exists — without reintroducing the shared-resource
# cross-binding bug.
# -----------------------------------------------------------------------------


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


def get_brick_ahu_supply_air_temp_sensor_with_ref_pattern():
    """BRICK Supply_Air_Temperature_Sensor on an AHU, with a Brick timeseries reference.

    Topology::

        Supply_Air_Temperature_Sensor  isPointOf             AHU
        Supply_Air_Temperature_Sensor  hasExternalReference  <ExternalRef/BNode>
                                                                 └─ hasTimeseriesId → <uuid>

    Paired with :func:`get_brick_ahu_supply_air_temp_sensor_virtual_pattern`;
    see the module-level note above.
    """
    sensor = Node(cls=core.namespace.BRICK.Supply_Air_Temperature_Sensor)
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
        output_port_index=room,
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
        output_port_index=room,
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
    sensor = Node(cls=core.namespace.BRICK.Supply_Air_Temperature_Sensor)
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


@autoreset_print
class SensorSystem(core.System):
    """A system representing a physical or virtual sensor in the building.

    This class implements sensor functionality, supporting both physical sensors
    (reading from time series data) and virtual sensors (computing values from
    other inputs). It integrates with TimeSeriesInputSystem for data handling.

    Args:
        filename: Path to sensor readings file.
            Defaults to None.
        df: DataFrame containing readings.
            Defaults to None.
        uuid: UUID identifying the time series in the database.
            Defaults to None.
        dbconfig: Configuration of the database to read sensor values from.
            Defaults to None.
        datecolumn: Column index containing date/time information.
            Defaults to 0.
        valuecolumn: Column index containing sensor values.
            Defaults to 1.
        use_spreadsheet: Whether to use a spreadsheet for input.
            Defaults to False.
        use_database: Whether to use a database for input.
            Defaults to False.
        use_df: Whether to use the provided DataFrame for input.
            Defaults to False.
        transformation: Optional function to transform the value.
            Defaults to None.
        **kwargs: Additional keyword arguments passed to parent class.

    Note:
        A sensor must either have connections to other systems (virtual sensor) or
        have data input through filename/df/database (physical sensor). Flags are
        auto-detected if only one data source is provided.
    """

    sp = [
        get_temperature_before_air_to_air_supply_side(),
        get_temperature_before_air_to_air_exhaust_side(),
        get_temperature_after_air_to_air_supply_side(),
        get_temperature_after_air_to_air_exhaust_side(),
        get_signature_pattern_input(),
        get_flow_signature_pattern_after_coil_air_side(),
        get_flow_signature_pattern_after_coil_water_side(),
        get_flow_signature_pattern_before_coil_water_side(),
        get_space_temperature_signature_pattern(),
        get_space_co2_signature_pattern(),
        get_position_signature_pattern(),
        # BRICK-specific patterns (Mortar / BRICK-annotated datasets)
        get_brick_command_sensor_pattern(),
        get_brick_damper_command_sensor_pattern(),
        # Each air-temperature virtual sensor has a mutually-exclusive pair of
        # patterns: one that requires a Brick timeseries reference (preferred
        # by the MILP when available) and one "virtual" fallback that still
        # wires the modelled temperature to the SensorSystem when no
        # timeseries is attached.  See the comment above
        # ``get_brick_zone_air_temp_sensor_with_ref_pattern`` for why this
        # split is required.
        get_brick_zone_air_temp_sensor_with_ref_pattern(),
        get_brick_zone_air_temp_sensor_virtual_pattern(),
        get_brick_ahu_supply_air_temp_sensor_with_ref_pattern(),
        get_brick_ahu_supply_air_temp_sensor_virtual_pattern(),
        get_brick_supply_air_flow_sensor_with_ref_pattern(),
        get_brick_supply_air_flow_sensor_virtual_pattern(),
        get_brick_sensor_leaf_pattern(),
    ]

    def __init__(
        self,
        filename: Optional[str] = None,
        df: Optional[pd.DataFrame] = None,
        uuid: Optional[str] = None,
        dbconfig: Optional[Dict[str, Any]] = None,
        datecolumn: int = 0,
        valuecolumn: int = 1,
        use_spreadsheet: bool = False,
        use_database: bool = False,
        use_df: bool = False,
        transformation: Optional[callable] = None,
        **kwargs,
    ) -> None:
        """Initialize the sensor system.

        Args:
            filename: Path to sensor readings file.
                Defaults to None.
            df: DataFrame containing readings.
                Defaults to None.
            datecolumn: Column index containing date/time information.
                Defaults to 0.
            valuecolumn: Column index containing sensor values.
                Defaults to 1.
            use_spreadsheet: Whether to use a spreadsheet for input.
                Defaults to False.
            use_database: Whether to use a database for input.
                Defaults to False.
            use_df: Whether to use the provided DataFrame for input.
                Defaults to False.
            transformation: Optional function to transform the value.
                Defaults to None.
            **kwargs: Additional keyword arguments passed to parent class.

        Note:
            Either filename/df must be provided for physical sensors, or
            the sensor must have connections defined for virtual sensors.
            Flags are auto-detected if only one data source is provided.
        """
        for legacy_key, new_key in (
            ("useSpreadsheet", "use_spreadsheet"),
            ("useDatabase", "use_database"),
            ("usedf", "use_df"),
        ):
            if legacy_key in kwargs:
                raise TypeError(
                    f"`{legacy_key}` has been removed. Use `{new_key}` instead."
                )

        # Count how many data sources are provided
        has_df = df is not None
        has_filename = filename is not None
        has_database = dbconfig is not None or uuid is not None
        n_sources = sum([has_df, has_filename, has_database])
        n_flags = sum([use_spreadsheet, use_database, use_df])

        # If multiple sources provided, user must explicitly set a flag
        assert not (n_sources > 1 and n_flags == 0), (
            "Multiple data sources provided (df, filename, database). "
            "You must explicitly set one of use_df=True, use_spreadsheet=True, or use_database=True "
            "to specify which source to use."
        )

        # Auto-detect data source if no flags are explicitly set
        if not use_spreadsheet and not use_database and not use_df:
            if has_df:
                use_df = True
            elif has_filename:
                use_spreadsheet = True
            elif has_database:
                use_database = True

        assert (
            sum([use_spreadsheet, use_database, use_df]) <= 1
        ), "Only one of use_spreadsheet, use_database, or use_df can be True."
        super().__init__(**kwargs)

        # Define inputs and outputs as private variables
        self._input = {"measuredValue": tps.Scalar()}
        self._output = {
            "measuredValue": tps.Scalar(0)
        }  # TODO: Not necessary to be a leaf scalar, if the sensor has inputs. Need to implement check in initialize()

        # Store attributes as private variables
        self._use_spreadsheet = use_spreadsheet
        self._use_database = use_database
        self._use_df = use_df
        self._filename = filename
        self._df = df
        self._datecolumn = datecolumn
        self._valuecolumn = valuecolumn
        self._uuid = uuid
        self._dbconfig = dbconfig
        self._is_leaf = None
        self._time_series_input = None
        self._transformation = transformation

        self._config = {
            "parameters": ["use_spreadsheet", "use_database", "use_df"],
            "spreadsheet": ["filename", "datecolumn", "valuecolumn"],
            "database": ["uuid", "dbconfig"],
        }

    @property
    def config(self):
        return self._config

    @property
    def input(self) -> dict:
        """
        Get the input ports of the sensor system.

        Returns:
            dict: Dictionary containing input ports:
                - "measuredValue": Measured value input for virtual sensors
        """
        return self._input

    @property
    def output(self) -> dict:
        """
        Get the output ports of the sensor system.

        Returns:
            dict: Dictionary containing output ports:
                - "measuredValue": Measured value output [units depend on sensor type]
        """
        return self._output

    @property
    def filename(self) -> Optional[str]:
        """
        Get the path to sensor readings file.
        """
        return self._filename

    @filename.setter
    def filename(self, value: Optional[str]) -> None:
        """
        Set the path to sensor readings file.
        Automatically sets use_spreadsheet=True if a value is provided.
        """
        self._filename = value
        if value is not None:
            self._use_spreadsheet = True
            self._use_database = False
            self._use_df = False

    @property
    def df(self) -> Optional[pd.DataFrame]:
        """
        Get the direct DataFrame input of sensor readings.
        """
        return self._df

    @df.setter
    def df(self, value: Optional[pd.DataFrame]) -> None:
        """
        Set the direct DataFrame input of sensor readings.
        Automatically sets use_df=True if a value is provided.
        """
        self._df = value
        if value is not None:
            self._use_df = True
            self._use_spreadsheet = False
            self._use_database = False

    @property
    def datecolumn(self) -> int:
        """
        Get the column index for date_time values.
        """
        return self._datecolumn

    @datecolumn.setter
    def datecolumn(self, value: int) -> None:
        """
        Set the column index for date_time values.
        """
        self._datecolumn = value

    @property
    def valuecolumn(self) -> int:
        """
        Get the column index for sensor readings.
        """
        return self._valuecolumn

    @valuecolumn.setter
    def valuecolumn(self, value: int) -> None:
        """
        Set the column index for sensor readings.
        """
        self._valuecolumn = value

    @property
    def is_leaf(self) -> bool:
        """
        Get whether the sensor reads from file/DataFrame (True) or is virtual (False).
        """
        return self._is_leaf

    @is_leaf.setter
    def is_leaf(self, value: bool) -> None:
        """
        Set whether the sensor reads from file/DataFrame (True) or is virtual (False).
        """
        self._is_leaf = value

    @property
    def time_series_input(self) -> Optional[TimeSeriesInputSystem]:
        """
        Get the data handling system for physical sensors.
        """
        return self._time_series_input

    @time_series_input.setter
    def time_series_input(self, value: Optional[TimeSeriesInputSystem]) -> None:
        """
        Set the data handling system for physical sensors.
        """
        self._time_series_input = value

    @property
    def use_spreadsheet(self) -> bool:
        """
        Get whether to use a spreadsheet for input.
        """
        return self._use_spreadsheet

    @use_spreadsheet.setter
    def use_spreadsheet(self, value: bool) -> None:
        """
        Set whether to use a spreadsheet for input.
        """
        self._use_spreadsheet = value

    @property
    def use_database(self) -> bool:
        """
        Get whether to use a database for input.
        """
        return self._use_database

    @use_database.setter
    def use_database(self, value: bool) -> None:
        """
        Set whether to use a database for input.
        """
        self._use_database = value

    @property
    def use_df(self) -> bool:
        """
        Get whether to use a DataFrame for input.
        """
        return self._use_df

    @use_df.setter
    def use_df(self, value: bool) -> None:
        """
        Set whether to use a DataFrame for input.
        """
        self._use_df = value

    @property
    def uuid(self) -> Optional[str]:
        """
        Get the UUID for database operations.
        """
        return self._uuid

    @uuid.setter
    def uuid(self, value: Optional[str]) -> None:
        """
        Set the UUID for database operations.
        Automatically sets use_database=True if a value is provided.
        """
        self._uuid = value
        if value is not None:
            self._use_database = True
            self._use_spreadsheet = False
            self._use_df = False

    @property
    def dbconfig(self) -> Optional[Dict[str, Any]]:
        """
        Get the database configuration parameters.
        """
        return self._dbconfig

    @dbconfig.setter
    def dbconfig(self, value: Optional[Dict[str, Any]]) -> None:
        """
        Set the database configuration parameters.
        Automatically sets use_database=True if a value is provided.
        """
        self._dbconfig = value
        if value is not None:
            self._use_database = True
            self._use_spreadsheet = False
            self._use_df = False

    def set_dbconfig(self, dbconfig: Optional[Dict[str, Any]]) -> None:
        """Set the database configuration on this sensor.

        Functional sibling of the ``dbconfig`` property setter, exposed
        explicitly so model-level helpers (e.g.
        :meth:`SimulationModel.set_dbconfigs`) can dispatch via duck-typed
        method lookup instead of touching the ``dbconfig`` property.
        """
        self.dbconfig = dbconfig

    @property
    def transformation(self) -> Optional[Callable]:
        """Unit-conversion callable applied to loaded timeseries before they
        are emitted on the sensor's ``measuredValue`` output.  ``None`` means
        no conversion (raw values pass through)."""
        return self._transformation

    @transformation.setter
    def transformation(self, fn: Optional[Callable]) -> None:
        self._transformation = fn

    def set_transformation(self, fn: Optional[Callable]) -> None:
        """Set the unit-conversion callable applied to loaded timeseries.

        Companion to :meth:`SimulationModel.set_transformations` (plural):
        the bulk model-level setter dispatches a per-component call here
        for every match.  Idempotent; subsequent calls overwrite.
        """
        self._transformation = fn

    def validate(self, p) -> tuple[bool, bool, bool, bool]:
        """Validate the sensor system configuration.

        Checks if the sensor has proper inputs for different operational modes.

        Args:
            p: Logging function for validation messages.

        Returns:
            tuple[bool, bool, bool, bool]: Validation status for:
                - Simulator
                - Estimator
                - Evaluator
                - Monitor
        """
        validated_for_simulator = True
        validated_for_estimator = True
        validated_for_optimizer = True

        if (
            len(self.connects_at) == 0
            and self.filename is None
            and self.df is None
            and self.uuid is None
        ):
            message = f"|CLASS: {self.__class__.__name__}|ID: {self.id}|: filename or df or uuid must be provided to enable use of Simulator, Estimator, and Optimizer."
            p(message, status="WARNING")
            validated_for_simulator = False
            validated_for_estimator = False
            validated_for_optimizer = False

        elif (
            len(self.connects_at) > 0
            and self.filename is None
            and self.df is None
            and self.uuid is None
        ):
            message = f"|CLASS: {self.__class__.__name__}|ID: {self.id}|: filename or df or uuid must be provided to enable use of Estimator."
            p(message, status="WARNING")
            validated_for_estimator = False

        self.is_leaf = len(self.connects_at) == 0  # No inputs -> leaf scalar
        self.output["measuredValue"].is_leaf = self.is_leaf

        return (
            validated_for_simulator,
            validated_for_estimator,
            validated_for_optimizer,
        )

    def validate_connections(self, p) -> bool:
        validated = True
        if (
            self.is_leaf
            and self.use_spreadsheet == False
            and self.use_database == False
            and self.use_df == False
        ):
            message = f"|CLASS: {self.__class__.__name__}|ID: {self.id}|: Missing connections for the following input(s) to enable use of Simulator, Estimator, and Optimizer:"
            p(message, status="[WARNING]")
            p.add_level()
            p("measuredValue")
            p.remove_level()
            validated = False
        return validated

    def initialize(
        self,
        start_time: List[datetime.datetime],
        end_time: List[datetime.datetime],
        step_size: List[float],
    ) -> None:
        """Initialize the sensor system.

        Sets up the physical or virtual sensor system and initializes the step instance.

        Args:
            start_time (Optional[datetime.datetime]): Start time for the simulation.
            end_time (Optional[datetime.datetime]): End time for the simulation.
            step_size (Optional[float]): Time step size in seconds.
            model (Optional[Any]): Model object (not used in this class).
        """

        self.validate(LOGGER)
        self.validate_connections(LOGGER)

        if self.use_spreadsheet or self.use_database or self.use_df:
            if self.use_df:
                if self.df is None:
                    raise ValueError("df must be provided when use_df=True.")
            self.time_series_input = TimeSeriesInputSystem(
                id=f"time series input - {self.id}",
                df=self.df,
                filename=self.filename,
                date_column=self.datecolumn,
                value_column=self.valuecolumn,
                use_spreadsheet=self.use_spreadsheet,
                use_database=self.use_database,
                uuid=self.uuid,
                dbconfig=self.dbconfig,
                transformation=self._transformation,
            )
            self.time_series_input.initialize(
                start_time=start_time,
                end_time=end_time,
                step_size=step_size,
            )

        else:
            self.time_series_input = None

        assert (
            len(self.connects_at) == 0 and self.time_series_input is None
        ) == False, f'Sensor object "{self.id}" has no inputs and and holds no data.'

        if self.is_leaf:
            # The batch initialization args are calculated in the TimeSeriesInputSystem.initialize() method.
            # They are stored in the physicalSystem object and reused here.
            self.output["measuredValue"].initialize(
                n_t=self.time_series_input.n_timesteps,
                n_s=self.time_series_input.batch_size,
                n_c=1,
                values=self.time_series_input.values,
            )
        else:
            _, _, max_timesteps, _ = core.Simulator.get_simulation_timesteps(
                start_time, end_time, step_size
            )
            batch_size = len(start_time)
            self.input["measuredValue"].initialize(
                n_t=max_timesteps,
                n_s=batch_size,
            )
            self.output["measuredValue"].initialize(
                n_t=max_timesteps,
                n_s=batch_size,
            )

    def do_step(
        self,
        second_time: Optional[float] = None,
        date_time: Optional[datetime.datetime] = None,
        step_size: Optional[float] = None,
        step_index: Optional[int] = None,
    ) -> None:
        """Execute one time step of the sensor system.

        Updates sensor outputs based on either physical readings or virtual calculations.

        Args:
            second_time (Optional[float]): Current simulation time in seconds.
            date_time (Optional[datetime.datetime]): Current simulation date_time.
            step_size (Optional[float]): Time step size in seconds.
        """
        if self.is_leaf:
            self.output["measuredValue"]._set(i_t=step_index)
        else:
            self.output["measuredValue"]._set(
                self.input["measuredValue"].get(), step_index
            )

    def get_physical_readings(
        self,
        start_time: List[datetime.datetime],
        end_time: List[datetime.datetime],
        step_size: List[float],
    ) -> pd.DataFrame:
        """Retrieve physical sensor readings for a specified time period.

        Args:
            start_time (Optional[datetime.datetime]): Start time for readings.
            end_time (Optional[datetime.datetime]): End time for readings.
            step_size (Optional[float]): Time step size in seconds.

        Returns:
            pd.DataFrame: DataFrame containing sensor readings.

        Raises:
            AssertionError: If called on a virtual sensor (no physical readings available).
        """
        self.initialize(start_time, end_time, step_size)
        assert (
            self.time_series_input is not None
        ), f'Cannot return physical readings for Sensor with id "{self.id}" as time_series_input is None.\nEither this sensor has not been intialized or the arguments filename/df/dbconfig were not provided when the object was initialized or the sensor is virtual and has no time_series_input.'
        self.time_series_input.initialize(start_time, end_time, step_size)
        return self.time_series_input.df
