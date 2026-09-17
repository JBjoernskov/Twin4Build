"""Example signature patterns for :mod:`twin4build.systems.outdoor_environment.outdoor_environment_system`.

Moved out of the system module (#200): patterns describe how one kind
of graph maps onto the component, and are examples of that, not a
standard.  Bound to their classes by :mod:`twin4build.examples.patterns`.
"""

from twin4build.translator.translator import (
    StepRule,
    Node,
    OptionalRule,
    SignaturePattern,
    PathRule,
)
import twin4build.core as core


def saref_signature_pattern():
    """
    Get the SAREF signature pattern of the outdoor environment component.

    Returns:
        SignaturePattern: The SAREF signature pattern of the outdoor environment component.
    """
    node0 = Node(cls=core.namespace.S4BLDG.OutdoorEnvironment)
    sp = SignaturePattern(id="outdoor_environment_signature_pattern")
    sp.add_modeled_node(node0)
    return sp


def brick_signature_pattern():
    """
    Get the BRICK signature pattern of the outdoor environment component.

    Returns:
        SignaturePattern: The BRICK signature pattern of the outdoor environment component.
    """
    weather_station = Node(cls=core.namespace.BRICK.Weather_Station)
    temp = Node(cls=core.namespace.BRICK.Outside_Air_Temperature_Sensor)
    irrad = Node(
        cls=(
            # ``Global_Solar_Irradiation_Sensor`` is not a Brick class (it
            # survives for graphs that extend Brick with it);
            # ``Solar_Irradiance_Sensor`` is the Brick 1.4 class (W/m2).
            core.namespace.BRICK.Global_Solar_Irradiation_Sensor,
            core.namespace.BRICK.Solar_Irradiance_Sensor,
        )
    )
    externalref_temp = Node(cls=(core.namespace.BRICKREF.ExternalReference, core.BlankNode))
    externalref_irrad = Node(cls=(core.namespace.BRICKREF.ExternalReference, core.BlankNode))
    timeseriesid_temp = Node(cls=core.namespace.XSD.string)
    timeseriesid_irrad = Node(cls=core.namespace.XSD.string)
    senaps_temp = Node(cls=core.namespace.XSD.string)
    senaps_irrad = Node(cls=core.namespace.XSD.string)

    # node3 = Node(cls=core.namespace.BRICK.Outdoor_CO2_Concentration_Sensor)
    sp = SignaturePattern(id="outdoor_environment_signature_pattern_brick")
    sp.add_rule(
        OptionalRule(
            subject=weather_station,
            object=temp,
            predicate=core.namespace.BRICK.hasPoint,
        )
    )
    sp.add_rule(
        OptionalRule(
            subject=weather_station,
            object=irrad,
            predicate=core.namespace.BRICK.hasPoint,
        )
    )
    sp.add_rule(
        OptionalRule(
            subject=temp,
            object=externalref_temp,
            predicate=core.namespace.BRICKREF.hasExternalReference,
        )  # Used in mortar
    )
    sp.add_rule(
        OptionalRule(
            subject=irrad,
            object=externalref_irrad,
            predicate=core.namespace.BRICKREF.hasExternalReference,
        )  # Used in mortar
    )
    sp.add_rule(
        OptionalRule(
            subject=externalref_temp,
            object=timeseriesid_temp,
            predicate=core.namespace.BRICKREF.hasTimeseriesId,
        )  # Used in mortar
    )
    sp.add_rule(
        OptionalRule(
            subject=externalref_irrad,
            object=timeseriesid_irrad,
            predicate=core.namespace.BRICKREF.hasTimeseriesId,
        )  # Used in mortar
    )
    sp.add_rule(
        OptionalRule(
            subject=temp, object=senaps_temp, predicate=core.namespace.SENAPS.senaps_id
        )  # Used in bts
    )
    sp.add_rule(
        OptionalRule(
            subject=irrad,
            object=senaps_irrad,
            predicate=core.namespace.SENAPS.senaps_id,
        )  # Used in bts
    )
    sp.add_modeled_node(temp)
    sp.add_modeled_node(irrad)
    sp.add_parameter("_uuid_outdoorTemperature", senaps_temp)
    sp.add_parameter("_uuid_globalIrradiation", senaps_irrad)
    sp.add_parameter("_uuid_outdoorTemperature", timeseriesid_temp)
    sp.add_parameter("_uuid_globalIrradiation", timeseriesid_irrad)

    return sp


def brick_signature_pattern_standalone():
    """
    BRICK signature pattern for a standalone Outside_Air_Temperature_Sensor
    and/or Global_Solar_Irradiation_Sensor not attached to a Weather_Station.

    Both sensor nodes are optional so the pattern matches even when only one
    is present. Instantiates a single OutdoorEnvironmentSystem with both
    uuid_outdoorTemperature and uuid_globalIrradiation populated from each
    sensor's external reference timeseries ID.
    """
    temp = Node(cls=core.namespace.BRICK.Outside_Air_Temperature_Sensor)
    irrad = Node(
        cls=(
            # ``Global_Solar_Irradiation_Sensor`` is not a Brick class (it
            # survives for graphs that extend Brick with it);
            # ``Solar_Irradiance_Sensor`` is the Brick 1.4 class (W/m2).
            core.namespace.BRICK.Global_Solar_Irradiation_Sensor,
            core.namespace.BRICK.Solar_Irradiance_Sensor,
        )
    )
    externalref_temp = Node(cls=(core.namespace.BRICKREF.ExternalReference, core.BlankNode))
    externalref_irrad = Node(cls=(core.namespace.BRICKREF.ExternalReference, core.BlankNode))
    timeseriesid_temp = Node(cls=core.namespace.XSD.string)
    timeseriesid_irrad = Node(cls=core.namespace.XSD.string)
    senaps_temp = Node(cls=core.namespace.XSD.string)
    senaps_irrad = Node(cls=core.namespace.XSD.string)

    sp = SignaturePattern(id="outdoor_environment_signature_pattern_brick_standalone")
    sp.add_node(temp, optional=False)
    sp.add_node(irrad, optional=False)
    sp.add_rule(
        OptionalRule(subject=temp, object=externalref_temp, predicate=core.namespace.BRICKREF.hasExternalReference)
    )
    sp.add_rule(
        OptionalRule(subject=externalref_temp, object=timeseriesid_temp, predicate=core.namespace.BRICKREF.hasTimeseriesId)
    )
    sp.add_rule(
        OptionalRule(subject=temp, object=senaps_temp, predicate=core.namespace.SENAPS.senaps_id)
    )
    sp.add_rule(
        OptionalRule(subject=irrad, object=externalref_irrad, predicate=core.namespace.BRICKREF.hasExternalReference)
    )
    sp.add_rule(
        OptionalRule(subject=externalref_irrad, object=timeseriesid_irrad, predicate=core.namespace.BRICKREF.hasTimeseriesId)
    )
    sp.add_rule(
        OptionalRule(subject=irrad, object=senaps_irrad, predicate=core.namespace.SENAPS.senaps_id)
    )
    sp.add_modeled_node(temp)
    sp.add_modeled_node(irrad)
    sp.add_parameter("_uuid_outdoorTemperature", senaps_temp)
    sp.add_parameter("_uuid_outdoorTemperature", timeseriesid_temp)
    sp.add_parameter("_uuid_globalIrradiation", senaps_irrad)
    sp.add_parameter("_uuid_globalIrradiation", timeseriesid_irrad)

    return sp
