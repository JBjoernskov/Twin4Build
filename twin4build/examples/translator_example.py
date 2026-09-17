# %pip install twin4build # Uncomment in google colab
# Standard library imports
import datetime

# Third party imports
from dateutil import tz

# Local application imports
import twin4build as tb
from twin4build.examples import patterns as example_patterns
import twin4build.examples.utils as utils


def fcn(self):
    """
    Custom configuration function to set up the model after translation.
    This function adds missing connections, configures data sources, and sets up control parameters.
    """
    # Add supply water temperature schedule
    supply_water_schedule = tb.ScheduleSystem(
        weekday_ruleset={
            "ruleset_default_value": 60,
            "ruleset_start_minute": [],
            "ruleset_end_minute": [],
            "ruleset_start_hour": [],
            "ruleset_end_hour": [],
            "ruleset_value": [],
        },
        id="supply_water_schedule",
    )

    # Add boundary temperature schedule
    boundary_temp_schedule = tb.ScheduleSystem(
        weekday_ruleset={
            "ruleset_default_value": 21,
            "ruleset_start_minute": [],
            "ruleset_end_minute": [],
            "ruleset_start_hour": [],
            "ruleset_end_hour": [],
            "ruleset_value": [],
        },
        id="boundary_temp_schedule",
    )

    # Add missing connections.  The office couples to the boundary-temperature
    # schedule through a 2R1C WallSystem (energy-consistent wall model).
    boundary_wall = tb.WallSystem(
        C=1e6,
        R_a=0.02,
        R_b=0.02,
        id="office_boundary_wall",
    )
    self.add_connection(
        self.components["office"],
        boundary_wall,
        "indoorTemperature",
        "temperatureA",
    )
    self.add_connection(
        boundary_temp_schedule,
        boundary_wall,
        "scheduleValue",
        "temperatureB",
    )
    self.add_connection(
        boundary_wall,
        self.components["office"],
        "heatFlowRateA",
        "wallHeatGain",
        input_port_index=0,
    )
    self.add_connection(
        supply_water_schedule,
        self.components["office_space_heater"],
        "scheduleValue",
        "supplyWaterTemperature",
    )

    # Configure sensor data sources
    self.components["office_temperature_sensor"].use_spreadsheet = True
    self.components["office_temperature_sensor"].filename = utils.get_path(
        ["estimator_example", "temperature_sensor.csv"]
    )

    self.components["office_co2_sensor"].use_spreadsheet = True
    self.components["office_co2_sensor"].filename = utils.get_path(
        ["estimator_example", "co2_sensor.csv"]
    )

    self.components["office_valve_position_sensor"].use_spreadsheet = True
    self.components["office_valve_position_sensor"].filename = utils.get_path(
        ["estimator_example", "valve_position_sensor.csv"]
    )

    self.components["office_damper_position_sensor"].use_spreadsheet = True
    self.components["office_damper_position_sensor"].filename = utils.get_path(
        ["estimator_example", "damper_position_sensor.csv"]
    )

    self.components["supply_air_temperature_sensor"].use_spreadsheet = True
    self.components["supply_air_temperature_sensor"].filename = utils.get_path(
        ["estimator_example", "supply_air_temperature.csv"]
    )

    # Configure control setpoints
    self.components["office_co2_setpoint"].weekDayRulesetDict = {
        "ruleset_default_value": 900,
        "ruleset_start_minute": [],
        "ruleset_end_minute": [],
        "ruleset_end_hour": [],
        "ruleset_start_hour": [],
        "ruleset_value": [],
    }

    self.components["office_occupancy_profile"].weekDayRulesetDict = {
        "ruleset_default_value": 0,
        "ruleset_start_minute": [],
        "ruleset_end_minute": [],
        "ruleset_start_hour": [],
        "ruleset_end_hour": [],
        "ruleset_value": [],
    }

    self.components["office_temperature_heating_setpoint"].use_spreadsheet = True
    self.components["office_temperature_heating_setpoint"].filename = utils.get_path(
        ["estimator_example", "temperature_heating_setpoint.csv"]
    )

    # Configure outdoor environment data
    self.components["outdoor_environment"].use_spreadsheet = True
    self.components["outdoor_environment"].filename_outdoorTemperature = utils.get_path(
        ["estimator_example", "outdoor_environment.csv"]
    )
    self.components["outdoor_environment"].datecolumn_outdoorTemperature = 0
    self.components["outdoor_environment"].valuecolumn_outdoorTemperature = 1

    self.components["outdoor_environment"].filename_globalIrradiation = utils.get_path(
        ["estimator_example", "outdoor_environment.csv"]
    )
    self.components["outdoor_environment"].datecolumn_globalIrradiation = 0
    self.components["outdoor_environment"].valuecolumn_globalIrradiation = 2

    self.components["outdoor_environment"].filename_outdoorCo2Concentration = (
        utils.get_path(["estimator_example", "outdoor_environment.csv"])
    )
    self.components["outdoor_environment"].datecolumn_outdoorCo2Concentration = 0
    self.components["outdoor_environment"].valuecolumn_outdoorCo2Concentration = 3


# Create a new model
# Local application imports
from twin4build.utils.logger import LOGGER
import twin4build.core as core
from twin4build.translator.translator import Node, OptionalRule, SignaturePattern, StepRule

LOGGER.hide_status("debug")

# ---------------------------------------------------------------------------
# Signature patterns: how a graph shape maps onto a component.
#
# Patterns are user-defined.  There is no pattern to rule them all: how a
# building is described differs per ontology, per BMS vendor and per site,
# so a deployment writes the patterns that fit *its* graphs and passes them
# to the translator explicitly.  ``twin4build.patterns`` is the public
# example set.  Two of its patterns are spelled out here to show the idea;
# the rest are imported below.
# ---------------------------------------------------------------------------


def schedule_pattern():
    """The simplest pattern: one node, bound to one component class.

    Every ``s4bldg:Schedule`` in the graph becomes a ``ScheduleSystem``.  The
    modeled node is the graph node the component stands for; its identity
    becomes the component id.
    """
    schedule = Node(cls=core.namespace.S4BLDG.Schedule)
    sp = SignaturePattern(id="example_schedule", system=tb.ScheduleSystem)
    sp.add_modeled_node(schedule)
    return sp


def damper_pattern():
    """A pattern with structure and an optional parameter.

    A damper is recognised by its controlled opening position: a controller
    ``controls`` an ``OpeningPosition`` that ``isPropertyOf`` the damper, and
    the controller ``observes`` some property.  ``StepRule`` edges must exist
    for a match; ``OptionalRule`` edges are read when present -- here the
    damper's nominal air flow rate, which ``add_parameter`` hands to the
    component's ``nominalAirFlowRate``.  ``add_input`` wires the controller's
    output to the damper's ``damperPosition``; ``add_modeled_node`` marks the
    node the component stands for.
    """
    damper = Node(cls=core.namespace.S4BLDG.Damper)
    controller = Node(cls=core.namespace.S4BLDG.Controller)
    position = Node(cls=core.namespace.SAREF.OpeningPosition)
    observed = Node(cls=core.namespace.SAREF.Property)
    value = Node(cls=core.namespace.SAREF.PropertyValue)
    number = Node(cls=core.namespace.XSD.float)
    nominal_flow = Node(cls=core.namespace.S4BLDG.NominalAirFlowRate)
    sp = SignaturePattern(id="example_damper", system=tb.DamperSystem)
    sp.add_rule(StepRule(subject=controller, object=position, predicate=core.namespace.SAREF.controls))
    sp.add_rule(StepRule(subject=position, object=damper, predicate=core.namespace.SAREF.isPropertyOf))
    sp.add_rule(StepRule(subject=controller, object=observed, predicate=core.namespace.SAREF.observes))
    sp.add_rule(OptionalRule(subject=value, object=number, predicate=core.namespace.SAREF.hasValue))
    sp.add_rule(OptionalRule(subject=value, object=nominal_flow, predicate=core.namespace.SAREF.isValueOfProperty))
    sp.add_rule(OptionalRule(subject=damper, object=value, predicate=core.namespace.SAREF.hasPropertyValue))
    # The controller's output drives the damper: an input declared on the pattern
    sp.add_input("damperPosition", controller, "inputSignal")
    sp.add_parameter("nominalAirFlowRate", number)
    sp.add_modeled_node(damper)
    return sp


# The pattern set for this example: the two patterns above replace the
# library's schedule and damper examples; everything else comes from the
# public example set.
patterns = [
    sp for sp in example_patterns.default_patterns()
    if sp.system not in (tb.ScheduleSystem, tb.DamperSystem)
] + [schedule_pattern(), damper_pattern()]

# Load the semantic model and translate it with those patterns
filename = utils.get_path(["estimator_example", "one_room_example_model.xlsm"])
sm = tb.SemanticModel(rdf_file=filename, id="translator_example")
model = tb.Translator().translate(sm, patterns=patterns, id="translator_example")
model.load(fcn=fcn)


forest_green = "#2D6A4F"
pink = "#FB9A99"
terracotta = "#C75B3A"
light_purple = "#CAB2D6"
purple = "#5B5EA6"
brown = "#7A6855"
white = "#FFFFFF"

# Named aliases (easy to swap)
light_black = "#3B3838"
dark_blue = "#44546A"
orange = "#DC8665"
red = "#873939"
grey = "#666666"
light_blue = "#8497B0"
green = "#83AF9B"
magenta = "#660066"
instance_style = {
    "http://example.org/building#cooling_coil_airside": {
        "fill_color": [light_blue, light_blue, None, None],
    },
    "http://example.org/building#cooling_coil_waterside": {
        "fill_color": [light_blue, light_blue, None, None],
    },
    "http://example.org/building#cooling_coil": {
        "fill_color": [light_blue, light_blue, None, None],
    },
    "http://example.org/building#cooling_system": {
        "fill_color": [light_blue, light_blue, None, None],
    },
    "http://example.org/building#heating_system": {
        "fill_color": [red, red, None, None],
    },
    "http://example.org/building#ventilation_system": {
        "fill_color": [dark_blue, dark_blue, None, None],
    },
}

query = """
                CONSTRUCT {
                    ?s ?p ?o
                }
                WHERE {
                    ?s ?p ?o .
                    FILTER (?p != rdf:type && 
                            ?p != rdfs:subClassOf
                            )
                    FILTER NOT EXISTS { ?s rdf:type s4syst:System }
                    FILTER NOT EXISTS { ?o rdf:type s4syst:System }
                }
                """
initial_node = "http://example.org/building#office"
model.semantic_model.visualize(
    format="png",
    instance_style=instance_style,
    deduplicate_inverse=True,
    include_full_uri=False,
    query=query,
    initial_node=initial_node,
    node_limit=35,
    traversal_mode="bfs",
    dpi=800,
)
model.simulation_model.visualize(
    format="png",
    instance_style=instance_style,
    deduplicate_inverse=True,
    include_full_uri=False,
    literals=False,
    dpi=800,
    compressed=True,
)
print(model.simulation_model.get_dir())
