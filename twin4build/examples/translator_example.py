# %pip install twin4build # Uncomment in google colab
# Standard library imports
import datetime

# Third party imports
from dateutil import tz

# Local application imports
import twin4build as tb
import twin4build.examples.utils as utils


def fcn(self):
    """
    Custom configuration function to set up the model after translation.
    This function adds missing connections, configures data sources, and sets up control parameters.
    """
    # Add supply water temperature schedule
    supply_water_schedule = tb.ScheduleSystem(
        weekDayRulesetDict={
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
        weekDayRulesetDict={
            "ruleset_default_value": 21,
            "ruleset_start_minute": [],
            "ruleset_end_minute": [],
            "ruleset_start_hour": [],
            "ruleset_end_hour": [],
            "ruleset_value": [],
        },
        id="boundary_temp_schedule",
    )

    # Add missing connections
    self.add_connection(
        boundary_temp_schedule,
        self.components["office"],
        "scheduleValue",
        "boundaryTemperature",
    )
    self.add_connection(
        supply_water_schedule,
        self.components["office_space_heater"],
        "scheduleValue",
        "supplyWaterTemperature",
    )

    # Configure sensor data sources
    self.components["office_temperature_sensor"].useSpreadsheet = True
    self.components["office_temperature_sensor"].filename = utils.get_path(
        ["estimator_example", "temperature_sensor.csv"]
    )

    self.components["office_co2_sensor"].useSpreadsheet = True
    self.components["office_co2_sensor"].filename = utils.get_path(
        ["estimator_example", "co2_sensor.csv"]
    )

    self.components["office_valve_position_sensor"].useSpreadsheet = True
    self.components["office_valve_position_sensor"].filename = utils.get_path(
        ["estimator_example", "valve_position_sensor.csv"]
    )

    self.components["office_damper_position_sensor"].useSpreadsheet = True
    self.components["office_damper_position_sensor"].filename = utils.get_path(
        ["estimator_example", "damper_position_sensor.csv"]
    )

    self.components["supply_air_temperature_sensor"].useSpreadsheet = True
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

    self.components["office_temperature_heating_setpoint"].useSpreadsheet = True
    self.components["office_temperature_heating_setpoint"].filename = utils.get_path(
        ["estimator_example", "temperature_heating_setpoint.csv"]
    )

    # Configure outdoor environment data
    self.components["outdoor_environment"].useSpreadsheet = True
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
model = tb.Model(id="translator_example")

# Local application imports
from twin4build.utils.print_progress import LOGGER

# from twin4build.systems.building_space.building_space_torch_system import saref_signature_pattern_sensor
LOGGER.hide_debug()
# Load the model from semantic file

filename = utils.get_path(["estimator_example", "one_room_example_model.xlsm"])
sm = tb.SemanticModel(rdf_file=filename, id="translator_example")

# translator = tb.Translator()

# tb.BuildingSpaceTorchSystem.sp = [saref_signature_pattern_sensor()]
# translator.translate(semantic_model=sm, systems_=[tb.BuildingSpaceTorchSystem], verbose=999999)

model.load(semantic_model_filename=filename, fcn=fcn, verbose=999999)


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
