# %pip install twin4build # Uncomment in google colab
import twin4build as tb
import datetime
from dateutil import tz
import twin4build.examples.utils as utils


def fcn(self):
    """
    Custom configuration function to set up the model after translation.
    This function adds missing connections, configures data sources, and sets up control parameters.
    """
    # Add supply water temperature schedule
    supply_water_schedule = tb.ScheduleSystem(
        weekDayRulesetDict = {
            "ruleset_default_value": 60,
            "ruleset_start_minute": [],
            "ruleset_end_minute": [],
            "ruleset_start_hour": [],
            "ruleset_end_hour": [],
            "ruleset_value": []
        },
        id="supply_water_schedule"
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
        id="boundary_temp_schedule"
    )

    print([c.id for c in self.components.values()])

    # Add missing connections
    self.add_connection(boundary_temp_schedule, self.components["020B"], "scheduleValue", "boundaryTemperature")
    self.add_connection(supply_water_schedule, self.components["020B_space_heater"], "scheduleValue", "supplyWaterTemperature")

    # Configure sensor data sources
    self.components["020B_temperature_sensor"].useSpreadsheet = True
    self.components["020B_temperature_sensor"].filename = utils.get_path(["estimator_example", "temperature_sensor.csv"])

    self.components["020B_co2_sensor"].useSpreadsheet = True
    self.components["020B_co2_sensor"].filename = utils.get_path(["estimator_example", "co2_sensor.csv"])

    self.components["020B_valve_position_sensor"].useSpreadsheet = True
    self.components["020B_valve_position_sensor"].filename = utils.get_path(["estimator_example", "valve_position_sensor.csv"])

    self.components["020B_damper_position_sensor"].useSpreadsheet = True
    self.components["020B_damper_position_sensor"].filename = utils.get_path(["estimator_example", "damper_position_sensor.csv"])

    self.components["BTA004"].useSpreadsheet = True
    self.components["BTA004"].filename = utils.get_path(["estimator_example", "supply_air_temperature.csv"])

    # Configure control setpoints
    self.components["020B_co2_setpoint"].weekDayRulesetDict = {
        "ruleset_default_value": 900,
        "ruleset_start_minute": [],
        "ruleset_end_minute": [],
        "ruleset_end_hour": [],
        "ruleset_start_hour": [],
        "ruleset_value": []
    }
    
    self.components["020B_occupancy_profile"].weekDayRulesetDict = {
        "ruleset_default_value": 0,
        "ruleset_start_minute": [],
        "ruleset_end_minute": [],
        "ruleset_start_hour": [],
        "ruleset_end_hour": [],
        "ruleset_value": []
    }
    
    self.components["020B_temperature_heating_setpoint"].useSpreadsheet = True
    self.components["020B_temperature_heating_setpoint"].filename = utils.get_path(["estimator_example", "temperature_heating_setpoint.csv"])
    
    # Configure outdoor environment data
    self.components["outdoor_environment"].useSpreadsheet = True
    self.components["outdoor_environment"].filename_outdoorTemperature = utils.get_path(["estimator_example", "outdoor_environment.csv"])
    self.components["outdoor_environment"].datecolumn_outdoorTemperature = 0
    self.components["outdoor_environment"].valuecolumn_outdoorTemperature = 1
    
    self.components["outdoor_environment"].filename_globalIrradiation = utils.get_path(["estimator_example", "outdoor_environment.csv"])
    self.components["outdoor_environment"].datecolumn_globalIrradiation = 0
    self.components["outdoor_environment"].valuecolumn_globalIrradiation = 2
    
    self.components["outdoor_environment"].filename_outdoorCo2Concentration = utils.get_path(["estimator_example", "outdoor_environment.csv"])
    self.components["outdoor_environment"].datecolumn_outdoorCo2Concentration = 0
    self.components["outdoor_environment"].valuecolumn_outdoorCo2Concentration = 3


# Create a new model
model = tb.Model(id="translator_example")

# Load the model from semantic file
filename = utils.get_path(["estimator_example", "one_room_example_model.xlsm"])
# sm = tb.SemanticModel(id="translator_example_semantic_model", rdf_file=filename, verbose=0)
# sm.reason()
# sm.serialize()
# aa
# sm.visualize()
# aa
# translator = tb.Translator()
# simulation_model = translator.translate(sm, systems_=[tb.PIDControllerSystem], verbose=100)
# aa
# simulation_model = translator.translate(sm, verbose=100)
# simulation_model.serialize()
# simulation_model.visualize()


model.load(semantic_model_filename=filename, fcn=fcn, verbose=100)


simulator = tb.Simulator(model)
step_size = 1200  # 20 minutes in seconds
start_time = [datetime.datetime(year=2023, month=11, day=27, hour=0, minute=0, second=0,
                                tzinfo=tz.gettz("Europe/Copenhagen")), 
            datetime.datetime(year=2023, month=11, day=28, hour=0, minute=0, second=0,
            tzinfo=tz.gettz("Europe/Copenhagen"))]
end_time = [datetime.datetime(year=2023, month=11, day=28, hour=0, minute=0, second=0,
                            tzinfo=tz.gettz("Europe/Copenhagen")), 
            datetime.datetime(year=2023, month=11, day=30, hour=0, minute=0, second=0,
            tzinfo=tz.gettz("Europe/Copenhagen"))]


simulator.simulate(step_size=step_size, start_time=start_time, end_time=end_time)


tb.plot.plot(simulator.date_time_steps, 
            [tb.plot.Entry(data=model.components["020B"].output["indoorTemperature"].history.detach().numpy(), 
                            label="Indoor Temperature", fmt="-", linewidth=2)], ylabel_1axis="Indoor Temperature [°C]", show=True)

aa





# simulation_model = translator.simulation_model

# 