import twin4build as tb
from twin4build.utils.uppath import uppath
def fcn(self):
    '''
        The fcn() function adds connections between components in a system model, 
        creates a schedule object, and adds it to the component dictionary.
        The test() function sets simulation parameters and runs a simulation of the system 
        model using the Simulator() class. It then generates several plots of the simulation results using functions from the plot module.
    '''


    space = tb.BuildingSpaceSystem(id="space", airVolume=466.54)
    temperature_controller = tb.ControllerSystem(id="temperature controller", K_p=2.50773924e-01, K_i=4.38174242e-01, K_d=0)
    co2_controller = tb.RulebasedControllerSystem(id="CO2 controller")
    supply_damper = tb.DamperSystem(id="supply damper", nominalAirFlowRate=tb.PropertyValue(hasValue=0.544444444))
    exhaust_damper = tb.DamperSystem(id="exhaust damper", nominalAirFlowRate=tb.PropertyValue(hasValue=0.544444444))
    space_heater = tb.SpaceHeaterSystem(id="space heater", 
                                        heatTransferCoefficient=8.31495759e+01,
                                        thermalMassHeatCapacity=tb.PropertyValue(hasValue=2.72765272e+06),
                                        temperatureClassification=tb.PropertyValue("45/30-21"))
    valve = tb.ValveSystem(id="valve", waterFlowRateMax=0.0202, valveAuthority=1)
    heating_meter = tb.MeterSystem(id="heating meter")
    temperature_sensor = tb.SensorSystem(id="temperature sensor")
    co2_sensor = tb.SensorSystem(id="CO2 sensor")
    valve_position_sensor = tb.SensorSystem(id="valve position sensor")
    damper_position_sensor = tb.SensorSystem(id="damper position sensor")

    supply_water_temperature_setpoint_schedule = tb.PiecewiseLinearScheduleSystem(
            weekDayRulesetDict = {
                "ruleset_default_value": {"X": [-5, 5, 7],
                                          "Y": [58, 65, 60.5]},
                "ruleset_start_minute": [0],
                "ruleset_end_minute": [0],
                "ruleset_start_hour": [5],
                "ruleset_end_hour": [7],
                "ruleset_value": [{"X": [-7, 5, 9],
                                    "Y": [72, 55, 50]}]},
            saveSimulationResult = True,
            id = "Supply water temperature")
    
    supply_air_temperature_schedule = tb.ScheduleSystem(
            weekDayRulesetDict = {
                "ruleset_default_value": 21,
                "ruleset_start_minute": [],
                "ruleset_end_minute": [],
                "ruleset_start_hour": [],
                "ruleset_end_hour": [],
                "ruleset_value": []},
            saveSimulationResult = True,
            id = "Supply air temperature")
    

    

    self.add_connection(supply_water_temperature_setpoint_schedule, space, "scheduleValue", "supplyWaterTemperature")
    self.add_connection(supply_air_temperature_schedule, space, "scheduleValue", "supplyAirTemperature")
    self.add_connection(co2_controller, supply_damper, "inputSignal", "damperPosition")
    self.add_connection(co2_controller, exhaust_damper, "inputSignal", "damperPosition")
    self.add_connection(co2_sensor, co2_controller, "indoorCo2Concentration", "actualValue")
    self.add_connection(supply_damper, space, "airFlowRate", "supplyAirFlowRate")
    self.add_connection(exhaust_damper, space, "airFlowRate", "returnAirFlowRate")
    self.add_connection(supply_damper, space, "damperPosition", "supplyDamperPosition")
    self.add_connection(exhaust_damper, space, "damperPosition", "returnDamperPosition")
    self.add_connection(space, temperature_sensor, "indoorTemperature", "indoorTemperature")
    self.add_connection(space, co2_sensor, "indoorCo2Concentration", "indoorCo2Concentration")
    self.add_connection(temperature_sensor, temperature_controller, "indoorTemperature", "actualValue")
    self.add_connection(valve, space, "valvePosition", "valvePosition")
    self.add_connection(temperature_controller, valve, "inputSignal", "valvePosition")
    self.add_connection(valve, space_heater, "waterFlowRate", "waterFlowRate")
    self.add_connection(space, space_heater, "indoorTemperature", "indoorTemperature")
    self.add_connection(valve, valve_position_sensor, "valvePosition", "valvePosition")
    self.add_connection(supply_damper, damper_position_sensor, "damperPosition", "damperPosition")
    self.add_connection(space_heater, heating_meter, "Energy", "Energy")

    t = tb.Temperature()
    c = tb.Co2()
    temperature_controller.observes = t
    co2_controller.observes = c
    t.isPropertyOf = space
    c.isPropertyOf = space