import twin4build as tb
from twin4build.utils.uppath import uppath
def fcn(self):
    '''
        The fcn() function adds connections between components in a system model, 
        creates a schedule object, and adds it to the component dictionary.
        The test() function sets simulation parameters and runs a simulation of the system 
        model using the Simulator() class. It then generates several plots of the simulation results using functions from the plot module.
    '''


    space = tb.BuildingSpaceSystem(id="OE20-601b-2", airVolume=466.54)
    temperature_controller = tb.ControllerSystem(id="Temperature controller", K_p=2.50773924e-01, K_i=4.38174242e-01, K_d=0)
    co2_controller = tb.RulebasedControllerSystem(id="CO2 controller")
    supply_damper = tb.DamperSystem(id="Supply damper", airFlowRateMax=tb.PropertyValue(hasValue=0.544444444))
    exhaust_damper = tb.DamperSystem(id="Exhaust damper", airFlowRateMax=tb.PropertyValue(hasValue=0.544444444))
    space_heater = tb.SpaceHeaterSystem(id="Space heater", 
                                        heatTransferCoefficient=8.31495759e+01,
                                        thermalMassHeatCapacity=tb.PropertyValue(hasValue=2.72765272e+06),
                                        temperatureClassification=tb.PropertyValue("45/30-21"))
    valve = tb.ValveSystem(id="Valve", waterFlowRateMax=0.0202, valveAuthority=1)
    heating_meter = tb.MeterSystem(id="Heating meter")
    temperature_sensor = tb.SensorSystem(id="OE20-601b-2 temperature sensor")
    co2_sensor = tb.SensorSystem(id="CO2 sensor")





    self.add_connection(co2_controller, supply_damper, "inputSignal", "damperPosition")
    self.add_connection(co2_controller, exhaust_damper, "inputSignal", "damperPosition")
    self.add_connection(co2_sensor, co2_controller, "measuredValue", "actualValue")
    self.add_connection(supply_damper, space, "airFlowRate", "supplyAirFlowRate")
    self.add_connection(exhaust_damper, space, "airFlowRate", "returnAirFlowRate")
    self.add_connection(supply_damper, space, "damperPosition", "supplyDamperPosition")
    self.add_connection(exhaust_damper, space, "damperPosition", "returnDamperPosition")
    self.add_connection(space, temperature_sensor, "indoorTemperature", "measuredValue")
    self.add_connection(space, co2_sensor, "indoorCo2Concentration", "measuredValue")
    self.add_connection(temperature_sensor, temperature_controller, "measuredValue", "actualValue")
    self.add_connection(valve, space, "valvePosition", "valvePosition")
    self.add_connection(temperature_controller, valve, "inputSignal", "valvePosition")
    self.add_connection(valve, space_heater, "waterFlowRate", "waterFlowRate")
    self.add_connection(space, space_heater, "indoorTemperature", "indoorTemperature")


    t = tb.Temperature()
    c = tb.Co2()
    temperature_controller.observes = t
    co2_controller.observes = c
    t.isPropertyOf = space
    c.isPropertyOf = space