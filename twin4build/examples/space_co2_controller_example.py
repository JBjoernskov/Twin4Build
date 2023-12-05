

import os
import sys
import datetime
from dateutil.tz import tzutc

# Only for testing before distributing package
if __name__ == '__main__':
    uppath = lambda _path,n: os.sep.join(_path.split(os.sep)[:-n])
    file_path = uppath(os.path.abspath(__file__), 3)
    sys.path.append(file_path)

from twin4build.model.model import Model
from twin4build.simulator.simulator import Simulator
from twin4build.saref4bldg.physical_object.building_object.building_device.distribution_device.distribution_flow_device.flow_controller.damper.damper_system import DamperSystem
from twin4build.saref4bldg.physical_object.building_object.building_device.distribution_device.distribution_control_device.controller.controller_system import ControllerSystem
from twin4build.saref4bldg.building_space.building_space_system_co2 import BuildingSpaceSystem
from twin4build.saref.measurement.measurement import Measurement
from twin4build.utils.schedule import Schedule
from twin4build.utils.plot import plot
from twin4build.saref.property_.Co2.Co2 import Co2
import twin4build.utils.plot.plot as plot


def fcn(self):
    ##############################################################
    ################## First, define components ##################
    ##############################################################
    occupancy_schedule = Schedule(
        weekDayRulesetDict={
            "ruleset_default_value": 0,
            "ruleset_start_minute": [0, 0, 0, 0, 0, 0, 0],
            "ruleset_end_minute": [0, 0, 0, 0, 0, 0, 0],
            "ruleset_start_hour": [6, 7, 8, 12, 14, 16, 18],
            "ruleset_end_hour": [7, 8, 12, 14, 16, 18, 22],
            "ruleset_value": [3, 5, 20, 25, 27, 7, 3]},
        add_noise=True,
        saveSimulationResult=True,
        id="Occupancy schedule")

    co2_setpoint_schedule = Schedule(
        weekDayRulesetDict={
            "ruleset_default_value": 600,
            "ruleset_start_minute": [],
            "ruleset_end_minute": [],
            "ruleset_start_hour": [],
            "ruleset_end_hour": [],
            "ruleset_value": []},
        saveSimulationResult=True,
        id="CO2 setpoint schedule")

    co2_property = Co2()
    co2_controller = ControllerSystem(
        controlsProperty=co2_property,
        K_p=-0.001,
        K_i=-0.001,
        K_d=0,
        saveSimulationResult=True,
        id="CO2 controller")

    supply_damper = DamperSystem(
        nominalAirFlowRate=Measurement(hasValue=1.6),
        a=5,
        saveSimulationResult=True,
        id="Supply damper")

    return_damper = DamperSystem(
        nominalAirFlowRate=Measurement(hasValue=1.6),
        a=5,
        saveSimulationResult=True,
        id="Return damper")

    space = BuildingSpaceSystem(
        airVolume=466.54,
        outdoorCo2Concentration=500,
        infiltration=0.005,
        generationCo2Concentration=0.0042*1000*1.225,
        saveSimulationResult=True,
        id="Space")
    co2_property.isPropertyOf = space

    #################################################################
    ################## Add connections to the model #################
    #################################################################
    self.add_connection(co2_controller, supply_damper,
                         "inputSignal", "damperPosition")
    self.add_connection(co2_controller, return_damper,
                         "inputSignal", "damperPosition")
    self.add_connection(supply_damper, space,
                         "airFlowRate", "supplyAirFlowRate")
    self.add_connection(return_damper, space,
                         "airFlowRate", "returnAirFlowRate")
    self.add_connection(occupancy_schedule, space,
                         "scheduleValue", "numberOfPeople")
    self.add_connection(space, co2_controller,
                         "indoorCo2Concentration", "actualValue")
    self.add_connection(co2_setpoint_schedule, co2_controller,
                         "scheduleValue", "setpointValue")

def space_co2_controller_example():
    '''
        The code defines a simulation model for a building's CO2 control system using 
        various components like schedules, controllers, and damper models, and establishes 
        connections between them. The simulation period is set, and the simulation results can be saved. 
        The code also includes a test function that initializes and adds components to the model and establishes
        connections between them.
    '''
    stepSize = 600 #Seconds
    startTime = datetime.datetime(year=2021, month=1, day=10, hour=0, minute=0, second=0)
    endTime = datetime.datetime(year=2021, month=1, day=12, hour=0, minute=0, second=0)
    model = Model(id="example_model")
    model.load_model(fcn=fcn, infer_connections=False)
    
    # Create a simulator instance 
    simulator = Simulator()

    # Simulate the model
    simulator.simulate(model=model,
                        stepSize=stepSize,
                        startTime=startTime,
                        endTime=endTime)
    
    plot.plot_damper(model=model, simulator=simulator, damper_id="Supply damper")
    plot.plot_space_CO2(model=model, simulator=simulator, space_id="Space")
    plot.plot_CO2_controller(model=model, simulator=simulator, CO2_controller_id="CO2 controller", show=True) #Set show=True to plot

if __name__ == '__main__':
    space_co2_controller_example()
