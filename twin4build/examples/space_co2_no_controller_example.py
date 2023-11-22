

import os
import sys
import datetime
from dateutil.tz import tzutc

###Only for testing before distributing package
if __name__ == '__main__':
    uppath = lambda _path,n: os.sep.join(_path.split(os.sep)[:-n])
    file_path = uppath(os.path.abspath(__file__), 3)
    sys.path.append(file_path)

from twin4build.model.model import Model
from twin4build.simulator.simulator import Simulator
from twin4build.saref4bldg.physical_object.building_object.building_device.distribution_device.distribution_flow_device.flow_controller.damper.damper_system import DamperSystem
from twin4build.saref4bldg.building_space.building_space_system_co2 import BuildingSpaceSystem
from twin4build.saref.measurement.measurement import Measurement
from twin4build.utils.schedule import Schedule
import twin4build.utils.plot.plot as plot
from twin4build.utils.time_series_input import TimeSeriesInput



def extend_model(self):
        
    ##############################################################
    ################## First, define components ##################
    ##############################################################
    occupancy_schedule = Schedule(
            weekDayRulesetDict = {
                "ruleset_default_value": 0,
                "ruleset_start_minute": [0,0,0,0,0,0,0],
                "ruleset_end_minute": [0,0,0,0,0,0,0],
                "ruleset_start_hour": [6,7,8,12,14,16,18],
                "ruleset_end_hour": [7,8,12,14,16,18,22],
                "ruleset_value": [3,5,20,25,27,7,3]}, #35
            add_noise = True,
            saveSimulationResult = True,
            id = "Occupancy schedule")
    
    # filename = "occ.csv"
    # occupancy = TimeSeriesInput(id="test", filename=filename)

    position_schedule = Schedule(
            weekDayRulesetDict = {
                "ruleset_default_value": 0,
                "ruleset_start_minute": [0,0,0,0,0,0,0],
                "ruleset_end_minute": [0,0,0,0,0,0,0],
                "ruleset_start_hour": [6,7,8,12,14,16,18],
                "ruleset_end_hour": [7,8,12,14,16,18,22],
                "ruleset_value": [0,0.1,1,0,0,0.5,0.7]}, #35
            add_noise = False,
            saveSimulationResult = True,
            id = "Position schedule")

    supply_damper = DamperSystem(
        nominalAirFlowRate = Measurement(hasValue=1.6),
        a = 5,
        saveSimulationResult = True,
        id = "Supply damper")


    return_damper = DamperSystem(
        nominalAirFlowRate = Measurement(hasValue=1.6),
        a = 5,
        saveSimulationResult = True,
        id = "Return damper")

    space = BuildingSpaceSystem(
        airVolume=466.54,
        outdoorCo2Concentration=500,
        infiltration=0.005,
        generationCo2Concentration=0.0042*1000*1.225,
        saveSimulationResult=True,
        id="Space")


    #################################################################
    ################## Add connections to the model #################
    #################################################################
    self.add_connection(position_schedule, supply_damper, "scheduleValue", "damperPosition")
    self.add_connection(position_schedule, return_damper, "scheduleValue", "damperPosition")
    self.add_connection(supply_damper, space, "airFlowRate", "supplyAirFlowRate")
    self.add_connection(return_damper, space, "airFlowRate", "returnAirFlowRate")
    self.add_connection(occupancy_schedule, space, "scheduleValue", "numberOfPeople")

    # Cycles are not allowed (with the exeption of controllers - see the controller example). If the follwing line is commented in, 
    # a cycle is introduced and the model will generate an error when "model.get_execution_order()" is run". 
    # You can see the generated graph with the cycle in the "system_graph.png" file.
    # model.add_connection(supply_damper, position_schedule, "test1234", "test1234") #<-------------------------- comment in to create a cycle

def space_co2_no_controller_example():

    stepSize = 60 #Seconds
    startPeriod = datetime.datetime(year=2018, month=1, day=1, hour=0, minute=0, second=0, tzinfo=tzutc())
    endPeriod = datetime.datetime(year=2018, month=1, day=5, hour=0, minute=0, second=0, tzinfo=tzutc())
    model = Model(id="Co2 model")

    model.load_model(extend_model=extend_model, infer_connections=False)


    # Create a simulator instance 
    simulator = Simulator()

    # Simulate the model
    simulator.simulate(model=model,
                       stepSize=stepSize,
                        startPeriod=startPeriod,
                        endPeriod=endPeriod)
    
    plot.plot_damper(model=model, simulator=simulator, damper_id="Supply damper")
    plot.plot_space_CO2(model=model, simulator=simulator, space_id="Space", show=False, ylim_3ax=[0,2]) #Set show=True to plot


if __name__ == '__main__':
    space_co2_no_controller_example()

