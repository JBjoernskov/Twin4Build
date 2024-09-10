

import os
import sys
import datetime
from dateutil.tz import tzutc
from dateutil import tz
###Only for testing before distributing package
if __name__ == '__main__':
    uppath = lambda _path,n: os.sep.join(_path.split(os.sep)[:-n])
    file_path = uppath(os.path.abspath(__file__), 3)
    sys.path.append(file_path)

import twin4build as tb
import twin4build.utils.plot.plot as plot

def fcn(self):
        
    ##############################################################
    ################## First, define components ##################
    ##############################################################
    occupancy_schedule = tb.ScheduleSystem(
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
    # occupancy = TimeSeriesInputSystem(id="test", filename=filename)

    position_schedule = tb.ScheduleSystem(
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

    supply_damper = tb.DamperSystem(
        nominalAirFlowRate = tb.PropertyValue(hasValue=1.6),
        a = 5,
        saveSimulationResult = True,
        id = "Supply damper")


    return_damper = tb.DamperSystem(
        nominalAirFlowRate = tb.PropertyValue(hasValue=1.6),
        a = 5,
        saveSimulationResult = True,
        id = "Return damper")

    space = tb.BuildingSpaceCo2System(
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
    startTime = datetime.datetime(year=2021, month=1, day=10, hour=0, minute=0, second=0, tzinfo=tz.gettz("Europe/Copenhagen")) 
    endTime = datetime.datetime(year=2021, month=1, day=12, hour=0, minute=0, second=0, tzinfo=tz.gettz("Europe/Copenhagen"))
    model = tb.Model(id="Co2 model")
    model.load(fcn=fcn, infer_connections=False)


    # Create a simulator instance 
    simulator = tb.Simulator()

    # Simulate the model
    simulator.simulate(model=model,
                       stepSize=stepSize,
                        startTime=startTime,
                        endTime=endTime)
    
    plot.plot_damper(model=model, simulator=simulator, damper_id="Supply damper")
    plot.plot_space_CO2(model=model, simulator=simulator, space_id="Space", show=False, ylim_3ax=[0,2]) #Set show=True to plot


if __name__ == '__main__':
    space_co2_no_controller_example()

