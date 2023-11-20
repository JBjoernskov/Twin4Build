

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
from twin4build.saref4bldg.building_space.building_space_system_co2 import BuildingSpaceSystemCo2
from twin4build.saref.measurement.measurement import Measurement
from twin4build.utils.schedule import Schedule

from twin4build.logger.Logging import Logging

logger = Logging.get_logger("ai_logfile")

def test():

    
    logger.info("[Space CO2 NO Controller Example] : Test function Entered")


    #If True, inputs and outputs are saved for each timestep during simulation
    saveSimulationResult = True

    # This creates a default plot for each component 
    do_plot = True
    stepSize = 600 #Seconds
    startPeriod = datetime.datetime(year=2018, month=1, day=1, hour=0, minute=0, second=0, tzinfo=tzutc())
    endPeriod = datetime.datetime(year=2018, month=1, day=5, hour=0, minute=0, second=0, tzinfo=tzutc())
    model = Model(stepSize = stepSize,
                        startPeriod = startPeriod,
                        endPeriod = endPeriod,
                        saveSimulationResult = saveSimulationResult)
    
    ##############################################################
    ################## First, define components ##################
    ##############################################################
    occupancy_schedule = Schedule(
            startPeriod = startPeriod,
            stepSize = stepSize,
            rulesetDict = {
                "ruleset_default_value": 0,
                "ruleset_start_minute": [0,0,0,0,0,0,0],
                "ruleset_end_minute": [0,0,0,0,0,0,0],
                "ruleset_start_hour": [6,7,8,12,14,16,18],
                "ruleset_end_hour": [7,8,12,14,16,18,22],
                "ruleset_value": [3,5,20,25,27,7,3]}, #35
            add_noise = True,
            input = {},
            output = {},
            savedInput = {},
            savedOutput = {},
            saveSimulationResult = saveSimulationResult,
            connectedThrough = [],
            connectsAt = [],
            id = "Occupancy schedule")


    position_schedule = Schedule(
            startPeriod = model.startPeriod,
            stepSize = model.stepSize,
            rulesetDict = {
                "ruleset_default_value": 0,
                "ruleset_start_minute": [0,0,0,0,0,0,0],
                "ruleset_end_minute": [0,0,0,0,0,0,0],
                "ruleset_start_hour": [6,7,8,12,14,16,18],
                "ruleset_end_hour": [7,8,12,14,16,18,22],
                "ruleset_value": [0,0.1,1,0,0,0.5,0.7]}, #35
            add_noise = False,
            input = {},
            output = {},
            savedInput = {},
            savedOutput = {},
            saveSimulationResult = model.saveSimulationResult,
            connectedThrough = [],
            connectsAt = [],
            id = "Position schedule") 

    supply_damper = DamperSystem(
        nominalAirFlowRate = Measurement(hasValue=1.6),
        a = 5,
        subSystemOf = [],
        input = {},
        output = {},
        savedInput = {},
        savedOutput = {},
        saveSimulationResult = model.saveSimulationResult,
        connectedThrough = [],
        connectsAt = [],
        id = "Supply damper")


    return_damper = DamperSystem(
        nominalAirFlowRate = Measurement(hasValue=1.6),
        a = 5,
        subSystemOf = [],
        input = {},
        output = {},
        savedInput = {},
        savedOutput = {},
        saveSimulationResult = model.saveSimulationResult,
        connectedThrough = [],
        connectsAt = [],
        id = "Return damper")

    space = BuildingSpaceSystemCo2(
        stepSize = model.stepSize,
        airVolume=466.54,
        densityAir = 1.225,
        startPeriod = startPeriod,
        input = {"generationCo2Concentration": 0.000008316,
                "outdoorCo2Concentration": 400,
                "infiltration": 0.07},
        output = {"indoorCo2Concentration": 500},
        savedInput = {},
        savedOutput = {},
        saveSimulationResult = saveSimulationResult,
        connectedThrough = [],
        connectsAt = [],
        id = "Space")


    #################################################################
    ################## Add components to the model ##################
    #################################################################
    model.add_component(occupancy_schedule)
    model.add_component(position_schedule)
    model.add_component(supply_damper)
    model.add_component(return_damper)
    model.add_component(space)




    #################################################################
    ################## Add connections to the model #################
    #################################################################
    model.add_connection(position_schedule, supply_damper, "scheduleValue", "damperPosition")
    model.add_connection(position_schedule, return_damper, "scheduleValue", "damperPosition")
    model.add_connection(supply_damper, space, "airFlowRate", "supplyAirFlowRate")
    model.add_connection(return_damper, space, "airFlowRate", "returnAirFlowRate")
    model.add_connection(occupancy_schedule, space, "scheduleValue", "numberOfPeople")

    # Cycles are not allowed (with the exeption of controllers - see the controller example). If the follwing line is commented in, 
    # a cycle is introduced and the model will generate an error when "model.get_execution_order()" is run". 
    # You can see the generated graph with the cycle in the "system_graph.png" file.
    # model.add_connection(supply_damper, position_schedule, "test1234", "test1234") #<-------------------------- comment in to create a cycle




    # This will draw a graph of all components and connections in the model.
    # The file is called "system_graph.png"
    model.draw_system_graph()

    # This generates the correct execution order for the different components, given that the model contains no cycles
    model.get_execution_order()
    model.draw_system_graph_no_cycles()

    # This is the order by which component models are executed in each timestep
    model.draw_flat_execution_graph()


    
    # Create a simulator instance 
    simulator = Simulator(stepSize = model.stepSize,
                            startPeriod = model.startPeriod,
                            endPeriod = model.endPeriod,
                            do_plot = do_plot)

    # Simulate the model
    simulator.simulate(model)


    if do_plot:
        import matplotlib.pyplot as plt
        plt.show()

        
    logger.info("[Space CO2 NO Controller Example] : Test function Entered")


if __name__ == '__main__':
    test()

