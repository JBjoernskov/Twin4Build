

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
from twin4build.saref4bldg.physical_object.building_object.building_device.distribution_device.distribution_flow_device.flow_controller.damper.damper_model import DamperModel
from twin4build.saref4bldg.building_space.building_space_model_co2 import BuildingSpaceModelCo2
from twin4build.saref.measurement.measurement import Measurement
from twin4build.utils.schedule import Schedule



def test():

    #If True, inputs and outputs are saved for each timestep during simulation
    createReport = True

    # This creates a default plot for each component 
    do_plot = True
    timeStep = 600 #Seconds
    startPeriod = datetime.datetime(year=2018, month=1, day=1, hour=0, minute=0, second=0, tzinfo=tzutc())
    endPeriod = datetime.datetime(year=2018, month=1, day=5, hour=0, minute=0, second=0, tzinfo=tzutc())
    model = Model(timeStep = timeStep,
                        startPeriod = startPeriod,
                        endPeriod = endPeriod,
                        createReport = createReport)
    
    ##############################################################
    ################## First, define components ##################
    ##############################################################
    occupancy_schedule = Schedule(
            startPeriod = startPeriod,
            timeStep = timeStep,
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
            createReport = createReport,
            connectedThrough = [],
            connectsAt = [],
            id = "Occupancy schedule")


    position_schedule = Schedule(
            startPeriod = model.startPeriod,
            timeStep = model.timeStep,
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
            createReport = model.createReport,
            connectedThrough = [],
            connectsAt = [],
            id = "Position schedule") 

    supply_damper = DamperModel(
        nominalAirFlowRate = Measurement(hasValue=1.6),
        a = 5,
        subSystemOf = [],
        input = {},
        output = {},
        savedInput = {},
        savedOutput = {},
        createReport = model.createReport,
        connectedThrough = [],
        connectsAt = [],
        id = "Supply damper")


    return_damper = DamperModel(
        nominalAirFlowRate = Measurement(hasValue=1.6),
        a = 5,
        subSystemOf = [],
        input = {},
        output = {},
        savedInput = {},
        savedOutput = {},
        createReport = model.createReport,
        connectedThrough = [],
        connectsAt = [],
        id = "Return damper")

    space = BuildingSpaceModelCo2(
        timeStep = model.timeStep,
        airVolume=466.54,
        densityAir = 1.225,
        startPeriod = startPeriod,
        input = {"generationCo2Concentration": 0.000008316,
                "outdoorCo2Concentration": 400,
                "infiltration": 0.07},
        output = {"indoorCo2Concentration": 500},
        savedInput = {},
        savedOutput = {},
        createReport = createReport,
        connectedThrough = [],
        connectsAt = [],
        id = "Space")


    ######################################################################################
    ################## Add components to the model and add a connection ##################
    ######################################################################################
    model.add_component(occupancy_schedule)
    model.add_component(position_schedule)
    model.add_component(supply_damper)
    model.add_component(return_damper)
    model.add_component(space)
    model.add_connection(position_schedule, supply_damper, "scheduleValue", "damperPosition")
    model.add_connection(position_schedule, return_damper, "scheduleValue", "damperPosition")
    model.add_connection(supply_damper, space, "airFlowRate", "supplyAirFlowRate")
    model.add_connection(return_damper, space, "airFlowRate", "returnAirFlowRate")
    model.add_connection(occupancy_schedule, space, "scheduleValue", "numberOfPeople")

    # Cycles are not allowed (with the exeption of controllers). If the follwing line is commented in, 
    # a cycle is introduced and the model will generate an error when "model.get_execution_order()" is run". 
    # You can see the generated graph with the cycle in the "system_graph.png" file.
    # model.add_connection(damper, position_schedule, "airFlow", "damperPosition") <- comment in to create a cycle




    # This will draw a graph of all components and connections in the model.
    # The file is called "system_graph.png"
    model.draw_system_graph()

    # This generates the correct execution order for the different components
    model.get_execution_order()
    model.init_building_space_models()
    model.draw_system_graph_no_cycles()

    # This is the order by which component models are executed in each timestep
    model.draw_flat_execution_graph()


    
    # Create a simulator instance 
    simulator = Simulator(timeStep = timeStep,
                            startPeriod = startPeriod,
                            endPeriod = endPeriod,
                            do_plot = do_plot)

    # Simulate the model
    simulator.simulate(model)


    if do_plot:
        import matplotlib.pyplot as plt
        plt.show()

if __name__ == '__main__':
    test()

