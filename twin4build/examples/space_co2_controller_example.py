

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
from twin4build.saref4bldg.physical_object.building_object.building_device.distribution_device.distribution_control_device.controller.controller_model import ControllerModel
from twin4build.saref4bldg.building_space.building_space_model_co2 import BuildingSpaceModelCo2
from twin4build.saref.measurement.measurement import Measurement
from twin4build.utils.schedule import Schedule

from twin4build.saref.property_.co2.co2 import Co2

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
            startPeriod = model.startPeriod,
            timeStep = model.timeStep,
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
            createReport = model.createReport,
            connectedThrough = [],
            connectsAt = [],
            id = "Occupancy schedule")
        
    co2_setpoint_schedule = Schedule(
        startPeriod = model.startPeriod,
        timeStep = model.timeStep,
        rulesetDict = {
            "ruleset_default_value": 600,
            "ruleset_start_minute": [],
            "ruleset_end_minute": [],
            "ruleset_start_hour": [],
            "ruleset_end_hour": [],
            "ruleset_value": []},
        input = {},
        output = {},
        savedInput = {},
        savedOutput = {},
        createReport = model.createReport,
        connectedThrough = [],
        connectsAt = [],
        id = "CO2 setpoint schedule")

    co2_property = Co2()
    co2_controller = ControllerModel(
        controlsProperty = co2_property,
        K_p = -0.001,
        K_i = -0.001,
        K_d = 0,
        input = {},
        output = {},
        savedInput = {},
        savedOutput = {},
        createReport = model.createReport,
        connectedThrough = [],
        connectsAt = [],
        id = "CO2 controller") 

    supply_damper = DamperModel(
        nominalAirFlowRate = Measurement(hasValue=1.6),
        a = 5,
        subSystemOf = [],
        input = {},
        output = {"airFlowRate": 0}, #Because we have a controller in the system, an inital value is needed
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
        output = {"airFlowRate": 0}, #Because we have a controller in the system, an inital value is needed
        savedInput = {},
        savedOutput = {},
        createReport = model.createReport,
        connectedThrough = [],
        connectsAt = [],
        id = "Return damper")

    space = BuildingSpaceModelCo2(
        hasProperty = [co2_property],
        timeStep = model.timeStep,
        airVolume=466.54,
        densityAir = 1.225,
        startPeriod = startPeriod,
        input = {"generationCo2Concentration": 0.000008316,
                "outdoorCo2Concentration": 400,
                "infiltration": 0.07},
        output = {"indoorCo2Concentration": 500}, #Initial value
        savedInput = {},
        savedOutput = {},
        createReport = createReport,
        connectedThrough = [],
        connectsAt = [],
        id = "Space")
    co2_property.isPropertyOf = space


    #################################################################
    ################## Add components to the model ##################
    #################################################################
    model.add_component(occupancy_schedule)
    model.add_component(co2_setpoint_schedule)
    model.add_component(co2_controller)
    model.add_component(supply_damper)
    model.add_component(return_damper)
    model.add_component(space)




    #################################################################
    ################## Add connections to the model #################
    #################################################################
    model.add_connection(co2_controller, supply_damper, "inputSignal", "damperPosition")
    model.add_connection(co2_controller, return_damper, "inputSignal", "damperPosition")
    model.add_connection(supply_damper, space, "airFlowRate", "supplyAirFlowRate")
    model.add_connection(return_damper, space, "airFlowRate", "returnAirFlowRate")
    model.add_connection(occupancy_schedule, space, "scheduleValue", "numberOfPeople")
    model.add_connection(space, co2_controller, "indoorCo2Concentration", "actualValue")
    model.add_connection(co2_setpoint_schedule, co2_controller, "scheduleValue", "setpointValue")


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
    simulator = Simulator(timeStep = model.timeStep,
                            startPeriod = model.startPeriod,
                            endPeriod = model.endPeriod,
                            do_plot = do_plot)

    # Simulate the model
    simulator.simulate(model)


    if do_plot:
        import matplotlib.pyplot as plt
        plt.show()

if __name__ == '__main__':
    test()

