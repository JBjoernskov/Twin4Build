import saref4bldg
import saref4syst
# from saref4bldg.physical_object.building_object.building_device.distribution_device import distribution_device
import saref4bldg.physical_object.building_object.building_device.distribution_device.distribution_device as distribution_device
import saref4bldg.physical_object.building_object.building_device.distribution_device.distribution_flow_device as distribution_flow_device
import saref4bldg.physical_object.building_object.building_device.distribution_device.distribution_control_device as distribution_control_device
import weather_station
import schedule



from dateutil.tz import tzutc
import datetime
import matplotlib.pyplot as plt
import networkx as nx
import pydot
from netgraph import Graph#; help(Graph)


from space_data_collection import SpaceDataCollection
import os


class BuildingEnergyModel:
    def __init__(self,
                timeStep = None,
                startPeriod = None,
                endPeriod = None,
                createReport = False):
        self.timeStep = timeStep
        self.startPeriod = startPeriod
        self.endPeriod = endPeriod
        self.createReport = createReport

        self.system_graph = nx.MultiDiGraph() ###

        self.system_graph_node_attribute_dict = {}
        self.system_graph_edge_label_dict = {}

    def add_edge_(self, a, b, label):
        if (a, b) in self.system_graph.edges:
            max_rad = max(x[2]['rad'] for x in self.system_graph.edges(data=True) if sorted(x[:2]) == sorted([a,b]))
        else:
            max_rad = 0

        self.system_graph.add_edge(a, b, rad=max_rad+0, label=label)


    def add_connection(self, sender_obj, reciever_obj, senderPropertyName, recieverPropertyName):
        sender_obj_connection = saref4syst.connection.Connection(connectsSystem = sender_obj, senderPropertyName = senderPropertyName)
        sender_obj.connectedThrough.append(sender_obj_connection)
        reciever_obj_connection_point = saref4syst.connection_point.ConnectionPoint(connectionPointOf = reciever_obj, connectsSystemThrough = sender_obj_connection, recieverPropertyName = recieverPropertyName)
        sender_obj_connection.connectsSystemAt = reciever_obj_connection_point
        reciever_obj.connectsAt.append(reciever_obj_connection_point)

        self.add_edge_(sender_obj.systemId, reciever_obj.systemId, label=senderPropertyName) ###

        
        self.system_graph_node_attribute_dict[sender_obj.systemId] = {"label": sender_obj.__class__.__name__}
        self.system_graph_node_attribute_dict[reciever_obj.systemId] = {"label": reciever_obj.__class__.__name__}
        self.system_graph_edge_label_dict[(sender_obj.systemId, reciever_obj.systemId)] = senderPropertyName
    
    def load_model(self):

        hvac_system = distribution_device.DistributionDevice(subSystemOf = [], hasSubSystem = [])
        heating_system = distribution_device.DistributionDevice(subSystemOf = [hvac_system], hasSubSystem = [])
        ventilation_system = distribution_device.DistributionDevice(subSystemOf = [hvac_system], hasSubSystem = [])
        cooling_system = distribution_device.DistributionDevice(subSystemOf = [hvac_system], hasSubSystem = [])

        ventilation_system_dict = {"VentilationSystem1": distribution_device.DistributionDevice(subSystemOf = [hvac_system])}
        heating_system_dict = {"HeatingSystem1": distribution_device.DistributionDevice(subSystemOf = [hvac_system])}
        cooling_system_dict = {"CoolingSystem1": distribution_device.DistributionDevice(subSystemOf = [hvac_system])}

        

        _weather_station = weather_station.WeatherStation(startPeriod = self.startPeriod,
                                                        endPeriod = self.endPeriod,
                                                        input = {},
                                                        output = {},
                                                        savedInput = {},
                                                        savedOutput = {},
                                                        createReport = self.createReport,
                                                        connectedThrough = [],
                                                        connectsAt = [])

        occupancy_schedule = schedule.Schedule(startPeriod = self.startPeriod,
                                                timeStep = self.timeStep,
                                                rulesetDict = {
                                                    "ruleset_default_value": 0,
                                                    "ruleset_start_minute": [0,0,0,0,0],
                                                    "ruleset_end_minute": [0,0,0,0,0],
                                                    "ruleset_start_hour": [0,5,8,12,18],
                                                    "ruleset_end_hour": [6,8,12,18,22],
                                                    "ruleset_value": [0,0,0,0,0]}, #35
                                                input = {},
                                                output = {},
                                                savedInput = {},
                                                savedOutput = {},
                                                createReport = False,
                                                connectedThrough = [],
                                                connectsAt = [])

        temperature_setpoint_schedule = schedule.Schedule(startPeriod = self.startPeriod,
                                                timeStep = self.timeStep,
                                                rulesetDict = {
                                                    "ruleset_default_value": 20,
                                                    "ruleset_start_minute": [0,0],
                                                    "ruleset_end_minute": [0,0],
                                                    "ruleset_start_hour": [0,6],
                                                    "ruleset_end_hour": [6,18],
                                                    "ruleset_value": [20,24]},
                                                input = {},
                                                output = {},
                                                savedInput = {},
                                                savedOutput = {},
                                                createReport = False,
                                                connectedThrough = [],
                                                connectsAt = [])

        air_to_air_heat_recovery = distribution_flow_device.energy_conversion_device.air_to_air_heat_recovery.air_to_air_heat_recovery_model.AirToAirHeatRecoveryModel(primaryAirFlowRateMax = 1,
                                                                    secondaryAirFlowRateMax = 1,
                                                                    specificHeatCapacityAir = 1000,
                                                                    subSystemOf = [ventilation_system],
                                                                    input = {},
                                                                    output = {},
                                                                    savedInput = {},
                                                                    savedOutput = {},
                                                                    createReport = self.createReport,
                                                                    connectedThrough = [],
                                                                    connectsAt = [])
        ventilation_system.hasSubSystem.append(air_to_air_heat_recovery)

        heating_coil = distribution_flow_device.energy_conversion_device.coil.coil_model.CoilModel(isHeatingCoil = True,
                                        specificHeatCapacityAir = 1000,
                                        subSystemOf = [heating_system, ventilation_system],
                                        input = {"supplyAirTemperatureSetpoint": 23},
                                        output = {},
                                        savedInput = {},
                                        savedOutput = {},
                                        createReport = self.createReport,
                                        connectedThrough = [],
                                        connectsAt = [])
        heating_system.hasSubSystem.append(heating_coil)
        ventilation_system.hasSubSystem.append(heating_coil)

        cooling_coil = distribution_flow_device.energy_conversion_device.coil.coil_model.CoilModel(isCoolingCoil = True,
                                        specificHeatCapacityAir = 1000,
                                        subSystemOf = [cooling_system, ventilation_system],
                                        input = {"supplyAirTemperatureSetpoint": 23},
                                        output = {},
                                        savedInput = {},
                                        savedOutput = {},
                                        createReport = self.createReport,
                                        connectedThrough = [],
                                        connectsAt = [])
        heating_system.hasSubSystem.append(cooling_coil)
        ventilation_system.hasSubSystem.append(cooling_coil)

        supply_fan = distribution_flow_device.flow_moving_device.fan.fan_model.FanModel(isSupplyFan = True,
                                    nominalAirFlowRate = 250/3600*1.225,
                                    nominalPowerRate = 10000,
                                    subSystemOf = [ventilation_system],
                                    input = {},
                                    output = {},
                                    savedInput = {},
                                    savedOutput = {},
                                    createReport = self.createReport,
                                    connectedThrough = [],
                                    connectsAt = [])
        ventilation_system.hasSubSystem.append(supply_fan)

        return_fan = distribution_flow_device.flow_moving_device.fan.fan_model.FanModel(isReturnFan = True,
                                    nominalAirFlowRate = 250/3600*1.225,
                                    nominalPowerRate = 10000,
                                    subSystemOf = [ventilation_system],
                                    input = {},
                                    output = {},
                                    savedInput = {},
                                    savedOutput = {},
                                    createReport = self.createReport,
                                    connectedThrough = [],
                                    connectsAt = [])
        ventilation_system.hasSubSystem.append(return_fan)

        supply_flowmeter = distribution_flow_device.flow_controller.flow_meter.flow_meter_model.FlowMeterModel(isSupplyFlowMeter = True, 
                                                subSystemOf = [ventilation_system],
                                                input = {},
                                                output = {},
                                                connectedThrough = [],
                                    connectsAt = [])
        ventilation_system.hasSubSystem.append(supply_flowmeter)

        return_flowmeter = distribution_flow_device.flow_controller.flow_meter.flow_meter_model.FlowMeterModel(isReturnFlowMeter = True, 
                                                subSystemOf = [ventilation_system],
                                                input = {},
                                                output = {},
                                                savedInput = {},
                                                savedOutput = {},
                                                connectedThrough = [],
                                                connectsAt = [])
        ventilation_system.hasSubSystem.append(return_flowmeter)


        self.initComponents = []
        self.initComponents.append(temperature_setpoint_schedule)
        self.initComponents.append(_weather_station)
        self.initComponents.append(occupancy_schedule)
        for i in range(1):
            space_heater = distribution_flow_device.flow_terminal.space_heater.space_heater_model.SpaceHeaterModel(outputCapacity = 1000,
                                                    thermalMassHeatCapacity = 30000,
                                                    specificHeatCapacityWater = 4180,
                                                    timeStep = self.timeStep, 
                                                    subSystemOf = [heating_system],
                                                    input = {"supplyWaterTemperature": 60},
                                                    output = {"radiatorOutletTemperature": 22,
                                                                "Energy": 0},
                                                    savedInput = {},
                                                    savedOutput = {},
                                                    createReport = self.createReport,
                                                    connectedThrough = [],
                                                    connectsAt = [])
            heating_system.hasSubSystem.append(space_heater)

            valve = distribution_flow_device.flow_controller.valve.valve_model.ValveModel(waterFlowRateMax = 0.1,
                                        valveAuthority = 0.8, 
                                        subSystemOf = [heating_system],
                                        input = {},
                                        output = {},
                                        savedInput = {},
                                        savedOutput = {},
                                        createReport = self.createReport,
                                        connectedThrough = [],
                                        connectsAt = [])
            heating_system.hasSubSystem.append(valve)

            temperature_controller = distribution_control_device.controller.controller_model.ControllerModel(isTemperatureController = True,
                                                            k_p = 1,
                                                            k_i = 0.5,
                                                            k_d = 0,
                                                            subSystemOf = [heating_system],
                                                            input = {},
                                                            output = {"valveSignal": 0},
                                                            savedInput = {},
                                                            savedOutput = {},
                                                            createReport = self.createReport,
                                                            connectedThrough = [],
                                                            connectsAt = [])
            heating_system.hasSubSystem.append(temperature_controller)

            co2_controller = distribution_control_device.controller.controller_model.ControllerModel(isCo2Controller = True, 
                                                    k_p = 0.01, #0.01
                                                    k_i = 0,
                                                    k_d = 0,
                                                    subSystemOf = [ventilation_system],
                                                    input = {"indoorCo2ConcentrationSetpoint": 600},
                                                    output = {"supplyDamperSignal": 0,
                                                            "returnDamperSignal": 0},
                                                    savedInput = {},
                                                    savedOutput = {},
                                                    connectedThrough = [],
                                                    connectsAt = [])
            ventilation_system.hasSubSystem.append(co2_controller)

            supply_damper = distribution_flow_device.flow_controller.damper.damper_model.DamperModel(isSupplyDamper = True,
                                                nominalAirFlowRate = 250/3600*1.225,
                                                subSystemOf = [ventilation_system],
                                                input = {},
                                                output = {},
                                                savedInput = {},
                                                savedOutput = {},
                                                createReport = self.createReport,
                                                connectedThrough = [],
                                                connectsAt = [])
            ventilation_system.hasSubSystem.append(supply_damper)
            supply_damper.output[supply_damper.AirFlowRateName] = 0 ########################

            return_damper = distribution_flow_device.flow_controller.damper.damper_model.DamperModel(isReturnDamper = True,
                                                nominalAirFlowRate = 250/3600*1.225,
                                                subSystemOf = [ventilation_system],
                                                input = {},
                                                output = {},
                                                savedInput = {},
                                                savedOutput = {},
                                                connectedThrough = [],
                                                connectsAt = [])
            ventilation_system.hasSubSystem.append(return_damper)
            return_damper.output[return_damper.AirFlowRateName] = 0 ########################

            space = saref4bldg.building_space.building_space_model.BuildingSpaceModel(densityAir = 1.225,
                                                airVolume = 50,
                                                startPeriod = self.startPeriod,
                                                timeStep = self.timeStep,
                                                input = {"generationCo2Concentration": 0.06,
                                                        "outdoorCo2Concentration": 500,
                                                        "shadesSignal": 0},
                                                output = {"indoorTemperature": 21.5,
                                                        "indoorCo2Concentration": 500},
                                                savedInput = {},
                                                savedOutput = {},
                                                createReport = self.createReport,
                                                connectedThrough = [],
                                                connectsAt = [])



            
            self.add_connection(temperature_setpoint_schedule, temperature_controller, "scheduleValue", "indoorTemperatureSetpoint")
            self.add_connection(space, temperature_controller, "indoorTemperature", "indoorTemperature")
            self.add_connection(space, co2_controller, "indoorCo2Concentration", "indoorCo2Concentration")

            self.add_connection(supply_damper, supply_flowmeter, supply_damper.AirFlowRateName, supply_damper.AirFlowRateName)
            self.add_connection(return_damper, return_flowmeter, return_damper.AirFlowRateName, return_damper.AirFlowRateName)
        
            self.add_connection(space, space_heater, "indoorTemperature", "indoorTemperature")
            self.add_connection(valve, space_heater, "waterFlowRate", "waterFlowRate")

            self.add_connection(temperature_controller, valve, "valveSignal", "valveSignal")
            self.add_connection(temperature_controller, space, "valveSignal", "valveSignal")

            self.add_connection(co2_controller, space, "supplyDamperSignal", "supplyDamperSignal")
            self.add_connection(co2_controller, space, "returnDamperSignal", "returnDamperSignal")

            self.add_connection(supply_damper, space, supply_damper.AirFlowRateName, "supplyAirFlowRate")
            self.add_connection(return_damper, space, return_damper.AirFlowRateName, "returnAirFlowRate")
            self.add_connection(occupancy_schedule, space, "scheduleValue", "numberOfPeople")

            self.add_connection(_weather_station, space, "directRadiation", "directRadiation")
            self.add_connection(_weather_station, space, "diffuseRadiation", "diffuseRadiation")
            self.add_connection(_weather_station, space, "outdoorTemperature", "outdoorTemperature")


            self.add_connection(co2_controller, supply_damper, "supplyDamperSignal", "supplyDamperSignal")
            self.add_connection(co2_controller, return_damper, "returnDamperSignal", "returnDamperSignal")

            self.initComponents.append(space)

        self.add_connection(_weather_station, air_to_air_heat_recovery, "outdoorTemperature", "outdoorTemperature")

        self.add_connection(space, air_to_air_heat_recovery, "indoorTemperature", "indoorTemperature") #########################
        self.add_connection(supply_flowmeter, air_to_air_heat_recovery, "supplyAirFlowRate", "supplyAirFlowRate")
        self.add_connection(return_flowmeter, air_to_air_heat_recovery, "returnAirFlowRate", "returnAirFlowRate")
        
        self.add_connection(air_to_air_heat_recovery, heating_coil, "supplyAirTemperature", "supplyAirTemperature")
        self.add_connection(air_to_air_heat_recovery, cooling_coil, "supplyAirTemperature", "supplyAirTemperature")

        self.add_connection(supply_flowmeter, heating_coil, "supplyAirFlowRate", "supplyAirFlowRate")
        self.add_connection(supply_flowmeter, cooling_coil, "supplyAirFlowRate", "supplyAirFlowRate")

        self.add_connection(supply_flowmeter, supply_fan, "supplyAirFlowRate", "supplyAirFlowRate")
        self.add_connection(return_flowmeter, return_fan, "returnAirFlowRate", "returnAirFlowRate")

        
        self.activeComponents = self.initComponents

    def show_system_graph(self):
        fig = plt.figure()

        rect = [0,0,1,1]
        ax = fig.add_axes(rect)
        # fig.set_size_inches(40, 13) 
        figManager = plt.get_current_fig_manager() ################
        figManager.window.showMaximized() #######################

        
        min_fontsize = 14
        max_fontsize = 18

        min_width = 1.2
        max_width = 3

        degree_list = [self.system_graph.degree(node) for node in self.system_graph.nodes]
        min_deg = min(degree_list)
        max_deg = max(degree_list)

        a_fontsize = (max_fontsize-min_fontsize)/(max_deg-min_deg)
        b_fontsize = max_fontsize-a_fontsize*max_deg

        a_width = (max_width-min_width)/(max_deg-min_deg)
        b_width = max_width-a_width*max_deg
        for node in self.system_graph.nodes:
            deg = self.system_graph.degree(node)
            fontsize = a_fontsize*deg + b_fontsize
            width = a_width*deg + b_width
            

            if node not in self.system_graph_node_attribute_dict:
                self.system_graph_node_attribute_dict[node] = {"fontsize": fontsize, "width": width}
            else:
                self.system_graph_node_attribute_dict[node]["fontsize"] = fontsize
                self.system_graph_node_attribute_dict[node]["width"] = width


        #get all node names
        # name_list = []
        # for node in self.system_graph.nodes:
        #     name_list.append(node.name)
        # name_list = list(set(name_list))

        


        nx.set_node_attributes(self.system_graph, values=self.system_graph_node_attribute_dict)

        # print(self.system_graph.nodes)
        # print(self.system_graph.nodes[9]["labels"])

        #####################################################
        # pos = nx.kamada_kawai_layout(self.system_graph)   
        # nx.draw_networkx_nodes(self.system_graph, pos=pos)
        # nx.draw_networkx_labels(self.system_graph, pos, labels=self.system_graph_node_label_dict)

        # nx.draw_networkx_edge_labels(self.system_graph, pos)

        # for edge in self.system_graph.edges(data=True):
        #     nx.draw_networkx_edges(self.system_graph, pos, edgelist=[(edge[0],edge[1])], connectionstyle=f'arc3, rad = {edge[2]["rad"]}')
        ###################################

        graph = nx.drawing.nx_pydot.to_pydot(self.system_graph)

        

        # graph.set_graph_defaults(pack="true", rankdir="TB", bgcolor="transparent", fontname="Helvetica", fontcolor="blue", fontsize=10, dpi=500, splines="ortho")
        graph.set_node_defaults(shape="circle", width=0.8, fixedsize="shape", margin=0, style="filled", fontname="Helvetica", color="#23a6db66", fontsize=10, colorscheme="oranges9")
        graph.set_edge_defaults(fontname="Helvetica", penwidth=2, color="#999999", fontcolor="#999999", fontsize=10, weight=3, minlen=1)

        self.system_graph = nx.drawing.nx_pydot.from_pydot(graph)

        nx.drawing.nx_pydot.write_dot(self.system_graph, 'system_graph.dot')
        # graph = nx.drawing.nx_pydot.to_pydot(self.system_graph)
        cmd_string = "\"C:/Program Files/Graphviz/bin/dot.exe\" -Tpng -Kdot -Grankdir=LR -o system_graph.png system_graph.dot"
        os.system(cmd_string)

        
        # graph = nx.from_edgelist(self.system_graph_edge_label_dict, nx.DiGraph())
        # Graph(graph, node_layout='dot', node_shape = "o", node_size=6, ax=ax,
        #         edge_width=1, edge_label_rotate=True, edge_layout="curved", arrows=True,
        #         node_labels=self.system_graph_node_label_dict, node_label_fontdict=dict(size=14),
        #         edge_labels=self.system_graph_edge_label_dict, edge_label_fontdict=dict(size=8), 
        #     )

# dot ###
# random
# circular
# spring ###
# community
# bipartite
        # plt.show()


    def show_execution_graph(self):
        self.execution_graph = nx.MultiDiGraph() ###
        self.execution_graph_node_attribute_dict = {}


        n = len(self.component_order)
        for i in range(n-1):
            sender_component = self.component_order[i]
            reciever_component = self.component_order[i+1]
            self.execution_graph.add_edge(sender_component.systemId, reciever_component.systemId) 

            self.execution_graph_node_attribute_dict[sender_component.systemId] = {"label": sender_component.__class__.__name__}
            self.execution_graph_node_attribute_dict[reciever_component.systemId] = {"label": reciever_component.__class__.__name__}



        min_fontsize = 14
        max_fontsize = 18

        min_width = 1.2
        max_width = 3

        degree_list = [self.execution_graph.degree(node) for node in self.execution_graph.nodes]
        min_deg = min(degree_list)
        max_deg = max(degree_list)

        a_fontsize = (max_fontsize-min_fontsize)/(max_deg-min_deg)
        b_fontsize = max_fontsize-a_fontsize*max_deg

        a_width = (max_width-min_width)/(max_deg-min_deg)
        b_width = max_width-a_width*max_deg
        for node in self.execution_graph.nodes:
            deg = self.execution_graph.degree(node)
            fontsize = a_fontsize*deg + b_fontsize
            width = a_width*deg + b_width
            

            if node not in self.execution_graph_node_attribute_dict:
                self.execution_graph_node_attribute_dict[node] = {"fontsize": fontsize, "width": width}
            else:
                self.execution_graph_node_attribute_dict[node]["fontsize"] = fontsize
                self.execution_graph_node_attribute_dict[node]["width"] = width



        nx.set_node_attributes(self.execution_graph, values=self.execution_graph_node_attribute_dict)

        graph = nx.drawing.nx_pydot.to_pydot(self.execution_graph)
        graph.set_node_defaults(shape="circle", width=0.8, fixedsize="shape", margin=0, style="filled", fontname="Helvetica", color="#23a6db66", fontsize=10, colorscheme="oranges9")
        graph.set_edge_defaults(fontname="Helvetica", penwidth=2, color="#999999", fontcolor="#999999", fontsize=10, weight=3, minlen=1)

        self.execution_graph = nx.drawing.nx_pydot.from_pydot(graph)

        nx.drawing.nx_pydot.write_dot(self.execution_graph, 'execution_graph.dot')
        cmd_string = "\"C:/Program Files/Graphviz/bin/dot.exe\" -Tpng -Kdot -Grankdir=LR -o execution_graph.png execution_graph.dot"
        os.system(cmd_string)

        

    def simulate(self):
        

        time = self.startPeriod
        time_list = []
        while time < self.endPeriod:
            for component in self.component_order:
                # print("----")
                # print(component.__class__.__name__)
                #Gather all needed inputs for the component through all ingoing connections
                for connection_point in component.connectsAt:
                    connection = connection_point.connectsSystemThrough
                    connected_component = connection.connectsSystem
                    # print("--h--")
                    # print(component.__class__.__name__)
                    # print(connected_component.__class__.__name__)
                    # print(connection.senderPropertyName)
                    # print(connection_point.recieverPropertyName)
                    # print(connected_component.output)
                    component.input[connection_point.recieverPropertyName] = connected_component.output[connection.senderPropertyName]

                component.update_output()
                component.update_report()

            time_list.append(time)
            time += datetime.timedelta(seconds=self.timeStep)
            


        print("-------")
        for component in self.component_order:
            if component.createReport:
                component.plot_report(time_list)
        plt.show()


        
    def find_path(self):
        self.visitCount = {}
        self.component_order = []
        self.component_order.extend(self.initComponents)
        while len(self.activeComponents)>0:
            print("YYYYYYYYYYYY")
            self.traverse()


    def traverse(self):
        activeComponentsNew = []
        for component in self.activeComponents:
            for connection in component.connectedThrough:
                connection_point = connection.connectsSystemAt
                connected_component = connection_point.connectionPointOf
                if connected_component.connectionVisits is None:
                    connected_component.connectionVisits = [connection_point.recieverPropertyName]
                else:
                    connected_component.connectionVisits.append(connection_point.recieverPropertyName)

                has_connections = True
                for ingoing_connection_point in connected_component.connectsAt:
                    ingoing_connection = ingoing_connection_point.connectsSystemThrough
                    if ingoing_connection_point.recieverPropertyName not in connected_component.connectionVisits or isinstance(connected_component, saref4bldg.building_space.building_space.BuildingSpace):
                        has_connections = False
                        break
                
                if has_connections:
                    self.component_order.append(connected_component)

                    if connected_component.connectedThrough is not None:
                        activeComponentsNew.append(connected_component)

                # print("---")
                # print(component.__class__.__name__)
                # print(connected_component.__class__.__name__)
                # print(connection.connectionType)

                # print(connected_component.connectionVisits)

        

        self.activeComponents = activeComponentsNew



    def get_leaf_subsystems(self, system):
        for sub_system in system.hasSubSystem:
            if sub_system.hasSubSystem is None:
                self.leaf_subsystems.append(sub_system)
            else:
                self.get_leaf_subsystems(sub_system)



createReport = True
timeStep = 600
startPeriod = datetime.datetime(year=2019, month=12, day=8, hour=0, minute=0, second=0, tzinfo=tzutc())
endPeriod = datetime.datetime(year=2019, month=12, day=14, hour=0, minute=0, second=0, tzinfo=tzutc())
model = BuildingEnergyModel(timeStep = timeStep,
                            startPeriod = startPeriod,
                            endPeriod = endPeriod,
                            createReport = createReport)
model.load_model()
model.find_path()
model.show_execution_graph()
model.show_system_graph()
model.simulate()



