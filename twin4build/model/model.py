import networkx as nx
import pandas as pd
import warnings
import shutil
import subprocess
import sys
import os
import copy
import pydot
# import seaborn







from twin4build.saref4syst.connection import Connection 
from twin4build.saref4syst.connection_point import ConnectionPoint
from twin4build.saref4syst.system import System
from twin4build.utils.uppath import uppath
from twin4build.utils.outdoor_environment import OutdoorEnvironment
from twin4build.utils.schedule import Schedule
from twin4build.utils.node import Node
from twin4build.saref.measurement.measurement import Measurement

from twin4build.saref.property_.temperature.temperature import Temperature


from twin4build.saref.function.sensing_function.sensing_function import SensingFunction
from twin4build.saref.property_.temperature.temperature import Temperature
from twin4build.saref.property_.co2.co2 import Co2


from twin4build.saref4bldg.physical_object.building_object.building_device.distribution_device.distribution_device import DistributionDevice



from twin4build.saref4bldg.building_space.building_space import BuildingSpace
from twin4build.saref4bldg.physical_object.building_object.building_device.distribution_device.distribution_flow_device.energy_conversion_device.coil.coil import Coil
from twin4build.saref4bldg.physical_object.building_object.building_device.distribution_device.distribution_control_device.controller.controller import Controller
from twin4build.saref4bldg.physical_object.building_object.building_device.distribution_device.distribution_flow_device.energy_conversion_device.air_to_air_heat_recovery.air_to_air_heat_recovery import AirToAirHeatRecovery
from twin4build.saref4bldg.physical_object.building_object.building_device.distribution_device.distribution_flow_device.flow_controller.damper.damper import Damper
from twin4build.saref4bldg.physical_object.building_object.building_device.distribution_device.distribution_flow_device.flow_controller.valve.valve import Valve
from twin4build.saref4bldg.physical_object.building_object.building_device.distribution_device.distribution_flow_device.flow_moving_device.fan.fan import Fan
from twin4build.saref4bldg.physical_object.building_object.building_device.distribution_device.distribution_flow_device.flow_terminal.space_heater.space_heater import SpaceHeater
from twin4build.saref.device.sensor.sensor import Sensor

from twin4build.saref4bldg.building_space.building_space_model_co2 import BuildingSpaceModelCo2
from twin4build.saref4bldg.building_space.building_space_model import BuildingSpaceModel, NoSpaceModelException
from twin4build.saref4bldg.physical_object.building_object.building_device.distribution_device.distribution_flow_device.energy_conversion_device.coil.coil_heating_model import CoilHeatingModel
from twin4build.saref4bldg.physical_object.building_object.building_device.distribution_device.distribution_flow_device.energy_conversion_device.coil.coil_cooling_model import CoilCoolingModel
from twin4build.saref4bldg.physical_object.building_object.building_device.distribution_device.distribution_control_device.controller.controller_model import ControllerModel
from twin4build.saref4bldg.physical_object.building_object.building_device.distribution_device.distribution_flow_device.energy_conversion_device.air_to_air_heat_recovery.air_to_air_heat_recovery_model import AirToAirHeatRecoveryModel
from twin4build.saref4bldg.physical_object.building_object.building_device.distribution_device.distribution_flow_device.flow_controller.damper.damper_model import DamperModel
from twin4build.saref4bldg.physical_object.building_object.building_device.distribution_device.distribution_flow_device.flow_controller.valve.valve_model import ValveModel
from twin4build.saref4bldg.physical_object.building_object.building_device.distribution_device.distribution_flow_device.flow_moving_device.fan.fan_model import FanModel
from twin4build.saref4bldg.physical_object.building_object.building_device.distribution_device.distribution_flow_device.flow_terminal.space_heater.space_heater_model import SpaceHeaterModel
from twin4build.saref.device.sensor.sensor_model import SensorModel 
from twin4build.saref4bldg.physical_object.building_object.building_device.shading_device.shading_device_model import ShadingDeviceModel

# Sensor(
#     hasFunction=SensingFunction(
#         hasSensingRange=[],
#         hasSensorType=Temperature
#     ))

def str2Class(str):
    return getattr(sys.modules[__name__], str)


class Model:
    def __init__(self,
                timeStep = None,
                startPeriod = None,
                endPeriod = None,
                createReport = False):
        self.timeStep = timeStep
        self.startPeriod = startPeriod
        self.endPeriod = endPeriod
        self.createReport = createReport
        self.system_graph = pydot.Dot()#nx.MultiDiGraph() ###
        rank = None#"same" #Set to "same" to put all nodes with same class on same rank
        self.subgraph_dict = {
            OutdoorEnvironment.__name__: pydot.Subgraph(rank=rank),
            Schedule.__name__: pydot.Subgraph(rank=rank),
            BuildingSpaceModel.__name__: pydot.Subgraph(rank=rank),
            BuildingSpaceModelCo2.__name__: pydot.Subgraph(rank=rank),
            ControllerModel.__name__: pydot.Subgraph(rank=rank),
            AirToAirHeatRecoveryModel.__name__: pydot.Subgraph(),
            CoilHeatingModel.__name__: pydot.Subgraph(rank=rank),
            CoilCoolingModel.__name__: pydot.Subgraph(rank=rank),
            DamperModel.__name__: pydot.Subgraph(rank=rank),
            ValveModel.__name__: pydot.Subgraph(rank=rank),
            FanModel.__name__: pydot.Subgraph(rank=rank),
            SpaceHeaterModel.__name__: pydot.Subgraph(rank=rank),
            Node.__name__: pydot.Subgraph(rank=rank),
            ShadingDeviceModel.__name__: pydot.Subgraph(rank=rank),
            SensorModel.__name__: pydot.Subgraph(rank=rank),
            }


        # for subgraph in self.subgraph_dict.values():
        #     dummy_node = pydot.Node("dummy")
        #     dummy_node.obj_dict["attributes"].update({
        #         "shape": "point",
        #         "style": "invis",
        #     })
        #     subgraph.add_node(dummy_node)

        self.system_graph_node_attribute_dict = {}
        self.system_graph_edge_label_dict = {}
        for subgraph in self.subgraph_dict.values():
            self.system_graph.add_subgraph(subgraph)

        self.initComponents = []
        self.activeComponents = None
        self.system_dict = {"ventilation": {},
                            "heating": {},
                            "cooling": {},
                            }
        self.component_base_dict = {}
        self.component_dict = {}
        

    def add_edge_(self, graph, a, b, label):
        graph.add_edge(pydot.Edge(a, b, label=label))

    def del_edge_(self, graph, a, b):
        if " " in a or "Ø" in a:
            a = "\"" + a + "\""
        else:
            a = a

        if " " in b or "Ø" in a:
            b = "\"" + b + "\""
        else:
            b = b

        graph.del_edge(a, b)

    def add_component(self, component):
        self.component_dict[component.id] = component


    def add_connection(self, sender_component, reciever_component, sender_property_name, reciever_property_name):
        sender_obj_connection = Connection(connectsSystem = sender_component, sender_property_name = sender_property_name)
        sender_component.connectedThrough.append(sender_obj_connection)
        reciever_component_connection_point = ConnectionPoint(connectionPointOf=reciever_component, connectsSystemThrough=sender_obj_connection, recieverPropertyName=reciever_property_name)
        sender_obj_connection.connectsSystemAt = reciever_component_connection_point
        reciever_component.connectsAt.append(reciever_component_connection_point)

        end_space = "          "
        edge_label = ("C: " + sender_property_name.split("_")[0] + end_space + "\n"
                        "CP: " + reciever_property_name.split("_")[0] + end_space)


        self.add_edge_(self.system_graph, sender_component.id, reciever_component.id, label=edge_label) ###

        cond1 = not self.subgraph_dict[type(sender_component).__name__].get_node(sender_component.id)
        cond2 = not self.subgraph_dict[type(sender_component).__name__].get_node("\""+ sender_component.id +"\"")

        if cond1 and cond2:
            # print("added sender " + sender_obj.id)
            node = pydot.Node(sender_component.id)
            self.subgraph_dict[type(sender_component).__name__].add_node(node)



        cond1 = not self.subgraph_dict[type(reciever_component).__name__].get_node(reciever_component.id)
        cond2 = not self.subgraph_dict[type(reciever_component).__name__].get_node("\""+ reciever_component.id +"\"")

        if cond1 and cond2:
            # print("added reciever " + reciever_component.id)
            node = pydot.Node(reciever_component.id)
            self.subgraph_dict[type(reciever_component).__name__].add_node(node)
        


        # self.system_graph_node_attribute_dict[sender_obj.id] = {"label": sender_obj.__class__.__name__.replace("Model","")}
        # self.system_graph_node_attribute_dict[reciever_component.id] = {"label": reciever_component.__class__.__name__.replace("Model","")}

        self.system_graph_node_attribute_dict[sender_component.id] = {"label": sender_component.id}
        self.system_graph_node_attribute_dict[reciever_component.id] = {"label": reciever_component.id}

    
    def add_outdoor_environment(self):
        outdoor_environment = OutdoorEnvironment(
            startPeriod = self.startPeriod,
            endPeriod = self.endPeriod,
            input = {},
            output = {},
            savedInput = {},
            savedOutput = {},
            createReport = self.createReport,
            connectedThrough = [],
            connectsAt = [],
            id = "Outdoor environment")
        self.add_component(outdoor_environment)

    def add_occupancy_schedule(self):
        occupancy_schedule = Schedule(
            startPeriod = self.startPeriod,
            timeStep = self.timeStep,
            rulesetDict = {
                "ruleset_default_value": 0,
                "ruleset_start_minute": [0,0,0,0,0,0,0],
                "ruleset_end_minute": [0,0,0,0,0,0,0],
                "ruleset_start_hour": [6,7,8,12,14,16,18],
                "ruleset_end_hour": [7,8,12,14,16,18,22],
                "ruleset_value": [3,5,20,25,27,7,3]}, #35
                # "ruleset_value": [0,0,0,0,0,0,0]}, #35
                # rulesetDict = {
                # "ruleset_default_value": 0,
                # "ruleset_start_minute": [],
                # "ruleset_end_minute": [],
                # "ruleset_start_hour": [],
                # "ruleset_end_hour": [],
                # "ruleset_value": []}, #35
                # "ruleset_value": [0,0,0,0,0,0]}, #35
                # # "ruleset_value": [0,0,0,0,0,0]}, #35
            add_noise = True,
            input = {},
            output = {},
            savedInput = {},
            savedOutput = {},
            createReport = self.createReport,
            connectedThrough = [],
            connectsAt = [],
            id = "Occupancy schedule")
        self.add_component(occupancy_schedule)

    def add_indoor_temperature_setpoint_schedule(self):
        indoor_temperature_setpoint_schedule = Schedule(
            startPeriod = self.startPeriod,
            timeStep = self.timeStep,
            rulesetDict = {
                "ruleset_default_value": 23,
                "ruleset_start_minute": [0,0],
                "ruleset_end_minute": [0,0],
                "ruleset_start_hour": [0,4],
                "ruleset_end_hour": [4,18],
                "ruleset_value": [23,23]},
            input = {},
            output = {},
            savedInput = {},
            savedOutput = {},
            createReport = self.createReport,
            connectedThrough = [],
            connectsAt = [],
            id = "Temperature setpoint schedule")
        self.add_component(indoor_temperature_setpoint_schedule)

    def add_co2_setpoint_schedule(self):
        co2_setpoint_schedule = Schedule(
            startPeriod = self.startPeriod,
            timeStep = self.timeStep,
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
            createReport = self.createReport,
            connectedThrough = [],
            connectsAt = [],
            id = "CO2 setpoint schedule")
        self.add_component(co2_setpoint_schedule)

    def add_supply_air_temperature_setpoint_schedule(self):
        supply_air_temperature_setpoint_schedule = Schedule(
            startPeriod = self.startPeriod,
            timeStep = self.timeStep,
            rulesetDict = {
                "ruleset_default_value": 23,
                "ruleset_start_minute": [],
                "ruleset_end_minute": [],
                "ruleset_start_hour": [],
                "ruleset_end_hour": [],
                "ruleset_value": []},
            input = {},
            output = {},
            savedInput = {},
            savedOutput = {},
            createReport = self.createReport,
            connectedThrough = [],
            connectsAt = [],
            id = "Supply temperature setpoint schedule")
        self.add_component(supply_air_temperature_setpoint_schedule)

    def add_shade_setpoint_schedule(self):
        shade_setpoint_schedule = Schedule(
            startPeriod = self.startPeriod,
            timeStep = self.timeStep,
            rulesetDict = {
                "ruleset_default_value": 0,
                "ruleset_start_minute": [30],
                "ruleset_end_minute": [0],
                "ruleset_start_hour": [11],
                "ruleset_end_hour": [18],
                "ruleset_value": [0]},
            input = {},
            output = {},
            savedInput = {},
            savedOutput = {},
            createReport = self.createReport,
            connectedThrough = [],
            connectsAt = [],
            id = "Shade setpoint schedule")
        self.add_component(shade_setpoint_schedule)

    def add_shading_device(self):
        shade_setpoint_schedule = ShadingDeviceModel(
            isExternal = True,
            input = {},
            output = {},
            savedInput = {},
            savedOutput = {},
            createReport = self.createReport,
            connectedThrough = [],
            connectsAt = [],
            id = "Shade")
        self.add_component(shade_setpoint_schedule)


    def read_config(self):
        """
        Reads configuration file and instantiates a base SAREF4BLDG object for each entry in the file.  
        It assumes that the sheets of the configuration file follows a certain naming pattern.
        
        xlsx:
            Sheets:
                Systems
                Space
                Damper
                SpaceHeater
                Valve
                Coil
                AirToAirHeatRecovery
                Fan
                Controller
        """

        file_name = "configuration_template_1space_1v_1h_0c_with_simple_naming.xlsx"
        file_path = os.path.join(uppath(os.path.abspath(__file__), 2), "test", "data", file_name)

        df_Systems = pd.read_excel(file_path, sheet_name="Systems")
        df_Space = pd.read_excel(file_path, sheet_name="BuildingSpace")
        df_Damper = pd.read_excel(file_path, sheet_name="Damper")
        df_SpaceHeater = pd.read_excel(file_path, sheet_name="SpaceHeater")
        df_Valve = pd.read_excel(file_path, sheet_name="Valve")
        df_Coil = pd.read_excel(file_path, sheet_name="Coil")
        df_AirToAirHeatRecovery = pd.read_excel(file_path, sheet_name="AirToAirHeatRecovery")
        df_Fan = pd.read_excel(file_path, sheet_name="Fan")
        df_Controller = pd.read_excel(file_path, sheet_name="Controller")
        df_Sensor = pd.read_excel(file_path, sheet_name="Sensor")

        for ventilation_system_name in df_Systems["Ventilation system name"].dropna():
            ventilation_system = DistributionDevice(subSystemOf = [], hasSubSystem = [], id = ventilation_system_name)
            self.system_dict["ventilation"][ventilation_system_name] = ventilation_system
        
        for heating_system_name in df_Systems["Heating system name"].dropna():
            heating_system = DistributionDevice(subSystemOf = [], hasSubSystem = [], id = heating_system_name)
            self.system_dict["heating"][heating_system_name] = heating_system

        for cooling_system_name in df_Systems["Cooling system name"].dropna():
            cooling_system = DistributionDevice(subSystemOf = [], hasSubSystem = [], id = cooling_system_name)
            self.system_dict["cooling"][cooling_system_name] = cooling_system

        for row in df_Space.dropna(subset=["id"]).itertuples(index=False):
            space_name = row[df_Space.columns.get_loc("id")]
            try: 
                space = BuildingSpace(
                    airVolume = row[df_Space.columns.get_loc("airVolume")],
                    contains = [],
                    hasProperty = [Temperature(), 
                                    Co2()],
                    connectedThrough = [],
                    connectsAt = [],
                    id = space_name)
                for property_ in space.hasProperty:
                    property_.isPropertyOf = space
                self.component_base_dict[space_name] = space
            except NoSpaceModelException:
                print("No fitting space model for space " + "\"" + space_name + "\"")
                print("Continuing...")
            

        for row in df_Damper.dropna(subset=["id"]).itertuples(index=False):
            damper_name = row[df_Damper.columns.get_loc("id")]
            #Check that an appropriate space object exists
            if row[df_Damper.columns.get_loc("isContainedIn")] not in self.component_base_dict:
                warnings.warn("Cannot find a matching mathing BuildingSpace object for damper \"" + damper_name + "\"")
            else:
                systems = row[df_Damper.columns.get_loc("subSystemOf")].split(";")
                systems = [system for system_dict in self.system_dict.values() for system in system_dict.values() if system.id in systems]
                damper = Damper(
                    subSystemOf = systems,
                    isContainedIn = self.component_base_dict[row[df_Damper.columns.get_loc("isContainedIn")]],
                    operationMode = row[df_Damper.columns.get_loc("operationMode")],
                    nominalAirFlowRate = Measurement(hasValue=row[df_Damper.columns.get_loc("nominalAirFlowRate")]),
                    connectedThrough = [],
                    connectsAt = [],
                    id = damper_name)
                self.component_base_dict[damper_name] = damper

        for row in df_SpaceHeater.dropna(subset=["id"]).itertuples(index=False):
            space_heater_name = row[df_SpaceHeater.columns.get_loc("id")]
            #Check that an appropriate space object exists
            if row[df_SpaceHeater.columns.get_loc("isContainedIn")] not in self.component_base_dict:
                warnings.warn("Cannot find a matching mathing BuildingSpace object for space heater \"" + space_heater_name + "\"")
            else:
                systems = row[df_SpaceHeater.columns.get_loc("subSystemOf")].split(";")
                systems = [system for system_dict in self.system_dict.values() for system in system_dict.values() if system.id in systems]
                space_heater = SpaceHeater(
                    subSystemOf = systems,
                    isContainedIn = self.component_base_dict[row[df_SpaceHeater.columns.get_loc("isContainedIn")]],
                    outputCapacity = Measurement(hasValue=row[df_SpaceHeater.columns.get_loc("outputCapacity")]),
                    temperatureClassification = row[df_SpaceHeater.columns.get_loc("temperatureClassification")],
                    thermalMassHeatCapacity = Measurement(hasValue=row[df_SpaceHeater.columns.get_loc("thermalMassHeatCapacity")]),
                    connectedThrough = [],
                    connectsAt = [],
                    id = space_heater_name)
                self.component_base_dict[space_heater_name] = space_heater

        for row in df_Valve.dropna(subset=["id"]).itertuples(index=False):
            valve_name = row[df_Valve.columns.get_loc("id")]
            #Check that an appropriate space object exists
            if row[df_Valve.columns.get_loc("isContainedIn")] not in self.component_base_dict:
                warnings.warn("Cannot find a matching mathing BuildingSpace object for valve \"" + valve_name + "\"")
            else:
                systems = row[df_Valve.columns.get_loc("subSystemOf")].split(";")
                systems = [system for system_dict in self.system_dict.values() for system in system_dict.values() if system.id in systems]
                valve = Valve(
                    subSystemOf = systems,
                    isContainedIn = self.component_base_dict[row[df_Valve.columns.get_loc("isContainedIn")]],
                    flowCoefficient = Measurement(hasValue=row[df_Valve.columns.get_loc("flowCoefficient")]),
                    testPressure = Measurement(hasValue=row[df_Valve.columns.get_loc("testPressure")]),
                    connectedThrough = [],
                    connectsAt = [],
                    id = valve_name)
                self.component_base_dict[valve_name] = valve

        for row in df_Coil.dropna(subset=["id"]).itertuples(index=False):
            coil_name = row[df_Coil.columns.get_loc("id")]
            systems = row[df_Coil.columns.get_loc("subSystemOf")].split(";")
            systems = [system for system_dict in self.system_dict.values() for system in system_dict.values() if system.id in systems]
            coil = Coil(
                subSystemOf = systems,
                operationMode = row[df_Coil.columns.get_loc("operationMode")],
                connectedThrough = [],
                connectsAt = [],
                id = coil_name)
            self.component_base_dict[coil_name] = coil
            
        for row in df_AirToAirHeatRecovery.dropna(subset=["id"]).itertuples(index=False):
            air_to_air_heat_recovery_name = row[df_AirToAirHeatRecovery.columns.get_loc("id")]
            systems = row[df_AirToAirHeatRecovery.columns.get_loc("subSystemOf")].split(";")
            systems = [system for system_dict in self.system_dict.values() for system in system_dict.values() if system.id in systems]
            air_to_air_heat_recovery = air_to_air_heat_recovery = AirToAirHeatRecovery(
                subSystemOf = systems,
                primaryAirFlowRateMax = Measurement(hasValue=row[df_AirToAirHeatRecovery.columns.get_loc("primaryAirFlowRateMax")]),
                secondaryAirFlowRateMax = Measurement(hasValue=row[df_AirToAirHeatRecovery.columns.get_loc("secondaryAirFlowRateMax")]),
                connectedThrough = [],
                connectsAt = [],
                id = air_to_air_heat_recovery_name)
            self.component_base_dict[air_to_air_heat_recovery_name] = air_to_air_heat_recovery


        for row in df_Fan.dropna(subset=["id"]).itertuples(index=False):
            fan_name = row[df_Fan.columns.get_loc("id")]
            systems = row[df_Fan.columns.get_loc("subSystemOf")].split(";")
            systems = [system for system_dict in self.system_dict.values() for system in system_dict.values() if system.id in systems]
            fan = Fan(
                subSystemOf = systems,
                operationMode = row[df_Fan.columns.get_loc("operationMode")],
                nominalAirFlowRate = Measurement(hasValue=row[df_Fan.columns.get_loc("nominalAirFlowRate")]),
                nominalPowerRate = Measurement(hasValue=row[df_Fan.columns.get_loc("nominalPowerRate")]),
                connectedThrough = [],
                connectsAt = [],
                id = fan_name)
            self.component_base_dict[fan_name] = fan

 
        for row in df_Controller.dropna(subset=["id"]).itertuples(index=False):
            controller_name = row[df_Controller.columns.get_loc("id")]
            if row[df_Controller.columns.get_loc("isContainedIn")] not in self.component_base_dict:
                warnings.warn("Cannot find a matching mathing BuildingSpace object for controller \"" + controller_name + "\"")
            else:
                systems = row[df_Controller.columns.get_loc("subSystemOf")].split(";")
                systems = [system for system_dict in self.system_dict.values() for system in system_dict.values() if system.id in systems]
                device = self.component_base_dict[row[df_Controller.columns.get_loc("isContainedIn")]] #Must be made more general to cover more than spaces
                Property = getattr(sys.modules[__name__], row[df_Controller.columns.get_loc("controlsProperty")])
                property_ = [property_ for property_ in device.hasProperty if isinstance(property_, Property)][0]
                controller = Controller(
                    subSystemOf = systems,
                    isContainedIn = self.component_base_dict[row[df_Controller.columns.get_loc("isContainedIn")]],
                    controlsProperty = property_,
                    connectedThrough = [],
                    connectsAt = [],
                    id = controller_name)
                property_.isControlledByDevice = controller
                self.component_base_dict[controller_name] = controller

        for row in df_Sensor.dropna(subset=["id"]).itertuples(index=False):
            sensor_name = row[df_Sensor.columns.get_loc("id")]
            if row[df_Sensor.columns.get_loc("isContainedIn")] not in self.component_base_dict:
                warnings.warn("Cannot find a matching mathing BuildingSpace object for sensor \"" + sensor_name + "\"")
            else:
                device = self.component_base_dict[row[df_Sensor.columns.get_loc("isContainedIn")]]
                Property = getattr(sys.modules[__name__], row[df_Sensor.columns.get_loc("measuresProperty")])
                property_ = [property_ for property_ in device.hasProperty if isinstance(property_, Property)][0]
                sensor = Sensor(
                    subSystemOf = [],
                    isContainedIn = self.component_base_dict[row[df_Sensor.columns.get_loc("isContainedIn")]],
                    measuresProperty = property_,
                    connectedThrough = [],
                    connectsAt = [],
                    id = sensor_name)
                property_.isMeasuredByDevice = sensor
                self.component_base_dict[sensor_name] = sensor

    def get_object_properties(self, object_):
        return {key: value for (key, value) in vars(object_).items()}

    def apply_model_extensions(self):

        space_instances = self.get_component_by_class(self.component_base_dict, BuildingSpace)
        damper_instances = self.get_component_by_class(self.component_base_dict, Damper)
        space_heater_instances = self.get_component_by_class(self.component_base_dict, SpaceHeater)
        valve_instances = self.get_component_by_class(self.component_base_dict, Valve)
        coil_instances = self.get_component_by_class(self.component_base_dict, Coil)
        air_to_air_heat_recovery_instances = self.get_component_by_class(self.component_base_dict, AirToAirHeatRecovery)
        fan_instances = self.get_component_by_class(self.component_base_dict, Fan)
        controller_instances = self.get_component_by_class(self.component_base_dict, Controller)
        sensor_instances = self.get_component_by_class(self.component_base_dict, Sensor)

        for space in space_instances:
            base_kwargs = self.get_object_properties(space)
            extension_kwargs = {
                "densityAir": 1.225,
                "startPeriod": self.startPeriod,
                "timeStep": self.timeStep,
                "input": {"generationCo2Concentration": 0.000008316,
                        "outdoorCo2Concentration": 400,
                        "infiltration": 0.07},
                "output": {"indoorTemperature": 21,
                        "indoorCo2Concentration": 500},
                "savedInput": {},
                "savedOutput": {},
                "createReport": self.createReport,
            }
            base_kwargs.update(extension_kwargs)
            space = BuildingSpaceModel(**base_kwargs)
            for property_ in space.hasProperty:
                property_.isPropertyOf = space
            self.add_component(space)

        for damper in damper_instances:
            base_kwargs = self.get_object_properties(damper)
            extension_kwargs = {
                "a": 5,
                "input": {},
                "output": {"airFlowRate": 0},
                "savedInput": {},
                "savedOutput": {},
                "createReport": self.createReport,
            }
            base_kwargs.update(extension_kwargs)
            damper = DamperModel(**base_kwargs)
            self.add_component(damper)
            damper.isContainedIn = self.component_dict[damper.isContainedIn.id]
            damper.isContainedIn.contains.append(damper)
            for system in damper.subSystemOf:
                system.hasSubSystem.append(damper)

        for space_heater in space_heater_instances:
            base_kwargs = self.get_object_properties(space_heater)
            extension_kwargs = {
                "specificHeatCapacityWater": Measurement(hasValue=4180),
                "timeStep": self.timeStep,
                "input": {"supplyWaterTemperature": 60},
                "output": {"radiatorOutletTemperature": 22,
                            "Energy": 0},
                "savedInput": {},
                "savedOutput": {},
                "createReport": self.createReport,
            }
            base_kwargs.update(extension_kwargs)
            space_heater = SpaceHeaterModel(**base_kwargs)
            self.add_component(space_heater)
            space_heater.isContainedIn = self.component_dict[space_heater.isContainedIn.id]
            space_heater.isContainedIn.contains.append(space_heater)
            for system in space_heater.subSystemOf:
                system.hasSubSystem.append(space_heater)

        for valve in valve_instances:
            base_kwargs = self.get_object_properties(valve)
            extension_kwargs = {
                "valveAuthority": Measurement(hasValue=0.8),
                "input": {},
                "output": {},
                "savedInput": {},
                "savedOutput": {},
                "createReport": self.createReport,
            }
            base_kwargs.update(extension_kwargs)
            valve = ValveModel(**base_kwargs)
            self.add_component(valve)
            valve.isContainedIn = self.component_dict[valve.isContainedIn.id]
            valve.isContainedIn.contains.append(valve)
            for system in valve.subSystemOf:
                system.hasSubSystem.append(valve)

        for coil in coil_instances:
            base_kwargs = self.get_object_properties(coil)
            extension_kwargs = {
                "specificHeatCapacityAir": Measurement(hasValue=1000),
                "input": {"supplyAirTemperatureSetpoint": 23},
                "output": {},
                "savedInput": {},
                "savedOutput": {},
                "createReport": self.createReport,
            }
            base_kwargs.update(extension_kwargs)
            if coil.operationMode=="heating":
                coil = CoilHeatingModel(**base_kwargs)
            elif coil.operationMode=="cooling":
                coil = CoilCoolingModel(**base_kwargs)
            self.add_component(coil)
            for system in coil.subSystemOf:
                system.hasSubSystem.append(coil)



        for air_to_air_heat_recovery in air_to_air_heat_recovery_instances:
            base_kwargs = self.get_object_properties(air_to_air_heat_recovery)
            extension_kwargs = {
                "specificHeatCapacityAir": Measurement(hasValue=1000),
                "eps_75_h": Measurement(hasValue=0.8),
                "eps_75_c": Measurement(hasValue=0.8),
                "eps_100_h": Measurement(hasValue=0.8),
                "eps_100_c": Measurement(hasValue=0.8),
                "input": {},
                "output": {},
                "savedInput": {},
                "savedOutput": {},
                "createReport": self.createReport,
            }
            base_kwargs.update(extension_kwargs)
            air_to_air_heat_recovery = AirToAirHeatRecoveryModel(**base_kwargs)
            self.add_component(air_to_air_heat_recovery)
            for system in air_to_air_heat_recovery.subSystemOf:
                system.hasSubSystem.append(air_to_air_heat_recovery)

            

        for fan in fan_instances:
            base_kwargs = self.get_object_properties(fan)
            extension_kwargs = {
                "c1": Measurement(hasValue=0.027828),
                "c2": Measurement(hasValue=0.026583),
                "c3": Measurement(hasValue=-0.087069),
                "c4": Measurement(hasValue=1.030920),
                "timeStep": self.timeStep,
                "input": {},
                "output": {"Energy": 0},
                "savedInput": {},
                "savedOutput": {},
                "createReport": self.createReport,
            }
            base_kwargs.update(extension_kwargs)
            fan = FanModel(**base_kwargs)
            self.add_component(fan)
            for system in fan.subSystemOf:
                system.hasSubSystem.append(fan)

        for controller in controller_instances:
            base_kwargs = self.get_object_properties(controller)
            if isinstance(controller.controlsProperty, Temperature):
                K_p = 0.05
                K_i = 0.8
                K_d = 0
            elif isinstance(controller.controlsProperty, Co2):
                K_p = -0.001
                K_i = -0.001
                K_d = 0
            extension_kwargs = {
                "K_p": K_p,
                "K_i": K_i,
                "K_d": K_d,
                "timeStep": self.timeStep,
                "input": {},
                "output": {"inputSignal": 0},
                "savedInput": {},
                "savedOutput": {},
                "createReport": self.createReport,
            }
            base_kwargs.update(extension_kwargs)
            controller = ControllerModel(**base_kwargs)
            self.add_component(controller)
            controller.isContainedIn = self.component_dict[controller.isContainedIn.id]
            controller.isContainedIn.contains.append(controller)
            controller.controlsProperty.isControlledByDevice = self.component_dict[controller.id]
            for system in controller.subSystemOf:
                system.hasSubSystem.append(controller)

        for sensor in sensor_instances:
            base_kwargs = self.get_object_properties(sensor)
            extension_kwargs = {
                "input": {},
                "output": {},
                "savedInput": {},
                "savedOutput": {},
                "createReport": self.createReport,
            }
            base_kwargs.update(extension_kwargs)
            sensor = SensorModel(**base_kwargs)
            self.add_component(sensor)
            sensor.isContainedIn = self.component_dict[sensor.isContainedIn.id]
            sensor.isContainedIn.contains.append(sensor)
            sensor.measuresProperty.isMeasuredByDevice = self.component_dict[sensor.id]
            for system in sensor.subSystemOf:
                system.hasSubSystem.append(sensor)


        # Add supply and exhaust node for each ventilation system
        for ventilation_system in self.system_dict["ventilation"].values():
            node_S = Node(
                    subSystemOf = [ventilation_system],
                    operationMode = "supply",
                    input = {},
                    output = {},
                    savedInput = {},
                    savedOutput = {},
                    createReport = self.createReport,
                    connectedThrough = [],
                    connectsAt = [],
                    # id = f"N_supply_{ventilation_system.id}")
                    id = "Supply node") ####
            self.add_component(node_S)
            ventilation_system.hasSubSystem.append(node_S)
            node_E = Node(
                    subSystemOf = [ventilation_system],
                    operationMode = "exhaust",
                    input = {},
                    output = {},
                    savedInput = {},
                    savedOutput = {},
                    createReport = self.createReport,
                    connectedThrough = [],
                    connectsAt = [],
                    # id = f"N_exhaust_{ventilation_system.id}") ##############################
                    id = "Exhaust node") ####
            self.add_component(node_E)
            ventilation_system.hasSubSystem.append(node_E)





        
    def get_component_by_class(self, dict_, class_):
        return [v for v in dict_.values() if isinstance(v, class_)]

    def get_dampers_by_space(self, space):
        return [component for component in space.contains if isinstance(component, Damper)]

    def get_space_heaters_by_space(self, space):
        return [component for component in space.contains if isinstance(component, BuildingSpace)]

    def get_valves_by_space(self, space):
        return [component for component in space.contains if isinstance(component, Valve)]

    def get_controllers_by_space(self, space):
        return [component for component in space.contains if isinstance(component, Controller)]


    def connect(self):
        """
        Connects component instances using the saref4syst extension.
        It currently assumes that components comply with a certain naming pattern:
        C_T_{space.id}: Temperature controller used to control temperature in space.
        C_C_{space.id}: CO2 controller used to control CO2-concentration in space.
        D_S_{space.id}: Supply damper in space.
        D_E_{space.id}: Exhaust damper in space.
        V_{space.id}: Valve in space.
        SH_{space.id}: Space heater in space.

        HC_{ventilation_system.id}_{heating_system.id}: Heating coil in ventilation_system and heating_system.
        CC_{ventilation_system.id}_{cooling_system.id}: Cooling coil in ventilation_system and cooling_system.
        HR_{ventilation_system.id}: Heat recovery unit in ventilation_system.
        F_S_{ventilation_system.id}: Supply fan in ventilation_system.
        F_E_{ventilation_system.id}: Exhaust fan in ventilation_system.
        """


        space_instances = self.get_component_by_class(self.component_dict, BuildingSpaceModel)
        damper_instances = self.get_component_by_class(self.component_dict, DamperModel)
        space_heater_instances = self.get_component_by_class(self.component_dict, SpaceHeaterModel)
        valve_instances = self.get_component_by_class(self.component_dict, ValveModel)
        coil_heating_instances = self.get_component_by_class(self.component_dict, CoilHeatingModel)
        coil_cooling_instances = self.get_component_by_class(self.component_dict, CoilCoolingModel)
        air_to_air_heat_recovery_instances = self.get_component_by_class(self.component_dict, AirToAirHeatRecoveryModel)
        fan_instances = self.get_component_by_class(self.component_dict, FanModel)
        node_instances = self.get_component_by_class(self.component_dict, Node)
        controller_instances = self.get_component_by_class(self.component_dict, ControllerModel)
        sensor_instances = self.get_component_by_class(self.component_dict, SensorModel)
        # controller_instances.extend(self.get_component_by_class(ControllerModelRulebased)) #######################


        outdoor_environment = self.component_dict["Outdoor environment"]
        occupancy_schedule = self.component_dict["Occupancy schedule"]
        indoor_temperature_setpoint_schedule = self.component_dict["Temperature setpoint schedule"]
        co2_setpoint_schedule = self.component_dict["CO2 setpoint schedule"]
        supply_air_temperature_setpoint_schedule = self.component_dict["Supply temperature setpoint schedule"]
        shade_setpoint_schedule = self.component_dict["Shade setpoint schedule"]
        shading_device = self.component_dict["Shade"]


        for space in space_instances:
            dampers = self.get_dampers_by_space(space)
            controllers = self.get_controllers_by_space(space)

            for controller in controllers:
                sensor = controller.controlsProperty.isMeasuredByDevice
                if isinstance(controller.controlsProperty, Temperature):
                    self.add_connection(space, sensor, "indoorTemperature", "indoorTemperature") ###
                    self.add_connection(sensor, controller, "indoorTemperature", "actualValue") ###
                    self.add_connection(controller, space, "inputSignal", "valvePosition")
                elif isinstance(controller.controlsProperty, Co2):
                    self.add_connection(space, sensor, "indoorCo2Concentration", "indoorCo2Concentration") ###
                    self.add_connection(sensor, controller, "indoorCo2Concentration", "actualValue") ###
                    self.add_connection(controller, space, "inputSignal", "damperPosition") ###
                    # self.add_connection(controller, space, "inputSignal", "returnDamperPosition")

            for damper in dampers:
                if damper.operationMode=="supply":
                    self.add_connection(damper, space, "airFlowRate", "supplyAirFlowRate")
                    ventilation_system = damper.subSystemOf[0]
                    node = [v for v in ventilation_system.hasSubSystem if isinstance(v, Node) and v.operationMode=="supply"][0]
                    self.add_connection(damper, node, "airFlowRate", "flowRate_" + space.id) ###
                    self.add_connection(space, node, "indoorTemperature", "flowTemperatureIn_" + space.id) ###
                elif damper.operationMode=="exhaust":
                    self.add_connection(damper, space, "airFlowRate", "returnAirFlowRate")
                    ventilation_system = damper.subSystemOf[0]
                    node = [v for v in ventilation_system.hasSubSystem if isinstance(v, Node) and v.operationMode=="exhaust"][0]
                    self.add_connection(damper, node, "airFlowRate", "flowRate_" + space.id) ###
                    self.add_connection(space, node, "indoorTemperature", "flowTemperatureIn_" + space.id) ###

            self.add_connection(outdoor_environment, space, "shortwaveRadiation", "shortwaveRadiation")
            self.add_connection(outdoor_environment, space, "outdoorTemperature", "outdoorTemperature")
            self.add_connection(occupancy_schedule, space, "scheduleValue", "numberOfPeople")
            self.add_connection(shade_setpoint_schedule, shading_device, "scheduleValue", "shadePosition")
            self.add_connection(shading_device, space, "shadePosition", "shadePosition")
            

        for damper in damper_instances:
            controllers = self.get_controllers_by_space(damper.isContainedIn)
            controller = [controller for controller in controllers if isinstance(controller.controlsProperty, Co2)][0]
            self.add_connection(controller, damper, "inputSignal", "damperPosition")

        for space_heater in space_heater_instances:
            space = space_heater.isContainedIn
            valve = self.get_valves_by_space(space)[0]
            self.add_connection(space, space_heater, "indoorTemperature", "indoorTemperature") 
            self.add_connection(valve, space_heater, "waterFlowRate", "waterFlowRate")

        for valve in valve_instances:
            controllers = self.get_controllers_by_space(valve.isContainedIn)
            controller = [controller for controller in controllers if isinstance(controller.controlsProperty, Temperature)][0]
            self.add_connection(controller, valve, "inputSignal", "valvePosition")

        for coil_heating in coil_heating_instances:
            for system in coil_heating.subSystemOf:
                air_to_air_heat_recovery = [v for v in system.hasSubSystem if isinstance(v, AirToAirHeatRecoveryModel)]
                if len(air_to_air_heat_recovery)!=0:
                    air_to_air_heat_recovery = air_to_air_heat_recovery[0]
                    node = [v for v in system.hasSubSystem if isinstance(v, Node) and v.operationMode == "supply"][0]
                    self.add_connection(air_to_air_heat_recovery, coil_heating, "primaryTemperatureOut", "supplyAirTemperature")
                    self.add_connection(node, coil_heating, "flowRate", "airFlowRate")
                    self.add_connection(supply_air_temperature_setpoint_schedule, coil_heating, "scheduleValue", "supplyAirTemperatureSetpoint")

        for coil_cooling in coil_cooling_instances:
            for system in coil_cooling.subSystemOf:
                air_to_air_heat_recovery = [v for v in system.hasSubSystem if isinstance(v, AirToAirHeatRecoveryModel)]
                if len(air_to_air_heat_recovery)!=0:
                    air_to_air_heat_recovery = air_to_air_heat_recovery[0]
                    node = [v for v in system.hasSubSystem if isinstance(v, Node) and v.operationMode == "supply"][0]
                    self.add_connection(air_to_air_heat_recovery, coil_cooling, "primaryTemperatureOut", "supplyAirTemperature")
                    self.add_connection(node, coil_cooling, "flowRate", "airFlowRate")
                    self.add_connection(supply_air_temperature_setpoint_schedule, coil_cooling, "scheduleValue", "supplyAirTemperatureSetpoint")

        for air_to_air_heat_recovery in air_to_air_heat_recovery_instances:
            ventilation_system = air_to_air_heat_recovery.subSystemOf[0]
            node_S = [v for v in ventilation_system.hasSubSystem if isinstance(v, Node) and v.operationMode == "supply"][0]
            node_E = [v for v in ventilation_system.hasSubSystem if isinstance(v, Node) and v.operationMode == "exhaust"][0]
            self.add_connection(outdoor_environment, air_to_air_heat_recovery, "outdoorTemperature", "primaryTemperatureIn")
            self.add_connection(node_E, air_to_air_heat_recovery, "flowTemperatureOut", "secondaryTemperatureIn")
            self.add_connection(node_S, air_to_air_heat_recovery, "flowRate", "primaryAirFlowRate")
            self.add_connection(node_E, air_to_air_heat_recovery, "flowRate", "secondaryAirFlowRate")

        for fan in fan_instances:
            ventilation_system = fan.subSystemOf[0]
            if fan.operationMode == "supply":
                node_S = [v for v in ventilation_system.hasSubSystem if isinstance(v, Node) and v.operationMode=="supply"][0]
                self.add_connection(node_S, fan, "flowRate", "airFlowRate")
            elif fan.operationMode == "exhaust":
                node_E = [v for v in ventilation_system.hasSubSystem if isinstance(v, Node) and v.operationMode=="exhaust"][0]
                self.add_connection(node_E, fan, "flowRate", "airFlowRate")

        for controller in controller_instances:
            if isinstance(controller.controlsProperty, Temperature):
                self.add_connection(indoor_temperature_setpoint_schedule, controller, "scheduleValue", "setpointValue")
            elif isinstance(controller.controlsProperty, Co2):
                self.add_connection(co2_setpoint_schedule, controller, "scheduleValue", "setpointValue")

    def read_config_name_based(self):
        """
        Reads configuration file and instantiates an object for each entry in the file.  
        It assumes that the sheets of the configuration file follows a certain naming pattern.
        
        xlsx:
            Sheets:
                Systems
                Spaces
                Dampers
                SpaceHeaters
                Valves
                HeatingCoils
                CoolingCoils
                AirToAirHeatRecovery
                Fan
                Controller
                Node

        """

        file_name = "configuration_template_Automation_paper_10space_1v_1h_0c.xlsx"
        file_path = os.path.join(uppath(os.path.abspath(__file__), 2), "test", "data", file_name)

        df_Systems = pd.read_excel(file_path, sheet_name="Systems")
        df_Spaces = pd.read_excel(file_path, sheet_name="Spaces")
        df_Dampers = pd.read_excel(file_path, sheet_name="Dampers")
        df_SpaceHeaters = pd.read_excel(file_path, sheet_name="SpaceHeaters")
        df_Valves = pd.read_excel(file_path, sheet_name="Valves")
        df_HeatingCoils = pd.read_excel(file_path, sheet_name="HeatingCoils")
        df_CoolingCoils = pd.read_excel(file_path, sheet_name="CoolingCoils")
        df_AirToAirHeatRecovery = pd.read_excel(file_path, sheet_name="AirToAirHeatRecovery")
        df_Fans = pd.read_excel(file_path, sheet_name="Fan")
        df_Controller = pd.read_excel(file_path, sheet_name="Controller")
        df_Nodes = pd.read_excel(file_path, sheet_name="Node")

        for ventilation_system_name in df_Systems["Ventilation system name"].dropna():
            ventilation_system = DistributionDevice(subSystemOf = [], hasSubSystem = [], id = ventilation_system_name)
            self.system_dict[ventilation_system_name] = ventilation_system
        
        for heating_system_name in df_Systems["Heating system name"].dropna():
            heating_system = DistributionDevice(subSystemOf = [], hasSubSystem = [], id = heating_system_name)
            self.system_dict[heating_system_name] = heating_system

        for cooling_system_name in df_Systems["Cooling system name"].dropna():
            cooling_system = DistributionDevice(subSystemOf = [], hasSubSystem = [], id = cooling_system_name)
            self.system_dict[cooling_system_name] = cooling_system

        for row in df_Spaces.dropna(subset=["Space name"]).itertuples(index=False):
            space_name = row[df_Spaces.columns.get_loc("Space name")]
            try: 
                space = BuildingSpaceModel(
                    densityAir = 1.225,
                    airVolume = 466.54,
                    startPeriod = self.startPeriod,
                    timeStep = self.timeStep,
                    input = {"generationCo2Concentration": 0.000009504,
                            "outdoorCo2Concentration": 400},
                    output = {"indoorTemperature": 21,
                            "indoorCo2Concentration": 500},
                    savedInput = {},
                    savedOutput = {},
                    createReport = self.createReport,
                    connectedThrough = [],
                    connectsAt = [],
                    id = space_name)
                self.component_dict[space_name] = space
            except NoSpaceModelException: 
                print("No fitting space model for space " + "\"" + space_name + "\"")
                print("Continuing...")
            

        for row in df_Dampers.dropna(subset=["Damper name"]).itertuples(index=False):
            damper_name = row[df_Dampers.columns.get_loc("Damper name")]
            #Check that an appropriate space object exists
            if damper_name[4:] not in self.component_dict:
                warnings.warn("Cannot find a matching mathing BuildingSpace object for damper \"" + damper_name + "\"")
            else:
                ventilation_system = self.system_dict[row[df_Dampers.columns.get_loc("Ventilation system")]]
                damper = DamperModel(
                    nominalAirFlowRate = Measurement(hasValue=row[df_Dampers.columns.get_loc("nominalAirFlowRate")]),
                    subSystemOf = [ventilation_system],
                    input = {},
                    output = {"airFlowRate": 0},
                    savedInput = {},
                    savedOutput = {},
                    createReport = self.createReport,
                    connectedThrough = [],
                    connectsAt = [],
                    id = damper_name)
                self.component_dict[damper_name] = damper
                ventilation_system.hasSubSystem.append(damper)

        for row in df_SpaceHeaters.dropna(subset=["Space heater name"]).itertuples(index=False):
            space_heater_name = row[df_SpaceHeaters.columns.get_loc("Space heater name")]
            #Check that an appropriate space object exists
            if space_heater_name[3:] not in self.component_dict:
                warnings.warn("Cannot find a matching mathing BuildingSpace object for space heater \"" + space_heater_name + "\"")
            else:
                heating_system = self.system_dict[row[df_SpaceHeaters.columns.get_loc("Heating system")]]
                space_heater = SpaceHeaterModel(
                    specificHeatCapacityWater = Measurement(hasValue=4180),
                    outputCapacity = Measurement(hasValue=row[df_SpaceHeaters.columns.get_loc("outputCapacity")]),
                    temperatureClassification = row[df_SpaceHeaters.columns.get_loc("temperatureClassification")],
                    thermalMassHeatCapacity = Measurement(hasValue=row[df_SpaceHeaters.columns.get_loc("thermalMassHeatCapacity")]),
                    timeStep = self.timeStep, 
                    subSystemOf = [heating_system],
                    input = {"supplyWaterTemperature": 60},
                    output = {"radiatorOutletTemperature": 22,
                                "Energy": 0},
                    savedInput = {},
                    savedOutput = {},
                    createReport = self.createReport,#self.createReport,
                    connectedThrough = [],
                    connectsAt = [],
                    id = space_heater_name)
                self.component_dict[space_heater_name] = space_heater
                heating_system.hasSubSystem.append(space_heater)

        for row in df_Valves.dropna(subset=["Valve name"]).itertuples(index=False):
            valve_name = row[df_Valves.columns.get_loc("Valve name")]
            #Check that an appropriate space object exists
            if valve_name[2:] not in self.component_dict:
                warnings.warn("Cannot find a matching mathing BuildingSpace object for valve \"" + valve_name + "\"")
            else:
                heating_system = self.component_dict[valve_name.replace("V_", "SH_")].subSystemOf[0]
                valve = ValveModel(
                    valveAuthority = Measurement(hasValue=0.8),
                    flowCoefficient = Measurement(hasValue=row[df_Valves.columns.get_loc("flowCoefficient")]),
                    testPressure = Measurement(hasValue=row[df_Valves.columns.get_loc("testPressure")]),
                    subSystemOf = [heating_system],
                    input = {},
                    output = {},
                    savedInput = {},
                    savedOutput = {},
                    createReport = self.createReport,
                    connectedThrough = [],
                    connectsAt = [],
                    id = valve_name)
                self.component_dict[valve_name] = valve
                heating_system.hasSubSystem.append(valve)

        for row in df_HeatingCoils.dropna(subset=["Heating coil name"]).itertuples(index=False):
            heating_coil_name = row[df_HeatingCoils.columns.get_loc("Heating coil name")]
            ventilation_system = self.system_dict[row[df_HeatingCoils.columns.get_loc("Ventilation system")]]
            heating_system = self.system_dict[row[df_HeatingCoils.columns.get_loc("Heating system")]]
            heating_coil = CoilHeatingModel(
                specificHeatCapacityAir = Measurement(hasValue=1000),
                subSystemOf = [ventilation_system, heating_system],
                input = {"supplyAirTemperatureSetpoint": 23},
                output = {},
                savedInput = {},
                savedOutput = {},
                createReport = self.createReport,
                connectedThrough = [],
                connectsAt = [],
                id = heating_coil_name)
            self.component_dict[heating_coil_name] = heating_coil
            ventilation_system.hasSubSystem.append(heating_coil)
            heating_system.hasSubSystem.append(heating_coil)

        for row in df_CoolingCoils.dropna(subset=["Cooling coil name"]).itertuples(index=False):
            cooling_coil_name = row[df_CoolingCoils.columns.get_loc("Cooling coil name")]
            ventilation_system = self.system_dict[row[df_CoolingCoils.columns.get_loc("Ventilation system")]]
            cooling_system = self.system_dict[row[df_CoolingCoils.columns.get_loc("Cooling system")]]
            cooling_coil = CoilCoolingModel(
                specificHeatCapacityAir = Measurement(hasValue=1000),
                subSystemOf = [ventilation_system, cooling_system],
                input = {"supplyAirTemperatureSetpoint": 23},
                output = {},
                savedInput = {},
                savedOutput = {},
                createReport = self.createReport,
                connectedThrough = [],
                connectsAt = [],
                id = cooling_coil_name)
            self.component_dict[cooling_coil_name] = cooling_coil
            ventilation_system.hasSubSystem.append(cooling_coil)
            cooling_system.hasSubSystem.append(cooling_coil)

        for row in df_AirToAirHeatRecovery.dropna(subset=["Air to air heat recovery name"]).itertuples(index=False):
            air_to_air_heat_recovery_name = row[df_AirToAirHeatRecovery.columns.get_loc("Air to air heat recovery name")]
            ventilation_system = self.system_dict[row[df_AirToAirHeatRecovery.columns.get_loc("Ventilation system")]]
            air_to_air_heat_recovery = air_to_air_heat_recovery = AirToAirHeatRecoveryModel(
                specificHeatCapacityAir = Measurement(hasValue=1000),
                eps_75_h = Measurement(hasValue=0.8),
                eps_75_c = Measurement(hasValue=0.8),
                eps_100_h = Measurement(hasValue=0.8),
                eps_100_c = Measurement(hasValue=0.8),
                primaryAirFlowRateMax = Measurement(hasValue=row[df_AirToAirHeatRecovery.columns.get_loc("primaryAirFlowRateMax")]),
                secondaryAirFlowRateMax = Measurement(hasValue=row[df_AirToAirHeatRecovery.columns.get_loc("secondaryAirFlowRateMax")]),
                subSystemOf = [ventilation_system],
                input = {},
                output = {},
                savedInput = {},
                savedOutput = {},
                createReport = self.createReport,
                connectedThrough = [],
                connectsAt = [],
                id = air_to_air_heat_recovery_name)
            self.component_dict[air_to_air_heat_recovery_name] = air_to_air_heat_recovery
            ventilation_system.hasSubSystem.append(air_to_air_heat_recovery)

        for row in df_Fans.dropna(subset=["Fan name"]).itertuples(index=False):
            fan_name = row[df_Fans.columns.get_loc("Fan name")]
            ventilation_system = self.system_dict[row[df_Fans.columns.get_loc("Ventilation system")]]
            fan = FanModel(
                c1=Measurement(hasValue=0.0015302446),
                c2=Measurement(hasValue=0.0052080574),
                c3=Measurement(hasValue=1.1086242),
                c4=Measurement(hasValue=-0.11635563),
                timeStep = self.timeStep,
                nominalAirFlowRate = Measurement(hasValue=row[df_Fans.columns.get_loc("nominalAirFlowRate")]),
                nominalPowerRate = Measurement(hasValue=row[df_Fans.columns.get_loc("nominalPowerRate")]),
                subSystemOf = [ventilation_system],
                input = {},
                output = {"Energy": 0},
                savedInput = {},
                savedOutput = {},
                createReport = self.createReport,
                connectedThrough = [],
                connectsAt = [],
                id = fan_name)
            self.component_dict[fan_name] = fan
            ventilation_system.hasSubSystem.append(fan)

        for row in df_Nodes.dropna(subset=["Node name"]).itertuples(index=False):
            node_name = row[df_Nodes.columns.get_loc("Node name")]
            ventilation_system = self.system_dict[row[df_Nodes.columns.get_loc("Ventilation system")]]
            node = Node(
                subSystemOf = [ventilation_system],
                input = {},
                output = {},
                savedInput = {},
                savedOutput = {},
                createReport = self.createReport,
                connectedThrough = [],
                connectsAt = [],
                id = node_name)
            self.component_dict[node_name] = node
            ventilation_system.hasSubSystem.append(node)

        for row in df_Controller.dropna(subset=["Controller name"]).itertuples(index=False):
            controller_name = row[df_Controller.columns.get_loc("Controller name")]
            if controller_name[4:] not in self.component_dict:
                warnings.warn("Cannot find a matching mathing BuildingSpace object for controller \"" + controller_name + "\"")
            else: #controller_name[0:4] == "C_T_":
                controller = ControllerModel(
                    K_p = row[df_Controller.columns.get_loc("K_p")],
                    K_i = row[df_Controller.columns.get_loc("K_i")],
                    K_d = row[df_Controller.columns.get_loc("K_d")],
                    subSystemOf = [],
                    input = {},
                    output = {"inputSignal": 0},
                    savedInput = {},
                    savedOutput = {},
                    createReport = self.createReport,
                    connectedThrough = [],
                    connectsAt = [],
                    id = controller_name)
                self.component_dict[controller_name] = controller
            # elif controller_name[:4]=="C_C_":
            #     controller = ControllerModelRulebased(
            #         subSystemOf = [],
            #         input = {},
            #         output = {"inputSignal": 0},
            #         savedInput = {},
            #         savedOutput = {},
            #         createReport = self.createReport,
            #         connectedThrough = [],
            #         connectsAt = [],
            #         id = controller_name)
            #     self.component_dict[controller_name] = controller
                ventilation_system.hasSubSystem.append(controller)

    def connect_name_based(self):
        """
        Connects component instances using the saref4syst extension.
        It currently assumes that components comply with a certain naming pattern:
        C_T_{space.id}: Temperature controller used to control temperature in space.
        C_C_{space.id}: CO2 controller used to control CO2-concentration in space.
        D_S_{space.id}: Supply damper in space.
        D_E_{space.id}: Exhaust damper in space.
        V_{space.id}: Valve in space.
        SH_{space.id}: Space heater in space.

        HC_{ventilation_system.id}_{heating_system.id}: Heating coil in ventilation_system and heating_system.
        CC_{ventilation_system.id}_{cooling_system.id}: Cooling coil in ventilation_system and cooling_system.
        HR_{ventilation_system.id}: Heat recovery unit in ventilation_system.
        F_S_{ventilation_system.id}: Supply fan in ventilation_system.
        F_E_{ventilation_system.id}: Exhaust fan in ventilation_system.
        """
        space_instances = self.get_component_by_class(self.component_dict, BuildingSpaceModel)
        damper_instances = self.get_component_by_class(self.component_dict, DamperModel)
        space_heater_instances = self.get_component_by_class(self.component_dict, SpaceHeaterModel)
        valve_instances = self.get_component_by_class(self.component_dict, ValveModel)
        coil_heating_instances = self.get_component_by_class(self.component_dict, CoilHeatingModel)
        coil_cooling_instances = self.get_component_by_class(self.component_dict, CoilCoolingModel)
        air_to_air_heat_recovery_instances = self.get_component_by_class(self.component_dict, AirToAirHeatRecoveryModel)
        fan_instances = self.get_component_by_class(self.component_dict, FanModel)
        node_instances = self.get_component_by_class(self.component_dict, Node)
        controller_instances = self.get_component_by_class(self.component_dict, ControllerModel)
        # controller_instances.extend(self.get_component_by_class(ControllerModelRulebased)) #######################


        outdoor_environment = self.component_dict["outdoor_environment"]
        occupancy_schedule = self.component_dict["occupancy_schedule"]
        indoor_temperature_setpoint_schedule = self.component_dict["indoor_temperature_setpoint_schedule"]
        co2_setpoint_schedule = self.component_dict["co2_setpoint_schedule"]
        supply_air_temperature_setpoint_schedule = self.component_dict["supply_air_temperature_setpoint_schedule"]
        shade_setpoint_schedule = self.component_dict["shade_setpoint_schedule"]


        for space in space_instances:
            if "C_T_" + space.id in self.component_dict:
                temperature_controller = self.component_dict["C_T_" + space.id]
                self.add_connection(space, temperature_controller, "indoorTemperature", "actualValue") ###
                self.add_connection(temperature_controller, space, "inputSignal", "valvePosition")

            if "C_C_" + space.id in self.component_dict:
                co2_controller = self.component_dict["C_C_" + space.id]
                self.add_connection(space, co2_controller, "indoorCo2Concentration", "actualValue") ###
                self.add_connection(co2_controller, space, "inputSignal", "supplyDamperPosition") ###
                self.add_connection(co2_controller, space, "inputSignal", "returnDamperPosition")

            if "D_S_" + space.id in self.component_dict:
                damper = self.component_dict["D_S_" + space.id]
                self.add_connection(damper, space, "airFlowRate", "supplyAirFlowRate")
                ventilation_system = damper.subSystemOf[0]
                node = [v for v in ventilation_system.hasSubSystem if isinstance(v, Node) and v.id[0:4] == "N_S_"][0]
                self.add_connection(damper, node, "airFlowRate", "flowRate_" + space.id) ###
                self.add_connection(space, node, "indoorTemperature", "flowTemperatureIn_" + space.id) ###
                
            if "D_E_" + space.id in self.component_dict:
                damper = self.component_dict["D_E_" + space.id]
                self.add_connection(damper, space, "airFlowRate", "returnAirFlowRate")
                ventilation_system = damper.subSystemOf[0]
                node = [v for v in ventilation_system.hasSubSystem if isinstance(v, Node) and v.id[0:4] == "N_E_"][0]
                self.add_connection(damper, node, "airFlowRate", "flowRate_" + space.id) ###
                self.add_connection(space, node, "indoorTemperature", "flowTemperatureIn_" + space.id) ###

            self.add_connection(outdoor_environment, space, "shortwaveRadiation", "shortwaveRadiation")
            # self.add_connection(outdoor_environment, space, "longwaveRadiation", "longwaveRadiation")
            self.add_connection(outdoor_environment, space, "outdoorTemperature", "outdoorTemperature")
            self.add_connection(occupancy_schedule, space, "scheduleValue", "numberOfPeople")
            self.add_connection(shade_setpoint_schedule, space, "scheduleValue", "shadePosition")

        for damper in damper_instances:
            if "C_C_" + damper.id[4:] in self.component_dict:
                co2_controller = self.component_dict["C_C_" + damper.id[4:]]
                self.add_connection(co2_controller, damper, "inputSignal", "damperPosition")

        for space_heater in space_heater_instances:
            space = self.component_dict[space_heater.id[3:]]
            valve = self.component_dict["V_" + space_heater.id[3:]]
            self.add_connection(space, space_heater, "indoorTemperature", "indoorTemperature") 
            self.add_connection(valve, space_heater, "waterFlowRate", "waterFlowRate")

        for valve in valve_instances:
            if "C_T_" + valve.id[2:] in self.component_dict:
                temperature_controller = self.component_dict["C_T_" + valve.id[2:]]
                self.add_connection(temperature_controller, valve, "inputSignal", "valvePosition")

        for coil_heating in coil_heating_instances:
            for system in coil_heating.subSystemOf:
                air_to_air_heat_recovery = [v for v in system.hasSubSystem if isinstance(v, AirToAirHeatRecoveryModel)]
                if len(air_to_air_heat_recovery)!=0:
                    air_to_air_heat_recovery = air_to_air_heat_recovery[0]
                    node = [v for v in system.hasSubSystem if isinstance(v, Node) and v.id[0:4] == "N_S_"][0]
                    self.add_connection(air_to_air_heat_recovery, coil_heating, "primaryTemperatureOut", "supplyAirTemperature")
                    self.add_connection(node, coil_heating, "flowRate", "airFlowRate")
                    self.add_connection(supply_air_temperature_setpoint_schedule, coil_heating, "scheduleValue", "supplyAirTemperatureSetpoint")

        for coil_cooling in coil_cooling_instances:
            for system in coil_cooling.subSystemOf:
                air_to_air_heat_recovery = [v for v in system.hasSubSystem if isinstance(v, AirToAirHeatRecoveryModel)]
                if len(air_to_air_heat_recovery)!=0:
                    air_to_air_heat_recovery = air_to_air_heat_recovery[0]
                    node = [v for v in system.hasSubSystem if isinstance(v, Node) and v.id[0:4] == "N_S_"][0]
                    self.add_connection(air_to_air_heat_recovery, coil_cooling, "primaryTemperatureOut", "supplyAirTemperature")
                    self.add_connection(node, coil_cooling, "flowRate", "airFlowRate")
                    self.add_connection(supply_air_temperature_setpoint_schedule, coil_cooling, "scheduleValue", "supplyAirTemperatureSetpoint")

        for air_to_air_heat_recovery in air_to_air_heat_recovery_instances:
            ventilation_system = air_to_air_heat_recovery.subSystemOf[0]
            node_S = [v for v in ventilation_system.hasSubSystem if isinstance(v, Node) and v.id[0:4] == "N_S_"][0]
            node_E = [v for v in ventilation_system.hasSubSystem if isinstance(v, Node) and v.id[0:4] == "N_E_"][0]
            self.add_connection(outdoor_environment, air_to_air_heat_recovery, "outdoorTemperature", "primaryTemperatureIn")
            self.add_connection(node_E, air_to_air_heat_recovery, "flowTemperatureOut", "secondaryTemperatureIn")
            self.add_connection(node_S, air_to_air_heat_recovery, "flowRate", "primaryAirFlowRate")
            self.add_connection(node_E, air_to_air_heat_recovery, "flowRate", "secondaryAirFlowRate")

        for fan in fan_instances:
            ventilation_system = fan.subSystemOf[0]
            if fan.id[0:4] == "F_S_":
                node_S = [v for v in ventilation_system.hasSubSystem if isinstance(v, Node) and v.id[0:4] == "N_S_"][0]
                self.add_connection(node_S, fan, "flowRate", "airFlowRate")
            elif fan.id[0:4] == "F_E_":
                node_E = [v for v in ventilation_system.hasSubSystem if isinstance(v, Node) and v.id[0:4] == "N_E_"][0]
                self.add_connection(node_E, fan, "flowRate", "airFlowRate")

        for controller in controller_instances:
            if controller.id[0:4] == "C_T_":
                self.add_connection(indoor_temperature_setpoint_schedule, controller, "scheduleValue", "setpointValue")
            elif controller.id[0:4] == "C_C_":
                self.add_connection(co2_setpoint_schedule, controller, "scheduleValue", "setpointValue")

    def init_building_space_models(self):
        for space in self.get_component_by_class(self.component_dict, BuildingSpaceModel):
            space.get_model()

    
    def load_model(self, read_config=True):
        self.add_outdoor_environment()
        self.add_occupancy_schedule()
        self.add_indoor_temperature_setpoint_schedule()
        self.add_co2_setpoint_schedule()
        self.add_supply_air_temperature_setpoint_schedule()
        self.add_shade_setpoint_schedule()
        self.add_shading_device()

        if read_config:

            # self.read_config_name_based()
            # self.connect_name_based()
            self.read_config()
            self.apply_model_extensions()
            self.connect()

        print("Finished loading model")


    def draw_system_graph_no_cycles(self):
        light_black = "#3B3838"
        dark_blue = "#44546A"
        orange = "#C55A11"
        red = "#873939"
        grey = "#666666"
        light_grey = "#71797E"

        file_name = "system_graph_no_cycles"
        self.system_graph_no_cycles.write(f"{file_name}.dot", prog="dot")

        # If Python can't find the dot executeable, change "app_path" variable to the full path
        app_path = shutil.which("dot")
        

        args = [app_path,
                "-Tpng",
                "-Kdot",
                "-Nstyle=filled", #rounded,filled
                "-Nshape=box",
                "-Nfontcolor=white",
                "-Nfontname=Times-Roman",
                "-Nfixedsize=true",
                # "-Gnodesep=3",
                "-Nnodesep=0.05",
                # "-Esamehead=true",
                "-Efontname=Helvetica",
                "-Epenwidth=2",
                "-Eminlen=1",
                f"-Ecolor={light_grey}",
                "-Gcompound=true",
                "-Grankdir=TB",
                "-Goverlap=scale",
                "-Gsplines=true",
                "-Gmargin=0",
                "-Gratio=compress",
                "-Gsize=5!",
                # "-Gratio=0.4", #0.5
                "-Gpack=true",
                "-Gdpi=1000",
                "-Grepulsiveforce=0.5",
                "-Gremincross=true",
                # "-Gbgcolor=#EDEDED",
                f"-o{file_name}.png",
                f"{file_name}.dot"]
        subprocess.run(args=args)


    def draw_system_graph(self):

        light_black = "#3B3838"
        dark_blue = "#44546A"
        orange = "#C55A11"
        red = "#873939"
        grey = "#666666"
        light_grey = "#71797E"
        light_blue = "#8497B0"
        yellow = "#BF9000"

        fill_color_dict = {"OutdoorEnvironment": grey,
                            "Schedule": grey,
                            "BuildingSpaceModel": light_black,
                            "BuildingSpaceModelCo2": light_black,
                            "ControllerModel": orange,
                            "AirToAirHeatRecoveryModel": dark_blue,
                            "CoilHeatingModel": red,
                            "CoilCoolingModel": dark_blue,
                            "DamperModel": dark_blue,
                            "ValveModel": red,
                            "FanModel": dark_blue,
                            "SpaceHeaterModel": red,
                            "Node": grey,
                            "ShadingDeviceModel": light_blue,
                            "SensorModel": yellow}


        border_color_dict = {"OutdoorEnvironment": "black",
                            "Schedule": "black",
                            "BuildingSpaceModel": "black",#"#2F528F",
                            "BuildingSpaceModelCo2": "black",
                            "ControllerModel": "black",
                            "AirToAirHeatRecoveryModel": "black",
                            "CoilHeatingModel": "black",
                            "CoilCoolingModel": "black",
                            "DamperModel": "black",
                            "ValveModel": "black",
                            "FanModel": "black",
                            "SpaceHeaterModel": "black",
                            "Node": "black",
                            "ShadingDeviceModel": "black",
                            "SensorModel": "black"}


        # K = 10
        K = 1
        min_fontsize = 22*K
        max_fontsize = 30*K

        min_width = 3.5*K
        max_width = 6*K

        min_width_char = 3.5*K
        max_width_char = 5*K

        min_height = 0.4*K
        max_height = 1*K

        nx_graph = nx.drawing.nx_pydot.from_pydot(self.system_graph)

        degree_list = [nx_graph.degree(node) for node in nx_graph.nodes()]
        min_deg = min(degree_list)
        max_deg = max(degree_list)

        char_list = [len(self.system_graph_node_attribute_dict[node]["label"]) for node in nx_graph.nodes()]
        min_char = min(char_list)
        max_char = max(char_list)

        if max_deg!=min_deg:
            a_fontsize = (max_fontsize-min_fontsize)/(max_deg-min_deg)
            b_fontsize = max_fontsize-a_fontsize*max_deg
        else:
            a_fontsize = 0
            b_fontsize = max_fontsize

        # a_width = (max_width-min_width)/(max_deg-min_deg)
        # b_width = max_width-a_width*max_deg


        if max_deg!=min_deg:
            a_width_char = (max_width_char-min_width_char)/(max_char-min_char)
            b_width_char = max_width_char-a_width_char*max_char
        else:
            a_width_char = 0
            b_width_char = max_width_char




        if max_deg!=min_deg:
            a_height = (max_height-min_height)/(max_deg-min_deg)
            b_height = max_height-a_height*max_deg
        else:
            a_height = 0
            b_height = max_height

        


        for node in nx_graph.nodes():
            deg = nx_graph.degree(node)
            fontsize = a_fontsize*deg + b_fontsize
            # width = a_width*deg + b_width
            width = a_width_char*len(self.system_graph_node_attribute_dict[node]["label"]) + b_width_char
            height = a_height*deg + b_height
            if node not in self.system_graph_node_attribute_dict:
                self.system_graph_node_attribute_dict[node] = {}

            self.system_graph_node_attribute_dict[node]["fontsize"] = fontsize
            self.system_graph_node_attribute_dict[node]["width"] = width
            self.system_graph_node_attribute_dict[node]["height"] = height
            self.system_graph_node_attribute_dict[node]["fillcolor"] = fill_color_dict[self.component_dict[node].__class__.__name__]
            self.system_graph_node_attribute_dict[node]["color"] = border_color_dict[self.component_dict[node].__class__.__name__]

            subgraph = self.subgraph_dict[type(self.component_dict[node]).__name__]

            if " " in node or "Ø" in node:
                name = "\"" + node + "\""
            else:
                name = node

            


            if len(subgraph.get_node(name))==1:
                subgraph.get_node(name)[0].obj_dict["attributes"].update(self.system_graph_node_attribute_dict[node])
            else:
                aa

        
        file_name = "system_graph"
        self.system_graph.write(f'{file_name}.dot')

        # If Python can't find the dot executeable, change "app_path" variable to the full path
        app_path = shutil.which("dot")
        

        args = [app_path,
                "-Tpng",
                "-Kdot",
                "-Nstyle=filled",
                "-Nshape=box",
                "-Nfontcolor=white",
                "-Nfontname=Times-Roman",
                "-Nfixedsize=true",
                # "-Gnodesep=3",
                "-Nnodesep=0.05",
                "-Efontname=Helvetica",
                "-Epenwidth=2",
                "-Eminlen=1",
                f"-Ecolor={light_grey}",
                "-Gcompound=true",
                "-Grankdir=TB",
                "-Goverlap=scale",
                "-Gsplines=true",
                "-Gmargin=0",
                "-Gratio=compress",
                "-Gsize=5!",
                # "-Gratio=0.4", #0.5
                "-Gpack=true",
                "-Gdpi=1000",
                "-Grepulsiveforce=0.5",
                "-Gremincross=true",
                # "-Gbgcolor=#EDEDED",
                f"-o{file_name}.png",
                f"{file_name}.dot"]
        subprocess.run(args=args)


    def draw_flat_execution_graph(self):
        light_black = "#3B3838"
        dark_blue = "#44546A"
        orange = "#C55A11"
        red = "#873939"
        grey = "#666666"
        light_grey = "#71797E"

        self.execution_graph = pydot.Dot()

        prev_node = None
        for i,component_group in enumerate(self.execution_order):
            subgraph = pydot.Subgraph()#graph_name=f"cluster_{i}", style="dotted", penwidth=8)
            for component in component_group:
                node = pydot.Node('"' + component.id + '"')
                node.obj_dict["attributes"].update(self.system_graph_node_attribute_dict[component.id])
                subgraph.add_node(node)
                if prev_node:
                    self.add_edge_(self.execution_graph, prev_node.obj_dict["name"], node.obj_dict["name"], "")
                prev_node = node

            self.execution_graph.add_subgraph(subgraph)
        self.execution_graph.write('execution_graph.dot')

         # If Python can't find the dot executeable, change "app_path" variable to the full path
        app_path = shutil.which("dot")
        file_name = "execution_graph"
        args = [app_path,
                "-Tpng",
                "-Kdot",
                "-Nstyle=filled",
                "-Nshape=box",
                "-Nfontcolor=white",
                "-Nfontname=Times-Roman",
                "-Nfixedsize=true",
                # "-Gnodesep=3",
                "-Nnodesep=0.01",
                "-Efontname=Helvetica",
                "-Epenwidth=2",
                "-Eminlen=0.1",
                f"-Ecolor={light_grey}",
                "-Gcompound=true",
                "-Grankdir=LR", #LR
                "-Goverlap=scale",
                "-Gsplines=true",
                "-Gmargin=0",
                "-Gratio=fill",
                "-Gsize=5!",
                "-Gratio=8", #8
                "-Gpack=true",
                "-Gdpi=1000",
                "-Grepulsiveforce=0.5",
                "-o" + file_name + ".png",
                file_name + ".dot"]
        subprocess.run(args=args)


    def flatten(self, _list):
        return [item for sublist in _list for item in sublist]

    def depth_first_search_recursive(self, component):
        self.visited.add(component)

        # Recur for all the vertices
        # adjacent to this vertex
        for connection in component.connectedThrough:
            connection_point = connection.connectsSystemAt
            reciever_component = connection_point.connectionPointOf
            if reciever_component not in self.visited:
                self.depth_first_search_recursive(reciever_component)
 
        
    def depth_first_search(self, component):
        self.visited = set()
        self.depth_first_search_recursive(component)

    def get_subgraph_dict_no_cycles(self):
        self.subgraph_dict_no_cycles = copy.deepcopy(self.subgraph_dict)
        subgraphs = self.system_graph_no_cycles.get_subgraphs()
        for subgraph in subgraphs:
            if len(subgraph.get_nodes())>0:
                node = subgraph.get_nodes()[0].obj_dict["name"].replace('"',"")
                self.subgraph_dict_no_cycles[type(self.component_dict_no_cycles[node]).__name__] = subgraph


    def get_component_dict_no_cycles(self):
        self.component_dict_no_cycles = copy.deepcopy(self.component_dict)
        self.system_graph_no_cycles = copy.deepcopy(self.system_graph)
        self.get_subgraph_dict_no_cycles()

        controller_instances = [v for v in self.component_dict_no_cycles.values() if isinstance(v, Controller)]
        for controller in controller_instances:
            # controlled_component = [connection_point.connectsSystemThrough.connectsSystem for connection_point in controller.connectsAt if connection_point.reciever_property_name=="actualValue"][0]
            controlled_component = controller.controlsProperty.isPropertyOf
            # print(controlled_component)
            self.depth_first_search(controller)

            for reachable_component in self.visited:
                for connection in reachable_component.connectedThrough:
                    connection_point = connection.connectsSystemAt
                    reciever_component = connection_point.connectionPointOf
                    if controlled_component == reciever_component:
                        controlled_component.connectsAt.remove(connection_point)
                        reachable_component.connectedThrough.remove(connection)
                        self.del_edge_(self.system_graph_no_cycles, reachable_component.id, controlled_component.id)

    def map_execution_order(self):
        self.execution_order = [[self.component_dict[component.id] for component in component_group] for component_group in self.execution_order]
        
    def get_execution_order(self):
        self.get_component_dict_no_cycles()
        self.initComponents = [v for v in self.component_dict_no_cycles.values() if len(v.connectsAt)==0]
        self.activeComponents = self.initComponents
        self.execution_order = []
        while len(self.activeComponents)>0:
            self.traverse()

        self.map_execution_order()
        
        self.flat_execution_order = self.flatten(self.execution_order)
        assert len(self.flat_execution_order)==len(self.component_dict_no_cycles), f"Cycles detected in the model. Inspect the generated \"system_graph.png\" to see where."


    def traverse(self):
        activeComponentsNew = []
        self.component_group = []
        for component in self.activeComponents:
            self.component_group.append(component)
            for connection in component.connectedThrough:
                connection_point = connection.connectsSystemAt
                reciever_component = connection_point.connectionPointOf
                reciever_component.connectsAt.remove(connection_point)

                if len(reciever_component.connectsAt)==0:
                    activeComponentsNew.append(reciever_component)

        self.activeComponents = activeComponentsNew
        self.execution_order.append(self.component_group)


    def get_leaf_subsystems(self, system):
        for sub_system in system.hasSubSystem:
            if sub_system.hasSubSystem is None:
                self.leaf_subsystems.append(sub_system)
            else:
                self.get_leaf_subsystems(sub_system)


# class ModelProxy(NamespaceProxy):
#     # We need to expose the same __dunder__ methods as NamespaceProxy,
#     # in addition to the b method.
#     _exposed_ = ('__getattribute__', '__setattr__', '__delattr__', "load_model", "get_execution_order")

#     def load_model(self):
#         callmethod = object.__getattribute__(self, '_callmethod')
#         return callmethod('load_model')

#     def get_execution_order(self):
#         callmethod = object.__getattribute__(self, '_callmethod')
#         return callmethod('get_execution_order')
    






    

