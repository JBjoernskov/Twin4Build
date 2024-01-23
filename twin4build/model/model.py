import networkx as nx
import pandas as pd
import warnings
import shutil
import subprocess
import sys
import os
import copy
import pydot
import inspect
import numpy as np
import pandas as pd
import datetime
import torch
import pickle
from openpyxl import load_workbook
from dateutil.parser import parse
from twin4build.utils.get_object_attributes import get_object_attributes
from twin4build.utils.mkdir_in_root import mkdir_in_root
from twin4build.utils.rsetattr import rsetattr
from twin4build.utils.rgetattr import rgetattr
from twin4build.utils.istype import istype
from twin4build.utils.data_loaders.fiwareReader import fiwareReader
from twin4build.utils.preprocessing.data_sampler import data_sampler
from twin4build.utils.data_loaders.load_spreadsheet import sample_from_df
from twin4build.saref4syst.connection import Connection 
from twin4build.saref4syst.connection_point import ConnectionPoint
from twin4build.saref4syst.system import System
from twin4build.utils.uppath import uppath
from twin4build.utils.outdoor_environment import OutdoorEnvironmentSystem
from twin4build.utils.schedule import ScheduleSystem
from twin4build.utils.node import FlowJunctionSystem
from twin4build.utils.piecewise_linear import PiecewiseLinearSystem
from twin4build.utils.piecewise_linear_supply_water_temperature import PiecewiseLinearSupplyWaterTemperatureSystem
from twin4build.utils.on_off_system import OnOffSystem
from twin4build.utils.data_loaders.load_spreadsheet import load_spreadsheet
from twin4build.saref.measurement.measurement import Measurement
from twin4build.utils.time_series_input import TimeSeriesInputSystem
from twin4build.utils.piecewise_linear_schedule import PiecewiseLinearScheduleSystem
import twin4build.saref.property_.property_ as property_
from twin4build.saref.property_.temperature.temperature import Temperature
from twin4build.saref.property_.Co2.Co2 import Co2
from twin4build.saref.property_.opening_position.opening_position import OpeningPosition #This is in use
from twin4build.saref.property_.energy.energy import Energy #This is in use
from twin4build.saref.property_.power.power import Power #This is in use
from twin4build.saref.property_.pressure.pressure import Pressure #This is in use
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
from twin4build.saref.device.meter.meter import Meter
from twin4build.saref4bldg.physical_object.building_object.building_device.distribution_device.distribution_flow_device.flow_moving_device.pump.pump import Pump
from twin4build.saref4bldg.physical_object.building_object.building_device.shading_device.shading_device import ShadingDevice
from twin4build.saref4bldg.building_space.building_space_adjacent_system import BuildingSpaceSystem, NoSpaceModelException
from twin4build.saref4bldg.building_space.building_space_co2_system import BuildingSpaceCo2System
from twin4build.saref4bldg.physical_object.building_object.building_device.distribution_device.distribution_flow_device.energy_conversion_device.coil.coil_system_fmu import CoilSystem
from twin4build.saref4bldg.physical_object.building_object.building_device.distribution_device.distribution_flow_device.energy_conversion_device.coil.coil_heating_system import CoilHeatingSystem
from twin4build.saref4bldg.physical_object.building_object.building_device.distribution_device.distribution_flow_device.energy_conversion_device.coil.coil_cooling_system import CoilCoolingSystem
from twin4build.saref4bldg.physical_object.building_object.building_device.distribution_device.distribution_control_device.controller.controller_system import ControllerSystem
from twin4build.saref4bldg.physical_object.building_object.building_device.distribution_device.distribution_control_device.controller.controller_system_rulebased import ControllerSystemRuleBased
from twin4build.saref4bldg.physical_object.building_object.building_device.distribution_device.distribution_flow_device.energy_conversion_device.air_to_air_heat_recovery.air_to_air_heat_recovery_system import AirToAirHeatRecoverySystem
from twin4build.saref4bldg.physical_object.building_object.building_device.distribution_device.distribution_flow_device.flow_controller.damper.damper_system import DamperSystem
from twin4build.saref4bldg.physical_object.building_object.building_device.distribution_device.distribution_flow_device.flow_controller.valve.valve_system import ValveSystem
from twin4build.saref4bldg.physical_object.building_object.building_device.distribution_device.distribution_flow_device.flow_moving_device.fan.fan_system import FanSystem
from twin4build.saref4bldg.physical_object.building_object.building_device.distribution_device.distribution_flow_device.flow_terminal.space_heater.space_heater_system import SpaceHeaterSystem
from twin4build.saref.device.sensor.sensor_system import SensorSystem
from twin4build.saref.device.meter.meter_system import MeterSystem
from twin4build.saref4bldg.physical_object.building_object.building_device.shading_device.shading_device_system import ShadingDeviceSystem
from twin4build.logger.Logging import Logging
import twin4build as tb
import twin4build.base as base
import twin4build.components as components

logger = Logging.get_logger("ai_logfile")

def str2Class(str):
    return getattr(sys.modules[__name__], str)


class Model:
    def __init__(self,
                 id=None,
                saveSimulationResult=False):
        assert isinstance(id, str), f"Argument \"id\" must be of type {str(type(str))}"
        self.id = id
        self.saveSimulationResult = saveSimulationResult
        # self.system_graph = pydot.Dot()#nx.MultiDiGraph() ###
        
        # self.system_graph_rank=None #Set to string "same" to put all nodes with same class on same rank
        # self.object_graph_rank=None #Set to string "same" to put all nodes with same class on same rank
        # self.system_subgraph_dict = {}
        # self.object_subgraph_dict = {}
        # self.system_graph_node_attribute_dict = {}
        # self.system_graph_edge_label_dict = {}
        
        self._initialize_graph("system")
        self._initialize_graph("object")


        self.system_dict = {"ventilation": {},
                            "heating": {},
                            "cooling": {},
                            }
        self.component_base_dict = {} #Subset of object_dict
        self.component_dict = {} #Subset of object_dict
        self.object_dict = {}
        self.object_dict_reversed = {}
        self.object_counter_dict = {}
        self.property_dict = {}
        self.custom_initial_dict = None
        self.initial_dict = None

        self.graph_path = mkdir_in_root(folder_list=["generated_files", "graphs"])

        logger.info("[Model Class] : Exited from Initialise Function")

    def _add_edge(self, graph, a, b, sender_property_name=None, receiver_property_name=None, edge_label=None):
        if edge_label is None:
            edge_label = self.get_edge_label(sender_property_name, receiver_property_name)
            graph.add_edge(pydot.Edge(a, b, label=edge_label, tailport=sender_property_name))
        else:
            graph.add_edge(pydot.Edge(a, b, label=edge_label))

    def _del_edge(self, graph, a, b, label):
        if pydot.needs_quotes(a):
            a = f"\"{a}\""
        if pydot.needs_quotes(b):
            b = f"\"{b}\""

        edges = graph.get_edge(a, b)
        is_matched = [el.obj_dict["attributes"]["label"]==label for el in edges]
        match_idx = [i for i, x in enumerate(is_matched) if x]
        assert len(match_idx)==1, "Wrong length"
        status = graph.del_edge(a, b, match_idx[0])
        return status

    def _add_component(self, component):
        assert isinstance(component, System), f"The argument \"component\" must be of type {System.__name__}"
        if component.id not in self.component_dict:
            self.component_dict[component.id] = component

        # class_name = type(component).__name__
        # if class_name not in self.system_subgraph_dict:
        #     self.system_subgraph_dict[class_name] = pydot.Subgraph(rank=self.system_graph_rank)
        #     self.system_subgraph_dict[class_name].add_subgraph(self.system_subgraph_dict[class_name])

        # if not self.system_subgraph_dict[class_name].get_node(component.id):
        #     node = pydot.Node(component.id)
        #     self.system_subgraph_dict[class_name].add_node(node)

        self._add_object(component)

    def get_new_object_name(self, obj):
        if obj.__class__.__name__ not in self.object_counter_dict:
            self.object_counter_dict[obj.__class__.__name__] = 0

        name = f"{obj.__class__.__name__} {str(self.object_counter_dict[obj.__class__.__name__])}"
        self.object_counter_dict[obj.__class__.__name__] += 1
        return name

    def make_pickable(self):
        """
        This method is responsible to remove all references to unpickable objects for the Model instance.
        This prepares the Model instance to be used with multiprocessing in the Estimator class.
        """
        self.object_dict = {} 
        self.object_dict_reversed = {}

    def _add_object(self, obj):
        if obj in self.component_dict.values() or obj in self.component_base_dict.values():
            name = obj.id
            self.object_dict[name] = obj
            self.object_dict_reversed[obj] = name
        elif obj not in self.object_dict_reversed:
            name = self.get_new_object_name(obj)
            self.object_dict[name] = obj
            self.object_dict_reversed[obj] = name
        # else:
        #     warnings.warn(f"Cannot add object with id \"{self.object_dict_reversed[obj]}\" as it already exists in model. Skipping component.")
            

    def remove_component(self, component):
        for connection in component.connectedThrough:
            connection.connectsSystem.remove(connection)

        for connection_point in component.connectsAt:
            connection_point.connectPointOf = None
        del self.component_dict[component.id]

    def get_edge_label(self, sender_property_name, receiver_property_name):
        end_space = "          "
        edge_label = ("Out: " + sender_property_name.split("_")[0] + end_space + "\n"
                        "In: " + receiver_property_name.split("_")[0] + end_space)
        return edge_label


    def add_connection(self, sender_component, receiver_component, sender_property_name, receiver_property_name):
        '''
            It that adds a connection between two components in a system. 
            It creates a Connection object between the sender and receiver components, 
            updates their respective lists of connected components, and adds a ConnectionPoint object 
            to the receiver component's list of connection points. The function also validates that the output/input 
            property names are valid for their respective components, and updates their dictionaries of inputs/outputs 
            accordingly. Finally, it adds a labeled edge between the two components in a system graph, and adds the components 
            as nodes in their respective subgraphs.
        '''

        logger.info("[Model Class] : Entered in Add Connection Function")

        self._add_component(sender_component)
        self._add_component(receiver_component)

        sender_obj_connection = Connection(connectsSystem=sender_component, senderPropertyName=sender_property_name)
        sender_component.connectedThrough.append(sender_obj_connection)
        receiver_component_connection_point = ConnectionPoint(connectionPointOf=receiver_component, connectsSystemThrough=sender_obj_connection, receiverPropertyName=receiver_property_name)
        sender_obj_connection.connectsSystemAt = receiver_component_connection_point
        receiver_component.connectsAt.append(receiver_component_connection_point)

        exception_classes = (TimeSeriesInputSystem, FlowJunctionSystem, PiecewiseLinearSystem, PiecewiseLinearSupplyWaterTemperatureSystem, PiecewiseLinearScheduleSystem, Sensor, Meter) # These classes are exceptions because their inputs and outputs can take any form 
        if isinstance(sender_component, exception_classes):
            sender_component.output.update({sender_property_name: None})
        else:
            message = f"The property \"{sender_property_name}\" is not a valid output for the component \"{sender_component.id}\" of type \"{type(sender_component)}\".\nThe valid output properties are: {','.join(list(sender_component.output.keys()))}"
            assert sender_property_name in (set(sender_component.input.keys()) | set(sender_component.output.keys())), message
        
        if isinstance(receiver_component, exception_classes):
            receiver_component.input.update({receiver_property_name: None})
        else:
            message = f"The property \"{receiver_property_name}\" is not a valid input for the component \"{receiver_component.id}\" of type \"{type(receiver_component)}\".\nThe valid input properties are: {','.join(list(receiver_component.input.keys()))}"
            assert receiver_property_name in receiver_component.input.keys(), message

        

        self._add_graph_relation(graph=self.system_graph, sender_component=sender_component, receiver_component=receiver_component, sender_property_name=sender_property_name, receiver_property_name=receiver_property_name)
        # class_name = type(sender_component).__name__
        # if class_name not in self.system_subgraph_dict:
        #     self.system_subgraph_dict[class_name] = pydot.Subgraph(rank=self.system_graph_rank)
        # class_name = type(receiver_component).__name__
        # if class_name not in self.system_subgraph_dict:
        #     self.system_subgraph_dict[class_name] = pydot.Subgraph(rank=self.system_graph_rank)

        # self._add_edge(self.system_graph, sender_component.id, receiver_component.id, label=edge_label) ###
        # cond1 = not self.system_subgraph_dict[type(sender_component).__name__].get_node(sender_component.id)
        # cond2 = not self.system_subgraph_dict[type(sender_component).__name__].get_node("\""+ sender_component.id +"\"")
        # if cond1 and cond2:
        #     node = pydot.Node(sender_component.id)
        #     self.system_subgraph_dict[type(sender_component).__name__].add_node(node)
        # cond1 = not self.system_subgraph_dict[type(receiver_component).__name__].get_node(receiver_component.id)
        # cond2 = not self.system_subgraph_dict[type(receiver_component).__name__].get_node("\""+ receiver_component.id +"\"")
        # if cond1 and cond2:
        #     node = pydot.Node(receiver_component.id)
        #     self.system_subgraph_dict[type(receiver_component).__name__].add_node(node)
        # # self.system_graph_node_attribute_dict[sender_obj.id] = {"label": sender_obj.__class__.__name__.replace("Model","")}
        # # self.system_graph_node_attribute_dict[receiver_component.id] = {"label": receiver_component.__class__.__name__.replace("Model","")}
        # self.system_graph_node_attribute_dict[sender_component.id] = {"label": sender_component.id}
        # self.system_graph_node_attribute_dict[receiver_component.id] = {"label": receiver_component.id}
        logger.info("[Model Class] : Exited from Add Connection Function")

    def _add_graph_relation(self, graph, sender_component, receiver_component, sender_property_name=None, receiver_property_name=None, edge_label=None):
        if graph is self.system_graph:
            rank = self.system_graph_rank
            subgraph_dict = self.system_subgraph_dict
            graph_node_attribute_dict = self.system_graph_node_attribute_dict
            graph_edge_label_dict = self.system_graph_edge_label_dict
        elif graph is self.object_graph:
            rank = self.object_graph_rank
            subgraph_dict = self.object_subgraph_dict
            graph_node_attribute_dict = self.object_graph_node_attribute_dict
            graph_edge_label_dict = self.object_graph_edge_label_dict
        else:
            if isinstance(graph, pydot.Dot):
                raise ValueError("Unknown graph object. Currently implemented graph objects are \"self.system_graph\" and \"self.object_graph\"")
            else:
                raise TypeError(f"The supplied \"graph\" argument must be of type \"{pydot.Dot.__name__}\"")
        
        
        
        if sender_component not in self.component_dict.values():
            self._add_object(sender_component)

        if receiver_component not in self.component_dict.values():
            self._add_object(receiver_component)

        

        sender_class_name = type(sender_component).__name__
        receiver_class_name = type(receiver_component).__name__
        if sender_class_name not in subgraph_dict:
            subgraph_dict[sender_class_name] = pydot.Subgraph(rank=rank)
            graph.add_subgraph(subgraph_dict[sender_class_name])
        
        if receiver_class_name not in subgraph_dict:
            subgraph_dict[receiver_class_name] = pydot.Subgraph(rank=rank)
            graph.add_subgraph(subgraph_dict[receiver_class_name])
        
        sender_component_name = self.object_dict_reversed[sender_component]
        receiver_component_name = self.object_dict_reversed[receiver_component]
        self._add_edge(graph, sender_component_name, receiver_component_name, sender_property_name, receiver_property_name, edge_label) ###
        
        cond1 = not subgraph_dict[type(sender_component).__name__].get_node(sender_component_name)
        cond2 = not subgraph_dict[type(sender_component).__name__].get_node("\""+ sender_component_name +"\"")
        if cond1 and cond2:
            node = pydot.Node(sender_component_name)
            subgraph_dict[type(sender_component).__name__].add_node(node)
        
        cond1 = not subgraph_dict[type(receiver_component).__name__].get_node(receiver_component_name)
        cond2 = not subgraph_dict[type(receiver_component).__name__].get_node("\""+ receiver_component_name +"\"")
        if cond1 and cond2:
            node = pydot.Node(receiver_component_name)
            subgraph_dict[type(receiver_component).__name__].add_node(node)
        graph_node_attribute_dict[sender_component_name] = {"label": sender_component_name}
        graph_node_attribute_dict[receiver_component_name] = {"label": receiver_component_name}

    def remove_connection(self):
        """
        A method for removing connections will be implemented here.
        The method must ensure that all references to the removed node are properly removed also.
        """
        pass
    
    def add_outdoor_environment(self, filename):
        outdoor_environment = OutdoorEnvironmentSystem(
            filename=filename,
            saveSimulationResult = self.saveSimulationResult,
            id = "outdoor_environment")
        self.component_base_dict["outdoor_environment"] = outdoor_environment
        self._add_component(outdoor_environment)

    def add_supply_air_temperature_setpoint_schedule_from_csv(self, ventilation_id=None):
        logger.info("[Model Class] : Entered in add_supply_air_temperature_setpoint_schedule Function")
        stepSize = 600
        startTime = datetime.datetime(year=2021, month=12, day=10, hour=0, minute=0, second=0) #piecewise 20.5-23
        endTime = datetime.datetime(year=2022, month=2, day=15, hour=0, minute=0, second=0) #piecewise 20.5-23


        # startTime = datetime.datetime(year=2022, month=10, day=28, hour=0, minute=0, second=0) #Constant 19
        # endTime = datetime.datetime(year=2022, month=12, day=23, hour=0, minute=0, second=0) #Constant 19
        # startTime = datetime.datetime(year=2022, month=2, day=16, hour=0, minute=0, second=0) ##Commissioning piecewise 20-23
        # endTime = datetime.datetime(year=2022, month=10, day=26, hour=0, minute=0, second=0) ##Commissioning piecewise 20-23
        date_format = "%m/%d/%Y %I:%M:%S %p"
        filename = os.path.join(os.path.abspath(uppath(os.path.abspath(__file__), 2)), "test", "data", "time_series_data", "VE02_FTU1.csv")
        VE02_FTU1 = load_spreadsheet(filename=filename, stepSize=stepSize, start_time=startTime, end_time=endTime, date_format=date_format, dt_limit=9999)
        # VE02_FTU1["FTU1"] = (VE02_FTU1["FTU1"]-32)*5/9 #convert from fahrenheit to celcius
        filename = os.path.join(os.path.abspath(uppath(os.path.abspath(__file__), 2)), "test", "data", "time_series_data", "VE02_FTI_KALK_SV.csv")
        VE02_FTI_KALK_SV = load_spreadsheet(filename=filename, stepSize=stepSize, start_time=startTime, end_time=endTime, date_format=date_format, dt_limit=9999)
        # VE02_FTI_KALK_SV["FTI_KALK_SV"] = (VE02_FTI_KALK_SV["FTI_KALK_SV"]-32)*5/9 #convert from fahrenheit to celcius
        input = pd.DataFrame()
        input.insert(0, "FTU1", VE02_FTU1["FTU1"])
        input.insert(0, "FTI_KALK_SV", VE02_FTI_KALK_SV["FTI_KALK_SV"])
        input.insert(0, "time", VE02_FTI_KALK_SV["Time stamp"])
        input = input.replace([np.inf, -np.inf], np.nan).dropna()
        output = input["FTI_KALK_SV"]
        input.drop(columns=["time", "FTI_KALK_SV"], inplace=True)
        if ventilation_id is not None:
            supply_air_temperature_setpoint_schedule = PiecewiseLinearSystem(id=f"{ventilation_id} Supply air temperature setpoint", saveSimulationResult = self.saveSimulationResult)
        else:
            supply_air_temperature_setpoint_schedule = PiecewiseLinearSystem(id=f"Supply air temperature setpoint", saveSimulationResult = self.saveSimulationResult)
        supply_air_temperature_setpoint_schedule.calibrate(input=input, output=output, n_line_segments=4)
        self._add_component(supply_air_temperature_setpoint_schedule)
        logger.info("[Model Class] : Exited from add_supply_air_temperature_setpoint_schedule Function")


    def add_supply_water_temperature_setpoint_schedule_from_csv(self, heating_id=None):
        logger.info("[Model Class] : Entered in Add Supply Water Temperature Setpoint ScheduleSystem Function")

        stepSize = 600
        startTime = datetime.datetime(year=2022, month=12, day=6, hour=0, minute=0, second=0)
        endTime = datetime.datetime(year=2023, month=1, day=1, hour=0, minute=0, second=0)
        format = "%m/%d/%Y %I:%M:%S %p"
        filename = os.path.join(os.path.abspath(uppath(os.path.abspath(__file__), 2)), "test", "data", "time_series_data", "weather_BMS.csv")
        weather_BMS = load_spreadsheet(filename=filename, stepSize=stepSize, start_time=startTime, end_time=endTime, format=format, dt_limit=60)
        # weather_BMS["outdoorTemperature"] = (weather_BMS["outdoorTemperature"]-32)*5/9 #convert from fahrenheit to celcius
        filename = os.path.join(os.path.abspath(uppath(os.path.abspath(__file__), 2)), "test", "data", "time_series_data", "VA01_FTF1_SV.csv")
        VA01_FTF1_SV = load_spreadsheet(filename=filename, stepSize=stepSize, start_time=startTime, end_time=endTime, format=format, dt_limit=999999)
        # VA01["FTF1_SV"] = (VA01["FTF1_SV"]-32)*5/9 #convert from fahrenheit to celcius

        # filename = os.path.join(os.path.abspath(uppath(os.path.abspath(__file__), 2)), "test", "data", "time_series_data", "VA01.csv")
        # VA01_FTF1_SV = load_spreadsheet(filename=filename, stepSize=stepSize, start_time=startTime, end_time=endTime, format=format, dt_limit=999999)
        # VA01_FTF1_SV["FTF1_SV"] = (VA01_FTF1_SV["FTF1"]-32)*5/9 #convert from fahrenheit to celcius

        input = {"normal": pd.DataFrame(), "boost": pd.DataFrame()}
        output = {"normal": None, "boost": None}
        input["normal"].insert(0, "outdoorTemperature", weather_BMS["outdoorTemperature"])
        input["normal"].insert(0, "FTF1_SV", VA01_FTF1_SV["FTF1_SV"])
        input["normal"].insert(0, "time", weather_BMS["Time stamp"])
        input["normal"][(input["normal"]["time"].dt.hour < 10) & (input["normal"]["time"].dt.hour > 3)] = np.nan # exclude boost function, which is typically active in the excluded hours
        input["normal"] = input["normal"].replace([np.inf, -np.inf], np.nan).dropna()#.reset_index()
        output["normal"] = input["normal"]["FTF1_SV"]
        input["normal"].drop(columns=["time", "FTF1_SV"], inplace=True)
        input["boost"].insert(0, "outdoorTemperature", weather_BMS["outdoorTemperature"])
        input["boost"].insert(0, "FTF1_SV", VA01_FTF1_SV["FTF1_SV"])
        input["boost"].insert(0, "time", weather_BMS["Time stamp"])

        if heating_id is not None:
            id = f"{heating_id} Supply water temperature setpoint"
        else:
            id = f"Supply water temperature setpoint"


        supply_water_temperature_setpoint_schedule = PiecewiseLinearSystem(id=id, saveSimulationResult=self.saveSimulationResult)
        supply_water_temperature_setpoint_schedule.calibrate(input=input["normal"], output=output["normal"], n_line_segments=2)

        points = supply_water_temperature_setpoint_schedule.model.predict(input["boost"]["outdoorTemperature"])
        tol = 0.2 #degrees
        input["boost"][(input["boost"]["FTF1_SV"]-points).abs()<=tol] = np.nan
        input["boost"] = input["boost"].replace([np.inf, -np.inf], np.nan).dropna()#.reset_index()
        output["boost"] = input["boost"]["FTF1_SV"]
        input["boost"].drop(columns=["time", "FTF1_SV"], inplace=True)

        import matplotlib.pyplot as plt
        fig,ax = plt.subplots()
        fig.set_size_inches(7, 5/2)
        ax.plot(input["normal"]["outdoorTemperature"].sort_values(), supply_water_temperature_setpoint_schedule.model.predict(input["normal"]["outdoorTemperature"].sort_values()), color="blue")
        ax.scatter(input["normal"]["outdoorTemperature"], output["normal"], color="red", s=1)

        n_line_segments = {"normal": 2, "boost": 2}
        supply_water_temperature_setpoint_schedule = PiecewiseLinearSupplyWaterTemperatureSystem(id=id, saveSimulationResult = self.saveSimulationResult)
        supply_water_temperature_setpoint_schedule.calibrate(input=input, output=output, n_line_segments=n_line_segments)
        # Sort out outliers
        points = supply_water_temperature_setpoint_schedule.model["boost"].predict(input["boost"]["outdoorTemperature"])
        tol = 0.5 #degrees
        input["boost"][(output["boost"]-points).abs()>=tol] = np.nan
        input["boost"] = input["boost"].replace([np.inf, -np.inf], np.nan).dropna()#.reset_index()
        output["boost"][(output["boost"]-points).abs()>=tol] = np.nan
        output["boost"] = output["boost"].replace([np.inf, -np.inf], np.nan).dropna()#.reset_index()
        supply_water_temperature_setpoint_schedule.calibrate(input=input, output=output, n_line_segments=n_line_segments)
        self._add_component(supply_water_temperature_setpoint_schedule)

        ax.plot(input["boost"]["outdoorTemperature"].sort_values(), supply_water_temperature_setpoint_schedule.model["boost"].predict(input["boost"]["outdoorTemperature"].sort_values()), color="yellow")
        ax.scatter(input["boost"]["outdoorTemperature"], output["boost"], color="red", s=1)
        logger.info("[Model Class] : Exited from Add Supply Water Temperature Setpoint ScheduleSystem Function")

    def read_config_from_fiware(self):
        fr = fiwareReader()
        fr.read_config_from_fiware()
        self.system_dict = fr.system_dict
        self.component_base_dict = fr.component_base_dict

    def _instantiate_objects(self, df_dict):
        """
        All components listed in the configuration file are instantiated with their id.

        Arguments
        df_dict: A dictionary of dataframes read from the configuration file with sheet names as keys and dataframes as values.  
        """
        logger.info("[Model Class] : Entered in Intantiate Object Function")
        for ventilation_system_name in df_dict["System"]["Ventilation system name"].dropna():
            ventilation_system = DistributionDevice(id=ventilation_system_name)
            self.system_dict["ventilation"][ventilation_system_name] = ventilation_system
        
        for heating_system_name in df_dict["System"]["Heating system name"].dropna():
            heating_system = DistributionDevice(id=heating_system_name)
            self.system_dict["heating"][heating_system_name] = heating_system

        for cooling_system_name in df_dict["System"]["Cooling system name"].dropna():
            cooling_system = DistributionDevice(id=cooling_system_name)
            self.system_dict["cooling"][cooling_system_name] = cooling_system

        for row in df_dict["BuildingSpace"].dropna(subset=["id"]).itertuples(index=False):
            space_name = row[df_dict["BuildingSpace"].columns.get_loc("id")]
            try: 
                space = BuildingSpace(id=space_name)
                self.component_base_dict[space_name] = space
            except NoSpaceModelException:
                logger.error("No fitting space model for space " + "\"" + space_name + "\"")
                logger.error("Continuing...")
            
        for row in df_dict["Damper"].dropna(subset=["id"]).itertuples(index=False):
            damper_name = row[df_dict["Damper"].columns.get_loc("id")]
            damper = Damper(id=damper_name)
            self.component_base_dict[damper_name] = damper
            #Check that an appropriate space object exists
            if row[df_dict["Damper"].columns.get_loc("isContainedIn")] not in self.component_base_dict:
                warnings.warn("Cannot find a matching mathing BuildingSpace object for damper \"" + damper_name + "\"")                

        for row in df_dict["SpaceHeater"].dropna(subset=["id"]).itertuples(index=False):
            space_heater_name = row[df_dict["SpaceHeater"].columns.get_loc("id")]
            space_heater = SpaceHeater(id=space_heater_name)
            self.component_base_dict[space_heater_name] = space_heater
            #Check that an appropriate object exists
            if row[df_dict["SpaceHeater"].columns.get_loc("isContainedIn")] not in self.component_base_dict:
                warnings.warn("Cannot find a matching mathing SpaceHeater object for space heater \"" + space_heater_name + "\"")                

        for row in df_dict["Valve"].dropna(subset=["id"]).itertuples(index=False):
            valve_name = row[df_dict["Valve"].columns.get_loc("id")]
            valve = Valve(id=valve_name)
            self.component_base_dict[valve_name] = valve
            #Check that an appropriate object exists
            if row[df_dict["Valve"].columns.get_loc("isContainedIn")] not in self.component_base_dict:
                warnings.warn("Cannot find a matching mathing Valve object for valve \"" + valve_name + "\"")

        for row in df_dict["Coil"].dropna(subset=["id"]).itertuples(index=False):
            coil_name = row[df_dict["Coil"].columns.get_loc("id")]
            coil = Coil(id=coil_name)
            self.component_base_dict[coil_name] = coil
            
        for row in df_dict["AirToAirHeatRecovery"].dropna(subset=["id"]).itertuples(index=False):
            air_to_air_heat_recovery_name = row[df_dict["AirToAirHeatRecovery"].columns.get_loc("id")]
            air_to_air_heat_recovery = AirToAirHeatRecovery(id=air_to_air_heat_recovery_name)
            self.component_base_dict[air_to_air_heat_recovery_name] = air_to_air_heat_recovery

        for row in df_dict["Fan"].dropna(subset=["id"]).itertuples(index=False):
            fan_name = row[df_dict["Fan"].columns.get_loc("id")]
            fan = Fan(id=fan_name)
            self.component_base_dict[fan_name] = fan

        for row in df_dict["Controller"].dropna(subset=["id"]).itertuples(index=False):
            controller_name = row[df_dict["Controller"].columns.get_loc("id")]
            controller = Controller(id=controller_name)
            self.component_base_dict[controller_name] = controller
            if row[df_dict["Controller"].columns.get_loc("isContainedIn")] not in self.component_base_dict:
                warnings.warn("Cannot find a matching mathing BuildingSpace object for controller \"" + controller_name + "\"")
                
        for row in df_dict["ShadingDevice"].dropna(subset=["id"]).itertuples(index=False):
            shading_device_name = row[df_dict["ShadingDevice"].columns.get_loc("id")]
            shading_device = ShadingDevice(id=shading_device_name)
            self.component_base_dict[shading_device_name] = shading_device
            if row[df_dict["ShadingDevice"].columns.get_loc("isContainedIn")] not in self.component_base_dict:
                warnings.warn("Cannot find a matching mathing BuildingSpace object for sensor \"" + shading_device_name + "\"")                

        for row in df_dict["Sensor"].dropna(subset=["id"]).itertuples(index=False):
            sensor_name = row[df_dict["Sensor"].columns.get_loc("id")]
            sensor = Sensor(id=sensor_name)
            self.component_base_dict[sensor_name] = sensor

        for row in df_dict["Meter"].dropna(subset=["id"]).itertuples(index=False):
            meter_name = row[df_dict["Meter"].columns.get_loc("id")]
            meter = Meter(id=meter_name)
            self.component_base_dict[meter_name] = meter

        for row in df_dict["Pump"].dropna(subset=["id"]).itertuples(index=False):
            pump_name = row[df_dict["Pump"].columns.get_loc("id")]
            pump = Pump(id=pump_name)
            self.component_base_dict[pump_name] = pump
            

        for row in df_dict["Property"].dropna(subset=["id"]).itertuples(index=False):
            property_name = row[df_dict["Property"].columns.get_loc("id")]
            Property = getattr(sys.modules[__name__], row[df_dict["Property"].columns.get_loc("type")])
            property_ = Property()
            self.property_dict[property_name] = property_

        logger.info("[Model Class] : Exited from Intantiate Object Function")

    def _populate_objects(self, df_dict):
        """
        All components listed in the configuration file are populated with data and connections are defined.

        Arguments
        df_dict: A dictionary of dataframes read from the configuration file with sheet names as keys and dataframes as values.  
        """
        logger.info("[Model Class] : Entered in Populate Object Function")


        for row in df_dict["BuildingSpace"].dropna(subset=["id"]).itertuples(index=False):
            space_name = row[df_dict["BuildingSpace"].columns.get_loc("id")]
            space = self.component_base_dict[space_name]
            if isinstance(row[df_dict["BuildingSpace"].columns.get_loc("hasProperty")], str):
                properties = [self.property_dict[property_name] for property_name in row[df_dict["BuildingSpace"].columns.get_loc("hasProperty")].split(";")]
                space.hasProperty = properties
            else:
                message = f"Required property \"hasProperty\" not set for BuildingSpace object \"{space.id}\""
                raise(ValueError(message))
            
            if isinstance(row[df_dict["BuildingSpace"].columns.get_loc("connectedAfter")], str):
                connected_after = row[df_dict["BuildingSpace"].columns.get_loc("connectedAfter")].split(";")
                connected_after = [self.component_base_dict[component_name] for component_name in connected_after]
                space.connectedAfter = connected_after

            space.airVolume = row[df_dict["BuildingSpace"].columns.get_loc("airVolume")]
            
        for row in df_dict["Damper"].dropna(subset=["id"]).itertuples(index=False):
            damper_name = row[df_dict["Damper"].columns.get_loc("id")]
            damper = self.component_base_dict[damper_name]
            systems = row[df_dict["Damper"].columns.get_loc("subSystemOf")].split(";")
            systems = [system for system_dict in self.system_dict.values() for system in system_dict.values() if system.id in systems]
            damper.subSystemOf = systems

            if isinstance(row[df_dict["Damper"].columns.get_loc("connectedAfter")], str):
                connected_after = row[df_dict["Damper"].columns.get_loc("connectedAfter")].split(";")
                connected_after = [self.component_base_dict[component_name] for component_name in connected_after]
                damper.connectedAfter = connected_after

            if isinstance(row[df_dict["Damper"].columns.get_loc("hasProperty")], str):
                properties = [self.property_dict[property_name] for property_name in row[df_dict["Damper"].columns.get_loc("hasProperty")].split(";")]
                damper.hasProperty = properties
            else:
                message = f"Required property \"hasProperty\" not set for Damper object \"{damper.id}\""
                raise(ValueError(message))
            
            
            damper.isContainedIn = self.component_base_dict[row[df_dict["Damper"].columns.get_loc("isContainedIn")]]
            damper.nominalAirFlowRate = Measurement(hasValue=row[df_dict["Damper"].columns.get_loc("nominalAirFlowRate")])
            
        for row in df_dict["SpaceHeater"].dropna(subset=["id"]).itertuples(index=False):
            space_heater_name = row[df_dict["SpaceHeater"].columns.get_loc("id")]
            space_heater = self.component_base_dict[space_heater_name]

            if isinstance(row[df_dict["SpaceHeater"].columns.get_loc("subSystemOf")], str):
                systems = row[df_dict["SpaceHeater"].columns.get_loc("subSystemOf")].split(";")
                systems = [system for system_dict in self.system_dict.values() for system in system_dict.values() if system.id in systems]
                space_heater.subSystemOf = systems
            else:
                message = f"Required property \"subSystemOf\" not set for SpaceHeater object \"{space_heater.id}\""
                raise(ValueError(message))

            if isinstance(row[df_dict["SpaceHeater"].columns.get_loc("connectedAfter")], str):
                connected_after = row[df_dict["SpaceHeater"].columns.get_loc("connectedAfter")].split(";")
                connected_after = [self.component_base_dict[component_name] for component_name in connected_after]
                space_heater.connectedAfter = connected_after
            else:
                message = f"Required property \"connectedAfter\" not set for SpaceHeater object \"{space_heater.id}\""
                raise(ValueError(message))

            properties = [self.property_dict[property_name] for property_name in row[df_dict["SpaceHeater"].columns.get_loc("hasProperty")].split(";")]
            space_heater.hasProperty = properties
            
            space_heater.isContainedIn = self.component_base_dict[row[df_dict["SpaceHeater"].columns.get_loc("isContainedIn")]]
            space_heater.outputCapacity = Measurement(hasValue=row[df_dict["SpaceHeater"].columns.get_loc("outputCapacity")])
            space_heater.temperatureClassification = row[df_dict["SpaceHeater"].columns.get_loc("temperatureClassification")]
            space_heater.thermalMassHeatCapacity = Measurement(hasValue=row[df_dict["SpaceHeater"].columns.get_loc("thermalMassHeatCapacity")])

        for row in df_dict["Valve"].dropna(subset=["id"]).itertuples(index=False):
            valve_name = row[df_dict["Valve"].columns.get_loc("id")]
            valve = self.component_base_dict[valve_name]

            if isinstance(row[df_dict["Valve"].columns.get_loc("subSystemOf")], str):
                systems = row[df_dict["Valve"].columns.get_loc("subSystemOf")].split(";")
                systems = [system for system_dict in self.system_dict.values() for system in system_dict.values() if system.id in systems]
                valve.subSystemOf = systems
            else:
                message = f"Required property \"subSystemOf\" not set for Valve object \"{valve.id}\""
                raise(ValueError(message))

            if isinstance(row[df_dict["Valve"].columns.get_loc("connectedAfter")], str):
                connected_after = row[df_dict["Valve"].columns.get_loc("connectedAfter")].split(";")
                connected_after = [self.component_base_dict[component_name] for component_name in connected_after]
                valve.connectedAfter = connected_after

            if isinstance(row[df_dict["Valve"].columns.get_loc("hasProperty")], str):
                properties = [self.property_dict[property_name] for property_name in row[df_dict["Valve"].columns.get_loc("hasProperty")].split(";")]
                valve.hasProperty = properties

            if isinstance(row[df_dict["Valve"].columns.get_loc("isContainedIn")], str):
                valve.isContainedIn = self.component_base_dict[row[df_dict["Valve"].columns.get_loc("isContainedIn")]]
            
            
            if isinstance(row[df_dict["Valve"].columns.get_loc("flowCoefficient")], str):
                valve.flowCoefficient = Measurement(hasValue=row[df_dict["Valve"].columns.get_loc("flowCoefficient")])
            
            if isinstance(row[df_dict["Valve"].columns.get_loc("testPressure")], str):
                valve.testPressure = Measurement(hasValue=row[df_dict["Valve"].columns.get_loc("testPressure")])

        for row in df_dict["Coil"].dropna(subset=["id"]).itertuples(index=False):
            coil_name = row[df_dict["Coil"].columns.get_loc("id")]
            coil = self.component_base_dict[coil_name]

            if isinstance(row[df_dict["Coil"].columns.get_loc("subSystemOf")], str):
                systems = row[df_dict["Coil"].columns.get_loc("subSystemOf")].split(";")
                systems = [system for system_dict in self.system_dict.values() for system in system_dict.values() if system.id in systems]
                has_ventilation_system = any([system in self.system_dict["ventilation"].values() for system in systems])
                has_heating_system = any([system in self.system_dict["heating"].values() for system in systems])
                has_cooling_system = any([system in self.system_dict["cooling"].values() for system in systems])
                assert has_ventilation_system and (has_heating_system or has_cooling_system), f"Required property \"subSystemOf\" must contain both a Ventilation system and either a Heating or Cooling system for Coil object \"{coil.id}\""
                coil.subSystemOf = systems
            else:
                message = f"Required property \"subSystemOf\" not set for Coil object \"{coil.id}\""
                raise(ValueError(message))
            
            if isinstance(row[df_dict["Coil"].columns.get_loc("connectedAfter")], str):
                connected_after = row[df_dict["Coil"].columns.get_loc("connectedAfter")].split(";")
                connected_after = [self.component_base_dict[component_name] for component_name in connected_after]
                coil.connectedAfter = connected_after

            if isinstance(row[df_dict["Coil"].columns.get_loc("hasProperty")], str):
                properties = [self.property_dict[property_name] for property_name in row[df_dict["Coil"].columns.get_loc("hasProperty")].split(";")]
                coil.hasProperty = properties
            
        for row in df_dict["AirToAirHeatRecovery"].dropna(subset=["id"]).itertuples(index=False):
            air_to_air_heat_recovery_name = row[df_dict["AirToAirHeatRecovery"].columns.get_loc("id")]
            air_to_air_heat_recovery = self.component_base_dict[air_to_air_heat_recovery_name]
            systems = row[df_dict["AirToAirHeatRecovery"].columns.get_loc("subSystemOf")].split(";")
            systems = [system for system_dict in self.system_dict.values() for system in system_dict.values() if system.id in systems]
            air_to_air_heat_recovery.subSystemOf = systems

            if isinstance(row[df_dict["AirToAirHeatRecovery"].columns.get_loc("connectedAfter")], str):
                connected_after = row[df_dict["AirToAirHeatRecovery"].columns.get_loc("connectedAfter")].split(";")
                connected_after = [self.component_base_dict[component_name] for component_name in connected_after]
                air_to_air_heat_recovery.connectedAfter = connected_after
            else:
                message = f"Required property \"connectedAfter\" not set for AirToAirHeatRecovery object \"{air_to_air_heat_recovery.id}\""
                raise(ValueError(message))
            properties = [self.property_dict[property_name] for property_name in row[df_dict["AirToAirHeatRecovery"].columns.get_loc("hasProperty")].split(";")]
            air_to_air_heat_recovery.hasProperty = properties
            air_to_air_heat_recovery.primaryAirFlowRateMax = Measurement(hasValue=row[df_dict["AirToAirHeatRecovery"].columns.get_loc("primaryAirFlowRateMax")])
            air_to_air_heat_recovery.secondaryAirFlowRateMax = Measurement(hasValue=row[df_dict["AirToAirHeatRecovery"].columns.get_loc("secondaryAirFlowRateMax")])

        for row in df_dict["Fan"].dropna(subset=["id"]).itertuples(index=False):
            fan_name = row[df_dict["Fan"].columns.get_loc("id")]
            fan = self.component_base_dict[fan_name]

            if isinstance(row[df_dict["Fan"].columns.get_loc("subSystemOf")], str):
                systems = row[df_dict["Fan"].columns.get_loc("subSystemOf")].split(";")
                systems = [system for system_dict in self.system_dict.values() for system in system_dict.values() if system.id in systems]
                fan.subSystemOf = systems
            else:
                message = f"Required property \"subSystemOf\" not set for fan object \"{fan.id}\""
                raise(ValueError(message))

            if isinstance(row[df_dict["Fan"].columns.get_loc("connectedAfter")], str):
                connected_after = row[df_dict["Fan"].columns.get_loc("connectedAfter")].split(";")
                connected_after = [self.component_base_dict[component_name] for component_name in connected_after]
                fan.connectedAfter = connected_after
            # else:
            #     message = f"Required property \"connectedAfter\" not set for fan object \"{fan.id}\""
            #     raise(ValueError(message))
            
            if isinstance(row[df_dict["Fan"].columns.get_loc("hasProperty")], str):
                properties = [self.property_dict[property_name] for property_name in row[df_dict["Fan"].columns.get_loc("hasProperty")].split(";")]
                fan.hasProperty = properties
            # else:
            #     message = f"Required property \"hasProperty\" not set for fan object \"{fan.id}\""
            #     raise(ValueError(message))
            fan.nominalAirFlowRate = Measurement(hasValue=row[df_dict["Fan"].columns.get_loc("nominalAirFlowRate")])
            fan.nominalPowerRate = Measurement(hasValue=row[df_dict["Fan"].columns.get_loc("nominalPowerRate")])

            
        for row in df_dict["Controller"].dropna(subset=["id"]).itertuples(index=False):
            controller_name = row[df_dict["Controller"].columns.get_loc("id")]
            controller = self.component_base_dict[controller_name]

            if isinstance(row[df_dict["Controller"].columns.get_loc("subSystemOf")], str):
                systems = row[df_dict["Controller"].columns.get_loc("subSystemOf")].split(";")
                systems = [system for system_dict in self.system_dict.values() for system in system_dict.values() if system.id in systems]
            else:
                message = f"Required property \"subSystemOf\" not set for controller object \"{controller.id}\""
                raise(ValueError(message))
            
            controller.subSystemOf = systems

            if isinstance(row[df_dict["Controller"].columns.get_loc("isContainedIn")], str):
                controller.isContainedIn = self.component_base_dict[row[df_dict["Controller"].columns.get_loc("isContainedIn")]]
            
            _property = self.property_dict[row[df_dict["Controller"].columns.get_loc("controlsProperty")]]
            controller.controlsProperty = _property


            if "actuatesProperty" not in df_dict["Controller"].columns:
                warnings.warn("The property \"actuatesProperty\" is not found in \"Controller\" sheet. This is ignored for now but will raise an error in the future. It probably is caused by using an outdated configuration file.")
            else:
                if isinstance(row[df_dict["Controller"].columns.get_loc("actuatesProperty")], str):
                    _property = self.property_dict[row[df_dict["Controller"].columns.get_loc("actuatesProperty")]]
                    controller.actuatesProperty = _property
                else:
                    message = f"Required property \"actuatesProperty\" not set for controller object \"{controller.id}\""
                    raise(ValueError(message))

 
        for row in df_dict["ShadingDevice"].dropna(subset=["id"]).itertuples(index=False):
            shading_device_name = row[df_dict["ShadingDevice"].columns.get_loc("id")]
            shading_device = self.component_base_dict[shading_device_name]

            if isinstance(row[df_dict["ShadingDevice"].columns.get_loc("subSystemOf")], str):
                systems = row[df_dict["ShadingDevice"].columns.get_loc("subSystemOf")].split(";")
                systems = [system for system_dict in self.system_dict.values() for system in system_dict.values() if system.id in systems]
                shading_device.subSystemOf = systems
            else:
                message = f"Required property \"subSystemOf\" not set for shading_device object \"{shading_device.id}\""
                raise(ValueError(message))

            properties = [self.property_dict[property_name] for property_name in row[df_dict["ShadingDevice"].columns.get_loc("hasProperty")].split(";")]
            shading_device.isContainedIn = self.component_base_dict[row[df_dict["ShadingDevice"].columns.get_loc("isContainedIn")]]
            shading_device.hasProperty = properties
      
        for row in df_dict["Sensor"].dropna(subset=["id"]).itertuples(index=False):
            sensor_name = row[df_dict["Sensor"].columns.get_loc("id")]
            sensor = self.component_base_dict[sensor_name]

            if isinstance(row[df_dict["Sensor"].columns.get_loc("measuresProperty")], str):
                properties = self.property_dict[row[df_dict["Sensor"].columns.get_loc("measuresProperty")]]
                sensor.measuresProperty = properties
            else:
                message = f"Required property \"measuresProperty\" not set for Sensor object \"{sensor.id}\""
                raise(ValueError(message))
            
            if isinstance(row[df_dict["Sensor"].columns.get_loc("isContainedIn")], str):
                sensor.isContainedIn = self.component_base_dict[row[df_dict["Sensor"].columns.get_loc("isContainedIn")]]
            
            if isinstance(row[df_dict["Sensor"].columns.get_loc("connectedAfter")], str):
                connected_after = row[df_dict["Sensor"].columns.get_loc("connectedAfter")].split(";")
                connected_after = [self.component_base_dict[component_name] for component_name in connected_after]
                sensor.connectedAfter = connected_after

            if isinstance(row[df_dict["Sensor"].columns.get_loc("subSystemOf")], str):
                systems = row[df_dict["Sensor"].columns.get_loc("subSystemOf")].split(";")
                systems = [system for system_dict in self.system_dict.values() for system in system_dict.values() if system.id in systems]
                sensor.subSystemOf = systems
 
        for row in df_dict["Meter"].dropna(subset=["id"]).itertuples(index=False):
            meter_name = row[df_dict["Meter"].columns.get_loc("id")]
            meter = self.component_base_dict[meter_name]
            if isinstance(row[df_dict["Meter"].columns.get_loc("measuresProperty")], str):
                properties = self.property_dict[row[df_dict["Meter"].columns.get_loc("measuresProperty")]]
                meter.measuresProperty = properties
            else:
                message = f"Required property \"measuresProperty\" not set for Sensor object \"{sensor.id}\""
                raise(ValueError(message))
            
            if isinstance(row[df_dict["Meter"].columns.get_loc("isContainedIn")], str):
                meter.isContainedIn = self.component_base_dict[row[df_dict["Meter"].columns.get_loc("isContainedIn")]]

            if isinstance(row[df_dict["Meter"].columns.get_loc("connectedAfter")], str):
                connected_after = row[df_dict["Meter"].columns.get_loc("connectedAfter")].split(";")
                connected_after = [self.component_base_dict[component_name] for component_name in connected_after]
                meter.connectedAfter = connected_after

            if isinstance(row[df_dict["Meter"].columns.get_loc("subSystemOf")], str):
                systems = row[df_dict["Meter"].columns.get_loc("subSystemOf")].split(";")
                systems = [system for system_dict in self.system_dict.values() for system in system_dict.values() if system.id in systems]
                meter.subSystemOf = systems

        for row in df_dict["Pump"].dropna(subset=["id"]).itertuples(index=False):
            pump_name = row[df_dict["Pump"].columns.get_loc("id")]
            pump = self.component_base_dict[pump_name]

            if isinstance(row[df_dict["Pump"].columns.get_loc("subSystemOf")], str):
                systems = row[df_dict["Pump"].columns.get_loc("subSystemOf")].split(";")
                systems = [system for system_dict in self.system_dict.values() for system in system_dict.values() if system.id in systems]
                pump.subSystemOf = systems
            else:
                message = f"Required property \"subSystemOf\" not set for Pump object \"{pump.id}\""
                raise(ValueError(message))
            
            if isinstance(row[df_dict["Pump"].columns.get_loc("connectedAfter")], str):
                connected_after = row[df_dict["Pump"].columns.get_loc("connectedAfter")].split(";")
                connected_after = [self.component_base_dict[component_name] for component_name in connected_after]
                pump.connectedAfter = connected_after
            else:
                message = f"Required property \"connectedAfter\" not set for Pump object \"{pump.id}\""
                raise(ValueError(message))
            
            if isinstance(row[df_dict["Pump"].columns.get_loc("hasProperty")], str):
                pump.connectedAfter = row[df_dict["Pump"].columns.get_loc("hasProperty")]


        logger.info("[Model Class] : Exited from Populate Object Function")

                        

    def read_datamodel_config(self, semantic_model_filename):
        '''
            This is a method that reads a configuration file in the Excel format, 
            and instantiates and populates objects based on the information in the file. 
            The method reads various sheets in the Excel file and stores the data in separate 
            pandas dataframes, one for each sheet. Then, it calls two other methods, _instantiate_objects 
            and _populate_objects, to create and populate objects based on the data in the dataframes.        
        '''

        logger.info("[Model Class] : Entered in read_config Function")
        wb = load_workbook(semantic_model_filename, read_only=True)
        df_Systems = pd.read_excel(semantic_model_filename, sheet_name="System") if 'System' in wb.sheetnames else pd.DataFrame([np.nan], columns=["id"])
        df_Space = pd.read_excel(semantic_model_filename, sheet_name="BuildingSpace") if 'BuildingSpace' in wb.sheetnames else pd.DataFrame([np.nan], columns=["id"])
        df_Damper = pd.read_excel(semantic_model_filename, sheet_name="Damper") if 'Damper' in wb.sheetnames else pd.DataFrame([np.nan], columns=["id"])
        df_SpaceHeater = pd.read_excel(semantic_model_filename, sheet_name="SpaceHeater") if 'SpaceHeater' in wb.sheetnames else pd.DataFrame([np.nan], columns=["id"])
        df_Valve = pd.read_excel(semantic_model_filename, sheet_name="Valve") if 'Valve' in wb.sheetnames else pd.DataFrame([np.nan], columns=["id"])
        df_Coil = pd.read_excel(semantic_model_filename, sheet_name="Coil") if 'Coil' in wb.sheetnames else pd.DataFrame([np.nan], columns=["id"])
        df_AirToAirHeatRecovery = pd.read_excel(semantic_model_filename, sheet_name="AirToAirHeatRecovery") if 'AirToAirHeatRecovery' in wb.sheetnames else pd.DataFrame([np.nan], columns=["id"])
        df_Fan = pd.read_excel(semantic_model_filename, sheet_name="Fan") if 'Fan' in wb.sheetnames else pd.DataFrame([np.nan], columns=["id"])
        df_Controller = pd.read_excel(semantic_model_filename, sheet_name="Controller") if 'Controller' in wb.sheetnames else pd.DataFrame([np.nan], columns=["id"])
        df_ShadingDevice = pd.read_excel(semantic_model_filename, sheet_name="ShadingDevice") if 'ShadingDevice' in wb.sheetnames else pd.DataFrame([np.nan], columns=["id"])
        df_Sensor = pd.read_excel(semantic_model_filename, sheet_name="Sensor") if 'Sensor' in wb.sheetnames else pd.DataFrame([np.nan], columns=["id"])
        df_Meter = pd.read_excel(semantic_model_filename, sheet_name="Meter") if 'Meter' in wb.sheetnames else pd.DataFrame([np.nan], columns=["id"])
        df_Pump = pd.read_excel(semantic_model_filename, sheet_name="Pump") if 'Pump' in wb.sheetnames else pd.DataFrame([np.nan], columns=["id"])
        df_Property = pd.read_excel(semantic_model_filename, sheet_name="Property") if 'Property' in wb.sheetnames else pd.DataFrame([np.nan], columns=["id"])

        df_dict = {"System": df_Systems,
                   "BuildingSpace": df_Space,
                   "Damper": df_Damper,
                   "SpaceHeater": df_SpaceHeater,
                   "Valve": df_Valve,
                   "Coil": df_Coil,
                   "AirToAirHeatRecovery": df_AirToAirHeatRecovery,
                   "Fan": df_Fan,
                   "Controller": df_Controller,
                   "ShadingDevice": df_ShadingDevice,
                   "Sensor": df_Sensor,
                   "Meter": df_Meter,
                   "Pump": df_Pump,
                   "Property": df_Property}

        self._instantiate_objects(df_dict)
        self._populate_objects(df_dict)
        logger.info("[Model Class] : Exited from read_config Function")
        
    def read_input_config(self, input_dict):
        """
        This method reads from an input dictionary and populates the corresponding objects.
        """
        logger.info("[Model Class] : Entered in read_input_config Function")

        time_format = '%Y-%m-%d %H:%M:%S%z'
        startTime = datetime.datetime.strptime(input_dict["metadata"]["start_time"], time_format)
        endTime = datetime.datetime.strptime(input_dict["metadata"]["end_time"], time_format)
        stepSize = input_dict["metadata"]['stepSize']
        sensor_inputs = input_dict["inputs_sensor"] #Change naming to be consistent
        schedule_inputs = input_dict["input_schedules"] #Change naming to be consistent
        weather_inputs = sensor_inputs["ml_inputs_dmi"]
        
        df_raw = pd.DataFrame()
        df_raw.insert(0, "datetime", weather_inputs["observed"])
        df_raw.insert(1, "outdoorTemperature", weather_inputs["temp_dry"])
        df_raw.insert(2, "globalIrradiation", weather_inputs["radia_glob"])
        df_sample = sample_from_df(df_raw,
                                    stepSize=stepSize,
                                    start_time=startTime,
                                    end_time=endTime,
                                    resample=True,
                                    clip=True,
                                    tz="Europe/Copenhagen",
                                    preserve_order=True)

        outdoor_environment = OutdoorEnvironmentSystem(df_input=df_sample,
                                                        saveSimulationResult = self.saveSimulationResult,
                                                        id = "outdoor_environment")

        # Initialize the room temperature 

        '''
            MODIFIED BY NEC-INDIA

            temperature_list = sensor_inputs["ml_inputs"]["temperature"]

            # take the first occurence of the value in the list 
            # if the [0] is 'None' then iterate for the next value till end 
            # if temperature_list has all 'None' then default temperature taken as 21
        '''
        if sensor_inputs["ml_inputs"]["temperature"][0]=="None":
            initial_temperature = 21
        else:
            initial_temperature = float(sensor_inputs["ml_inputs"]["temperature"][0])
        custom_initial_dict = {"OE20-601b-2": {"indoorTemperature": initial_temperature}}
        self.set_custom_initial_dict(custom_initial_dict)

        indoor_temperature_setpoint_schedule = ScheduleSystem(
            **schedule_inputs["temperature_setpoint_schedule"],
            add_noise = False,
            saveSimulationResult = True,
            id = "OE20-601b-2| Temperature setpoint schedule")

        occupancy_schedule = ScheduleSystem(
            **schedule_inputs["occupancy_schedule"],
            add_noise = True,
            saveSimulationResult = True,
            id = "OE20-601b-2| Occupancy schedule")

        supply_water_temperature_setpoint_schedule = PiecewiseLinearScheduleSystem(
            **schedule_inputs["supply_water_temperature_schedule_pwlf"],
            saveSimulationResult = True,
            id = "Heating system| Supply water temperature schedule")
        
        supply_air_temperature_schedule = ScheduleSystem(
            **schedule_inputs["supply_air_temperature_schedule"],
            saveSimulationResult = True,
            id = "Ventilation system| Supply air temperature schedule")

        self._add_component(outdoor_environment)
        self._add_component(occupancy_schedule)
        self._add_component(indoor_temperature_setpoint_schedule)
        self._add_component(supply_water_temperature_setpoint_schedule)
        self._add_component(supply_air_temperature_schedule)

        logger.info("[Model Class] : Exited from read_input_config Function")

    def parse_semantic_model(self):
        space_instances = self.get_component_by_class(self.component_base_dict, BuildingSpace)
        damper_instances = self.get_component_by_class(self.component_base_dict, Damper)
        space_heater_instances = self.get_component_by_class(self.component_base_dict, SpaceHeater)
        valve_instances = self.get_component_by_class(self.component_base_dict, Valve)
        coil_instances = self.get_component_by_class(self.component_base_dict, Coil)
        air_to_air_heat_recovery_instances = self.get_component_by_class(self.component_base_dict, AirToAirHeatRecovery)
        fan_instances = self.get_component_by_class(self.component_base_dict, Fan)
        controller_instances = self.get_component_by_class(self.component_base_dict, Controller)
        shading_device_instances = self.get_component_by_class(self.component_base_dict, ShadingDevice)
        sensor_instances = self.get_component_by_class(self.component_base_dict, Sensor)
        meter_instances = self.get_component_by_class(self.component_base_dict, Meter)

        for space in space_instances:
            for property_ in space.hasProperty:
                property_.isPropertyOf = space
            for component in space.connectedAfter:
                component.connectedBefore.append(space)
            
        for damper in damper_instances:
            damper.isContainedIn.contains.append(damper)
            for system in damper.subSystemOf:
                system.hasSubSystem.append(damper)
            for property_ in damper.hasProperty:
                property_.isPropertyOf = damper
            for component in damper.connectedAfter:
                component.connectedBefore.append(damper)

        for space_heater in space_heater_instances:
            for component in space_heater.connectedAfter:
                component.connectedBefore.append(space_heater)
            space_heater.isContainedIn.contains.append(space_heater)
            for system in space_heater.subSystemOf:
                system.hasSubSystem.append(space_heater)
            for property_ in space_heater.hasProperty:
                property_.isPropertyOf = space_heater

        for valve in valve_instances:
            if valve.isContainedIn is not None:
                valve.isContainedIn.contains.append(valve)
            for system in valve.subSystemOf:
                system.hasSubSystem.append(valve)
            for property_ in valve.hasProperty:
                property_.isPropertyOf = valve
            for component in valve.connectedAfter:
                component.connectedBefore.append(valve)

        for coil in coil_instances:
            for system in coil.subSystemOf:
                system.hasSubSystem.append(coil)
            for property_ in coil.hasProperty:
                property_.isPropertyOf = coil
            for component in coil.connectedAfter:
                component.connectedBefore.append(coil)

        for air_to_air_heat_recovery in air_to_air_heat_recovery_instances:
            for system in air_to_air_heat_recovery.subSystemOf:
                system.hasSubSystem.append(air_to_air_heat_recovery)
            for property_ in air_to_air_heat_recovery.hasProperty:
                property_.isPropertyOf = air_to_air_heat_recovery
            for component in air_to_air_heat_recovery.connectedAfter:
                component.connectedBefore.append(air_to_air_heat_recovery)

        for fan in fan_instances:
            for system in fan.subSystemOf:
                system.hasSubSystem.append(fan)
            for property_ in fan.hasProperty:
                property_.isPropertyOf = fan
            for component in fan.connectedAfter:
                component.connectedBefore.append(fan)

        for controller in controller_instances:
            if controller.isContainedIn is not None:
                controller.isContainedIn.contains.append(controller)
            controller.controlsProperty.isControlledByDevice = controller
            controller.actuatesProperty.isActuatedByDevice = controller
            for system in controller.subSystemOf:
                system.hasSubSystem.append(controller)

        for shading_device in shading_device_instances:
            shading_device.isContainedIn.contains.append(shading_device)
            for system in shading_device.subSystemOf:
                system.hasSubSystem.append(shading_device)
            for property_ in shading_device.hasProperty:
                property_.isPropertyOf = shading_device

        for sensor in sensor_instances:
            if sensor.isContainedIn is not None:
                sensor.isContainedIn.contains.append(sensor)
            sensor.measuresProperty.isMeasuredByDevice = sensor
            for system in sensor.subSystemOf:
                system.hasSubSystem.append(sensor)
            for component in sensor.connectedAfter:
                component.connectedBefore.append(sensor)

        for meter in meter_instances:
            if meter.isContainedIn is not None:
                meter.isContainedIn.contains.append(meter)
            meter.measuresProperty.isMeasuredByDevice = meter
            for system in meter.subSystemOf:
                system.hasSubSystem.append(meter)
            for component in meter.connectedAfter:
                component.connectedBefore.append(meter)

        #Add systems
        for heating_system in self.system_dict["heating"].values():
            self._add_object(heating_system)

        for cooling_system in self.system_dict["cooling"].values():
            self._add_object(cooling_system)

        for ventilation_system in self.system_dict["ventilation"].values():
            self._add_object(ventilation_system)
        
        logger.info("[Model Class] : Exited from Apply Model Extensions Function")
        
    def apply_model_extensions(self):
        logger.info("[Model Class] : Entered in Apply Model Extensions Function")
        space_instances = self.get_component_by_class(self.component_base_dict, BuildingSpace)
        damper_instances = self.get_component_by_class(self.component_base_dict, Damper)
        space_heater_instances = self.get_component_by_class(self.component_base_dict, SpaceHeater)
        valve_instances = self.get_component_by_class(self.component_base_dict, Valve)
        coil_instances = self.get_component_by_class(self.component_base_dict, Coil)
        air_to_air_heat_recovery_instances = self.get_component_by_class(self.component_base_dict, AirToAirHeatRecovery)
        fan_instances = self.get_component_by_class(self.component_base_dict, Fan)
        controller_instances = self.get_component_by_class(self.component_base_dict, Controller)
        shading_device_instances = self.get_component_by_class(self.component_base_dict, ShadingDevice)
        sensor_instances = self.get_component_by_class(self.component_base_dict, Sensor)
        meter_instances = self.get_component_by_class(self.component_base_dict, Meter)

        for space in space_instances:
            base_kwargs = self.get_object_properties(space)
            extension_kwargs = {
                "saveSimulationResult": self.saveSimulationResult,
            }
            base_kwargs.update(extension_kwargs)
            space = BuildingSpaceSystem(**base_kwargs)
            self._add_component(space)
            for property_ in space.hasProperty:
                property_.isPropertyOf = space
            for component in space.connectedAfter:
                component.connectedBefore.append(space)
            
        for damper in damper_instances:
            base_kwargs = self.get_object_properties(damper)
            extension_kwargs = {
                "a": 1,
                "saveSimulationResult": self.saveSimulationResult,
            }
            base_kwargs.update(extension_kwargs)
            damper = DamperSystem(**base_kwargs)
            self._add_component(damper)
            damper.isContainedIn = self.component_dict[damper.isContainedIn.id]
            damper.isContainedIn.contains.append(damper)
            for system in damper.subSystemOf:
                system.hasSubSystem.append(damper)
            for property_ in damper.hasProperty:
                property_.isPropertyOf = damper
            for component in damper.connectedAfter:
                component.connectedBefore.append(damper)

        for space_heater in space_heater_instances:
            base_kwargs = self.get_object_properties(space_heater)
            extension_kwargs = {
                "saveSimulationResult": self.saveSimulationResult,
            }
            base_kwargs.update(extension_kwargs)
            space_heater = SpaceHeaterSystem(**base_kwargs)
            space_heater.heatTransferCoefficient = 8.31495759e+01
            space_heater.thermalMassHeatCapacity.hasvalue = 2.72765272e+06
            for component in space_heater.connectedAfter:
                component.connectedBefore.append(space_heater)

            self._add_component(space_heater)
            space_heater.isContainedIn = self.component_dict[space_heater.isContainedIn.id]
            space_heater.isContainedIn.contains.append(space_heater)
            for system in space_heater.subSystemOf:
                system.hasSubSystem.append(space_heater)
            for property_ in space_heater.hasProperty:
                property_.isPropertyOf = space_heater

        for valve in valve_instances:
            base_kwargs = self.get_object_properties(valve)
            extension_kwargs = {
                "waterFlowRateMax": 0.0202,
                "valveAuthority": 1.,
                "saveSimulationResult": self.saveSimulationResult,
            }
            base_kwargs.update(extension_kwargs)
            valve = ValveSystem(**base_kwargs)
            self._add_component(valve)
            valve.isContainedIn = self.component_dict[valve.isContainedIn.id]
            valve.isContainedIn.contains.append(valve)
            for system in valve.subSystemOf:
                system.hasSubSystem.append(valve)
            for property_ in valve.hasProperty:
                property_.isPropertyOf = valve
            for component in valve.connectedAfter:
                component.connectedBefore.append(valve)

        for coil in coil_instances:
            base_kwargs = self.get_object_properties(coil)
            extension_kwargs = {
                "saveSimulationResult": self.saveSimulationResult,
            }
            base_kwargs.update(extension_kwargs)
            if len([v for v in coil.subSystemOf if v in self.system_dict["heating"].values()])==1:
                coil = CoilHeatingSystem(**base_kwargs)
            elif len([v for v in coil.subSystemOf if v in self.system_dict["cooling"].values()])==1:
                coil = CoilCoolingSystem(**base_kwargs)
            else:
                raise(ValueError(f"The system of the Coil with id \"{coil.id}\" is not set."))
            self._add_component(coil)
            for system in coil.subSystemOf:
                system.hasSubSystem.append(coil)
            for property_ in coil.hasProperty:
                property_.isPropertyOf = coil
            for component in coil.connectedAfter:
                component.connectedBefore.append(coil)

        for air_to_air_heat_recovery in air_to_air_heat_recovery_instances:
            base_kwargs = self.get_object_properties(air_to_air_heat_recovery)
            extension_kwargs = {
                "specificHeatCapacityAir": Measurement(hasValue=1000),
                "eps_75_h": 0.84918046,
                "eps_75_c": 0.82754917,
                "eps_100_h": 0.85202735,
                "eps_100_c": 0.8215695,
                "saveSimulationResult": self.saveSimulationResult,
            }
            base_kwargs.update(extension_kwargs)
            air_to_air_heat_recovery = AirToAirHeatRecoverySystem(**base_kwargs)
            self._add_component(air_to_air_heat_recovery)
            for system in air_to_air_heat_recovery.subSystemOf:
                system.hasSubSystem.append(air_to_air_heat_recovery)
            for property_ in air_to_air_heat_recovery.hasProperty:
                property_.isPropertyOf = air_to_air_heat_recovery
            for component in air_to_air_heat_recovery.connectedAfter:
                component.connectedBefore.append(air_to_air_heat_recovery)

        for fan in fan_instances:
            base_kwargs = self.get_object_properties(fan)
            extension_kwargs = {
                "c1": 0.027828,
                "c2": 0.026583,
                "c3": -0.087069,
                "c4": 1.030920,
                "saveSimulationResult": self.saveSimulationResult,
            }
            base_kwargs.update(extension_kwargs)
            fan = FanSystem(**base_kwargs)
            self._add_component(fan)
            for system in fan.subSystemOf:
                system.hasSubSystem.append(fan)
            for property_ in fan.hasProperty:
                property_.isPropertyOf = fan
            for component in fan.connectedAfter:
                component.connectedBefore.append(fan)

        for controller in controller_instances:
            base_kwargs = self.get_object_properties(controller)
            if isinstance(controller.controlsProperty, Temperature):
                K_i = 2.50773924e-01
                K_p = 4.38174242e-01
                K_d = 0
                extension_kwargs = {
                    "K_p": K_p,
                    "K_i": K_i,
                    "K_d": K_d,
                    "saveSimulationResult": self.saveSimulationResult,
                }
                base_kwargs.update(extension_kwargs)
                controller = ControllerSystem(**base_kwargs)
            elif isinstance(controller.controlsProperty, Co2):
                extension_kwargs = {
                    "saveSimulationResult": self.saveSimulationResult,
                }
                base_kwargs.update(extension_kwargs)
                controller = ControllerSystemRuleBased(**base_kwargs)
            self._add_component(controller)
            controller.isContainedIn = self.component_dict[controller.isContainedIn.id]
            controller.isContainedIn.contains.append(controller)
            controller.controlsProperty.isControlledByDevice = self.component_dict[controller.id]
            for system in controller.subSystemOf:
                system.hasSubSystem.append(controller)

        for shading_device in shading_device_instances:
            base_kwargs = self.get_object_properties(shading_device)
            extension_kwargs = {
                "saveSimulationResult": self.saveSimulationResult,
            }
            base_kwargs.update(extension_kwargs)
            shading_device = ShadingDeviceSystem(**base_kwargs)
            self._add_component(shading_device)
            shading_device.isContainedIn = self.component_dict[shading_device.isContainedIn.id]
            shading_device.isContainedIn.contains.append(shading_device)
            for system in shading_device.subSystemOf:
                system.hasSubSystem.append(shading_device)
            for property_ in shading_device.hasProperty:
                property_.isPropertyOf = shading_device

        for sensor in sensor_instances:
            base_kwargs = self.get_object_properties(sensor)
            extension_kwargs = {
                "saveSimulationResult": self.saveSimulationResult,
            }
            base_kwargs.update(extension_kwargs)
            sensor = SensorSystem(**base_kwargs)
            self._add_component(sensor)
            if sensor.isContainedIn is not None:
                sensor.isContainedIn = self.component_dict[sensor.isContainedIn.id]
                sensor.isContainedIn.contains.append(sensor)
            sensor.measuresProperty.isMeasuredByDevice = self.component_dict[sensor.id]
            for system in sensor.subSystemOf:
                system.hasSubSystem.append(sensor)
            for component in sensor.connectedAfter:
                component.connectedBefore.append(sensor)

        for meter in meter_instances:
            base_kwargs = self.get_object_properties(meter)
            extension_kwargs = {
                "saveSimulationResult": self.saveSimulationResult,
            }
            base_kwargs.update(extension_kwargs)
            meter = MeterSystem(**base_kwargs)
            self._add_component(meter)
            if meter.isContainedIn is not None:
                meter.isContainedIn = self.component_dict[meter.isContainedIn.id]
                meter.isContainedIn.contains.append(meter)
            meter.measuresProperty.isMeasuredByDevice = self.component_dict[meter.id]
            for system in meter.subSystemOf:
                system.hasSubSystem.append(meter)
            for component in meter.connectedAfter:
                component.connectedBefore.append(meter)

        # # Add supply and return node for each ventilation system
        for ventilation_system in self.system_dict["ventilation"].values():
            node_S = FlowJunctionSystem(
                    subSystemOf = [ventilation_system],
                    operationMode = "supply",
                    saveSimulationResult = self.saveSimulationResult,
                    # id = f"N_supply_{ventilation_system.id}")
                    id = "Supply node") ####
            self._add_component(node_S)
            ventilation_system.hasSubSystem.append(node_S)
            node_E = FlowJunctionSystem(
                    subSystemOf = [ventilation_system],
                    operationMode = "return",
                    saveSimulationResult = self.saveSimulationResult,
                    # id = f"N_return_{ventilation_system.id}") ##############################  ####################################################################################
                    id = "Exhaust node") ####
            self._add_component(node_E)
            ventilation_system.hasSubSystem.append(node_E)

        #Map all connectedTo properties
        for component in self.component_dict.values():
            connectedTo_new = []
            if component.connectedTo is not None:
                for base_component in component.connectedTo:
                    connectedTo_new.append(self.component_dict[base_component.id])
            component.connectedTo = connectedTo_new

        for component in self.component_dict.values():
            connectedBefore_new = []
            if len(component.connectedBefore)>0:
                for base_component in component.connectedBefore:
                    connectedBefore_new.append(self.component_dict[base_component.id])
            component.connectedBefore = connectedBefore_new

        for component in self.component_dict.values():
            connectedAfter_new = []
            if len(component.connectedAfter)>0:
                for base_component in component.connectedAfter:
                    connectedAfter_new.append(self.component_dict[base_component.id])
            component.connectedAfter = connectedAfter_new

        #Add systems
        for heating_system in self.system_dict["heating"].values():
            self._add_object(heating_system)

        for cooling_system in self.system_dict["cooling"].values():
            self._add_object(cooling_system)

        for ventilation_system in self.system_dict["ventilation"].values():
            self._add_object(ventilation_system)
        
        logger.info("[Model Class] : Exited from Apply Model Extensions Function")

    def get_object_properties(self, object_):
        return {key: value for (key, value) in vars(object_).items()}
        
    def get_component_by_class(self, dict_, class_, filter=None):
        if filter is None:
            filter = lambda v: True
        return [v for v in dict_.values() if (isinstance(v, class_) and filter(v))]

    def get_dampers_by_space(self, space):
        return [component for component in space.contains if isinstance(component, Damper)]

    def get_space_heaters_by_space(self, space):
        return [component for component in space.contains if isinstance(component, SpaceHeater)]

    def get_valves_by_space(self, space):
        return [component for component in space.contains if isinstance(component, Valve)]

    def get_controllers_by_space(self, space):
        return [component for component in space.contains if isinstance(component, Controller)]

    def get_shading_devices_by_space(self, space):
        return [component for component in space.contains if isinstance(component, ShadingDevice)]

    def _get_leaf_node_old(self, component, last_component, ref_component, found_ref=False):
        # if isinstance(component, AirToAirHeatRecovery)==False or len(list(set(component.connectedTo) - set([component])))>1:
        if isinstance(component, AirToAirHeatRecovery) or len(component.connectedTo)<2:
            node = component
            found_ref = True if component is ref_component else False
        else:
            for connected_component in component.connectedTo:
                if connected_component is not last_component:
                    if isinstance(connected_component, AirToAirHeatRecovery)==False and len(connected_component.connectedTo)>1:
                        node, found_ref = self._get_leaf_node_old(connected_component, component, ref_component, found_ref=found_ref)
                        found_ref = True if connected_component is ref_component else False
                    else:
                        node = connected_component
        return node, found_ref

    def _get_flow_placement_old(self, ref_component, component):
        """
         _______________________________________________________
        |                                                       |
    ref | ------------------------------------> flow direction  | component
        |_______________________________________________________|

        The above example would yield placement = "after"
        """
        for connected_component in component.connectedTo:
            placement=None
            node, found_ref = self._get_leaf_node_old(connected_component, component, ref_component)
            if isinstance(node, Damper):
                if found_ref:
                    if node.operationMode=="supply":
                        placement = "before"
                        side = "supply"
                        # print(1)
                    else:
                        placement = "after"
                        side = "return"
                        # print(2)
                else:
                    if node.operationMode=="supply":
                        placement = "after"
                        side = "supply"
                        # print(3)
                    else:
                        placement = "before"
                        side = "return"
                        # print(4)
                break

            elif isinstance(node, OutdoorEnvironmentSystem):
                if found_ref:
                    placement = "after"
                    side = "supply"
                    # print(5)
                else:
                    placement = "before"
                    side = "supply"
                    # print(6)
                break

            elif isinstance(node, AirToAirHeatRecovery):
                saved_found_ref = found_ref

        if placement is None:
            if saved_found_ref:
                placement = "after"
                side = "return"
                # print(7)
            else:
                placement = "before"
                side = "return"
                # print(8)
                
        return placement, side

    def _get_leaf_nodes_before(self, ref_component, component, leaf_nodes=None, found_ref=False):
        if leaf_nodes is None:
            leaf_nodes = []
        if len(component.connectedAfter)>0:
            for connected_component in component.connectedAfter:
                found_ref = True if connected_component is ref_component else False
                leaf_nodes, found_ref = self._get_leaf_nodes_before(ref_component, connected_component, leaf_nodes, found_ref=found_ref)
        else:
            leaf_nodes.append(component)
        return leaf_nodes, found_ref
    
    def _get_leaf_nodes_after(self, component, leaf_nodes=None, visited=None):
        if leaf_nodes is None:
            leaf_nodes = []
        if visited is None:
            visited = [component.id]
        if len(component.connectedBefore)>0:
            for connected_component in component.connectedBefore:
                if connected_component.id not in visited:
                    visited.append(connected_component.id)
                    leaf_nodes, visited = self._get_leaf_nodes_after(connected_component, leaf_nodes, visited)
                else:
                    visited.append(connected_component.id)
                    raise RecursionError(f"The component of class \"{connected_component.__class__.__name__}\" with id \"{connected_component.id}\" is part of a cycle. The following components form a cycle with the \"connectedBefore\" property: {' -> '.join(visited)}")
        else:
            leaf_nodes.append(component)
        return leaf_nodes, visited

    def component_is_before(self, ref_component, component, is_before=False):
        if len(component.connectedAfter)>0:
            for connected_component in component.connectedAfter:
                is_before = True if connected_component is ref_component else False
                is_before = self.component_is_before(ref_component, connected_component, is_before=is_before)
        return is_before

    def component_is_after(self, ref_component, component, is_after=False):
        if len(component.connectedBefore)>0:
            for connected_component in component.connectedBefore:
                is_after = True if connected_component is ref_component else False
                is_after = self.component_is_after(ref_component, connected_component, is_after=is_after)
        return is_after
    
    def _classes_are_before(self, ref_classes, component, is_before=False, visited=None):
        if visited is None:
            visited = [component.id]
        if len(component.connectedAfter)>0:
            for connected_component in component.connectedAfter:
                if connected_component.id not in visited:
                    is_before = True if istype(connected_component, ref_classes) else False
                    if is_before==False:
                        is_before = self._classes_are_before(ref_classes, connected_component, is_before, visited)
                else:
                    visited.append(connected_component.id)
                    raise RecursionError(f"The component of class \"{connected_component.__class__.__name__}\" with id \"{connected_component.id}\" is part of a cycle. The following components form a cycle with the \"connectedBefore\" property: {' -> '.join(visited)}")

        return is_before

    def _classes_are_after(self, ref_classes, component, is_after=False, visited=None):
        if visited is None:
            visited = [component.id]
        if len(component.connectedBefore)>0:
            for connected_component in component.connectedBefore:
                if connected_component.id not in visited:
                    visited.append(connected_component.id)
                    is_after = True if istype(connected_component, ref_classes) else False
                    if is_after==False:
                        is_after = self._classes_are_after(ref_classes, connected_component, is_after, visited)
                else:
                    visited.append(connected_component.id)
                    raise RecursionError(f"The component of class \"{connected_component.__class__.__name__}\" with id \"{connected_component.id}\" is part of a cycle. The following components form a cycle with the \"connectedBefore\" property: {' -> '.join(visited)}")

        return is_after

    def _get_instance_of_type_before(self, ref_classes, component, found_instance=None, visited=None):
        if found_instance is None:
            found_instance = []
        if visited is None:
            visited = [component.id]
        if len(component.connectedAfter)>0:
            for connected_component in component.connectedAfter:
                if connected_component.id not in visited:
                    visited.append(connected_component.id)
                    is_type = True if istype(connected_component, ref_classes) else False
                    if is_type==False:
                        found_instance = self._get_instance_of_type_before(ref_classes, connected_component, found_instance, visited)
                    else:
                        found_instance.append(connected_component)
                else:
                    visited.append(connected_component.id)
                    raise RecursionError(f"The component of class \"{connected_component.__class__.__name__}\" with id \"{connected_component.id}\" is part of a cycle. The following components form a cycle with the \"connectedBefore\" property: {' -> '.join(visited)}")

        return found_instance

    def _get_instance_of_type_after(self, ref_classes, component, found_instance=None, visited=None):
        if found_instance is None:
            found_instance = []
        if visited is None:
            visited = [component.id]
        if len(component.connectedBefore)>0:
            for connected_component in component.connectedBefore:
                if connected_component.id not in visited:
                    visited.append(connected_component.id)
                    is_type = True if istype(connected_component, ref_classes) else False
                    if is_type==False:
                        found_instance = self._get_instance_of_type_after(ref_classes, connected_component, found_instance, visited)
                    else:
                        found_instance.append(connected_component)
                else:
                    visited.append(connected_component.id)
                    raise RecursionError(f"The component of class \"{connected_component.__class__.__name__}\" with id \"{connected_component.id}\" is part of a cycle. The following components form a cycle with the \"connectedBefore\" property: {' -> '.join(visited)}")

        return found_instance

    def _get_flow_placement(self, component):
        """
         _______________________________________________________
        |                                                       |
    ref | ------------------------------------> flow direction  | component
        |_______________________________________________________|

        The above example would yield placement = "after"
        """
        # leaf_nodes_after, visited = self._get_leaf_nodes_after(component)
        # if leaf_nodes_after is None: #No leaf nodes exists as there is a cycle.
        #     space = [v for v in visited if isinstance(v, BuildingSpace)]
        #     if len(space)>0:
        #         leaf_nodes_after = space
                

        # if any([isinstance(component, BuildingSpace) for component in leaf_nodes_after]): #We assume that a BuildingSpace object is always present in ventilation systems
        if self._classes_are_after((BuildingSpaceSystem, ), component):
            side = "supply"
        else:
            side = "return"
        return side

    def _get_component_system_type(self, component):
        """
        Assumes that the component only has one supersystem
        """
        if component.subSystemOf[0].id in self.system_dict["ventilation"]:
            system_type = "ventilation"
        elif component.subSystemOf[0].id in self.system_dict["heating"]:
            system_type = "heating"
        elif component.subSystemOf[0].id in self.system_dict["cooling"]:
            system_type = "cooling"
        return system_type

    def get_occupancy_schedule(self, space_id):
        if space_id is not None:
            if f"{space_id}_occupancy_schedule" in self.component_dict:
                id = f"{space_id}_occupancy_schedule"
            else:
                id = f"{space_id}| Occupancy schedule"
        else:
            id = f"occupancy_schedule"
        occupancy_schedule = self.component_dict[id]
        return occupancy_schedule

    def get_indoor_temperature_setpoint_schedule(self, space_id=None):
        if space_id is not None:
            if f"{space_id}_temperature_setpoint_schedule" in self.component_dict:
                id = f"{space_id}_temperature_setpoint_schedule"
            else:
                id = f"{space_id}| Temperature setpoint schedule"
        else:
            id = f"temperature_setpoint_schedule"
        indoor_temperature_setpoint_schedule = self.component_dict[id]
        return indoor_temperature_setpoint_schedule

    def get_co2_setpoint_schedule(self, space_id):
        if space_id is not None:
            if f"{space_id}_co2_setpoint_schedule" in self.component_dict:
                id = f"{space_id}_co2_setpoint_schedule"
            else:
                id = f"CO2 setpoint schedule {space_id}"
        else:
            id = f"co2_setpoint_schedule"
        co2_setpoint_schedule = self.component_dict[id]
        return co2_setpoint_schedule
        
    def get_shade_setpoint_schedule(self, space_id):
        if space_id is not None:
            id = f"{space_id}_shade_setpoint_schedule"
        else:
            id = f"shade_setpoint_schedule"
        shade_setpoint_schedule = self.component_dict[id]
        return shade_setpoint_schedule

    def get_supply_air_temperature_setpoint_schedule(self, ventilation_id):
        if ventilation_id is not None:
            if f"{ventilation_id}_supply_air_temperature_schedule" in self.component_dict:
                id = f"{ventilation_id}_supply_air_temperature_schedule"
            else:
                id = f"{ventilation_id}| Supply air temperature schedule"
        else:
            id = f"supply_air_temperature_schedule"
        supply_air_temperature_setpoint_schedule = self.component_dict[id]
        return supply_air_temperature_setpoint_schedule

    def get_supply_water_temperature_setpoint_schedule(self, heating_id):
        if heating_id is not None:
            if f"{heating_id}_supply_water_temperature_schedule" in self.component_dict:
                id = f"{heating_id}_supply_water_temperature_schedule"
            else:
                id = f"{heating_id}| Supply water temperature schedule"
        else:
            id = f"supply_water_temperature_schedule"
        supply_water_temperature_setpoint_schedule = self.component_dict[id]
        return supply_water_temperature_setpoint_schedule

    # def get_supply_water_temperature_setpoint_schedule(self, heating_id):
    #     id = f"{heating_id} Supply water temperature setpoint"
    #     return self.component_dict[id]

    def connect_new(self):
        def _prune_recursive(component, visited, exception_classes, exception_classes_exact):
            visited.add(component)

            attributes = get_object_attributes()
            for attr in attributes:
                obj = rgetattr(component, attr)
                if obj is not None and inspect.ismethod(obj)==False:
                    if isinstance(obj, list):
                        for receiver_component in obj:
                            if isinstance(receiver_component, exception_classes)==False and receiver_component not in visited and istype(receiver_component, exception_classes_exact)==False:
                                visited = _prune_recursive(receiver_component, visited, exception_classes, exception_classes_exact)
                    else:
                        receiver_component = obj
                        if isinstance(receiver_component, exception_classes)==False and receiver_component not in visited and istype(receiver_component, exception_classes_exact)==False:
                            visited = _prune_recursive(receiver_component, visited, exception_classes, exception_classes_exact)
            return visited

        def _prune(obj):
            visited = set()
            visited = _prune_recursive(obj, visited)
            return visited


        classes = [cls[1] for cls in inspect.getmembers(components, inspect.isclass) if (issubclass(cls[1], (System, )) and hasattr(cls[1], "cs"))]
        
        for component_cls in classes:
            cs = component_cls.cs
            visited = set()
            # visited.add(component)
            for node in cs.nodes:
                match_nodes = [c for c in self.component_base_dict.values() if (isinstance(c, node.cls))]
                for match_node in match_nodes:
                    _prune(match_node)


    def connect(self):
        """
        Connects component instances using the saref4syst extension.
        """
        logger.info("[Model Class] : Entered in Connect Function")

        space_instances = self.get_component_by_class(self.component_dict, BuildingSpaceSystem)
        damper_instances = self.get_component_by_class(self.component_dict, DamperSystem)
        space_heater_instances = self.get_component_by_class(self.component_dict, SpaceHeaterSystem)
        valve_instances = self.get_component_by_class(self.component_dict, ValveSystem)
        coil_heating_instances = self.get_component_by_class(self.component_dict, CoilHeatingSystem)
        coil_cooling_instances = self.get_component_by_class(self.component_dict, CoilCoolingSystem)
        air_to_air_heat_recovery_instances = self.get_component_by_class(self.component_dict, AirToAirHeatRecoverySystem)
        fan_instances = self.get_component_by_class(self.component_dict, FanSystem)
        controller_instances = self.get_component_by_class(self.component_dict, Controller)
        shading_device_instances = self.get_component_by_class(self.component_dict, ShadingDevice)
        sensor_instances = self.get_component_by_class(self.component_dict, SensorSystem, filter=lambda v: isinstance(v.measuresProperty, Pressure)==False)
        meter_instances = self.get_component_by_class(self.component_dict, MeterSystem)
        node_instances = self.get_component_by_class(self.component_dict, FlowJunctionSystem)
        # flow_temperature_change_types = (AirToAirHeatRecoverySystem, FanSystem, CoilHeatingSystem, CoilCoolingSystem)



        flow_temperature_change_types = (AirToAirHeatRecoverySystem, CoilHeatingSystem, CoilCoolingSystem)
        flow_change_types = (FlowJunctionSystem, )

        outdoor_environment = self.component_dict["outdoor_environment"]
        
        for id, heating_system in self.system_dict["heating"].items():
            supply_water_temperature_setpoint_schedule = self.get_supply_water_temperature_setpoint_schedule(id)
            self.add_connection(outdoor_environment, supply_water_temperature_setpoint_schedule, "outdoorTemperature", "outdoorTemperature")

        for space in space_instances:
            dampers = self.get_dampers_by_space(space)
            valves = self.get_valves_by_space(space)
            shading_devices = self.get_shading_devices_by_space(space)

            for damper in dampers:
                if space in damper.connectedBefore:
                    self.add_connection(damper, space, "airFlowRate", "supplyAirFlowRate")
                    self.add_connection(damper, space, "damperPosition", "supplyDamperPosition")
                    ventilation_system = damper.subSystemOf[0] #Logic might be needed here in the future if multiple systems are returned
                    supply_air_temperature_setpoint_schedule = self.get_supply_air_temperature_setpoint_schedule(ventilation_system.id)
                    self.add_connection(supply_air_temperature_setpoint_schedule, space, "scheduleValue", "supplyAirTemperature")
                    
                elif space in damper.connectedAfter:
                    self.add_connection(damper, space, "airFlowRate", "returnAirFlowRate")
                    self.add_connection(damper, space, "damperPosition", "returnDamperPosition")
            
            for valve in valves:
                self.add_connection(valve, space, "valvePosition", "valvePosition")
                heating_system = valve.subSystemOf[0] #Logic might be needed here in the fututre if multiple systems are returned
                supply_water_temperature_setpoint_schedule = self.get_supply_water_temperature_setpoint_schedule(heating_system.id)
                self.add_connection(supply_water_temperature_setpoint_schedule, space, "scheduleValue", "supplyWaterTemperature")

            for shading_device in shading_devices:
                self.add_connection(shading_device, space, "shadePosition", "shadePosition")

                        
            self.add_connection(outdoor_environment, space, "globalIrradiation", "globalIrradiation")
            self.add_connection(outdoor_environment, space, "outdoorTemperature", "outdoorTemperature")
            occupancy_schedule = self.get_occupancy_schedule(space.id)
            self.add_connection(occupancy_schedule, space, "scheduleValue", "numberOfPeople")
            
        for damper in damper_instances:
            controllers = self.get_controllers_by_space(damper.isContainedIn)
            controller = [controller for controller in controllers if isinstance(controller.controlsProperty, Co2)]
            if len(controller)!=0:
                controller = controller[0]
                self.add_connection(controller, damper, "inputSignal", "damperPosition")
            else:
                filename = os.path.join(os.path.abspath(uppath(os.path.abspath(__file__), 2)), "test", "data", "time_series_data", "OE20-601b-2_Damper position.csv")
                warnings.warn(f"No CO2 controller found in BuildingSpace: \"{damper.isContainedIn.id}\".\nAssigning historic values by file: \"{filename}\"")
                if " Damper position data" not in self.component_dict:
                    damper_position_schedule = TimeSeriesInputSystem(id=" Damper position data", filename=filename)
                    self._add_component(damper_position_schedule)
                else:
                    damper_position_schedule = self.component_dict[" Damper position data"]
                self.add_connection(damper_position_schedule, damper, "damperPosition", "damperPosition")

        for space_heater in space_heater_instances:
            valve = space_heater.connectedAfter[0] #The space heater is currently a terminal component meaning that only 1 component can be connected
            self.add_connection(space, space_heater, "indoorTemperature", "indoorTemperature") 
            self.add_connection(valve, space_heater, "waterFlowRate", "waterFlowRate")
            heating_system = [v for v in space_heater.subSystemOf if v in self.system_dict["heating"].values()][0]
            supply_water_temperature_setpoint_schedule = self.get_supply_water_temperature_setpoint_schedule(heating_system.id)
            self.add_connection(supply_water_temperature_setpoint_schedule, space_heater, "scheduleValue", "supplyWaterTemperature")
            
        for valve in valve_instances:
            controllers = self.get_controllers_by_space(valve.isContainedIn)
            controller = [controller for controller in controllers if isinstance(controller.controlsProperty, Temperature)]
            # property_ = valve.hasProperty
            # controller = property_.isControlledByDevice
            if len(controller)!=0:
                controller = controller[0]
                self.add_connection(controller, valve, "inputSignal", "valvePosition")
            else:
                filename = os.path.join(os.path.abspath(uppath(os.path.abspath(__file__), 2)), "test", "data", "time_series_data", "OE20-601b-2_Space heater valve position.csv")
                warnings.warn(f"No Temperature controller found in BuildingSpace: \"{valve.isContainedIn.id}\".\nAssigning historic values by file: \"{filename}\"")
                if "Valve position schedule" not in self.component_dict:
                    valve_position_schedule = TimeSeriesInputSystem(id="Valve position schedule", filename=filename)
                    self._add_component(valve_position_schedule)
                else:
                    valve_position_schedule = self.component_dict["Valve position schedule"]
                self.add_connection(valve_position_schedule, valve, "valvePosition", "valvePosition")

        for coil_heating in coil_heating_instances:
            instance_of_type_before = self._get_instance_of_type_before(flow_temperature_change_types, coil_heating)
            if len(instance_of_type_before)==0:
                self.add_connection(outdoor_environment, coil_heating, "outdoorTemperature", "inletAirTemperature")
            else:
                instance_of_type_before = instance_of_type_before[0]
                if isinstance(instance_of_type_before, AirToAirHeatRecoverySystem):
                    self.add_connection(instance_of_type_before, coil_heating, "primaryTemperatureOut", "inletAirTemperature")
                elif isinstance(instance_of_type_before, FanSystem):
                    self.add_connection(instance_of_type_before, coil_heating, "outletAirTemperature", "inletAirTemperature")
                elif isinstance(instance_of_type_before, CoilHeatingSystem):
                    self.add_connection(instance_of_type_before, coil_heating, "outletAirTemperature", "inletAirTemperature")
                elif isinstance(instance_of_type_before, CoilCoolingSystem):
                    self.add_connection(instance_of_type_before, coil_heating, "outletAirTemperature", "inletAirTemperature")
            ventilation_system = [v for v in coil_heating.subSystemOf if v in self.system_dict["ventilation"].values()][0]
            supply_air_temperature_setpoint_schedule = self.get_supply_air_temperature_setpoint_schedule(ventilation_system.id)
            self.add_connection(supply_air_temperature_setpoint_schedule, coil_heating, "scheduleValue", "outletAirTemperatureSetpoint")
            # node = self._get_instance_of_type_after((FlowJunctionSystem, ), coil_heating)[0]
            supply_node = [node for node in node_instances if node.operationMode=="supply"][0]
            self.add_connection(supply_node, coil_heating, "flowRate", "airFlowRate")

        for coil_cooling in coil_cooling_instances:
            instance_of_type_before = self._get_instance_of_type_before(flow_temperature_change_types, coil_cooling)
            if len(instance_of_type_before)==0:
                self.add_connection(outdoor_environment, coil_cooling, "outdoorTemperature", "inletAirTemperature")
            else:
                instance_of_type_before = instance_of_type_before[0]
                if isinstance(instance_of_type_before, AirToAirHeatRecoverySystem):
                    self.add_connection(instance_of_type_before, coil_cooling, "primaryTemperatureOut", "inletAirTemperature")
                elif isinstance(instance_of_type_before, FanSystem):
                    self.add_connection(instance_of_type_before, coil_cooling, "outletAirTemperature", "inletAirTemperature")
                elif isinstance(instance_of_type_before, CoilHeatingSystem):
                    self.add_connection(instance_of_type_before, coil_cooling, "outletAirTemperature", "inletAirTemperature")
                elif isinstance(instance_of_type_before, CoilCoolingSystem):
                    self.add_connection(instance_of_type_before, coil_cooling, "outletAirTemperature", "inletAirTemperature")
            ventilation_system = [v for v in coil_cooling.subSystemOf if v in self.system_dict["ventilation"].values()][0]
            supply_air_temperature_setpoint_schedule = self.get_supply_air_temperature_setpoint_schedule(ventilation_system.id)
            self.add_connection(supply_air_temperature_setpoint_schedule, coil_cooling, "scheduleValue", "outletAirTemperatureSetpoint")
            supply_node = [node for node in node_instances if node.operationMode=="supply"][0]
            self.add_connection(supply_node, coil_cooling, "flowRate", "airFlowRate")

        for air_to_air_heat_recovery in air_to_air_heat_recovery_instances:
            ventilation_system = air_to_air_heat_recovery.subSystemOf[0]
            node_S = [v for v in ventilation_system.hasSubSystem if isinstance(v, FlowJunctionSystem) and v.operationMode == "supply"][0]
            node_E = [v for v in ventilation_system.hasSubSystem if isinstance(v, FlowJunctionSystem) and v.operationMode == "return"][0]
            self.add_connection(outdoor_environment, air_to_air_heat_recovery, "outdoorTemperature", "primaryTemperatureIn")
            self.add_connection(node_E, air_to_air_heat_recovery, "flowTemperatureOut", "secondaryTemperatureIn")
            self.add_connection(node_S, air_to_air_heat_recovery, "flowRate", "primaryAirFlowRate")
            self.add_connection(node_E, air_to_air_heat_recovery, "flowRate", "secondaryAirFlowRate")

            supply_air_temperature_setpoint_schedule = self.get_supply_air_temperature_setpoint_schedule(ventilation_system.id)
            self.add_connection(supply_air_temperature_setpoint_schedule, air_to_air_heat_recovery, "scheduleValue", "primaryTemperatureOutSetpoint")

        for fan in fan_instances:
            side = self._get_flow_placement(component=fan)
            if side=="supply":
                nodes = [node for node in node_instances if node.operationMode=="supply"]
            else:
                nodes = [node for node in node_instances if node.operationMode=="return"]
            
            if len(nodes)==0:
                raise(Exception("No Nodes after or before fanFan"))
            else:
                node = nodes[0]
                self.add_connection(node, fan, "flowRate", "airFlowRate")

        for controller in controller_instances:
            property_ = controller.controlsProperty
            property_of = property_.isPropertyOf
            measuring_device = property_.isMeasuredByDevice

            if isinstance(controller, ControllerSystemRuleBased)==False:
                if isinstance(property_of, BuildingSpace):
                    if isinstance(property_, Temperature):
                        self.add_connection(measuring_device, controller, "indoorTemperature", "actualValue")
                        setpoint_schedule = self.get_indoor_temperature_setpoint_schedule(property_of.id)
                    elif isinstance(property_, Co2):
                        self.add_connection(measuring_device, controller, "indoorCo2Concentration", "actualValue")
                        setpoint_schedule = self.get_co2_setpoint_schedule(property_of.id)
                elif isinstance(property_of, CoilHeatingSystem):
                    system = [v for v in property_of.subSystemOf if v in self.system_dict["heating"].values()][0]
                    setpoint_schedule = self.get_supply_air_temperature_setpoint_schedule(system.id)
                elif isinstance(property_of, CoilCoolingSystem):
                    system = [v for v in property_of.subSystemOf if v in self.system_dict["cooling"].values()][0]
                    setpoint_schedule = self.get_supply_air_temperature_setpoint_schedule(system.id)
                else:
                    logger.error("[Model Class] : " f"Unknown property {str(type(property_))} of {str(type(property_of))}")
                    raise Exception(f"Unknown property {str(type(property_))} of {str(type(property_of))}")
                self.add_connection(setpoint_schedule, controller, "scheduleValue", "setpointValue")
            else:
                if isinstance(property_of, BuildingSpace):
                    if isinstance(property_, Temperature):
                        self.add_connection(measuring_device, controller, "indoorTemperature", "actualValue")
                    elif isinstance(property_, Co2):
                        self.add_connection(measuring_device, controller, "indoorCo2Concentration", "actualValue")
                elif isinstance(property_of, CoilHeatingSystem):
                    system = [v for v in property_of.subSystemOf if v in self.system_dict["heating"].values()][0]
                elif isinstance(property_of, CoilCoolingSystem):
                    system = [v for v in property_of.subSystemOf if v in self.system_dict["cooling"].values()][0]
                else:
                    logger.error("[Model Class] : " f"Unknown property {str(type(property_))} of {str(type(property_of))}")
                    raise Exception(f"Unknown property {str(type(property_))} of {str(type(property_of))}")
                

        for shading_device in shading_device_instances:
            shade_setpoint_schedule = self.get_shade_setpoint_schedule(shading_device.id)
            self.add_connection(shade_setpoint_schedule, shading_device, "scheduleValue", "shadePosition")

        for sensor in sensor_instances:
            property_ = sensor.measuresProperty
            property_of = property_.isPropertyOf
            if property_of is None:
                instance_of_type_before = self._get_instance_of_type_before(flow_temperature_change_types, sensor)
                if len(instance_of_type_before)==0:
                    self.add_connection(outdoor_environment, sensor, "outdoorTemperature", "inletAirTemperature")
                elif len(instance_of_type_before)==1:
                    instance_of_type_before = instance_of_type_before[0]
                    if isinstance(instance_of_type_before, Coil):
                        if isinstance(property_, Temperature):
                            self.add_connection(instance_of_type_before, sensor, "outletAirTemperature", "flowAirTemperature")
                        else:
                            logger.error("[Model Class] :" f"Unknown property {str(type(property_))} of {str(type(instance_of_type_before))}")
                            raise Exception(f"Unknown property {str(type(property_))} of {str(type(instance_of_type_before))}")

                    elif isinstance(instance_of_type_before, AirToAirHeatRecovery):
                        side = self._get_flow_placement(sensor)
                        if isinstance(property_, Temperature):
                            if side=="supply":
                                self.add_connection(instance_of_type_before, sensor, "primaryTemperatureOut", "primaryTemperatureOut")
                            else:
                                self.add_connection(instance_of_type_before, sensor, "secondaryTemperatureOut", "secondaryTemperatureOut")
                        else:
                            logger.error("[Model Class] :" f"Unknown property {str(type(property_))} of {str(type(instance_of_type_before))}")
                            raise Exception(f"Unknown property {str(type(property_))} of {str(type(instance_of_type_before))}")
                        
                    # elif isinstance(instance_of_type_before, Fan):
                    #     if isinstance(property_, Temperature):
                    #         self.add_connection(instance_of_type_before, sensor, "outletAirTemperature", "flowAirTemperature")
                    #     else:
                    #         logger.error("[Model Class] :" f"Unknown property {str(type(property_))} of {str(type(instance_of_type_before))}")
                    #         raise Exception(f"Unknown property {str(type(property_))} of {str(type(instance_of_type_before))}")
            else:
                if isinstance(property_of, BuildingSpace):
                    if isinstance(property_, Temperature):
                        self.add_connection(property_of, sensor, "indoorTemperature", "indoorTemperature")
                    elif isinstance(property_, Co2): 
                        self.add_connection(property_of, sensor, "indoorCo2Concentration", "indoorCo2Concentration")
                    else:
                        logger.error("[Model Class] :" f"Unknown property {str(type(property_))} of {str(type(property_of))}")
                        raise Exception(f"Unknown property {str(type(property_))} of {str(type(property_of))}")

                if isinstance(property_of, Damper):
                    if isinstance(property_, OpeningPosition):
                        self.add_connection(property_of, sensor, "damperPosition", "damperPosition")
                    else:
                        logger.error("[Model Class] :" f"Unknown property {str(type(property_))} of {str(type(property_of))}")
                        raise Exception(f"Unknown property {str(type(property_))} of {str(type(property_of))}")

                if isinstance(property_of, Valve):
                    if isinstance(property_, OpeningPosition):
                        self.add_connection(property_of, sensor, "valvePosition", "valvePosition")
                    else:
                        logger.error("[Model Class] :" f"Unknown property {str(type(property_))} of {str(type(property_of))}")
                        raise Exception(f"Unknown property {str(type(property_))} of {str(type(property_of))}")

                if isinstance(property_of, ShadingDevice):
                    if isinstance(property_, OpeningPosition):
                        self.add_connection(property_of, sensor, "shadePosition", "shadePosition")
                    else:
                        logger.error("[Model Class] :" f"Unknown property {str(type(property_))} of {str(type(property_of))}")
                        raise Exception(f"Unknown property {str(type(property_))} of {str(type(property_of))}")

        for meter in meter_instances:
            property_ = meter.measuresProperty
            property_of = property_.isPropertyOf
            if isinstance(property_of, SpaceHeater):
                if isinstance(property_, Energy):
                    self.add_connection(property_of, meter, "Energy", "Energy")
                elif isinstance(property_, Power):
                    self.add_connection(property_of, meter, "Power", "Power")

            elif isinstance(property_of, Coil):
                if isinstance(property_, Energy):
                    self.add_connection(property_of, meter, "Energy", "Energy")
                elif isinstance(property_, Power):
                    self.add_connection(property_of, meter, "Power", "Power")

            elif isinstance(property_of, Fan):
                if isinstance(property_, Energy):
                    self.add_connection(property_of, meter, "Energy", "Energy")
                elif isinstance(property_, Power):
                    self.add_connection(property_of, meter, "Power", "Power")


        for node in node_instances:
            ventilation_system = node.subSystemOf[0]
            supply_dampers = [v for v in ventilation_system.hasSubSystem if isinstance(v, Damper) and len(v.connectedAfter)>0 and isinstance(v.connectedAfter[0], BuildingSpaceSystem)]
            exhaust_dampers = [v for v in ventilation_system.hasSubSystem if isinstance(v, Damper) and len(v.connectedBefore)>0 and isinstance(v.connectedBefore[0], BuildingSpaceSystem)]
            if node.operationMode=="return":
                for damper in exhaust_dampers:
                    space = damper.isContainedIn
                    self.add_connection(damper, node, "airFlowRate", "flowRate_" + space.id)
                    self.add_connection(space, node, "indoorTemperature", "flowTemperatureIn_" + space.id)
            else:
                for damper in supply_dampers:
                    self.add_connection(damper, node, "airFlowRate", "flowRate_" + space.id)
                # self.add_connection(supply_air_temperature_setpoint_schedule, node, "supplyAirTemperature", "flowTemperatureIn")
        logger.info("[Model Class] : Exited from Connect Function")


    def init_building_space_systems(self):
        for space in self.get_component_by_class(self.component_dict, BuildingSpaceSystem):
            space.get_model()

    def init_building_space_systems(self):
        for space in self.get_component_by_class(self.component_dict, BuildingSpaceSystem):
            space.get_model()


    def set_custom_initial_dict(self, custom_initial_dict):
        #validate input
        np_custom_initial_dict_ids = np.array(list(custom_initial_dict.keys()))
        legal_ids = np.array([dict_id in self.component_dict for dict_id in custom_initial_dict])
        assert np.all(legal_ids), f"Unknown component id(s) provided in \"custom_initial_dict\": {np_custom_initial_dict_ids[legal_ids==False]}"
        self.custom_initial_dict = custom_initial_dict

    def set_initial_values(self):
        """
        Arguments
        use_default: If True, set default initial values, e.g. damper position=0. If False, use initial_dict.
        initial_dict: Dictionary with component id as key and dictionary as values containing output property
        """
        default_initial_dict = {
            OutdoorEnvironmentSystem.__name__: {},
            ScheduleSystem.__name__: {},
            BuildingSpaceSystem.__name__: {"indoorTemperature": 21,
                                "indoorCo2Concentration": 500},
            BuildingSpaceCo2System.__name__: {"indoorCo2Concentration": 500},
            ControllerSystem.__name__: {"inputSignal": 0},
            ControllerSystemRuleBased.__name__: {"inputSignal": 0},
            AirToAirHeatRecoverySystem.__name__: {},
            CoilSystem.__name__: {},
            CoilHeatingSystem.__name__: {"outletAirTemperature": 21},
            CoilCoolingSystem.__name__: {},
            DamperSystem.__name__: {"airFlowRate": 0,
                            "damperPosition": 0},
            ValveSystem.__name__: {"waterFlowRate": 0,
                            "valvePosition": 0},
            FanSystem.__name__: {}, #Energy
            SpaceHeaterSystem.__name__: {"outletWaterTemperature": 21,
                                        "Energy": 0},
            FlowJunctionSystem.__name__: {},
            ShadingDeviceSystem.__name__: {},
            SensorSystem.__name__: {},
            MeterSystem.__name__: {},
            PiecewiseLinearSystem.__name__: {},
            PiecewiseLinearSupplyWaterTemperatureSystem.__name__: {},
            PiecewiseLinearScheduleSystem.__name__: {},
            TimeSeriesInputSystem.__name__: {},
            OnOffSystem.__name__: {},
        }
        initial_dict = {}
        for component in self.component_dict.values():
            initial_dict[component.id] = default_initial_dict[type(component).__name__]
        if self.custom_initial_dict is not None:
            for key, value in self.custom_initial_dict.items():
                initial_dict[key].update(value)

        for component in self.component_dict.values():
            component.output.update(initial_dict[component.id])

    def set_parameters_from_array(self, parameters, component_list, attr_list):
        for i, (obj, attr) in enumerate(zip(component_list, attr_list)):
            rsetattr(obj, attr, parameters[i])

    def set_parameters_from_dict(self, parameters, component_list, attr_list):
        for (obj, attr) in zip(component_list, attr_list):
            rsetattr(obj, attr, parameters[attr])

    def cache(self,
                startTime=None,
                endTime=None,
                stepSize=None):
        """
        This method is called once before using multiprocessing on the Simulator.
        It calls the customizable "initialize" method for specific components to cache and create the folder structure for time series data.
        """
        components = self.get_component_by_class(self.component_dict, (SensorSystem, MeterSystem, OutdoorEnvironmentSystem, TimeSeriesInputSystem))
        for component in components:
            component.initialize(startTime=startTime,
                                endTime=endTime,
                                stepSize=stepSize)

    def initialize(self,
                    startTime=None,
                    endTime=None,
                    stepSize=None):
        """
        This method is always called before simulation. 
        It sets initial values for the different components and further calls the customizable "initialize" method for each component. 
        """
        logger.info("Initializing model for simulation...")
        self.set_initial_values()
        self.check_for_for_missing_initial_values()
        for component in self.component_dict.values():
            component.clear_results()
            component.initialize(startTime=startTime,
                                endTime=endTime,
                                stepSize=stepSize)
            
    def validate_model(self):
        components = list(self.component_dict.values())
        for component in components:
            if len(component.connectedThrough)==0 and len(component.connectsAt)==0:
                warnings.warn(f"The component with class \"{component.__class__.__name__}\" and id \"{component.id}\" has no connections. It has been removed from the model.")
                self.remove_component(component)

    
    def load_model(self, semantic_model_filename=None, input_config=None, infer_connections=True, fcn=None):
        """
        This method loads component models and creates connections between the models. 
        In addition, it creates and draws graphs of the simulation model and the semantic model. 
        """
        print("Loading model...")
        # if infer_connections:
            # self.add_outdoor_environment()
        if semantic_model_filename is not None:
            self.read_datamodel_config(semantic_model_filename)
            self._create_object_graph(self.component_base_dict)
            self.draw_object_graph(filename="object_graph_input")
            self.apply_model_extensions()
        if input_config is not None:
            self.read_input_config(input_config)
        if fcn is not None:
            Model.fcn = fcn
        self.fcn()
        self._create_object_graph(self.component_dict)
        self.draw_object_graph(filename="object_graph_completed")
        if infer_connections:
            self.connect()
        self.validate_model()
        self._create_system_graph()
        self.draw_system_graph()
        self._get_execution_order()
        self._create_flat_execution_graph()
        self.draw_system_graph_no_cycles()
        self.draw_execution_graph()

    def load_model_new(self, semantic_model_filename=None, input_config=None, infer_connections=True, fcn=None):
        """
        This method loads component models and creates connections between the models. 
        In addition, it creates and draws graphs of the simulation model and the semantic model. 
        """
        print("Loading model...")
        # if infer_connections:
            # self.add_outdoor_environment()
        if semantic_model_filename is not None:
            self.read_datamodel_config(semantic_model_filename)
            self._create_object_graph(self.component_base_dict)
            self.draw_object_graph(filename="object_graph_input")
            self.parse_semantic_model()
        if input_config is not None:
            self.read_input_config(input_config)
        if fcn is not None:
            Model.fcn = fcn
        self.fcn()
        self._create_object_graph(self.component_base_dict)
        self.draw_object_graph(filename="object_graph_parsed")
        self._create_object_graph(self.component_dict)
        self.draw_object_graph(filename="object_graph_completed")
        if infer_connections:
            self.connect_new()
        # self.validate_model()
        # self._create_system_graph()
        # self.draw_system_graph()
        # self._get_execution_order()
        # self._create_flat_execution_graph()
        # self.draw_system_graph_no_cycles()
        # self.draw_execution_graph()

    def fcn(self):
        pass
            
    def draw_system_graph_no_cycles(self):
        light_black = "#3B3838"
        dark_blue = "#44546A"
        orange = "#C55A11"
        red = "#873939"
        grey = "#666666"
        light_grey = "#71797E"

        file_name = "system_graph_no_cycles"
        graph_file_name = os.path.join(self.graph_path, f"{file_name}.png")
        self.system_graph_no_cycles.write(f"{file_name}.dot", prog="dot")
        # If Python can't find the dot executeable, change "app_path" variable to the full path
        app_path = shutil.which("dot")
        args = [app_path,
                "-q",
                "-Tpng",
                "-Kdot",
                "-Nstyle=filled", #rounded,filled
                "-Nshape=box",
                "-Nfontcolor=white",
                "-Nfontname=Sans bold",
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
                f"-o{graph_file_name}",
                f"{file_name}.dot"]
        subprocess.run(args=args)
        os.remove(f"{file_name}.dot")


    def _create_system_graph(self):
        
        logger.info("[Model Class] : Entered in Create System Graph Function")


        light_black = "#3B3838"
        dark_blue = "#44546A"
        orange = "#DC8665"#"#C55A11"
        red = "#873939"
        grey = "#666666"
        light_grey = "#71797E"
        light_blue = "#8497B0"
        yellow = "#83AF9B"#"#BF9000"
        buttercream = "#B89B72"
        green = "#83AF9B"        

        fill_color_dict = {"OutdoorEnvironmentSystem": grey,
                            "ScheduleSystem": grey,
                            "BuildingSpaceSystem": light_black,
                            "ControllerSystem": orange,
                            "ControllerSystemRuleBased": orange,
                            "AirToAirHeatRecoverySystem": dark_blue,
                            "CoilSystem": dark_blue,
                            "CoilHeatingSystem": red,
                            "CoilCoolingSystem": light_blue,
                            "DamperSystem": dark_blue,
                            "ValveSystem": red,
                            "FanSystem": dark_blue,
                            "SpaceHeaterSystem": red,
                            "FlowJunctionSystem": buttercream,
                            "ShadingDeviceSystem": light_blue,
                            "SensorSystem": yellow,
                            "MeterSystem": yellow,
                            "PiecewiseLinearSystem": grey,
                            "PiecewiseLinearSupplyWaterTemperatureSystem": grey,
                            "PiecewiseLinearScheduleSystem": grey,
                            "TimeSeriesInputSystem": grey}
        fill_default = "grey"
        border_color_dict = {"OutdoorEnvironmentSystem": "black",
                            "ScheduleSystem": "black",
                            "BuildingSpaceSystem": "black",#"#2F528F",
                            "ControllerSystem": "black",
                            "ControllerSystemRuleBased": "black",
                            "AirToAirHeatRecoverySystem": "black",
                            "CoilSystem": "black",
                            "CoilHeatingSystem": "black",
                            "CoilCoolingSystem": "black",
                            "DamperSystem": "black",
                            "ValveSystem": "black",
                            "FanSystem": "black",
                            "SpaceHeaterSystem": "black",
                            "FlowJunctionSystem": "black",
                            "ShadingDeviceSystem": "black",
                            "SensorSystem": "black",
                            "MeterSystem": "black",
                            "PiecewiseLinearSystem": "black",
                            "PiecewiseLinearSupplyWaterTemperatureSystem": "black",
                            "PiecewiseLinearScheduleSystem": "black",
                            "TimeSeriesInputSystem": "black"}
        border_default = "black"

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
        for node in nx_graph.nodes():
            name = self.system_graph_node_attribute_dict[node]["label"]
            if "_" in name:
                split_delimiter = "_"
            else:
                split_delimiter = " "
            char_len = len(name)
            char_limit = 20
            if char_len>char_limit:
                name_split = name.split(split_delimiter)
                
                char_cumsum = np.cumsum(np.array([len(s) for s in name_split]))
                add_space_char = np.arange(char_cumsum.shape[0])
                char_cumsum = char_cumsum + add_space_char
                idx_arr = np.where(char_cumsum>char_limit)[0]
                if idx_arr.size!=0:
                    idx = idx_arr[0]
                    name_before_line_break = split_delimiter.join(name_split[0:idx])
                    name_after_line_break = split_delimiter.join(name_split[idx:])
                    new_name = name_before_line_break + "\n" + name_after_line_break
                    self.system_graph_node_attribute_dict[node]["label"] = new_name
                    self.system_graph_node_attribute_dict[node]["labelcharcount"] = len(name_before_line_break) if len(name_before_line_break)>len(name_after_line_break) else len(name_after_line_break)
                else:
                    self.system_graph_node_attribute_dict[node]["labelcharcount"] = len(name)
            else:
                self.system_graph_node_attribute_dict[node]["labelcharcount"] = len(name)

        degree_list = [nx_graph.degree(node) for node in nx_graph.nodes()]
        min_deg = min(degree_list)
        max_deg = max(degree_list)

        charcount_list = [self.system_graph_node_attribute_dict[node]["labelcharcount"] for node in nx_graph.nodes()]
        min_char = min(charcount_list)
        max_char = max(charcount_list)

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
            name = self.system_graph_node_attribute_dict[node]["label"]
            if "\n" in name:
                name_split = name.split("\n")[0]
                width = a_width_char*len(name_split) + b_width_char
                height = (a_height*deg + b_height)*2
            else:
                width = a_width_char*len(self.system_graph_node_attribute_dict[node]["label"]) + b_width_char
                height = a_height*deg + b_height


            if node not in self.system_graph_node_attribute_dict:
                self.system_graph_node_attribute_dict[node] = {}

            self.system_graph_node_attribute_dict[node]["fontsize"] = fontsize
            self.system_graph_node_attribute_dict[node]["width"] = width
            self.system_graph_node_attribute_dict[node]["height"] = height
            class_name = self.component_dict[node].__class__.__name__
            if class_name not in fill_color_dict:
                self.system_graph_node_attribute_dict[node]["fillcolor"] = fill_default
            else:
                self.system_graph_node_attribute_dict[node]["fillcolor"] = fill_color_dict[class_name]
            
            if class_name not in fill_color_dict:
                self.system_graph_node_attribute_dict[node]["color"] = border_default
            else:
                self.system_graph_node_attribute_dict[node]["color"] = border_color_dict[class_name]

            subgraph = self.system_subgraph_dict[type(self.component_dict[node]).__name__]

            # if " " in node or "Ø" in node:
            #     name = "\"" + node + "\""
            # else:
            #     name = node

            # print(node)
            # print(name)
            # print(len(subgraph.get_node(name)))
            name = node
            if len(subgraph.get_node(name))==1:
                subgraph.get_node(name)[0].obj_dict["attributes"].update(self.system_graph_node_attribute_dict[node])
            elif len(subgraph.get_node(name))==0: #If the name is not present, try with quotes
                 name = "\"" + node + "\""
                 subgraph.get_node(name)[0].obj_dict["attributes"].update(self.system_graph_node_attribute_dict[node])
            else:
                print([el.id for el in self.component_dict.values()])
                raise Exception(f"Multiple identical node names found in subgraph")
        logger.info("[Model Class] : Exited from Create System Graph Function")

    def draw_system_graph(self):
        light_grey = "#71797E"
        file_name = "system_graph"
        graph_file_name = os.path.join(self.graph_path, f"{file_name}.png")
        self.system_graph.write(f'{file_name}.dot')
        # If Python can't find the dot executeable, change "app_path" variable to the full path
        app_path = shutil.which("dot")
        args = [app_path,
                "-q",
                "-Tpng",
                "-Kdot",
                "-Nstyle=filled",
                "-Nshape=box",
                "-Nfontcolor=white",
                "-Nfontname=Sans bold",
                "-Nfixedsize=true",
                # "-Gnodesep=3",
                "-Nnodesep=0.05",
                "-Efontname=Helvetica",
                "-Efontsize=14",
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
                # "-Gratio=auto", #0.5
                "-Gpack=true",
                "-Gdpi=1000",
                "-Grepulsiveforce=0.5",
                "-Gremincross=true",
                "-Gstart=5",
                "-q",
                # "-Gbgcolor=#EDEDED",
                f"-o{graph_file_name}",
                f"{file_name}.dot"]
        subprocess.run(args=args)
        os.remove(f"{file_name}.dot")

    def _create_flat_execution_graph(self):
        self.execution_graph = pydot.Dot()
        prev_node=None
        for i,component_group in enumerate(self.execution_order):
            subgraph = pydot.Subgraph()#graph_name=f"cluster_{i}", style="dotted", penwidth=8)
            for component in component_group:
                node = pydot.Node('"' + component.id + '"')
                if component.id in self.system_graph_node_attribute_dict:
                    node.obj_dict["attributes"].update(self.system_graph_node_attribute_dict[component.id])
                subgraph.add_node(node)
                if prev_node:
                    self._add_edge(self.execution_graph, prev_node.obj_dict["name"], node.obj_dict["name"], edge_label="")
                prev_node = node

            self.execution_graph.add_subgraph(subgraph)

    def draw_execution_graph(self):
        light_grey = "#71797E" 
        file_name = "execution_graph"
        graph_file_name = os.path.join(self.graph_path, f"{file_name}.png")       
        self.execution_graph.write(f'{file_name}.dot')

         # If Python can't find the dot executeable, change "app_path" variable to the full path
        app_path = shutil.which("dot")
        args = [app_path,
                "-q",
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
                f"-o{graph_file_name}",
                f"{file_name}.dot"]
        subprocess.run(args=args)
        os.remove(f"{file_name}.dot")

    def _initialize_graph(self, graph_type):
        if graph_type=="system":
            self.system_graph = pydot.Dot()
            self.system_subgraph_dict = {}
            self.system_graph_node_attribute_dict = {}
            self.system_graph_edge_label_dict = {}
            self.system_graph_rank=None
        elif graph_type=="object":
            self.object_graph = pydot.Dot()
            self.object_subgraph_dict = {}
            self.object_graph_node_attribute_dict = {}
            self.object_graph_edge_label_dict = {}
            self.object_graph_rank=None
        else:
            raise(ValueError(f"Unknown graph type: \"{graph_type}\""))

    def _create_object_graph(self, object_dict):
        logger.info("[Model Class] : Entered in Create Object Graph Function")
        self._initialize_graph("object")
        # self._reset_object_dict()
        # exception_classes = (dict, float, str, int, Connection, ConnectionPoint, np.ndarray, torch.device, pd.DataFrame, property_.Property, Measurement) # These classes are excluded from the graph 
        exception_classes = (dict, float, str, int, Connection, ConnectionPoint, np.ndarray, torch.device, pd.DataFrame) # These classes are excluded from the graph 
        exception_classes_exact = (DistributionDevice,)
        visited = set()

        

        for component in object_dict.values():
            if component not in visited:
                visited = self._depth_first_search_recursive(component, visited, exception_classes, exception_classes_exact)

        for component in visited:
            # if not self.object_graph.get_node(id(component)):
            attributes = dir(component)
            attributes = [attr for attr in attributes if attr[:2]!="__"]#Remove callables
                
            for attr in attributes:
                obj = rgetattr(component, attr)
                if obj is not None and inspect.ismethod(obj)==False:
                    if isinstance(obj, list):
                        for receiver_component in obj:
                            if isinstance(receiver_component, exception_classes)==False and istype(receiver_component, exception_classes_exact)==False:
                                self._add_graph_relation(self.object_graph, component, receiver_component, edge_label=attr)
                    else:
                        receiver_component = obj
                        if isinstance(receiver_component, exception_classes)==False and istype(receiver_component, exception_classes_exact)==False:
                            self._add_graph_relation(self.object_graph, component, receiver_component, edge_label=attr)
        light_black = "#3B3838"
        dark_blue = "#44546A"
        orange = "#DC8665"#"#C55A11"
        red = "#873939"
        grey = "#666666"
        light_grey = "#71797E"
        light_blue = "#8497B0"
        yellow = "#83AF9B"#"#BF9000"
        buttercream = "#B89B72"
        green = "#83AF9B"

        fill_color_dict = {"OutdoorEnvironment": grey,
                            "Schedule": grey,
                            "BuildingSpace": light_black,
                            "Controller": orange,
                            "ControllerRuleBased": orange,
                            "AirToAirHeatRecovery": dark_blue,
                            "Coil": dark_blue,
                            "CoilHeating": red,
                            "CoilCooling": light_blue,
                            "Damper": dark_blue,
                            "Valve": red,
                            "Fan": dark_blue,
                            "SpaceHeater": red,
                            "Node": buttercream,
                            "ShadingDevice": light_blue,
                            "Sensor": yellow,
                            "Meter": yellow,
                            "PiecewiseLinearSystem": grey,
                            "PiecewiseLinearSupplyWaterTemperatureSystem": grey,
                            "PiecewiseLinearScheduleSystem": grey,
                            "TimeSeriesInputSystem": grey}
        fill_default = grey
        border_color_dict = {"OutdoorEnvironment": "black",
                            "Schedule": "black",
                            "BuildingSpace": "black",#"#2F528F",
                            "Controller": "black",
                            "ControllerRuleBased": "black",
                            "AirToAirHeatRecovery": "black",
                            "Coil": "black",
                            "CoilHeating": "black",
                            "CoilCooling": "black",
                            "Damper": "black",
                            "Valve": "black",
                            "Fan": "black",
                            "SpaceHeater": "black",
                            "Node": "black",
                            "ShadingDevice": "black",
                            "Sensor": "black",
                            "Meter": "black",
                            "PiecewiseLinearSystem": "black",
                            "PiecewiseLinearSupplyWaterTemperatureSystem": "black",
                            "PiecewiseLinearScheduleSystem": "black",
                            "TimeSeriesInputSystem": "black"}
        border_default = "black"

        # K = 10
        K = 1
        min_fontsize = 22*K
        max_fontsize = 30*K

        min_width_char = 3.5*K
        max_width_char = 5*K

        min_height = 0.4*K
        max_height = 1*K

        nx_graph = nx.drawing.nx_pydot.from_pydot(self.object_graph)

        if len(nx_graph.nodes())>0:
            for node in nx_graph.nodes():
                name = self.object_graph_node_attribute_dict[node]["label"]
                if "_" in name:
                    split_delimiter = "_"
                else:
                    split_delimiter = " "
                char_len = len(name)
                char_limit = 20
                if char_len>char_limit:
                    name_split = name.split(split_delimiter)
                    char_cumsum = np.cumsum(np.array([len(s) for s in name_split]))
                    add_space_char = np.arange(char_cumsum.shape[0])
                    char_cumsum = char_cumsum + add_space_char
                    idx_arr = np.where(char_cumsum>char_limit)[0]
                    if idx_arr.size!=0:
                        idx = idx_arr[0]
                        name_before_line_break = split_delimiter.join(name_split[0:idx])
                        name_after_line_break = split_delimiter.join(name_split[idx:])
                        new_name = name_before_line_break + "\n" + name_after_line_break
                        self.object_graph_node_attribute_dict[node]["label"] = new_name
                        self.object_graph_node_attribute_dict[node]["labelcharcount"] = len(name_before_line_break) if len(name_before_line_break)>len(name_after_line_break) else len(name_after_line_break)
                    else:
                        self.object_graph_node_attribute_dict[node]["labelcharcount"] = len(name)
                else:
                    self.object_graph_node_attribute_dict[node]["labelcharcount"] = len(name)

            degree_list = [nx_graph.degree(node) for node in nx_graph.nodes()]
            min_deg = min(degree_list)
            max_deg = max(degree_list)

            charcount_list = [self.object_graph_node_attribute_dict[node]["labelcharcount"] for node in nx_graph.nodes()]
            min_char = min(charcount_list)
            max_char = max(charcount_list)

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
                name = self.object_graph_node_attribute_dict[node]["label"]
                if "\n" in name:
                    name_split = name.split("\n")[0]
                    width = a_width_char*len(name_split) + b_width_char
                    height = (a_height*deg + b_height)*2
                else:
                    width = a_width_char*len(self.object_graph_node_attribute_dict[node]["label"]) + b_width_char
                    height = a_height*deg + b_height


                if node not in self.object_graph_node_attribute_dict:
                    self.object_graph_node_attribute_dict[node] = {}##################

                self.object_graph_node_attribute_dict[node]["fontsize"] = fontsize
                self.object_graph_node_attribute_dict[node]["width"] = width
                self.object_graph_node_attribute_dict[node]["height"] = height

                class_name = self.object_dict[node].__class__.__name__.replace("System", "")
                if class_name not in fill_color_dict:
                    self.object_graph_node_attribute_dict[node]["fillcolor"] = fill_default
                else:
                    self.object_graph_node_attribute_dict[node]["fillcolor"] = fill_color_dict[class_name]
                
                if class_name not in border_color_dict:
                    self.object_graph_node_attribute_dict[node]["color"] = border_default
                else:
                    self.object_graph_node_attribute_dict[node]["color"] = border_color_dict[class_name]

                subgraph = self.object_subgraph_dict[type(self.object_dict[node]).__name__]

                # if " " in node or "Ø" in node:
                #     name = "\"" + node + "\""
                # else:
                #     name = node

                # print(node)
                # print(name)
                # print(len(subgraph.get_node(name)))
                name = node
                if len(subgraph.get_node(name))==1:
                    subgraph.get_node(name)[0].obj_dict["attributes"].update(self.object_graph_node_attribute_dict[node])
                elif len(subgraph.get_node(name))==0: #If the name is not present, try with quotes
                    name = "\"" + node + "\""
                    subgraph.get_node(name)[0].obj_dict["attributes"].update(self.object_graph_node_attribute_dict[node])
                else:
                    raise Exception(f"Multiple identical node names found in subgraph")

        logger.info("[Model Class] : Exited from Create System Graph Function")

    def draw_object_graph(self, filename=None):
        light_grey = "#71797E"
        if filename is None:
            filename = "object_graph"
        graph_filename = os.path.join(self.graph_path, f"{filename}.png")     
        self.object_graph.write(f'{filename}.dot')
        # If Python can't find the dot executeable, change "app_path" variable to the full path
        app_path = shutil.which("dot")
        args = [app_path,
                "-q",
                "-Tpng",
                "-Kdot",
                "-Nstyle=filled",
                "-Nshape=box",
                "-Nfontcolor=white",
                "-Nfontname=Sans bold",
                "-Nfixedsize=true",
                # "-Gnodesep=3",
                "-Nnodesep=0.05",
                "-Efontname=Helvetica",
                "-Efontsize=14",
                "-Epenwidth=2",
                "-Eminlen=1",
                f"-Ecolor={light_grey}",
                "-Gcompound=true",
                "-Grankdir=TB",
                "-Goverlap=scale",
                "-Gsplines=true", #true
                "-Gmargin=0",
                # "-Gratio=compress",
                "-Gsize=5!",
                "-Gratio=auto", #0.5
                "-Gpack=true",
                "-Gdpi=1000",
                "-Grepulsiveforce=0.5",
                "-Gremincross=true",
                "-Gstart=5",
                "-q",
                # "-Gbgcolor=#EDEDED",
                f"-o{graph_filename}",
                f"{filename}.dot"]
        subprocess.run(args=args)
        os.remove(f"{filename}.dot")

    def _depth_first_search_recursive(self, component, visited, exception_classes, exception_classes_exact):
        visited.add(component)

        attributes = dir(component)
        attributes = [attr for attr in attributes if attr[:2]!="__"]#Remove callables
        for attr in attributes:
            obj = rgetattr(component, attr)
            if obj is not None and inspect.ismethod(obj)==False:
                if isinstance(obj, list):
                    for receiver_component in obj:
                        if isinstance(receiver_component, exception_classes)==False and receiver_component not in visited and istype(receiver_component, exception_classes_exact)==False:
                            visited = self._depth_first_search_recursive(receiver_component, visited, exception_classes, exception_classes_exact)
                else:
                    receiver_component = obj
                    if isinstance(receiver_component, exception_classes)==False and receiver_component not in visited and istype(receiver_component, exception_classes_exact)==False:
                        visited = self._depth_first_search_recursive(receiver_component, visited, exception_classes, exception_classes_exact)
        return visited

 
    def _depth_first_search(self, obj):
        visited = set()
        visited = self._depth_first_search_recursive(obj, visited)
        return visited

    def _flatten(self, _list):
        return [item for sublist in _list for item in sublist]

    def _depth_first_search_recursive_system(self, component, visited):
        visited.add(component)

        # Recur for all the vertices
        # adjacent to this vertex
        for connection in component.connectedThrough:
            connection_point = connection.connectsSystemAt
            receiver_component = connection_point.connectionPointOf
            if receiver_component not in visited:
                visited = self._depth_first_search_recursive_system(receiver_component, visited)
        return visited
 
    def _depth_first_search_system(self, component):
        visited = set()
        visited = self._depth_first_search_recursive_system(component, visited)
        return visited

    def get_subgraph_dict_no_cycles(self):
        self.system_subgraph_dict_no_cycles = copy.deepcopy(self.system_subgraph_dict)
        subgraphs = self.system_graph_no_cycles.get_subgraphs()
        for subgraph in subgraphs:
            subgraph.get_nodes()
            if len(subgraph.get_nodes())>0:
                node = subgraph.get_nodes()[0].obj_dict["name"].replace('"',"")
                self.system_subgraph_dict_no_cycles[type(self._component_dict_no_cycles[node]).__name__] = subgraph


    def get_component_dict_no_cycles(self):
        self._component_dict_no_cycles = copy.deepcopy(self.component_dict)
        self.system_graph_no_cycles = copy.deepcopy(self.system_graph)
        self.get_subgraph_dict_no_cycles()
        self.required_initialization_connections = []

        controller_instances = [v for v in self._component_dict_no_cycles.values() if isinstance(v, Controller)]
        for controller in controller_instances:
            controlled_component = controller.controlsProperty.isPropertyOf
            assert controlled_component is not None, f"The attribute \"isPropertyOf\" is None for property \"{controller.controlsProperty}\" of component \"{controller.id}\""
            visited = self._depth_first_search_system(controller)

            for reachable_component in visited:
                for connection in reachable_component.connectedThrough:
                    connection_point = connection.connectsSystemAt
                    receiver_component = connection_point.connectionPointOf
                    if controlled_component==receiver_component:
                        controlled_component.connectsAt.remove(connection_point)
                        reachable_component.connectedThrough.remove(connection)
                        edge_label = self.get_edge_label(connection.senderPropertyName, connection_point.receiverPropertyName)
                        status = self._del_edge(self.system_graph_no_cycles, reachable_component.id, controlled_component.id, label=edge_label)
                        assert status, "del_edge returned False. Check if additional characters should be added to \"disallowed_characters\"."

                        self.required_initialization_connections.append(connection)

    def _set_addUncertainty(self, addUncertainty=None, filter="physicalSystem"):
        allowed_filters = ["all", "physicalSystem"]
        assert isinstance(addUncertainty, bool), "Argument addUncertainty must be True or False"
        assert filter in allowed_filters, f"The \"filter\" argument must be one of the following: {', '.join(allowed_filters)} - \"{filter}\" was provided."
        sensor_instances = self.get_component_by_class(self.component_dict, Sensor)
        meter_instances = self.get_component_by_class(self.component_dict, Meter)
        instances = sensor_instances
        instances.extend(meter_instances)
        if filter=="all":
            for instance in instances:
                instance.addUncertainty = addUncertainty
        elif filter=="physicalSystem":
            for instance in instances:
                if instance.isPhysicalSystem: #Only set the variable if the measuring device is real data supplied as input to the model
                    instance.initialize()
                    instance.addUncertainty = addUncertainty

    def load_chain_log(self, filename):
        with open(filename, 'rb') as handle:
            self.chain_log = pickle.load(handle)
            self.chain_log["chain.T"] = 1/self.chain_log["chain.betas"] ##################################
            list_ = ["integratedAutoCorrelatedTime", "chain.jumps_accepted", "chain.jumps_proposed", "chain.swaps_accepted", "chain.swaps_proposed"]
            for key in list_:
                self.chain_log[key] = np.array(self.chain_log[key])
        # time_format = '%Y-%m-%d %H:%M:%S%z'
        # self.stepSize_train = datetime.datetime.strptime(self.chain_log["stepSize_train"], time_format)
        # self.startTime_train = datetime.datetime.strptime(self.chain_log["startTime_train"], time_format)
        # self.endTime_train = datetime.datetime.strptime(self.chain_log["endTime_train"], time_format)

    def set_trackGradient(self, trackGradient):
        assert isinstance(trackGradient, bool), "Argument trackGradient must be True or False" 
        for component in self.flat_execution_order:
            component.trackGradient = trackGradient

    def map_execution_order(self):
        self.execution_order = [[self.component_dict[component.id] for component in component_group] for component_group in self.execution_order]

    def map_required_initialization_connections(self):
        self.required_initialization_connections = [connection for no_cycle_connection in self.required_initialization_connections for connection in self.component_dict[no_cycle_connection.connectsSystem.id].connectedThrough if connection.senderPropertyName==no_cycle_connection.senderPropertyName]

    def check_for_for_missing_initial_values(self):
        for connection in self.required_initialization_connections:
            component = connection.connectsSystem
            if connection.senderPropertyName not in component.output:
                raise Exception(f"The component with id: \"{component.id}\" and class: \"{component.__class__.__name__}\" is missing an initial value for the output: {connection.senderPropertyName}")
            elif component.output[connection.senderPropertyName] is None:
                raise Exception(f"The component with id: \"{component.id}\" and class: \"{component.__class__.__name__}\" is missing an initial value for the output: {connection.senderPropertyName}")
                
    def _get_execution_order(self):
        self.get_component_dict_no_cycles()
        initComponents = [v for v in self._component_dict_no_cycles.values() if len(v.connectsAt)==0]
        self.activeComponents = initComponents
        self.execution_order = []
        while len(self.activeComponents)>0:
            self._traverse()

        self.map_execution_order()
        self.map_required_initialization_connections()
        self.flat_execution_order = self._flatten(self.execution_order)
        assert len(self.flat_execution_order)==len(self._component_dict_no_cycles), f"Cycles detected in the model. Inspect the generated file \"system_graph.png\" to see where."

    def _traverse(self):
        activeComponentsNew = []
        self.component_group = []
        for component in self.activeComponents:
            self.component_group.append(component)
            for connection in component.connectedThrough:
                connection_point = connection.connectsSystemAt
                receiver_component = connection_point.connectionPointOf
                receiver_component.connectsAt.remove(connection_point)
                if len(receiver_component.connectsAt)==0:
                    activeComponentsNew.append(receiver_component)
        self.activeComponents = activeComponentsNew
        self.execution_order.append(self.component_group)



    ###### METHODS NOT CURRENTLY USED ######
    def get_leaf_subsystems(self, system):
        for sub_system in system.hasSubSystem:
            if sub_system.hasSubSystem is None:
                self.leaf_subsystems.append(sub_system)
            else:
                self.get_leaf_subsystems(sub_system)


    






    

