import networkx as nx
import pandas as pd
import warnings
import math
import shutil
import subprocess
import sys
import os
import copy
import pydot
import inspect
import numpy as np
import pandas as pd
import numbers
import datetime
import torch
import json
import builtins
import pickle
import matplotlib.font_manager
from PIL import ImageFont
from itertools import count
from prettytable import PrettyTable
from prettytable.colortable import ColorTable, Themes
from twin4build.utils.print_progress import PrintProgress
# import fmpy.fmi2 as fmi2

from openpyxl import load_workbook
from dateutil.parser import parse
from twin4build.utils.fmu.fmu_component import FMUComponent
from twin4build.utils.isnumeric import isnumeric
from twin4build.utils.get_object_attributes import get_object_attributes
from twin4build.utils.mkdir_in_root import mkdir_in_root
from twin4build.utils.rsetattr import rsetattr
from twin4build.utils.rgetattr import rgetattr
from twin4build.utils.rhasattr import rhasattr
from twin4build.utils.istype import istype
from twin4build.utils.data_loaders.fiwareReader import fiwareReader
from twin4build.utils.preprocessing.data_sampler import data_sampler
from twin4build.utils.data_loaders.load_spreadsheet import sample_from_df
from twin4build.saref4syst.connection import Connection 
from twin4build.saref4syst.connection_point import ConnectionPoint
from twin4build.saref4syst.system import System
import twin4build.utils.signature_pattern.signature_pattern as signature_pattern
from twin4build.utils.uppath import uppath
from twin4build.logger.Logging import Logging
import twin4build.base as base
import twin4build.components as components

logger = Logging.get_logger("ai_logfile")

def str2Class(str):
    return getattr(sys.modules[__name__], str)


class Model:
    def __str__(self):
        t = PrettyTable(["Number of components in simulation model: ", len(self.component_dict)])
        title = f"Model overview    id: {self.id}"
        t.title = title
        t.add_row(["Number of objects in semantic model: ", len(self.object_dict)], divider=True)
        t.add_row(["", ""])
        t.add_row(["", ""], divider=True)
        t.add_row(["id", "Class"], divider=True)
        unique_class_list = []
        for component in self.component_dict.values():
            cls = component.__class__
            if cls not in unique_class_list:
                unique_class_list.append(cls)
        unique_class_list = sorted(unique_class_list, key=lambda x: x.__name__.lower())

        for cls in unique_class_list:
            cs = self.get_component_by_class(self.component_dict, cls, filter=lambda v, class_: v.__class__ is class_)
            n = len(cs)
            for i,c in enumerate(cs):
                t.add_row([c.id, cls.__name__], divider=True if i==n-1 else False)
            
        return t.get_string()

    def __init__(self,
                 id=None,
                saveSimulationResult=False):
        self.valid_chars = ["_", "-", " ", "(", ")", "[", "]"]
        assert isinstance(id, str), f"Argument \"id\" must be of type {str(type(str))}"
        isvalid = np.array([x.isalnum() or x in self.valid_chars for x in id])
        np_id = np.array(list(id))
        violated_characters = list(np_id[isvalid==False])
        assert all(isvalid), f"The model with id \"{id}\" has an invalid id. The characters \"{', '.join(violated_characters)}\" are not allowed."
        self.id = id
        self.saveSimulationResult = saveSimulationResult
        self._initialize_graph("system")
        self._initialize_graph("object")

        self.system_dict = {"ventilation": {},
                            "heating": {},
                            "cooling": {},
                            }
        self.heat_exchanger_dict = {}
        self.component_base_dict = {} #Subset of object_dict
        self.component_dict = {} #Subset of object_dict
        self.object_dict = {}
        self.object_dict_reversed = {}
        self.object_counter_dict = {}
        self.property_dict = {}
        self.custom_initial_dict = None
        self.initial_dict = None
        self.heatexchanger_types = (base.AirToAirHeatRecovery, base.Coil)
        self.water_types = (base.Pump, base.Valve, base.SpaceHeater)
        self.air_types = (base.Fan, base.Damper)

        self.graph_path, isfile = self.get_dir(folder_list=["graphs"])
        logger.info("[Model Class] : Exited from Initialise Function")
    
    def get_dir(self, folder_list=[], filename=None):
        f = ["generated_files", "models", self.id]
        f.extend(folder_list)
        folder_list = f
        filename, isfile = mkdir_in_root(folder_list=folder_list, filename=filename)
        return filename, isfile

    def _add_edge(self, graph, a, b, sender_property_name=None, receiver_property_name=None, edge_kwargs=None):
        if edge_kwargs is None:
            edge_label = self.get_edge_label(sender_property_name, receiver_property_name)
            edge_kwargs = {"label": edge_label}
            graph.add_edge(pydot.Edge(a, b, **edge_kwargs))
        else:
            graph.add_edge(pydot.Edge(a, b, **edge_kwargs))

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

        self._add_object(component)

    def get_new_object_name(self, obj):
        if "id" not in get_object_attributes(obj):
            if obj.__class__.__name__ not in self.object_counter_dict:
                self.object_counter_dict[obj.__class__.__name__] = 0
            name = f"{obj.__class__.__name__.lower()} {str(self.object_counter_dict[obj.__class__.__name__])}"# [{id(obj)}]"
            self.object_counter_dict[obj.__class__.__name__] += 1
        else:
            name = obj.id
        return name

    def make_pickable(self):
        """
        This method is responsible to remove all references to unpickable objects for the Model instance.
        This prepares the Model instance to be used with multiprocessing in the Estimator class.
        """
        self.object_dict = {} 
        self.object_dict_reversed = {}
        fmu_components = self.get_component_by_class(self.component_dict, FMUComponent)
        for fmu_component in fmu_components:
            if "fmu" in get_object_attributes(fmu_component):
                # fmu_component.fmu.freeInstance()
                # fmu_component.fmu.terminate()
                del fmu_component.fmu
                del fmu_component.fmu_initial_state
                fmu_component.INITIALIZED = False

    def _add_object(self, obj):
        if obj in self.component_dict.values() or obj in self.component_base_dict.values():
            name = obj.id
            self.object_dict[name] = obj
            self.object_dict_reversed[obj] = name
        elif obj not in self.object_dict_reversed:
            name = self.get_new_object_name(obj)
            self.object_dict[name] = obj
            self.object_dict_reversed[obj] = name

    def remove_component(self, component):
        for connection in component.connectedThrough:
            connection_point = connection.connectsSystemAt
            connected_component = connection_point.connectionPointOf
            self.remove_connection(component, connected_component, connection.senderPropertyName, connection_point.receiverPropertyName)
            connection.connectsSystem = None

        for connection_point in component.connectsAt:
            connection_point.connectPointOf = None
        del self.component_dict[component.id]
        #Remove from subgraph dict
        subgraph_dict = self.system_subgraph_dict
        component_class_name = component.__class__
        if component_class_name in subgraph_dict:
            subgraph_dict[component_class_name].del_node(component.id)



    def get_edge_label(self, sender_property_name, receiver_property_name):
        end_space = "          "
        # edge_label = ("Out: " + sender_property_name.split("_")[0] + end_space + "\n"
        #                 "In: " + receiver_property_name.split("_")[0] + end_space)
        edge_label = ("Out: " + sender_property_name + end_space + "\n"
                        "In: " + receiver_property_name + end_space)
        return edge_label

    def wrap_component_output(self, wrapped_component, output, component):
        pass


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

        exception_classes = (components.TimeSeriesInputSystem, components.FlowJunctionSystem, components.JunctionDuctSystem ,components.PiecewiseLinearSystem, components.PiecewiseLinearSupplyWaterTemperatureSystem, components.PiecewiseLinearScheduleSystem, base.Sensor, base.Meter, components.MaxSystem) # These classes are exceptions because their inputs and outputs can take any form 
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

    def _add_graph_relation(self, graph, sender_component, receiver_component, sender_property_name=None, receiver_property_name=None, edge_kwargs=None, sender_node_kwargs=None, receiver_node_kwargs=None):
        if sender_node_kwargs is None:
            sender_node_kwargs = {}

        if receiver_node_kwargs is None:
            receiver_node_kwargs = {}


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

        

        sender_class_name = sender_component.__class__
        receiver_class_name = receiver_component.__class__
        if sender_class_name not in subgraph_dict:
            subgraph_dict[sender_class_name] = pydot.Subgraph(rank=rank)
            graph.add_subgraph(subgraph_dict[sender_class_name])
        
        if receiver_class_name not in subgraph_dict:
            subgraph_dict[receiver_class_name] = pydot.Subgraph(rank=rank)
            graph.add_subgraph(subgraph_dict[receiver_class_name])
        
        sender_component_name = self.object_dict_reversed[sender_component]
        receiver_component_name = self.object_dict_reversed[receiver_component]
        self._add_edge(graph, sender_component_name, receiver_component_name, sender_property_name, receiver_property_name, edge_kwargs) ###
        
        cond1 = not subgraph_dict[sender_class_name].get_node(sender_component_name)
        cond2 = not subgraph_dict[sender_class_name].get_node("\""+ sender_component_name +"\"")
        if cond1 and cond2:
            node = pydot.Node(sender_component_name)
            subgraph_dict[sender_class_name].add_node(node)
        
        cond1 = not subgraph_dict[receiver_class_name].get_node(receiver_component_name)
        cond2 = not subgraph_dict[receiver_class_name].get_node("\""+ receiver_component_name +"\"")
        if cond1 and cond2:
            node = pydot.Node(receiver_component_name)
            subgraph_dict[receiver_class_name].add_node(node)


        if "label" not in sender_node_kwargs:
            sender_node_kwargs.update({"label": sender_component_name})
            graph_node_attribute_dict[sender_component_name] = sender_node_kwargs
        graph_node_attribute_dict[sender_component_name] = sender_node_kwargs

        if "label" not in receiver_node_kwargs:
            receiver_node_kwargs.update({"label": receiver_component_name})
            graph_node_attribute_dict[receiver_component_name] = receiver_node_kwargs
        graph_node_attribute_dict[receiver_component_name] = receiver_node_kwargs


    def remove_connection(self, sender_component, receiver_component, sender_property_name, receiver_property_name):
        """
        Deletes a connection between two components in a system
        Updates the respective lists of connected components and deletes the ConnectionPoint objects from the receiver component's list of connection points
        It also updates the dictionaries of inputs/outputs for the sender and receiver components accordingly
        Finally, it deletes the labeled edge between the two components in the system graph
        """
        logger.info("[Model Class] : Entered in Remove Connection Function")

        #print("==============================")
        #print("Removing connection between: ", sender_component.id, " and ", receiver_component.id)
        #print("==============================")

        sender_obj_connection = None
        for connection in sender_component.connectedThrough:
            if connection.senderPropertyName == sender_property_name:
                sender_obj_connection = connection
                break
        if sender_obj_connection is None:
            raise ValueError(f"The sender component \"{sender_component.id}\" does not have a connection with the property \"{sender_property_name}\"")
        sender_component.connectedThrough.remove(sender_obj_connection)

        receiver_component_connection_point = None
        for connection_point in receiver_component.connectsAt:
            if connection_point.receiverPropertyName == receiver_property_name:
                receiver_component_connection_point = connection_point
                break
        if receiver_component_connection_point is None:
            raise ValueError(f"The receiver component \"{receiver_component.id}\" does not have a connection point with the property \"{receiver_property_name}\"")
        receiver_component.connectsAt.remove(receiver_component_connection_point)

        del sender_obj_connection
        del receiver_component_connection_point
        
        self._del_edge(self.system_graph, sender_component.id, receiver_component.id, self.get_edge_label(sender_property_name, receiver_property_name))

        #Exception classes 
        exception_classes = (components.TimeSeriesInputSystem, components.FlowJunctionSystem, components.PiecewiseLinearSystem, components.PiecewiseLinearSupplyWaterTemperatureSystem, components.PiecewiseLinearScheduleSystem, base.Sensor, base.Meter) # These classes are exceptions because their inputs and outputs can take any form

        if isinstance(sender_component, exception_classes):
            del sender_component.output[sender_property_name]

        if isinstance(receiver_component, exception_classes):
            del receiver_component.input[receiver_property_name]
        
        logger.info("[Model Class] : Exited from Remove Connection Function")
    
    def add_outdoor_environment(self, filename=None):
        outdoor_environment = base.OutdoorEnvironment(
            filename=filename,
            saveSimulationResult = self.saveSimulationResult,
            id = "outdoor_environment")
        self.component_base_dict["outdoor_environment"] = outdoor_environment

    def add_outdoor_environment_system(self, filename=None):
        outdoor_environment = components.OutdoorEnvironmentSystem(
            filename=filename,
            saveSimulationResult = self.saveSimulationResult,
            id = "outdoor_environment")
        self._add_component(outdoor_environment)

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
            ventilation_system = base.DistributionDevice(id=ventilation_system_name)
            self.system_dict["ventilation"][ventilation_system_name] = ventilation_system
        
        for heating_system_name in df_dict["System"]["Heating system name"].dropna():
            heating_system = base.DistributionDevice(id=heating_system_name)
            self.system_dict["heating"][heating_system_name] = heating_system

        for cooling_system_name in df_dict["System"]["Cooling system name"].dropna():
            cooling_system = base.DistributionDevice(id=cooling_system_name)
            self.system_dict["cooling"][cooling_system_name] = cooling_system

        for row in df_dict["BuildingSpace"].dropna(subset=["id"]).itertuples(index=False):
            space_name = row[df_dict["BuildingSpace"].columns.get_loc("id")]
            space = base.BuildingSpace(id=space_name)
            self.component_base_dict[space_name] = space
            
        for row in df_dict["Damper"].dropna(subset=["id"]).itertuples(index=False):
            damper_name = row[df_dict["Damper"].columns.get_loc("id")]
            damper = base.Damper(id=damper_name)
            self.component_base_dict[damper_name] = damper
            if row[df_dict["Damper"].columns.get_loc("isContainedIn")] not in self.component_base_dict:
                warnings.warn("Cannot find a matching BuildingSpace object for damper \"" + damper_name + "\"")                

        for row in df_dict["SpaceHeater"].dropna(subset=["id"]).itertuples(index=False):
            space_heater_name = row[df_dict["SpaceHeater"].columns.get_loc("id")]
            space_heater = base.SpaceHeater(id=space_heater_name)
            self.component_base_dict[space_heater_name] = space_heater
            if row[df_dict["SpaceHeater"].columns.get_loc("isContainedIn")] not in self.component_base_dict:
                warnings.warn("Cannot find a matching SpaceHeater object for space heater \"" + space_heater_name + "\"")                

        for row in df_dict["Valve"].dropna(subset=["id"]).itertuples(index=False):
            valve_name = row[df_dict["Valve"].columns.get_loc("id")]
            valve = base.Valve(id=valve_name)
            self.component_base_dict[valve_name] = valve
            if row[df_dict["Valve"].columns.get_loc("isContainedIn")] not in self.component_base_dict:
                warnings.warn("Cannot find a matching Valve object for valve \"" + valve_name + "\"")

        for row in df_dict["Coil"].dropna(subset=["id"]).itertuples(index=False):
            coil_name = row[df_dict["Coil"].columns.get_loc("id")]
            coil = base.Coil(id=coil_name)
            self.component_base_dict[coil_name] = coil
            
        for row in df_dict["AirToAirHeatRecovery"].dropna(subset=["id"]).itertuples(index=False):
            air_to_air_heat_recovery_name = row[df_dict["AirToAirHeatRecovery"].columns.get_loc("id")]
            air_to_air_heat_recovery = base.AirToAirHeatRecovery(id=air_to_air_heat_recovery_name)
            self.component_base_dict[air_to_air_heat_recovery_name] = air_to_air_heat_recovery

        for row in df_dict["Fan"].dropna(subset=["id"]).itertuples(index=False):
            fan_name = row[df_dict["Fan"].columns.get_loc("id")]
            fan = base.Fan(id=fan_name)
            self.component_base_dict[fan_name] = fan

        for row in df_dict["Controller"].dropna(subset=["id"]).itertuples(index=False):
            controller_name = row[df_dict["Controller"].columns.get_loc("id")]
            controller = base.Controller(id=controller_name)
            self.component_base_dict[controller_name] = controller

        for row in df_dict["SetpointController"].dropna(subset=["id"]).itertuples(index=False):
            controller_name = row[df_dict["SetpointController"].columns.get_loc("id")]
            controller = base.SetpointController(id=controller_name)
            self.component_base_dict[controller_name] = controller

        for row in df_dict["RulebasedController"].dropna(subset=["id"]).itertuples(index=False):
            controller_name = row[df_dict["RulebasedController"].columns.get_loc("id")]
            controller = base.RulebasedController(id=controller_name)
            self.component_base_dict[controller_name] = controller

        for row in df_dict["Schedule"].dropna(subset=["id"]).itertuples(index=False):
            schedule_name = row[df_dict["Schedule"].columns.get_loc("id")]
            schedule = base.Schedule(id=schedule_name)
            self.component_base_dict[schedule_name] = schedule
                
        for row in df_dict["ShadingDevice"].dropna(subset=["id"]).itertuples(index=False):
            shading_device_name = row[df_dict["ShadingDevice"].columns.get_loc("id")]
            shading_device = base.ShadingDevice(id=shading_device_name)
            self.component_base_dict[shading_device_name] = shading_device            

        for row in df_dict["Sensor"].dropna(subset=["id"]).itertuples(index=False):
            sensor_name = row[df_dict["Sensor"].columns.get_loc("id")]
            sensor = base.Sensor(id=sensor_name)
            self.component_base_dict[sensor_name] = sensor

        for row in df_dict["Meter"].dropna(subset=["id"]).itertuples(index=False):
            meter_name = row[df_dict["Meter"].columns.get_loc("id")]
            meter = base.Meter(id=meter_name)
            self.component_base_dict[meter_name] = meter

        for row in df_dict["Pump"].dropna(subset=["id"]).itertuples(index=False):
            pump_name = row[df_dict["Pump"].columns.get_loc("id")]
            pump = base.Pump(id=pump_name)
            self.component_base_dict[pump_name] = pump
            
        for row in df_dict["Property"].dropna(subset=["id"]).itertuples(index=False):
            property_name = row[df_dict["Property"].columns.get_loc("id")]
            Property = getattr(base, row[df_dict["Property"].columns.get_loc("type")])
            property_ = Property()
            self.property_dict[property_name] = property_

        self.add_heatexchanger_subsystems()

        logger.info("[Model Class] : Exited from Intantiate Object Function")

    def add_heatexchanger_subsystems(self):
        added_components = []
        for component in self.component_base_dict.values():
            if isinstance(component, self.heatexchanger_types):
                if isinstance(component, base.AirToAirHeatRecovery):
                    component_supply = base.AirToAirHeatRecovery(subSystemOf=[component], id=f"{component.id} (supply)")
                    component_exhaust = base.AirToAirHeatRecovery(subSystemOf=[component], id=f"{component.id} (exhaust)")
                    component.hasSubSystem.append(component_supply)
                    component.hasSubSystem.append(component_exhaust)
                    added_components.append(component_supply)
                    added_components.append(component_exhaust)

                elif isinstance(component, base.Coil):
                    component_waterside = base.Coil(subSystemOf=[component], id=f"{component.id} (waterside)")
                    component_airside = base.Coil(subSystemOf=[component], id=f"{component.id} (airside)")
                    component.hasSubSystem.append(component_waterside)
                    component.hasSubSystem.append(component_airside)
                    added_components.append(component_waterside)
                    added_components.append(component_airside)

        for component in added_components:
            self.component_base_dict[component.id] = component
                
                    

    def _populate_objects(self, df_dict):
        """
        All components listed in the configuration file are populated with data and connections are defined.

        Arguments
        df_dict: A dictionary of dataframes read from the configuration file with sheet names as keys and dataframes as values.  
        """
        logger.info("[Model Class] : Entered in Populate Object Function")
        allowed_numeric_types = (float, int)
        true_list = ["True", "true", "TRUE"]

        for row in df_dict["BuildingSpace"].dropna(subset=["id"]).itertuples(index=False):
            space_name = row[df_dict["BuildingSpace"].columns.get_loc("id")]
            space = self.component_base_dict[space_name]
            if isinstance(row[df_dict["BuildingSpace"].columns.get_loc("hasProperty")], str):
                properties = [self.property_dict[property_name] for property_name in row[df_dict["BuildingSpace"].columns.get_loc("hasProperty")].split(";")]
                space.hasProperty.extend(properties)
            else:
                message = f"Required property \"hasProperty\" not set for BuildingSpace object \"{space.id}\""
                raise(ValueError(message))
            

            if "connectedTo" not in df_dict["BuildingSpace"].columns:
                warnings.warn("The property \"connectedTo\" is not found in \"BuildingSpace\" sheet. This is ignored for now but will raise an error in the future. It probably is caused by using an outdated configuration file.")
            elif isinstance(row[df_dict["BuildingSpace"].columns.get_loc("connectedTo")], str):
                connected_to = row[df_dict["BuildingSpace"].columns.get_loc("connectedTo")].split(";")
                connected_to = [self.component_base_dict[component_name] for component_name in connected_to]
                space.connectedTo.extend(connected_to)
            
            if "hasFluidSuppliedBy" not in df_dict["BuildingSpace"].columns:
                warnings.warn("The property \"hasFluidSuppliedBy\" is not found in \"BuildingSpace\" sheet. This is ignored for now but will raise an error in the future. It probably is caused by using an outdated configuration file.")
            else:
                if isinstance(row[df_dict["BuildingSpace"].columns.get_loc("hasFluidSuppliedBy")], str):
                    connected_after = row[df_dict["BuildingSpace"].columns.get_loc("hasFluidSuppliedBy")].split(";")
                    connected_after = [self.component_base_dict[component_name] for component_name in connected_after]
                    space.hasFluidSuppliedBy.extend(connected_after)
                else:
                    message = f"Property \"hasFluidSuppliedBy\" not set for BuildingSpace object \"{space.id}\""
                    warnings.warn(message)


            if "hasProfile" not in df_dict["BuildingSpace"].columns:
                warnings.warn("The property \"hasProfile\" is not found in \"BuildingSpace\" sheet. This is ignored for now but will raise an error in the future. It probably is caused by using an outdated configuration file.")
            elif isinstance(row[df_dict["BuildingSpace"].columns.get_loc("hasProfile")], str):
                schedule_name = row[df_dict["BuildingSpace"].columns.get_loc("hasProfile")]
                space.hasProfile = self.component_base_dict[schedule_name]

            space.airVolume = row[df_dict["BuildingSpace"].columns.get_loc("airVolume")]
            
        for row in df_dict["Damper"].dropna(subset=["id"]).itertuples(index=False):
            damper_name = row[df_dict["Damper"].columns.get_loc("id")]
            damper = self.component_base_dict[damper_name]
            systems = row[df_dict["Damper"].columns.get_loc("subSystemOf")].split(";")
            systems = [system for system_dict in self.system_dict.values() for system in system_dict.values() if system.id in systems]
            damper.subSystemOf.extend(systems)

            if isinstance(row[df_dict["Damper"].columns.get_loc("hasFluidSuppliedBy")], str):
                connected_after = row[df_dict["Damper"].columns.get_loc("hasFluidSuppliedBy")].split(";")
                connected_after = [self.component_base_dict[component_name] for component_name in connected_after]
                damper.hasFluidSuppliedBy.extend(connected_after)
            else:
                message = f"Property \"hasFluidSuppliedBy\" not set for Damper object \"{damper.id}\""
                warnings.warn(message)

            if isinstance(row[df_dict["Damper"].columns.get_loc("hasFluidReturnedBy")], str):
                connected_after = row[df_dict["Damper"].columns.get_loc("hasFluidReturnedBy")].split(";")
                connected_after = [self.component_base_dict[component_name] for component_name in connected_after]
                damper.hasFluidReturnedBy.extend(connected_after)
            else:
                message = f"Property \"hasFluidSuppliedBy\" not set for Damper object \"{damper.id}\""
                warnings.warn(message)

            if isinstance(row[df_dict["Damper"].columns.get_loc("hasProperty")], str):
                properties = [self.property_dict[property_name] for property_name in row[df_dict["Damper"].columns.get_loc("hasProperty")].split(";")]
                damper.hasProperty.extend(properties)
            else:
                message = f"Required property \"hasProperty\" not set for Damper object \"{damper.id}\""
                raise(ValueError(message))
            
            
            damper.isContainedIn = self.component_base_dict[row[df_dict["Damper"].columns.get_loc("isContainedIn")]]
            rsetattr(damper, "nominalAirFlowRate.hasValue", row[df_dict["Damper"].columns.get_loc("nominalAirFlowRate")])
            
        for row in df_dict["SpaceHeater"].dropna(subset=["id"]).itertuples(index=False):
            space_heater_name = row[df_dict["SpaceHeater"].columns.get_loc("id")]
            space_heater = self.component_base_dict[space_heater_name]

            if isinstance(row[df_dict["SpaceHeater"].columns.get_loc("subSystemOf")], str):
                systems = row[df_dict["SpaceHeater"].columns.get_loc("subSystemOf")].split(";")
                systems = [system for system_dict in self.system_dict.values() for system in system_dict.values() if system.id in systems]
                space_heater.subSystemOf.extend(systems)
            else:
                message = f"Required property \"subSystemOf\" not set for SpaceHeater object \"{space_heater.id}\""
                raise(ValueError(message))

            if isinstance(row[df_dict["SpaceHeater"].columns.get_loc("hasFluidSuppliedBy")], str):
                connected_after = row[df_dict["SpaceHeater"].columns.get_loc("hasFluidSuppliedBy")].split(";")
                connected_after = [self.component_base_dict[component_name] for component_name in connected_after]
                space_heater.hasFluidSuppliedBy.extend(connected_after)
            else:
                message = f"Property \"hasFluidSuppliedBy\" not set for SpaceHeater object \"{space_heater.id}\""
                warnings.warn(message)

            properties = [self.property_dict[property_name] for property_name in row[df_dict["SpaceHeater"].columns.get_loc("hasProperty")].split(";")]
            space_heater.hasProperty.extend(properties)
            
            space_heater.isContainedIn = self.component_base_dict[row[df_dict["SpaceHeater"].columns.get_loc("isContainedIn")]]
            rsetattr(space_heater, "outputCapacity.hasValue", row[df_dict["SpaceHeater"].columns.get_loc("outputCapacity")])

            if isinstance(row[df_dict["SpaceHeater"].columns.get_loc("temperatureClassification")], str):
                rsetattr(space_heater, "temperatureClassification.hasValue", row[df_dict["SpaceHeater"].columns.get_loc("temperatureClassification")])
            else:
                message = f"Property \"temperatureClassification\" not set for SpaceHeater object \"{space_heater.id}\""
                warnings.warn(message)
                # raise(ValueError(message))
            rsetattr(space_heater, "thermalMassHeatCapacity.hasValue", row[df_dict["SpaceHeater"].columns.get_loc("thermalMassHeatCapacity")])

        for row in df_dict["Valve"].dropna(subset=["id"]).itertuples(index=False):
            valve_name = row[df_dict["Valve"].columns.get_loc("id")]
            valve = self.component_base_dict[valve_name]

            if isinstance(row[df_dict["Valve"].columns.get_loc("subSystemOf")], str):
                systems = row[df_dict["Valve"].columns.get_loc("subSystemOf")].split(";")
                systems = [system for system_dict in self.system_dict.values() for system in system_dict.values() if system.id in systems]
                valve.subSystemOf.extend(systems)
            else:
                message = f"Required property \"subSystemOf\" not set for Valve object \"{valve.id}\""
                raise(ValueError(message))

            if isinstance(row[df_dict["Valve"].columns.get_loc("hasFluidSuppliedBy")], str):
                connected_after = row[df_dict["Valve"].columns.get_loc("hasFluidSuppliedBy")].split(";")
                connected_after = [self.component_base_dict[component_name] for component_name in connected_after]
                valve.hasFluidSuppliedBy.extend(connected_after)
            else:
                message = f"Property \"hasFluidSuppliedBy\" not set for Valve object \"{valve.id}\""
                warnings.warn(message)

            if isinstance(row[df_dict["Valve"].columns.get_loc("hasFluidReturnedBy")], str):
                connected_after = row[df_dict["Valve"].columns.get_loc("hasFluidReturnedBy")].split(";")
                connected_after = [self.component_base_dict[component_name] for component_name in connected_after]
                valve.hasFluidReturnedBy.extend(connected_after)
            else:
                message = f"Property \"hasFluidReturnedBy\" not set for Valve object \"{valve.id}\""
                warnings.warn(message)


            if isinstance(row[df_dict["Valve"].columns.get_loc("hasProperty")], str):
                properties = [self.property_dict[property_name] for property_name in row[df_dict["Valve"].columns.get_loc("hasProperty")].split(";")]
                valve.hasProperty.extend(properties)

            if isinstance(row[df_dict["Valve"].columns.get_loc("isContainedIn")], str):
                valve.isContainedIn = self.component_base_dict[row[df_dict["Valve"].columns.get_loc("isContainedIn")]]
            
            
            if isinstance(row[df_dict["Valve"].columns.get_loc("flowCoefficient")], float) and np.isnan(row[df_dict["Valve"].columns.get_loc("flowCoefficient")])==False:
                rsetattr(valve, "flowCoefficient.hasValue", row[df_dict["Valve"].columns.get_loc("flowCoefficient")])
            
            if isinstance(row[df_dict["Valve"].columns.get_loc("testPressure")], float) and np.isnan(row[df_dict["Valve"].columns.get_loc("testPressure")])==False:
                rsetattr(valve, "testPressure.hasValue", row[df_dict["Valve"].columns.get_loc("testPressure")])

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
                coil.subSystemOf.extend(systems)
            else:
                message = f"Required property \"subSystemOf\" not set for Coil object \"{coil.id}\""
                raise(ValueError(message))

            if isinstance(row[df_dict["Coil"].columns.get_loc("hasFluidSuppliedBy (airside)")], str):
                connected_after = row[df_dict["Coil"].columns.get_loc("hasFluidSuppliedBy (airside)")].split(";")
                connected_after = [self.component_base_dict[component_name] for component_name in connected_after]
                coil_ = self.component_base_dict[coil_name + " (airside)"]
                coil_.hasFluidSuppliedBy.extend(connected_after)
            else:
                message = f"Property \"hasFluidSuppliedBy (airside)\" not set for Coil object \"{coil.id}\""
                warnings.warn(message)

            if isinstance(row[df_dict["Coil"].columns.get_loc("hasFluidSuppliedBy (waterside)")], str):
                connected_after = row[df_dict["Coil"].columns.get_loc("hasFluidSuppliedBy (waterside)")].split(";")
                connected_after = [self.component_base_dict[component_name] for component_name in connected_after]
                coil_ = self.component_base_dict[coil_name + " (waterside)"]
                coil_.hasFluidSuppliedBy.extend(connected_after)
            else:
                message = f"Property \"hasFluidSuppliedBy (waterside)\" not set for Coil object \"{coil.id}\""
                warnings.warn(message)

            if isinstance(row[df_dict["Coil"].columns.get_loc("hasProperty")], str):
                properties = [self.property_dict[property_name] for property_name in row[df_dict["Coil"].columns.get_loc("hasProperty")].split(";")]
                coil.hasProperty.extend(properties)
            
        for row in df_dict["AirToAirHeatRecovery"].dropna(subset=["id"]).itertuples(index=False):
            air_to_air_heat_recovery_name = row[df_dict["AirToAirHeatRecovery"].columns.get_loc("id")]
            air_to_air_heat_recovery = self.component_base_dict[air_to_air_heat_recovery_name]
            systems = row[df_dict["AirToAirHeatRecovery"].columns.get_loc("subSystemOf")].split(";")
            systems = [system for system_dict in self.system_dict.values() for system in system_dict.values() if system.id in systems]
            air_to_air_heat_recovery.subSystemOf = systems

            if isinstance(row[df_dict["AirToAirHeatRecovery"].columns.get_loc("hasFluidSuppliedBy")], str):
                connected_after = row[df_dict["AirToAirHeatRecovery"].columns.get_loc("hasFluidSuppliedBy")].split(";")
                connected_after = [self.component_base_dict[component_name] for component_name in connected_after]
                air_to_air_heat_recovery.hasFluidSuppliedBy.extend(connected_after)
            else:
                message = f"Property \"hasFluidSuppliedBy\" not set for AirToAirHeatRecovery object \"{air_to_air_heat_recovery.id}\""
                warnings.warn(message)

            if isinstance(row[df_dict["AirToAirHeatRecovery"].columns.get_loc("hasFluidReturnedBy")], str):
                connected_after = row[df_dict["AirToAirHeatRecovery"].columns.get_loc("hasFluidReturnedBy")].split(";")
                connected_after = [self.component_base_dict[component_name] for component_name in connected_after]
                air_to_air_heat_recovery.hasFluidReturnedBy.extend(connected_after)
            else:
                message = f"Property \"hasFluidReturnedBy\" not set for AirToAirHeatRecovery object \"{air_to_air_heat_recovery.id}\""
                warnings.warn(message)
            

            properties = [self.property_dict[property_name] for property_name in row[df_dict["AirToAirHeatRecovery"].columns.get_loc("hasProperty")].split(";")]
            air_to_air_heat_recovery.hasProperty.extend(properties)
            rsetattr(air_to_air_heat_recovery, "primaryAirFlowRateMax.hasValue", row[df_dict["AirToAirHeatRecovery"].columns.get_loc("primaryAirFlowRateMax")])
            rsetattr(air_to_air_heat_recovery, "secondaryAirFlowRateMax.hasValue", row[df_dict["AirToAirHeatRecovery"].columns.get_loc("secondaryAirFlowRateMax")])

        for row in df_dict["Fan"].dropna(subset=["id"]).itertuples(index=False):
            fan_name = row[df_dict["Fan"].columns.get_loc("id")]
            fan = self.component_base_dict[fan_name]

            if isinstance(row[df_dict["Fan"].columns.get_loc("subSystemOf")], str):
                systems = row[df_dict["Fan"].columns.get_loc("subSystemOf")].split(";")
                systems = [system for system_dict in self.system_dict.values() for system in system_dict.values() if system.id in systems]
                fan.subSystemOf.extend(systems)
            else:
                message = f"Required property \"subSystemOf\" not set for fan object \"{fan.id}\""
                raise(ValueError(message))

            if isinstance(row[df_dict["Fan"].columns.get_loc("hasFluidSuppliedBy")], str):
                connected_after = row[df_dict["Fan"].columns.get_loc("hasFluidSuppliedBy")].split(";")
                connected_after = [self.component_base_dict[component_name] for component_name in connected_after]
                fan.hasFluidSuppliedBy.extend(connected_after)
            else:
                message = f"Property \"hasFluidSuppliedBy\" not set for Fan object \"{fan.id}\""
                warnings.warn(message)

            if isinstance(row[df_dict["Fan"].columns.get_loc("hasFluidReturnedBy")], str):
                connected_after = row[df_dict["Fan"].columns.get_loc("hasFluidReturnedBy")].split(";")
                connected_after = [self.component_base_dict[component_name] for component_name in connected_after]
                fan.hasFluidReturnedBy.extend(connected_after)
            else:
                message = f"Property \"hasFluidReturnedBy\" not set for Fan object \"{fan.id}\""
                warnings.warn(message)
            
            if isinstance(row[df_dict["Fan"].columns.get_loc("hasProperty")], str):
                properties = [self.property_dict[property_name] for property_name in row[df_dict["Fan"].columns.get_loc("hasProperty")].split(";")]
                fan.hasProperty.extend(properties)
            rsetattr(fan, "nominalAirFlowRate.hasValue", row[df_dict["Fan"].columns.get_loc("nominalAirFlowRate")])
            rsetattr(fan, "nominalPowerRate.hasValue", row[df_dict["Fan"].columns.get_loc("nominalPowerRate")])

            
        for row in df_dict["Controller"].dropna(subset=["id"]).itertuples(index=False):
            controller_name = row[df_dict["Controller"].columns.get_loc("id")]
            controller = self.component_base_dict[controller_name]

            if isinstance(row[df_dict["Controller"].columns.get_loc("subSystemOf")], str):
                systems = row[df_dict["Controller"].columns.get_loc("subSystemOf")].split(";")
                systems = [system for system_dict in self.system_dict.values() for system in system_dict.values() if system.id in systems]
            else:
                message = f"Required property \"subSystemOf\" not set for controller object \"{controller.id}\""
                raise(ValueError(message))
            
            controller.subSystemOf.extend(systems)

            if isinstance(row[df_dict["Controller"].columns.get_loc("isContainedIn")], str):
                controller.isContainedIn = self.component_base_dict[row[df_dict["Controller"].columns.get_loc("isContainedIn")]]

            _property = self.property_dict[row[df_dict["Controller"].columns.get_loc("observes")]]
            controller.observes = _property


            if "controls" not in df_dict["Controller"].columns:
                warnings.warn("The property \"controls\" is not found in \"Controller\" sheet. This is ignored for now but will raise an error in the future. It probably is caused by using an outdated configuration file.")
            else:
                if isinstance(row[df_dict["Controller"].columns.get_loc("controls")], str):
                    controls = row[df_dict["Controller"].columns.get_loc("controls")].split(";")
                    controls = [self.property_dict[component_name] for component_name in controls]
                    controller.controls.extend(controls)
                else:
                    message = f"Required property \"controls\" not set for controller object \"{controller.id}\""
                    raise(ValueError(message))

        for row in df_dict["SetpointController"].dropna(subset=["id"]).itertuples(index=False):
            controller_name = row[df_dict["SetpointController"].columns.get_loc("id")]
            controller = self.component_base_dict[controller_name]

            if isinstance(row[df_dict["SetpointController"].columns.get_loc("subSystemOf")], str):
                systems = row[df_dict["SetpointController"].columns.get_loc("subSystemOf")].split(";")
                systems = [system for system_dict in self.system_dict.values() for system in system_dict.values() if system.id in systems]
                controller.subSystemOf.extend(systems)
            else:
                message = f"Required property \"subSystemOf\" not set for controller object \"{controller.id}\""
                raise(ValueError(message))
            
            

            if isinstance(row[df_dict["SetpointController"].columns.get_loc("isContainedIn")], str):
                controller.isContainedIn = self.component_base_dict[row[df_dict["SetpointController"].columns.get_loc("isContainedIn")]]
            _property = self.property_dict[row[df_dict["SetpointController"].columns.get_loc("observes")]]
            controller.observes = _property

            if "controls" not in df_dict["SetpointController"].columns:
                warnings.warn("The property \"controls\" is not found in \"SetpointController\" sheet. This is ignored for now but will raise an error in the future. It probably is caused by using an outdated configuration file.")
            else:
                if isinstance(row[df_dict["SetpointController"].columns.get_loc("controls")], str):
                    controls = row[df_dict["SetpointController"].columns.get_loc("controls")].split(";")
                    controls = [self.property_dict[component_name] for component_name in controls]
                    controller.controls.extend(controls)
                else:
                    s = str(row[df_dict["SetpointController"].columns.get_loc("controls")])
                    message = f"Required property \"controls\" not set for controller object \"{controller.id}\", {s} was provided"
                    raise(ValueError(message))

            if "isReverse" not in df_dict["SetpointController"].columns:
                warnings.warn("The property \"isReverse\" is not found in \"SetpointController\" sheet. This is ignored for now but will raise an error in the future. It probably is caused by using an outdated configuration file.")
            else:
                if isinstance(row[df_dict["SetpointController"].columns.get_loc("isReverse")], bool):
                    is_reverse = row[df_dict["SetpointController"].columns.get_loc("isReverse")]
                    controller.isReverse = is_reverse
                elif isinstance(row[df_dict["SetpointController"].columns.get_loc("isReverse")], str):
                    is_reverse = row[df_dict["SetpointController"].columns.get_loc("isReverse")] in true_list
                    controller.isReverse = is_reverse
                elif isinstance(row[df_dict["SetpointController"].columns.get_loc("isReverse")], allowed_numeric_types):
                    is_reverse = int(row[df_dict["SetpointController"].columns.get_loc("isReverse")])==1
                    controller.isReverse = is_reverse
                else:
                    message = f"Required property \"isReverse\" not set to Bool value for controller object \"{controller.id}\""
                    raise(ValueError(message))

            if isinstance(row[df_dict["SetpointController"].columns.get_loc("hasProfile")], str):
                schedule_name = row[df_dict["SetpointController"].columns.get_loc("hasProfile")]
                controller.hasProfile = self.component_base_dict[schedule_name]
            else:
                message = f"Required property \"hasProfile\" not set for controller object \"{controller.id}\""
                raise(ValueError(message))

        for row in df_dict["RulebasedController"].dropna(subset=["id"]).itertuples(index=False):
            controller_name = row[df_dict["RulebasedController"].columns.get_loc("id")]
            controller = self.component_base_dict[controller_name]

            if isinstance(row[df_dict["RulebasedController"].columns.get_loc("subSystemOf")], str):
                systems = row[df_dict["RulebasedController"].columns.get_loc("subSystemOf")].split(";")
                systems = [system for system_dict in self.system_dict.values() for system in system_dict.values() if system.id in systems]
            else:
                message = f"Required property \"subSystemOf\" not set for controller object \"{controller.id}\""
                raise(ValueError(message))
            
            controller.subSystemOf.extend(systems)

            if isinstance(row[df_dict["RulebasedController"].columns.get_loc("isContainedIn")], str):
                controller.isContainedIn = self.component_base_dict[row[df_dict["RulebasedController"].columns.get_loc("isContainedIn")]]
            
            _property = self.property_dict[row[df_dict["RulebasedController"].columns.get_loc("observes")]]
            controller.observes = _property


            if "controls" not in df_dict["RulebasedController"].columns:
                warnings.warn("The property \"controls\" is not found in \"RulebasedController\" sheet. This is ignored for now but will raise an error in the future. It probably is caused by using an outdated configuration file.")
            else:
                if isinstance(row[df_dict["RulebasedController"].columns.get_loc("controls")], str):
                    controls = row[df_dict["RulebasedController"].columns.get_loc("controls")].split(";")
                    controls = [self.property_dict[component_name] for component_name in controls]
                    controller.controls.extend(controls)
                else:
                    message = f"Required property \"controls\" not set for controller object \"{controller.id}\""
                    raise(ValueError(message))

            if "isReverse" not in df_dict["RulebasedController"].columns:
                warnings.warn("The property \"isReverse\" is not found in \"RulebasedController\" sheet. This is ignored for now but will raise an error in the future. It probably is caused by using an outdated configuration file.")
            else:
                if np.isnan(row[df_dict["RulebasedController"].columns.get_loc("isReverse")])==False:
                    if isinstance(row[df_dict["RulebasedController"].columns.get_loc("isReverse")], bool):
                        is_reverse = row[df_dict["RulebasedController"].columns.get_loc("isReverse")]
                        controller.isReverse = is_reverse
                    elif isinstance(row[df_dict["RulebasedController"].columns.get_loc("isReverse")], str):
                        is_reverse = row[df_dict["RulebasedController"].columns.get_loc("isReverse")] in true_list
                        controller.isReverse = is_reverse
                    elif isinstance(row[df_dict["RulebasedController"].columns.get_loc("isReverse")], allowed_numeric_types):
                        is_reverse = int(row[df_dict["RulebasedController"].columns.get_loc("isReverse")])==1
                        controller.isReverse = is_reverse
                    else:
                        message = f"Required property \"isReverse\" not set to Bool value for controller object \"{controller.id}\""
                        raise(ValueError(message))


            if isinstance(row[df_dict["RulebasedController"].columns.get_loc("hasProfile")], str):
                schedule_name = row[df_dict["RulebasedController"].columns.get_loc("hasProfile")]
                controller.hasProfile = self.component_base_dict[schedule_name]
            else:
                message = f"Required property \"hasProfile\" not set for controller object \"{controller.id}\""
                warnings.warn(message)

            

 
        for row in df_dict["ShadingDevice"].dropna(subset=["id"]).itertuples(index=False):
            shading_device_name = row[df_dict["ShadingDevice"].columns.get_loc("id")]
            shading_device = self.component_base_dict[shading_device_name]

            if isinstance(row[df_dict["ShadingDevice"].columns.get_loc("subSystemOf")], str):
                systems = row[df_dict["ShadingDevice"].columns.get_loc("subSystemOf")].split(";")
                systems = [system for system_dict in self.system_dict.values() for system in system_dict.values() if system.id in systems]
                shading_device.subSystemOf.extend(systems)
            else:
                message = f"Required property \"subSystemOf\" not set for shading_device object \"{shading_device.id}\""
                raise(ValueError(message))

            properties = [self.property_dict[property_name] for property_name in row[df_dict["ShadingDevice"].columns.get_loc("hasProperty")].split(";")]
            shading_device.isContainedIn = self.component_base_dict[row[df_dict["ShadingDevice"].columns.get_loc("isContainedIn")]]
            shading_device.hasProperty.extend(properties)
      
        for row in df_dict["Sensor"].dropna(subset=["id"]).itertuples(index=False):
            sensor_name = row[df_dict["Sensor"].columns.get_loc("id")]
            sensor = self.component_base_dict[sensor_name]

            if isinstance(row[df_dict["Sensor"].columns.get_loc("observes")], str):
                properties = self.property_dict[row[df_dict["Sensor"].columns.get_loc("observes")]]
                sensor.observes = properties
            else:
                message = f"Required property \"observes\" not set for Sensor object \"{sensor.id}\""
                raise(ValueError(message))
            
            if isinstance(row[df_dict["Sensor"].columns.get_loc("isContainedIn")], str):
                sensor.isContainedIn = self.component_base_dict[row[df_dict["Sensor"].columns.get_loc("isContainedIn")]]
            
            if isinstance(row[df_dict["Sensor"].columns.get_loc("hasFluidSuppliedBy")], str):
                connected_after = row[df_dict["Sensor"].columns.get_loc("hasFluidSuppliedBy")].split(";")
                connected_after = [self.component_base_dict[component_name] for component_name in connected_after]
                sensor.hasFluidSuppliedBy.extend(connected_after)
            else:
                message = f"Property \"hasFluidSuppliedBy\" not set for Sensor object \"{sensor.id}\""
                warnings.warn(message)

            if isinstance(row[df_dict["Sensor"].columns.get_loc("hasFluidReturnedBy")], str):
                connected_after = row[df_dict["Sensor"].columns.get_loc("hasFluidReturnedBy")].split(";")
                connected_after = [self.component_base_dict[component_name] for component_name in connected_after]
                sensor.hasFluidReturnedBy.extend(connected_after)
            else:
                message = f"Property \"hasFluidReturnedBy\" not set for Sensor object \"{sensor.id}\""
                warnings.warn(message)

            if isinstance(row[df_dict["Sensor"].columns.get_loc("subSystemOf")], str):
                systems = row[df_dict["Sensor"].columns.get_loc("subSystemOf")].split(";")
                systems = [system for system_dict in self.system_dict.values() for system in system_dict.values() if system.id in systems]
                sensor.subSystemOf.extend(systems)
 
        for row in df_dict["Meter"].dropna(subset=["id"]).itertuples(index=False):
            meter_name = row[df_dict["Meter"].columns.get_loc("id")]
            meter = self.component_base_dict[meter_name]
            if isinstance(row[df_dict["Meter"].columns.get_loc("observes")], str):
                properties = self.property_dict[row[df_dict["Meter"].columns.get_loc("observes")]]
                meter.observes = properties
            else:
                message = f"Required property \"observes\" not set for Sensor object \"{meter.id}\""
                raise(ValueError(message))
            
            if isinstance(row[df_dict["Meter"].columns.get_loc("isContainedIn")], str):
                meter.isContainedIn = self.component_base_dict[row[df_dict["Meter"].columns.get_loc("isContainedIn")]]

            if isinstance(row[df_dict["Meter"].columns.get_loc("hasFluidSuppliedBy")], str):
                connected_after = row[df_dict["Meter"].columns.get_loc("hasFluidSuppliedBy")].split(";")
                connected_after = [self.component_base_dict[component_name] for component_name in connected_after]
                meter.hasFluidSuppliedBy.extend(connected_after)
            else:
                message = f"Property \"hasFluidSuppliedBy\" not set for Meter object \"{meter.id}\""
                warnings.warn(message)

            if isinstance(row[df_dict["Meter"].columns.get_loc("hasFluidReturnedBy")], str):
                connected_after = row[df_dict["Meter"].columns.get_loc("hasFluidReturnedBy")].split(";")
                connected_after = [self.component_base_dict[component_name] for component_name in connected_after]
                meter.hasFluidReturnedBy.extend(connected_after)
            else:
                message = f"Property \"hasFluidReturnedBy\" not set for Meter object \"{meter.id}\""
                warnings.warn(message)

            if isinstance(row[df_dict["Meter"].columns.get_loc("subSystemOf")], str):
                systems = row[df_dict["Meter"].columns.get_loc("subSystemOf")].split(";")
                systems = [system for system_dict in self.system_dict.values() for system in system_dict.values() if system.id in systems]
                meter.subSystemOf.extend(systems)

        for row in df_dict["Pump"].dropna(subset=["id"]).itertuples(index=False):
            pump_name = row[df_dict["Pump"].columns.get_loc("id")]
            pump = self.component_base_dict[pump_name]

            if isinstance(row[df_dict["Pump"].columns.get_loc("subSystemOf")], str):
                systems = row[df_dict["Pump"].columns.get_loc("subSystemOf")].split(";")
                systems = [system for system_dict in self.system_dict.values() for system in system_dict.values() if system.id in systems]
                pump.subSystemOf.extend(systems)
            else:
                message = f"Required property \"subSystemOf\" not set for Pump object \"{pump.id}\""
                raise(ValueError(message))
            
            if isinstance(row[df_dict["Pump"].columns.get_loc("hasFluidSuppliedBy")], str):
                connected_after = row[df_dict["Pump"].columns.get_loc("hasFluidSuppliedBy")].split(";")
                connected_after = [self.component_base_dict[component_name] for component_name in connected_after]
                pump.hasFluidSuppliedBy.extend(connected_after)
            else:
                message = f"Property \"hasFluidSuppliedBy\" not set for Pump object \"{pump.id}\""
                warnings.warn(message)

            if isinstance(row[df_dict["Pump"].columns.get_loc("hasFluidReturnedBy")], str):
                connected_after = row[df_dict["Pump"].columns.get_loc("hasFluidReturnedBy")].split(";")
                connected_after = [self.component_base_dict[component_name] for component_name in connected_after]
                pump.hasFluidReturnedBy.extend(connected_after)
            else:
                message = f"Property \"hasFluidReturnedBy\" not set for Pump object \"{pump.id}\""
                warnings.warn(message)
            
            if isinstance(row[df_dict["Pump"].columns.get_loc("hasProperty")], str):
                pump.hasProperty.extend(row[df_dict["Pump"].columns.get_loc("hasProperty")])


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
        df_SetpointController = pd.read_excel(semantic_model_filename, sheet_name="SetpointController") if 'SetpointController' in wb.sheetnames else pd.DataFrame([np.nan], columns=["id"])
        df_Schedule = pd.read_excel(semantic_model_filename, sheet_name="Schedule") if 'Schedule' in wb.sheetnames else pd.DataFrame([np.nan], columns=["id"])
        df_RulebasedController = pd.read_excel(semantic_model_filename, sheet_name="RulebasedController") if 'RulebasedController' in wb.sheetnames else pd.DataFrame([np.nan], columns=["id"])
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
                   "SetpointController": df_SetpointController,
                   "Schedule": df_Schedule,
                   "RulebasedController": df_RulebasedController,
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

        #print(input_dict.keys())
        
        
        if "rooms_sensor_data" in input_dict:
            """
            Reacting to different input combinations for custom ventilation system model
            """
            
            for component_id, component_data in input_dict["rooms_sensor_data"].items():
                data_available = bool(component_data["sensor_data_available"]) 
                if data_available:
                    timestamp = component_data["time"]
                    co2_values = component_data["co2"]
                    damper_values = component_data["damper_position"]
                    
                    df_co2 = pd.DataFrame({"timestamp": timestamp, "co2": co2_values})
                    df_damper = pd.DataFrame({"timestamp": timestamp, "damper_position": damper_values})
                    
                    df_co2 = sample_from_df(df_co2,
                                            stepSize=stepSize,
                                            start_time=startTime,
                                            end_time=endTime,
                                            resample=True,
                                            clip=True,
                                            tz="Europe/Copenhagen",
                                            preserve_order=True)
                    df_damper = sample_from_df(df_damper,
                                            stepSize=stepSize,
                                            start_time=startTime,
                                            end_time=endTime,
                                            resample=True,
                                            clip=True,
                                            tz="Europe/Copenhagen",
                                            preserve_order=True)
                    df_damper["damper_position"] = df_damper["damper_position"]/100
                    
                    components_ = self.component_dict.keys()
                    #Make it an array of strings
                    components_ = list(components_)
                    # find all the components that contain the last part of the component_id, after the first dash
                    
                    #Extract the substring after the first dash
                    substring_id = component_id.split("-")[1] + "-" + component_id.split("-")[2]
                    substring_id = substring_id.lower().replace("-", "_")
                    
                    # Find all the components that contain the substring
                    filtered_components = [component_ for component_ in components_ if substring_id in component_]

                    #If the component contains "co2" in the id, add the co2 data
                    for component_ in filtered_components:
                        if "CO2_sensor" in component_:
                            co2_sensor = self.component_dict[component_]
                            co2_sensor.df_input = df_co2
                        elif "Damper_position_sensor" in component_:
                            damper_position_sensor = self.component_dict[component_]
                            damper_position_sensor.df_input = df_damper
                else:
                    """
                    Reading schedules and reacting to different controller configurations
                    """
                    logger.info("[Model Class] : No sensor data available in the input dictionary, using schedules instead.")

                    rooms_volumes = {
                        "22_601b_00": 500,
                        "22_601b_0": 417,
                        "22_601b_1": 417,
                        "22_601b_2": 417,
                        "22_603_0" : 240,
                        "22_603_1" : 240,
                        "22_604_0" : 486,
                        "22_604_1" : 375,
                        "22_603b_2" : 159,
                        "22_603a_2" : 33,
                        "22_604a_2" : 33,
                        "22_604b_2" : 33,
                        "22_605a_2" : 33,
                        "22_605b_2" : 33,
                        "22_604e_2" : 33,
                        "22_604d_2" : 33,
                        "22_604c_2" : 33,
                        "22_605e_2" : 33,
                        "22_605d_2" : 33,
                        "22_605c_2" : 30,

                    }

                    components_ = self.component_dict.keys()
                    components_ = list(components_)
                    substring_id = component_id.split("-")[1] + "-" + component_id.split("-")[2]
                    substring_id = substring_id.lower().replace("-", "_")
                    substring_id = "22_" + substring_id
                    room_filtered_components = [component_ for component_ in components_ if substring_id in component_]

                    if substring_id == "22_601b_0":
                        components_601b_00 = {
                            "CO2_controller_sensor_22_601b_00",                             
                            "Damper_position_sensor_22_601b_00", 
                            "Supply_damper_22_601b_00", 
                            "Return_damper_22_601b_00", 
                            "CO2_sensor_22_601b_00"
                        }
                        filtered_components = []
                        for component_ in room_filtered_components:
                            if component_ not in components_601b_00:
                                filtered_components.append(component_)
                        room_filtered_components = filtered_components

                    controller_type = component_data["controller_type"]

                    if controller_type == "PID":
                        #Assert that a co2 setpoint schedule is available, if not raise an error with a message
                        assert "co2_setpoint_schedule" in component_data["schedules"], "No CO2 setpoint schedule in input dict. available for PID controller"
                        schedule_input = component_data["schedules"]["co2_setpoint_schedule"]
                        #Assert that the controller constants are available, if not raise an error with a message
                        assert "kp" in component_data, "No kp value in input dict. available for PID controller"
                        assert "ki" in component_data, "No ki value in input dict. available for PID controller"
                        assert "kd" in component_data, "No kd value in input dict. available for PID controller"
                        #get the id of the sensor from the filtered components
                        co2_sensor_component_id = next((component_ for component_ in room_filtered_components if "CO2_sensor" in component_), None)
                        co2_sensor_component = self.component_dict[co2_sensor_component_id]
                        co2_sensor_observed_property = co2_sensor_component.observes

                        #Create the schedule object
                        co2_setpoint_schedule = components.ScheduleSystem(
                            **schedule_input,
                            add_noise = False,
                            saveSimulationResult = True,
                            id = f"{substring_id}_co2_setpoint_schedule")
                        #Create the PID controller object
                        pid_controller = components.ControllerSystem(
                            observes = co2_sensor_observed_property,
                            K_p = component_data["kp"],
                            K_i = component_data["ki"],
                            K_d = component_data["kd"],
                            saveSimulationResult = True,
                            id = f"{substring_id}_CO2_PID_controller")
                 
                        #Remove the connections to the previous controller and delete it
                        ann_controller_component_id = next((component_ for component_ in room_filtered_components if "CO2_controller" in component_), None)
                        ann_controller_component = self.component_dict[ann_controller_component_id]
                        co2_sensor_component_id = next((component_ for component_ in room_filtered_components if "CO2_sensor" in component_), None)
                        co2_sensor_component = self.component_dict[co2_sensor_component_id]
                        supply_damper_id = next((component_ for component_ in room_filtered_components if "Supply_damper" in component_), None)
                        return_damper_id = next((component_ for component_ in room_filtered_components if "Return_damper" in component_), None)
                        supply_damper = self.component_dict[supply_damper_id]
                        return_damper = self.component_dict[return_damper_id]

                        self.remove_connection(co2_sensor_component, ann_controller_component, "indoorCo2Concentration", "actualValue")
                        self.remove_connection(ann_controller_component, return_damper, "inputSignal", "damperPosition")
                        self.remove_connection(ann_controller_component, supply_damper, "inputSignal", "damperPosition")
                        self.remove_component(ann_controller_component)

                        #Add the components to the model
                        self._add_component(co2_setpoint_schedule)
                        self._add_component(pid_controller)
                        #Add the connection between the schedule and the controller
                        self.add_connection(co2_setpoint_schedule, pid_controller, "scheduleValue", "setpointValue")
                        self.add_connection(co2_sensor_component, pid_controller, "indoorCo2Concentration", "actualValue")
                        #Add the connection between the controller and the dampers
                        self.add_connection(pid_controller, supply_damper, "inputSignal", "damperPosition")
                        self.add_connection(pid_controller, return_damper, "inputSignal", "damperPosition")
                        pid_controller.observes.isPropertyOf = co2_sensor_component
                
                        #Recalculate the filtered components
                        components_ = self.component_dict.keys()
                        components_ = list(components_)
                        substring_id = component_id.split("-")[1] + "-" + component_id.split("-")[2]
                        substring_id = substring_id.lower().replace("-", "_")
                        substring_id = "22_" + substring_id
                        room_filtered_components = [component_ for component_ in components_ if substring_id in component_]

                        if substring_id == "22_601b_0":
                            components_601b_00 = {
                                "CO2_controller_sensor_22_601b_00",                             
                                "Damper_position_sensor_22_601b_00", 
                                "Supply_damper_22_601b_00", 
                                "Return_damper_22_601b_00", 
                                "CO2_sensor_22_601b_00"
                            }
                            filtered_components = []
                            for component_ in room_filtered_components:
                                if component_ not in components_601b_00:
                                    filtered_components.append(component_)
                            room_filtered_components = filtered_components
                    elif controller_type == "RBC":
                        #Assert that a co2 setpoint schedule is available, if not raise an error with a message
                        assert "co2_setpoint_schedule" in component_data["schedules"], "No CO2 setpoint schedule in input dict. available for PID controller"
                        schedule_input = component_data["schedules"]["co2_setpoint_schedule"]
                        #get the id of the sensor from the filtered components
                        co2_sensor_component_id = next((component_ for component_ in room_filtered_components if "CO2_sensor" in component_), None)
                        co2_sensor_component = self.component_dict[co2_sensor_component_id]
                        co2_sensor_observed_property = co2_sensor_component.observes

                        #Create the schedule object
                        co2_setpoint_schedule = components.ScheduleSystem(
                            **schedule_input,
                            add_noise = False,
                            saveSimulationResult = True,
                            id = f"{substring_id}_co2_setpoint_schedule")
                        #Create the RBC controller object
                        rbc_controller = components.RulebasedSetpointInputControllerSystem(
                            observes = co2_sensor_observed_property,
                            saveSimulationResult = True,
                            id = f"{substring_id}_CO2_RBC_controller")
                        
                 
                        #Remove the connections to the previous controller and delete it
                        ann_controller_component_id = next((component_ for component_ in room_filtered_components if "CO2_controller" in component_), None)
                        ann_controller_component = self.component_dict[ann_controller_component_id]
                        co2_sensor_component_id = next((component_ for component_ in room_filtered_components if "CO2_sensor" in component_), None)
                        co2_sensor_component = self.component_dict[co2_sensor_component_id]
                        supply_damper_id = next((component_ for component_ in room_filtered_components if "Supply_damper" in component_), None)
                        return_damper_id = next((component_ for component_ in room_filtered_components if "Return_damper" in component_), None)
                        supply_damper = self.component_dict[supply_damper_id]
                        return_damper = self.component_dict[return_damper_id]

                        self.remove_connection(co2_sensor_component, ann_controller_component, "indoorCo2Concentration", "actualValue")
                        self.remove_connection(ann_controller_component, return_damper, "inputSignal", "damperPosition")
                        self.remove_connection(ann_controller_component, supply_damper, "inputSignal", "damperPosition")
                        self.remove_component(ann_controller_component)

                        #Add the components to the model
                        self._add_component(co2_setpoint_schedule)
                        self._add_component(rbc_controller)
                        #Add the connection between the schedule and the controller
                        self.add_connection(co2_setpoint_schedule, rbc_controller, "scheduleValue", "setpointValue")
                        self.add_connection(co2_sensor_component, rbc_controller, "indoorCo2Concentration", "actualValue")
                        #Add the connection between the controller and the dampers
                        self.add_connection(rbc_controller, supply_damper, "inputSignal", "damperPosition")
                        self.add_connection(rbc_controller, return_damper, "inputSignal", "damperPosition")
                        rbc_controller.observes.isPropertyOf = co2_sensor_component
                
                        #Recalculate the filtered components
                        components_ = self.component_dict.keys()
                        components_ = list(components_)
                        substring_id = component_id.split("-")[1] + "-" + component_id.split("-")[2]
                        substring_id = substring_id.lower().replace("-", "_")
                        substring_id = "22_" + substring_id
                        room_filtered_components = [component_ for component_ in components_ if substring_id in component_]

                        if substring_id == "22_601b_0":
                            components_601b_00 = {
                                "CO2_controller_sensor_22_601b_00",                             
                                "Damper_position_sensor_22_601b_00", 
                                "Supply_damper_22_601b_00", 
                                "Return_damper_22_601b_00", 
                                "CO2_sensor_22_601b_00"
                            }
                            filtered_components = []
                            for component_ in room_filtered_components:
                                if component_ not in components_601b_00:
                                    filtered_components.append(component_)
                            room_filtered_components = filtered_components
                    
                    
                    for component_ in room_filtered_components:
                        if "CO2_sensor" in component_:
                            sender_component = self.component_dict[component_]
                            receiver_component_id = next((component_ for component_ in room_filtered_components if "_controller" in component_), None)
                            receiver_component = self.component_dict[receiver_component_id]
                            self.remove_connection(sender_component, receiver_component, "indoorCo2Concentration", "actualValue")
                            self.remove_component(sender_component)
                            
                            schedule_input = component_data["schedules"]["occupancy_schedule"] 
                            #Create the schedule object
                            room_occupancy_schedule = components.ScheduleSystem(
                                **schedule_input,
                                add_noise = False,
                                saveSimulationResult = True,
                                id = f"{substring_id}_occupancy_schedule")
                            #Create the space co2 object
                            room_volume = int(rooms_volumes[substring_id]) 
                            room_space_co2 = components.BuildingSpaceCo2System(
                                airVolume = room_volume,
                                saveSimulationResult = True,
                                id = f"{substring_id}_CO2_space")
                            
                            self._add_component(room_occupancy_schedule)
                            self._add_component(room_space_co2)

                            supply_damper_id = next((component_ for component_ in room_filtered_components if "Supply_damper" in component_), None)
                            return_damper_id = next((component_ for component_ in room_filtered_components if "Return_damper" in component_), None)
                            supply_damper = self.component_dict[supply_damper_id]
                            return_damper = self.component_dict[return_damper_id]

                            self.add_connection(room_occupancy_schedule, room_space_co2,
                                "scheduleValue", "numberOfPeople")
                            self.add_connection(supply_damper, room_space_co2,
                                "airFlowRate", "supplyAirFlowRate")
                            self.add_connection(return_damper, room_space_co2,
                                "airFlowRate", "returnAirFlowRate")
                            self.add_connection(room_space_co2, receiver_component,
                                "indoorCo2Concentration", "actualValue")
                            
                            receiver_component.observes.isPropertyOf = room_space_co2           


            ## ADDED FOR DAMPER CONTROL of 601b_00, missing data

            oe_601b_00_component = self.component_dict["CO2_controller_sensor_22_601b_00"]
            freq = f"{stepSize}S"
            df_damper_control = pd.DataFrame({"timestamp": pd.date_range(start=startTime, end=endTime, freq=freq)})
            df_damper_control["damper_position"] = 1
            
            df_damper_control = sample_from_df(df_damper_control,
                        stepSize=stepSize,
                        start_time=startTime,
                        end_time=endTime,
                        resample=True,
                        clip=True,
                        tz="Europe/Copenhagen",
                        preserve_order=True)
            oe_601b_00_component.df_input = df_damper_control

        else:
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

            outdoor_environment = components.OutdoorEnvironmentSystem(df_input=df_sample,
                                                            saveSimulationResult = self.saveSimulationResult,
                                                            id = "outdoor_environment")

            

            '''
                MODIFIED BY NEC-INDIA

                temperature_list = sensor_inputs["ml_inputs"]["temperature"]
            '''
            if sensor_inputs["ml_inputs"]["temperature"][0]=="None":
                initial_temperature = 21
            else:
                initial_temperature = float(sensor_inputs["ml_inputs"]["temperature"][0])
            custom_initial_dict = {"OE20-601b-2": {"indoorTemperature": initial_temperature}}
            self.set_custom_initial_dict(custom_initial_dict)

            indoor_temperature_setpoint_schedule = components.ScheduleSystem(
                **schedule_inputs["temperature_setpoint_schedule"],
                add_noise = False,
                saveSimulationResult = True,
                id = "OE20-601b-2_temperature_setpoint_schedule")

            occupancy_schedule = components.ScheduleSystem(
                **schedule_inputs["occupancy_schedule"],
                add_noise = True,
                saveSimulationResult = True,
                id = "OE20-601b-2_occupancy_schedule")

            supply_water_temperature_setpoint_schedule = components.PiecewiseLinearScheduleSystem(
                **schedule_inputs["supply_water_temperature_schedule_pwlf"],
                saveSimulationResult = True,
                id = "Heating system_supply_water_temperature_schedule")
            
            supply_air_temperature_schedule = components.ScheduleSystem(
                **schedule_inputs["supply_air_temperature_schedule"],
                saveSimulationResult = True,
                id = "Ventilation system_supply_air_temperature_schedule")

            # self._add_component(outdoor_environment)
            # self._add_component(occupancy_schedule)
            # self._add_component(indoor_temperature_setpoint_schedule)
            # self._add_component(supply_water_temperature_setpoint_schedule)
            # self._add_component(supply_air_temperature_schedule)


            space = self.component_dict["OE20-601b-2"]
            space_heater = self.component_dict["Space heater"]
            temperature_controller = self.component_dict["Temperature controller"]
            # outdoor_environment = self.component_dict["outdoor_environment"]
            # supply_water_temperature_setpoint_schedule = self.component_dict["Heating system_supply_water_temperature_schedule"]
            # supply_air_temperature_schedule = self.component_dict["Ventilation system_supply_air_temperature_schedule"]
            # indoor_temperature_setpoint_schedule = self.component_dict["OE20-601b-2_temperature_setpoint_schedule"]
            # occupancy_schedule = self.component_dict["OE20-601b-2_occupancy_schedule"]
    

            self.add_connection(outdoor_environment, space, "outdoorTemperature", "outdoorTemperature")
            self.add_connection(outdoor_environment, space, "globalIrradiation", "globalIrradiation")
            self.add_connection(outdoor_environment, supply_water_temperature_setpoint_schedule, "outdoorTemperature", "outdoorTemperature")
            self.add_connection(supply_water_temperature_setpoint_schedule, space_heater, "scheduleValue", "supplyWaterTemperature")
            self.add_connection(supply_water_temperature_setpoint_schedule, space, "scheduleValue", "supplyWaterTemperature")
            self.add_connection(supply_air_temperature_schedule, space, "scheduleValue", "supplyAirTemperature")
            self.add_connection(indoor_temperature_setpoint_schedule, temperature_controller, "scheduleValue", "setpointValue")
            self.add_connection(occupancy_schedule, space, "scheduleValue", "numberOfPeople")

        logger.info("[Model Class] : Exited from read_input_config Function")

    def update_attribute(self, component, attribute, value):
        """
        This method updates the value of an attribute of a component.
        """
        if component is None:
            return
            
        attr = rgetattr(component, attribute)
        if isinstance(attr, list):
            if isinstance(value, list):
                for v in value:
                    if v not in attr:
                        attr.append(v)
            if value not in attr:
                attr.append(value)
        else:
            rsetattr(component, attribute, value)

    def parse_semantic_model(self):
        space_instances = self.get_component_by_class(self.component_base_dict, base.BuildingSpace)
        damper_instances = self.get_component_by_class(self.component_base_dict, base.Damper)
        space_heater_instances = self.get_component_by_class(self.component_base_dict, base.SpaceHeater)
        valve_instances = self.get_component_by_class(self.component_base_dict, base.Valve)
        coil_instances = self.get_component_by_class(self.component_base_dict, base.Coil)
        air_to_air_heat_recovery_instances = self.get_component_by_class(self.component_base_dict, base.AirToAirHeatRecovery)
        fan_instances = self.get_component_by_class(self.component_base_dict, base.Fan)
        controller_instances = self.get_component_by_class(self.component_base_dict, base.Controller)
        setpoint_controller_instances = self.get_component_by_class(self.component_base_dict, base.SetpointController)
        rulebased_controller_instances = self.get_component_by_class(self.component_base_dict, base.RulebasedController)
        shading_device_instances = self.get_component_by_class(self.component_base_dict, base.ShadingDevice)
        sensor_instances = self.get_component_by_class(self.component_base_dict, base.Sensor)
        meter_instances = self.get_component_by_class(self.component_base_dict, base.Meter)
        pump_instances = self.get_component_by_class(self.component_base_dict, base.Pump)

        for space in space_instances:
            for property_ in space.hasProperty:
                self.update_attribute(property_, "isPropertyOf", space)
            for component in space.hasFluidSuppliedBy:
                self.update_attribute(component, "suppliesFluidTo", space)
            space.hasFluidFedBy = space.hasFluidSuppliedBy
            for component in space.hasFluidFedBy:
                self.update_attribute(component, "feedsFluidTo", space)
            for component in space.connectedTo:
                self.update_attribute(component, "connectedTo", space)
            
        for damper in damper_instances:
            self.update_attribute(damper, "isContainedIn.contains", damper)
            for system in damper.subSystemOf:
                self.update_attribute(system, "hasSubSystem", damper)
            for property_ in damper.hasProperty:
                self.update_attribute(property_, "isPropertyOf", damper)
            for component in damper.hasFluidSuppliedBy:
                self.update_attribute(component, "suppliesFluidTo", damper)
            for component in damper.hasFluidReturnedBy:
                self.update_attribute(component, "returnsFluidTo", damper)

            damper.hasFluidFedBy = damper.hasFluidSuppliedBy
            for component in damper.hasFluidFedBy:
                self.update_attribute(component, "feedsFluidTo", damper)

        for space_heater in space_heater_instances:
            for component in space_heater.hasFluidSuppliedBy:
                self.update_attribute(component, "suppliesFluidTo", space_heater)
            space_heater.hasFluidFedBy = space_heater.hasFluidSuppliedBy
            for component in space_heater.hasFluidFedBy:
                self.update_attribute(component, "feedsFluidTo", space_heater)
            self.update_attribute(space_heater, "isContainedIn.contains", space_heater)
            for system in space_heater.subSystemOf:
                self.update_attribute(system, "hasSubSystem", space_heater)
            for property_ in space_heater.hasProperty:
                self.update_attribute(property_, "isPropertyOf", space_heater)

        for valve in valve_instances:
            self.update_attribute(valve.isContainedIn, "contains", valve)
            for system in valve.subSystemOf:
                self.update_attribute(system, "hasSubSystem", valve)
            for property_ in valve.hasProperty:
                self.update_attribute(property_, "isPropertyOf", valve)
            for component in valve.hasFluidSuppliedBy:
                self.update_attribute(component, "suppliesFluidTo", valve)
            for component in valve.hasFluidReturnedBy:
                self.update_attribute(component, "returnsFluidTo", valve)
            valve.hasFluidFedBy = valve.hasFluidSuppliedBy
            for component in valve.hasFluidFedBy:
                self.update_attribute(component, "feedsFluidTo", valve)

        for coil in coil_instances:
            for system in coil.subSystemOf:
                self.update_attribute(system, "hasSubSystem", coil)
            for property_ in coil.hasProperty:
                self.update_attribute(property_, "isPropertyOf", coil)

            for component in coil.hasFluidSuppliedBy:
                self.update_attribute(component, "suppliesFluidTo", coil)
            for component in coil.hasFluidReturnedBy:
                self.update_attribute(component, "returnsFluidTo", coil)

            coil.hasFluidFedBy = coil.hasFluidSuppliedBy
            for component in coil.hasFluidFedBy:
                self.update_attribute(component, "feedsFluidTo", coil)

        for air_to_air_heat_recovery in air_to_air_heat_recovery_instances:
            for system in air_to_air_heat_recovery.subSystemOf:
                self.update_attribute(system, "hasSubSystem", air_to_air_heat_recovery)
            for property_ in air_to_air_heat_recovery.hasProperty:
                self.update_attribute(property_, "isPropertyOf", air_to_air_heat_recovery)
            for component in air_to_air_heat_recovery.hasFluidFedBy:
                self.update_attribute(component, "feedsFluidTo", air_to_air_heat_recovery)

            for component in air_to_air_heat_recovery.hasFluidSuppliedBy:
                self.update_attribute(component, "suppliesFluidTo", air_to_air_heat_recovery)
            for component in air_to_air_heat_recovery.hasFluidReturnedBy:
                self.update_attribute(component, "returnsFluidTo", air_to_air_heat_recovery)

            air_to_air_heat_recovery.hasFluidFedBy = air_to_air_heat_recovery.hasFluidSuppliedBy
            for component in air_to_air_heat_recovery.hasFluidFedBy:
                self.update_attribute(component, "feedsFluidTo", air_to_air_heat_recovery)
                component.feedsFluidTo.append(air_to_air_heat_recovery)

        for fan in fan_instances:
            for system in fan.subSystemOf:
                self.update_attribute(system, "hasSubSystem", fan)
            for property_ in fan.hasProperty:
                self.update_attribute(property_, "isPropertyOf", fan)
            for component in fan.hasFluidFedBy:
                self.update_attribute(component, "feedsFluidTo", fan)

            for component in fan.hasFluidSuppliedBy:
                self.update_attribute(component, "suppliesFluidTo", fan)
            for component in fan.hasFluidReturnedBy:
                self.update_attribute(component, "returnsFluidTo", fan)

            fan.hasFluidFedBy = fan.hasFluidSuppliedBy
            for component in fan.hasFluidFedBy:
                self.update_attribute(component, "feedsFluidTo", fan)

        for controller in controller_instances:
            if controller.isContainedIn is not None:
                self.update_attribute(controller.isContainedIn, "contains", controller)
            self.update_attribute(controller.observes, "isObservedBy", controller)
            for property_ in controller.controls:
                self.update_attribute(property_, "isControlledBy", controller)
            for system in controller.subSystemOf:
                self.update_attribute(system, "hasSubSystem", controller)

        for setpoint_controller in setpoint_controller_instances:
            self.update_attribute(setpoint_controller.isContainedIn, "contains", setpoint_controller)
            self.update_attribute(setpoint_controller, "observes.isObservedBy", setpoint_controller)
            for property_ in setpoint_controller.controls:
                self.update_attribute(property_, "isControlledBy", setpoint_controller)
            for system in setpoint_controller.subSystemOf:
                self.update_attribute(system, "hasSubSystem", setpoint_controller)


        for rulebased_controller in rulebased_controller_instances:
            self.update_attribute(rulebased_controller.isContainedIn, "contains", rulebased_controller)
            self.update_attribute(rulebased_controller.observes, "isObservedBy", rulebased_controller)
            for property_ in rulebased_controller.controls:
                self.update_attribute(property_, "isControlledBy", rulebased_controller)
            for system in rulebased_controller.subSystemOf:
                self.update_attribute(system, "hasSubSystem", rulebased_controller)

        for shading_device in shading_device_instances:
            self.update_attribute(shading_device.isContainedIn, "contains", shading_device)
            for system in shading_device.subSystemOf:
                self.update_attribute(system, "hasSubSystem", shading_device)
            for property_ in shading_device.hasProperty:
                self.update_attribute(property_, "isPropertyOf", shading_device)

        for sensor in sensor_instances:
            self.update_attribute(sensor.isContainedIn, "contains", sensor)
            self.update_attribute(sensor.observes, "isObservedBy", sensor)
            for system in sensor.subSystemOf:
                self.update_attribute(system, "hasSubSystem", sensor)
            for component in sensor.hasFluidFedBy:
                self.update_attribute(component, "feedsFluidTo", sensor)

            for component in sensor.hasFluidSuppliedBy:
                self.update_attribute(component, "suppliesFluidTo", sensor)
            for component in sensor.hasFluidReturnedBy:
                self.update_attribute(component, "returnsFluidTo", sensor)

            sensor.hasFluidFedBy = sensor.hasFluidSuppliedBy
            for component in sensor.hasFluidFedBy:
                self.update_attribute(component, "feedsFluidTo", sensor)


        for meter in meter_instances:
            self.update_attribute(meter.isContainedIn, "contains", meter)
            self.update_attribute(meter.observes, "isObservedBy", meter)
            for system in meter.subSystemOf:
                self.update_attribute(system, "hasSubSystem", meter)
            for component in meter.hasFluidFedBy:
                self.update_attribute(component, "feedsFluidTo", meter)

            for component in meter.hasFluidSuppliedBy:
                self.update_attribute(component, "suppliesFluidTo", meter)
            for component in meter.hasFluidReturnedBy:
                self.update_attribute(component, "returnsFluidTo", meter)

            
            meter.hasFluidFedBy = meter.hasFluidSuppliedBy
            for component in meter.hasFluidFedBy:
                self.update_attribute(component, "feedsFluidTo", meter)

        for pump in pump_instances:
            for system in pump.subSystemOf:
                self.update_attribute(system, "hasSubSystem", pump)
            for component in pump.hasFluidFedBy:
                self.update_attribute(component, "feedsFluidTo", pump)
            for property_ in pump.hasProperty:
                self.update_attribute(property_, "isPropertyOf", pump)

            for component in pump.hasFluidSuppliedBy:
                self.update_attribute(component, "suppliesFluidTo", pump)
            for component in pump.hasFluidReturnedBy:
                self.update_attribute(component, "returnsFluidTo", pump)

            pump.hasFluidFedBy = pump.hasFluidSuppliedBy
            for component in pump.hasFluidFedBy:
                self.update_attribute(component, "feedsFluidTo", pump)
        for heating_system in self.system_dict["heating"].values():
            self._add_object(heating_system)

        for cooling_system in self.system_dict["cooling"].values():
            self._add_object(cooling_system)

        for ventilation_system in self.system_dict["ventilation"].values():
            self._add_object(ventilation_system)
        
        logger.info("[Model Class] : Exited from Apply Model Extensions Function")
        
    def apply_model_extensions(self):
        logger.info("[Model Class] : Entered in Apply Model Extensions Function")
        space_instances = self.get_component_by_class(self.component_base_dict, base.BuildingSpace)
        damper_instances = self.get_component_by_class(self.component_base_dict, base.Damper)
        space_heater_instances = self.get_component_by_class(self.component_base_dict, base.SpaceHeater)
        valve_instances = self.get_component_by_class(self.component_base_dict, base.Valve)
        coil_instances = self.get_component_by_class(self.component_base_dict, base.Coil)
        air_to_air_heat_recovery_instances = self.get_component_by_class(self.component_base_dict, base.AirToAirHeatRecovery)
        fan_instances = self.get_component_by_class(self.component_base_dict, base.Fan)
        controller_instances = self.get_component_by_class(self.component_base_dict, base.Controller)
        shading_device_instances = self.get_component_by_class(self.component_base_dict, base.ShadingDevice)
        sensor_instances = self.get_component_by_class(self.component_base_dict, base.Sensor)
        meter_instances = self.get_component_by_class(self.component_base_dict, base.Meter)

        for space in space_instances:
            base_kwargs = self.get_object_properties(space)
            extension_kwargs = {
                "saveSimulationResult": self.saveSimulationResult,
            }
            base_kwargs.update(extension_kwargs)
            space = components.BuildingSpaceSystem(**base_kwargs)
            self._add_component(space)
            for property_ in space.hasProperty:
                property_.isPropertyOf = space
            for component in space.hasFluidFedBy:
                component.feedsFluidTo.append(space)
            
        for damper in damper_instances:
            base_kwargs = self.get_object_properties(damper)
            extension_kwargs = {
                "a": 1,
                "saveSimulationResult": self.saveSimulationResult,
            }
            base_kwargs.update(extension_kwargs)
            damper = components.DamperSystem(**base_kwargs)
            self._add_component(damper)
            damper.isContainedIn = self.component_dict[damper.isContainedIn.id]
            damper.isContainedIn.contains.append(damper)
            for system in damper.subSystemOf:
                system.hasSubSystem.append(damper)
            for property_ in damper.hasProperty:
                property_.isPropertyOf = damper
            for component in damper.hasFluidReturnedBy:
                component.returnsFluidTo.append(damper)
            damper.hasFluidFedBy = damper.hasFluidReturnedBy
            

        for space_heater in space_heater_instances:
            base_kwargs = self.get_object_properties(space_heater)
            extension_kwargs = {
                "saveSimulationResult": self.saveSimulationResult,
            }
            base_kwargs.update(extension_kwargs)
            space_heater = components.SpaceHeaterSystem(**base_kwargs)
            space_heater.heatTransferCoefficient = 8.31495759e+01
            space_heater.thermalMassHeatCapacity.hasvalue = 2.72765272e+06
            for component in space_heater.hasFluidSuppliedBy:
                component.suppliesFluidTo.append(space_heater)

            space_heater.hasFluidFedBy = space_heater.hasFluidSuppliedBy

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
            valve = components.ValveSystem(**base_kwargs)
            self._add_component(valve)
            valve.isContainedIn = self.component_dict[valve.isContainedIn.id]
            valve.isContainedIn.contains.append(valve)
            for system in valve.subSystemOf:
                system.hasSubSystem.append(valve)
            for property_ in valve.hasProperty:
                property_.isPropertyOf = valve
            for component in valve.hasFluidFedBy:
                component.feedsFluidTo.append(valve)

        for coil in coil_instances:
            base_kwargs = self.get_object_properties(coil)
            extension_kwargs = {
                "saveSimulationResult": self.saveSimulationResult,
            }
            base_kwargs.update(extension_kwargs)
            if len([v for v in coil.subSystemOf if v in self.system_dict["heating"].values()])==1:
                coil = components.CoilHeatingSystem(**base_kwargs)
            elif len([v for v in coil.subSystemOf if v in self.system_dict["cooling"].values()])==1:
                coil = components.CoilCoolingSystem(**base_kwargs)
            else:
                raise(ValueError(f"The system of the Coil with id \"{coil.id}\" is not set."))
            self._add_component(coil)
            for system in coil.subSystemOf:
                system.hasSubSystem.append(coil)
            for property_ in coil.hasProperty:
                property_.isPropertyOf = coil
            for component in coil.hasFluidFedBy:
                component.feedsFluidTo.append(coil)

        for air_to_air_heat_recovery in air_to_air_heat_recovery_instances:
            base_kwargs = self.get_object_properties(air_to_air_heat_recovery)
            extension_kwargs = {
                "specificHeatCapacityAir": base.PropertyValue(hasValue=1000),
                "eps_75_h": 0.84918046,
                "eps_75_c": 0.82754917,
                "eps_100_h": 0.85202735,
                "eps_100_c": 0.8215695,
                "saveSimulationResult": self.saveSimulationResult,
            }
            base_kwargs.update(extension_kwargs)
            air_to_air_heat_recovery = components.AirToAirHeatRecoverySystem(**base_kwargs)
            self._add_component(air_to_air_heat_recovery)
            for system in air_to_air_heat_recovery.subSystemOf:
                system.hasSubSystem.append(air_to_air_heat_recovery)
            for property_ in air_to_air_heat_recovery.hasProperty:
                property_.isPropertyOf = air_to_air_heat_recovery
            for component in air_to_air_heat_recovery.hasFluidFedBy:
                component.feedsFluidTo.append(air_to_air_heat_recovery)

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
            fan = components.FanSystem(**base_kwargs)
            self._add_component(fan)
            for system in fan.subSystemOf:
                system.hasSubSystem.append(fan)
            for property_ in fan.hasProperty:
                property_.isPropertyOf = fan
            for component in fan.hasFluidFedBy:
                component.feedsFluidTo.append(fan)

        for controller in controller_instances:
            base_kwargs = self.get_object_properties(controller)
            if isinstance(controller.observes, base.Temperature):
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
                controller = components.ControllerSystem(**base_kwargs)
            elif isinstance(controller.observes, base.Co2):
                extension_kwargs = {
                    "saveSimulationResult": self.saveSimulationResult,
                }
                base_kwargs.update(extension_kwargs)
                controller = components.RulebasedControllerSystem(**base_kwargs)
            self._add_component(controller)
            controller.isContainedIn = self.component_dict[controller.isContainedIn.id]
            controller.isContainedIn.contains.append(controller)
            controller.observes.isControlledBy = self.component_dict[controller.id]
            for system in controller.subSystemOf:
                system.hasSubSystem.append(controller)

        for shading_device in shading_device_instances:
            base_kwargs = self.get_object_properties(shading_device)
            extension_kwargs = {
                "saveSimulationResult": self.saveSimulationResult,
            }
            base_kwargs.update(extension_kwargs)
            shading_device = components.ShadingDeviceSystem(**base_kwargs)
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
            sensor = components.SensorSystem(**base_kwargs)
            self._add_component(sensor)
            if sensor.isContainedIn is not None:
                sensor.isContainedIn = self.component_dict[sensor.isContainedIn.id]
                sensor.isContainedIn.contains.append(sensor)
            sensor.observes.isObservedBy = self.component_dict[sensor.id]
            for system in sensor.subSystemOf:
                system.hasSubSystem.append(sensor)
            for component in sensor.hasFluidFedBy:
                component.feedsFluidTo.append(sensor)

        for meter in meter_instances:
            base_kwargs = self.get_object_properties(meter)
            extension_kwargs = {
                "saveSimulationResult": self.saveSimulationResult,
            }
            base_kwargs.update(extension_kwargs)
            meter = components.MeterSystem(**base_kwargs)
            self._add_component(meter)
            if meter.isContainedIn is not None:
                meter.isContainedIn = self.component_dict[meter.isContainedIn.id]
                meter.isContainedIn.contains.append(meter)
            meter.observes.isObservedBy = self.component_dict[meter.id]
            for system in meter.subSystemOf:
                system.hasSubSystem.append(meter)
            for component in meter.hasFluidFedBy:
                component.feedsFluidTo.append(meter)
        for ventilation_system in self.system_dict["ventilation"].values():
            node_S = components.FlowJunctionSystem(
                    subSystemOf = [ventilation_system],
                    operationMode = "supply",
                    saveSimulationResult = self.saveSimulationResult,
                    id = "Supply node") ####
            self._add_component(node_S)
            ventilation_system.hasSubSystem.append(node_S)
            node_E = components.FlowJunctionSystem(
                    subSystemOf = [ventilation_system],
                    operationMode = "return",
                    saveSimulationResult = self.saveSimulationResult,
                    id = "Exhaust node") ####
            self._add_component(node_E)
            ventilation_system.hasSubSystem.append(node_E)
        for component in self.component_dict.values():
            connectedTo_new = []
            if component.connectedTo is not None:
                for base_component in component.connectedTo:
                    connectedTo_new.append(self.component_dict[base_component.id])
            component.connectedTo = connectedTo_new

        for component in self.component_dict.values():
            feedsFluidTo_new = []
            if len(component.feedsFluidTo)>0:
                for base_component in component.feedsFluidTo:
                    feedsFluidTo_new.append(self.component_dict[base_component.id])
            component.feedsFluidTo = feedsFluidTo_new

        for component in self.component_dict.values():
            hasFluidFedBy_new = []
            if len(component.hasFluidFedBy)>0:
                for base_component in component.hasFluidFedBy:
                    hasFluidFedBy_new.append(self.component_dict[base_component.id])
            component.hasFluidFedBy = hasFluidFedBy_new
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
            filter = lambda v, class_: True
        return [v for v in dict_.values() if (isinstance(v, class_) and filter(v, class_))]

    def get_dampers_by_space(self, space):
        return [component for component in space.contains if isinstance(component, base.Damper)]

    def get_space_heaters_by_space(self, space):
        return [component for component in space.contains if isinstance(component, base.SpaceHeater)]

    def get_valves_by_space(self, space):
        return [component for component in space.contains if isinstance(component, base.Valve)]

    def get_controllers_by_space(self, space):
        return [component for component in space.contains if isinstance(component, base.Controller)]

    def get_shading_devices_by_space(self, space):
        return [component for component in space.contains if isinstance(component, base.ShadingDevice)]

    def _get_leaf_node_old(self, component, last_component, ref_component, found_ref=False):
        if isinstance(component, base.AirToAirHeatRecovery) or len(component.connectedTo)<2:
            node = component
            found_ref = True if component is ref_component else False
        else:
            for connected_component in component.connectedTo:
                if connected_component is not last_component:
                    if isinstance(connected_component, base.AirToAirHeatRecovery)==False and len(connected_component.connectedTo)>1:
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
            if isinstance(node, base.Damper):
                if found_ref:
                    if node.operationMode=="supply":
                        placement = "before"
                        side = "supply"
                    else:
                        placement = "after"
                        side = "return"
                else:
                    if node.operationMode=="supply":
                        placement = "after"
                        side = "supply"
                    else:
                        placement = "before"
                        side = "return"
                break

            elif isinstance(node, components.OutdoorEnvironmentSystem):
                if found_ref:
                    placement = "after"
                    side = "supply"
                else:
                    placement = "before"
                    side = "supply"
                break

            elif isinstance(node, base.AirToAirHeatRecovery):
                saved_found_ref = found_ref

        if placement is None:
            if saved_found_ref:
                placement = "after"
                side = "return"
            else:
                placement = "before"
                side = "return"
                
        return placement, side

    def _get_leaf_nodes_before(self, ref_component, component, leaf_nodes=None, found_ref=False):
        if leaf_nodes is None:
            leaf_nodes = []
        if len(component.hasFluidFedBy)>0:
            for connected_component in component.hasFluidFedBy:
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
        if len(component.feedsFluidTo)>0:
            for connected_component in component.feedsFluidTo:
                if connected_component.id not in visited:
                    visited.append(connected_component.id)
                    leaf_nodes, visited = self._get_leaf_nodes_after(connected_component, leaf_nodes, visited)
                else:
                    visited.append(connected_component.id)
                    raise RecursionError(f"The component of class \"{connected_component.__class__.__name__}\" with id \"{connected_component.id}\" is part of a cycle. The following components form a cycle with the \"feedsFluidTo\" property: {' -> '.join(visited)}")
        else:
            leaf_nodes.append(component)
        return leaf_nodes, visited

    def component_is_before(self, ref_component, component, is_before=False):
        if len(component.hasFluidFedBy)>0:
            for connected_component in component.hasFluidFedBy:
                is_before = True if connected_component is ref_component else False
                is_before = self.component_is_before(ref_component, connected_component, is_before=is_before)
        return is_before

    def component_is_after(self, ref_component, component, is_after=False):
        if len(component.feedsFluidTo)>0:
            for connected_component in component.feedsFluidTo:
                is_after = True if connected_component is ref_component else False
                is_after = self.component_is_after(ref_component, connected_component, is_after=is_after)
        return is_after
    
    def _classes_are_before(self, ref_classes, component, is_before=False, visited=None):
        if visited is None:
            visited = [component.id]
        if len(component.hasFluidFedBy)>0:
            for connected_component in component.hasFluidFedBy:
                if connected_component.id not in visited:
                    is_before = True if istype(connected_component, ref_classes) else False
                    if is_before==False:
                        is_before = self._classes_are_before(ref_classes, connected_component, is_before, visited)
                else:
                    visited.append(connected_component.id)
                    raise RecursionError(f"The component of class \"{connected_component.__class__.__name__}\" with id \"{connected_component.id}\" is part of a cycle. The following components form a cycle with the \"feedsFluidTo\" property: {' -> '.join(visited)}")

        return is_before

    def _classes_are_after(self, ref_classes, component, is_after=False, visited=None):
        if visited is None:
            visited = [component.id]
        if len(component.feedsFluidTo)>0:
            for connected_component in component.feedsFluidTo:
                if connected_component.id not in visited:
                    visited.append(connected_component.id)
                    is_after = True if istype(connected_component, ref_classes) else False
                    if is_after==False:
                        is_after = self._classes_are_after(ref_classes, connected_component, is_after, visited)
                else:
                    visited.append(connected_component.id)
                    raise RecursionError(f"The component of class \"{connected_component.__class__.__name__}\" with id \"{connected_component.id}\" is part of a cycle. The following components form a cycle with the \"feedsFluidTo\" property: {' -> '.join(visited)}")

        return is_after

    def _get_instance_of_type_before(self, ref_classes, component, found_instance=None, visited=None):
        if found_instance is None:
            found_instance = []
        if visited is None:
            visited = [component.id]
        if len(component.hasFluidFedBy)>0:
            for connected_component in component.hasFluidFedBy:
                if connected_component.id not in visited:
                    visited.append(connected_component.id)
                    is_type = True if istype(connected_component, ref_classes) else False
                    if is_type==False:
                        found_instance = self._get_instance_of_type_before(ref_classes, connected_component, found_instance, visited)
                    else:
                        found_instance.append(connected_component)
                else:
                    visited.append(connected_component.id)
                    raise RecursionError(f"The component of class \"{connected_component.__class__.__name__}\" with id \"{connected_component.id}\" is part of a cycle. The following components form a cycle with the \"feedsFluidTo\" property: {' -> '.join(visited)}")

        return found_instance

    def _get_instance_of_type_after(self, ref_classes, component, found_instance=None, visited=None):
        if found_instance is None:
            found_instance = []
        if visited is None:
            visited = [component.id]
        if len(component.feedsFluidTo)>0:
            for connected_component in component.feedsFluidTo:
                if connected_component.id not in visited:
                    visited.append(connected_component.id)
                    is_type = True if istype(connected_component, ref_classes) else False
                    if is_type==False:
                        found_instance = self._get_instance_of_type_after(ref_classes, connected_component, found_instance, visited)
                    else:
                        found_instance.append(connected_component)
                else:
                    visited.append(connected_component.id)
                    raise RecursionError(f"The component of class \"{connected_component.__class__.__name__}\" with id \"{connected_component.id}\" is part of a cycle. The following components form a cycle with the \"feedsFluidTo\" property: {' -> '.join(visited)}")

        return found_instance

    def _get_flow_placement(self, component):
        """
         _______________________________________________________
        |                                                       |
    ref | ------------------------------------> flow direction  | component
        |_______________________________________________________|

        The above example would yield placement = "after"
        """
        if self._classes_are_after((components.BuildingSpaceSystem, ), component):
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




    def connect_new(self):
        def copy_nodemap(nodemap):
            return {k: v.copy() for k, v in nodemap.items()}
        
        def _prune_recursive(match_node, sp_node, node_map, node_map_list, feasible, comparison_table, ruleset):
            match_node_id = match_node.id if "id" in get_object_attributes(match_node) else match_node.__class__.__name__ + " [" + str(id(match_node)) +"]"
            # print(f"sp_node: {sp_node.id} match_node: {match_node_id}")
            if sp_node not in feasible: feasible[sp_node] = set()
            if sp_node not in comparison_table: comparison_table[sp_node] = set()
            feasible[sp_node].add(match_node)
            comparison_table[sp_node].add(match_node)
            

            match_name_attributes = get_object_attributes(match_node)


            sp_node_pairs = sp_node.attributes
            sp_node_pairs_ = sp_node._attributes
            sp_node_pairs_list = sp_node._list_attributes
            if len(sp_node_pairs)==0:
                node_map[sp_node] = {match_node}
                node_map_list = [node_map]


            # print("so far feasible:", [a.id if "id" in get_object_attributes(a) else a.__class__.__name__ + " [" + str(id(a)) +"]" for f in feasible.values() for a in f])

            # print("ENTERED===============")
            # for sp_node_, match_node_set in node_map.items():
                # id_sp = sp_node_.id
                # id_sp = id_sp.replace(r"\n", "")
                # id_m = [match_node.id if "id" in get_object_attributes(match_node) else match_node.__class__.__name__ + " [" + str(id(match_node)) +"]" for match_node in match_node_set]
                # print(id_sp, id_m)

            for sp_attr_name, sp_node_child in sp_node_pairs_.items(): #iterate the required attributes/predicates of the signature node
                # print("sp_attr_name1:", sp_attr_name)
                # print("match_name_attributes1:", match_name_attributes)
                if sp_attr_name in match_name_attributes: #is there a match with the semantic node?
                    match_node_child = rgetattr(match_node, sp_attr_name)
                    if match_node_child is not None:
                        rule = ruleset[(sp_node, sp_node_child, sp_attr_name)]
                        pairs, rule_applies, ruleset = rule.apply(match_node, match_node_child, ruleset, node_map=node_map)
                        if len(pairs)==1:
                            filtered_match_node_child, filtered_sp_node_child = next(iter(pairs))
                            if filtered_match_node_child not in comparison_table[sp_node_child]:
                                node_map_list, node_map, feasible, comparison_table, prune = _prune_recursive(filtered_match_node_child, filtered_sp_node_child, copy_nodemap(node_map), node_map_list, feasible, comparison_table, ruleset)
                                if prune and isinstance(rule, signature_pattern.Optional)==False:
                                    feasible[sp_node].discard(match_node)
                                    # print("PRUNING1:", f"sp_node: {sp_node.id} match_node: {match_node_id}")
                                    return node_map_list, node_map, feasible, comparison_table, True
                                else:
                                    for node_map_ in node_map_list:
                                        node_map_[sp_node] = {match_node} #Multiple nodes might be added if multiple branches match
                            elif filtered_match_node_child in feasible[sp_node_child]:
                                node_map_list = [node_map] if len(node_map_list)==0 else node_map_list ####################################################################################
                                for node_map_ in node_map_list:
                                    node_map_[sp_node] = {match_node} #Multiple nodes might be added if multiple branches match
                                    node_map_[sp_node_child] = {filtered_match_node_child}
                            else:
                                feasible[sp_node].discard(match_node)
                                # print("PRUNING2:", f"sp_node: {sp_node.id} match_node: {match_node_id}")
                                return node_map_list, node_map, feasible, comparison_table, True
                        else:
                            feasible[sp_node].discard(match_node)
                            # print("PRUNING3:", f"sp_node: {sp_node.id} match_node: {match_node_id}")
                            return node_map_list, node_map, feasible, comparison_table, True
                    else:
                        if isinstance(sp_node_child, list):
                            for sp_node_child_ in sp_node_child:
                                rule = ruleset[(sp_node, sp_node_child_, sp_attr_name)]
                                if isinstance(rule, signature_pattern.Optional)==False:
                                    feasible[sp_node].discard(match_node)
                                    # print("PRUNING4:", f"sp_node: {sp_node.id} match_node: {match_node_id}")
                                    return node_map_list, node_map, feasible, comparison_table, True
                        else:
                            rule = ruleset[(sp_node, sp_node_child, sp_attr_name)]
                            if isinstance(rule, signature_pattern.Optional)==False:
                                feasible[sp_node].discard(match_node)
                                # print("PRUNING5:", f"sp_node: {sp_node.id} match_node: {match_node_id}")
                                return node_map_list, node_map, feasible, comparison_table, True
                else:
                    if isinstance(sp_node_child, list):
                        for sp_node_child_ in sp_node_child:
                            rule = ruleset[(sp_node, sp_node_child_, sp_attr_name)]
                            if isinstance(rule, signature_pattern.Optional)==False:
                                feasible[sp_node].discard(match_node)
                                # print("PRUNING6:", f"sp_node: {sp_node.id} match_node: {match_node_id}")
                                return node_map_list, node_map, feasible, comparison_table, True
                    else:
                        rule = ruleset[(sp_node, sp_node_child, sp_attr_name)]
                        if isinstance(rule, signature_pattern.Optional)==False:
                            feasible[sp_node].discard(match_node)
                            # print("PRUNING7:", f"sp_node: {sp_node.id} match_node: {match_node_id}")
                            return node_map_list, node_map, feasible, comparison_table, True
                        
            for sp_attr_name, sp_node_child in sp_node_pairs_list.items(): #iterate the required attributes/predicates of the signature node
                # print("sp_attr_name2:", sp_attr_name)
                # print("match_name_attributes2:", match_name_attributes)
                if sp_attr_name in match_name_attributes: #is there a match with the semantic node?
                    match_node_child = rgetattr(match_node, sp_attr_name)
                    if match_node_child is not None:
                        for sp_node_child_ in sp_node_child:
                            rule = ruleset[(sp_node, sp_node_child_, sp_attr_name)]
                            pairs, rule_applies, ruleset = rule.apply(match_node, match_node_child, ruleset, node_map=node_map)
                            found = False
                            new_node_map_list = []
                            for filtered_match_node_child, filtered_sp_node_child in pairs:
                                # print("filtered_match_node_child:", filtered_match_node_child)
                                if filtered_match_node_child not in comparison_table[sp_node_child_]:
                                    # print("ENTERED recursive")
                                    comparison_table[sp_node_child_].add(filtered_match_node_child)
                                    
                                    node_map_list_, node_map, feasible, comparison_table, prune = _prune_recursive(filtered_match_node_child, filtered_sp_node_child, copy_nodemap(node_map), node_map_list.copy(), feasible, comparison_table, ruleset)
                                        
                                    if found and prune==False:# and isinstance(rule, signature_pattern.MultipleMatches)==False:
                                        name = match_node.id if "id" in get_object_attributes(match_node) else match_node.__class__.__name__
                                        warnings.warn(f"Multiple matches found for context signature node \"{sp_node.id}\" and semantic model node \"{name}\".")
                                    
                                    if prune==False: #be careful here - multiple branches might match - how to account for?
                                        
                                        new_node_map_list.extend(node_map_list_)
                                        # print("EXTENDED list")
                                        # print(id(a) for a in new_node_map_list)
                                        found = True

                                elif filtered_match_node_child in feasible[sp_node_child_]:
                                    # print("ENTERED feasible")
                                    node_map[sp_node_child_] = {filtered_match_node_child}
                                    new_node_map_list.extend([copy_nodemap(node_map)])
                                    found = True

                            if found==False and isinstance(rule, signature_pattern.Optional)==False:
                                feasible[sp_node].discard(match_node)
                                # print("PRUNING8:", f"sp_node: {sp_node.id} match_node: {match_node_id}")
                                return node_map_list, node_map, feasible, comparison_table, True
                            else:
                                node_map_list = new_node_map_list
                                for node_map_ in node_map_list:
                                    node_map_[sp_node] = set() if sp_node not in node_map_ else node_map_[sp_node]
                                    node_map_[sp_node].add(match_node) #Multiple nodes might be added if multiple branches match
                    else:
                        if isinstance(sp_node_child, list):
                            for sp_node_child_ in sp_node_child:
                                rule = ruleset[(sp_node, sp_node_child_, sp_attr_name)]
                                if isinstance(rule, signature_pattern.Optional)==False:
                                    feasible[sp_node].discard(match_node)
                                    # print("PRUNING9:", f"sp_node: {sp_node.id} match_node: {match_node_id}")
                                    return node_map_list, node_map, feasible, comparison_table, True
                        else:
                            rule = ruleset[(sp_node, sp_node_child, sp_attr_name)]
                            if isinstance(rule, signature_pattern.Optional)==False:
                                feasible[sp_node].discard(match_node)
                                # print("PRUNING10:", f"sp_node: {sp_node.id} match_node: {match_node_id}")
                                return node_map_list, node_map, feasible, comparison_table, True
                else:
                    if isinstance(sp_node_child, list):
                        for sp_node_child_ in sp_node_child:
                            rule = ruleset[(sp_node, sp_node_child_, sp_attr_name)]
                            if isinstance(rule, signature_pattern.Optional)==False:
                                feasible[sp_node].discard(match_node)
                                # print("PRUNING11:", f"sp_node: {sp_node.id} match_node: {match_node_id}")
                                return node_map_list, node_map, feasible, comparison_table, True
                    else:
                        rule = ruleset[(sp_node, sp_node_child, sp_attr_name)]
                        if isinstance(rule, signature_pattern.Optional)==False:
                            feasible[sp_node].discard(match_node)
                            # print("PRUNING12:", f"sp_node: {sp_node.id} match_node: {match_node_id}")
                            return node_map_list, node_map, feasible, comparison_table, True

            # print("Returning")
            # for i, node_map_ in enumerate(node_map_list):
            #     print(f"-----nodemap {i} ------")
            #     for sp_node_, match_node_set in node_map.items():
            #         id_sp = sp_node_.id
            #         id_sp = id_sp.replace(r"\n", "")
            #         id_m = [match_node.id if "id" in get_object_attributes(match_node) else match_node.__class__.__name__ + " [" + str(id(match_node)) +"]" for match_node in match_node_set]
            #         print(id_sp, id_m)
            return node_map_list, node_map, feasible, comparison_table, False


        def match(group, node_map_, sp, cg, new_ig):
            can_match = all([group[sp_node_]==node_map_[sp_node_] if len(group[sp_node_])!=0 and len(node_map_[sp_node_])!=0 else True for sp_node_ in sp.nodes])
            is_match = False
            if can_match:
                node_map_no_None = {sp_node_: match_node_set for sp_node_,match_node_set in node_map_.items() if len(match_node_set)!=0}

                # print("CAN MATCH")
                # print("--------- node_map_no_None -------------")
                # for sp_node_, match_node_set in node_map_no_None.items():
                #     id_sp = sp_node_.id
                #     id_sp = id_sp.replace(r"\n", "")
                #     id_m = [match_node.id if "id" in get_object_attributes(match_node) else match_node.__class__.__name__ + " [" + str(id(match_node)) +"]" for match_node in match_node_set]
                #     print(id_sp, id_m)
                
                # print("--------- group -------------")
                # for sp_node_, match_node_set in group.items():
                #     id_sp = sp_node_.id if "id" in get_object_attributes(sp_node_) else sp_node_.__class__.__name__ + " [" + str(id(sp_node_)) +"]"
                #     id_sp = id_sp.replace(r"\n", "")
                #     id_m = [match_node.id if "id" in get_object_attributes(match_node) else match_node.__class__.__name__ + " [" + str(id(match_node)) +"]" for match_node in match_node_set]
                #     print(id_sp, id_m)

                is_match = False
                for sp_node_, match_node_set_nm in node_map_no_None.items():
                    attributes = sp_node_.attributes
                    for attr, subject in attributes.items():
                        for match_node_nm in match_node_set_nm:
                            node_map_child = getattr(match_node_nm, attr)
                            if node_map_child is not None and (isinstance(node_map_child, list) and len(node_map_child)==0)==False:
                                if isinstance(node_map_child, list)==False:
                                    node_map_child_ = [node_map_child]
                                else:
                                    node_map_child_ = node_map_child
                                if isinstance(subject, list)==False:
                                    subject_ = [subject]
                                else:
                                    subject_ = subject
                        
                                for subject__ in subject_:
                                    group_child_ = group[subject__]
                                    if len(group_child_)!=0 and len(node_map_child_)!=0:
                                        for group_child__ in group_child_:
                                            for node_map_child__ in node_map_child_:
                                                if group_child__ is node_map_child__: #Only check equality of objects for first match
                                                    is_match = True
                                                        
                                                if is_match:
                                                    break
                                            if is_match:
                                                break

                                    if is_match:
                                        break
                            if is_match:
                                break
                        if is_match:
                            break
                    if is_match:
                        break
                
                if is_match:
                    for sp_node_, match_node_set_nm in node_map_no_None.items():
                        for match_node_ in match_node_set_nm:
                            feasible = {sp_node: set() for sp_node in sp.nodes}
                            comparison_table = {sp_node: set() for sp_node in sp.nodes}
                            # match_node_id = match_node_.id if "id" in get_object_attributes(match_node_) else match_node_.__class__.__name__ + " [" + str(id(match_node_)) +"]"

                            # print("TRYING TO PRUNE 111111111111111111111111111111111111111111111111111111111111111")
                            # print(f"sp_node: {subject__.id} match_node: {match_node_id}")
                            sp.reset_ruleset()
                            group_prune = copy_nodemap(group)
                            group_prune = {sp_node___: group_prune[sp_node___] for sp_node___ in sp.nodes}
                            _, _, _, _, prune = _prune_recursive(match_node_, sp_node_, group_prune, [], feasible, comparison_table, sp.ruleset)
                            # prune = False
                            if prune:
                                is_match = False
                                break
                        if is_match==False:
                            break
                            
                    if is_match:
                        # print("--------------------------------MATCHED-----------------------------------------")
                        for sp_node__, match_node__ in node_map_no_None.items(): #Add all elements
                            group[sp_node__] = match_node__
                        if all([len(group[sp_node_])!=0 for sp_node_ in sp.nodes]):
                            cg.append(group)
                            new_ig.remove(group)
                
                if is_match==False:
                    is_match = False
                    group_no_None = {sp_node_: match_node_set for sp_node_,match_node_set in group.items() if len(match_node_set)!=0}
                    for sp_node_, match_node_set_group in group_no_None.items():
                        attributes = sp_node_.attributes
                        for attr, subject in attributes.items():
                            for match_node_group in match_node_set_group:
                                group_child = getattr(match_node_group, attr)
                                if group_child is not None and (isnumeric(group_child) and math.isnan(group_child))==False and (isinstance(group_child, list) and len(group_child)==0)==False:
                                    if isinstance(group_child, list)==False:
                                        group_child_ = [group_child]
                                    else:
                                        group_child_ = group_child
                                    if isinstance(subject, list)==False:
                                        subject_ = [subject]
                                    else:
                                        subject_ = subject

                                    for subject__ in subject_:
                                        node_map_child_ = node_map_[subject__]
                                        if len(group_child_)!=0 and len(node_map_child_)!=0:
                                            for group_child__ in group_child_:
                                                for node_map_child__ in node_map_child_:
                                                    if group_child__ is node_map_child__: #Only check equality of objects for first match
                                                        is_match = True

                                                    if is_match:
                                                        break
                                                if is_match:
                                                    break

                                        if is_match:
                                            break
                                if is_match:
                                    break
                            if is_match:
                                break
                        if is_match:
                            break
                    if is_match:
                        for sp_node_, match_node_set_nm in node_map_no_None.items():
                            for match_node_ in match_node_set_nm:
                                feasible = {sp_node: set() for sp_node in sp.nodes}
                                comparison_table = {sp_node: set() for sp_node in sp.nodes}
                                # match_node_id = match_node_.id if "id" in get_object_attributes(match_node_) else match_node_.__class__.__name__ + " [" + str(id(match_node_)) +"]"

                                # print("TRYING TO PRUNE 2222222222222222222222222222222222222222222222222222222")
                                # print(f"sp_node: {subject__.id} match_node: {match_node_id}")
                                sp.reset_ruleset()
                                group_prune = copy_nodemap(group)
                                group_prune = {sp_node___: group_prune[sp_node___] for sp_node___ in sp.nodes}
                                _, _, _, _, prune = _prune_recursive(match_node_, sp_node_, group_prune, [], feasible, comparison_table, sp.ruleset)
                                # prune = False
                                if prune:
                                    is_match = False
                                    break
                            if is_match==False:
                                break

                        if is_match:
                            # print("--------------------------------MATCHED123-----------------------------------------")
                            for sp_node__, match_node__ in node_map_no_None.items(): #Add all elements
                                group[sp_node__] = match_node__
                            if all([len(group[sp_node_])!=0 for sp_node_ in sp.nodes]):
                                cg.append(group)
                                new_ig.remove(group)
            return is_match, group, cg, new_ig

        classes = [cls[1] for cls in inspect.getmembers(components, inspect.isclass) if (issubclass(cls[1], (System, )) and hasattr(cls[1], "sp"))]
        complete_groups = {}
        incomplete_groups = {}
        counter = 0
        for component_cls in classes:
            # print(component_cls.__name__)
            complete_groups[component_cls] = {}
            incomplete_groups[component_cls] = {}
            sps = component_cls.sp
            for sp in sps:
                complete_groups[component_cls][sp] = []
                incomplete_groups[component_cls][sp] = []
                cg = complete_groups[component_cls][sp]
                ig = incomplete_groups[component_cls][sp]
                feasible = {sp_node: set() for sp_node in sp.nodes}
                comparison_table = {sp_node: set() for sp_node in sp.nodes}
                for sp_node in sp.nodes:
                    # print("START: SP NODE: ", sp_node.id) if component_cls is components.CoilPumpValveFMUSystem else None
                    l = list(sp_node.cls)
                    l.remove(signature_pattern.NodeBase)
                    l = tuple(l)
                    match_nodes = [c for c in self.object_dict.values() if (isinstance(c, l)) and isinstance(c, signature_pattern.NodeBase)==False] ######################################
                        
                    for match_node in match_nodes:
                        # print("MATCH NODE: ", match_node.id if "id" in get_object_attributes(match_node) else match_node.__class__.__name__ + " [" + str(id(match_node)) +"]") if component_cls is components.CoilPumpValveFMUSystem else None
                        node_map = {sp_node_: set() for sp_node_ in sp.nodes}
                        node_map_list = []
                        prune = True
                        if match_node not in comparison_table[sp_node]:
                            # print("ENTERED prune")
                            sp.reset_ruleset()
                            node_map_list, node_map, feasible, comparison_table, prune = _prune_recursive(match_node, sp_node, node_map, node_map_list, feasible, comparison_table, sp.ruleset)
                            # print("EXITED prune") if component_cls is components.CoilPumpValveFMUSystem else None
                            # print("PRUNE: ", prune) if component_cls is components.CoilPumpValveFMUSystem else None

                            
                        elif match_node in feasible[sp_node]:
                            # print("IS FEASIBLE") if component_cls is components.CoilPumpValveFMUSystem else None
                            node_map[sp_node] = {match_node}
                            node_map_list = [node_map]
                            prune = False
                        # print("prune: ", prune) if component_cls is components.CoilPumpValveFMUSystem else None
                        
                        if prune==False:
                            modeled_nodes = []
                            for node_map_ in node_map_list:
                                
                                node_map_set = set()
                                for sp_modeled_node in sp.modeled_nodes:
                                    node_map_set.update(node_map_[sp_modeled_node])
                                modeled_nodes.append(node_map_set)
                            node_map_list_new = []
                            for i,(node_map_, node_map_set) in enumerate(zip(node_map_list, modeled_nodes)):
                                active_set = node_map_set
                                passive_set = set().union(*[v for k,v in enumerate(modeled_nodes) if k!=i])
                                if len(active_set.intersection(passive_set))==0:
                                    node_map_list_new.append(node_map_)
                            node_map_list = node_map_list_new
                            
                            # print("LEN list: ", len(node_map_list)) if component_cls is components.CoilPumpValveFMUSystem else None
                            for node_map_ in node_map_list:
                                if all([len(match_node_set)!=0 for sp_node_,match_node_set in node_map_.items()]):#all([sp_node_ in node_map_ for sp_node_ in sp.nodes]):
                                    cg.append(node_map_)
                                else:
                                    if len(ig)==0: #If there are no groups in the incomplete group list, add the node map
                                        ig.append(node_map_)
                                    else:
                                        new_ig = ig.copy()
                                        is_match_ = False
                                        for group in ig: #Iterate over incomplete groups
                                            is_match, group, cg, new_ig = match(group, node_map_, sp, cg, new_ig)
                                            if is_match:
                                                is_match_ = True
                                        if is_match_==False:
                                            new_ig.append(node_map_)
                                        ig = new_ig
                                        # print("MATCHED: ", is_match_) if component_cls is components.CoilPumpValveFMUSystem else None
                                        # for sp_node_, match_node_set in node_map_.items():
                                            # id_sp = sp_node_.id if "id" in get_object_attributes(sp_node_) else sp_node_.__class__.__name__ + " [" + str(id(sp_node_)) +"]"
                                            # id_sp = id_sp.replace(r"\n", "")
                                            # id_m = [match_node.id if "id" in get_object_attributes(match_node) else match_node.__class__.__name__ + " [" + str(id(match_node)) +"]" for match_node in match_node_set]
                                            # print(id_sp, id_m, "comparison: ", [m in comparison_table[sp_node_] if sp_node_ in comparison_table else None for m in match_node_set], "feasible: ", [m in feasible[sp_node_] if sp_node_ in comparison_table else None for m in match_node_set])
                
                
                ig_len = np.inf
                while len(ig)<ig_len:
                    ig_len = len(ig)
                    new_ig = ig.copy()
                    is_match = False
                    for group_i in ig:
                        for group_j in ig:
                            if group_i!=group_j:
                                is_match, group, cg, new_ig = match(group_i, group_j, sp, cg, new_ig)
                                # if is_match:
                                    # new_ig.remove(group_j) #Not removed in match - therefore we remove it here
                            if is_match:
                                break
                        if is_match:
                            break
                    ig = new_ig
                    
                
                # if component_cls is components.BuildingSpace1AdjBoundaryFMUSystem:
                #     print("INCOMPLETE GROUPS================================================================================")
                #     for group in ig:
                #         print("GROUP------------------------------")
                #         for sp_node_, match_node_set in group.items():
                #             id_sp = sp_node_.id if "id" in get_object_attributes(sp_node_) else sp_node_.__class__.__name__ + " [" + str(id(sp_node_)) +"]"
                #             id_sp = id_sp.replace(r"\n", "")
                #             id_m = [match_node.id if "id" in get_object_attributes(match_node) else match_node.__class__.__name__ + " [" + str(id(match_node)) +"]" for match_node in match_node_set]
                #             print(id_sp, id_m, "comparison: ", [m in comparison_table[sp_node_] if sp_node_ in comparison_table else None for m in match_node_set], "feasible: ", [m in feasible[sp_node_] if sp_node_ in comparison_table else None for m in match_node_set])


                #     print("COMPLETE GROUPS================================================================================")
                #     for group in cg:
                #         print("GROUP------------------------------")
                #         for sp_node_, match_node_set in group.items():
                #             id_sp = sp_node_.id if "id" in get_object_attributes(sp_node_) else sp_node_.__class__.__name__ + " [" + str(id(sp_node_)) +"]"
                #             id_sp = id_sp.replace(r"\n", "")
                #             id_m = [match_node.id if "id" in get_object_attributes(match_node) else match_node.__class__.__name__ + " [" + str(id(match_node)) +"]" for match_node in match_node_set]
                #             print(id_sp, id_m)


                    
                
                
                new_ig = ig.copy()
                for group in ig: #Iterate over incomplete groups
                    if all([len(group[sp_node_])!=0 for sp_node_ in sp.required_nodes]):
                        cg.append(group)
                        new_ig.remove(group)
                ig = new_ig


        for component_cls, sps in complete_groups.items():
            complete_groups[component_cls] = {sp: groups for sp, groups in sorted(complete_groups[component_cls].items(), key=lambda item: item[0].priority, reverse=True)}
        complete_groups = {k: v for k, v in sorted(complete_groups.items(), key=lambda item: max(sp.priority for sp in item[1]), reverse=True)}
        self.instance_map = {}
        self.instance_map_reversed = {}
        self.instance_to_group_map = {} ############### if changed to self.instance_to_group_map, it cannot be pickled
        modeled_components = set()
        for i, (component_cls, sps) in enumerate(complete_groups.items()):
            for sp, groups in sps.items():
                for group in groups:
                    modeled_match_nodes = {c for sp_node in sp.modeled_nodes for c in group[sp_node]}
                    if len(modeled_components.intersection(modeled_match_nodes))==0:
                        modeled_components |= modeled_match_nodes #Union/add set
                        if len(modeled_match_nodes)==1:
                            component = next(iter(modeled_match_nodes))
                            id_ = component.id
                            base_kwargs = self.get_object_properties(component)
                            extension_kwargs = {"id": id_,
                                                "saveSimulationResult": self.saveSimulationResult,}
                        else:
                            id_ = ""
                            modeled_match_nodes_sorted = sorted(modeled_match_nodes, key=lambda x: x.id)
                            for component in modeled_match_nodes_sorted:
                                id_ += f"[{component.id}]"
                            base_kwargs = {}
                            extension_kwargs = {"id": id_,
                                                "saveSimulationResult": self.saveSimulationResult,
                                                "base_components": list(modeled_match_nodes_sorted)}
                            for component in modeled_match_nodes_sorted:
                                kwargs = self.get_object_properties(component)
                                base_kwargs.update(kwargs)
                        base_kwargs.update(extension_kwargs)
                        component = component_cls(**base_kwargs)
                        # print("INSTANCE CREATED: ", component.id)
                        self.instance_to_group_map[component] = (modeled_match_nodes, (component_cls, sp, group))
                        self.instance_map[component] = modeled_match_nodes
                        for modeled_match_node in modeled_match_nodes:
                            self.instance_map_reversed[modeled_match_node] = component
        for component, (modeled_match_nodes, (component_cls, sp, group)) in self.instance_to_group_map.items():
            for key, (sp_node, source_keys) in sp.inputs.items():
                match_node_set = group[sp_node]
                if match_node_set.issubset(modeled_components):
                    for component_inner, (modeled_match_nodes_inner, group_inner) in self.instance_to_group_map.items():
                        if match_node_set.issubset(modeled_match_nodes_inner) and component_inner is not component:
                            source_key = [source_key for c, source_key in source_keys.items() if isinstance(component_inner, c)][0]
                            self.add_connection(component_inner, component, source_key, key)
                else:
                    for match_node in match_node_set:
                        warnings.warn(f"\nThe component with class \"{match_node.__class__.__name__}\" and id \"{match_node.id}\" is not modeled. The input \"{key}\" of the component with class \"{component_cls.__name__}\" and id \"{component.id}\" is not connected.\n")
            for key, node in sp.parameters.items():
                if len(group[node])==1:
                    (value, ) = group[node] #unpack set
                    rsetattr(component, key, value)




    def connect(self):
        """
        Connects component instances using the saref4syst extension.
        """
        logger.info("[Model Class] : Entered in Connect Function")

        import twin4build.components as components
        space_models = Model.get_component_by_class(Model.component_dict, components.BuildingSpace11AdjBoundaryOutdoorFMUSystem)
        for space_model in space_models:
            space_model.id

        space_instances = self.get_component_by_class(self.component_dict, components.BuildingSpaceSystem)
        damper_instances = self.get_component_by_class(self.component_dict, components.DamperSystem)
        space_heater_instances = self.get_component_by_class(self.component_dict, components.SpaceHeaterSystem)
        valve_instances = self.get_component_by_class(self.component_dict, components.ValveSystem)
        coil_heating_instances = self.get_component_by_class(self.component_dict, components.CoilHeatingSystem)
        coil_cooling_instances = self.get_component_by_class(self.component_dict, components.CoilCoolingSystem)
        air_to_air_heat_recovery_instances = self.get_component_by_class(self.component_dict, components.AirToAirHeatRecoverySystem)
        fan_instances = self.get_component_by_class(self.component_dict, components.FanSystem)
        controller_instances = self.get_component_by_class(self.component_dict, base.Controller)
        shading_device_instances = self.get_component_by_class(self.component_dict, components.ShadingDeviceSystem)
        sensor_instances = self.get_component_by_class(self.component_dict, components.SensorSystem)#, filter=lambda v: isinstance(v.observes, base.Pressure)==False)
        meter_instances = self.get_component_by_class(self.component_dict, components.MeterSystem)
        node_instances = self.get_component_by_class(self.component_dict, components.FlowJunctionSystem)



        flow_temperature_change_types = (components.AirToAirHeatRecoverySystem, components.CoilHeatingSystem, components.CoilCoolingSystem)
        flow_change_types = (components.FlowJunctionSystem, )

        outdoor_environment = self.component_dict["outdoor_environment"]
        
        for id, heating_system in self.system_dict["heating"].items():
            supply_water_temperature_setpoint_schedule = self.get_supply_water_temperature_setpoint_schedule(id)
            self.add_connection(outdoor_environment, supply_water_temperature_setpoint_schedule, "outdoorTemperature", "outdoorTemperature")

        for space in space_instances:
            dampers = self.get_dampers_by_space(space)
            valves = self.get_valves_by_space(space)
            shading_devices = self.get_shading_devices_by_space(space)

            for damper in dampers:
                if space in damper.feedsFluidTo:
                    self.add_connection(damper, space, "airFlowRate", "supplyAirFlowRate")
                    self.add_connection(damper, space, "damperPosition", "supplyDamperPosition")
                    ventilation_system = damper.subSystemOf[0] #Logic might be needed here in the future if multiple systems are returned
                    supply_air_temperature_setpoint_schedule = self.get_supply_air_temperature_setpoint_schedule(ventilation_system.id)
                    self.add_connection(supply_air_temperature_setpoint_schedule, space, "scheduleValue", "supplyAirTemperature")
                    
                elif space in damper.hasFluidFedBy:
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
            controller = [controller for controller in controllers if isinstance(controller.observes, base.Co2)]
            if len(controller)!=0:
                controller = controller[0]
                self.add_connection(controller, damper, "inputSignal", "damperPosition")
            else:
                filename = os.path.join(os.path.abspath(uppath(os.path.abspath(__file__), 2)), "test", "data", "time_series_data", "OE20-601b-2_Damper position.csv")
                warnings.warn(f"No CO2 controller found in BuildingSpace: \"{damper.isContainedIn.id}\".\nAssigning historic values by file: \"{filename}\"")
                if " Damper position data" not in self.component_dict:
                    damper_position_schedule = components.TimeSeriesInputSystem(id=" Damper position data", filename=filename)
                    self._add_component(damper_position_schedule)
                else:
                    damper_position_schedule = self.component_dict[" Damper position data"]
                self.add_connection(damper_position_schedule, damper, "damperPosition", "damperPosition")

        for space_heater in space_heater_instances:
            valve = space_heater.hasFluidFedBy[0] #The space heater is currently a terminal component meaning that only 1 component can be connected
            self.add_connection(space, space_heater, "indoorTemperature", "indoorTemperature") 
            self.add_connection(valve, space_heater, "waterFlowRate", "waterFlowRate")
            heating_system = [v for v in space_heater.subSystemOf if v in self.system_dict["heating"].values()][0]
            supply_water_temperature_setpoint_schedule = self.get_supply_water_temperature_setpoint_schedule(heating_system.id)
            self.add_connection(supply_water_temperature_setpoint_schedule, space_heater, "scheduleValue", "supplyWaterTemperature")
            
        for valve in valve_instances:
            controllers = self.get_controllers_by_space(valve.isContainedIn)
            controller = [controller for controller in controllers if isinstance(controller.observes, base.Temperature)]
            if len(controller)!=0:
                controller = controller[0]
                self.add_connection(controller, valve, "inputSignal", "valvePosition")
            else:
                filename = os.path.join(os.path.abspath(uppath(os.path.abspath(__file__), 2)), "test", "data", "time_series_data", "OE20-601b-2_Space heater valve position.csv")
                warnings.warn(f"No Temperature controller found in BuildingSpace: \"{valve.isContainedIn.id}\".\nAssigning historic values by file: \"{filename}\"")
                if "Valve position schedule" not in self.component_dict:
                    valve_position_schedule = components.TimeSeriesInputSystem(id="Valve position schedule", filename=filename)
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
                if isinstance(instance_of_type_before, components.AirToAirHeatRecoverySystem):
                    self.add_connection(instance_of_type_before, coil_heating, "primaryTemperatureOut", "inletAirTemperature")
                elif isinstance(instance_of_type_before, components.FanSystem):
                    self.add_connection(instance_of_type_before, coil_heating, "outletAirTemperature", "inletAirTemperature")
                elif isinstance(instance_of_type_before, components.CoilHeatingSystem):
                    self.add_connection(instance_of_type_before, coil_heating, "outletAirTemperature", "inletAirTemperature")
                elif isinstance(instance_of_type_before, components.CoilCoolingSystem):
                    self.add_connection(instance_of_type_before, coil_heating, "outletAirTemperature", "inletAirTemperature")
            ventilation_system = [v for v in coil_heating.subSystemOf if v in self.system_dict["ventilation"].values()][0]
            supply_air_temperature_setpoint_schedule = self.get_supply_air_temperature_setpoint_schedule(ventilation_system.id)
            self.add_connection(supply_air_temperature_setpoint_schedule, coil_heating, "scheduleValue", "outletAirTemperatureSetpoint")
            supply_node = [node for node in node_instances if node.operationMode=="supply"][0]
            self.add_connection(supply_node, coil_heating, "flowRate", "airFlowRate")

        for coil_cooling in coil_cooling_instances:
            instance_of_type_before = self._get_instance_of_type_before(flow_temperature_change_types, coil_cooling)
            if len(instance_of_type_before)==0:
                self.add_connection(outdoor_environment, coil_cooling, "outdoorTemperature", "inletAirTemperature")
            else:
                instance_of_type_before = instance_of_type_before[0]
                if isinstance(instance_of_type_before, components.AirToAirHeatRecoverySystem):
                    self.add_connection(instance_of_type_before, coil_cooling, "primaryTemperatureOut", "inletAirTemperature")
                elif isinstance(instance_of_type_before, components.FanSystem):
                    self.add_connection(instance_of_type_before, coil_cooling, "outletAirTemperature", "inletAirTemperature")
                elif isinstance(instance_of_type_before, components.CoilHeatingSystem):
                    self.add_connection(instance_of_type_before, coil_cooling, "outletAirTemperature", "inletAirTemperature")
                elif isinstance(instance_of_type_before, components.CoilCoolingSystem):
                    self.add_connection(instance_of_type_before, coil_cooling, "outletAirTemperature", "inletAirTemperature")
            ventilation_system = [v for v in coil_cooling.subSystemOf if v in self.system_dict["ventilation"].values()][0]
            supply_air_temperature_setpoint_schedule = self.get_supply_air_temperature_setpoint_schedule(ventilation_system.id)
            self.add_connection(supply_air_temperature_setpoint_schedule, coil_cooling, "scheduleValue", "outletAirTemperatureSetpoint")
            supply_node = [node for node in node_instances if node.operationMode=="supply"][0]
            self.add_connection(supply_node, coil_cooling, "flowRate", "airFlowRate")

        for air_to_air_heat_recovery in air_to_air_heat_recovery_instances:
            ventilation_system = air_to_air_heat_recovery.subSystemOf[0]
            node_S = [v for v in ventilation_system.hasSubSystem if isinstance(v, components.FlowJunctionSystem) and v.operationMode == "supply"][0]
            node_E = [v for v in ventilation_system.hasSubSystem if isinstance(v, components.FlowJunctionSystem) and v.operationMode == "return"][0]
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
            property_ = controller.observes
            property_of = property_.isPropertyOf
            measuring_device = property_.isObservedBy
            if isinstance(controller, components.RulebasedControllerSystem)==False:
                if isinstance(property_of, base.BuildingSpace):
                    if isinstance(property_, base.Temperature):
                        self.add_connection(measuring_device, controller, "indoorTemperature", "actualValue")
                        setpoint_schedule = self.get_indoor_temperature_setpoint_schedule(property_of.id)
                    elif isinstance(property_, base.Co2):
                        self.add_connection(measuring_device, controller, "indoorCo2Concentration", "actualValue")
                        setpoint_schedule = self.get_co2_setpoint_schedule(property_of.id)
                elif isinstance(property_of, components.CoilHeatingSystem):
                    system = [v for v in property_of.subSystemOf if v in self.system_dict["heating"].values()][0]
                    setpoint_schedule = self.get_supply_air_temperature_setpoint_schedule(system.id)
                elif isinstance(property_of, components.CoilCoolingSystem):
                    system = [v for v in property_of.subSystemOf if v in self.system_dict["cooling"].values()][0]
                    setpoint_schedule = self.get_supply_air_temperature_setpoint_schedule(system.id)
                else:
                    logger.error("[Model Class] : " f"Unknown property {str(type(property_))} of {str(type(property_of))}")
                    raise Exception(f"Unknown property {str(type(property_))} of {str(type(property_of))}")
                self.add_connection(setpoint_schedule, controller, "scheduleValue", "setpointValue")
            else:
                if isinstance(property_of, base.BuildingSpace):
                    if isinstance(property_, base.Temperature):
                        self.add_connection(measuring_device, controller, "indoorTemperature", "actualValue")
                    elif isinstance(property_, base.Co2):
                        self.add_connection(measuring_device, controller, "indoorCo2Concentration", "actualValue")
                elif isinstance(property_of, components.CoilHeatingSystem):
                    system = [v for v in property_of.subSystemOf if v in self.system_dict["heating"].values()][0]
                elif isinstance(property_of, components.CoilCoolingSystem):
                    system = [v for v in property_of.subSystemOf if v in self.system_dict["cooling"].values()][0]
                else:
                    logger.error("[Model Class] : " f"Unknown property {str(type(property_))} of {str(type(property_of))}")
                    raise Exception(f"Unknown property {str(type(property_))} of {str(type(property_of))}")
                

        for shading_device in shading_device_instances:
            shade_setpoint_schedule = self.get_shade_setpoint_schedule(shading_device.id)
            self.add_connection(shade_setpoint_schedule, shading_device, "scheduleValue", "shadePosition")

        for sensor in sensor_instances:
            property_ = sensor.observes
            property_of = property_.isPropertyOf
            if property_of is None:
                instance_of_type_before = self._get_instance_of_type_before(flow_temperature_change_types, sensor)
                if len(instance_of_type_before)==0:
                    self.add_connection(outdoor_environment, sensor, "outdoorTemperature", "inletAirTemperature")
                elif len(instance_of_type_before)==1:
                    instance_of_type_before = instance_of_type_before[0]
                    if isinstance(instance_of_type_before, base.Coil):
                        if isinstance(property_, base.Temperature):
                            self.add_connection(instance_of_type_before, sensor, "outletAirTemperature", "flowAirTemperature")
                        else:
                            logger.error("[Model Class] :" f"Unknown property {str(type(property_))} of {str(type(instance_of_type_before))}")
                            raise Exception(f"Unknown property {str(type(property_))} of {str(type(instance_of_type_before))}")

                    elif isinstance(instance_of_type_before, base.AirToAirHeatRecovery):
                        side = self._get_flow_placement(sensor)
                        if isinstance(property_, base.Temperature):
                            if side=="supply":
                                self.add_connection(instance_of_type_before, sensor, "primaryTemperatureOut", "primaryTemperatureOut")
                            else:
                                self.add_connection(instance_of_type_before, sensor, "secondaryTemperatureOut", "secondaryTemperatureOut")
                        else:
                            logger.error("[Model Class] :" f"Unknown property {str(type(property_))} of {str(type(instance_of_type_before))}")
                            raise Exception(f"Unknown property {str(type(property_))} of {str(type(instance_of_type_before))}")
            else:
                if isinstance(property_of, base.BuildingSpace):
                    if isinstance(property_, base.Temperature):
                        self.add_connection(property_of, sensor, "indoorTemperature", "indoorTemperature")
                    elif isinstance(property_, base.Co2): 
                        self.add_connection(property_of, sensor, "indoorCo2Concentration", "indoorCo2Concentration")
                    else:
                        logger.error("[Model Class] :" f"Unknown property {str(type(property_))} of {str(type(property_of))}")
                        raise Exception(f"Unknown property {str(type(property_))} of {str(type(property_of))}")

                if isinstance(property_of, base.Damper):
                    if isinstance(property_, base.OpeningPosition):
                        self.add_connection(property_of, sensor, "damperPosition", "damperPosition")
                    else:
                        logger.error("[Model Class] :" f"Unknown property {str(type(property_))} of {str(type(property_of))}")
                        raise Exception(f"Unknown property {str(type(property_))} of {str(type(property_of))}")

                if isinstance(property_of, base.Valve):
                    if isinstance(property_, base.OpeningPosition):
                        self.add_connection(property_of, sensor, "valvePosition", "valvePosition")
                    else:
                        logger.error("[Model Class] :" f"Unknown property {str(type(property_))} of {str(type(property_of))}")
                        raise Exception(f"Unknown property {str(type(property_))} of {str(type(property_of))}")

                if isinstance(property_of, base.ShadingDevice):
                    if isinstance(property_, base.OpeningPosition):
                        self.add_connection(property_of, sensor, "shadePosition", "shadePosition")
                    else:
                        logger.error("[Model Class] :" f"Unknown property {str(type(property_))} of {str(type(property_of))}")
                        raise Exception(f"Unknown property {str(type(property_))} of {str(type(property_of))}")

        for meter in meter_instances:
            property_ = meter.observes
            property_of = property_.isPropertyOf
            if isinstance(property_of, base.SpaceHeater):
                if isinstance(property_, base.Energy):
                    self.add_connection(property_of, meter, "Energy", "Energy")
                elif isinstance(property_, base.Power):
                    self.add_connection(property_of, meter, "Power", "Power")

            elif isinstance(property_of, base.Coil):
                if isinstance(property_, base.Energy):
                    self.add_connection(property_of, meter, "Energy", "Energy")
                elif isinstance(property_, base.Power):
                    self.add_connection(property_of, meter, "Power", "Power")

            elif isinstance(property_of, base.Fan):
                if isinstance(property_, base.Energy):
                    self.add_connection(property_of, meter, "Energy", "Energy")
                elif isinstance(property_, base.Power):
                    self.add_connection(property_of, meter, "Power", "Power")


        for node in node_instances:
            ventilation_system = node.subSystemOf[0]
            supply_dampers = [v for v in ventilation_system.hasSubSystem if isinstance(v, base.Damper) and len(v.hasFluidFedBy)>0 and isinstance(v.hasFluidFedBy[0], components.BuildingSpaceSystem)]
            exhaust_dampers = [v for v in ventilation_system.hasSubSystem if isinstance(v, base.Damper) and len(v.feedsFluidTo)>0 and isinstance(v.feedsFluidTo[0], components.BuildingSpaceSystem)]
            if node.operationMode=="return":
                for damper in exhaust_dampers:
                    space = damper.isContainedIn
                    self.add_connection(damper, node, "airFlowRate", "flowRate_" + space.id)
                    self.add_connection(space, node, "indoorTemperature", "flowTemperatureIn_" + space.id)
            else:
                for damper in supply_dampers:
                    self.add_connection(damper, node, "airFlowRate", "flowRate_" + space.id)
        logger.info("[Model Class] : Exited from Connect Function")


    def init_building_space_systems(self):
        for space in self.get_component_by_class(self.component_dict, components.BuildingSpaceSystem):
            space.get_model()

    def init_building_space_systems(self):
        for space in self.get_component_by_class(self.component_dict, components.BuildingSpaceSystem):
            space.get_model()


    def set_custom_initial_dict(self, custom_initial_dict):
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
            components.OutdoorEnvironmentSystem.__name__: {},
            components.OccupancySystem.__name__: {"scheduleValue": 0},
            components.ScheduleSystem.__name__: {},
            components.BuildingSpaceSystem.__name__: {"indoorTemperature": 21,
                                                      "indoorCo2Concentration": 500},
            components.BuildingSpaceCo2System.__name__: {"indoorCo2Concentration": 500},
            components.BuildingSpaceOccSystem.__name__: {"numberOfPeople": 0},
            components.BuildingSpaceFMUSystem.__name__: {"indoorTemperature": 21,
                                                        "indoorCo2Concentration": 500},
            components.BuildingSpace0AdjBoundaryFMUSystem.__name__: {"indoorTemperature": 21,
                                                        "indoorCo2Concentration": 500},
            components.BuildingSpace0AdjBoundaryOutdoorFMUSystem.__name__: {"indoorTemperature": 21,
                                                        "indoorCo2Concentration": 500},
            components.BuildingSpace1AdjFMUSystem.__name__: {"indoorTemperature": 21,
                                                        "indoorCo2Concentration": 500},
            components.BuildingSpace2AdjFMUSystem.__name__: {"indoorTemperature": 21,
                                                        "indoorCo2Concentration": 500},
            components.BuildingSpace1AdjBoundaryFMUSystem.__name__: {"indoorTemperature": 21,
                                                        "indoorCo2Concentration": 500},
            components.BuildingSpace2AdjBoundaryFMUSystem.__name__: {"indoorTemperature": 21,
                                                        "indoorCo2Concentration": 500},
            components.BuildingSpace2AdjBoundaryOutdoorFMUSystem.__name__: {"indoorTemperature": 21,
                                                        "indoorCo2Concentration": 500},
            components.BuildingSpaceNoSH1AdjBoundaryOutdoorFMUSystem.__name__: {"indoorTemperature": 21,
                                                        "indoorCo2Concentration": 500},
            components.BuildingSpace1AdjBoundaryOutdoorFMUSystem.__name__: {"indoorTemperature": 21,
                                                        "indoorCo2Concentration": 500},      
            components.BuildingSpace11AdjBoundaryOutdoorFMUSystem.__name__: {"indoorTemperature": 21,
                                                        "indoorCo2Concentration": 500},                                                                                   
            components.ControllerSystem.__name__: {"inputSignal": 0},
            components.RulebasedControllerSystem.__name__: {"inputSignal": 0},
            components.RulebasedSetpointInputControllerSystem.__name__: {"inputSignal": 0},
            components.ClassificationAnnControllerSystem.__name__: {"inputSignal": 0},
            components.PIControllerFMUSystem.__name__: {"inputSignal": 0},
            components.SequenceControllerSystem.__name__: {"inputSignal": 0},  
            components.OnOffControllerSystem.__name__: {"inputSignal": 0},  
            components.AirToAirHeatRecoverySystem.__name__: {},
            components.CoilPumpValveFMUSystem.__name__: {},
            components.CoilFMUSystem.__name__: {},
            components.CoilHeatingSystem.__name__: {"outletAirTemperature": 21},
            components.CoilCoolingSystem.__name__: {},
            components.DamperSystem.__name__: {"airFlowRate": 0,
                                                "damperPosition": 0},
            components.ValveSystem.__name__: {"waterFlowRate": 0,
                                                "valvePosition": 0},
            components.ValveFMUSystem.__name__: {"waterFlowRate": 0,
                                                "valvePosition": 0},
            components.ValvePumpFMUSystem.__name__: {"waterFlowRate": 0,
                                                "valvePosition": 0},
            components.FanSystem.__name__: {}, #Energy
            components.FanFMUSystem.__name__: {}, #Energy
            components.SpaceHeaterSystem.__name__: {"outletWaterTemperature": 21,
                                        "Energy": 0},
            components.FlowJunctionSystem.__name__: {},
            components.JunctionDuctSystem.__name__: {},
            components.ShadingDeviceSystem.__name__: {},
            components.SensorSystem.__name__: {},
            components.MeterSystem.__name__: {},
            components.PiecewiseLinearSystem.__name__: {},
            components.PiecewiseLinearSupplyWaterTemperatureSystem.__name__: {},
            components.PiecewiseLinearScheduleSystem.__name__: {},
            components.TimeSeriesInputSystem.__name__: {},
            components.OnOffSystem.__name__: {},
            
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
        for i, (p, obj, attr) in enumerate(zip(parameters, component_list, attr_list)):
            assert rhasattr(obj, attr), f"The component with class \"{obj.__class__.__name__}\" and id \"{obj.id}\" has no attribute \"{attr}\"."
            rsetattr(obj, attr, p)

    def set_parameters_from_dict(self, parameters, component_list, attr_list):
        for (obj, attr) in zip(component_list, attr_list):
            assert rhasattr(obj, attr), f"The component with class \"{obj.__class__.__name__}\" and id \"{obj.id}\" has no attribute \"{attr}\"."
            rsetattr(obj, attr, parameters[attr])

    def cache(self,
                startTime=None,
                endTime=None,
                stepSize=None):
        """
        This method is called once before using multiprocessing on the Simulator.
        It calls the customizable "initialize" method for specific components to cache and create the folder structure for time series data.
        """
        c = self.get_component_by_class(self.component_dict, (components.SensorSystem, components.MeterSystem, components.OutdoorEnvironmentSystem, components.TimeSeriesInputSystem))
        for component in c:
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
        for component in self.flat_execution_order:
            component.clear_results()
            component.initialize(startTime=startTime,
                                endTime=endTime,
                                stepSize=stepSize,
                                model=self)
            
    def validate_model(self):
        self.validate_ids()
        self.validate_connections()
                
    def validate_ids(self):
        components = list(self.component_dict.values())
        for component in components:
            isvalid = np.array([x.isalnum() or x in self.valid_chars for x in component.id])
            np_id = np.array(list(component.id))
            violated_characters = list(np_id[isvalid==False])
            assert all(isvalid), f"The component with class \"{component.__class__.__name__}\" and id \"{component.id}\" has an invalid id. The characters \"{', '.join(violated_characters)}\" are not allowed."

    def validate_connections(self):
        components = list(self.component_dict.values())
        for component in components:
            if len(component.connectedThrough)==0 and len(component.connectsAt)==0:
                warnings.warn(f"The component with class \"{component.__class__.__name__}\" and id \"{component.id}\" has no connections. It has been removed from the model.")
                self.remove_component(component)

            input_labels = [cp.receiverPropertyName for cp in component.connectsAt]
            for req_input_label in component.input.keys():
                assert req_input_label in input_labels, f"The component with class \"{component.__class__.__name__}\" and id \"{component.id}\" is missing the input: \"{req_input_label}\""

        

    
    def load_model(self, semantic_model_filename=None, input_config=None, infer_connections=True, fcn=None, do_load_parameters=True):
        """
        This method loads component models and creates connections between the models. 
        In addition, it creates and draws graphs of the simulation model and the semantic model. 
        """
        print("Loading model...")
        if semantic_model_filename is not None:
            self.read_datamodel_config(semantic_model_filename)
            self._create_object_graph(self.component_base_dict)
            self.draw_object_graph(filename="object_graph_input")
            self.apply_model_extensions()
            # self.parse_semantic_model()

        if fcn is not None:
            self.fcn = fcn.__get__(self, Model)


        self.fcn()
        if input_config is not None:
            self.read_input_config(input_config)
        if infer_connections:
            self.connect()
        
        self._create_system_graph()
        self.draw_system_graph()
        self._get_execution_order_old()
        self._create_flat_execution_graph()
        self.draw_system_graph_no_cycles()
        self.draw_execution_graph()
        if do_load_parameters:
            self._load_parameters()

        self.validate_model()



    def _load_parameters(self):
        for component in self.component_dict.values():
            assert hasattr(component, "config"), f"The class \"{component.__class__.__name__}\" has no \"config\" attribute."
            config = component.config.copy()
            assert "parameters" in config, f"The \"config\" attribute of class \"{component.__class__.__name__}\" has no \"parameters\" key."
            filename, isfile = self.get_dir(folder_list=["model_parameters", component.__class__.__name__], filename=f"{component.id}.json")
            config["parameters"] = {attr: rgetattr(component, attr) for attr in config["parameters"]}
            if isfile==False:
                with open(filename, 'w') as f:
                    json.dump(config, f, indent=4)
            else:
                with open(filename) as f:
                    config = json.load(f)
                parameters = {k: float(v) if isnumeric(v) else v for k, v in config["parameters"].items()}
                self.set_parameters_from_dict(parameters, [component for k in config["parameters"].keys()], [k for k in config["parameters"].keys()])

                if "readings" in config:
                    filename_ = config["readings"]["filename"]
                    datecolumn = config["readings"]["datecolumn"]
                    valuecolumn = config["readings"]["valuecolumn"]
                    if filename_ is not None:
                        component.filename = filename_

                        if datecolumn is not None:
                            component.datecolumn = datecolumn
                        elif isinstance(component, components.OutdoorEnvironmentSystem)==False:
                            raise(ValueError(f"\"datecolumn\" is not defined in the \"readings\" key of the config file: {filename}"))

                        if valuecolumn is not None:
                            component.valuecolumn = valuecolumn
                        elif isinstance(component, components.OutdoorEnvironmentSystem)==False:
                            raise(ValueError(f"\"valuecolumn\" is not defined in the \"readings\" key of the config file: {filename}"))
                    
    def load_model_new(self, semantic_model_filename=None, input_config=None, infer_connections=True, fcn=None, create_signature_graphs=False, verbose=False, validate_model=True):
        if verbose:
            self._load_model_new(semantic_model_filename=semantic_model_filename, input_config=input_config, infer_connections=infer_connections, fcn=fcn, create_signature_graphs=create_signature_graphs, validate_model=validate_model)
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._load_model_new(semantic_model_filename=semantic_model_filename, input_config=input_config, infer_connections=infer_connections, fcn=fcn, create_signature_graphs=create_signature_graphs, validate_model=validate_model)

    def _load_model_new(self, semantic_model_filename=None, input_config=None, infer_connections=True, fcn=None, create_signature_graphs=False, validate_model=True):
        """
        This method loads component models and creates connections between the models. 
        In addition, it creates and draws graphs of the simulation model and the semantic model. 
        """


        p = PrintProgress()
        p("Loading model")
        p.add_level()
        self.add_outdoor_environment()
        if semantic_model_filename is not None:
            p(f"Reading semantic model")
            self.read_datamodel_config(semantic_model_filename)
            
            p(f"Creating input object graph")
            self._create_object_graph(self.component_base_dict)
            self.draw_object_graph(filename="object_graph_input")

            p(f"Parsing semantic model")
            self.parse_semantic_model()


        if input_config is not None:
            p(f"Reading input config")
            self.read_input_config(input_config)

        p(f"Creating parsed object graph")
        self._create_object_graph(self.component_base_dict)
        self.draw_object_graph(filename="object_graph_parsed")

        if create_signature_graphs:
            p(f"Creating signature graphs")
            self._create_signature_graphs()
        
        if infer_connections:
            p(f"Connecting components")
            self.connect_new()

        
        if fcn is not None:
            p(f"Applying user defined function")
            self.fcn = fcn.__get__(self, Model) # This is done to avoid the fcn to be shared between instances (https://stackoverflow.com/questions/28127874/monkey-patching-python-an-instance-method)
        self.fcn()

        
        
        p(f"Creating system graph")
        self._create_system_graph()
        self.draw_system_graph()

        if validate_model:
            p("Validating model")
            self.validate_model()

        p("Determining execution order")
        self._get_execution_order()

        p("Creating execution graph")
        self._create_flat_execution_graph()

        p("Creating system graph without cycles")
        self.draw_system_graph_no_cycles()
        self.draw_execution_graph()

        p("Loading parameters")
        self._load_parameters()
        p()

        print(self)

    def fcn(self):
        pass

    def split_name(self, name, linesep="\n"):

        split_delimiters = [" ", ")(", "_", "]", "|"]
        new_name = name
        char_len = len(name)
        char_limit = 20
        if any([s in name for s in split_delimiters]):
            if char_len>char_limit:
                name_splits = [name]
                for split_delimiter_ in split_delimiters:
                    new_name_splits = []
                    for name_split in name_splits:
                        splitted = name_split.split(split_delimiter_)
                        n = [e+split_delimiter_ if e and i<len(splitted)-1 else e for i,e in enumerate(splitted)]
                        new_name_splits.extend(n)                    
                    name_splits = new_name_splits

                char_cumsum = np.cumsum(np.array([len(s) for s in name_splits]))
                add_space_char = np.arange(char_cumsum.shape[0])
                char_cumsum = char_cumsum + add_space_char
                idx_arr = np.where(char_cumsum>char_limit)[0]
                if idx_arr.size!=0 and (idx_arr[0]==0 and idx_arr.size==1)==False:
                    if idx_arr[0]==0:
                        idx = idx_arr[1]
                    else:
                        idx = idx_arr[0]
                    name_before_line_break = "".join(name_splits[0:idx])
                    name_after_line_break = "".join(name_splits[idx:])
                    if len(name_after_line_break)>char_limit:
                        name_after_line_break = self.split_name(name_after_line_break, linesep=linesep)
                    
                    if name_before_line_break!="" and name_after_line_break!="":
                        new_name = name_before_line_break + linesep + name_after_line_break
                    else:
                        new_name = name
        return new_name
    
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

    def _create_signature_graphs(self):
        classes = [cls[1] for cls in inspect.getmembers(components, inspect.isclass) if (issubclass(cls[1], (System, )) and hasattr(cls[1], "sp"))]
        for component_cls in classes:
            sps = component_cls.sp
            for sp in sps:
                d = {s.id: s for s in sp.nodes}
                filename = self.get_dir(folder_list=["graphs", "signatures"], filename=f"signature_{component_cls.__name__}_{sp.id}")[0]
                self._create_object_graph(d, sp.ruleset, add_brackets=False)
                self.draw_graph(filename, self.object_graph)

    def _create_object_graph(self, object_dict, ruleset=None, add_brackets=True):
        logger.info("[Model Class] : Entered in Create Object Graph Function")
        self._initialize_graph("object")
        exceptions = []
        builtin_types = [getattr(builtins, d) for d in dir(builtins) if isinstance(getattr(builtins, d), type)]
        for exception in exceptions: builtin_types.remove(exception)
        exception_classes = (Connection, ConnectionPoint, np.ndarray, torch.device, pd.DataFrame) # These classes are excluded from the graph 
        exception_classes_exact = (base.DistributionDevice, *builtin_types, count)
        visited = []


        ruleset_applies = True if ruleset is not None else False
        shape_dict = {signature_pattern.IgnoreIntermediateNodes: "circle",
                      signature_pattern.Optional: "diamond",}
        dummy_dim = 0.3
        
        for component in object_dict.values():
            if component not in visited:
                visited = self._depth_first_search_recursive(component, visited, exception_classes, exception_classes_exact)
        
        ignore_nodes = [v.isValueOfProperty for v in visited if isinstance(v, base.PropertyValue) and v.hasValue is None]
        ignore_nodes.extend([v for v in visited if isinstance(v, base.PropertyValue) and v.hasValue is None])
        include_nodes = [v.hasValue for v in visited if isinstance(v, base.PropertyValue) and v.hasValue is not None]
        ignore_edges = ["feedsFluidTo", "hasFluidFedBy"]
        # ignore_edges = []
        end_space = "  "
        for component in visited:
            attributes = get_object_attributes(component)
                
            for attr in attributes:
                if attr not in ignore_edges:
                    edge_label = attr+end_space
                    obj = rgetattr(component, attr)

                    if hasattr(component.__class__, attr) and isinstance(rgetattr(component.__class__, attr), property):
                        class_property = True
                    else:
                        class_property = False

                    if obj is not None and inspect.ismethod(obj)==False and component not in ignore_nodes and class_property==False:
                        if isinstance(obj, list):
                            for receiver_component in obj:
                                cond1 = receiver_component in include_nodes
                                cond2 = isinstance(receiver_component, exception_classes)==False and istype(receiver_component, exception_classes_exact)==False and receiver_component not in ignore_nodes
                                if cond1 or cond2:
                                    if ruleset_applies and attr in component.attributes and receiver_component in component.attributes[attr] and type(ruleset[(component, receiver_component, attr)]) in shape_dict:
                                        dummy = signature_pattern.Node(tuple())
                                        shape = shape_dict[type(ruleset[(component, receiver_component, attr)])]
                                        self._add_graph_relation(self.object_graph, component, dummy, edge_kwargs={"label": edge_label, "arrowhead":"none", "fontname":"CMU Typewriter Text"}, receiver_node_kwargs={"label": "", "shape":shape, "width":dummy_dim, "height":dummy_dim})#, "fontname":"Helvetica", "fontsize":"21", "fontcolor":"black"})
                                        self._add_graph_relation(self.object_graph, dummy, receiver_component, edge_kwargs={"label": ""}, sender_node_kwargs={"label": "", "shape":shape, "width":dummy_dim, "height":dummy_dim})#, "fontname":"Helvetica", "fontsize":"21", "fontcolor":"black"})
                                    else:
                                        self._add_graph_relation(self.object_graph, component, receiver_component, edge_kwargs={"label": edge_label, "fontname":"CMU Typewriter Text"})
                        else:
                            receiver_component = obj
                            cond1 = receiver_component in include_nodes
                            cond2 = isinstance(receiver_component, exception_classes)==False and istype(receiver_component, exception_classes_exact)==False and receiver_component not in ignore_nodes

                            if cond1 or cond2:
                                if ruleset_applies and attr in component.attributes and receiver_component == component.attributes[attr] and type(ruleset[(component, receiver_component, attr)]) in shape_dict:
                                    dummy = signature_pattern.Node(tuple())
                                    shape = shape_dict[type(ruleset[(component, receiver_component, attr)])]
                                    self._add_graph_relation(self.object_graph, component, dummy, edge_kwargs={"label": edge_label, "arrowhead":"none", "fontname":"CMU Typewriter Text"}, receiver_node_kwargs={"label": "", "shape":shape, "width":dummy_dim, "height":dummy_dim})#, "fontname":"Helvetica", "fontsize":"21", "fontcolor":"black"})
                                    self._add_graph_relation(self.object_graph, dummy, receiver_component, edge_kwargs={"label": ""}, sender_node_kwargs={"label": "", "shape":shape, "width":dummy_dim, "height":dummy_dim})#, "fontname":"Helvetica", "fontsize":"21", "fontcolor":"black"})
                                else:
                                    self._add_graph_relation(self.object_graph, component, receiver_component, edge_kwargs={"label": edge_label, "fontname":"CMU Typewriter Text"})
        graph = self.object_graph
        attributes = self.object_graph_node_attribute_dict
        subgraphs = self.object_subgraph_dict
        self._create_graph(graph, attributes, subgraphs, add_brackets=add_brackets)

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
                    self._add_edge(self.execution_graph, prev_node.obj_dict["name"], node.obj_dict["name"], edge_kwargs={"label": ""})
                prev_node = node
            self.execution_graph.add_subgraph(subgraph)

    def _create_system_graph(self):
        graph = self.system_graph
        attributes = self.system_graph_node_attribute_dict
        subgraphs = self.system_subgraph_dict
        self._create_graph(graph, attributes, subgraphs)

    def is_html(self, name):
        if len(name)>=2 and name[0]=="<" and name[-1]==">":
            return True
        else:
            return False

    def _create_graph(self, graph, attributes, subgraphs, add_brackets=False):
        logger.info("[Model Class] : Entered in Create System Graph Function")
        light_black = "#3B3838"
        dark_blue = "#44546A"
        orange = "#DC8665"#"#C55A11"
        red = "#873939"
        grey = "#666666"
        light_grey = "#71797E"
        light_blue = "#8497B0"
        green = "#83AF9B"#"#BF9000"
        buttercream = "#B89B72"
        green = "#83AF9B"        

        fill_colors = {base.BuildingSpace: light_black,
                            base.Controller: orange,
                            base.AirToAirHeatRecovery: dark_blue,
                            base.Coil: red,
                            base.Damper: dark_blue,
                            base.Valve: red,
                            base.Fan: dark_blue,
                            base.SpaceHeater: red,
                            base.Sensor: green,
                            base.Meter: green,
                            base.Schedule: grey,
                            base.Pump: red}
        fill_default = light_grey
        border_colors = {base.BuildingSpace: "black",
                            base.Controller: "black",
                            base.AirToAirHeatRecovery: "black",
                            base.Coil: "black",
                            base.Damper: "black",
                            base.Valve: "black",
                            base.Fan: "black",
                            base.SpaceHeater: "black",
                            base.Sensor: "black",
                            base.Meter: "black"}
        border_default = "black"
        K = 1
        min_fontsize = 22*K
        max_fontsize = 28*K

        fontpath, fontname = self.get_font()


        delta_box_width = 0.2
        delta_box_height = 0.5
        width_pad = 2*delta_box_width
        height_pad = 0.1
        nx_graph = nx.drawing.nx_pydot.from_pydot(graph)
        labelwidths = []
        labelheights = []
        for node in nx_graph.nodes():
            name = attributes[node]["label"]
            linesep = "\n"
            is_html = self.is_html(name)
            name = self.split_name(name, linesep=linesep)
            html_chars = ["<", ">", "SUB," ,"/SUB"]
            no_html_name = name
            for s in html_chars:
                no_html_name = no_html_name.replace(s, "")
            names = no_html_name.split(linesep)

            if is_html==False: #Convert to html 
                name ="<"+name+">"
            name = name.replace(linesep, "<br />")
            if add_brackets:
                name ="<&#60;" + name[1:]
                name = name[:-1] + "&#62;>"
            char_count_list = [len(s) for s in names if s]
            if len(char_count_list)>0:
                char_count = max(char_count_list)
                linecount = len(names)
            else:
                char_count = 0
                linecount = 0
            attributes[node]["label"] = name
            attributes[node]["labelcharcount"] = char_count
            attributes[node]["labellinecount"] = linecount
            attributes[node]["fontname"] = fontname


        a_fontsize = 0
        b_fontsize = max_fontsize

        a_char_width = delta_box_width
        b_char_width = width_pad

        a_line_height = delta_box_height
        b_line_height = height_pad

        for node in nx_graph.nodes():
            deg = nx_graph.degree(node)
            fontsize = a_fontsize*deg + b_fontsize
            name = attributes[node]["label"]
            labelwidth = attributes[node]["labelcharcount"]
            labelheight = attributes[node]["labellinecount"]
            
            width = a_char_width*labelwidth + b_char_width
            height = a_line_height*labelheight + b_line_height


            if node not in attributes:
                attributes[node] = {}

            attributes[node]["fontsize"] = fontsize

            if "width" not in attributes[node]:
                attributes[node]["width"] = width
            if "height" not in attributes[node]:
                attributes[node]["height"] = height
            cls = self.object_dict[node].__class__
            if cls in fill_colors:
                attributes[node]["fillcolor"] = fill_colors[cls] 
            elif issubclass(cls, tuple(fill_colors.keys())):
                c = [color for c, color in fill_colors.items() if issubclass(cls, c)]
                if len(c)>1:
                    colors = c[0]
                    for color in c[1:]:
                        colors += ":" + color #Currently, the gradient colors are limited to 2
                    c = f"\"{colors}\""
                else:
                    c = c[0]
                attributes[node]["fillcolor"] = c
                
            else:
                attributes[node]["fillcolor"] = fill_default
            
            if cls in border_colors:
                attributes[node]["color"] = border_colors[cls] 
            elif issubclass(cls, tuple(border_colors.keys())):
                c = [c for c in border_colors.keys() if issubclass(cls, c)][0]
                attributes[node]["color"] = border_colors[c]
            else:
                attributes[node]["color"] = border_default

            c = [c for c in subgraphs.keys() if cls is c][0]
            subgraph = subgraphs[c]
            name = node
            if len(subgraph.get_node(name))==1:
                subgraph.get_node(name)[0].obj_dict["attributes"].update(attributes[node])
            elif len(subgraph.get_node(name))==0: #If the name is not present, try with quotes
                 name = "\"" + node + "\""
                 subgraph.get_node(name)[0].obj_dict["attributes"].update(attributes[node])
            else:
                print([el.id for el in self.object_dict.values()])
                raise Exception(f"Multiple identical node names found in subgraph")
        logger.info("[Model Class] : Exited from Create System Graph Function")

    def draw_object_graph(self, filename="object_graph"):
        graph = self.object_graph
        self.draw_graph(filename, graph)

    def draw_system_graph_no_cycles(self):
        filename = "system_graph_no_cycles"
        graph = self.system_graph_no_cycles
        self.draw_graph(filename, graph)

    def draw_system_graph(self):
        filename = "system_graph"
        graph = self.system_graph
        self.draw_graph(filename, graph)

    def draw_execution_graph(self):
        filename = "execution_graph"
        graph = self.execution_graph
        self.draw_graph(filename, graph)

    def get_font(self):
        font_files = matplotlib.font_manager.findSystemFonts(fontpaths=None)
        preferred_font = "Helvetica-Bold".lower()
        preferred_font = "CMUNBTL".lower()
        found_preferred_font = False
        for font in font_files:
            s = os.path.split(font)
            if preferred_font in s[1].lower():
                found_preferred_font = True
                break

        if found_preferred_font==False:
            font = matplotlib.font_manager.findfont(matplotlib.font_manager.FontProperties(family=['sans-serif']))

        fontpath = os.path.split(font)[0]
        fontname = os.path.split(font)[1]

        fontname = "CMU Typewriter Text"

        return fontpath, fontname

    def unflatten(self, filename):
        app_path = shutil.which("unflatten")
        args = [app_path, "-f", f"-l 3", f"-o{filename}_unflatten.dot", f"{filename}.dot"]
        subprocess.run(args=args)

    def draw_graph(self, filename, graph, args=None):
        fontpath, fontname = self.get_font()
        
        light_grey = "#71797E"
        graph_filename = os.path.join(self.graph_path, f"{filename}.png")
        graph.write(f'{filename}.dot', prog="dot")
        app_path = shutil.which("dot")
        if args is None:
            args = [app_path,
                    "-q",
                    "-Tpng",
                    "-Kdot",
                    f"-Gfontpath={fontpath}",
                    "-Nstyle=filled",
                    "-Nshape=box",
                    "-Nfontcolor=white",
                    f"-Nfontname=Helvetica bold",
                    "-Nfixedsize=true",
                    "-Gnodesep=0.1",
                    "-Efontname=Helvetica",
                    "-Efontsize=21",
                    "-Epenwidth=2",
                    "-Eminlen=1",
                    f"-Ecolor={light_grey}",
                    "-Gcompound=true",
                    "-Grankdir=TB",
                    "-Gsplines=true", #true
                    "-Gmargin=0",
                    "-Gsize=10!",
                    # "-Gratio=compress", #0.5 #auto
                    "-Gpack=true",
                    "-Gdpi=5000",
                    "-Grepulsiveforce=0.5",
                    "-Gremincross=true",
                    "-Gstart=1",
                    "-q",
                    f"-o{graph_filename}",
                    f"{filename}.dot"] #_unflatten
        else:
            args_ = [app_path]
            args_.extend(args)
            args_.extend([f"-o{graph_filename}", f"{filename}.dot"])
            args = args_
        subprocess.run(args=args)
        os.remove(f"{filename}.dot")

    def _depth_first_search_recursive(self, component, visited, exception_classes, exception_classes_exact):
        visited.append(component)
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
        visited = []
        visited = self._depth_first_search_recursive(obj, visited)
        return visited

    def _flatten(self, _list):
        return [item for sublist in _list for item in sublist]


    def _shortest_path(self, component):
        def _shortest_path_recursive(shortest_path, exhausted, unvisited):
            while len(unvisited)>0:
                component = unvisited[0]
                current_path_length = shortest_path[component]
                for connection in component.connectedThrough:
                    connection_point = connection.connectsSystemAt
                    receiver_component = connection_point.connectionPointOf

                    if receiver_component not in exhausted:
                        unvisited.append(receiver_component)
                        if receiver_component not in shortest_path: shortest_path[receiver_component] = np.inf
                        if current_path_length+1<shortest_path[receiver_component]:
                            shortest_path[receiver_component] = current_path_length+1
                exhausted.append(component)
                unvisited.remove(component)
            return shortest_path
                
        shortest_path = {}
        shortest_path[component] = 0
        exhausted = []
        unvisited = [component]
        shortest_path = _shortest_path_recursive(shortest_path, exhausted, unvisited)
        return shortest_path
 
    def _depth_first_search_system(self, component):
        def _depth_first_search_recursive_system(component, visited):
            visited.append(component)
            for connection in component.connectedThrough:
                connection_point = connection.connectsSystemAt
                receiver_component = connection_point.connectionPointOf
                if receiver_component not in visited:
                    visited = _depth_first_search_recursive_system(receiver_component, visited)
            return visited
        visited = []
        visited = _depth_first_search_recursive_system(component, visited)
        return visited
    
    
 
    def _depth_first_search_cycle_system(self, component):
        def _depth_first_search_recursive_system(component, visited):
            visited.append(component)
            for connection in component.connectedThrough:
                connection_point = connection.connectsSystemAt
                receiver_component = connection_point.connectionPointOf
                if len(receiver_component.connectedThrough)==0:
                    return visited
                if receiver_component not in visited:
                    visited = _depth_first_search_recursive_system(receiver_component, visited.copy())
            return visited
        visited = []
        visited = _depth_first_search_recursive_system(component, visited)
        return visited

    def get_subgraph_dict_no_cycles(self):
        self.system_subgraph_dict_no_cycles = copy.deepcopy(self.system_subgraph_dict)
        subgraphs = self.system_graph_no_cycles.get_subgraphs()
        for subgraph in subgraphs:
            subgraph.get_nodes()
            if len(subgraph.get_nodes())>0:
                node = subgraph.get_nodes()[0].obj_dict["name"].replace('"',"")
                self.system_subgraph_dict_no_cycles[self._component_dict_no_cycles[node].__class__] = subgraph


    def get_component_dict_no_cycles_old(self):
        self._component_dict_no_cycles = copy.deepcopy(self.component_dict)
        self.system_graph_no_cycles = copy.deepcopy(self.system_graph)
        self.get_subgraph_dict_no_cycles()
        self.required_initialization_connections = []
        controller_instances = [v for v in self._component_dict_no_cycles.values() if isinstance(v, base.Controller)]
        for controller in controller_instances:
            controlled_component = controller.observes.isPropertyOf
            assert controlled_component is not None, f"The attribute \"isPropertyOf\" is None for property \"{controller.observes}\" of component \"{controller.id}\""
            visited = self._depth_first_search_system(controller)
            for reachable_component in visited:
                for connection in reachable_component.connectedThrough.copy():
                    connection_point = connection.connectsSystemAt
                    receiver_component = connection_point.connectionPointOf
                    if controlled_component==receiver_component:
                        controlled_component.connectsAt.remove(connection_point)
                        reachable_component.connectedThrough.remove(connection)
                        edge_label = self.get_edge_label(connection.senderPropertyName, connection_point.receiverPropertyName)
                        status = self._del_edge(self.system_graph_no_cycles, reachable_component.id, controlled_component.id, label=edge_label)
                        assert status, "del_edge returned False. Check if additional characters should be added to \"disallowed_characters\"."

                        self.required_initialization_connections.append(connection)

    def get_base_component(self, key):
        """
        Assumes that there is a 1-to-1 mapping
        """
        assert len(self.instance_map[self.component_dict[key]])==1, f"The mapping for component \"{key}\" is not 1-to-1"
        return next(iter(self.instance_map[self.component_dict[key]]))

    def get_component_dict_no_cycles(self):
        self._component_dict_no_cycles = copy.deepcopy(self.component_dict)
        self.system_graph_no_cycles = copy.deepcopy(self.system_graph)
        self.get_subgraph_dict_no_cycles()
        self.required_initialization_connections = []

        controller_instances = [v for v in self._component_dict_no_cycles.values() if isinstance(v, base.Controller)]
        for controller in controller_instances:
            modeled_components = self.instance_map[self.component_dict[controller.id]]
            base_controller = [v for v in modeled_components if isinstance(v, base.Controller)][0]
            controlled_components = [self._component_dict_no_cycles[self.instance_map_reversed[c.isPropertyOf].id] for c in base_controller.controls]
            assert len(controlled_components)!=0, f"No controlled components found for controller with id: {controller.id}"
            visited = self._depth_first_search_system(controller)

            for controlled_component in controlled_components:
                for reachable_component in visited:
                    for connection in reachable_component.connectedThrough.copy():
                        connection_point = connection.connectsSystemAt
                        receiver_component = connection_point.connectionPointOf
                        if controlled_component==receiver_component:
                            controlled_component.connectsAt.remove(connection_point)
                            reachable_component.connectedThrough.remove(connection)
                            edge_label = self.get_edge_label(connection.senderPropertyName, connection_point.receiverPropertyName)
                            status = self._del_edge(self.system_graph_no_cycles, reachable_component.id, controlled_component.id, label=edge_label)
                            assert status, "del_edge returned False. Check if additional characters should be added to \"disallowed_characters\"."

                            self.required_initialization_connections.append(connection)


        # Temporary fix for removing connections between spaces - should be handled in a more general way
        # Maybe implement Johnsons method to detect and locate cycles 
        space_instances = [v for v in self._component_dict_no_cycles.values() if isinstance(v, base.BuildingSpace)]
        for space in space_instances:
            modeled_components = self.instance_map[self.component_dict[space.id]]
            base_space = [v for v in modeled_components if isinstance(v, base.BuildingSpace)][0]
            connected_spaces = [self._component_dict_no_cycles[self.instance_map_reversed[c].id] for c in base_space.connectedTo if isinstance(c, base.BuildingSpace)]
            for connected_space in connected_spaces:
                for connection in space.connectedThrough.copy():
                    connection_point = connection.connectsSystemAt
                    receiver_component = connection_point.connectionPointOf
                    if receiver_component==connected_space:
                        space.connectedThrough.remove(connection)
                        connected_space.connectsAt.remove(connection_point)
                        edge_label = self.get_edge_label(connection.senderPropertyName, connection_point.receiverPropertyName)
                        status = self._del_edge(self.system_graph_no_cycles, space.id, connected_space.id, label=edge_label)
                        assert status, "del_edge returned False. Check if additional characters should be added to \"disallowed_characters\"."
                        self.required_initialization_connections.append(connection)

        # # Temporary fix for removing connections between spaces - should be handled in a more general way
        # # Maybe implement Johnsons method to detect and locate cycles 
        # occupancy_instances = [v for v in self._component_dict_no_cycles.values() if isinstance(v, components.OccupancySystem)]
        # for occ in occupancy_instances:
        #     for connection in occ.connectedThrough.copy():
        #         connection_point = connection.connectsSystemAt
        #         receiver_component = connection_point.connectionPointOf
        #         occ.connectedThrough.remove(connection)
        #         receiver_component.connectsAt.remove(connection_point)
        #         edge_label = self.get_edge_label(connection.senderPropertyName, connection_point.receiverPropertyName)
        #         status = self._del_edge(self.system_graph_no_cycles, occ.id, receiver_component.id, label=edge_label)
        #         assert status, "del_edge returned False. Check if additional characters should be added to \"disallowed_characters\"."
        #         self.required_initialization_connections.append(connection)

    def _set_addUncertainty(self, addUncertainty=None, filter="physicalSystem"):
        allowed_filters = ["all", "physicalSystem"]
        assert isinstance(addUncertainty, bool), "Argument addUncertainty must be True or False"
        assert filter in allowed_filters, f"The \"filter\" argument must be one of the following: {', '.join(allowed_filters)} - \"{filter}\" was provided."
        sensor_instances = self.get_component_by_class(self.component_dict, base.Sensor)
        meter_instances = self.get_component_by_class(self.component_dict, base.Meter)
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
        _, ext = os.path.splitext(filename)
        
        if ext==".pickle":
            with open(filename, 'rb') as handle:
                self.chain_log = pickle.load(handle)
                
        elif ext==".npz":
            to_list_keys = ["startTime_train", "endTime_train", "stepSize_train"]
            self.chain_log = dict(np.load(filename, allow_pickle=True))
            for key, value in self.chain_log.items():
                print(key, self.chain_log[key], type(self.chain_log[key]))

                # if key in to_list_keys:
                    # self.chain_log[key] = value.tolist()

                if value.size==1:
                #     print(value)
                    self.chain_log[key] = value.tolist()
                print(key, self.chain_log[key], type(self.chain_log[key]))
                
            # print(self.chain_log["n_par"])
            # print(self.chain_log["n_par_map"])
        
        self.chain_log["chain.T"] = 1/self.chain_log["chain.betas"]

        # self.chain_log["startTime_train"] = self.chain_log["startTime_train"][0]
        # self.chain_log["endTime_train"] = self.chain_log["endTime_train"][0]
        # self.chain_log["stepSize_train"] = self.chain_log["stepSize_train"][0]

        # with open(filename, 'wb') as handle:
            # pickle.dump(self.chain_log, handle, protocol=pickle.HIGHEST_PROTOCOL)

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
                
    def _get_execution_order_old(self):
        self.get_component_dict_no_cycles_old()
        initComponents = [v for v in self._component_dict_no_cycles.values() if len(v.connectsAt)==0]
        self.activeComponents = initComponents
        self.execution_order = []
        while len(self.activeComponents)>0:
            self._traverse()

        self.map_execution_order()
        self.map_required_initialization_connections()
        self.flat_execution_order = self._flatten(self.execution_order)
        assert len(self.flat_execution_order)==len(self._component_dict_no_cycles), f"Cycles detected in the model. Inspect the generated file \"system_graph.png\" to see where."

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
    def get_leaf_subsystems(self, system):
        for sub_system in system.hasSubSystem:
            if sub_system.hasSubSystem is None:
                self.leaf_subsystems.append(sub_system)
            else:
                self.get_leaf_subsystems(sub_system)


    






    

