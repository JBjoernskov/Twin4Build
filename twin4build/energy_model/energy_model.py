from dateutil.tz import tzutc
import datetime
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import warnings
import shutil
import subprocess
import sys
import os
# import torch
# import threading

import numpy as np ################################################################## test

from torch.multiprocessing import Pool
# from pathos.pools import ProcessPool
# from concurrent.futures import ThreadPoolExecutor, wait
# from multiprocessing import Pool
# from ray.util.multiprocessing import Pool

# import os
# import ray
# num_cpus = os.cpu_count()
# ray.init(num_cpus=num_cpus)

import copy
import math
from tqdm import tqdm






test = False

###Only for testing before distributing package
if test:
    uppath = lambda _path,n: os.sep.join(_path.split(os.sep)[:-n])
    file_path = uppath(os.path.abspath(__file__), 3)
    sys.path.append(file_path)


import twin4build.utils.building_data_collection_dict as building_data_collection_dict
from twin4build.saref4syst.connection import Connection 
from twin4build.saref4syst.connection_point import ConnectionPoint
from twin4build.saref4syst.system import System
from twin4build.utils.uppath import uppath
from twin4build.utils.weather_station import WeatherStation
from twin4build.utils.schedule import Schedule
from twin4build.utils.node import Node
from twin4build.saref.measurement.measurement import Measurement
from twin4build.saref.date_time.date_time import DateTime
from twin4build.saref4bldg.building_space.building_space import BuildingSpace
from twin4build.saref4bldg.building_space.building_space_model import BuildingSpaceModel, NoSpaceModelException
from twin4build.saref4bldg.physical_object.building_object.building_device.distribution_device.distribution_device import DistributionDevice
from twin4build.saref4bldg.physical_object.building_object.building_device.distribution_device.distribution_control_device.controller.controller_model import ControllerModel
from twin4build.saref4bldg.physical_object.building_object.building_device.distribution_device.distribution_flow_device.energy_conversion_device.air_to_air_heat_recovery.air_to_air_heat_recovery_model import AirToAirHeatRecoveryModel
from twin4build.saref4bldg.physical_object.building_object.building_device.distribution_device.distribution_flow_device.energy_conversion_device.coil.coil_heating_model import CoilHeatingModel
from twin4build.saref4bldg.physical_object.building_object.building_device.distribution_device.distribution_flow_device.energy_conversion_device.coil.coil_cooling_model import CoilCoolingModel
from twin4build.saref4bldg.physical_object.building_object.building_device.distribution_device.distribution_flow_device.flow_controller.damper.damper_model import DamperModel
from twin4build.saref4bldg.physical_object.building_object.building_device.distribution_device.distribution_flow_device.flow_controller.valve.valve_model import ValveModel
from twin4build.saref4bldg.physical_object.building_object.building_device.distribution_device.distribution_flow_device.flow_moving_device.fan.fan_model import FanModel
from twin4build.saref4bldg.physical_object.building_object.building_device.distribution_device.distribution_flow_device.flow_terminal.space_heater.space_heater_model import SpaceHeaterModel



class EnergyModel:
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

        self.initComponents = []
        self.activeComponents = None
        self.system_dict = {}
        self.component_dict = {}


        # self.executor = ThreadPoolExecutor(max_workers=4)

        # self.pool = ProcessPool(nodes=4)

        

        

    def add_edge_(self, a, b, label):
        if (a, b) in self.system_graph.edges:
            max_rad = max(x[2]['rad'] for x in self.system_graph.edges(data=True) if sorted(x[:2]) == sorted([a,b]))
        else:
            max_rad = 0
        self.system_graph.add_edge(a, b, rad=max_rad+0, label=label)


    def add_connection(self, sender_obj, reciever_obj, senderPropertyName, recieverPropertyName):
        sender_obj_connection = Connection(connectsSystem = sender_obj, senderPropertyName = senderPropertyName)
        sender_obj.connectedThrough.append(sender_obj_connection)
        reciever_obj_connection_point = ConnectionPoint(connectionPointOf = reciever_obj, connectsSystemThrough = sender_obj_connection, recieverPropertyName = recieverPropertyName)
        sender_obj_connection.connectsSystemAt = reciever_obj_connection_point
        reciever_obj.connectsAt.append(reciever_obj_connection_point)

        self.add_edge_(sender_obj.systemId, reciever_obj.systemId, label=senderPropertyName) ###

        
        self.system_graph_node_attribute_dict[sender_obj.systemId] = {"label": sender_obj.__class__.__name__}
        self.system_graph_node_attribute_dict[reciever_obj.systemId] = {"label": reciever_obj.__class__.__name__}
        self.system_graph_edge_label_dict[(sender_obj.systemId, reciever_obj.systemId)] = senderPropertyName

    
    def add_weather_station(self):
        weather_station = WeatherStation(
            startPeriod = self.startPeriod,
            endPeriod = self.endPeriod,
            input = {},
            output = {},
            savedInput = {},
            savedOutput = {},
            createReport = self.createReport,
            connectedThrough = [],
            connectsAt = [],
            systemId = "weather_station")
        self.component_dict["weather_station"] = weather_station

    def add_occupancy_schedule(self):
        occupancy_schedule = Schedule(
            startPeriod = self.startPeriod,
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
            createReport = self.createReport,
            connectedThrough = [],
            connectsAt = [],
            systemId = "occupancy_schedule")
        self.component_dict["occupancy_schedule"] = occupancy_schedule

    def add_temperature_setpoint_schedule(self):
        temperature_setpoint_schedule = Schedule(
            startPeriod = self.startPeriod,
            timeStep = self.timeStep,
            rulesetDict = {
                "ruleset_default_value": 20,
                "ruleset_start_minute": [0,0],
                "ruleset_end_minute": [0,0],
                "ruleset_start_hour": [0,6],
                "ruleset_end_hour": [6,18],
                "ruleset_value": [20,22]},
            input = {},
            output = {},
            savedInput = {},
            savedOutput = {},
            createReport = self.createReport,
            connectedThrough = [],
            connectsAt = [],
            systemId = "temperature_setpoint_schedule")
        self.component_dict["temperature_setpoint_schedule"] = temperature_setpoint_schedule

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
            systemId = "co2_setpoint_schedule")
        self.component_dict["co2_setpoint_schedule"] = co2_setpoint_schedule

    
    def read_config(self):
        file_path = os.path.join(uppath(os.path.abspath(__file__), 2), "test", "data", "configuration_template.xlsx")

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
            ventilation_system = DistributionDevice(subSystemOf = [], hasSubSystem = [], systemId = ventilation_system_name)
            self.system_dict[ventilation_system_name] = ventilation_system
        
        for heating_system_name in df_Systems["Heating system name"].dropna():
            heating_system = DistributionDevice(subSystemOf = [], hasSubSystem = [], systemId = heating_system_name)
            self.system_dict[heating_system_name] = heating_system

        for cooling_system_name in df_Systems["Cooling system name"].dropna():
            cooling_system = DistributionDevice(subSystemOf = [], hasSubSystem = [], systemId = cooling_system_name)
            self.system_dict[cooling_system_name] = cooling_system

        for row in df_Spaces.dropna(subset=["Space name"]).itertuples(index=False):
            space_name = row[df_Spaces.columns.get_loc("Space name")]
            try: 
                space = BuildingSpaceModel(
                    densityAir = 1.225,
                    airVolume = 50,
                    startPeriod = self.startPeriod,
                    timeStep = self.timeStep,
                    input = {"generationCo2Concentration": 0.06,
                            "outdoorCo2Concentration": 500,
                            "shadesPosition": 0},
                    output = {"indoorTemperature": 21.5,
                            "indoorCo2Concentration": 500},
                    savedInput = {},
                    savedOutput = {},
                    createReport = self.createReport,
                    connectedThrough = [],
                    connectsAt = [],
                    systemId = space_name)
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
                    systemId = damper_name)
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
                    createReport = self.createReport,
                    connectedThrough = [],
                    connectsAt = [],
                    systemId = space_heater_name)
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
                    systemId = valve_name)
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
                systemId = heating_coil_name)
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
                systemId = cooling_coil_name)
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
                systemId = air_to_air_heat_recovery_name)
            self.component_dict[air_to_air_heat_recovery_name] = air_to_air_heat_recovery
            ventilation_system.hasSubSystem.append(air_to_air_heat_recovery)

        for row in df_Fans.dropna(subset=["Fan name"]).itertuples(index=False):
            fan_name = row[df_Fans.columns.get_loc("Fan name")]
            ventilation_system = self.system_dict[row[df_Fans.columns.get_loc("Ventilation system")]]
            fan = FanModel(
                c1=Measurement(hasValue=0),
                c2=Measurement(hasValue=0),
                c3=Measurement(hasValue=0),
                c4=Measurement(hasValue=1),
                nominalAirFlowRate = Measurement(hasValue=row[df_Fans.columns.get_loc("nominalAirFlowRate")]),
                nominalPowerRate = Measurement(hasValue=row[df_Fans.columns.get_loc("nominalPowerRate")]),
                subSystemOf = [ventilation_system],
                input = {},
                output = {},
                savedInput = {},
                savedOutput = {},
                createReport = self.createReport,
                connectedThrough = [],
                connectsAt = [],
                systemId = fan_name)
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
                systemId = node_name)
            self.component_dict[node_name] = node
            ventilation_system.hasSubSystem.append(node)

        for row in df_Controller.dropna(subset=["Controller name"]).itertuples(index=False):
            controller_name = row[df_Controller.columns.get_loc("Controller name")]
            if controller_name[4:] not in self.component_dict:
                warnings.warn("Cannot find a matching mathing BuildingSpace object for controller \"" + controller_name + "\"")
            else:
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
                    systemId = controller_name)
                self.component_dict[controller_name] = controller
            # ventilation_system.hasSubSystem.append(controller)
        
    def get_models_by_instance(self, type_):
        return [v for v in self.component_dict.values() if isinstance(v, type_)]

    def connect(self):
        """
        Connects component instances 
        """
        space_instances = self.get_models_by_instance(BuildingSpaceModel)
        damper_instances = self.get_models_by_instance(DamperModel)
        space_heater_instances = self.get_models_by_instance(SpaceHeaterModel)
        valve_instances = self.get_models_by_instance(ValveModel)
        coil_heating_instances = self.get_models_by_instance(CoilHeatingModel)
        coil_cooling_instances = self.get_models_by_instance(CoilCoolingModel)
        air_to_air_heat_recovery_instances = self.get_models_by_instance(AirToAirHeatRecoveryModel)
        fan_instances = self.get_models_by_instance(FanModel)
        node_instances = self.get_models_by_instance(Node)
        controller_instances = self.get_models_by_instance(ControllerModel)


        weather_station = self.component_dict["weather_station"]
        occupancy_schedule = self.component_dict["occupancy_schedule"]
        temperature_setpoint_schedule = self.component_dict["temperature_setpoint_schedule"]
        co2_setpoint_schedule = self.component_dict["co2_setpoint_schedule"]


        for space in space_instances:
            if "C_T_" + space.systemId in self.component_dict:
                temperature_controller = self.component_dict["C_T_" + space.systemId]
                self.add_connection(space, temperature_controller, "indoorTemperature", "actualValue") ###
                self.add_connection(temperature_controller, space, "inputSignal", "valvePosition")

            if "C_C_" + space.systemId in self.component_dict:
                co2_controller = self.component_dict["C_C_" + space.systemId]
                self.add_connection(space, co2_controller, "indoorCo2Concentration", "actualValue") ###
                self.add_connection(co2_controller, space, "inputSignal", "supplyDamperPosition") ###
                self.add_connection(co2_controller, space, "inputSignal", "returnDamperPosition")

            if "D_S_" + space.systemId in self.component_dict:
                damper = self.component_dict["D_S_" + space.systemId]
                self.add_connection(damper, space, "airFlowRate", "supplyAirFlowRate")
                ventilation_system = damper.subSystemOf[0]
                node = [v for v in ventilation_system.hasSubSystem if isinstance(v, Node) and v.systemId[0:4] == "N_S_"][0]
                self.add_connection(damper, node, "airFlowRate", "flowRate_" + space.systemId) ###
                self.add_connection(space, node, "indoorTemperature", "flowTemperatureIn_" + space.systemId) ###
                
            if "D_E_" + space.systemId in self.component_dict:
                damper = self.component_dict["D_E_" + space.systemId]
                self.add_connection(damper, space, "airFlowRate", "returnAirFlowRate")
                ventilation_system = damper.subSystemOf[0]
                node = [v for v in ventilation_system.hasSubSystem if isinstance(v, Node) and v.systemId[0:4] == "N_E_"][0]
                self.add_connection(damper, node, "airFlowRate", "flowRate_" + space.systemId) ###
                self.add_connection(space, node, "indoorTemperature", "flowTemperatureIn_" + space.systemId) ###

            self.add_connection(weather_station, space, "shortwaveRadiation", "shortwaveRadiation")
            self.add_connection(weather_station, space, "longwaveRadiation", "longwaveRadiation")
            self.add_connection(weather_station, space, "outdoorTemperature", "outdoorTemperature")
            self.add_connection(occupancy_schedule, space, "scheduleValue", "numberOfPeople")

        for damper in damper_instances:
            if "C_C_" + damper.systemId[4:] in self.component_dict:
                co2_controller = self.component_dict["C_C_" + damper.systemId[4:]]
                self.add_connection(co2_controller, damper, "inputSignal", "damperPosition")

        for space_heater in space_heater_instances:
            space = self.component_dict[space_heater.systemId[3:]]
            valve = self.component_dict["V_" + space_heater.systemId[3:]]
            self.add_connection(space, space_heater, "indoorTemperature", "indoorTemperature") 
            self.add_connection(valve, space_heater, "waterFlowRate", "waterFlowRate")

        for valve in valve_instances:
            if "C_T_" + valve.systemId[2:] in self.component_dict:
                temperature_controller = self.component_dict["C_T_" + valve.systemId[2:]]
                self.add_connection(temperature_controller, valve, "inputSignal", "valvePosition")

        for coil_heating in coil_heating_instances:
            for system in coil_heating.subSystemOf:
                air_to_air_heat_recovery = [v for v in system.hasSubSystem if isinstance(v, AirToAirHeatRecoveryModel)]
                if len(air_to_air_heat_recovery)!=0:
                    air_to_air_heat_recovery = air_to_air_heat_recovery[0]
                    node = [v for v in system.hasSubSystem if isinstance(v, Node) and v.systemId[0:4] == "N_S_"][0]
                    self.add_connection(air_to_air_heat_recovery, coil_heating, "primaryTemperatureOut", "supplyAirTemperature")
                    self.add_connection(node, coil_heating, "flowRate", "airFlowRate")

        for coil_cooling in coil_cooling_instances:
            for system in coil_cooling.subSystemOf:
                air_to_air_heat_recovery = [v for v in system.hasSubSystem if isinstance(v, AirToAirHeatRecoveryModel)]
                if len(air_to_air_heat_recovery)!=0:
                    air_to_air_heat_recovery = air_to_air_heat_recovery[0]
                    node = [v for v in system.hasSubSystem if isinstance(v, Node) and v.systemId[0:4] == "N_S_"][0]
                    self.add_connection(air_to_air_heat_recovery, coil_cooling, "primaryTemperatureOut", "supplyAirTemperature")
                    self.add_connection(node, coil_cooling, "flowRate", "airFlowRate")

        for air_to_air_heat_recovery in air_to_air_heat_recovery_instances:
            ventilation_system = air_to_air_heat_recovery.subSystemOf[0]
            node_S = [v for v in ventilation_system.hasSubSystem if isinstance(v, Node) and v.systemId[0:4] == "N_S_"][0]
            node_E = [v for v in ventilation_system.hasSubSystem if isinstance(v, Node) and v.systemId[0:4] == "N_E_"][0]
            self.add_connection(weather_station, air_to_air_heat_recovery, "outdoorTemperature", "primaryTemperatureIn")
            self.add_connection(node_E, air_to_air_heat_recovery, "flowTemperatureOut", "secondaryTemperatureIn")
            self.add_connection(node_S, air_to_air_heat_recovery, "flowRate", "primaryAirFlowRate")
            self.add_connection(node_E, air_to_air_heat_recovery, "flowRate", "secondaryAirFlowRate")

        for fan in fan_instances:
            ventilation_system = fan.subSystemOf[0]
            if fan.systemId[0:4] == "F_S_":
                node_S = [v for v in ventilation_system.hasSubSystem if isinstance(v, Node) and v.systemId[0:4] == "N_S_"][0]
                self.add_connection(node_S, fan, "flowRate", "airFlowRate")
            elif fan.systemId[0:4] == "F_E_":
                node_E = [v for v in ventilation_system.hasSubSystem if isinstance(v, Node) and v.systemId[0:4] == "N_E_"][0]
                self.add_connection(node_E, fan, "flowRate", "airFlowRate")

        for controller in controller_instances:
            if controller.systemId[0:4] == "C_T_":
                self.add_connection(temperature_setpoint_schedule, controller, "scheduleValue", "setpointValue")
            elif controller.systemId[0:4] == "C_C_":
                self.add_connection(co2_setpoint_schedule, controller, "scheduleValue", "setpointValue")

        
    
    def load_model(self, read_config=True):
        self.add_weather_station()
        self.add_occupancy_schedule()
        self.add_temperature_setpoint_schedule()
        self.add_co2_setpoint_schedule()

        if read_config:
            self.read_config()
            self.connect()

        print("Finished loading model")


    def show_system_graph(self):
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

        nx.set_node_attributes(self.system_graph, values=self.system_graph_node_attribute_dict)
        graph = nx.drawing.nx_pydot.to_pydot(self.system_graph)

        graph.set_node_defaults(shape="circle", width=0.8, fixedsize="shape", margin=0, style="filled", fontname="Helvetica", color="#23a6db66", fontsize=10, colorscheme="oranges9")
        graph.set_edge_defaults(fontname="Helvetica", penwidth=2, color="#999999", fontcolor="#999999", fontsize=10, weight=3, minlen=1)

        self.system_graph = nx.drawing.nx_pydot.from_pydot(graph)

        nx.drawing.nx_pydot.write_dot(self.system_graph, 'system_graph.dot')
        # If Python can't find the dot executeable, change "app_path" variable to the full path
        app_path = shutil.which("dot")
        file_name = "system_graph"
        args = [app_path,
                "-Tpng",
                "-Kdot",
                "-Nstyle=filled",
                "-Nfixedsize=true",
                "-Grankdir=LR",
                "-Goverlap=scale",
                "-Gsplines=true",
                "-Gmargin=0",
                "-Gratio=fill",
                "-Gsize=15!",
                "-Gpack=true",
                "-Gdpi=1000",
                "-Grepulsiveforce=0.5",
                "-o" + file_name + ".png",
                file_name + ".dot"]
        subprocess.run(args=args)


    def show_execution_graph(self):
        self.execution_graph = nx.MultiDiGraph() ###
        self.execution_graph_node_attribute_dict = {}


        n = len(self.flat_execution_order)
        for i in range(n-1):
            sender_component = self.flat_execution_order[i]
            reciever_component = self.flat_execution_order[i+1]
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



         # If Python can't find the dot executeable, change "app_path" variable to the full path
        app_path = shutil.which("dot")
        file_name = "execution_graph"
        args = [app_path,
                "-Tpng",
                "-Kdot",
                "-Nstyle=filled",
                "-Nfixedsize=true",
                "-Grankdir=LR",
                "-Goverlap=scale",
                "-Gsplines=true",
                "-Gmargin=0",
                "-Gratio=fill",
                "-Gsize=7,5!",
                "-Gpack=true",
                "-Gdpi=1000",
                "-Grepulsiveforce=10",
                "-o" + file_name + ".png",
                file_name + ".dot"]
        subprocess.run(args=args)

    # @torch.jit.script
    def do_component_timestep(self, component):
        # print("----")
        # print(component.__class__.__name__)

        #Gather all needed inputs for the component through all ingoing connections
        for connection_point in component.connectsAt:
            connection = connection_point.connectsSystemThrough
            connected_component = connection.connectsSystem
            # print("--------------------------------")
            # print("------")
            # print(component.__class__.__name__)
            # print(connected_component.__class__.__name__)
            # print("------")
            # print(connection.senderPropertyName)
            # print(connection_point.recieverPropertyName)
            # print("------")
            # print(component.input)
            # print(connected_component.output)
            component.input[connection_point.recieverPropertyName] = connected_component.output[connection.senderPropertyName]
        component.update_output()
        component.update_report()



    
    # @torch.jit.script
    def do_system_time_step(self):
        for component_group in self.execution_order:
            # self.executor = ThreadPoolExecutor(max_workers=8)
            # self.executor.map(self.do_component_timestep, component_group)
            # self.executor.shutdown(wait=True)

            # futures = [self.executor.submit(self.do_component_timestep, component) for component in component_group]
            # wait(futures)

            # POOL = Pool()
            # POOL.map(self.do_component_timestep, component_group)
            # POOL.close()
            # POOL.join()

            # ray.get([self.do_component_timestep.remote(component) for component in component_group])

            # it = np.arange(len(component_group))
            # np.random.shuffle(it)
            # for i in it:
            #     self.do_component_timestep(component_group[i])


            for component in component_group:
                self.do_component_timestep(component)

            

            


    def get_simulation_timesteps(self):
        n_timesteps = math.floor((self.endPeriod-self.startPeriod).total_seconds()/self.timeStep)
        self.timeSteps = [self.startPeriod+datetime.timedelta(seconds=i*self.timeStep) for i in range(n_timesteps)]


    
    def simulate(self):
        self.get_simulation_timesteps()
        for time in tqdm(self.timeSteps):
            self.do_system_time_step()
            # print(time)
        for component in self.flat_execution_order:
            if component.createReport:
                component.plot_report(self.timeSteps)
        plt.show()


    def flatten(self, _list):
        return [item for sublist in _list for item in sublist]


    def get_component_dict_no_cycles(self):
        self.component_dict_no_cycles = copy.deepcopy(self.component_dict)
        space_instances = [v for v in self.component_dict_no_cycles.values() if isinstance(v, BuildingSpaceModel)]
        for space in space_instances:
            new_connectsAt = []
            for connection_point in space.connectsAt:
                connection = connection_point.connectsSystemThrough
                connected_component = connection.connectsSystem
                if len(connected_component.connectsAt)==0:
                    new_connectsAt.append(connection_point)
                else:
                    connected_component.connectedThrough.remove(connection)
            space.connectsAt = new_connectsAt

    def map_execution_order(self):
        self.execution_order = [[self.component_dict[component.systemId] for component in component_group] for component_group in self.execution_order]
        
    def get_execution_order(self):
        self.get_component_dict_no_cycles()
        self.initComponents = [v for v in self.component_dict_no_cycles.values() if len(v.connectsAt)==0]
        self.activeComponents = self.initComponents
        self.visitCount = {}
        self.execution_order = []
        self.execution_order.append(self.initComponents)
        while len(self.activeComponents)>0:
            self.traverse()

        self.map_execution_order()
        
        self.flat_execution_order = self.flatten(self.execution_order)
        assert len(self.flat_execution_order)==len(self.component_dict_no_cycles)

        


    def traverse(self):
        activeComponentsNew = []
        self.component_group = []
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
                    if ingoing_connection_point.recieverPropertyName not in connected_component.connectionVisits:
                        has_connections = False
                        break
                
                if has_connections:
                    self.component_group.append(connected_component)
                    if connected_component.connectedThrough is not None:
                        activeComponentsNew.append(connected_component)

        

        self.activeComponents = activeComponentsNew
        self.execution_order.append(self.component_group)


    def get_leaf_subsystems(self, system):
        for sub_system in system.hasSubSystem:
            if sub_system.hasSubSystem is None:
                self.leaf_subsystems.append(sub_system)
            else:
                self.get_leaf_subsystems(sub_system)




def run():
    createReport = False
    timeStep = 600
    startPeriod = datetime.datetime(year=2019, month=12, day=8, hour=0, minute=0, second=0, tzinfo=tzutc())
    endPeriod = datetime.datetime(year=2019, month=12, day=20, hour=0, minute=0, second=0, tzinfo=tzutc())
    model = EnergyModel(timeStep = timeStep,
                                startPeriod = startPeriod,
                                endPeriod = endPeriod,
                                createReport = createReport)

    model.load_model()
    model.get_execution_order()
    # model.show_execution_graph()
    # model.show_system_graph()

    del building_data_collection_dict.building_data_collection_dict
    
    model.simulate()


# import cProfile
# import pstats
# import io

# pr = cProfile.Profile()
# pr.enable()

# my_result = run()

# pr.disable()
# s = io.StringIO()
# ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
# ps.print_stats()

# with open('test.txt', 'w+') as f:
#     f.write(s.getvalue())


if __name__ == '__main__':
    if test:
        run()