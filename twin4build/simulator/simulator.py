from tqdm import tqdm
import datetime
import math
import numpy as np
import pandas as pd

import os
import sys

uppath = lambda _path,n: os.sep.join(_path.split(os.sep)[:-n])
file_path = uppath(os.path.abspath(__file__), 3)
sys.path.append(file_path)

from twin4build.saref4bldg.building_space.building_space_model import BuildingSpaceSystem
from twin4build.saref.device.sensor.sensor import Sensor
from twin4build.saref.device.meter.meter import Meter
import warnings
from twin4build.saref4bldg.physical_object.building_object.building_device.distribution_device.distribution_flow_device.energy_conversion_device.coil.coil import Coil
from twin4build.saref4bldg.physical_object.building_object.building_device.distribution_device.distribution_flow_device.flow_moving_device.fan.fan import Fan

from twin4build.utils.data_loaders.load_from_file import load_from_file
from twin4build.utils.uppath import uppath
from twin4build.utils.node import Node
import copy
from twin4build.logger.Logging import Logging

logger = Logging.get_logger("ai_logfile")

class Simulator():
    """
    The Simulator class simulates a model for a certain time period 
    using the <Simulator>.simulate(<Model>) method.
    """
    def __init__(self, 
                model=None,
                do_plot=False):
        self.model = model
        self.do_plot = do_plot
        logger.info("[Simulator Class] : Entered in Initialise Function")

    def do_component_timestep(self, component):
        #Gather all needed inputs for the component through all ingoing connections
        for connection_point in component.connectsAt:
            connection = connection_point.connectsSystemThrough
            connected_component = connection.connectsSystem
            if isinstance(component, BuildingSpaceSystem):
                assert np.isnan(connected_component.output[connection.senderPropertyName])==False, f"Model output {connection.senderPropertyName} of component {connected_component.id} is NaN."
            component.input[connection_point.receiverPropertyName] = connected_component.output[connection.senderPropertyName]
            if component.doUncertaintyAnalysis:
                component.inputUncertainty[connection_point.receiverPropertyName] = connected_component.outputUncertainty[connection.senderPropertyName]

        component.do_step(secondTime=self.secondTime, dateTime=self.dateTime, stepSize=self.stepSize)

        # component.update_simulation_result()

        # if isinstance(component, Fan):
        #     y_ref = [component.fmu_outputs[key].valueReference for key in component.FMUoutput.values()]
        #     x_ref = [component.fmu_parameters["c4"].valueReference]
        #     dv = [1]
        #     grad = component.fmu.getDirectionalDerivative(vUnknown_ref=y_ref, vKnown_ref=x_ref, dvKnown=dv)
        #     print("do_step: grad")
        #     print(grad)

    
    def do_system_time_step(self, model):
        """
        Do a system time step, i.e. execute the "do_step" method for each component model. 

        Notes:
        The list model.execution_order currently consists of component groups that can be executed in parallel 
        because they dont require any inputs from each other. 
        However, in python neither threading or multiprocessing yields any performance gains.
        If the framework is implemented in another language, e.g. C++, parallel execution of components is expected to yield significant performance gains. 
        Another promising option for optimization is to group all components with identical classes/models as well as priority and perform a vectorized "do_step" on all such models at once.
        This can be done in python using numpy or torch.      
        """
  
        for component_group in model.execution_order:
            for component in component_group:
                self.do_component_timestep(component)

        if self.trackGradients:
            self.get_gradient(self.targetParameters, self.targetMeasuringDevices)

        for component in model.flat_execution_order:
            component.update_simulation_result()

    def get_execution_order_reversed(self):
        self.execution_order_reversed = {}
        for targetMeasuringDevice in self.targetMeasuringDevices:
            n_inputs = {component: len(component.connectsAt) for component in self.model.component_dict.values()}
            n_outputs = {component: len(component.connectedThrough) for component in self.model.component_dict.values()}
            target_index = self.model.flat_execution_order.index(targetMeasuringDevice)
            self.execution_order_reversed[targetMeasuringDevice] = list(reversed(self.model.flat_execution_order[:target_index+1]))[:]
            items_removed = True
            while items_removed: # Assumes that a component must have at least 1 input or 1 output in the graph
                items_removed = False
                for component in self.execution_order_reversed[targetMeasuringDevice]:
                    if n_inputs[component]==0: # No inputs
                        if component not in self.targetParameters.keys():
                            self.execution_order_reversed[targetMeasuringDevice].remove(component)
                            for connection in component.connectedThrough:
                                connection_point = connection.connectsSystemAt
                                receiver_component = connection_point.connectionPointOf
                                n_inputs[receiver_component]-=1
                            items_removed = True
                    elif n_outputs[component]==0: # No outputs
                        if component is not targetMeasuringDevice:
                            self.execution_order_reversed[targetMeasuringDevice].remove(component)
                            for connection_point in component.connectsAt:
                                connection = connection_point.connectsSystemThrough
                                sender_component = connection.connectsSystem
                                n_outputs[sender_component]-=1
                            items_removed = True
                    elif targetMeasuringDevice not in self.model.depth_first_search(component):
                        self.execution_order_reversed[targetMeasuringDevice].remove(component)
                        for connection_point in component.connectsAt:
                            connection = connection_point.connectsSystemThrough
                            sender_component = connection.connectsSystem
                            n_outputs[sender_component]-=1
                        for connection in component.connectedThrough:
                            connection_point = connection.connectsSystemAt
                            receiver_component = connection_point.connectionPointOf
                            n_inputs[receiver_component]-=1
                        items_removed = True

            # Make parameterGradient dicts to hold values
            for component, attr_list in self.targetParameters.items():
                for attr in attr_list:
                    if targetMeasuringDevice not in component.parameterGradient:
                        component.parameterGradient[targetMeasuringDevice] = {attr: None}
                    else:
                        component.parameterGradient[targetMeasuringDevice][attr] = None

            # Make outputGradient dicts to hold values
            targetMeasuringDevice.outputGradient[targetMeasuringDevice] = {next(iter(targetMeasuringDevice.input)): None}
            for component in self.execution_order_reversed[targetMeasuringDevice]:
                for connection_point in component.connectsAt:
                    connection = connection_point.connectsSystemThrough
                    sender_component = connection.connectsSystem
                    sender_property_name = connection.senderPropertyName
                    if targetMeasuringDevice not in sender_component.outputGradient:
                        sender_component.outputGradient[targetMeasuringDevice] = {sender_property_name: None}
                    else:
                        sender_component.outputGradient[targetMeasuringDevice][sender_property_name] = None

    def reset_grad(self, targetMeasuringDevice):
        """
        Resets the gradients
        """
        # Make parameterGradient dicts to hold values
        for component, attr_list in self.targetParameters.items():
            for attr in attr_list:
                component.parameterGradient[targetMeasuringDevice][attr] = 0

        targetMeasuringDevice.outputGradient[targetMeasuringDevice][next(iter(targetMeasuringDevice.input))] = 1
        for component in self.execution_order_reversed[targetMeasuringDevice]:
            for connection_point in component.connectsAt:
                connection = connection_point.connectsSystemThrough
                sender_component = connection.connectsSystem
                sender_property_name = connection.senderPropertyName
                sender_component.outputGradient[targetMeasuringDevice][sender_property_name] = 0

        
    def get_gradient(self, targetParameters, targetMeasuringDevices):
        """
        The list execution_order_reversed can be pruned by recursively removing nodes 
        with input size=0 for components which does not contain target parameters components
        Same thing for nodes with output size=0 (measuring devices)
        """
        for targetMeasuringDevice in targetMeasuringDevices:
            self.reset_grad(targetMeasuringDevice)
            for component in self.execution_order_reversed[targetMeasuringDevice]:
                if component in targetParameters:
                    for attr in targetParameters[component]:
                        grad_dict = component.get_subset_gradient(attr, y_keys=component.outputGradient[targetMeasuringDevice].keys(), as_dict=True)
                        for key in grad_dict.keys():
                            component.parameterGradient[targetMeasuringDevice][attr] += component.outputGradient[targetMeasuringDevice][key]*grad_dict[key]
            
                for connection_point in component.connectsAt:
                    connection = connection_point.connectsSystemThrough
                    sender_component = connection.connectsSystem
                    sender_property_name = connection.senderPropertyName
                    receiver_property_name = connection_point.receiverPropertyName
                    grad_dict = component.get_subset_gradient(receiver_property_name, y_keys=component.outputGradient[targetMeasuringDevice].keys(), as_dict=True)
                    for key in grad_dict.keys():
                        sender_component.outputGradient[targetMeasuringDevice][sender_property_name] += component.outputGradient[targetMeasuringDevice][key]*grad_dict[key]
                    # print("-----")
                    # print(targetMeasuringDevice.id)
                    # print(sender_component.id)
                    # print(component.id)
                    # print(component.input)
                    
                    # print(grad_dict, receiver_property_name, component.outputGradient[targetMeasuringDevice].keys())
                    # print(component.outputGradient[targetMeasuringDevice])
                    # print(sender_component.outputGradient[targetMeasuringDevice])

    def get_simulation_timesteps(self, startPeriod, endPeriod, stepSize):
        n_timesteps = math.floor((endPeriod-startPeriod).total_seconds()/stepSize)
        self.secondTimeSteps = [i*stepSize for i in range(n_timesteps)]
        self.dateTimeSteps = [startPeriod+datetime.timedelta(seconds=i*stepSize) for i in range(n_timesteps)]
 
    
    def simulate(self, model, startPeriod, endPeriod, stepSize, trackGradients=False, targetParameters=None, targetMeasuringDevices=None, show_progress_bar=True):
        """
        Simulate the "model" between the dates "startPeriod" and "endPeriod" with timestep equal to "stepSize" in seconds. 
        """
        assert targetParameters is not None and targetMeasuringDevices is not None if trackGradients else True, "Arguments targetParameters and targetMeasuringDevices must be set if trackGradients=True"
        self.model = model
        self.startPeriod = startPeriod
        self.endPeriod = endPeriod
        self.stepSize = stepSize
        self.trackGradients = trackGradients
        self.targetParameters = targetParameters
        self.targetMeasuringDevices = targetMeasuringDevices
        if trackGradients:
            assert isinstance(targetParameters, dict), "The argument targetParameters must be a dictionary"
            assert isinstance(targetMeasuringDevices, list), "The argument targetMeasuringDevices must be a list of Sensor and Meter objects"
            self.model.set_trackGradient(True)
            self.get_execution_order_reversed()
        self.model.initialize(startPeriod=startPeriod, endPeriod=endPeriod, stepSize=stepSize)
        self.get_simulation_timesteps(startPeriod, endPeriod, stepSize)
        logger.info("Running simulation")
        if show_progress_bar:
            for self.secondTime, self.dateTime in tqdm(zip(self.secondTimeSteps,self.dateTimeSteps), total=len(self.dateTimeSteps)):
                self.do_system_time_step(self.model)
        else:
            for self.secondTime, self.dateTime in zip(self.secondTimeSteps,self.dateTimeSteps):
                self.do_system_time_step(self.model)


        for component in self.model.flat_execution_order:
            if component.saveSimulationResult and self.do_plot:
                component.plot_report(self.dateTimeSteps)

    
    def get_simulation_readings(self):
        df_simulation_readings = pd.DataFrame()
        time = self.dateTimeSteps
        df_simulation_readings.insert(0, "time", time)
        sensor_instances = self.model.get_component_by_class(self.model.component_dict, Sensor)
        meter_instances = self.model.get_component_by_class(self.model.component_dict, Meter)
                
        for sensor in sensor_instances:
            savedOutput = self.model.component_dict[sensor.id].savedOutput
            key = list(savedOutput.keys())[0]
            simulation_readings = savedOutput[key]
            df_simulation_readings.insert(0, sensor.id, simulation_readings)

        for meter in meter_instances:
            savedOutput = self.model.component_dict[meter.id].savedOutput
            key = list(savedOutput.keys())[0]
            simulation_readings = savedOutput[key]
            df_simulation_readings.insert(0, meter.id, simulation_readings)
        return df_simulation_readings
    
    def get_actual_readings(self, startPeriod, endPeriod, stepSize):
        print("Collecting actual readings...")
        """
        This is a temporary method for retrieving actual sensor readings.
        Currently it simply reads from csv files containing historic data.
        In the future, it should read from quantumLeap.  
        """
        self.get_simulation_timesteps(startPeriod, endPeriod, stepSize)
        logger.info("[Simulator Class] : Entered in Get Actual Readings Function")
        format = "%m/%d/%Y %I:%M:%S %p" # Date format used for loading data from csv files
        id_to_csv_map = {"Space temperature sensor": "OE20-601b-2_Indoor air temperature (Celcius)",
                         "Space CO2 sensor": "OE20-601b-2_CO2 (ppm)",
                         "Valve position sensor": "OE20-601b-2_Space heater valve position",
                         "Damper position sensor": "OE20-601b-2_Damper position",
                         "Shading position sensor": "",
                         "VE02 Primary Airflow Temperature BHR sensor": "weather_BMS",
                         "Heat recovery temperature sensor": "VE02_FTG_MIDDEL",
                         "Heating coil temperature sensor": "VE02_FTI1",
                         "VE02 Secondary Airflow Temperature BHR sensor": "VE02_FTU1",
                         "Heating meter": "",
                         "test123": "VE02_airflowrate_supply_kg_s",
                         "VE02 Primary Airflow Temperature AHR sensor": "VE02_FTG_MIDDEL",
                         "VE02 Primary Airflow Temperature AHC sensor": "VE02_FTI1",
                         "fan power meter": "VE02_power_VI",
                         "coil outlet air temperature sensor": "VE02_FTI1",
                         "fan inlet air temperature sensor": "",
                         "coil outlet water temperature sensor": "VE02_FTT1",
                         "fan outlet air temperature sensor": "VE02_FTG_MIDDEL",
                         "valve position sensor": "VE02_MVV1"
                         }
        
        df_actual_readings = pd.DataFrame()
        time = self.dateTimeSteps
        df_actual_readings.insert(0, "time", time)
        sensor_instances = self.model.get_component_by_class(self.model.component_dict, Sensor)
        meter_instances = self.model.get_component_by_class(self.model.component_dict, Meter)
                
        for sensor in sensor_instances:
            if sensor.id in id_to_csv_map:
                filename = f"{id_to_csv_map[sensor.id]}.csv"
                filename = os.path.join(os.path.abspath(uppath(os.path.abspath(__file__), 2)), "test", "data", "time_series_data", filename)
                if os.path.isfile(filename):
                    actual_readings = load_from_file(filename=filename, stepSize=stepSize, start_time=startPeriod, end_time=endPeriod, format=format, dt_limit=999999)
                    df_actual_readings.insert(0, sensor.id, actual_readings.iloc[:,1])
                else:
                    warnings.warn(f"No file named: \"{filename}\"\n Skipping sensor: \"{sensor.id}\"")
                    logger.error(f"No file named: \"{filename}\"\n Skipping sensor: \"{sensor.id}\"")

        for meter in meter_instances:
            if meter.id in id_to_csv_map:
                filename = f"{id_to_csv_map[meter.id]}.csv"
                filename = os.path.join(os.path.abspath(uppath(os.path.abspath(__file__), 2)), "test", "data", "time_series_data", filename)
                if os.path.isfile(filename):
                    actual_readings = load_from_file(filename=filename, stepSize=stepSize, start_time=startPeriod, end_time=endPeriod, format=format, dt_limit=999999)
                    df_actual_readings.insert(0, meter.id, actual_readings.iloc[:,1])
                else:
                    warnings.warn(f"No file named: \"{filename}\"\n Skipping meter: \"{meter.id}\"")
                    logger.error(f"No file named: \"{filename}\"\n Skipping meter: \"{meter.id}\"")

                
        logger.info("[Simulator Class] : Exited from Get Actual Readings Function")


        return df_actual_readings