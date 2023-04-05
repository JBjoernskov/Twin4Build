from tqdm import tqdm
import datetime
import math
import numpy as np
import pandas as pd
from twin4build.saref4bldg.building_space.building_space_model import BuildingSpaceModel
from twin4build.saref.device.sensor.sensor import Sensor
from twin4build.saref.device.meter.meter import Meter
import warnings
import os
from twin4build.utils.data_loaders.load_from_file import load_from_file
from twin4build.utils.uppath import uppath
from twin4build.utils.node import Node
class Simulator:
    """
    The Simulator class simulates a model for a certain time period 
    using the <Simulator>.simulate(<Model>) method.
    """
    def __init__(self, 
                do_plot=False):
        self.do_plot = do_plot

    def do_component_timestep(self, component):
        #Gather all needed inputs for the component through all ingoing connections
        for connection_point in component.connectsAt:
            connection = connection_point.connectsSystemThrough
            connected_component = connection.connectsSystem
            if isinstance(component, BuildingSpaceModel):
                assert np.isnan(connected_component.output[connection.senderPropertyName])==False, f"Model output {connection.senderPropertyName} of component {connected_component.id} is NaN."
            component.input[connection_point.recieverPropertyName] = connected_component.output[connection.senderPropertyName]
        component.do_step(secondTime=self.secondTime, dateTime=self.dateTime, stepSize=self.stepSize)
        component.update_report()

    def do_system_time_step(self, model):
        """
        Do a system time step, i.e. execute the "do_step" for each component model. 

        
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

    def get_simulation_timesteps(self):
        n_timesteps = math.floor((self.endPeriod-self.startPeriod).total_seconds()/self.stepSize)
        self.secondTimeSteps = [i*self.stepSize for i in range(n_timesteps)]
        self.dateTimeSteps = [self.startPeriod+datetime.timedelta(seconds=i*self.stepSize) for i in range(n_timesteps)]
 
    def simulate(self, model, startPeriod, endPeriod, stepSize):
        """
        Simulate the "model" between the dates "startPeriod" and "endPeriod" with timestep equal to "stepSize" in seconds. 
        """
        self.startPeriod = startPeriod
        self.endPeriod = endPeriod
        self.stepSize = stepSize
        self.model = model
        self.model.initialize(startPeriod=startPeriod, endPeriod=endPeriod, stepSize=stepSize)
        self.get_simulation_timesteps()
        print("Running simulation")
        for self.secondTime, self.dateTime in tqdm(zip(self.secondTimeSteps,self.dateTimeSteps), total=len(self.dateTimeSteps)):
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
        This is a temporary method for retieving actual sensor readings.
        Currently it simply reads from csv files.
        In the future, it should read from quantumLeap.  
        """
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
                         }
        
        df_actual_readings = pd.DataFrame()
        time = self.dateTimeSteps
        df_actual_readings.insert(0, "time", time)
        sensor_instances = self.model.get_component_by_class(self.model.component_dict, Sensor)
        meter_instances = self.model.get_component_by_class(self.model.component_dict, Meter)
                
        for sensor in sensor_instances:
            filename = f"{id_to_csv_map[sensor.id]}.csv"
            filename = os.path.join(os.path.abspath(uppath(os.path.abspath(__file__), 2)), "test", "data", "time_series_data", filename)
            if os.path.isfile(filename):
                actual_readings = load_from_file(filename=filename, stepSize=stepSize, start_time=startPeriod, end_time=endPeriod, format=format, dt_limit=999999)
                df_actual_readings.insert(0, sensor.id, actual_readings.iloc[:,1])
            else:
                warnings.warn(f"No file named: \"{filename}\"\n Skipping sensor: \"{sensor.id}\"")

        for meter in meter_instances:
            filename = f"{id_to_csv_map[meter.id]}.csv"
            filename = os.path.join(os.path.abspath(uppath(os.path.abspath(__file__), 2)), "test", "data", "time_series_data", filename)
            if os.path.isfile(filename):
                actual_readings = load_from_file(filename=filename, stepSize=stepSize, start_time=startPeriod, end_time=endPeriod, format=format, dt_limit=999999)
                df_actual_readings.insert(0, meter.id, actual_readings.iloc[:,1])
            else:
                warnings.warn(f"No file named: \"{filename}\"\n Skipping meter: \"{meter.id}\"")

        return df_actual_readings