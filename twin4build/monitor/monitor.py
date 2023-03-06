
from twin4build.simulator.simulator import Simulator
from twin4build.saref.device.sensor.sensor import Sensor
from twin4build.saref.device.meter.meter import Meter
from twin4build.utils.data_loaders.load_from_file import load_from_file
from twin4build.utils.uppath import uppath
from twin4build.utils.plot.plot import get_fig_axes, load_params
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings


class Monitor:
    """
    This Monitor class monitors the performance of a building.
    """
    def __init__(self, model):
        self.model = model
        self.simulator = Simulator()

    def get_actual_readings(self, startPeriod, endPeriod, stepSize):
        print("Collecting actual readings...")
        """
        This is a temporary method for retieving actual sensor readings.
        Currently it simply reads from csv files.
        In the future, it should read from quantumLeap.  
        """
        format = "%m/%d/%Y %I:%M:%S %p" # Date format used for loading data from csv files
        id_to_csv_map = {"Indoor temperature sensor": "OE20-601b-2_Indoor air temperature (Celcius)",
                         "Indoor CO2 sensor": "OE20-601b-2_CO2 (ppm)",
                         "Valve position sensor": "OE20-601b-2_Space heater valve position",
                         "Damper position sensor": "OE20-601b-2_Damper position",
                         "Shading position sensor": "",
                         "VE02 Primary Airflow Temperature BHR sensor": "weather_BMS",
                         "VE02 Primary Airflow Temperature AHR sensor": "VE02_FTG_MIDDEL",
                         "VE02 Primary Airflow Temperature AHC sensor": "VE02_FTI1",
                         "VE02 Secondary Airflow Temperature BHR sensor": "VE02_FTU1",
                         "Heating meter": "",
                         }
        
        df_actual_readings = pd.DataFrame()
        time = self.simulator.dateTimeSteps
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

    def get_simulation_readings(self):
        df_simulation_readings = pd.DataFrame()
        time = self.simulator.dateTimeSteps
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
    
    def plot_performance(self, df_simulation_readings, df_actual_readings):
        colors = sns.color_palette("deep")
        blue = colors[0]
        orange = colors[1]
        green = colors[2]
        red = colors[3]
        purple = colors[4]
        brown = colors[5]
        pink = colors[6]
        grey = colors[7]
        beis = colors[8]
        sky_blue = colors[9]
        load_params()
        
        for key in list(df_actual_readings.columns): #iterate thorugh keys and skip first key which is "time"
            if key!="time":
                fig,axes = get_fig_axes(key)
                axes[0].plot(df_simulation_readings["time"], df_simulation_readings[key], color="black", linestyle="dashed")
                axes[0].plot(df_actual_readings["time"], df_actual_readings[key], color=blue)
                fig.suptitle(key)

        plt.show()

    def monitor(self, 
                startPeriod=None,
                endPeriod=None,
                stepSize=600):
        
        self.simulator.simulate(self.model,
                                stepSize=stepSize,
                                startPeriod=startPeriod,
                                endPeriod=endPeriod)
        


        df_simulation_readings = self.get_simulation_readings()
        df_actual_readings = self.get_actual_readings(startPeriod, endPeriod, stepSize)

        self.plot_performance(df_simulation_readings, df_actual_readings)


        # metrics

        # RMSE 
        # MSE
        # MAE
        # Relative error

