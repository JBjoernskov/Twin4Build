
from twin4build.simulator.simulator import Simulator
from twin4build.saref.device.sensor.sensor import Sensor
from twin4build.saref.device.meter.meter import Meter
from twin4build.utils.data_loaders.load_from_file import load_from_file
from twin4build.utils.uppath import uppath
from twin4build.utils.plot.plot import get_fig_axes, load_params
from twin4build.saref.property_.temperature.temperature import Temperature
from twin4build.saref.property_.Co2.Co2 import Co2
from twin4build.saref.property_.opening_position.opening_position import OpeningPosition #This is in use
from twin4build.saref.property_.energy.energy import Energy #This is in use
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
import matplotlib.dates as mdates


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
                         "test123": "VE02_airflowrate_supply_kg_s",
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
    
    def get_ylabel(self, key):
        property_ = self.model.component_dict[key].measuresProperty
        if isinstance(property_, Temperature):
            ylabel = r"Temperature [$^\circ$C]"
        elif isinstance(property_, OpeningPosition):
            ylabel = r"Position [0-100 \%]"
        elif isinstance(property_, Energy):
            ylabel = r"Energy [kWh]"
        else:
            #More properties should be added if needed
            raise Exception(f"Unknown Property {str(type(property_))}")
        return ylabel
    
    def get_legend_label(self, key):
        property_ = self.model.component_dict[key].measuresProperty
        if isinstance(property_, Temperature):
            legend = r"Temperature"
        elif isinstance(property_, OpeningPosition):
            legend = r"Position"
        elif isinstance(property_, Energy):
            legend = r"Energy"
        else:
            #More properties should be added if needed
            raise Exception(f"Unknown Property {str(type(property_))}")
        return legend

    def save_plots(self):
        for key, (fig,axes) in self.plot_dict.items():
            fig.savefig(f"{key}.png", dpi=300)
    
    def plot_performance(self, df_simulation_readings, df_actual_readings, save_plots=False):
        self.colors = sns.color_palette("deep")
        blue = self.colors[0]
        orange = self.colors[1]
        green = self.colors[2]
        red = self.colors[3]
        purple = self.colors[4]
        brown = self.colors[5]
        pink = self.colors[6]
        grey = self.colors[7]
        beis = self.colors[8]
        sky_blue = self.colors[9]
        load_params()
        
        self.plot_dict = {}
        for key in list(df_actual_readings.columns): #iterate thorugh keys and skip first key which is "time"
            if key!="time":
                ylabel = self.get_ylabel(key)
                legend_label = self.get_legend_label(key)
                fig,axes = plt.subplots(2, sharex=True)
                self.plot_dict[key] = (fig,axes)
                
                axes[0].plot(df_actual_readings["time"], df_actual_readings[key], color=blue, label=f"{legend_label} measured")
                axes[0].plot(df_simulation_readings["time"], df_simulation_readings[key], color="black", linestyle="dashed", label=f"{legend_label} predicted", linewidth=2)
                axes[0].set_ylabel(ylabel=ylabel)
                err = (df_actual_readings[key]-df_simulation_readings[key])/np.abs(df_actual_readings[key])*100
                err_moving_average = self.get_moving_average(err)
                axes[1].plot(df_actual_readings["time"], err, color=blue, label="Relative error")
                axes[1].plot(df_actual_readings["time"], err_moving_average, color=orange, label=f"Moving average")
                axes[1].set_ylabel(ylabel=r"Perfomance gap [%]")

                facecolor = tuple(list(beis)+[0.5])
                edgecolor = tuple(list((0,0,0))+[0.5])
                fill_limit = 5
                axes[1].fill_between(df_actual_readings["time"], y1=-fill_limit, y2=fill_limit, facecolor=facecolor, edgecolor=edgecolor, label=r"5% error band")
                axes[1].set_ylim([-30,30])

                myFmt = mdates.DateFormatter('%d-%m')
                axes[0].xaxis.set_major_formatter(myFmt)
                axes[1].xaxis.set_major_formatter(myFmt)

                axes[0].xaxis.set_tick_params(rotation=45)
                axes[1].xaxis.set_tick_params(rotation=45)
                fig.suptitle(key)
                fig.set_size_inches(14, 8)
                axes[0].legend()
                axes[1].legend()


    def get_moving_average(self, x):
        moving_average = x.rolling(144).mean()
        return moving_average

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

