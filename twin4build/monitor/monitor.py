
import sys 
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
import matplotlib.dates as mdates
import matplotlib.ticker as ticker


uppath = lambda _path,n: os.sep.join(_path.split(os.sep)[:-n])
file_path = uppath(os.path.abspath(__file__), 3)
sys.path.append(file_path)

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

from twin4build.logger.Logging import Logging

logger = Logging.get_logger("ai_logfile")

class Monitor:
    """
    This Monitor class monitors the performance of a building.
    """
    def __init__(self, model):
        self.model = model
        self.simulator = Simulator()
    
    def get_ylabel(self, key):
        property_ = self.model.component_dict[key].measuresProperty
        if isinstance(property_, Temperature):
            ylabel = r"Temperature [$^\circ$C]"
        elif isinstance(property_, Co2):
            ylabel = r"CO$_2$ [ppm]"
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
        elif isinstance(property_, Co2):
            legend = r"CO$_2$-concentration"
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
        
        '''
            plot_performance that takes two dataframes as inputs and plots the 
            performance of a simulation against actual data. The function uses the 
            Seaborn library for plotting and generates a set of subplots showing physical and virtual data, 
            performance gap, and anomaly signals. The function also applies moving averages and error bands to the 
            plots to make them easier to interpret.
        '''
        
        logger.info("[Monitor Class] : Entered in Plot Performance Function")
        
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

        error_band = 5
        for key in list(df_actual_readings.columns): #iterate thorugh keys and skip first key which is "time"
            if key!="time":
                ylabel = self.get_ylabel(key)
                legend_label = self.get_legend_label(key)
                fig,axes = plt.subplots(2, sharex=True)
                self.plot_dict[key] = (fig,axes)
                
                axes[0].plot(df_actual_readings["time"], df_actual_readings[key], color=blue, label=f"Physical")
                axes[0].plot(df_simulation_readings["time"], df_simulation_readings[key], color="black", linestyle="dashed", label=f"Virtual", linewidth=2)
                # axes[0].set_ylabel(ylabel=ylabel)
                fig.text(0.015, 0.74, ylabel, va='center', ha='center', rotation='vertical', fontsize=13, color="black")
                err = (df_actual_readings[key]-df_simulation_readings[key])/np.abs(df_actual_readings[key])*100
                err_moving_average = self.get_moving_average(err)
                axes[1].plot(df_actual_readings["time"], err, color=blue, label="Relative error")
                axes[1].plot(df_actual_readings["time"], err_moving_average, color=orange, label=f"Moving average")
                # axes[1].set_ylabel(ylabel=r"Perfomance gap [%]")
                fig.text(0.015, 0.3, r"Perfomance gap [%]", va='center', ha='center', rotation='vertical', fontsize=13, color="black")

                facecolor = tuple(list(beis)+[0.5])
                edgecolor = tuple(list((0,0,0))+[0.5])
                axes[1].fill_between(df_actual_readings["time"], y1=-error_band, y2=error_band, facecolor=facecolor, edgecolor=edgecolor, label=f"{error_band}% error band")
                axes[1].set_ylim([-30,30])

                myFmt = mdates.DateFormatter('%a')
                axes[0].xaxis.set_major_formatter(myFmt)
                axes[1].xaxis.set_major_formatter(myFmt)

                axes[0].xaxis.set_tick_params(rotation=45)
                axes[1].xaxis.set_tick_params(rotation=45)
                fig.suptitle(key, fontsize=18)
                fig.set_size_inches(7, 5)
                axes[0].legend()
                axes[1].legend()


        subset = ["Space temperature sensor", "Heat recovery temperature sensor", "Heating coil temperature sensor"]######
        # subset = ["Space temperature sensor", "VE02 Primary Airflow Temperature AHR sensor", "VE02 Primary Airflow Temperature AHC sensor"]
        fig,axes = plt.subplots(len(subset), sharex=True)
        fig.set_size_inches(7, 5)
        fig.suptitle("Anomaly signals", fontsize=18)
        self.plot_dict["monitor"] = (fig,axes)
        
        for ax, key in zip(axes, subset): #iterate thorugh keys and skip first key which is "time"
            if key!="time" and key in subset:
                df_err = (df_actual_readings[key]-df_simulation_readings[key])/np.abs(df_actual_readings[key])*100
                df_err_moving_average = self.get_moving_average(df_err)
                df_err_moving_average["signal"] = df_err_moving_average.abs()
                df_err_moving_average["signal"] = df_err_moving_average["signal"]>error_band
                ax.plot(df_actual_readings["time"], df_err_moving_average["signal"], color=blue, label=f"Signal: {key}")
                ax.legend(prop={'size': 8}, loc="best")
                ax.set_ylim([-0.05,1.05])

                myFmt = mdates.DateFormatter('%a')
                ax.xaxis.set_major_formatter(myFmt)
                ax.xaxis.set_tick_params(rotation=45)
                ax.xaxis.set_major_locator(ticker.MultipleLocator(1))

        
        logger.info("[Monitor Class] : Exited from Plot Performance Function")
        

    def get_moving_average(self, x):
        moving_average = x.rolling(144, min_periods=10).mean()
        return moving_average

    def monitor(self, 
                startPeriod=None,
                endPeriod=None,
                stepSize=600):
        
        self.simulator.simulate(self.model,
                                stepSize=stepSize,
                                startPeriod=startPeriod,
                                endPeriod=endPeriod)
        


        df_simulation_readings = self.simulator.get_simulation_readings()
        df_actual_readings = self.simulator.get_actual_readings(startPeriod, endPeriod, stepSize)
        self.plot_performance(df_simulation_readings, df_actual_readings)


        # metrics

        # RMSE 
        # MSE
        # MAE
        # Relative error

