
from twin4build.simulator.simulator import Simulator
from twin4build.saref.device.sensor.sensor import Sensor
from twin4build.saref.device.meter.meter import Meter
from twin4build.utils.data_loaders.load_spreadsheet import load_spreadsheet
from twin4build.utils.uppath import uppath
from twin4build.utils.plot.plot import get_fig_axes, load_params
from twin4build.saref.property_.temperature.temperature import Temperature
from twin4build.saref.property_.Co2.Co2 import Co2
from twin4build.saref.property_.opening_position.opening_position import OpeningPosition #This is in use
from twin4build.saref.property_.energy.energy import Energy #This is in use
from twin4build.saref.property_.power.power import Power #This is in use
from twin4build.saref.property_.flow.flow import Flow
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
import matplotlib.dates as mdates
import matplotlib.ticker as ticker


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
        elif isinstance(property_, Power):
            ylabel = r"Power [kW]"
        elif isinstance(property_, Flow):
            ylabel = r"Flow [$m^3$/h]"
        else:
            #More properties should be added if needed
            raise Exception(f"Unknown Property {str(type(property_))}")
        return ylabel
    
    # def get_legend_label(self, key):
    #     property_ = self.model.component_dict[key].measuresProperty
    #     if isinstance(property_, Temperature):
    #         legend_label = r"Temperature"

            
    #     elif isinstance(property_, Co2):
    #         legend_label = r"CO$_2$-concentration"
    #     elif isinstance(property_, OpeningPosition):
    #         legend_label = r"Position"
    #     elif isinstance(property_, Energy):
    #         legend_label = r"Energy"
    #     elif isinstance(property_, Power):
    #         legend_label = r"Power"
    #     else:
    #         #More properties should be added if needed
    #         raise Exception(f"Unknown Property {str(type(property_))}")
    #     return legend
    
    def get_error(self, key):
        err = (self.df_actual_readings[key]-self.df_simulation_readings[key])
        return err

    def get_relative_error(self, key):
        err = (self.df_actual_readings[key]-self.df_simulation_readings[key])/np.abs(self.df_actual_readings[key])*100
        return err
    
    def get_performance_gap(self, key):
        property_ = self.model.component_dict[key].measuresProperty
        error_band_abs = 2
        error_band_relative = 5 #%
        if isinstance(property_, Temperature):
            error_band = error_band_abs
            err = self.get_error(key)
            legend_label = f"{error_band}$^\circ$C error band"
        elif isinstance(property_, Co2):
            error_band = error_band_abs
            err = self.get_error(key)
            legend_label = f"{error_band} CO$_2$ error band"
        elif isinstance(property_, OpeningPosition):
            error_band = error_band_abs
            err = self.get_error(key)
            legend_label = f"{error_band}$ position error band"
        elif isinstance(property_, Energy):
            error_band = error_band_abs
            err = self.get_error(key)
            legend_label = f"{error_band} kWh error band"
        elif isinstance(property_, Power):
            error_band = error_band_relative
            err = self.get_relative_error(key)
            legend_label = f"{error_band}% error band"
        elif isinstance(property_, Flow):
            error_band = error_band_relative
            err = self.get_relative_error(key)
            legend_label = f"{error_band}% error band"
        else:
            #More properties should be added if needed
            raise Exception(f"Unknown Property {str(type(property_))}")
        return err, error_band, legend_label

    def save_plots(self):
        for key, (fig,axes) in self.plot_dict.items():
            fig.savefig(f"{key}.png", dpi=300)

            
    def plot_performance(self, save_plots=False):
        '''
            plot_performance that takes two dataframes as inputs and plots the 
            performance of a simulation against actual data. The function uses the 
            Seaborn library for plotting and generates a set of subplots showing physical and virtual data, 
            performance gap, and anomaly signals. The function also applies moving averages and error bands to the 
            plots to make them easier to interpret.
        '''
        
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
        error_band = 1
        for id_ in list(self.df_actual_readings.columns): #iterate thorugh keys and skip first key which is "time"
            facecolor = tuple(list(beis)+[0.5])
            edgecolor = tuple(list((0,0,0))+[0.5])
            ylabel = self.get_ylabel(id_)
            # legend_label = self.get_legend_label(key)
            fig,axes = plt.subplots(2, sharex=True)
            self.plot_dict[id_] = (fig,axes)

            if self.model.component_dict[id_].doUncertaintyAnalysis:
                key = list(self.model.component_dict[id_].inputUncertainty.keys())[0]
                output = np.array(self.model.component_dict[id_].savedOutput[key])
                outputUncertainty = np.array(self.model.component_dict[id_].savedOutputUncertainty[key])
                axes[0].fill_between(self.simulator.dateTimeSteps, y1=output-outputUncertainty, y2=output+outputUncertainty, facecolor=facecolor, edgecolor=edgecolor, label="Prediction uncertainty")
                
            axes[0].plot(self.df_actual_readings.index, self.df_actual_readings[id_], color=blue, label=f"Physical")
            axes[0].plot(self.df_simulation_readings.index, self.df_simulation_readings[id_], color="black", linestyle="dashed", label=f"Virtual", linewidth=2)
            # axes[0].set_ylabel(ylabel=ylabel)
            fig.text(0.015, 0.74, ylabel, va='center', ha='center', rotation='vertical', fontsize=13, color="black")
            # err = (df_actual_readings[key]-df_simulation_readings[key])/np.abs(df_actual_readings[key])*100
            pg, error_band, legend_label = self.get_performance_gap(id_)
            pg_moving_average = self.get_moving_average(pg)
            axes[1].plot(self.df_actual_readings.index, pg, color=blue, label="Residual")
            axes[1].plot(self.df_actual_readings.index, pg_moving_average, color=orange, label=f"Moving average")
            # axes[1].set_ylabel(ylabel=r"Perfomance gap [%]")
            fig.text(0.015, 0.3, r"Perfomance gap [$^\circ$C]", va='center', ha='center', rotation='vertical', fontsize=13, color="black")

            
            axes[1].fill_between(self.df_actual_readings.index, y1=-error_band, y2=error_band, facecolor=facecolor, edgecolor=edgecolor, label=legend_label)
            # axes[1].set_ylim([-30,30])
            # axes[1].set_ylim([-3,3])

            # myFmt = mdates.DateFormatter('%a') #Weekday
            myFmt = mdates.DateFormatter('%H')
            axes[0].xaxis.set_major_formatter(myFmt)
            axes[1].xaxis.set_major_formatter(myFmt)

            axes[0].xaxis.set_tick_params(rotation=45)
            axes[1].xaxis.set_tick_params(rotation=45)
            fig.suptitle(id_, fontsize=18)
            fig.set_size_inches(7, 5)
            axes[0].legend()
            axes[1].legend()

        subset = self.df_actual_readings.columns
        # subset = ["Space temperature sensor", "Heat recovery temperature sensor", "Heating coil temperature sensor"]#BS2023


        fig,axes = plt.subplots(len(subset), sharex=True)
        fig.set_size_inches(7, 5)
        fig.suptitle("Anomaly signals", fontsize=18)
        self.plot_dict["monitor"] = (fig,axes)
        
        for ax, key in zip(axes, subset): #iterate thorugh keys and skip first key which is "time"
            if key!="time" and key in subset:
                pg, error_band, legend_label = self.get_performance_gap(key)
                pg_moving_average = self.get_moving_average(pg)
                pg_moving_average["signal"] = pg_moving_average.abs()
                pg_moving_average["signal"] = pg_moving_average["signal"]>error_band
                ax.plot(self.df_actual_readings.index, pg_moving_average["signal"], color=blue, label=f"Signal: {key}")
                ax.legend(prop={'size': 8}, loc="best")
                ax.set_ylim([-0.05,1.05])

                myFmt = mdates.DateFormatter('%a')
                ax.xaxis.set_major_formatter(myFmt)
                ax.xaxis.set_tick_params(rotation=45)
                ax.xaxis.set_major_locator(ticker.MultipleLocator(1))

    def get_moving_average(self, x):
        moving_average = x.rolling(144, min_periods=10).mean()
        return moving_average
    
    def get_MSE(self):
        MSE = {}
        for key in list(self.df_actual_readings.columns): #iterate thorugh keys and skip first key which is "time"
            if key!="time":
                value = ((self.df_actual_readings[key]-self.df_simulation_readings[key])**2)
                MSE[key] = value.mean()
        return MSE
        
    def get_RMSE(self):
        RMSE = {}
        MSE = self.get_MSE()
        for key in MSE:
            RMSE[key] = MSE[key]**(0.5)
        return RMSE

    def monitor(self, 
                startTime=None,
                endTime=None,
                stepSize=None,
                show=False):
        
        self.simulator.simulate(self.model,
                                stepSize=stepSize,
                                startTime=startTime,
                                endTime=endTime)
        


        self.df_simulation_readings = self.simulator.get_simulation_readings()
        self.df_actual_readings = self.simulator.get_actual_readings(startTime, endTime, stepSize)
        if show:
            self.plot_performance()
            plt.show()


        # metrics

        # RMSE 
        # MSE
        # MAE
        # Relative error

