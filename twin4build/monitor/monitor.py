
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
import twin4build.utils.plot.plot as plot


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
            ylabel = r"Flow [kg/s]"
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

        if key=="total_fan_air_flow":
            property_ = Flow()
        else:
            property_ = self.model.component_dict[key].measuresProperty
        
        error_band_abs = 2
        error_band_relative = 15 #%
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
            error_band = 0.2
            err = self.get_error(key)
            legend_label = f"{error_band}% position error band"
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

        sensor_count = 0

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
            fig.text(0.020, 0.74, ylabel, va='center', ha='center', rotation='vertical', fontsize=13, color="black")
            # err = (df_actual_readings[key]-df_simulation_readings[key])/np.abs(df_actual_readings[key])*100
            pg, error_band, legend_label = self.get_performance_gap(id_)
            pg_moving_average = self.get_moving_average(pg)
            axes[1].plot(self.df_actual_readings.index, pg, color=blue, label="Residual")
            axes[1].plot(self.df_actual_readings.index, pg_moving_average, color=orange, label=f"Moving average")
            # axes[1].set_ylabel(ylabel=r"Perfomance gap [%]")s
            fig.text(0.020, 0.3, r"Perfomance gap [%]", va='center', ha='center', rotation='vertical', fontsize=13, color="black")

            
            axes[1].fill_between(self.df_actual_readings.index, y1=-error_band, y2=error_band, facecolor=facecolor, edgecolor=edgecolor, label=legend_label)
            # axes[1].set_ylim([-30,30])
            # axes[1].set_ylim([-3,3])

            # myFmt = mdates.DateFormatter('%a') #Weekday
            myFmt = mdates.DateFormatter('%H %d %b')
            axes[0].xaxis.set_major_formatter(myFmt)
            axes[1].xaxis.set_major_formatter(myFmt)

            axes[0].xaxis.set_tick_params(rotation=45)
            axes[1].xaxis.set_tick_params(rotation=45)
            if id_ == "Total_AirFlow_sensor":
                fig.suptitle("Total Air Flow", fontsize=18)
            else:
                fig.suptitle(sensor_count, fontsize=18)
                sensor_count += 1
            fig.set_size_inches(7, 5)
            axes[0].legend(fontsize=8)
            axes[1].legend(fontsize=8)

            fig.tight_layout()
            #Add a margin to the left of the figure
            fig.subplots_adjust(left=0.13)

        subset = self.df_actual_readings.columns
        # subset = ["Space temperature sensor", "Heat recovery temperature sensor", "Heating coil temperature sensor"]#BS2023


        fig,axes = plt.subplots(len(subset), sharex=True)
        fig.set_size_inches(7, 5)
        fig.suptitle("Anomaly signals", fontsize=18)
        self.plot_dict["monitor"] = (fig,axes)

        # Ensure axes is always iterable
        if len(subset) == 1:
            axes = [axes]  # Make it a list of one element

        sensor_count = 0
        for ax, key in zip(axes, subset): #iterate thorugh keys and skip first key which is "time"
            if key!="time" and key in subset:
                pg, error_band, legend_label = self.get_performance_gap(key)
                pg_moving_average = self.get_moving_average(pg)
                pg_moving_average["signal"] = pg_moving_average.abs()
                pg_moving_average["signal"] = pg_moving_average["signal"]>error_band
                if key == "Total_AirFlow_sensor":
                    ax.plot(self.df_actual_readings.index, pg_moving_average["signal"], color=blue, label=f"Signal: {key}")
                else:
                    ax.plot(self.df_actual_readings.index, pg_moving_average["signal"], color=blue, label=f"Signal: Room {sensor_count}")
                    sensor_count += 1
                ax.legend(prop={'size': 8}, loc="best")
                ax.set_ylim([-0.05,1.05])

                myFmt = mdates.DateFormatter('%a')
                ax.xaxis.set_major_formatter(myFmt)
                ax.xaxis.set_tick_params(rotation=45)
                ax.xaxis.set_major_locator(ticker.MultipleLocator(1))

    def get_moving_average(self, x):
        moving_average = x.rolling(144*2, min_periods=30).mean()
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
        


        #Get Damper_position_sensor_22_601b_00 data
        total_airflow_sensor = self.model.component_dict["Total_AirFlow_sensor"]
        # Loop through the component's savedOutput dictionary and make a new Series with the sum of the values
        
        # Assuming total_airflow_sensor.savedOutput[first_key] is a list
        first_key = next(iter(total_airflow_sensor.savedOutput))
        # Create a Series with an index based on the length of the list
        sum_series = pd.Series(0, index=range(len(total_airflow_sensor.savedOutput[first_key])))

        # Your existing loop for summing
        for key in total_airflow_sensor.savedOutput:
            # Assuming each item is compatible with being added to a Series; you might need further adjustments
            sum_series = sum_series.add(pd.Series(total_airflow_sensor.savedOutput[key], index=range(len(total_airflow_sensor.savedOutput[key]))), fill_value=0)

        sum_series = sum_series + 1.56 #Adding the constant value of air flow calculated from the error between sensor data and simulation.

        self.df_simulation_readings = self.simulator.get_simulation_readings()
        # Reindex sum_series to match the DataFrame's index
        #sum_series_aligned = sum_series.reindex(self.df_simulation_readings.index, fill_value=0)

        self.df_actual_readings = self.simulator.get_actual_readings(startTime, endTime, stepSize)


        #if the simulation readings are more than 22 kes
        if len(self.df_simulation_readings.columns) > 22:
            #Drop all columns with "co2" in the key from the simulation readings and actual readings
            self.df_simulation_readings = self.df_simulation_readings.loc[:,~self.df_simulation_readings.columns.str.contains("CO2")]
            self.df_actual_readings = self.df_actual_readings.loc[:,~self.df_actual_readings.columns.str.contains("CO2")]
            #Drop the Damper_position_sensor_22_601b_00 from the simulation and actual readings
            #self.df_simulation_readings = self.df_simulation_readings.drop(columns=["Damper_position_sensor_22_601b_00"])
            #self.df_actual_readings = self.df_actual_readings.drop(columns=["Damper_position_sensor_22_601b_00"])

        else:
        #Drop all columns but Total_AirFlow_Sensor and Damper_position_sensor_22_604_0 from the simulation readings
            self.df_simulation_readings = self.df_simulation_readings[["Total_AirFlow_sensor"]]
            self.df_actual_readings = self.df_actual_readings[["Total_AirFlow_sensor"]] 



        #transform the actual readings from feet3/min to m3/s and then to kg/s of air
        self.df_actual_readings["Total_AirFlow_sensor"] = self.df_actual_readings["Total_AirFlow_sensor"] * 0.00047194745 * 1.225

        self.df_simulation_readings["Total_AirFlow_sensor"] = sum_series.values


        if show:
            self.plot_performance()

            plot.plot_CO2_controller_rulebased(model=self.model, simulator=self.simulator, CO2_controller_id="CO2_controller_22_601b_0", show=False) #Set show=True to plot

            plt.show()
        
        # metrics

        # RMSE 
        # MSE
        # MAE
        # Relative error

