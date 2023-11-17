import os
import sys
from twin4build.simulator.simulator import Simulator
from twin4build.saref.device.sensor.sensor import Sensor
from twin4build.saref.device.meter.meter import Meter
from twin4build.utils.data_loaders.load_from_file import load_from_file
from twin4build.utils.uppath import uppath
from twin4build.utils.plot.plot import get_fig_axes, load_params
from twin4build.utils.plot.plot import bar_plot_line_format
from twin4build.saref.property_.temperature.temperature import Temperature
from twin4build.saref.property_.Co2.Co2 import Co2
from twin4build.saref.property_.opening_position.opening_position import OpeningPosition #This is in use
from twin4build.saref.property_.energy.energy import Energy #This is in use
from twin4build.model.model import Model
from twin4build.saref4bldg.building_space.building_space import BuildingSpace
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
import matplotlib.dates as mdates
import seaborn as sns
import matplotlib.ticker as ticker
from twin4build.utils.plot.plot import get_fig_axes, load_params
from twin4build.logger.Logging import Logging
logger = Logging.get_logger("ai_logfile")

class Evaluator:
    """
    This Evaluator class evaluates and compares different scenarios.
    """
    def __init__(self):
        self.simulator = Simulator()
        
        logger.info("[Evaluator] : Entered in Initialise Function")

    def get_kpi(self, df_simulation_readings, measuring_device, evaluation_metric, property_):
        
        '''
            The get_kpi function calculates a Key Performance Indicator (KPI) based on simulation readings, 
            a measuring device, an evaluation metric, and a property to be evaluated. 
            If the property is a Temperature, it calculates the discomfort of occupants based on the
            difference between the temperature readings and the setpoint value over time. If the property is an Energy, 
            it calculates the energy consumption over time. The KPI is then returned.
        '''
        logger.info("[Evaluator] : Entered in get_kpi Function")

        if isinstance(property_, Temperature):
            assert isinstance(property_.isPropertyOf, BuildingSpace), f"Measuring device \"{measuring_device}\" does not belong to a space. Only Temperature sensors belonging to a space can be evaluated (currently)."
            assert property_.isControlledByDevice is not None, f"Property belonging to measuring device \"{measuring_device}\" is not controlled and does not have a setpoint. Only properties that are controlled can be evaluated (currently)."
            schedule_readings = property_.isControlledByDevice.savedInput["setpointValue"]
            filtered_df = pd.DataFrame()
            filtered_df.insert(0, "time", df_simulation_readings["time"])
            filtered_df.insert(1, "schedule_readings", schedule_readings)
            filtered_df.insert(2, measuring_device, df_simulation_readings[measuring_device])
            dt = filtered_df['time'].diff().apply(lambda x: x.total_seconds()/3600)
            filtered_df["discomfort"] = (filtered_df["schedule_readings"]-filtered_df[measuring_device])*dt
            filtered_df["discomfort"].clip(lower=0, inplace=True)

            filtered_df.set_index("time", inplace=True)

            filtered_df.loc[filtered_df.between_time('17:00', '8:00').index] = 0 #Set times outside to 0
            # filtered_df.loc[(filtered_df.index.weekday==5)|(filtered_df.index.weekday==6)] = 0 #Set times outside to 0

            filtered_df["discomfort"] = filtered_df["discomfort"].cumsum()


            if evaluation_metric=="T":
                filtered_df = filtered_df.tail(n=1).set_index(pd.Index(["Total"]))
            else:
                # filtered_df = filtered_df.set_index('time').resample(f'1{evaluation_metric}')
                filtered_df = filtered_df.resample(f'1{evaluation_metric}')
                filtered_df = filtered_df.last() - filtered_df.first()
            kpi = filtered_df["discomfort"]

        elif isinstance(property_, Energy):
            if evaluation_metric=="T":
                filtered_df = df_simulation_readings.set_index('time').tail(n=1).set_index(pd.Index(["Total"]))
            else:
                filtered_df = df_simulation_readings.set_index('time').resample(f'1{evaluation_metric}')
                filtered_df = filtered_df.last() - filtered_df.first()
            kpi = filtered_df[measuring_device]

        logger.info("[Evaluator] : Exited from get KPI Function")

        return kpi

    def evaluate(self, 
                startPeriod=None,
                endPeriod=None,
                stepSize=None,
                models=None,
                measuring_devices=None,
                evaluation_metrics=None):
        
        '''
            startPeriod: start time of the simulation
            endPeriod: end time of the simulation
            stepSize: time step size of the simulation
            models: a list of model instances to evaluate
            measuring_devices: a list of strings indicating the components in the models to be evaluated
            evaluation_metrics: a list of strings indicating the evaluation metrics to use (hourly, daily, weekly, monthly, annually, total).
        '''

        
        logger.info("[Evaluator] : Entered in Evaluate Function")

        legal_evaluation_metrics = ["H", "D", "W", "M", "A", "T"] #hourly, daily, weekly, monthly, annually, Total

        assert isinstance(models, list) and all([isinstance(model, Model) for model in models]), "Argument \"models\" must be a list of Model instances."
        # assert isinstance(measuring_devices, list) and all([isinstance(measuring_device, Sensor) or isinstance(measuring_device, Meter) for measuring_device in measuring_devices]), "Argument \"measuring_devices\" must be a list of Sensor or Meter instances."
        assert isinstance(measuring_devices, list) and all([isinstance(measuring_device, str) for measuring_device in measuring_devices]) and all([measuring_device in model.component_dict.keys() for (model, measuring_device) in zip(models, measuring_devices)]), f"Argument \"measuring_devices\" must be a list of strings with components that are included in all models."
        assert isinstance(evaluation_metrics, list) and all([isinstance(evaluation_metric, str) for evaluation_metric in evaluation_metrics]) and all([evaluation_metric in legal_evaluation_metrics for evaluation_metric in evaluation_metrics]), f"Argument \"evaluation_metrics\" must be a list of strings of either: {','.join(legal_evaluation_metrics)}."
        assert len(measuring_devices)==len(evaluation_metrics), "Length of measuring device must be equal to length of evaluation metrics."

        load_params()
        self.result_dict = {}
        self.bar_plot_dict = {}
        self.acc_plot_dict = {}

        kpi_dict = {measuring_device:pd.DataFrame() for measuring_device in measuring_devices}
        self.simulation_readings_dict = {measuring_device:pd.DataFrame() for measuring_device in measuring_devices}


        for model in models:
            self.simulator.simulate(model,
                                stepSize=stepSize,
                                startPeriod=startPeriod,
                                endPeriod=endPeriod)
            df_simulation_readings = self.simulator.get_simulation_readings()
            
            
            for measuring_device, evaluation_metric in zip(measuring_devices, evaluation_metrics):
                property_ = model.component_dict[measuring_device].measuresProperty
                kpi = self.get_kpi(df_simulation_readings, measuring_device, evaluation_metric, property_)

                kpi_dict[measuring_device].insert(0, model.id, kpi)
                if "time" not in kpi_dict[measuring_device]:
                    kpi_dict[measuring_device].insert(0, "time", kpi.index)
                

                self.simulation_readings_dict[measuring_device].insert(0, model.id, df_simulation_readings[measuring_device])
                # schedule_readings = property_.isControlledByDevice.savedInput["setpointValue"]
                # simulation_readings_dict[measuring_device].insert(0, model.id, df_simulation_readings[measuring_device])
                if "time" not in self.simulation_readings_dict[measuring_device]:
                    self.simulation_readings_dict[measuring_device].insert(0, "time", df_simulation_readings["time"])

        for measuring_device, evaluation_metric in zip(measuring_devices, evaluation_metrics):
            kpi_dict[measuring_device].set_index("time", inplace=True)
            fig, ax = plt.subplots()
            self.bar_plot_dict[measuring_device] = (fig,ax)
            fig.set_size_inches(7, 5)
            fig.suptitle(measuring_device, fontsize=18)
            kpi_dict[measuring_device].plot(kind="bar", ax=ax, rot=0).legend(fontsize=8)
            ax.set_xticklabels(map(bar_plot_line_format, kpi_dict[measuring_device].index, [evaluation_metric]*len(kpi_dict[measuring_device].index)))
            for container in ax.containers:
                labels = ["{:.2f}".format(v) if v/kpi_dict[measuring_device].max().max() > 0.01 else "" for v in container.datavalues]
                ax.bar_label(container, labels=labels)
            ax.set_xlabel(None)
            

            self.simulation_readings_dict[measuring_device].set_index("time", inplace=True)
            fig, ax = plt.subplots()
            self.acc_plot_dict[measuring_device] = (fig,ax)
            fig.set_size_inches(7, 5)
            fig.suptitle(measuring_device, fontsize=18)
            self.simulation_readings_dict[measuring_device].plot(ax=ax, rot=0).legend(fontsize=8)

            
        logger.info("[Evaluator] : Exited from Evaluate Function")  



