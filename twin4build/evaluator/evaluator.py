import os
import sys
import datetime
from twin4build.simulator.simulator import Simulator
from twin4build.saref.device.sensor.sensor import Sensor
from twin4build.saref.device.meter.meter import Meter
from twin4build.utils.data_loaders.load_spreadsheet import load_spreadsheet
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
from twin4build.utils.plot.plot import get_fig_axes, load_params, _convert_limits, Colors
from twin4build.utils.bayesian_inference import generate_quantiles

class Evaluator:
    """
    This Evaluator class evaluates and compares different scenarios.
    """
    def __init__(self):
        self.simulator = Simulator()

    def get_kpi(self, df_simulation_readings, measuring_device, evaluation_metric, property_):
        
        '''
            The get_kpi function calculates a Key Performance Indicator (KPI) based on simulation readings, 
            a measuring device, an evaluation metric, and a property to be evaluated. 
            If the property is a Temperature, it calculates the discomfort of occupants based on the
            difference between the temperature readings and the setpoint value over time. If the property is an Energy, 
            it calculates the energy consumption over time. The KPI is then returned.
        '''

        if isinstance(property_, Temperature):
            assert isinstance(property_.isPropertyOf, BuildingSpace), f"Measuring device \"{measuring_device}\" does not belong to a space. Only Temperature sensors belonging to a space can be evaluated (currently)."

            # assert property_.isControlledBy is not None, f"Property belonging to measuring device \"{measuring_device}\" is not controlled and does not have a setpoint. Only properties that are controlled can be evaluated (currently)."
            
            controller = property_.isObservedBy[0] #We assume that there is only one controller for each property or that they have the same setpoint schedule
            schedule = controller.hasProfile
            # modeled_components = self.simulator.model.instance_map[self.components[controller.id]]
            # base_controller = [v for v in modeled_components if isinstance(v, base.Controller)][0]
            modeled_schedule = self.simulator.model.instance_map_reversed[schedule]
            schedule_readings = modeled_schedule.savedOutput["scheduleValue"]
            filtered_df = pd.DataFrame()
            filtered_df.insert(0, "time", df_simulation_readings.index)
            filtered_df.insert(1, "schedule_readings", schedule_readings)
            filtered_df.set_index("time", inplace=True) #Important for inserting in next line
            filtered_df.insert(1, measuring_device, df_simulation_readings[measuring_device])
            dt = filtered_df.index.to_series().diff().apply(lambda x: x.total_seconds()/3600)
            filtered_df["discomfort"] = (filtered_df["schedule_readings"]-filtered_df[measuring_device])*dt
            filtered_df["discomfort"] = filtered_df.discomfort.mask(filtered_df["discomfort"]<0, 0)
            filtered_df["discomfort"].clip(lower=0, inplace=True)

            

            # filtered_df.loc[filtered_df.between_time('17:00', '8:00').index] = 0 #Set times outside to 0
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
                print(df_simulation_readings)
                filtered_df = df_simulation_readings.tail(n=1).set_index(pd.Index(["Total"]))
            else:
                filtered_df = df_simulation_readings.resample(f'1{evaluation_metric}')
                filtered_df = filtered_df.last() - filtered_df.first()
            kpi = filtered_df[measuring_device]

        return kpi

    def evaluate(self,
                startTime=None,
                endTime=None,
                stepSize=None,
                models=None,
                measuring_devices=None,
                evaluation_metrics=None,
                method="simulate",
                single_plot=False,
                include_measured=False,
                measuring_device_name_map=None,
                options=None,
                show=True):
        figsize = (15, 4)
        '''
            startTime: start time of the simulation
            endTime: end time of the simulation
            stepSize: time step size of the simulation
            models: a list of model instances to evaluate
            measuring_devices: a list of strings indicating the components in the models to be evaluated
            evaluation_metrics: a list of strings indicating the evaluation metrics to use (H: hourly, D: daily, W: weekly, M: monthly, A: annually, T: total).
        '''

        load_params()
        legal_evaluation_metrics = ["H", "D", "W", "M", "A", "T"] #hourly, daily, weekly, monthly, annually, Total

        assert isinstance(models, list) and all([isinstance(model, Model) for model in models]), "Argument \"models\" must be a list of Model instances."
        # assert isinstance(measuring_devices, list) and all([isinstance(measuring_device, Sensor) or isinstance(measuring_device, Meter) for measuring_device in measuring_devices]), "Argument \"measuring_devices\" must be a list of Sensor or Meter instances."
        assert isinstance(measuring_devices, list) and all([isinstance(measuring_device, str) for measuring_device in measuring_devices]) and all([measuring_device in model.components.keys() for (model, measuring_device) in zip(models, measuring_devices)]), f"Argument \"measuring_devices\" must be a list of strings with components that are included in all models."
        assert isinstance(evaluation_metrics, list) and all([isinstance(evaluation_metric, str) for evaluation_metric in evaluation_metrics]) and all([evaluation_metric in legal_evaluation_metrics for evaluation_metric in evaluation_metrics]), f"Argument \"evaluation_metrics\" must be a list of strings of either: {','.join(legal_evaluation_metrics)}."
        assert len(measuring_devices)==len(evaluation_metrics), "Length of measuring device must be equal to length of evaluation metrics."
        allowed_methods = ["simulate","bayesian_inference"]
        assert method in allowed_methods, f"The \"method\" argument must be one of the following: {', '.join(allowed_methods)} - \"{method}\" was provided."
        load_params()
        self.result_dict = {}
        self.bar_plot_dict = {}
        self.acc_plot_dict = {}

        if isinstance(startTime, datetime.datetime):
            startTime = [startTime]
        if isinstance(endTime, datetime.datetime):
            endTime = [endTime]
        if isinstance(stepSize, (int,float)):
            stepSize = [stepSize]


        if include_measured:
            actual_readings_dict = {}
            self.simulator = Simulator(models[0])
            for i, (startTime_, endTime_, stepSize_)  in enumerate(zip(startTime, endTime, stepSize)):
                actual_readings = self.simulator.get_actual_readings(startTime=startTime_, endTime=endTime_, stepSize=stepSize_)
                if i==0:
                    actual_readings_dict["time"] = np.array(self.simulator.dateTimeSteps)
                else:
                    actual_readings_dict["time"] = np.concatenate((self.actual_readings["time"], np.array(self.simulator.dateTimeSteps)), axis=0)
                for measuring_device in measuring_devices:
                    if i==0:
                        actual_readings_dict[measuring_device] = actual_readings[measuring_device].to_numpy()
                    else:
                        actual_readings_dict[measuring_device] = np.concatenate((self.actual_readings[measuring_device], actual_readings[measuring_device].to_numpy()), axis=0)
    
            actual_readings = pd.DataFrame.from_dict(actual_readings_dict)
            actual_readings.set_index("time", inplace=True)
        kpi_dict = {measuring_device:pd.DataFrame() for measuring_device in measuring_devices}
        self.simulation_readings_dict = {measuring_device:pd.DataFrame() for measuring_device in measuring_devices}

        if measuring_device_name_map is None:
            measuring_device_name_map = {measuring_device:measuring_device for measuring_device in measuring_devices}
        else:
            for measuring_device in measuring_devices:
                if measuring_device not in measuring_device_name_map:
                    measuring_device_name_map[measuring_device] = measuring_device

        if method=="simulate":
            for model in models:
                self.simulator.simulate(model,
                                    stepSize=stepSize,
                                    startTime=startTime,
                                    endTime=endTime)
                df_simulation_readings = self.simulator.get_simulation_readings()
                for measuring_device, evaluation_metric in zip(measuring_devices, evaluation_metrics):
                    property_ = model.components[measuring_device].observes
                    kpi = self.get_kpi(df_simulation_readings, measuring_device, evaluation_metric, property_)
                    kpi_dict[measuring_device].insert(0, model.id, kpi)
                    if "time" not in kpi_dict[measuring_device]:
                        kpi_dict[measuring_device].insert(0, "time", kpi.index)
                    

                    # self.simulation_readings_dict[measuring_device].insert(0, model.id, df_simulation_readings[measuring_device])
                    # schedule_readings = property_.isControlledBy.savedInput["setpointValue"]
                    # simulation_readings_dict[measuring_device].insert(0, model.id, df_simulation_readings[measuring_device])
                    # if "time" not in self.simulation_readings_dict[measuring_device]:
                        # self.simulation_readings_dict[measuring_device].insert(0, "time", df_simulation_readings.index)

            for measuring_device, evaluation_metric in zip(measuring_devices, evaluation_metrics):
                kpi_dict[measuring_device].set_index("time", inplace=True)
                fig, ax = plt.subplots()
                self.bar_plot_dict[measuring_device] = (fig,ax)
                fig.set_size_inches(figsize)
                fig.suptitle(measuring_device, fontsize=18)
                kpi_dict[measuring_device].plot(kind="bar", ax=ax, rot=0).legend(fontsize=8)
                ax.set_xticklabels(map(bar_plot_line_format, kpi_dict[measuring_device].index, [evaluation_metric]*len(kpi_dict[measuring_device].index)))
                for container in ax.containers:
                    labels = ["{:.2f}".format(v) if v/kpi_dict[measuring_device].max().max() > 0.01 else "" for v in container.datavalues]
                    ax.bar_label(container, labels=labels)
                ax.set_xlabel(None)
                

                # self.simulation_readings_dict[measuring_device].set_index("time", inplace=True)
                # fig, ax = plt.subplots()
                # self.acc_plot_dict[measuring_device] = (fig,ax)
                # fig.set_size_inches(figsize)
                # fig.suptitle(measuring_device, fontsize=18)
                # self.simulation_readings_dict[measuring_device].plot(ax=ax, rot=0).legend(fontsize=8)

        elif method=="bayesian_inference":
            if options is None:
                options = {}

            

            if "compare_with" in options:
                compare_with = options["compare_with"]
                del options["compare_with"]
            else:
                compare_with = "model"

            if "limit" in options:
                limit = options["limit"]
                del options["limit"]
            else:
                limit = 68 #1 sigma

            quantile = _convert_limits([limit])[0]
            err_dict = {measuring_device: [] for measuring_device in measuring_devices}

            

            for model in models:
                result = self.simulator.bayesian_inference(model,
                                                            stepSize=stepSize,
                                                            startTime=startTime,
                                                            endTime=endTime,
                                                            **options)
                




                for measuring_device, evaluation_metric in zip(measuring_devices, evaluation_metrics):
                    property_ = model.components[measuring_device].observes
                    simulation_readings = [d for d in result["values"] if d["id"]==measuring_device][0][compare_with]
                    print("----")
                    print("measuring_device", measuring_device)
                    print("simulation_readings", simulation_readings)
                    median_simulation_readings = generate_quantiles(simulation_readings, np.array([0.5]))
                    n_samples = simulation_readings.shape[0]
                    kpis = []
                    for i in range(n_samples):
                        df_simulation_readings = pd.DataFrame()
                        time = result["time"]
                        df_simulation_readings.insert(0, "time", time)
                        df_simulation_readings.insert(0, measuring_device, simulation_readings[i,:])
                        df_simulation_readings.set_index("time", inplace=True)
                        kpi = self.get_kpi(df_simulation_readings, measuring_device, evaluation_metric, property_) ####################
                        kpis.append(kpi)
                    kpis = np.array(kpis)
                    # kpis = kpis.reshape((len(kpis), 1))
                    median_kpi = generate_quantiles(kpis, np.array([0.5]))
                    q = generate_quantiles(kpis, np.array(quantile))
                    q[0] = median_kpi-q[0]
                    q[1] = q[1]-median_kpi
                    err_dict[measuring_device].append(q)
                    kpi_dict[measuring_device].insert(len(kpi_dict[measuring_device].columns), model.id, median_kpi[0,:])
                    if "time" not in kpi_dict[measuring_device]:
                        # print(kpi.index)
                        # print(kpi_dict[measuring_device])
                        kpi_dict[measuring_device].insert(0, "time", kpi.index)
                    self.simulation_readings_dict[measuring_device].insert(len(self.simulation_readings_dict[measuring_device].columns), model.id, median_simulation_readings[0,:])
                    # schedule_readings = property_.isControlledBy.savedInput["setpointValue"]
                    # simulation_readings_dict[measuring_device].insert(0, model.id, df_simulation_readings[measuring_device])
                    if "time" not in self.simulation_readings_dict[measuring_device]:
                        self.simulation_readings_dict[measuring_device].insert(0, "time", result["time"])

            if include_measured:
                for measuring_device, evaluation_metric in zip(measuring_devices, evaluation_metrics):
                    property_ = model.components[measuring_device].observes
                    kpi = self.get_kpi(actual_readings, measuring_device, evaluation_metric, property_) ####################
                    kpi_dict[measuring_device].insert(0, "Baseline measured", kpi.to_numpy())
                    l = err_dict[measuring_device][::-1]
                    l.append(np.array([[0],[0]]))
                    l = l[::-1]
                    err_dict[measuring_device] = l

            for measuring_device, evaluation_metric in zip(measuring_devices, evaluation_metrics):
                kpi_dict[measuring_device].set_index("time", inplace=True)
            if single_plot:
                for measuring_device, evaluation_metric in zip(measuring_devices, evaluation_metrics):
                    property_ = model.components[measuring_device].observes
                    fig, ax = plt.subplots()
                    self.bar_plot_dict[measuring_device] = (fig,ax)
                    fig.set_size_inches(figsize)
                    fig.suptitle(measuring_device_name_map[measuring_device], fontsize=18)
                    print(kpi_dict[measuring_device])
                    print(err_dict[measuring_device])
                    kpi_dict[measuring_device].plot(kind="bar", ax=ax, rot=0, yerr=err_dict[measuring_device]).legend(fontsize=12)
                    ax.set_xticklabels(map(bar_plot_line_format, kpi_dict[measuring_device].index, [evaluation_metric]*len(kpi_dict[measuring_device].index)))
                    containers = [container for container in ax.containers if hasattr(container, "datavalues")]
                    for container in containers:
                        labels = ["{:.2f}".format(v) if v/kpi_dict[measuring_device].max().max() > 0.01 else "" for v in container.datavalues]
                        ax.bar_label(container, labels=labels, fontsize=11)
                    ax.set_xlabel(None)

                    if isinstance(property_, Temperature):
                        ax.set_ylabel(r"$d$ [Kh]", color="black")
                    

                    self.simulation_readings_dict[measuring_device].set_index("time", inplace=True)
                    fig, ax = plt.subplots()
                    # self.acc_plot_dict[measuring_device] = (fig,ax)
                    
                    fig.set_size_inches(figsize)
                    fig.suptitle(measuring_device_name_map[measuring_device], fontsize=18)
                    # self.simulation_readings_dict[measuring_device].plot(ax=ax, rot=0, zorder=1)
                    for column in self.simulation_readings_dict[measuring_device]:
                        ax.plot(self.simulation_readings_dict[measuring_device].index, self.simulation_readings_dict[measuring_device][column], label=column, zorder=1)

                    ##############
                    
                    if isinstance(property_, Temperature):
                        controller = property_.isObservedBy[0] #We assume that there is only one controller for each property or that they have the same setpoint schedule
                        schedule = controller.hasProfile
                        # modeled_components = self.simulator.model.instance_map[self.components[controller.id]]
                        # base_controller = [v for v in modeled_components if isinstance(v, base.Controller)][0]
                        modeled_schedule = self.simulator.model.instance_map_reversed[schedule]
                        schedule_readings = modeled_schedule.savedOutput["scheduleValue"]
                        ylim = ax.get_ylim()
                        ax.fill_between(self.simulation_readings_dict[measuring_device].index, 0, schedule_readings, facecolor="black", edgecolor=Colors.red ,alpha=0.1, label=r"Heating setpoint", linewidth=3, zorder=2)
                        ax.set_ylim(ylim)
                        ax.set_ylabel(r"$T_z$ [$^\circ$C]", color="black")
                    ax.legend(fontsize=12)
                    mylocator = mdates.HourLocator(interval=8, tz=None)
                    ax.xaxis.set_minor_locator(mylocator)
                    myFmt = mdates.DateFormatter('%H')
                    ax.xaxis.set_minor_formatter(myFmt)

                    mylocator = mdates.WeekdayLocator(
                        byweekday=[mdates.MO, mdates.TU, mdates.WE, mdates.TH, mdates.FR, mdates.SA, mdates.SU], interval=1,
                        tz=None)
                    ax.xaxis.set_major_locator(mylocator)
                    myFmt = mdates.DateFormatter('%a')
                    ax.xaxis.set_major_formatter(myFmt)

                    ax.tick_params(axis='x', which='major', pad=10)  # move the tick labels
            else:
                for key in kpi_dict.keys():
                    kpi_dict[key]['Space'] = measuring_device_name_map[key]
                
                df = pd.concat(kpi_dict.values())
                # df = pd.DataFrame(kpi_dict)
                fig, ax = plt.subplots()
                fig.set_size_inches(figsize)
                # fig.suptitle(measuring_device, fontsize=18)
                df.plot(kind="bar", ax=ax, x="Space", rot=0).legend(fontsize=8)
                # ax.set_xticklabels(map(bar_plot_line_format, df.index, [evaluation_metric]*len(df.index)))
                # containers = [container for container in ax.containers if hasattr(container, "datavalues")]
                # for container in containers:
                #     labels = ["{:.2f}".format(v) if v/kpi_dict[measuring_device].max().max() > 0.01 else "" for v in container.datavalues]
                #     ax.bar_label(container, labels=labels)
                # ax.set_xlabel(None)
                
                for measuring_device, evaluation_metric in zip(measuring_devices, evaluation_metrics):
                    self.simulation_readings_dict[measuring_device].set_index("time", inplace=True)
                    fig, ax = plt.subplots()
                    # self.acc_plot_dict[measuring_device] = (fig,ax)
                    fig.set_size_inches(figsize)
                    fig.suptitle(measuring_device, fontsize=18)
                    self.simulation_readings_dict[measuring_device].plot(ax=ax, rot=0).legend(fontsize=8)

        if show:
            plt.show()    



