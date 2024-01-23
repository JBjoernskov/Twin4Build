from tqdm import tqdm
import datetime
import math
import numpy as np
import pandas as pd
import george
from george import kernels
from fmpy.fmi2 import FMICallException
import twin4build.saref4bldg.building_space.building_space as building_space
from twin4build.saref.device.sensor.sensor import Sensor
from twin4build.saref.device.meter.meter import Meter
from twin4build.logger.Logging import Logging
from twin4build.utils.plot import plot
import multiprocessing
import matplotlib.pyplot as plt

logger = Logging.get_logger("ai_logfile")

class Simulator():
    """
    The Simulator class simulates a model for a certain time period 
    using the <Simulator>.simulate(<Model>) method.
    """
    def __init__(self, 
                model=None):
        self.model = model
        logger.info("[Simulator Class] : Entered in Initialise Function")

    def do_component_timestep(self, component):
        #Gather all needed inputs for the component through all ingoing connections
        for connection_point in component.connectsAt:
            connection = connection_point.connectsSystemThrough
            connected_component = connection.connectsSystem
            if isinstance(component, building_space.BuildingSpace):
                assert np.isnan(connected_component.output[connection.senderPropertyName])==False, f"Model output {connection.senderPropertyName} of component {connected_component.id} is NaN."
            component.input[connection_point.receiverPropertyName] = connected_component.output[connection.senderPropertyName]
            if component.doUncertaintyAnalysis:
                component.inputUncertainty[connection_point.receiverPropertyName] = connected_component.outputUncertainty[connection.senderPropertyName]

        component.do_step(secondTime=self.secondTime, dateTime=self.dateTime, stepSize=self.stepSize)
    
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
            component.update_results()

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

    def get_simulation_timesteps(self, startTime, endTime, stepSize):
        n_timesteps = math.floor((endTime-startTime).total_seconds()/stepSize)
        self.secondTimeSteps = [i*stepSize for i in range(n_timesteps)]
        self.dateTimeSteps = [startTime+datetime.timedelta(seconds=i*stepSize) for i in range(n_timesteps)]
    
    def simulate(self, model, startTime, endTime, stepSize, trackGradients=False, targetParameters=None, targetMeasuringDevices=None, show_progress_bar=True):
        """
        Simulate the "model" between the dates "startTime" and "endTime" with timestep equal to "stepSize" in seconds.
        """
        assert targetParameters is not None and targetMeasuringDevices is not None if trackGradients else True, "Arguments targetParameters and targetMeasuringDevices must be set if trackGradients=True"
        self.model = model
        self.startTime = startTime
        self.endTime = endTime
        self.stepSize = stepSize
        self.trackGradients = trackGradients
        self.targetParameters = targetParameters
        self.targetMeasuringDevices = targetMeasuringDevices
        if trackGradients:
            assert isinstance(targetParameters, dict), "The argument targetParameters must be a dictionary"
            assert isinstance(targetMeasuringDevices, list), "The argument targetMeasuringDevices must be a list of Sensor and Meter objects"
            self.model.set_trackGradient(True)
            self.get_execution_order_reversed()
        self.model.initialize(startTime=startTime, endTime=endTime, stepSize=stepSize)
        self.get_simulation_timesteps(startTime, endTime, stepSize)
        logger.info("Running simulation")
        if show_progress_bar:
            for self.secondTime, self.dateTime in tqdm(zip(self.secondTimeSteps,self.dateTimeSteps), total=len(self.dateTimeSteps)):
                self.do_system_time_step(self.model)
        else:
            for self.secondTime, self.dateTime in zip(self.secondTimeSteps,self.dateTimeSteps):
                self.do_system_time_step(self.model)

    def get_simulation_readings(self):
        df_simulation_readings = pd.DataFrame()
        time = self.dateTimeSteps
        df_simulation_readings.insert(0, "time", time)
        df_simulation_readings = df_simulation_readings.set_index("time")
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
    
    def get_actual_readings(self, startTime, endTime, stepSize, reading_type="all"):
        allowed_reading_types = ["all", "input"]
        assert reading_type in allowed_reading_types, f"The \"walker_initialization\" argument must be one of the following: {', '.join(allowed_reading_types)} - \"{reading_type}\" was provided."
        print("Collecting actual readings...")
        """
        This is a temporary method for retrieving actual sensor readings.
        Currently it simply reads from csv files containing historic data.
        In the future, it should read from quantumLeap.  
        """
        self.get_simulation_timesteps(startTime, endTime, stepSize)
        logger.info("[Simulator Class] : Entered in Get Actual Readings Function")
        df_actual_readings = pd.DataFrame()
        time = self.dateTimeSteps
        df_actual_readings.insert(0, "time", time)
        df_actual_readings = df_actual_readings.set_index("time")
        sensor_instances = self.model.get_component_by_class(self.model.component_dict, Sensor)
        meter_instances = self.model.get_component_by_class(self.model.component_dict, Meter)
   
        for sensor in sensor_instances:
            sensor.set_is_physical_system()
            if reading_type=="all":
                actual_readings = sensor.get_physical_readings(startTime, endTime, stepSize)
                df_actual_readings.insert(0, sensor.id, actual_readings)
            
            elif reading_type=="input" and sensor.isPhysicalSystem:
                actual_readings = sensor.get_physical_readings(startTime, endTime, stepSize)
                df_actual_readings.insert(0, sensor.id, actual_readings)
                
            
        for meter in meter_instances:
            meter.set_is_physical_system()
            if reading_type=="all":
                actual_readings = meter.get_physical_readings(startTime, endTime, stepSize)
                df_actual_readings.insert(0, meter.id, actual_readings)
            
            elif reading_type=="input" and meter.isPhysicalSystem:
                actual_readings = meter.get_physical_readings(startTime, endTime, stepSize)
                df_actual_readings.insert(0, meter.id, actual_readings)

        logger.info("[Simulator Class] : Exited from Get Actual Readings Function")
        return df_actual_readings
    
    def _sim_func(self, model, parameter_set, component_list, attr_list):
        try:
            # Set parameters for the model
            self.model.set_parameters_from_array(parameter_set, component_list, attr_list)
            self.simulate(model,
                            stepSize=self.stepSize,
                            startTime=self.startTime,
                            endTime=self.endTime,
                            trackGradients=False,
                            targetParameters=self.targetParameters,
                            targetMeasuringDevices=self.targetMeasuringDevices,
                            show_progress_bar=False)
            standardDeviation = model.chain_log["standardDeviation"]
            y_model = np.zeros((len(self.dateTimeSteps), len(self.targetMeasuringDevices)))
            y = np.zeros((len(self.dateTimeSteps), len(self.targetMeasuringDevices)))
            for i, measuring_device in enumerate(self.targetMeasuringDevices):
                simulation_readings = np.array(next(iter(measuring_device.savedInput.values())))
                y_model[:,i] = simulation_readings
                y[:,i] = simulation_readings + np.random.normal(0, scale=self.targetMeasuringDevices[measuring_device]["standardDeviation"], size=simulation_readings.shape)

                # standardDeviation = np.array([el["standardDeviation"] for el in targetMeasuringDevices.values()])
            # y_w_obs_error = y# + 
        except FMICallException as inst:
            return None
        return (y, y_model)

    def _sim_func_gaussian_process(self, model, theta, stepSize, startTime, endTime, df_actual_readings_train, x_train, x):
        try:
            n_par = model.chain_log["n_par"]
            n_par_map = model.chain_log["n_par_map"]
            theta_kernel = np.exp(theta[-n_par:])
            theta = theta[:-n_par]

            # Set parameters for the model
            component_list = [model.component_dict[com_id] for com_id in model.chain_log["component_id"]]
            attr_list = model.chain_log["component_attr"]
            self.model.set_parameters_from_array(theta, component_list, attr_list)
            self.simulate(model,
                            stepSize=model.chain_log["stepSize_train"],
                            startTime=model.chain_log["startTime_train"],
                            endTime=model.chain_log["endTime_train"],
                            trackGradients=False,
                            targetParameters=self.targetParameters,
                            targetMeasuringDevices=self.targetMeasuringDevices,
                            show_progress_bar=False)
            simulation_readings_train = np.zeros((len(self.dateTimeSteps), len(self.targetMeasuringDevices)))
            actual_readings_train = np.zeros((len(self.dateTimeSteps), len(self.targetMeasuringDevices)))
            x_train = {}
            # t_train = self.secondTimeSteps
            # x_train = self.get_actual_readings(startTime=model.chain_log["startTime_train"], endTime=model.chain_log["endTime_train"], stepSize=model.chain_log["stepSize_train"], reading_type="input").to_numpy()
            for j, measuring_device in enumerate(self.targetMeasuringDevices):
                simulation_readings_train[:,j] = np.array(next(iter(measuring_device.savedInput.values())))
                actual_readings_train[:,j] = df_actual_readings_train[measuring_device.id].to_numpy()
                source_component = [cp.connectsSystemThrough.connectsSystem for cp in measuring_device.connectsAt][0]
                x = np.array(list(source_component.savedInput.values())).transpose()
                x_train[measuring_device.id] = x
            
            self.simulate(model,
                            stepSize=stepSize,
                            startTime=startTime,
                            endTime=endTime,
                            trackGradients=False,
                            targetParameters=self.targetParameters,
                            targetMeasuringDevices=self.targetMeasuringDevices,
                            show_progress_bar=False)
            # y_model = np.zeros((len(self.dateTimeSteps), len(self.targetMeasuringDevices)))
            # y = np.zeros((len(self.dateTimeSteps), len(self.targetMeasuringDevices)))
            # t = self.secondTimeSteps
            # x = self.get_actual_readings(startTime=startTime, endTime=endTime, stepSize=stepSize, reading_type="input").to_numpy()
            # ndim = n_x+1
            # standardDeviation = model.chain_log["standardDeviation"]
            # n_samples = 1000
            # for j, measuring_device in enumerate(self.targetMeasuringDevices):
            #     simulation_readings = np.array(next(iter(measuring_device.savedInput.values())))
            #     # y[:,j] = simulation_readings
            #     scale_lengths = theta_kernel[ndim*j:ndim*j+ndim]
            #     a = scale_lengths[0]
            #     scale_lengths = scale_lengths[1:]
            #     kernel = kernels.ExpSquaredKernel(metric=scale_lengths, ndim=scale_lengths.size)
            #     gp = george.GP(a*kernel)
            #     gp.compute(x_train, standardDeviation[j])
            #     y_model[:,j] = simulation_readings
            #     y[:,j] = np.mean(gp.sample_conditional(actual_readings_train[:,j]-simulation_readings_train[:,j], x, n_samples), axis=0) + simulation_readings


            
            n_samples = 100
            y_model = np.zeros((len(self.dateTimeSteps), len(self.targetMeasuringDevices)))
            y_noise = np.zeros((n_samples, len(self.dateTimeSteps), len(self.targetMeasuringDevices)))
            y = np.zeros((n_samples, len(self.dateTimeSteps), len(self.targetMeasuringDevices)))
            n_prev = 0
            for j, measuring_device in enumerate(self.targetMeasuringDevices):
                source_component = [cp.connectsSystemThrough.connectsSystem for cp in measuring_device.connectsAt][0]
                x = np.array(list(source_component.savedInput.values())).transpose()
                n = n_par_map[measuring_device.id]
                simulation_readings = np.array(next(iter(measuring_device.savedInput.values())))
                scale_lengths = theta_kernel[n_prev:n_prev+n]
                a = scale_lengths[0]
                scale_lengths = scale_lengths[1:]
                kernel = kernels.Matern52Kernel(metric=scale_lengths, ndim=scale_lengths.size)
                # kernel = kernels.ExpSquaredKernel(metric=scale_lengths, ndim=scale_lengths.size)
                gp = george.GP(a*kernel)
                gp.compute(x_train[measuring_device.id], self.targetMeasuringDevices[measuring_device]["standardDeviation"])
                y_noise[:,:,j] = gp.sample_conditional(actual_readings_train[:,j]-simulation_readings_train[:,j], x, n_samples)
                y_model[:,j] = simulation_readings
                y[:,:,j] = y_noise[:,:,j] + y_model[:,j]
                n_prev = n

        except FMICallException as inst:
            return None

        return (y, y_model, y_noise)
    
    def _sim_func_wrapped(self, args):
            return self._sim_func(*args)
    
    def _sim_func_wrapped_gaussian_process(self, args):
        return self._sim_func_gaussian_process(*args)
    
    def run_emcee_inference(self, model, parameter_chain, targetParameters, targetMeasuringDevices, startTime, endTime, stepSize, show=False, assume_uncorrelated_noise=True):
        self.model = model
        self.startTime = startTime
        self.endTime = endTime
        self.stepSize = stepSize
        self.targetParameters = targetParameters
        self.targetMeasuringDevices = targetMeasuringDevices
        n_samples_max = 100
        n_samples = parameter_chain.shape[0] if parameter_chain.shape[0]<n_samples_max else n_samples_max #100
        sample_indices = np.random.randint(parameter_chain.shape[0], size=n_samples)
        parameter_chain_sampled = parameter_chain[sample_indices]

        component_list = [obj for obj, attr_list in targetParameters.items() for i in range(len(attr_list))]
        attr_list = [attr for attr_list in targetParameters.values() for attr in attr_list]

        self.get_simulation_timesteps(startTime, endTime, stepSize)
        time = self.dateTimeSteps
        df_actual_readings_test = self.get_actual_readings(startTime=startTime, endTime=endTime, stepSize=stepSize)
        df_actual_readings_train = self.get_actual_readings(startTime=model.chain_log["startTime_train"], endTime=model.chain_log["endTime_train"], stepSize=model.chain_log["stepSize_train"])

        x = self.get_actual_readings(startTime=startTime, endTime=endTime, stepSize=stepSize, reading_type="input").to_numpy()
        x_train = self.get_actual_readings(startTime=model.chain_log["startTime_train"], endTime=model.chain_log["endTime_train"], stepSize=model.chain_log["stepSize_train"], reading_type="input").to_numpy()
        
        print("Running inference...")
        # pbar = tqdm(total=len(sample_indices))
        # y_list = [_sim_func(self, parameter_set) for parameter_set in parameter_chain_sampled]
        
        # unique_pred = np.unique(np.array([pred.data.tobytes() for pred in y_list]))
        # unique_par = np.unique(np.array([par.data.tobytes() for par in parameter_chain_sampled]))
 
        if assume_uncorrelated_noise==False:
            sim_func = self._sim_func_wrapped_gaussian_process
            args = [(model, parameter_set, stepSize, startTime, endTime, df_actual_readings_train, x_train, x) for parameter_set in parameter_chain_sampled]#########################################
        else:
            sim_func = self._sim_func_wrapped
            args = [(model, parameter_set, component_list, attr_list) for parameter_set in parameter_chain_sampled]

        n_cores = 4#multiprocessing.cpu_count()
        pool = multiprocessing.Pool(n_cores, maxtasksperchild=100) #maxtasksperchild is set because FMUs are leaking memory
        chunksize = 1#math.ceil(len(args)/n_cores)
        # self.model._set_addUncertainty(True)
        y_list = list(tqdm(pool.imap(sim_func, args, chunksize=chunksize), total=len(args)))
        
        # y_list = [self._sim_func_wrapped(arg) for arg in args]
        pool.close()
        # self.model._set_addUncertainty(False)
        y_list = [el for el in y_list if el is not None]


        predictions_noise = [[] for i in range(len(targetMeasuringDevices))]
        predictions_model = [[] for i in range(len(targetMeasuringDevices))]
        predictions = [[] for i in range(len(targetMeasuringDevices))]

        for y in y_list:
            # standardDeviation = np.array([el["standardDeviation"] for el in targetMeasuringDevices.values()])
            # y_w_obs_error = y# + np.random.normal(0, standardDeviation, size=y.shape)
            for col in range(len(targetMeasuringDevices)):
                if assume_uncorrelated_noise==False:
                    predictions_noise[col].append(y[2][:,:,col])
                    predictions[col].append(y[0][:,:,col])
                else:
                    predictions[col].append(y[0][:,col])

                predictions_model[col].append(y[1][:,col])
                
                
        intervals = []
        for col in range(len(targetMeasuringDevices)):
            pn = np.array(predictions_noise[col])
            om = np.array(predictions_model[col])
            p = np.array(predictions[col])
            pn = pn.reshape((pn.shape[0]*pn.shape[1], pn.shape[2])) if assume_uncorrelated_noise==False else pn
            p = p.reshape((p.shape[0]*p.shape[1], p.shape[2])) if assume_uncorrelated_noise==False else p

            intervals.append({"noise": pn,
                            "model": om,
                            "prediction": p})
        ydata = []
        for measuring_device, value in targetMeasuringDevices.items():
            ydata.append(df_actual_readings_test[measuring_device.id].to_numpy())


        ydata = np.array(ydata).transpose()
        fig, axes = plot.plot_emcee_inference(intervals, time, ydata, show=show)
        
        return fig, axes

    def run_ls_inference(self, model, ls_params, targetParameters, targetMeasuringDevices, startTime, endTime, stepSize, show=False):
        """
        Run model estimation using parameters from least squares optimization.

        :param model: The model to be simulated.
        :param ls_params: Parameters obtained from least squares optimization.
        :param targetParameters: Target parameters for the model.
        :param targetMeasuringDevices: Target measuring devices for collecting simulation output.
        :param startTime: Start time for the simulation.
        :param endTime: End time for the simulation.
        :param stepSize: Step size for the simulation.
        :param show: Flag to show plots if applicable.
        :return: Results of the simulation with the least squares parameters.
        """
        component_list = [obj for obj, attr_list in targetParameters.items() for i in range(len(attr_list))]
        attr_list = [attr for attr_list in targetParameters.values() for attr in attr_list]

        self.get_simulation_timesteps(startTime, endTime, stepSize)
        time = self.dateTimeSteps
        actual_readings = self.get_actual_readings(startTime=startTime, endTime=endTime, stepSize=stepSize)

        print("Running inference with least squares parameters...")
        
        try:
            # Set parameters for the model
            self.model.set_parameters_from_array(ls_params, component_list, attr_list)
            self.simulate(model, stepSize=stepSize, startTime=startTime, endTime=endTime,
                        trackGradients=False, targetParameters=targetParameters,
                        targetMeasuringDevices=targetMeasuringDevices, show_progress_bar=False)
            predictions = np.zeros((len(time), len(targetMeasuringDevices)))
            for i, measuring_device in enumerate(targetMeasuringDevices):
                simulation_readings = np.array(next(iter(measuring_device.savedInput.values())))
                predictions[:, i] = simulation_readings

        except FMICallException as inst:
            predictions = None
            print("Simulation failed:", inst)

        ydata = []
        for measuring_device, value in targetMeasuringDevices.items():
            ydata.append(actual_readings[measuring_device.id].to_numpy())
        ydata = np.array(ydata).transpose()

        if show and predictions is not None:
            fig, axes = plot.plot_ls_inference(predictions, time, ydata, targetMeasuringDevices)
            return fig, axes
        
        print("Simulation finished.")

        return predictions