from tqdm import tqdm
import datetime
import math
import numpy as np
import pandas as pd
from fmpy.fmi2 import FMICallException
import twin4build.saref4bldg.building_space.building_space as building_space
from twin4build.saref.device.sensor.sensor import Sensor
from twin4build.saref.device.meter.meter import Meter
from twin4build.logger.Logging import Logging
from twin4build.utils.plot import plot
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
    
    def get_actual_readings(self, startTime, endTime, stepSize):
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
        sensor_instances = self.model.get_component_by_class(self.model.component_dict, Sensor)
        meter_instances = self.model.get_component_by_class(self.model.component_dict, Meter)
                
        for sensor in sensor_instances:
            actual_readings = sensor.get_physical_readings(startTime, endTime, stepSize)
            df_actual_readings.insert(0, sensor.id, actual_readings)

        for meter in meter_instances:
            actual_readings = meter.get_physical_readings(startTime, endTime, stepSize)
            df_actual_readings.insert(0, meter.id, actual_readings)

        logger.info("[Simulator Class] : Exited from Get Actual Readings Function")
        return df_actual_readings
    
    def run_emcee_inference(self, model, parameter_chain, targetParameters, targetMeasuringDevices, startTime, endTime, stepSize, show=False):
        n_samples_max = 500
        n_samples = parameter_chain.shape[0] if parameter_chain.shape[0]<n_samples_max else n_samples_max #100
        sample_indices = np.random.randint(parameter_chain.shape[0], size=n_samples)
        parameter_chain_sampled = parameter_chain[sample_indices]

        component_list = [obj for obj, attr_list in targetParameters.items() for i in range(len(attr_list))]
        attr_list = [attr for attr_list in targetParameters.values() for attr in attr_list]

        self.get_simulation_timesteps(startTime, endTime, stepSize)
        time = self.dateTimeSteps
        actual_readings = self.get_actual_readings(startTime=startTime, endTime=endTime, stepSize=stepSize)

        # n_cores = 5#multiprocessing.cpu_count()
        # pool = multiprocessing.Pool(n_cores)

        print("Running inference...")
        pbar = tqdm(total=len(sample_indices))
        cached_predictions = {}
        def _sim_func(simulator, parameter_set):
            try:
                # Set parameters for the model
                hashed = parameter_set.data.tobytes()
                if hashed not in cached_predictions:
                    self.model.set_parameters_from_array(parameter_set, component_list, attr_list)
                    self.simulate(model,
                                    stepSize=stepSize,
                                    startTime=startTime,
                                    endTime=endTime,
                                    trackGradients=False,
                                    targetParameters=targetParameters,
                                    targetMeasuringDevices=targetMeasuringDevices,
                                    show_progress_bar=False)
                    y = np.zeros((len(time), len(targetMeasuringDevices)))
                    for i, measuring_device in enumerate(targetMeasuringDevices):
                        simulation_readings = np.array(next(iter(measuring_device.savedInput.values())))
                        y[:,i] = simulation_readings
                    cached_predictions[hashed] = y
                else:
                    y = cached_predictions[hashed]
                pbar.update(1)
            except FMICallException as inst:
                y = None
            return y
        y_list = [_sim_func(self, parameter_set) for parameter_set in parameter_chain_sampled]
        y_list = [el for el in y_list if el is not None]

        # r = list(tqdm.tqdm(p.imap(_foo, range(30)), total=30))

        predictions = [[] for i in range(len(targetMeasuringDevices))]
        predictions_w_obs_error = [[] for i in range(len(targetMeasuringDevices))]

        for y in y_list:
            standardDeviation = np.array([el["standardDeviation"] for el in targetMeasuringDevices.values()])
            y_w_obs_error = y + np.random.normal(0, standardDeviation, size=y.shape)
            for col in range(len(targetMeasuringDevices)):
                predictions[col].append(y[:,col])
                predictions_w_obs_error[col].append(y_w_obs_error[:,col])
        intervals = []
        for col in range(len(targetMeasuringDevices)):
            intervals.append({"credible": np.array(predictions[col]),
                            "prediction": np.array(predictions_w_obs_error[col])})
        ydata = []
        for measuring_device, value in targetMeasuringDevices.items():
            ydata.append(actual_readings[measuring_device.id].to_numpy())
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