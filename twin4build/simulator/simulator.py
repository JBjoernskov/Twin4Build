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
import twin4build.components as components
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
        Another promising option for performance optimization is to group all components with identical classes/models as well as priority and perform a vectorized "do_step" on all such models at once.
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
        assert startTime.tzinfo is not None, "The argument startTime must have a timezone"
        assert endTime.tzinfo is not None, "The argument endTime must have a timezone"
        assert isinstance(stepSize, int), "The argument stepSize must be an integer"
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
    
    def get_model_inputs(self, startTime, endTime, stepSize):
        """
        Should return all inputs for the model for estimation.
        At some point, a more general implementation should be made, .
        """
        self.get_simulation_timesteps(startTime, endTime, stepSize)
        logger.info("[Simulator Class] : Entered in Get Actual Readings Function")
        readings_dict = {}

        filter_classes = (components.SensorSystem,
                          components.MeterSystem,
                          components.ScheduleSystem,
                          components.OutdoorEnvironmentSystem)

        for component in self.model.component_dict.values():
            if isinstance(component, filter_classes) and len(component.connectsAt)==0:
                component.initialize(startTime, endTime, stepSize)
                readings_dict[component.id] = {}
                
                if isinstance(component, components.OutdoorEnvironmentSystem):
                    for column in component.df.columns:
                        readings_dict[component.id][column] = component.df[column].to_numpy()
                elif isinstance(component, components.ScheduleSystem) and component.useFile:
                    actual_readings = component.do_step_instance.df
                    key = next(iter(component.output.keys()))
                    readings_dict[component.id][key] = actual_readings.values
                elif isinstance(component, (components.SensorSystem, components.MeterSystem)):
                    actual_readings = component.do_step_instance.df
                    key = next(iter(component.output.keys()))
                    readings_dict[component.id][key] = actual_readings.values
        return readings_dict
    
    def get_actual_readings(self, startTime, endTime, stepSize, reading_type="all"):
        allowed_reading_types = ["all", "input"]
        assert reading_type in allowed_reading_types, f"The \"reading_type\" argument must be one of the following: {', '.join(allowed_reading_types)} - \"{reading_type}\" was provided."
        # print("Collecting actual readings...")
        """
        This is a temporary method for retrieving actual sensor readings.
        Currently it simply reads from csv files containing historic data.
        In the future, it should read from quantumLeap.  

        Todo:
        Expand to return ALL inputs for the model for estimation.
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
            sensor.initialize(startTime, endTime, stepSize)
            # sensor.set_is_physical_system()
            if sensor.physicalSystem is not None:
                if reading_type=="all":
                    actual_readings = sensor.get_physical_readings(startTime, endTime, stepSize)
                    df_actual_readings.insert(0, sensor.id, actual_readings)                
                elif reading_type=="input" and sensor.isPhysicalSystem:
                    actual_readings = sensor.get_physical_readings(startTime, endTime, stepSize)
                    df_actual_readings.insert(0, sensor.id, actual_readings)
                
        for meter in meter_instances:
            meter.initialize(startTime, endTime, stepSize)
            # meter.set_is_physical_system()
            if meter.physicalSystem is not None:
                if reading_type=="all":
                    actual_readings = meter.get_physical_readings(startTime, endTime, stepSize)
                    df_actual_readings.insert(0, meter.id, actual_readings)
                elif reading_type=="input" and meter.isPhysicalSystem:
                    actual_readings = meter.get_physical_readings(startTime, endTime, stepSize)
                    df_actual_readings.insert(0, meter.id, actual_readings)

        logger.info("[Simulator Class] : Exited from Get Actual Readings Function")
        return df_actual_readings
    
    def _get_gp_input_wrapped(self, a):
        args, kwargs = a
        return self.get_gp_input(*args, **kwargs)

    def get_gp_input(self, targetMeasuringDevices, startTime, endTime, stepSize, input_type="boundary", add_time=True, max_inputs=3, run_simulation=False, x0_=None, gp_input_map=None):
        allowed_input_types = ["closest", "boundary", "time"]
        assert input_type in allowed_input_types, f"The \"input_type\" argument must be one of the following: {', '.join(allowed_input_types)} - \"{input_type}\" was provided."
        self.gp_input = {measuring_device.id: [] for measuring_device in targetMeasuringDevices}
        self.gp_input_map = {measuring_device.id: [] for measuring_device in targetMeasuringDevices}
        
        if gp_input_map is not None:
            use_gp_input_map = True
        else:
            use_gp_input_map = False
        
        if input_type=="time":
            t = np.array(self.secondTimeSteps)
            for measuring_device in targetMeasuringDevices:
                self.gp_input[measuring_device.id] = t.reshape((t.shape[0], 1))
                self.gp_input_map[measuring_device.id].append("time")
        elif input_type=="boundary":
            input_readings = self.get_model_inputs(startTime=startTime, endTime=endTime, stepSize=stepSize)
            # print("---------------BEFORE -------------------")
            temp_gp_input = {measuring_device.id: [] for measuring_device in targetMeasuringDevices}
            temp_gp_input_map = {measuring_device.id: [] for measuring_device in targetMeasuringDevices}
            temp_depths = {measuring_device.id: [] for measuring_device in targetMeasuringDevices}
            temp_variance = {measuring_device.id: [] for measuring_device in targetMeasuringDevices}
            for measuring_device in targetMeasuringDevices:
                for c_id in input_readings:
                    component = self.model.component_dict[c_id]
                    shortest_path = self.model._shortest_path(component)
                    if measuring_device in shortest_path.keys():
                        shortest_path_length = shortest_path[measuring_device]
                        for output in input_readings[c_id]:
                            readings = input_readings[c_id][output]
                            if np.allclose(readings, readings[0])==False and np.isnan(readings).any()==False: #Not all equal values and no nans
                                temp_gp_input[measuring_device.id].append(readings)
                                temp_gp_input_map[measuring_device.id].append((c_id, output))
                                temp_depths[measuring_device.id].append(shortest_path_length)
                                r = (readings-np.min(readings))/(np.max(readings)-np.min(readings))
                                temp_variance[measuring_device.id].append(np.var(r)/np.mean(r))

            # print("-------------------INPUTS-------------------")
            for measuring_device in targetMeasuringDevices:
                if len(temp_gp_input[measuring_device.id])<=max_inputs:
                    self.gp_input[measuring_device.id] = temp_gp_input[measuring_device.id]
                    self.gp_input_map[measuring_device.id] = temp_gp_input_map[measuring_device.id]
                else:
                    depths = np.array(temp_depths[measuring_device.id])
                    var = np.array(temp_variance[measuring_device.id])
                    # Use closest inputs first. We assume that the closest inputs are the most relevant.
                    idx = np.argsort(depths)
                    # Use highest variance inputs first. We assume that high variance inputs carry more information.
                    # idx = np.argsort(var)[::-1]
                    # print("--------------measuring_device.id", measuring_device.id)
                    # for i in idx:
                        # print("depth: ", depths[i])
                        # print("obj: ", temp_gp_input_map[measuring_device.id][i])
                    for i in idx[:max_inputs]:
                        self.gp_input[measuring_device.id].append(temp_gp_input[measuring_device.id][i])
                        self.gp_input_map[measuring_device.id].append(temp_gp_input_map[measuring_device.id][i])
                # print(f"{measuring_device.id}: {self.gp_input_map[measuring_device.id]}")
            
            if add_time:
                t = np.array(self.secondTimeSteps)
                for measuring_device in targetMeasuringDevices:
                    x = np.array(self.gp_input[measuring_device.id]).transpose()
                    x = np.concatenate((x, t.reshape((t.shape[0], 1))), axis=1)
                    self.gp_input[measuring_device.id] = x
                    self.gp_input_map[measuring_device.id].append("time")
                    # self.gp_input[measuring_device.id] = (x-np.mean(x, axis=0))/np.std(x, axis=0)


        elif input_type=="closest":
            # import matplotlib.pyplot as plt
            if run_simulation:
                self._sim_func(self.model, x0_, [startTime], [endTime], [stepSize])

            temp_gp_input = {measuring_device.id: [] for measuring_device in targetMeasuringDevices}
            temp_gp_input_map = {measuring_device.id: [] for measuring_device in targetMeasuringDevices}
            temp_variance = {measuring_device.id: [] for measuring_device in targetMeasuringDevices}

            
            for measuring_device in targetMeasuringDevices:
                if use_gp_input_map:
                    for (c_id, input_) in gp_input_map[measuring_device.id]:
                        connected_component = self.model.component_dict[c_id]
                        readings = np.array(connected_component.savedOutput[input_])
                        temp_gp_input[measuring_device.id].append(readings)
                        temp_gp_input_map[measuring_device.id].append((c_id, input_))


                else:
                    input_readings = {}
                    source_component = [cp.connectsSystemThrough.connectsSystem for cp in measuring_device.connectsAt][0]
                    for connection_point in source_component.connectsAt:
                        connection = connection_point.connectsSystemThrough
                        connected_component = connection.connectsSystem
                        input_readings[(connected_component.id, connection.senderPropertyName)] = connected_component.savedOutput[connection.senderPropertyName]

                    # input_readings = source_component.savedInput

                    # c_id = source_component.id
                    if use_gp_input_map:
                        gp_input_list = gp_input_map[measuring_device.id]
                        d = [d_[1] for d_ in gp_input_list if isinstance(d_, tuple)]

                    all_constant = True
                    for (c_id, input_) in input_readings:
                        readings = np.array(input_readings[(c_id, input_)])
                        is_not_constant = np.any(readings==None)==False and np.allclose(readings, readings[0])==False and np.isnan(readings).any()==False
                        if is_not_constant:
                            all_constant = False
                            break
                        
                    # plt.figure()
                    # plt.title(measuring_device.id)
                    for (c_id, input_) in input_readings:
                        # plt.plot(input_readings[input_], label=input_)
                        readings = np.array(input_readings[(c_id, input_)])
                        is_not_constant = np.any(readings==None)==False and np.allclose(readings, readings[0])==False and np.isnan(readings).any()==False
                        if (all_constant or is_not_constant):
                            temp_gp_input[measuring_device.id].append(readings)
                            temp_gp_input_map[measuring_device.id].append((c_id, input_))
                            if all_constant:
                                break
                        # r = (readings-np.min(readings))/(np.max(readings)-np.min(readings))
                        # temp_variance[measuring_device.id].append(np.var(r)/np.mean(r))
                # plt.legend()
                # plt.show()
                assert len(temp_gp_input[measuring_device.id])>0, f"No input readings found for {measuring_device.id}"
            for measuring_device in targetMeasuringDevices:
                
                if len(temp_gp_input[measuring_device.id])<=max_inputs:
                    self.gp_input[measuring_device.id] = temp_gp_input[measuring_device.id]
                    self.gp_input_map[measuring_device.id] = temp_gp_input_map[measuring_device.id]
                else:
                    # var = np.array(temp_variance[measuring_device.id])
                    # idx = np.argsort(var)[::-1] # Use highest variance inputs first. We assume that high variance inputs carry more information.
                    # for i in idx[:max_inputs]:
                    for i in range(max_inputs):
                        self.gp_input[measuring_device.id].append(temp_gp_input[measuring_device.id][i])
                        self.gp_input_map[measuring_device.id].append(temp_gp_input_map[measuring_device.id][i])
            
            
            t = np.array(self.secondTimeSteps)
            for measuring_device in targetMeasuringDevices:
                # print(f"{measuring_device.id}: {self.gp_input_map[measuring_device.id]}")
                x = np.array(self.gp_input[measuring_device.id]).transpose()
                if add_time:
                    x = np.concatenate((x, t.reshape((t.shape[0], 1))), axis=1)
                    self.gp_input_map[measuring_device.id].append("time")
                self.gp_input[measuring_device.id] = x
        if use_gp_input_map:
            assert gp_input_map==self.gp_input_map, "gp_input_map does not match self.gp_input_map"
        return self.gp_input, self.gp_input_map


    def get_gp_variance(self, targetMeasuringDevices, theta, startTime, endTime, stepSize):
        df_actual_readings = pd.DataFrame()
        for startTime_, endTime_, stepSize_  in zip(startTime, endTime, stepSize):
            df_actual_readings_ = self.get_actual_readings(startTime=startTime_, endTime=endTime_, stepSize=stepSize_)
            df_actual_readings = pd.concat([df_actual_readings, df_actual_readings_])

        # This is a temporary solution. The fmu.freeInstance() method fails with a segmentation fault. 
        # The following ensures that we run the simulation in a separate process.
        args = [(self.model, theta, startTime, endTime, stepSize)]
        pool = multiprocessing.Pool(1)
        chunksize = 1
        self.model.make_pickable()
        y_list = list(pool.imap(self._sim_func_wrapped, args, chunksize=chunksize))
        pool.close()
        y_list = [el for el in y_list if el is not None]
        if len(y_list)>0:
            simulation_readings = y_list[0][1]
        else:
            raise(Exception("Simulation failed."))
        
        self.gp_variance = {}
        for j, (measuring_device, value) in enumerate(targetMeasuringDevices.items()):
            actual_readings = df_actual_readings[measuring_device.id].to_numpy()
            res = (actual_readings-simulation_readings[:,j])
            std = self.targetMeasuringDevices[measuring_device]["standardDeviation"]
            var = np.var(res)-std**2
            tol = 1e-6
            if var>0:
                self.gp_variance[measuring_device.id] = var
            elif np.var(res)>tol:
                self.gp_variance[measuring_device.id] = np.var(res)
            else:
                self.gp_variance[measuring_device.id] = tol
            
            # signal_to_noise = 5
            # self.gp_variance[measuring_device.id] = (self.targetMeasuringDevices[measuring_device]["standardDeviation"]*signal_to_noise)**2

            self.gp_variance[measuring_device.id] = (self.targetMeasuringDevices[measuring_device]["standardDeviation"]*signal_to_noise)**2





            # var = np.var(res)
            # tol = 1e-8
            # if var>tol:
            #     self.gp_variance[measuring_device.id] = var
            #     self.targetMeasuringDevices[measuring_device]["standardDeviation"] = var**0.5/signal_to_noise
            # else:
            #     self.gp_variance[measuring_device.id] = tol
            #     self.targetMeasuringDevices[measuring_device]["standardDeviation"] = tol**0.5/signal_to_noise

            # print(measuring_device.id, self.gp_variance[measuring_device.id])
            # print("var", var)
            # print("signal/noise: ", (self.gp_variance[measuring_device.id]/self.targetMeasuringDevices[measuring_device]["standardDeviation"]**2)**(0.5))
        return self.gp_variance
        
    def get_gp_lengthscale(self, targetMeasuringDevices, gp_input, lambda_=1):
        self.gp_lengthscale = {}
        for measuring_device, value in targetMeasuringDevices.items():
            x = gp_input[measuring_device.id]
            tol = 1e-8
            var = np.var(x, axis=0) #handle var=0
            idx = np.abs(var)>tol
            # assert np.any(idx==False), f"An input for {measuring_device.id} has less than 1e-8 variance. Something is likely wrong."
            var[idx==False] = tol
            self.gp_lengthscale[measuring_device.id] = np.sqrt(var)/lambda_
            # print(measuring_device.id, self.gp_lengthscale[measuring_device.id])
        return self.gp_lengthscale

    def _sim_func(self, model, theta, startTime, endTime, stepSize):
        try:
            # Set parameters for the model
            theta = theta[self.theta_mask]
            model.set_parameters_from_array(theta, self.flat_component_list, self.flat_attr_list)

            n_timesteps = 0
            for startTime_, endTime_, stepSize_  in zip(startTime, endTime, stepSize):
                self.get_simulation_timesteps(startTime_, endTime_, stepSize_)
                n_timesteps += len(self.secondTimeSteps)
            y_model = np.zeros((n_timesteps, len(self.targetMeasuringDevices)))
            n_time_prev = 0
            for startTime_, endTime_, stepSize_  in zip(startTime, endTime, stepSize):
                self.simulate(model,
                                stepSize=stepSize_,
                                startTime=startTime_,
                                endTime=endTime_,
                                trackGradients=False,
                                targetParameters=self.targetParameters,
                                targetMeasuringDevices=self.targetMeasuringDevices,
                                show_progress_bar=False)
                n_time = len(self.dateTimeSteps)
                for j, measuring_device in enumerate(self.targetMeasuringDevices):
                    simulation_readings = np.array(next(iter(measuring_device.savedInput.values())))
                    y_model[n_time_prev:n_time_prev+n_time,j] = simulation_readings
        except FMICallException as inst:
            return None
        return (None, y_model, None)

    def _sim_func_gaussian_process(self, model, theta, startTime, endTime, stepSize):
        try:
            n_par = model.chain_log["n_par"]
            n_par_map = model.chain_log["n_par_map"]
            theta_kernel = np.exp(theta[-n_par:])
            theta = theta[:-n_par]

            # Set parameters for the model
            theta = theta[self.theta_mask]
            self.model.set_parameters_from_array(theta, self.flat_component_list, self.flat_attr_list)
            simulation_readings_train = {measuring_device.id: [] for measuring_device in self.targetMeasuringDevices}
            actual_readings_train = {measuring_device.id: [] for measuring_device in self.targetMeasuringDevices}
            x_train = {measuring_device.id: [] for measuring_device in self.targetMeasuringDevices}
            oldest_date = min(model.chain_log["startTime_train"])
            for stepSize_train, startTime_train, endTime_train in zip(model.chain_log["stepSize_train"], model.chain_log["startTime_train"], model.chain_log["endTime_train"]):
                df_actual_readings_train = self.get_actual_readings(startTime=startTime_train, endTime=endTime_train, stepSize=stepSize_train)
                self.simulate(model,
                                stepSize=stepSize_train,
                                startTime=startTime_train,
                                endTime=endTime_train,
                                trackGradients=False,
                                targetParameters=self.targetParameters,
                                targetMeasuringDevices=self.targetMeasuringDevices,
                                show_progress_bar=False)
                self.get_gp_input(self.targetMeasuringDevices, startTime=startTime_train, endTime=endTime_train, stepSize=stepSize_train, input_type="boundary", add_time=True, max_inputs=4)
                # self.get_gp_input(self.targetMeasuringDevices, startTime=startTime_train, endTime=endTime_train, stepSize=stepSize_train, input_type="closest", add_time=False, max_inputs=7, gp_input_map=self.model.chain_log["gp_input_map"])
                for measuring_device in self.targetMeasuringDevices:
                    simulation_readings_train[measuring_device.id].append(np.array(next(iter(measuring_device.savedInput.values())))[self.n_initialization_steps:])#self.targetMeasuringDevices[measuring_device]["scale_factor"])
                    actual_readings_train[measuring_device.id].append(df_actual_readings_train[measuring_device.id].to_numpy()[self.n_initialization_steps:])#self.targetMeasuringDevices[measuring_device]["scale_factor"])
                    x = self.gp_input[measuring_device.id]
                    x_train[measuring_device.id].append(x[self.n_initialization_steps:])
                        
            for measuring_device in self.targetMeasuringDevices:
                simulation_readings_train[measuring_device.id] = np.concatenate(simulation_readings_train[measuring_device.id])#-model.chain_log["mean_train"][measuring_device.id])/model.chain_log["sigma_train"][measuring_device.id]
                actual_readings_train[measuring_device.id] = np.concatenate(actual_readings_train[measuring_device.id])#-model.chain_log["mean_train"][measuring_device.id])/model.chain_log["sigma_train"][measuring_device.id]
                x_train[measuring_device.id] = np.concatenate(x_train[measuring_device.id])
                
            n_timesteps = 0
            for startTime_, endTime_, stepSize_  in zip(startTime, endTime, stepSize):
                self.get_simulation_timesteps(startTime_, endTime_, stepSize_)
                n_timesteps += len(self.secondTimeSteps)
            n_samples = 200
            y_model = np.zeros((n_timesteps, len(self.targetMeasuringDevices)))
            y_noise = np.zeros((n_samples, n_timesteps, len(self.targetMeasuringDevices)))
            y = np.zeros((n_samples, n_timesteps, len(self.targetMeasuringDevices)))
            n_time_prev = 0
            for startTime_, endTime_, stepSize_  in zip(startTime, endTime, stepSize):
                self.simulate(model,
                                stepSize=stepSize_,
                                startTime=startTime_,
                                endTime=endTime_,
                                trackGradients=False,
                                targetParameters=self.targetParameters,
                                targetMeasuringDevices=self.targetMeasuringDevices,
                                show_progress_bar=False)
                self.get_gp_input(self.targetMeasuringDevices, startTime=startTime_, endTime=endTime_, stepSize=stepSize_, input_type="boundary", add_time=True, max_inputs=4)
                # self.get_gp_input(self.targetMeasuringDevices, startTime=startTime_, endTime=endTime_, stepSize=stepSize_, input_type="closest", add_time=False, max_inputs=7, gp_input_map=self.model.chain_log["gp_input_map"])
                n_time = len(self.dateTimeSteps)
                n_prev = 0
                for j, measuring_device in enumerate(self.targetMeasuringDevices):
                    x = self.gp_input[measuring_device.id]
                    n = n_par_map[measuring_device.id]
                    simulation_readings = np.array(next(iter(measuring_device.savedInput.values())))                    
                    scale_lengths = theta_kernel[n_prev:n_prev+n]
                    a = scale_lengths[0]
                    scale_lengths = scale_lengths[1:]
                    s = int(scale_lengths.size)
                    scale_lengths_base = scale_lengths[:s]
                    axes = list(range(s))
                    kernel1 = kernels.Matern32Kernel(metric=scale_lengths_base, ndim=s, axes=axes)
                    kernel = kernel1
                    res_train = (actual_readings_train[measuring_device.id]-simulation_readings_train[measuring_device.id])/self.targetMeasuringDevices[measuring_device]["scale_factor"]
                    std = self.targetMeasuringDevices[measuring_device]["standardDeviation"]/self.targetMeasuringDevices[measuring_device]["scale_factor"]
                    gp = george.GP(a*kernel)#, solver=george.HODLRSolver, tol=1e-8, min_size=500)
                    # print(x_train[measuring_device.id].shape)
                    # print(s)
                    # print(n)
                    # print(x.shape)
                    # print(res_train.shape)

                    # import matplotlib.pyplot as plt
                    # df_train = pd.DataFrame(x_train[measuring_device.id])
                    # df_test = pd.DataFrame(x)
                    # df_train.plot(subplots=True, legend=True, title="Train")
                    # df_test.plot(subplots=True, legend=True, title="Test")
                    # plt.show()



                    gp.compute(x_train[measuring_device.id], std)
                    y_noise[:,n_time_prev:n_time_prev+n_time,j] = gp.sample_conditional(res_train, x, n_samples)*self.targetMeasuringDevices[measuring_device]["scale_factor"]
                    y_model[n_time_prev:n_time_prev+n_time,j] = simulation_readings
                    y[:,n_time_prev:n_time_prev+n_time,j] = y_noise[:,n_time_prev:n_time_prev+n_time,j] + y_model[n_time_prev:n_time_prev+n_time,j]
                    n_prev += n
                n_time_prev += n_time

        except FMICallException as inst:
            return None
        except np.linalg.LinAlgError as inst:
            return None

        return (y, y_model, y_noise)
    
    def _sim_func_wrapped(self, args):
            return self._sim_func(*args)
    
    def _sim_func_wrapped_gaussian_process(self, args):
        return self._sim_func_gaussian_process(*args)
    
    def bayesian_inference(self, 
                           model, 
                           startTime, 
                           endTime, 
                           stepSize, 
                           targetMeasuringDevices=None, 
                           n_initialization_steps=0,
                           show_progress_bar=True,
                           assume_uncorrelated_noise=True, 
                           burnin=None,
                           n_samples_max=100,
                           n_cores=multiprocessing.cpu_count(),
                           seed=None):
        if seed is not None:
            assert isinstance(seed, int), "The seed must be an integer."
            np.random.seed(seed)
        self.model = model
        self.startTime = startTime
        self.endTime = endTime
        self.stepSize = stepSize
        self.targetParameters = None
        self.targetMeasuringDevices = targetMeasuringDevices
        self.n_initialization_steps = n_initialization_steps
        self.n_samples_max = n_samples_max

        targetMeasuringDevices_new = {}
        for k,v in targetMeasuringDevices.items():
            if isinstance(k, str):
                assert k in model.component_dict.keys(), f"Measuring device {k} not found in the model."
                targetMeasuringDevices_new[model.component_dict[k]] = v
            else:
                assert k in model.component_dict.values(), f"Measuring device object {k} not found in the model."
                targetMeasuringDevices_new[k] = v
        self.targetMeasuringDevices = targetMeasuringDevices_new
        s = model.chain_log["chain.x"].shape[0]
        assert burnin<=model.chain_log["chain.x"].shape[0], f"The burnin parameter ({str(burnin)}) must be less than the number of samples in the chain ({str(s)})."

        parameter_chain = model.chain_log["chain.x"][burnin:,0,:,:]
        parameter_chain = parameter_chain.reshape((parameter_chain.shape[0]*parameter_chain.shape[1], parameter_chain.shape[2]))

        n_samples = parameter_chain.shape[0] if parameter_chain.shape[0]<self.n_samples_max else self.n_samples_max #100
        sample_indices = np.random.randint(parameter_chain.shape[0], size=n_samples)
        parameter_chain_sampled = parameter_chain[sample_indices]

        self.flat_component_list = [model.component_dict[com_id] for com_id in model.chain_log["component_id"]]
        self.flat_attr_list = model.chain_log["component_attr"]
        self.theta_mask = model.chain_log["theta_mask"]

        print("Running inference...")
        
        # pbar = tqdm(total=len(sample_indices))
        # y_list = [_sim_func(self, parameter_set) for parameter_set in parameter_chain_sampled]
        
        # unique_pred = np.unique(np.array([pred.data.tobytes() for pred in y_list]))
        # unique_par = np.unique(np.array([par.data.tobytes() for par in parameter_chain_sampled]))
 
        if assume_uncorrelated_noise==False:
            print([type(stepSize_) for stepSize_ in stepSize])
            sim_func = self._sim_func_wrapped_gaussian_process
            args = [(model, parameter_set, startTime, endTime, stepSize) for parameter_set in parameter_chain_sampled]#########################################
        else:
            sim_func = self._sim_func_wrapped
            args = [(model, parameter_set, startTime, endTime, stepSize) for parameter_set in parameter_chain_sampled]

        del model.chain_log["chain.x"] ########################################

        #################################
        if n_cores>1:
            pool = multiprocessing.Pool(n_cores, maxtasksperchild=100) #maxtasksperchild is set because FMUs are leaking memory ##################################
            chunksize = 1#math.ceil(len(args)/n_cores)
            self.model.make_pickable()
            if show_progress_bar:
                y_list = list(tqdm(pool.imap(sim_func, args, chunksize=chunksize), total=len(args)))
            else:
                y_list = list(pool.imap(sim_func, args, chunksize=chunksize))
            pool.close() ###############################
        else:
            if show_progress_bar:
                y_list = [sim_func(arg) for arg in tqdm(args)]
            else:
                y_list = [sim_func(arg) for arg in args]
        ############################################
        
        
        # self.model._set_addUncertainty(False)
        y_list = [el for el in y_list if el is not None]

        #To allow access to simulated schedules and inputs, we need to simulate the model once in the main process as a quick fix
        sim_func(args[0])

        print("Number of failed simulations: ", n_samples-len(y_list))
        


        predictions_noise = [[] for i in range(len(self.targetMeasuringDevices))]
        predictions_model = [[] for i in range(len(self.targetMeasuringDevices))]
        predictions = [[] for i in range(len(self.targetMeasuringDevices))]

        for y in y_list:
            # standardDeviation = np.array([el["standardDeviation"] for el in targetMeasuringDevices.values()])
            # y_w_obs_error = y# + np.random.normal(0, standardDeviation, size=y.shape)
            for col, key in enumerate(self.targetMeasuringDevices):
                if assume_uncorrelated_noise==False:
                    predictions_noise[col].append(y[2][:,:,col])
                    predictions[col].append(y[0][:,:,col])

                predictions_model[col].append(y[1][:,col])
                
                
        result = {"values": [], "time": None, "y_data": None}
        for col, key in enumerate(self.targetMeasuringDevices):
            
            pn = np.array(predictions_noise[col])
            om = np.array(predictions_model[col])
            p = np.array(predictions[col])
            pn = pn.reshape((pn.shape[0]*pn.shape[1], pn.shape[2])) if assume_uncorrelated_noise==False else pn
            p = p.reshape((p.shape[0]*p.shape[1], p.shape[2])) if assume_uncorrelated_noise==False else p
            result["values"].append({"noise": pn,
                                "model": om,
                                "prediction": p,
                                "id": key.id})
            
        # self.get_simulation_timesteps(startTime, endTime, stepSize)
        df_actual_readings_test = pd.DataFrame()
        time = []
        for startTime_, endTime_, stepSize_  in zip(startTime, endTime, stepSize):
            self.get_simulation_timesteps(startTime_, endTime_, stepSize_)
            time.extend(self.dateTimeSteps)
            df_actual_readings_test_ = self.get_actual_readings(startTime=startTime_, endTime=endTime_, stepSize=stepSize_)
            # df_actual_readings_test = df_actual_readings_test.append(df_actual_readings_test)
            df_actual_readings_test = pd.concat([df_actual_readings_test, df_actual_readings_test_])
        
        ydata = []
        for measuring_device, value in self.targetMeasuringDevices.items():
            if measuring_device.id in df_actual_readings_test.columns:
                ydata.append(df_actual_readings_test[measuring_device.id].to_numpy())
            else:
                ydata.append(np.empty(len(time)))
        ydata = np.array(ydata).transpose()
        result["time"] = time
        result["ydata"] = ydata
        
        
        return result

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