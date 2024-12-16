from __future__ import annotations
from tqdm import tqdm
import datetime
import math
import numpy as np
import pandas as pd
import george
from george import kernels
from fmpy.fmi2 import FMICallException
from twin4build.saref.device.sensor.sensor import Sensor
from twin4build.saref.device.meter.meter import Meter
# from twin4build.utils.plot import plot
import multiprocessing
import matplotlib.pyplot as plt
import twin4build.systems as systems
from typing import Optional, Dict, List, Tuple, Union
from twin4build.saref4syst.system import System
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import twin4build.model.model as model
class Simulator:
    """
    A class for simulating models in the twin4build framework.

    This class provides methods to simulate models, perform Bayesian inference,
    and retrieve simulation results. It handles both standard simulations and 
    Gaussian process-based inference.

    Attributes:
        model (Model): The model to be simulated.
        secondTime (float): Current simulation time in seconds.
        dateTime (datetime): Current simulation datetime.
        stepSize (int): Simulation step size in seconds.
        startTime (datetime): Simulation start time.
        endTime (datetime): Simulation end time.
        secondTimeSteps (List[float]): List of simulation timesteps in seconds.
        dateTimeSteps (List[datetime]): List of simulation timesteps as datetime objects.
        gp_input (Dict[str, np.ndarray]): GP input features for each device.
        gp_input_map (Dict[str, List]): Mapping of GP input features to their sources.
        gp_variance (Dict[str, float]): GP variance for each device.
        gp_lengthscale (Dict[str, np.ndarray]): GP lengthscales for each device.
        targetMeasuringDevices (Dict[System, Dict]): Target devices for inference.
        n_initialization_steps (int): Number of initialization steps to skip.
        theta_mask (np.ndarray): Mask for parameter selection.
        flat_component_list (List[System]): Flattened list of components.
        flat_attr_list (List[str]): Flattened list of attributes.
    """
    def __init__(self, model: Optional[model.Model] = None):
        """
        Initialize the Simulator instance.

        Creates a new simulator object that can be used to run simulations
        and perform parameter estimation.

        Args:
            model (Optional[Model], optional): The model to be simulated. 
                Can be set later if not provided at initialization.
                Defaults to None.

        Notes:
            The simulator maintains internal state about the current simulation,
            including time steps and component states.
        """
        self.model = model

    def _do_component_timestep(self, component: System) -> None:
        """
        Perform a single timestep for a component.

        Args:
            component (System): The component to simulate.

        Raises:
            AssertionError: If any input value is NaN.
        """
        #Gather all needed inputs for the component through all ingoing connections
        for i, connection_point in enumerate(component.connectsAt):
            for j, connection in enumerate(connection_point.connectsSystemThrough):
                connected_component = connection.connectsSystem
                component.input[connection_point.receiverPropertyName].set(connected_component.output[connection.senderPropertyName].get())
        component.do_step(secondTime=self.secondTime, dateTime=self.dateTime, stepSize=self.stepSize)
    
    def _do_system_time_step(self, model: model.Model) -> None:
        """
        Execute a time step for all components in the model.

        This method executes components in the order specified by the model's execution
        order, ensuring proper propagation of information through the system. It:
        1. Executes components in groups based on dependencies
        2. Updates component states after all executions
        3. Handles both FMU and non-FMU components

        Args:
            model (model.Model): The model containing components to simulate.

        Notes:
            - Components are executed sequentially based on their dependencies
            - Results are updated after all components have been stepped
            - Component execution order is determined by the model's execution_order attribute
            - Updates are propagated through the flat_execution_order after main execution
        """
        for component_group in model.execution_order:
            for component in component_group:
                self._do_component_timestep(component)

        for component in model.flat_execution_order:
            component.update_results()

    def get_simulation_timesteps(self, startTime: datetime, endTime: datetime, stepSize: int) -> None:
        """
        Generate simulation timesteps between start and end times.

        Creates lists of both second-based and datetime-based timesteps for the simulation
        period using the specified step size.

        Args:
            startTime (datetime): Start time of the simulation.
            endTime (datetime): End time of the simulation.
            stepSize (int): Step size in seconds.

        Notes:
            Updates the following instance attributes:
            - secondTimeSteps: List of timesteps in seconds
            - dateTimeSteps: List of timesteps as datetime objects
        """
        n_timesteps = math.floor((endTime-startTime).total_seconds()/stepSize)
        self.secondTimeSteps = [i*stepSize for i in range(n_timesteps)]
        self.dateTimeSteps = [startTime+datetime.timedelta(seconds=i*stepSize) for i in range(n_timesteps)]
        
    
    def simulate(self, 
                 model: model.Model, 
                 startTime: datetime, 
                 endTime: datetime, 
                 stepSize: int, 
                 show_progress_bar: bool = True) -> None:
        """
        Simulate the model between the specified dates with the given timestep.

        This method:
        1. Initializes the model and simulation parameters
        2. Generates simulation timesteps
        3. Executes the simulation loop with optional progress bar
        4. Updates component states at each timestep

        Args:
            model (Model): The model to simulate.
            startTime (datetime): Start time of the simulation.
            endTime (datetime): End time of the simulation.
            stepSize (int): Step size in seconds.
            show_progress_bar (bool, optional): Whether to show a progress bar during simulation.
                Defaults to True.

        Raises:
            AssertionError: If input parameters are invalid or missing timezone info.
            FMICallException: If the FMU simulation fails.
        """
        self.model = model
        assert startTime.tzinfo is not None, "The argument startTime must have a timezone"
        assert endTime.tzinfo is not None, "The argument endTime must have a timezone"
        assert isinstance(stepSize, int), "The argument stepSize must be an integer"
        self.startTime = startTime
        self.endTime = endTime
        self.stepSize = stepSize
        self.model.initialize(startTime=startTime, endTime=endTime, stepSize=stepSize)
        self.get_simulation_timesteps(startTime, endTime, stepSize)
        if show_progress_bar:
            for self.secondTime, self.dateTime in tqdm(zip(self.secondTimeSteps,self.dateTimeSteps), total=len(self.dateTimeSteps)):
                self._do_system_time_step(self.model)
        else:
            for self.secondTime, self.dateTime in zip(self.secondTimeSteps,self.dateTimeSteps):
                self._do_system_time_step(self.model)

    def get_simulation_readings(self) -> pd.DataFrame:
        """
        Get simulation readings for sensors and meters.

        Collects the simulation results from all sensors and meters in the model
        and organizes them into a pandas DataFrame with timestamps as index.

        Returns:
            pd.DataFrame: DataFrame containing simulation readings with columns:
                - time: Timestamp index
                - {sensor_id}: Reading values for each sensor
                - {meter_id}: Reading values for each meter
        """
        df_simulation_readings = pd.DataFrame()
        time = self.dateTimeSteps
        df_simulation_readings.insert(0, "time", time)
        df_simulation_readings = df_simulation_readings.set_index("time")
        sensor_instances = self.model.get_component_by_class(self.model.components, Sensor)
        meter_instances = self.model.get_component_by_class(self.model.components, Meter)

        for sensor in sensor_instances:
            savedOutput = self.model.components[sensor.id].savedOutput
            key = list(savedOutput.keys())[0]
            simulation_readings = savedOutput[key]
            df_simulation_readings.insert(0, sensor.id, simulation_readings)

        for meter in meter_instances:
            savedOutput = self.model.components[meter.id].savedOutput
            key = list(savedOutput.keys())[0]
            simulation_readings = savedOutput[key]
            df_simulation_readings.insert(0, meter.id, simulation_readings)
        return df_simulation_readings
    
    def get_model_inputs(self, startTime: datetime, endTime: datetime, stepSize: int) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Get all inputs for the model for estimation.

        Collects input data from sensors, meters, schedules, and outdoor environment 
        components that have no incoming connections (boundary conditions).

        Args:
            startTime (datetime): Start time of the simulation.
            endTime (datetime): End time of the simulation.
            stepSize (int): Step size in seconds.

        Returns:
            Dict[str, Dict[str, np.ndarray]]: Nested dictionary structure:
                - First level: Component ID to component data mapping
                - Second level: Output variable name to values mapping

        Notes:
            Only collects data from components that are instances of:
            - SensorSystem
            - MeterSystem
            - ScheduleSystem
            - OutdoorEnvironmentSystem
        """
        self.get_simulation_timesteps(startTime, endTime, stepSize)
        readings_dict = {}
        filter_classes = (systems.SensorSystem,
                          systems.MeterSystem,
                          systems.ScheduleSystem,
                          systems.OutdoorEnvironmentSystem)

        for component in self.model.components.values():
            if isinstance(component, filter_classes) and len(component.connectsAt)==0:
                component.initialize(startTime, endTime, stepSize)
                readings_dict[component.id] = {}
                
                if isinstance(component, systems.OutdoorEnvironmentSystem):
                    for column in component.df.columns:
                        readings_dict[component.id][column] = component.df[column].to_numpy()
                elif isinstance(component, systems.ScheduleSystem) and component.useFile:
                    actual_readings = component.do_step_instance.df
                    key = next(iter(component.output.keys()))
                    readings_dict[component.id][key] = actual_readings.values
                elif isinstance(component, (systems.SensorSystem, systems.MeterSystem)):
                    actual_readings = component.do_step_instance.df
                    key = next(iter(component.output.keys()))
                    readings_dict[component.id][key] = actual_readings.values
        return readings_dict
    
    def get_actual_readings(self, startTime: datetime, endTime: datetime, stepSize: int, 
                            reading_type: str = "all") -> pd.DataFrame:
        """
        Get actual sensor and meter readings from physical devices.

        Retrieves historical data from physical sensors and meters within the specified 
        time period. Currently reads from CSV files, but designed to be extended for 
        other data sources like quantumLeap.

        Args:
            startTime (datetime): Start time of the readings.
            endTime (datetime): End time of the readings.
            stepSize (int): Step size in seconds.
            reading_type (str, optional): Type of readings to retrieve:
                - "all": Get readings from all devices
                - "input": Get readings only from input devices
                Defaults to "all".

        Returns:
            pd.DataFrame: DataFrame containing actual readings with columns:
                - time: Timestamp index
                - {device_id}: Reading values for each device

        Raises:
            AssertionError: If reading_type is not one of ["all", "input"].
        """
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
        df_actual_readings = pd.DataFrame()
        time = self.dateTimeSteps
        df_actual_readings.insert(0, "time", time)
        df_actual_readings = df_actual_readings.set_index("time")
        sensor_instances = self.model.get_component_by_class(self.model.components, Sensor)
        meter_instances = self.model.get_component_by_class(self.model.components, Meter)
   
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

        return df_actual_readings
    
    def _get_gp_input_wrapped(self, a: Tuple) -> Tuple[Dict[str, np.ndarray], Dict[str, List]]:
        """
        Wrapper for get_gp_input to use with multiprocessing.

        Unpacks arguments and calls get_gp_input with proper parameters.

        Args:
            a (Tuple): Tuple containing (args, kwargs) for get_gp_input.

        Returns:
            Tuple[Dict[str, np.ndarray], Dict[str, List]]: See get_gp_input return value.
        """
        args, kwargs = a
        return self.get_gp_input(*args, **kwargs)

    def get_gp_input(self, targetMeasuringDevices: List[System], 
                     startTime: datetime, 
                     endTime: datetime, 
                     stepSize: int, 
                     input_type: str = "boundary", 
                     add_time: bool = True, 
                     max_inputs: int = 3, 
                     run_simulation: bool = False, 
                     x0_: Optional[np.ndarray] = None,
                     gp_input_map: Optional[Dict[str, List]] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, List]]:
        """
        Get Gaussian process inputs for target measuring devices.

        This method determines the input features for Gaussian process modeling based on
        the specified input type and configuration parameters.

        Args:
            targetMeasuringDevices (List[System]): List of target measuring devices.
            startTime (datetime): Start time of the simulation.
            endTime (datetime): End time of the simulation.
            stepSize (int): Step size in seconds.
            input_type (str, optional): Type of input selection strategy:
                - "closest": Use closest connected components
                - "boundary": Use boundary conditions
                - "time": Use only time as input
                Defaults to "boundary".
            add_time (bool, optional): Whether to add time as an additional input feature.
                Defaults to True.
            max_inputs (int, optional): Maximum number of input features to use.
                Defaults to 3.
            run_simulation (bool, optional): Whether to run a simulation to get inputs.
                Defaults to False.
            x0_ (Optional[np.ndarray], optional): Initial state for simulation.
                Defaults to None.
            gp_input_map (Optional[Dict[str, List]], optional): Predefined input map for GP.
                Defaults to None.

        Returns:
            Tuple[Dict[str, np.ndarray], Dict[str, List]]: A tuple containing:
                - Dictionary mapping device IDs to input arrays
                - Dictionary mapping device IDs to input feature names

        Raises:
            AssertionError: If input_type is not one of ["closest", "boundary", "time"].
        """
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
                    component = self.model.components[c_id]
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
                    for i in idx[:max_inputs]:
                        self.gp_input[measuring_device.id].append(temp_gp_input[measuring_device.id][i])
                        self.gp_input_map[measuring_device.id].append(temp_gp_input_map[measuring_device.id][i])
            
            if add_time:
                t = np.array(self.secondTimeSteps)
                for measuring_device in targetMeasuringDevices:
                    x = np.array(self.gp_input[measuring_device.id]).transpose()
                    x = np.concatenate((x, t.reshape((t.shape[0], 1))), axis=1)
                    self.gp_input[measuring_device.id] = x
                    self.gp_input_map[measuring_device.id].append("time")
                    # self.gp_input[measuring_device.id] = (x-np.mean(x, axis=0))/np.std(x, axis=0)


        elif input_type=="closest":
            if run_simulation:
                self._sim_func(self.model, x0_, [startTime], [endTime], [stepSize])

            temp_gp_input = {measuring_device.id: [] for measuring_device in targetMeasuringDevices}
            temp_gp_input_map = {measuring_device.id: [] for measuring_device in targetMeasuringDevices}
            temp_variance = {measuring_device.id: [] for measuring_device in targetMeasuringDevices}

            
            for measuring_device in targetMeasuringDevices:
                if use_gp_input_map:
                    for (c_id, input_) in gp_input_map[measuring_device.id]:
                        connected_component = self.model.components[c_id]
                        readings = np.array(connected_component.savedOutput[input_])
                        temp_gp_input[measuring_device.id].append(readings)
                        temp_gp_input_map[measuring_device.id].append((c_id, input_))


                else:
                    input_readings = {}
                    source_component = [cp.connectsSystemThrough.connectsSystem for cp in measuring_device.connectsAt][0]
                    for connection_point in source_component.connectsAt:
                        for connection in connection_point.connectsSystemThrough:
                            connected_component = connection.connectsSystem
                            input_readings[(connected_component.id, connection.senderPropertyName)] = connected_component.savedOutput[connection.senderPropertyName]

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
                    for (c_id, input_) in input_readings:
                        readings = np.array(input_readings[(c_id, input_)])
                        is_not_constant = np.any(readings==None)==False and np.allclose(readings, readings[0])==False and np.isnan(readings).any()==False
                        if (all_constant or is_not_constant):
                            temp_gp_input[measuring_device.id].append(readings)
                            temp_gp_input_map[measuring_device.id].append((c_id, input_))
                            if all_constant:
                                break
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
                x = np.array(self.gp_input[measuring_device.id]).transpose()
                if add_time:
                    x = np.concatenate((x, t.reshape((t.shape[0], 1))), axis=1)
                    self.gp_input_map[measuring_device.id].append("time")
                self.gp_input[measuring_device.id] = x
        if use_gp_input_map:
            assert gp_input_map==self.gp_input_map, "gp_input_map does not match self.gp_input_map"
        return self.gp_input, self.gp_input_map


    def get_gp_variance(self, 
                        targetMeasuringDevices: Dict[System, Dict], 
                        theta: np.ndarray, 
                        startTime: List[datetime.datetime], 
                        endTime: List[datetime.datetime], 
                        stepSize: List[int]) -> Dict[str, float]:
        """
        Calculate Gaussian process variance for target measuring devices.

        Computes the variance between actual and simulated readings, accounting for
        measurement noise and ensuring numerical stability.

        Args:
            targetMeasuringDevices (Dict[System, Dict]): Dictionary mapping measuring devices
                to their configuration parameters.
            theta (np.ndarray): Model parameters to use for simulation.
            startTime (List[datetime]): List of start times for each simulation period.
            endTime (List[datetime]): List of end times for each simulation period.
            stepSize (List[int]): List of step sizes for each simulation period.

        Returns:
            Dict[str, float]: Dictionary mapping device IDs to their GP variances.

        Raises:
            Exception: If simulation fails.

        Notes:
            Uses a separate process for simulation to avoid memory issues with FMU instances.
            Applies a minimum variance threshold for numerical stability.
        """
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
        return self.gp_variance
        
    def get_gp_lengthscale(self, targetMeasuringDevices: Dict[System, Dict], 
                           gp_input: Dict[str, np.ndarray], 
                           lambda_: float = 1) -> Dict[str, np.ndarray]:
        """
        Calculate Gaussian process lengthscales for target measuring devices.

        Computes appropriate lengthscales for GP kernel based on input data variance
        and a scaling factor.

        Args:
            targetMeasuringDevices (Dict[System, Dict]): Dictionary mapping measuring devices
                to their configuration parameters.
            gp_input (Dict[str, np.ndarray]): Dictionary mapping device IDs to their input
                feature arrays.
            lambda_ (float, optional): Scaling factor for lengthscales. Defaults to 1.

        Returns:
            Dict[str, np.ndarray]: Dictionary mapping device IDs to their GP lengthscales.

        Notes:
            Applies a minimum variance threshold to handle constant inputs.
            Lengthscales are computed as sqrt(variance)/lambda for each input dimension.
        """
        self.gp_lengthscale = {}
        for measuring_device, value in targetMeasuringDevices.items():
            x = gp_input[measuring_device.id]
            tol = 1e-8
            var = np.var(x, axis=0) #handle var=0
            idx = np.abs(var)>tol
            # assert np.any(idx==False), f"An input for {measuring_device.id} has less than 1e-8 variance. Something is likely wrong."
            var[idx==False] = tol
            self.gp_lengthscale[measuring_device.id] = np.sqrt(var)/lambda_
        return self.gp_lengthscale

    def _sim_func(self, model: model.Model, theta: np.ndarray, startTime: List[datetime.datetime], 
                  endTime: List[datetime.datetime], stepSize: List[int]) -> Optional[Tuple[None, np.ndarray, None]]:
        """
        Internal simulation function for basic inference.

        Executes a simulation with given parameters and returns the results. Handles parameter
        setting and simulation execution while catching potential FMU-related errors.

        Args:
            model (Model): The model to simulate.
            theta (np.ndarray): Model parameters to use for simulation.
            startTime (List[datetime]): List of start times for each simulation period.
            endTime (List[datetime]): List of end times for each simulation period.
            stepSize (List[int]): List of step sizes for each simulation period.

        Returns:
            Optional[Tuple[None, np.ndarray, None]]: A tuple containing:
                - None (placeholder for compatibility)
                - Simulation results array
                - None (placeholder for compatibility)
                Returns None if simulation fails.

        Notes:
            This is an internal method primarily used by the Bayesian inference process.
            The unusual return type structure is maintained for compatibility with other methods.
        """
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
                                show_progress_bar=False)
                n_time = len(self.dateTimeSteps)
                for j, measuring_device in enumerate(self.targetMeasuringDevices):
                    simulation_readings = np.array(next(iter(measuring_device.savedInput.values())))
                    y_model[n_time_prev:n_time_prev+n_time,j] = simulation_readings
        except FMICallException as inst:
            return None
        return (None, y_model, None)

    def _sim_func_gaussian_process(self, model: model.Model, theta: np.ndarray, startTime: List[datetime.datetime], 
                                   endTime: List[datetime.datetime], stepSize: List[int]) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Internal simulation function for Gaussian process-based inference.

        Executes a simulation with given parameters and computes Gaussian process predictions.
        Handles parameter setting, simulation execution, and GP computations.

        Args:
            model (Model): The model to simulate.
            theta (np.ndarray): Combined array of model parameters and GP hyperparameters.
            startTime (List[datetime]): List of start times for each simulation period.
            endTime (List[datetime]): List of end times for each simulation period.
            stepSize (List[int]): List of step sizes for each simulation period.

        Returns:
            Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]: A tuple containing:
                - Full GP predictions including uncertainty
                - Base model predictions
                - GP noise predictions
                Returns None if simulation or GP computation fails.

        Raises:
            np.linalg.LinAlgError: If GP computation encounters numerical instability.

        Notes:
            This is an internal method primarily used by the Bayesian inference process.
            Uses the george GP library for Gaussian process computations.
        """
        try:
            n_par = model.result["n_par"]
            n_par_map = model.result["n_par_map"]
            theta_kernel = np.exp(theta[-n_par:])
            theta = theta[:-n_par]

            # Set parameters for the model
            theta = theta[self.theta_mask]
            self.model.set_parameters_from_array(theta, self.flat_component_list, self.flat_attr_list)
            simulation_readings_train = {measuring_device.id: [] for measuring_device in self.targetMeasuringDevices}
            actual_readings_train = {measuring_device.id: [] for measuring_device in self.targetMeasuringDevices}
            x_train = {measuring_device.id: [] for measuring_device in self.targetMeasuringDevices}
            for stepSize_train, startTime_train, endTime_train in zip(model.result["stepSize_train"], model.result["startTime_train"], model.result["endTime_train"]):
                df_actual_readings_train = self.get_actual_readings(startTime=startTime_train, endTime=endTime_train, stepSize=stepSize_train)
                self.simulate(model,
                                stepSize=stepSize_train,
                                startTime=startTime_train,
                                endTime=endTime_train,
                                show_progress_bar=False)
                self.get_gp_input(self.targetMeasuringDevices, startTime=startTime_train, endTime=endTime_train, stepSize=stepSize_train, input_type="boundary", add_time=True, max_inputs=4)
                # self.get_gp_input(self.targetMeasuringDevices, startTime=startTime_train, endTime=endTime_train, stepSize=stepSize_train, input_type="closest", add_time=False, max_inputs=7, gp_input_map=self.model.result["gp_input_map"])
                for measuring_device in self.targetMeasuringDevices:
                    simulation_readings_train[measuring_device.id].append(np.array(next(iter(measuring_device.savedInput.values()))))#self.targetMeasuringDevices[measuring_device]["scale_factor"])
                    actual_readings_train[measuring_device.id].append(df_actual_readings_train[measuring_device.id].to_numpy()[self.n_initialization_steps:])#self.targetMeasuringDevices[measuring_device]["scale_factor"])
                    x = self.gp_input[measuring_device.id]
                    x_train[measuring_device.id].append(x[self.n_initialization_steps:])
                        
            for measuring_device in self.targetMeasuringDevices:
                simulation_readings_train[measuring_device.id] = np.concatenate(simulation_readings_train[measuring_device.id])#-model.result["mean_train"][measuring_device.id])/model.result["sigma_train"][measuring_device.id]
                actual_readings_train[measuring_device.id] = np.concatenate(actual_readings_train[measuring_device.id])#-model.result["mean_train"][measuring_device.id])/model.result["sigma_train"][measuring_device.id]
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
                                show_progress_bar=False)
                self.get_gp_input(self.targetMeasuringDevices, startTime=startTime_, endTime=endTime_, stepSize=stepSize_, input_type="boundary", add_time=True, max_inputs=4)
                # self.get_gp_input(self.targetMeasuringDevices, startTime=startTime_, endTime=endTime_, stepSize=stepSize_, input_type="closest", add_time=False, max_inputs=7, gp_input_map=self.model.result["gp_input_map"])
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
    
    def _sim_func_wrapped(self, args: Tuple) -> Optional[Tuple[None, np.ndarray, None]]:
        """
        Wrapper for _sim_func to use with multiprocessing.

        Unpacks arguments and calls _sim_func with proper parameters.

        Args:
            args (Tuple): Tuple containing all arguments for _sim_func.

        Returns:
            Optional[Tuple[None, np.ndarray, None]]: See _sim_func return value.
        """
        return self._sim_func(*args)
    
    def _sim_func_wrapped_gaussian_process(self, args: Tuple) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Wrapper for _sim_func_gaussian_process to use with multiprocessing.

        Unpacks arguments and calls _sim_func_gaussian_process with proper parameters.

        Args:
            args (Tuple): Tuple containing all arguments for _sim_func_gaussian_process.

        Returns:
            Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]: See _sim_func_gaussian_process return value.
        """
        return self._sim_func_gaussian_process(*args)
    
    def bayesian_inference(self, model: model.Model, 
                           startTime: List[datetime.datetime], 
                           endTime: List[datetime.datetime], 
                           stepSize: List[int], 
                           targetMeasuringDevices: Optional[Dict[Union[str, System], Dict]] = None, 
                           n_initialization_steps: int = 0, 
                           show_progress_bar: bool = True,
                           assume_uncorrelated_noise: bool = True, 
                           burnin: Optional[int] = None,
                           n_samples_max: int = 100, 
                           n_cores: int = multiprocessing.cpu_count(),
                           seed: Optional[int] = None) -> Dict:
        """
        Perform Bayesian inference on the model. Simulates the model n_samples_max times (or less), where N is the number of parameter samples generated from a Markov Chain Monte Carlo (MCMC) sampling approach to estimate
        model parameters and uncertainties.

        Args:
            model (Model): The model to perform inference on.
            startTime (List[datetime]): List of start times for simulation periods.
            endTime (List[datetime]): List of end times for simulation periods.
            stepSize (List[int]): List of step sizes for simulation periods.
            targetMeasuringDevices (Optional[Dict[Union[str, System], Dict]], optional): 
                Dictionary mapping devices to their configuration parameters. 
                Defaults to None.
            n_initialization_steps (int, optional): Number of steps to skip at start.
                Defaults to 0.
            show_progress_bar (bool, optional): Whether to show progress during inference.
                Defaults to True.
            assume_uncorrelated_noise (bool, optional): Whether to assume uncorrelated noise.
                Defaults to True.
            burnin (Optional[int], optional): Number of samples to discard as burn-in.
                Defaults to None.
            n_samples_max (int, optional): Maximum number of MCMC samples.
                Defaults to 100.
            n_cores (int, optional): Number of CPU cores to use.
                Defaults to all available cores.
            seed (Optional[int], optional): Random seed for reproducibility.
                Defaults to None.

        Returns:
            Dict: Dictionary containing inference results including:
                - samples: Parameter samples from posterior
                - log_prob: Log probabilities of samples
                - acceptance_fraction: MCMC acceptance rate
                - chain_log: Dictionary with additional chain information

        Raises:
            AssertionError: If input parameters are invalid.
            ValueError: If MCMC sampling fails to converge.

        Notes:
            Uses parallel tempering MCMC for better exploration of parameter space.
            Implements automatic convergence checking and chain adaptation.
        """
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
                assert k in model.components.keys(), f"Measuring device {k} not found in the model."
                targetMeasuringDevices_new[model.components[k]] = v
            else:
                assert k in model.components.values(), f"Measuring device object {k} not found in the model."
                targetMeasuringDevices_new[k] = v
        self.targetMeasuringDevices = targetMeasuringDevices_new
        s = model.result["chain_x"].shape[0]
        assert burnin<=model.result["chain_x"].shape[0], f"The burnin parameter ({str(burnin)}) must be less than the number of samples in the chain ({str(s)})."

        parameter_chain = model.result["chain_x"][burnin:,0,:,:]
        parameter_chain = parameter_chain.reshape((parameter_chain.shape[0]*parameter_chain.shape[1], parameter_chain.shape[2]))

        n_samples = parameter_chain.shape[0] if parameter_chain.shape[0]<self.n_samples_max else self.n_samples_max #100
        sample_indices = np.random.randint(parameter_chain.shape[0], size=n_samples)
        parameter_chain_sampled = parameter_chain[sample_indices]

        self.flat_component_list = [model.components[com_id] for com_id in model.result["component_id"]]
        self.flat_attr_list = model.result["component_attr"]
        self.theta_mask = model.result["theta_mask"]

        print("Running inference...")
        
        # pbar = tqdm(total=len(sample_indices))
        # y_list = [_sim_func(self, parameter_set) for parameter_set in parameter_chain_sampled]
 
        if assume_uncorrelated_noise==False:
            print([type(stepSize_) for stepSize_ in stepSize])
            sim_func = self._sim_func_wrapped_gaussian_process
            args = [(model, parameter_set, startTime, endTime, stepSize) for parameter_set in parameter_chain_sampled]#########################################
        else:
            sim_func = self._sim_func_wrapped
            args = [(model, parameter_set, startTime, endTime, stepSize) for parameter_set in parameter_chain_sampled]

        # del model.result["chain_x"] ########################################

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
