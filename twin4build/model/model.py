import pandas as pd
import warnings
import inspect
import numpy as np
import pandas as pd
import datetime
from prettytable import PrettyTable
from twin4build.utils.print_progress import PrintProgress
from twin4build.utils.mkdir_in_root import mkdir_in_root
from twin4build.utils.data_loaders.load_spreadsheet import sample_from_df
import twin4build.saref4syst.system as system
import twin4build.systems as systems
from typing import List, Dict, Any, Optional, Tuple, Type, Callable
import twin4build.translator.translator as translator
import twin4build.model.semantic_model.semantic_model as semantic_model
import twin4build.model.simulation_model as simulation_model

class Model:
    """
    A class representing a building system model.

    This class is responsible for creating, managing, and simulating a building system model.
    It handles component instantiation, connections between components, and execution order
    for simulation.

    Attributes:
        id (str): Unique identifier for the model.
        saveSimulationResult (bool): Flag to determine if simulation results should be saved.
        components (dict): Dictionary of all components in the model.
        object_dict (dict): Dictionary of all objects in the model.
        system_dict (dict): Dictionary of systems in the model (ventilation, heating, cooling).
        execution_order (list): Ordered list of component groups for execution.
        flat_execution_order (list): Flattened list of components in execution order.
    """

    __slots__ = (
        '_id', 
        '_simulation_model',
        '_semantic_model',
        '_dir_conf', 
        '_is_loaded', 
        "_is_validated", 
        "_p"
    )


    def __str__(self):
        t = PrettyTable(["Number of components in simulation model: ", len(self.components)])
        t.add_row(["Number of connections in simulation model: ", self.simulation_model.count_connections()], divider=True)
        title = f"Model overview    id: {self._id}"
        t.title = title
        t.add_row(["Number of instances in semantic model: ", self.semantic_model.count_instances()], divider=True)
        t.add_row(["Number of triples in semantic model: ", self.semantic_model.count_triples()], divider=True)
        t.add_row(["", ""])
        t.add_row(["", ""], divider=True)
        t.add_row(["id", "Class"], divider=True)
        unique_class_list = []
        for component in self.components.values():
            cls = component.__class__
            if cls not in unique_class_list:
                unique_class_list.append(cls)
        unique_class_list = sorted(unique_class_list, key=lambda x: x.__name__.lower())

        for cls in unique_class_list:
            cs = self.get_component_by_class(self.components, cls, filter=lambda v, class_: v.__class__ is class_)
            n = len(cs)
            for i,c in enumerate(cs):
                t.add_row([c.id, cls.__name__], divider=True if i==n-1 else False)
            
        return t.get_string()

    def __init__(self, id: str, saveSimulationResult: bool = True) -> None:
        """
        Initialize the Model instance.

        Args:
            id (str): Unique identifier for the model.
            saveSimulationResult (bool): Flag to determine if simulation results should be saved.

        Raises:
            AssertionError: If the id is not a string or contains invalid characters.
        """
        
        valid_chars = ["_", "-", " ", "(", ")", "[", "]"]
        assert isinstance(id, str), f"Argument \"id\" must be of type {str(type(str))}"
        isvalid = np.array([x.isalnum() or x in valid_chars for x in id])
        np_id = np.array(list(id))
        violated_characters = list(np_id[isvalid==False])
        assert all(isvalid), f"The model with id \"{id}\" has an invalid id. The characters \"{', '.join(violated_characters)}\" are not allowed."
        self._id = id
        self._dir_conf = ["generated_files", "models", self._id]

        self._simulation_model = simulation_model.SimulationModel(id=f"{id}_simulation_model", saveSimulationResult=saveSimulationResult, dir_conf=self.dir_conf)
        self._semantic_model = semantic_model.SemanticModel(id=f"{id}_semantic_model", dir_conf=self.dir_conf)

        self._is_loaded = False
        self._is_validated = False


    @property
    def id(self) -> str:
        return self._id

    @property
    def simulation_model(self) -> simulation_model.SimulationModel:
        return self._simulation_model
    
    @property
    def semantic_model(self) -> semantic_model.SemanticModel:
        return self._semantic_model

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded
    
    @property
    def is_validated(self) -> bool:
        return self._is_validated
    
    @property
    def saveSimulationResult(self) -> bool:
        return self.simulation_model.saveSimulationResult

    @property
    def components(self) -> dict:
        return self.simulation_model.components

    @property
    def component_dict(self) -> dict:
        """
        Deprecated property that provides backward compatibility for accessing components.
        Will be removed.
        
        Returns:
            dict: Dictionary of all components in the model
        """
        warnings.warn(
            "component_dict is deprecated and will be removed."
            "Use components instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return self.components
    
    @property
    def dir_conf(self) -> List[str]:
        return self._dir_conf
    
    @dir_conf.setter
    def dir_conf(self, dir_conf: List[str]) -> None:
        assert isinstance(dir_conf, list) and all(isinstance(x, str) for x in dir_conf), f"The set value must be of type {list} and contain strings"
        self._dir_conf = dir_conf


    @property
    def result(self) -> Any:
        return self.simulation_model.result
    
    @property
    def execution_order(self) -> List[str]:
        return self.simulation_model.execution_order
    
    @property
    def flat_execution_order(self) -> List[str]:
        return self.simulation_model.flat_execution_order
    

    def get_dir(self, folder_list: List[str] = [], filename: Optional[str] = None) -> Tuple[str, bool]:
        """
        Get the directory path for storing model-related files.

        Args:
            folder_list (List[str]): List of folder names to create.
            filename (Optional[str]): Name of the file to create.

        Returns:
            Tuple[str, bool]: The full path to the directory or file, and a boolean indicating if the file exists.
        """
        folder_list_ = self.dir_conf.copy()
        folder_list_.extend(folder_list)
        filename, isfile = mkdir_in_root(folder_list=folder_list_, filename=filename)
        return filename, isfile

    def add_component(self, component: system.System) -> None:
        """
        Add a component to the model.

        Args:
            component (system.System): The component to add.

        Raises:
            AssertionError: If the component is not an instance of system.System.
        """
        self.simulation_model.add_component(component=component)

    def make_pickable(self) -> None:
        """
        Make the model instance pickable by removing unpickable references.

        This method prepares the Model instance for use with multiprocessing in the Estimator class.
        """
        self.simulation_model.make_pickable()

    def remove_component(self, component: system.System) -> None:
        """
        Remove a component from the model.

        Args:
            component (system.System): The component to remove.
        """
        self.simulation_model.remove_component(component=component)

    def add_connection(self, sender_component: system.System, receiver_component: system.System, 
                       sender_property_name: str, receiver_property_name: str) -> None:
        """
        Add a connection between two components in the system.

        Args:
            sender_component (system.System): The component sending the connection.
            receiver_component (system.System): The component receiving the connection.
            sender_property_name (str): Name of the sender property.
            receiver_property_name (str): Name of the receiver property.
        Raises:
            AssertionError: If property names are invalid for the components.
            AssertionError: If a connection already exists.
        """
        self.simulation_model.add_connection(sender_component=sender_component, 
                                             receiver_component=receiver_component, 
                                             sender_property_name=sender_property_name, 
                                             receiver_property_name=receiver_property_name)

    def remove_connection(self, sender_component: system.System, receiver_component: system.System, 
                          sender_property_name: str, receiver_property_name: str) -> None:
        """
        Remove a connection between two components in the system.

        Args:
            sender_component (system.System): The component sending the connection.
            receiver_component (system.System): The component receiving the connection.
            sender_property_name (str): Name of the sender property.
            receiver_property_name (str): Name of the receiver property.

        Raises:
            ValueError: If the specified connection does not exist.
        """
        self.simulation_model.remove_connection(sender_component=sender_component, 
                                               receiver_component=receiver_component, 
                                               sender_property_name=sender_property_name, 
                                               receiver_property_name=receiver_property_name)
      
    def get_component_by_class(self, 
                               dict_: Dict, class_: Type, 
                               filter: Optional[Callable] = None) -> List:
        """
        Get components of a specific class from a dictionary.

        Args:
            dict_ (Dict): The dictionary to search.
            class_ (Type): The class to filter by.
            filter (Optional[Callable]): Additional filter function.

        Returns:
            List: List of components matching the class and filter.
        """
        return self.simulation_model.get_component_by_class(dict_=dict_, class_=class_, filter=filter)

    def set_custom_initial_dict(self, custom_initial_dict: Dict[str, Dict[str, Any]]) -> None:
        """
        Set custom initial values for components.

        Args:
            custom_initial_dict (Dict[str, Dict[str, Any]]): Dictionary of custom initial values.

        Raises:
            AssertionError: If unknown component IDs are provided.
        """
        self.simulation_model.set_custom_initial_dict(custom_initial_dict=custom_initial_dict)

    def set_initial_values(self) -> None:
        """
        Set initial values for all components in the model.
        """
        self.simulation_model.set_initial_values()

    def set_parameters_from_array(self, 
                                  parameters: List[Any], 
                                  component_list: List[system.System], 
                                  attr_list: List[str]) -> None:
        """
        Set parameters for components from an array.

        Args:
            parameters (List[Any]): List of parameter values.
            component_list (List[system.System]): List of components to set parameters for.
            attr_list (List[str]): List of attribute names corresponding to the parameters.

        Raises:
            AssertionError: If a component doesn't have the specified attribute.
        """
        self.simulation_model.set_parameters_from_array(parameters=parameters, 
                                                       component_list=component_list, 
                                                       attr_list=attr_list)

    def set_parameters_from_dict(self, 
                                 parameters: Dict[str, Any], 
                                 component_list: List[system.System], 
                                 attr_list: List[str]) -> None:
        """
        Set parameters for components from a dictionary.

        Args:
            parameters (Dict[str, Any]): Dictionary of parameter values.
            component_list (List[system.System]): List of components to set parameters for.
            attr_list (List[str]): List of attribute names corresponding to the parameters.

        Raises:
            AssertionError: If a component doesn't have the specified attribute.
        """
        self.simulation_model.set_parameters_from_dict(parameters=parameters, 
                                                       component_list=component_list, 
                                                       attr_list=attr_list)

    def cache(self, startTime: Optional[datetime.datetime] = None, endTime: Optional[datetime.datetime] = None, stepSize: Optional[int] = None) -> None:
        """
        Cache data and create folder structure for time series data.

        Args:
            startTime (Optional[datetime.datetime]): Start time for caching.
            endTime (Optional[datetime.datetime]): End time for caching.
            stepSize (Optional[int]): Time step size for caching.
        """
        self.simulation_model.cache(startTime=startTime, 
                                    endTime=endTime, 
                                    stepSize=stepSize)

    def initialize(self,
                   startTime: Optional[datetime.datetime] = None,
                   endTime: Optional[datetime.datetime] = None,
                   stepSize: Optional[int] = None) -> None:
        """
        Initialize the model for simulation.

        Args:
            startTime (Optional[datetime.datetime]): Start time for the simulation.
            endTime (Optional[datetime.datetime]): End time for the simulation.
            stepSize (Optional[int]): Time step size for the simulation.
        """
        self.simulation_model.initialize(startTime=startTime, 
                                         endTime=endTime, 
                                         stepSize=stepSize)



    def validate(self) -> None:
        """
        Validate the model by checking IDs and connections.
        """
        self.simulation_model.validate()

    def _load_parameters(self, force_config_update: bool = False) -> None:
        """
        Load parameters for all components from configuration files.

        Args:
            force_config_update (bool): If True, all parameters are read from the config file. If False, only the parameters that are None are read from the config file. If you want to use the fcn function
            to set the parameters, you should set force_config_update to False to avoid it being overwritten.
        """
        self.simulation_model._load_parameters(force_config_update=force_config_update)


    def _read_input_config(self, input_dict: Dict) -> None:
        """
        Read input configuration and populate corresponding objects.

        Args:
            input_dict (Dict): Dictionary containing input configuration data.
        """
        time_format = '%Y-%m-%d %H:%M:%S%z'
        startTime = datetime.datetime.strptime(input_dict["metadata"]["start_time"], time_format)
        endTime = datetime.datetime.strptime(input_dict["metadata"]["end_time"], time_format)
        stepSize = input_dict["metadata"]['stepSize']

        #print(input_dict.keys())
        
        
        if "rooms_sensor_data" in input_dict:
            """
            Reacting to different input combinations for custom ventilation system model
            """
            
            for component_id, component_data in input_dict["rooms_sensor_data"].items():
                data_available = bool(component_data["sensor_data_available"]) 
                if data_available:
                    timestamp = component_data["time"]
                    co2_values = component_data["co2"]
                    damper_values = component_data["damper_position"]
                    
                    df_co2 = pd.DataFrame({"timestamp": timestamp, "co2": co2_values})
                    df_damper = pd.DataFrame({"timestamp": timestamp, "damper_position": damper_values})
                    
                    df_co2 = sample_from_df(df_co2,
                                            stepSize=stepSize,
                                            start_time=startTime,
                                            end_time=endTime,
                                            resample=True,
                                            clip=True,
                                            tz="Europe/Copenhagen",
                                            preserve_order=True)
                    df_damper = sample_from_df(df_damper,
                                            stepSize=stepSize,
                                            start_time=startTime,
                                            end_time=endTime,
                                            resample=True,
                                            clip=True,
                                            tz="Europe/Copenhagen",
                                            preserve_order=True)
                    df_damper["damper_position"] = df_damper["damper_position"]/100
                    
                    components_ = self.components.keys()
                    #Make it an array of strings
                    components_ = list(components_)
                    # find all the components that contain the last part of the component_id, after the first dash
                    
                    #Extract the substring after the first dash
                    substring_id = component_id.split("-")[1] + "-" + component_id.split("-")[2]
                    substring_id = substring_id.lower().replace("-", "_")
                    
                    # Find all the components that contain the substring
                    filtered_components = [component_ for component_ in components_ if substring_id in component_]

                    #If the component contains "co2" in the id, add the co2 data
                    for component_ in filtered_components:
                        if "CO2_sensor" in component_:
                            co2_sensor = self.components[component_]
                            co2_sensor.df_input = df_co2
                        elif "Damper_position_sensor" in component_:
                            damper_position_sensor = self.components[component_]
                            damper_position_sensor.df_input = df_damper
                else:
                    """
                    Reading schedules and reacting to different controller configurations
                    """
                    rooms_volumes = {
                        "22_601b_00": 500,
                        "22_601b_0": 417,
                        "22_601b_1": 417,
                        "22_601b_2": 417,
                        "22_603_0" : 240,
                        "22_603_1" : 240,
                        "22_604_0" : 486,
                        "22_604_1" : 375,
                        "22_603b_2" : 159,
                        "22_603a_2" : 33,
                        "22_604a_2" : 33,
                        "22_604b_2" : 33,
                        "22_605a_2" : 33,
                        "22_605b_2" : 33,
                        "22_604e_2" : 33,
                        "22_604d_2" : 33,
                        "22_604c_2" : 33,
                        "22_605e_2" : 33,
                        "22_605d_2" : 33,
                        "22_605c_2" : 30,

                    }

                    components_ = self.components.keys()
                    components_ = list(components_)
                    substring_id = component_id.split("-")[1] + "-" + component_id.split("-")[2]
                    substring_id = substring_id.lower().replace("-", "_")
                    substring_id = "22_" + substring_id
                    room_filtered_components = [component_ for component_ in components_ if substring_id in component_]

                    if substring_id == "22_601b_0":
                        components_601b_00 = {
                            "CO2_controller_sensor_22_601b_00",                             
                            "Damper_position_sensor_22_601b_00", 
                            "Supply_damper_22_601b_00", 
                            "Return_damper_22_601b_00", 
                            "CO2_sensor_22_601b_00"
                        }
                        filtered_components = []
                        for component_ in room_filtered_components:
                            if component_ not in components_601b_00:
                                filtered_components.append(component_)
                        room_filtered_components = filtered_components

                    controller_type = component_data["controller_type"]

                    if controller_type == "PID":
                        #Assert that a co2 setpoint schedule is available, if not raise an error with a message
                        assert "co2_setpoint_schedule" in component_data["schedules"], "No CO2 setpoint schedule in input dict. available for PID controller"
                        schedule_input = component_data["schedules"]["co2_setpoint_schedule"]
                        #Assert that the controller constants are available, if not raise an error with a message
                        assert "kp" in component_data, "No kp value in input dict. available for PID controller"
                        assert "ki" in component_data, "No ki value in input dict. available for PID controller"
                        assert "kd" in component_data, "No kd value in input dict. available for PID controller"
                        #get the id of the sensor from the filtered components
                        co2_sensor_component_id = next((component_ for component_ in room_filtered_components if "CO2_sensor" in component_), None)
                        co2_sensor_component = self.components[co2_sensor_component_id]
                        co2_sensor_observed_property = co2_sensor_component.observes

                        #Create the schedule object
                        co2_setpoint_schedule = systems.ScheduleSystem(
                            **schedule_input,
                            add_noise = False,
                            saveSimulationResult = True,
                            id = f"{substring_id}_co2_setpoint_schedule")
                        #Create the PID controller object
                        pid_controller = systems.ControllerSystem(
                            observes = co2_sensor_observed_property,
                            K_p = component_data["kp"],
                            K_i = component_data["ki"],
                            K_d = component_data["kd"],
                            saveSimulationResult = True,
                            id = f"{substring_id}_CO2_PID_controller")
                 
                        #Remove the connections to the previous controller and delete it
                        ann_controller_component_id = next((component_ for component_ in room_filtered_components if "CO2_controller" in component_), None)
                        ann_controller_component = self.components[ann_controller_component_id]
                        co2_sensor_component_id = next((component_ for component_ in room_filtered_components if "CO2_sensor" in component_), None)
                        co2_sensor_component = self.components[co2_sensor_component_id]
                        supply_damper_id = next((component_ for component_ in room_filtered_components if "Supply_damper" in component_), None)
                        return_damper_id = next((component_ for component_ in room_filtered_components if "Return_damper" in component_), None)
                        supply_damper = self.components[supply_damper_id]
                        return_damper = self.components[return_damper_id]

                        self.remove_connection(co2_sensor_component, ann_controller_component, "indoorCo2Concentration", "actualValue")
                        self.remove_connection(ann_controller_component, return_damper, "inputSignal", "damperPosition")
                        self.remove_connection(ann_controller_component, supply_damper, "inputSignal", "damperPosition")
                        self.remove_component(ann_controller_component)

                        #Add the components to the model
                        self._add_component(co2_setpoint_schedule)
                        self._add_component(pid_controller)
                        #Add the connection between the schedule and the controller
                        self.add_connection(co2_setpoint_schedule, pid_controller, "scheduleValue", "setpointValue")
                        self.add_connection(co2_sensor_component, pid_controller, "indoorCo2Concentration", "actualValue")
                        #Add the connection between the controller and the dampers
                        self.add_connection(pid_controller, supply_damper, "inputSignal", "damperPosition")
                        self.add_connection(pid_controller, return_damper, "inputSignal", "damperPosition")
                        pid_controller.observes.isPropertyOf = co2_sensor_component
                
                        #Recalculate the filtered components
                        components_ = self.components.keys()
                        components_ = list(components_)
                        substring_id = component_id.split("-")[1] + "-" + component_id.split("-")[2]
                        substring_id = substring_id.lower().replace("-", "_")
                        substring_id = "22_" + substring_id
                        room_filtered_components = [component_ for component_ in components_ if substring_id in component_]

                        if substring_id == "22_601b_0":
                            components_601b_00 = {
                                "CO2_controller_sensor_22_601b_00",                             
                                "Damper_position_sensor_22_601b_00", 
                                "Supply_damper_22_601b_00", 
                                "Return_damper_22_601b_00", 
                                "CO2_sensor_22_601b_00"
                            }
                            filtered_components = []
                            for component_ in room_filtered_components:
                                if component_ not in components_601b_00:
                                    filtered_components.append(component_)
                            room_filtered_components = filtered_components
                    elif controller_type == "RBC":
                        #Assert that a co2 setpoint schedule is available, if not raise an error with a message
                        assert "co2_setpoint_schedule" in component_data["schedules"], "No CO2 setpoint schedule in input dict. available for PID controller"
                        schedule_input = component_data["schedules"]["co2_setpoint_schedule"]
                        #get the id of the sensor from the filtered components
                        co2_sensor_component_id = next((component_ for component_ in room_filtered_components if "CO2_sensor" in component_), None)
                        co2_sensor_component = self.components[co2_sensor_component_id]
                        co2_sensor_observed_property = co2_sensor_component.observes

                        #Create the schedule object
                        co2_setpoint_schedule = systems.ScheduleSystem(
                            **schedule_input,
                            add_noise = False,
                            saveSimulationResult = True,
                            id = f"{substring_id}_co2_setpoint_schedule")
                        #Create the RBC controller object
                        rbc_controller = systems.RulebasedSetpointInputControllerSystem(
                            observes = co2_sensor_observed_property,
                            saveSimulationResult = True,
                            id = f"{substring_id}_CO2_RBC_controller")
                        
                 
                        #Remove the connections to the previous controller and delete it
                        ann_controller_component_id = next((component_ for component_ in room_filtered_components if "CO2_controller" in component_), None)
                        ann_controller_component = self.components[ann_controller_component_id]
                        co2_sensor_component_id = next((component_ for component_ in room_filtered_components if "CO2_sensor" in component_), None)
                        co2_sensor_component = self.components[co2_sensor_component_id]
                        supply_damper_id = next((component_ for component_ in room_filtered_components if "Supply_damper" in component_), None)
                        return_damper_id = next((component_ for component_ in room_filtered_components if "Return_damper" in component_), None)
                        supply_damper = self.components[supply_damper_id]
                        return_damper = self.components[return_damper_id]

                        self.remove_connection(co2_sensor_component, ann_controller_component, "indoorCo2Concentration", "actualValue")
                        self.remove_connection(ann_controller_component, return_damper, "inputSignal", "damperPosition")
                        self.remove_connection(ann_controller_component, supply_damper, "inputSignal", "damperPosition")
                        self.remove_component(ann_controller_component)

                        #Add the components to the model
                        self._add_component(co2_setpoint_schedule)
                        self._add_component(rbc_controller)
                        #Add the connection between the schedule and the controller
                        self.add_connection(co2_setpoint_schedule, rbc_controller, "scheduleValue", "setpointValue")
                        self.add_connection(co2_sensor_component, rbc_controller, "indoorCo2Concentration", "actualValue")
                        #Add the connection between the controller and the dampers
                        self.add_connection(rbc_controller, supply_damper, "inputSignal", "damperPosition")
                        self.add_connection(rbc_controller, return_damper, "inputSignal", "damperPosition")
                        rbc_controller.observes.isPropertyOf = co2_sensor_component
                
                        #Recalculate the filtered components
                        components_ = self.components.keys()
                        components_ = list(components_)
                        substring_id = component_id.split("-")[1] + "-" + component_id.split("-")[2]
                        substring_id = substring_id.lower().replace("-", "_")
                        substring_id = "22_" + substring_id
                        room_filtered_components = [component_ for component_ in components_ if substring_id in component_]

                        if substring_id == "22_601b_0":
                            components_601b_00 = {
                                "CO2_controller_sensor_22_601b_00",                             
                                "Damper_position_sensor_22_601b_00", 
                                "Supply_damper_22_601b_00", 
                                "Return_damper_22_601b_00", 
                                "CO2_sensor_22_601b_00"
                            }
                            filtered_components = []
                            for component_ in room_filtered_components:
                                if component_ not in components_601b_00:
                                    filtered_components.append(component_)
                            room_filtered_components = filtered_components
                    
                    
                    for component_ in room_filtered_components:
                        if "CO2_sensor" in component_:
                            sender_component = self.components[component_]
                            receiver_component_id = next((component_ for component_ in room_filtered_components if "_controller" in component_), None)
                            receiver_component = self.components[receiver_component_id]
                            self.remove_connection(sender_component, receiver_component, "indoorCo2Concentration", "actualValue")
                            self.remove_component(sender_component)
                            
                            schedule_input = component_data["schedules"]["occupancy_schedule"] 
                            #Create the schedule object
                            room_occupancy_schedule = systems.ScheduleSystem(
                                **schedule_input,
                                add_noise = False,
                                saveSimulationResult = True,
                                id = f"{substring_id}_occupancy_schedule")
                            #Create the space co2 object
                            room_volume = int(rooms_volumes[substring_id]) 
                            room_space_co2 = systems.BuildingSpaceCo2System(
                                airVolume = room_volume,
                                saveSimulationResult = True,
                                id = f"{substring_id}_CO2_space")
                            
                            self._add_component(room_occupancy_schedule)
                            self._add_component(room_space_co2)

                            supply_damper_id = next((component_ for component_ in room_filtered_components if "Supply_damper" in component_), None)
                            return_damper_id = next((component_ for component_ in room_filtered_components if "Return_damper" in component_), None)
                            supply_damper = self.components[supply_damper_id]
                            return_damper = self.components[return_damper_id]

                            self.add_connection(room_occupancy_schedule, room_space_co2,
                                "scheduleValue", "numberOfPeople")
                            self.add_connection(supply_damper, room_space_co2,
                                "airFlowRate", "supplyAirFlowRate")
                            self.add_connection(return_damper, room_space_co2,
                                "airFlowRate", "returnAirFlowRate")
                            self.add_connection(room_space_co2, receiver_component,
                                "indoorCo2Concentration", "actualValue")
                            
                            receiver_component.observes.isPropertyOf = room_space_co2           


            ## ADDED FOR DAMPER CONTROL of 601b_00, missing data
            oe_601b_00_component = self.components["CO2_controller_sensor_22_601b_00"]
            freq = f"{stepSize}S"
            df_damper_control = pd.DataFrame({"timestamp": pd.date_range(start=startTime, end=endTime, freq=freq)})
            df_damper_control["damper_position"] = 1
            
            df_damper_control = sample_from_df(df_damper_control,
                        stepSize=stepSize,
                        start_time=startTime,
                        end_time=endTime,
                        resample=True,
                        clip=True,
                        tz="Europe/Copenhagen",
                        preserve_order=True)
            oe_601b_00_component.df_input = df_damper_control

        else:
            sensor_inputs = input_dict["inputs_sensor"] #Change naming to be consistent
            schedule_inputs = input_dict["input_schedules"] #Change naming to be consistent
            weather_inputs = sensor_inputs["ml_inputs_dmi"]
            
            df_raw = pd.DataFrame()
            df_raw.insert(0, "datetime", weather_inputs["observed"])
            df_raw.insert(1, "outdoorTemperature", weather_inputs["temp_dry"])
            df_raw.insert(2, "globalIrradiation", weather_inputs["radia_glob"])
            df_sample = sample_from_df(df_raw,
                                        stepSize=stepSize,
                                        start_time=startTime,
                                        end_time=endTime,
                                        resample=True,
                                        clip=True,
                                        tz="Europe/Copenhagen",
                                        preserve_order=True)

            outdoor_environment = systems.OutdoorEnvironmentSystem(df_input=df_sample,
                                                            saveSimulationResult = self.saveSimulationResult,
                                                            id = "outdoor_environment")

            '''
                MODIFIED BY NEC-INDIA

                temperature_list = sensor_inputs["ml_inputs"]["temperature"]
            '''
            if sensor_inputs["ml_inputs"]["temperature"][0]=="None":
                initial_temperature = 21
            else:
                initial_temperature = float(sensor_inputs["ml_inputs"]["temperature"][0])
            custom_initial_dict = {"OE20-601b-2": {"indoorTemperature": initial_temperature}}
            self.set_custom_initial_dict(custom_initial_dict)

            indoor_temperature_setpoint_schedule = systems.ScheduleSystem(
                **schedule_inputs["temperature_setpoint_schedule"],
                add_noise = False,
                saveSimulationResult = True,
                id = "OE20-601b-2_temperature_setpoint_schedule")

            occupancy_schedule = systems.ScheduleSystem(
                **schedule_inputs["occupancy_schedule"],
                add_noise = True,
                saveSimulationResult = True,
                id = "OE20-601b-2_occupancy_schedule")

            supply_water_temperature_setpoint_schedule = systems.PiecewiseLinearScheduleSystem(
                **schedule_inputs["supply_water_temperature_schedule_pwlf"],
                saveSimulationResult = True,
                id = "Heating system_supply_water_temperature_schedule")
            
            supply_air_temperature_schedule = systems.ScheduleSystem(
                **schedule_inputs["supply_air_temperature_schedule"],
                saveSimulationResult = True,
                id = "Ventilation system_supply_air_temperature_schedule")

            space = self.components["OE20-601b-2"]
            space_heater = self.components["Space heater"]
            temperature_controller = self.components["Temperature controller"]

            self.add_connection(outdoor_environment, space, "outdoorTemperature", "outdoorTemperature")
            self.add_connection(outdoor_environment, space, "globalIrradiation", "globalIrradiation")
            self.add_connection(outdoor_environment, supply_water_temperature_setpoint_schedule, "outdoorTemperature", "outdoorTemperature")
            self.add_connection(supply_water_temperature_setpoint_schedule, space_heater, "scheduleValue", "supplyWaterTemperature")
            self.add_connection(supply_water_temperature_setpoint_schedule, space, "scheduleValue", "supplyWaterTemperature")
            self.add_connection(supply_air_temperature_schedule, space, "scheduleValue", "supplyAirTemperature")
            self.add_connection(indoor_temperature_setpoint_schedule, temperature_controller, "scheduleValue", "setpointValue")
            self.add_connection(occupancy_schedule, space, "scheduleValue", "numberOfPeople")


    def load(self, semantic_model_filename: Optional[str] = None, 
             input_config: Optional[Dict] = None, 
             fcn: Optional[Callable] = None, 
             draw_semantic_model: bool = True, 
             create_signature_graphs: bool = False, 
             draw_simulation_model: bool = True, 
             verbose: bool = False, 
             validate_model: bool = True, 
             force_config_update: bool = False) -> None:
        """
        Load and set up the model for simulation.

        Args:
            semantic_model_filename (Optional[str]): Path to the semantic model configuration file.
            input_config (Optional[Dict]): Input configuration dictionary.
            fcn (Optional[Callable]): Custom function to be applied during model loading.
            draw_semantic_model (bool): Whether to create and save the object graph.
            create_signature_graphs (bool): Whether to create and save signature graphs.
            draw_simulation_model (bool): Whether to create and save the system graph.
            verbose (bool): Whether to print verbose output during loading.
            validate_model (bool): Whether to perform model validation.
        """
        if verbose:
            self._load(semantic_model_filename=semantic_model_filename, 
                       input_config=input_config, 
                       fcn=fcn, 
                       draw_semantic_model=draw_semantic_model, 
                       create_signature_graphs=create_signature_graphs, 
                       draw_simulation_model=draw_simulation_model,
                       verbose=verbose,
                       validate_model=validate_model, 
                       force_config_update=force_config_update)
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._load(semantic_model_filename=semantic_model_filename, 
                            input_config=input_config, 
                            fcn=fcn,
                            draw_semantic_model=draw_semantic_model,
                            create_signature_graphs=create_signature_graphs,
                            draw_simulation_model=draw_simulation_model,
                            verbose=verbose,
                            validate_model=validate_model, 
                            force_config_update=force_config_update)

    def _load(self, 
              semantic_model_filename: Optional[str] = None, 
              input_config: Optional[Dict] = None, 
              fcn: Optional[Callable] = None, 
              draw_semantic_model: bool = True, 
              create_signature_graphs: bool = False, 
              draw_simulation_model: bool = True,
              verbose: bool = False,
              validate_model: bool = True, 
              force_config_update: bool = False) -> None:
        """
        Internal method to load and set up the model for simulation.

        This method is called by load and performs the actual loading process.

        Args:
            semantic_model_filename (Optional[str]): Path to the semantic model configuration file.
            fcn (Optional[Callable]): Custom function to be applied during model loading.
            draw_semantic_model (bool): Whether to create and save the object graph.
            create_signature_graphs (bool): Whether to create and save signature graphs.
            draw_simulation_model (bool): Whether to create and save the system graph.
            validate_model (bool): Whether to perform model validation.
        """

        if self._is_loaded:
            warnings.warn("The model is already loaded. Resetting model.")
            self.reset()

        self._is_loaded = True

        self._p = PrintProgress()
        self._p("Loading model")
        self._p.add_level()
        # self.add_outdoor_environment()
        if semantic_model_filename is not None:
            apply_translator = True
            self._p(f"Parsing semantic model", status="")
            self._semantic_model = semantic_model.SemanticModel(semantic_model_filename, 
                                                               dir_conf=self.dir_conf.append("semantic_model"),
                                                               id=f"{self._id}_semantic_model")
            
            if draw_semantic_model:
                self._p(f"Drawing semantic model")
                self._semantic_model.visualize()

        else:
            apply_translator = False


        if input_config is not None:
            warnings.warn(
                "The input_config parameter is deprecated and will be removed in a future version. "
                "Please use the semantic model configuration instead.",
                DeprecationWarning,
                stacklevel=2
            )
            self._p(f"Reading input config")
            self._read_input_config(input_config)

        # if create_signature_graphs:
        #     self._p(f"Drawing signature graphs")
        #     self._create_signature_graphs()
        
        if apply_translator:
            self._p(f"Applying translator")
            translator_ = translator.Translator()
            systems_ = [cls[1] for cls in inspect.getmembers(systems, inspect.isclass) if (issubclass(cls[1], (system.System, )) and hasattr(cls[1], "sp"))]
            self._simulation_model = translator_.translate(systems_, self._semantic_model)
            self._simulation_model.dir_conf = self.dir_conf

        if draw_simulation_model:
            self._p(f"Drawing simulation model")
            self._simulation_model.visualize()
        

        self._simulation_model.load(fcn=fcn,
                                   verbose=verbose, 
                                   validate_model=validate_model, 
                                   force_config_update=force_config_update)

        self._p()
        if verbose:
            print(self)

    def fcn(self) -> None:
        """
        Placeholder for a custom function to be applied during model loading.
        """

    def set_save_simulation_result(self, flag: bool=True, c: list=None):
        self.simulation_model.set_save_simulation_result(flag=flag, c=c)


    def reset(self) -> None:
        """
        Reset the model to its initial state.
        """
        self._id = self._id  # Keep the original id
        # self.saveSimulationResult = self.saveSimulationResult  # Keep the original saveSimulationResult setting

        # Reset all the dictionaries and lists
        self.simulation_model.reset()

        # Reset the loaded state
        self._is_loaded = False ###
        self._is_validated = False ###

        # Reset any estimation results
        self._result = None ###


    def load_estimation_result(self, filename: Optional[str] = None, result: Optional[Dict] = None) -> None:
        """
        Load a chain log from a file or dictionary.

        Args:
            filename (Optional[str]): The filename to load the chain log from.
            result (Optional[Dict]): The chain log dictionary to load.

        Raises:
            AssertionError: If invalid arguments are provided.
        """
        self.simulation_model.load_estimation_result(filename=filename, result=result)
            


    def check_for_for_missing_initial_values(self) -> None:
        """
        Check for missing initial values in components.

        Raises:
            Exception: If any component is missing an initial value.
        """
        self.simulation_model.check_for_for_missing_initial_values()
    


    






    

