import warnings
import numpy as np
import pandas as pd
import datetime
from prettytable import PrettyTable
from twin4build.utils.print_progress import PRINTPROGRESS
from twin4build.utils.mkdir_in_root import mkdir_in_root
import twin4build.core as core
from typing import List, Dict, Any, Optional, Tuple, Type, Callable
# import twin4build.translator.translator as translator
# import twin4build.model.semantic_model.semantic_model as semantic_model
# import twin4build.model.simulation_model as simulation_model

class Model:
    r"""
    A class representing a building system model.

    This class is responsible for creating, managing, and simulating a building system model.
    It handles component instantiation, connections between components, and execution order
    for simulation.

    Attributes:
        id (str): Unique identifier for the model.
        components (dict): Dictionary of all components in the model.
        execution_order (list): Ordered list of component groups for execution.
        flat_execution_order (list): Flattened list of components in execution order.
    """

    __slots__ = (
        '_id', 
        '_simulation_model',
        '_semantic_model',
        '_translator',
        '_dir_conf',
    )


    def __str__(self):
        t = PrettyTable(["Number of components in simulation model: ", len(self.components)])
        t.add_row(["Number of connections in simulation model: ", self.simulation_model.count_connections()], divider=True)
        title = f"Model overview    id: {self._id}"
        t.title = title
        t.add_row(["Model directory: ", self.get_dir()[0]], divider=True)
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

    def __init__(self, id: str) -> None:
        """
        Initialize the Model instance.

        Args:
            id (str): Unique identifier for the model.

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

        self._semantic_model = core.SemanticModel(id=self._id, 
                                                  namespaces={"SIM": core.namespace.SIM,
                                                              "SAREF": core.namespace.SAREF,
                                                              "S4BLDG": core.namespace.S4BLDG,
                                                              "S4SYST": core.namespace.S4SYST,
                                                              "FSO": core.namespace.FSO},
                                                   dir_conf=self._dir_conf + ["semantic_model"])
        self._simulation_model = core.SimulationModel(dir_conf=self.dir_conf + ["simulation_model"],
                                                        id=f"{self._id}_simulation_model")



    @property
    def id(self) -> str:
        return self._id

    @property
    def simulation_model(self) -> "core.SimulationModel":
        return self._simulation_model
    
    @property
    def semantic_model(self) -> "core.SemanticModel":
        return self._semantic_model

    @property
    def is_loaded(self) -> bool:
        return self._simulation_model.is_loaded
    
    @property
    def is_validated(self) -> bool:
        return self._simulation_model.is_validated

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

    def add_component(self, component: core.System) -> None:
        """
        Add a component to the model.

        Args:
            component (core.System): The component to add.

        Raises:
            AssertionError: If the component is not an instance of core.System.
        """
        self.simulation_model.add_component(component=component)

    def make_pickable(self) -> None:
        """
        Make the model instance pickable by removing unpickable references.

        This method prepares the Model instance for use with multiprocessing in the Estimator class.
        """
        self.simulation_model.make_pickable()

    def remove_component(self, component: core.System) -> None:
        """
        Remove a component from the model.

        Args:
            component (core.System): The component to remove.
        """
        self.simulation_model.remove_component(component=component)

    def add_connection(self, sender_component: core.System, receiver_component: core.System, 
                       outputPort: str, inputPort: str) -> None:
        """
        Add a connection between two components in the system.

        Args:
            sender_component (core.System): The component sending the connection.
            receiver_component (core.System): The component receiving the connection.
            outputPort (str): Name of the sender property.
            inputPort (str): Name of the receiver property.
        Raises:
            AssertionError: If property names are invalid for the components.
            AssertionError: If a connection already exists.
        """
        self.simulation_model.add_connection(sender_component=sender_component, 
                                             receiver_component=receiver_component, 
                                             outputPort=outputPort, 
                                             inputPort=inputPort)

    def remove_connection(self, sender_component: core.System, receiver_component: core.System, 
                          outputPort: str, inputPort: str) -> None:
        """
        Remove a connection between two components in the system.

        Args:
            sender_component (core.System): The component sending the connection.
            receiver_component (core.System): The component receiving the connection.
            outputPort (str): Name of the sender property.
            inputPort (str): Name of the receiver property.

        Raises:
            ValueError: If the specified connection does not exist.
        """
        self.simulation_model.remove_connection(sender_component=sender_component, 
                                               receiver_component=receiver_component, 
                                               outputPort=outputPort, 
                                               inputPort=inputPort)
      
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
                                  values: List[Any], 
                                  components: List[core.System], 
                                  parameter_names: List[str],
                                  normalized: List[bool] = None,
                                  overwrite: bool = False) -> None:
        """
        Set parameters for components from an array.

        Args:
            values (List[Any]): List of parameter values.
            component_list (List[core.System]): List of components to set parameters for.
            attr_list (List[str]): List of attribute names corresponding to the parameters.

        Raises:
            AssertionError: If a component doesn't have the specified attribute.
        """
        self.simulation_model.set_parameters_from_array(values=values, 
                                                       components=components, 
                                                       parameter_names=parameter_names,
                                                       normalized=normalized,
                                                       overwrite=overwrite)

    def set_parameters_from_dict(self, 
                                 parameters: Dict[str, Any], 
                                 component_list: List[core.System], 
                                 attr_list: List[str]) -> None:
        """
        Set parameters for components from a dictionary.

        Args:
            parameters (Dict[str, Any]): Dictionary of parameter values.
            component_list (List[core.System]): List of components to set parameters for.
            attr_list (List[str]): List of attribute names corresponding to the parameters.

        Raises:
            AssertionError: If a component doesn't have the specified attribute.
        """
        self.simulation_model.set_parameters_from_dict(parameters=parameters, 
                                                       component_list=component_list, 
                                                       attr_list=attr_list)

    def cache(self, 
              startTime: Optional[datetime.datetime] = None, 
              endTime: Optional[datetime.datetime] = None, 
              stepSize: Optional[int] = None,
              simulator: Optional["core.Simulator"] = None) -> None:
        """
        Cache data and create folder structure for time series data.

        Args:
            startTime (Optional[datetime.datetime]): Start time for caching.
            endTime (Optional[datetime.datetime]): End time for caching.
            stepSize (Optional[int]): Time step size for caching.
        """
        self.simulation_model.cache(startTime=startTime, 
                                    endTime=endTime, 
                                    stepSize=stepSize,
                                    simulator=simulator)

    def initialize(self,
                   startTime: Optional[datetime.datetime] = None,
                   endTime: Optional[datetime.datetime] = None,
                   stepSize: Optional[int] = None,
                   simulator: Optional["core.Simulator"] = None) -> None:
        """
        Initialize the model for simulation.

        Args:
            startTime (Optional[datetime.datetime]): Start time for the simulation.
            endTime (Optional[datetime.datetime]): End time for the simulation.
            stepSize (Optional[int]): Time step size for the simulation.
            simulator (Optional[core.Simulator]): Simulator instance.
        """
        self.simulation_model.initialize(startTime=startTime, 
                                         endTime=endTime, 
                                         stepSize=stepSize,
                                         simulator=simulator)



    def validate(self) -> None:
        """
        Validate the model by checking IDs and connections.
        """
        self.simulation_model.validate()

    def _load_parameters(self, force_config_overwrite: bool = False) -> None:
        """
        Load parameters for all components from configuration files.

        Args:
            force_config_overwrite (bool): If True, all parameters are read from the config file. If False, only the parameters that are None are read from the config file. If you want to use the fcn function
            to set the parameters, you should set force_config_overwrite to False to avoid it being overwritten.
        """
        self.simulation_model._load_parameters(force_config_overwrite=force_config_overwrite)

    def load(self, 
             semantic_model_filename: Optional[str] = None,
             simulation_model_filename: Optional[str] = None,
             fcn: Optional[Callable] = None,
             draw_semantic_model: bool = True,
             draw_simulation_model: bool = True,
             verbose: bool = False,
             validate_model: bool = True,
             force_config_overwrite: bool = False) -> None:
        """
        Load and set up the model for simulation.

        Args:
            semantic_model_filename (Optional[str]): Path to the semantic model configuration file.
            fcn (Optional[Callable]): Custom function to be applied during model loading.
            draw_semantic_model (bool): Whether to create and save the object graph.
            draw_simulation_model (bool): Whether to create and save the system graph.
            verbose (bool): Whether to print verbose output during loading.
            validate_model (bool): Whether to perform model validation.
        """
        if verbose:
            self._load(semantic_model_filename=semantic_model_filename, 
                       simulation_model_filename=simulation_model_filename,
                       fcn=fcn, 
                       draw_semantic_model=draw_semantic_model, 
                       draw_simulation_model=draw_simulation_model,
                       verbose=verbose,
                       validate_model=validate_model, 
                       force_config_overwrite=force_config_overwrite)
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._load(semantic_model_filename=semantic_model_filename, 
                           simulation_model_filename=simulation_model_filename,
                           fcn=fcn,
                           draw_semantic_model=draw_semantic_model,
                           draw_simulation_model=draw_simulation_model,
                           verbose=verbose,
                           validate_model=validate_model, 
                           force_config_overwrite=force_config_overwrite)

    def _load(self, 
              semantic_model_filename: Optional[str] = None,
              simulation_model_filename: Optional[str] = None,
              fcn: Optional[Callable] = None, 
              draw_semantic_model: bool = True, 
              draw_simulation_model: bool = True,
              verbose: bool = False,
              validate_model: bool = True, 
              force_config_overwrite: bool = False) -> None:
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
        assert semantic_model_filename is None or simulation_model_filename is None, "Providing both semantic_model_filename and simulation_model_filename is currently not supported."

        # if self._is_loaded:
        #     warnings.warn("The model is already loaded. Resetting model.")
        #     self.reset()

        PRINTPROGRESS("Loading model")
        PRINTPROGRESS.add_level()
        # self.add_outdoor_environment()
        if semantic_model_filename is not None:
            apply_translator = True
            PRINTPROGRESS(f"Parsing semantic model", status="")
            self._semantic_model = core.SemanticModel(semantic_model_filename,
                                                               dir_conf=self.dir_conf + ["semantic_model"],
                                                               id=f"{self._id}_semantic_model")
            self._semantic_model.reason()
            if draw_semantic_model:
                PRINTPROGRESS(f"Drawing semantic model")
                self._semantic_model.visualize()

        else:
            apply_translator = False
        
        if apply_translator:
            PRINTPROGRESS(f"Applying translator")
            PRINTPROGRESS.add_level()
            self._translator = core.Translator()
            self._simulation_model = self._translator.translate(self._semantic_model)
            self._simulation_model.dir_conf = self.dir_conf + ["simulation_model"]
            PRINTPROGRESS.remove_level()

        
        self._simulation_model.load(
            rdf_file=simulation_model_filename,
            fcn=fcn,
            verbose=verbose, 
            validate_model=validate_model, 
            force_config_overwrite=force_config_overwrite)
        
        if draw_simulation_model:
            PRINTPROGRESS(f"Drawing simulation model")
            self._simulation_model.visualize()
        
        PRINTPROGRESS.remove_level()

        PRINTPROGRESS("Model loaded")
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
        # Reset all the dictionaries and lists
        self.simulation_model.reset()


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
    


    def get_semantic_object(self, key: str) -> "core.SemanticObject":
        """
        Get the semantic object for a given key.

        Args:
            key (str): The key of the component.

        Returns:
            core.System: The system component.

        Raises:
            AssertionError: If the mapping is not 1-to-1.
        """
        assert len(self._translator.sim2sem_map[self._simulation_model._components[key]])==1, f"The mapping for component \"{key}\" is not 1-to-1"
        return next(iter(self._translator.sim2sem_map[self._simulation_model._components[key]]))
    
    def serialize(self) -> None:
        """
        Serialize the model.
        """
        self._semantic_model.serialize()
        self._simulation_model.serialize()











    

