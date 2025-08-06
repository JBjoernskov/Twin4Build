# Standard library imports
import datetime
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

# Third party imports
import numpy as np
import pandas as pd
from prettytable import PrettyTable

# Local application imports
import twin4build.core as core
from twin4build.utils.mkdir_in_root import mkdir_in_root
from twin4build.utils.print_progress import PRINTPROGRESS


class Model:
    r"""
    A unified interface for building digital twin models.

    This class serves as a composed interface that integrates both simulation and semantic
    modeling capabilities for building digital twins. It combines the functionality of
    :class:`SimulationModel` and :class:`SemanticModel` into a single, user-friendly interface.

    **Composition Architecture:**

    The Model class acts as a facade that orchestrates two core components:

    1. **SimulationModel:** Handles the computational aspects including cycle removal
       (Algorithm 1), topological sorting (Algorithm 2), component management, and
       preparation for simulation execution.

    2. **SemanticModel:** Manages the ontological representation using SAREF4SYST,
       RDF graphs, semantic queries, and metadata for interoperability.

    **Key Responsibilities:**

    - **Unified Interface:** Provides a single entry point for both simulation and semantic operations
    - **Component Management:** Delegates component operations to the appropriate underlying model
    - **Model Lifecycle:** Orchestrates loading, validation, and execution preparation
    - **Data Integration:** Coordinates between semantic metadata and simulation execution
    - **Interoperability:** Ensures consistent SAREF-compliant representation across both models

    **Usage Pattern:**

    Users typically interact with this Model class rather than directly with SimulationModel
    or SemanticModel. The Model class automatically handles the coordination between the two
    underlying models, ensuring consistency and proper initialization order.

    Attributes
    ----------
    simulation_model : SimulationModel
        The underlying simulation model handling computational aspects and execution order.
    semantic_model : SemanticModel
        The underlying semantic model managing RDF graphs and ontological representations.
    components : Dict[str, System]
        Dictionary of all SAREF4SYST System components (delegated to simulation_model).
    execution_order : List[List[System]]
        Execution order determined by topological sorting (delegated to simulation_model).
    flat_execution_order : List[System]
        Flattened execution order for sequential processing (delegated to simulation_model).

    See Also
    --------
    SimulationModel : Detailed documentation on Algorithms 1-2, cycle removal, topological
                     sorting, component management, and simulation preparation
    SemanticModel : Detailed documentation on SAREF4SYST integration, RDF graph management,
                   semantic queries, and ontological operations
    Simulator : Algorithm 3 implementation for executing the prepared simulation model

    Examples
    --------
    Basic model creation and usage:

    >>> import twin4build as tb
    >>>
    >>> # Create unified model interface
    >>> model = tb.Model(id="building_model")
    >>>
    >>> # Add components (delegates to simulation_model)
    >>> space = tb.SpaceSystem(id="office_space")
    >>> heater = tb.SpaceHeaterSystem(id="radiator")
    >>> model.add_component(space)
    >>> model.add_component(heater)
    >>>
    >>> # Add connections (updates both simulation and semantic models)
    >>> model.add_connection(space, heater, "indoorTemperature", "zoneTemperature")
    >>>
    >>> # Load model (applies Algorithms 1-2, prepares semantic representation)
    >>> model.load()
    >>>
    >>> # Model is now ready for simulation or semantic queries
    >>> simulator = tb.Simulator(model)

    Working with semantic capabilities:

    >>> # Access semantic model directly when needed
    >>> model.semantic_model.visualize()  # Generate RDF graph visualization
    >>> model.semantic_model.serialize()  # Export to RDF format
    >>>
    >>> # Query semantic information
    >>> instances = model.semantic_model.get_instances_of_type("s4bldg:SpaceHeater")

    Working with simulation capabilities:

    >>> # Access simulation model directly when needed
    >>> print(f"Execution order: {model.simulation_model.execution_order}")
    >>> print(f"Components: {len(model.simulation_model.components)}")
    >>>
    >>> # Check if model is ready for simulation
    >>> if model.simulation_model._is_loaded:
    ...     simulator = tb.Simulator(model)
    ...     # Run simulation...

    Loading from RDF file:

    >>> # Load existing semantic model and convert to simulation model
    >>> model = tb.Model(id="restored_model")
    >>> model.load(rdf_file="my_building.ttl")
    >>> # Model now contains both semantic and simulation representations
    """

    __slots__ = (
        "_id",
        "_simulation_model",
        "_semantic_model",
        "_translator",
        "_dir_conf",
    )

    def __str__(self):
        t = PrettyTable(
            ["Number of components in simulation model: ", len(self.components)]
        )
        t.add_row(
            [
                "Number of connections in simulation model: ",
                self.simulation_model.count_connections(),
            ],
            divider=True,
        )
        title = f"Model overview    id: {self._id}"
        t.title = title
        t.add_row(["Model directory: ", self.get_dir()[0]], divider=True)
        t.add_row(
            [
                "Number of instances in semantic model: ",
                self.semantic_model.count_instances(),
            ],
            divider=True,
        )
        t.add_row(
            [
                "Number of triples in semantic model: ",
                self.semantic_model.count_triples(),
            ],
            divider=True,
        )
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
            cs = self.get_component_by_class(
                self.components, cls, filter=lambda v, class_: v.__class__ is class_
            )
            n = len(cs)
            for i, c in enumerate(cs):
                t.add_row([c.id, cls.__name__], divider=True if i == n - 1 else False)

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
        assert isinstance(id, str), f'Argument "id" must be of type {str(type(str))}'
        isvalid = np.array([x.isalnum() or x in valid_chars for x in id])
        np_id = np.array(list(id))
        violated_characters = list(np_id[isvalid == False])
        assert all(
            isvalid
        ), f"The model with id \"{id}\" has an invalid id. The characters \"{', '.join(violated_characters)}\" are not allowed."
        self._id = id
        self._dir_conf = ["generated_files", "models", self._id]

        self._semantic_model = core.SemanticModel(
            id=self._id,
            namespaces={
                "SIM": core.namespace.SIM,
                "SAREF": core.namespace.SAREF,
                "S4BLDG": core.namespace.S4BLDG,
                "S4SYST": core.namespace.S4SYST,
                "FSO": core.namespace.FSO,
            },
            dir_conf=self._dir_conf + ["semantic_model"],
        )
        self._simulation_model = core.SimulationModel(
            dir_conf=self.dir_conf + ["simulation_model"],
            id=f"{self._id}_simulation_model",
        )

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
    def dir_conf(self) -> List[str]:
        return self._dir_conf

    @dir_conf.setter
    def dir_conf(self, dir_conf: List[str]) -> None:
        assert isinstance(dir_conf, list) and all(
            isinstance(x, str) for x in dir_conf
        ), f"The set value must be of type {list} and contain strings"
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

    def get_dir(
        self, folder_list: List[str] = [], filename: Optional[str] = None
    ) -> Tuple[str, bool]:
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

    def add_component(self, component: "core.System") -> None:
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

    def remove_component(self, component: "core.System") -> None:
        """
        Remove a component from the model.

        Args:
            component (core.System): The component to remove.
        """
        self.simulation_model.remove_component(component=component)

    def add_connection(
        self,
        sender_component: "core.System",
        receiver_component: "core.System",
        outputPort: str,
        inputPort: str,
    ) -> None:
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
        self.simulation_model.add_connection(
            sender_component=sender_component,
            receiver_component=receiver_component,
            outputPort=outputPort,
            inputPort=inputPort,
        )

    def remove_connection(
        self,
        sender_component: "core.System",
        receiver_component: "core.System",
        outputPort: str,
        inputPort: str,
    ) -> None:
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
        self.simulation_model.remove_connection(
            sender_component=sender_component,
            receiver_component=receiver_component,
            outputPort=outputPort,
            inputPort=inputPort,
        )

    def get_component_by_class(
        self, dict_: Dict, class_: Type, filter: Optional[Callable] = None
    ) -> List:
        """
        Get components of a specific class from a dictionary.

        Args:
            dict_ (Dict): The dictionary to search.
            class_ (Type): The class to filter by.
            filter (Optional[Callable]): Additional filter function.

        Returns:
            List: List of components matching the class and filter.
        """
        return self.simulation_model.get_component_by_class(
            dict_=dict_, class_=class_, filter=filter
        )

    def set_custom_initial_dict(
        self, custom_initial_dict: Dict[str, Dict[str, Any]]
    ) -> None:
        """
        Set custom initial values for components.

        Args:
            custom_initial_dict (Dict[str, Dict[str, Any]]): Dictionary of custom initial values.

        Raises:
            AssertionError: If unknown component IDs are provided.
        """
        self.simulation_model.set_custom_initial_dict(
            custom_initial_dict=custom_initial_dict
        )

    def set_initial_values(self) -> None:
        """
        Set initial values for all components in the model.
        """
        self.simulation_model.set_initial_values()

    def set_parameters_from_array(
        self,
        values: List[Any],
        components: List["core.System"],
        parameter_names: List[str],
        normalized: List[bool] = None,
        overwrite: bool = False,
        save_original: bool = False,
    ) -> None:
        """
        Set parameters for components from an array.

        Args:
            values (List[Any]): List of parameter values.
            component_list (List[core.System]): List of components to set parameters for.
            attr_list (List[str]): List of attribute names corresponding to the parameters.

        Raises:
            AssertionError: If a component doesn't have the specified attribute.
        """
        self.simulation_model.set_parameters_from_array(
            values=values,
            components=components,
            parameter_names=parameter_names,
            normalized=normalized,
            overwrite=overwrite,
            save_original=save_original,
        )

    def restore_parameters(self, keep_values: bool = True) -> None:
        """
        Restore the parameters of the model.
        """
        self.simulation_model.restore_parameters(keep_values=keep_values)

    def cache(
        self,
        startTime: Optional[datetime.datetime] = None,
        endTime: Optional[datetime.datetime] = None,
        stepSize: Optional[int] = None,
        simulator: Optional["core.Simulator"] = None,
    ) -> None:
        """
        Cache data and create folder structure for time series data.

        Args:
            startTime (Optional[datetime.datetime]): Start time for caching.
            endTime (Optional[datetime.datetime]): End time for caching.
            stepSize (Optional[int]): Time step size for caching.
        """
        self.simulation_model.cache(
            startTime=startTime, endTime=endTime, stepSize=stepSize, simulator=simulator
        )

    def initialize(
        self,
        startTime: datetime.datetime,
        endTime: datetime.datetime,
        stepSize: int,
        simulator: "core.Simulator",
    ) -> None:
        """
        Initialize the model for simulation.

        Args:
            startTime (datetime.datetime): Start time for the simulation.
            endTime (datetime.datetime): End time for the simulation.
            stepSize (int): Time step size for the simulation.
            simulator (core.Simulator): Simulator instance.
        """
        self.simulation_model.initialize(startTime, endTime, stepSize, simulator)

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
        self.simulation_model._load_parameters(
            force_config_overwrite=force_config_overwrite
        )

    def load(
        self,
        semantic_model_filename: Optional[str] = None,
        simulation_model_filename: Optional[str] = None,
        fcn: Optional[Callable] = None,
        draw_semantic_model: bool = True,
        draw_simulation_model: bool = True,
        verbose: bool = False,
        validate_model: bool = True,
        force_config_overwrite: bool = False,
    ) -> None:
        """
        Load and set up the model for simulation.

        Args:
            semantic_model_filename (Optional[str]): Path to the semantic model configuration file.
            fcn (Optional[Callable]): Custom function to be applied during model loading.
            draw_semantic_model (bool): Whether to create and save the semantic model graph.
            draw_simulation_model (bool): Whether to create and save the simulation model graph.
            verbose (bool): Whether to print verbose output during loading.
            validate_model (bool): Whether to perform model validation.
        """
        if verbose:
            self._load(
                semantic_model_filename=semantic_model_filename,
                simulation_model_filename=simulation_model_filename,
                fcn=fcn,
                draw_semantic_model=draw_semantic_model,
                draw_simulation_model=draw_simulation_model,
                verbose=verbose,
                validate_model=validate_model,
                force_config_overwrite=force_config_overwrite,
            )
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._load(
                    semantic_model_filename=semantic_model_filename,
                    simulation_model_filename=simulation_model_filename,
                    fcn=fcn,
                    draw_semantic_model=draw_semantic_model,
                    draw_simulation_model=draw_simulation_model,
                    verbose=verbose,
                    validate_model=validate_model,
                    force_config_overwrite=force_config_overwrite,
                )

    def _load(
        self,
        semantic_model_filename: Optional[str] = None,
        simulation_model_filename: Optional[str] = None,
        fcn: Optional[Callable] = None,
        draw_semantic_model: bool = True,
        draw_simulation_model: bool = True,
        verbose: bool = False,
        validate_model: bool = True,
        force_config_overwrite: bool = False,
    ) -> None:
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
        assert (
            semantic_model_filename is None or simulation_model_filename is None
        ), "Providing both semantic_model_filename and simulation_model_filename is currently not supported."

        # if self._is_loaded:
        #     warnings.warn("The model is already loaded. Resetting model.")
        #     self.reset()

        PRINTPROGRESS("Loading model")
        PRINTPROGRESS.add_level()
        # self.add_outdoor_environment()
        if semantic_model_filename is not None:
            apply_translator = True
            PRINTPROGRESS("Parsing semantic model", status="")
            self._semantic_model = core.SemanticModel(
                semantic_model_filename,
                dir_conf=self.dir_conf + ["semantic_model"],
                id=f"{self._id}_semantic_model",
            )
            self._semantic_model.reason()
            if draw_semantic_model:
                PRINTPROGRESS("Drawing semantic model")
                self._semantic_model.visualize()

        else:
            apply_translator = False

        if apply_translator:
            PRINTPROGRESS("Applying translator")
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
            force_config_overwrite=force_config_overwrite,
        )

        if draw_simulation_model:
            PRINTPROGRESS("Drawing simulation model")
            self._simulation_model.visualize()

        PRINTPROGRESS.remove_level()

        PRINTPROGRESS("Model loaded")
        if verbose:
            print(self)

    def fcn(self) -> None:
        """
        Placeholder for a custom function to be applied during model loading.
        """

    def set_save_simulation_result(self, flag: bool = True, c: list = None):
        self.simulation_model.set_save_simulation_result(flag=flag, c=c)

    def reset(self) -> None:
        """
        Reset the model to its initial state.
        """
        # Reset all the dictionaries and lists
        self.simulation_model.reset()

    def load_estimation_result(
        self, filename: Optional[str] = None, result: Optional[Dict] = None
    ) -> None:
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
        assert (
            len(self._translator.sim2sem_map[self._simulation_model._components[key]])
            == 1
        ), f'The mapping for component "{key}" is not 1-to-1'
        return next(
            iter(self._translator.sim2sem_map[self._simulation_model._components[key]])
        )

    def serialize(self) -> None:
        """
        Serialize the model.
        """
        self._semantic_model.serialize()
        self._simulation_model.serialize()
