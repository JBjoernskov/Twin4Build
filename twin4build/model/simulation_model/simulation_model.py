from __future__ import annotations
import warnings
import os
import copy
import numpy as np
import datetime
import json
import pickle
from prettytable import PrettyTable
from twin4build.utils.print_progress import PrintProgress
import twin4build.systems.utils.fmu.fmu_component as fmu_component
from twin4build.utils.isnumeric import isnumeric
from twin4build.utils.get_object_attributes import get_object_attributes
from twin4build.utils.mkdir_in_root import mkdir_in_root
from twin4build.utils.rsetattr import rsetattr
from twin4build.utils.rgetattr import rgetattr
from twin4build.utils.rhasattr import rhasattr
from twin4build.utils.istype import istype
import twin4build.core as core
import twin4build.systems as systems
import twin4build.estimator.estimator as estimator
from typing import List, Dict, Any, Optional, Tuple, Type, Callable
from twin4build.utils.simple_cycle import simple_cycles
import twin4build.utils.input_output_types as tps
from rdflib import Namespace, Literal, RDF, RDFS

class SimulationModel:
    r"""
    A class representing a building system model.

    This class is responsible for creating, managing, and simulating a building system model.
    It handles component instantiation, connections between components, and execution order
    for simulation.

    Mathematical Formulation:

    1. Component Execution Order:
       The execution order is determined by a directed acyclic graph (DAG) of component dependencies:

       .. math::

           G = (V, E)

       where:
         - :math:`V` is the set of components
         - :math:`E` is the set of connections between components
         - Each edge :math:`(v_i, v_j) \in E` indicates that component :math:`v_i` must execute before :math:`v_j`

    2. Parameter Validation:
       For each parameter :math:`p` in component :math:`c`:

       .. math::

           p_{min} \leq p \leq p_{max}

       where:
         - :math:`p_{min}` is the minimum allowed value
         - :math:`p_{max}` is the maximum allowed value
         - Parameters must also satisfy any additional constraints defined by the component

    3. Connection Validation:
       For each connection between components :math:`c_i` and :math:`c_j`:

       .. math::

           \text{type}(c_i.\text{output}) \subseteq \text{type}(c_j.\text{input})

       where:
         - :math:`\text{type}(x)` is the data type of port :math:`x`
         - :math:`\subseteq` indicates type compatibility

    4. Component State Initialization:
       For each component :math:`c`:

       .. math::

           \mathbf{x}_c(0) = \mathbf{x}_c^{init}

       where:
         - :math:`\mathbf{x}_c(0)` is the initial state vector
         - :math:`\mathbf{x}_c^{init}` is the specified initial state

    5. Model Validation:
       The model is valid if:

       .. math::

           \forall c \in V: \text{valid}(c) \land \forall (c_i, c_j) \in E: \text{valid\_connection}(c_i, c_j)

       where:
         - :math:`\text{valid}(c)` checks if component :math:`c` is properly configured
         - :math:`\text{valid\_connection}(c_i, c_j)` checks if the connection between :math:`c_i` and :math:`c_j` is valid

    Attributes:
        id (str): Unique identifier for the model.
        components (dict): Dictionary of all components in the model.
        _execution_order (list): Ordered list of component groups for execution.
        _flat_execution_order (list): Flattened list of components in execution order.
    """

    __slots__ = (
        '_id', '_components',
        '_instance_map', 
        '_custom_initial_dict', '_execution_order', '_flat_execution_order',
        '_required_initialization_connections', '_components_no_cycles', '_is_loaded', "_is_validated", '_result',
        '_valid_chars', "_p", "_validated_for_simulator", "_validated_for_estimator",
        "_validated_for_evaluator", "_validated_for_monitor", "_dir_conf", "_connection_counter", "_semantic_model"
    )


    def __str__(self):
        t = PrettyTable(["Number of components in simulation model: ", self.count_components()])
        t.add_row(["Number of connections in simulation model: ", self.count_connections()], divider=True)
        title = f"Model overview    id: {self._id}"
        t.title = title
        t.add_row(["", ""])
        t.add_row(["", ""], divider=True)
        t.add_row(["id", "Class"], divider=True)
        unique_class_list = []
        for component in self._components.values():
            cls = component.__class__
            if cls not in unique_class_list:
                unique_class_list.append(cls)
        unique_class_list = sorted(unique_class_list, key=lambda x: x.__name__.lower())

        for cls in unique_class_list:
            cs = self.get_component_by_class(self._components, cls, filter=lambda v, class_: v.__class__ is class_)
            n = len(cs)
            for i,c in enumerate(cs):
                t.add_row([c.id, cls.__name__], divider=True if i==n-1 else False)
            
        return t.get_string()

    def __init__(self, id: str, dir_conf: List[str] = None) -> None:
        """
        Initialize the Model instance.

        Args:
            id (str): Unique identifier for the model.

        Raises:
            AssertionError: If the id is not a string or contains invalid characters.
        """
        self._id = id
        if dir_conf is None:
            self._dir_conf = ["generated_files", "models", self._id]
        else:
            self._dir_conf = dir_conf

        self._valid_chars = ["_", "-", " ", "(", ")", "[", "]"]
        assert isinstance(id, str), f"Argument \"id\" must be of type {str(type(str))}"
        isvalid = np.array([x.isalnum() or x in self._valid_chars for x in id])
        np_id = np.array(list(id))
        violated_characters = list(np_id[isvalid==False])
        assert all(isvalid), f"The model with id \"{id}\" has an invalid id. The characters \"{', '.join(violated_characters)}\" are not allowed."
        self._id = id
        self._components = {}
        self._custom_initial_dict = None
        self._is_loaded = False
        self._is_validated = False

        self._connection_counter = 0

        self._semantic_model = core.SemanticModel(id=self._id, 
                                                  namespaces={"SIM": core.SIM,
                                                              "SAREF": core.SAREF,
                                                              "S4BLDG": core.S4BLDG,
                                                              "S4SYST": core.S4SYST,
                                                              "FSO": core.FSO},
                                                   dir_conf=self._dir_conf + ["semantic_model"])

    @property
    def components(self) -> dict:
        return self._components

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
        return self._components

    @property
    def dir_conf(self) -> List[str]:
        return self._dir_conf

    @property
    def execution_order(self) -> List[str]:
        return self._execution_order
    
    @property
    def flat_execution_order(self) -> List[str]:
        return self._flat_execution_order
    
    @dir_conf.setter
    def dir_conf(self, dir_conf: List[str]) -> None:
        assert isinstance(dir_conf, list) and all(isinstance(x, str) for x in dir_conf), f"The set value must be of type {list} and contain strings"
        self._dir_conf = dir_conf
    
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
        assert isinstance(component, core.System), f"The argument \"component\" must be of type {core.System.__name__}"
        if component.id not in self._components:
            self._components[component.id] = component


    def make_pickable(self) -> None:
        """
        Make the model instance pickable by removing unpickable references.

        This method prepares the Model instance for use with multiprocessing, e.g. in the Estimator class.
        """
        fmus = self.get_component_by_class(self._components, fmu_component.FMUComponent)
        for fmu in fmus:
            if "fmu" in get_object_attributes(fmu):
                del fmu.fmu
                del fmu.fmu_initial_state
                fmu.INITIALIZED = False

    def remove_component(self, component: core.System) -> None:
        """
        Remove a component from the model.

        Args:
            component (core.System): The component to remove.
        """
        for connection in component.connectedThrough:
            for connection_point in connection.connectsSystemAt:
                connected_component = connection_point.connectionPointOf
                self.remove_connection(component, connected_component, connection.outputPort, connection_point.inputPort)
        
        del self._components[component.id]

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
        self.add_component(sender_component)
        self.add_component(receiver_component)

        found_connection_point = False
        # Check if there already is a connectionPoint with the same receiver_property_name
        for receiver_component_connection_point in receiver_component.connectsAt:
            if receiver_component_connection_point.inputPort == inputPort:
                found_connection_point = True
                break
        
        
        found_connection = False
        # Check if there already is a connection with the same sender_property_name
        for sender_obj_connection in sender_component.connectedThrough:
            if sender_obj_connection.outputPort == outputPort:
                found_connection = True
                break

        if found_connection_point and found_connection:
            message = f"core.Connection between \"{sender_component.id}\" and \"{receiver_component.id}\" with the properties \"{outputPort}\" and \"{inputPort}\" already exists."
            assert receiver_component_connection_point not in sender_obj_connection.connectsSystemAt, message
                    

        if found_connection==False:
            sender_obj_connection = core.Connection(connectsSystem=sender_component, outputPort=outputPort)
            sender_component.connectedThrough.append(sender_obj_connection)

        if found_connection_point==False:
            receiver_component_connection_point = core.ConnectionPoint(connectionPointOf=receiver_component, inputPort=inputPort)
            receiver_component.connectsAt.append(receiver_component_connection_point)
        
        sender_obj_connection.connectsSystemAt.append(receiver_component_connection_point)
        receiver_component_connection_point.connectsSystemThrough.append(sender_obj_connection)# if sender_obj_connection not in receiver_component_connection_point.connectsSystemThrough else None


        # Inputs and outputs of these classes can be set dynamically. Inputs and outputs of classes not in this tuple are set as part of their class definition.
        # exception_classes = (systems.TimeSeriesInputSystem,
        #                      systems.PiecewiseLinearSystem,
        #                     #  systems.PiecewiseLinearScheduleSystem,
        #                      systems.SensorSystem,
        #                      systems.MeterSystem,
        #                      systems.MaxSystem,
        #                      systems.NeuralPolicyControllerSystem)
        exception_classes = ()


        sender_component_uri = self._semantic_model.SIM.__getitem__(sender_component.id)
        receiver_component_uri = self._semantic_model.SIM.__getitem__(receiver_component.id)

        sender_component_class_name = sender_component.__class__.__name__
        receiver_component_class_name = receiver_component.__class__.__name__

        connection_uri = self._semantic_model.SIM.__getitem__(str(hash(sender_obj_connection)))#self._semantic_model.SIM.__getitem__(sender_component.id + " " + sender_property_name)
        connection_point_uri = self._semantic_model.SIM.__getitem__(str(hash(receiver_component_connection_point)))#self._semantic_model.SIM.__getitem__(receiver_component.id + " " + receiver_property_name)

        literal_sender_property = Literal(outputPort)
        literal_receiver_property = Literal(inputPort)

        # Add the class of the components to the semantic model
        self._semantic_model.graph.add((sender_component_uri, RDF.type, core.SIM.__getitem__(sender_component_class_name)))
        self._semantic_model.graph.add((receiver_component_uri, RDF.type, core.SIM.__getitem__(receiver_component_class_name)))

        self._semantic_model.graph.add((core.SIM.__getitem__(sender_component_class_name), RDFS.subClassOf, core.S4SYST.System))
        self._semantic_model.graph.add((core.SIM.__getitem__(receiver_component_class_name), RDFS.subClassOf, core.S4SYST.System))

        # Add the class of the connections and connection points to the semantic model
        self._semantic_model.graph.add((connection_uri, RDF.type, core.S4SYST.Connection))
        self._semantic_model.graph.add((connection_point_uri, RDF.type, core.S4SYST.ConnectionPoint))

        # Add the forward connection to the semantic model
        self._semantic_model.graph.add((sender_component_uri, core.S4SYST.connectedThrough, connection_uri))
        self._semantic_model.graph.add((connection_uri, core.S4SYST.connectsSystemAt, connection_point_uri))
        self._semantic_model.graph.add((connection_point_uri, core.S4SYST.connectionPointOf, receiver_component_uri))

        # Add the reverse connection to the semantic model
        self._semantic_model.graph.add((connection_uri, core.S4SYST.connectsSystem, sender_component_uri))
        self._semantic_model.graph.add((connection_point_uri, core.S4SYST.connectsSystemThrough, connection_uri))
        self._semantic_model.graph.add((receiver_component_uri, core.S4SYST.connectsAt, connection_point_uri))

        self._semantic_model.graph.add((connection_uri, core.SIM.outputPort, literal_sender_property))
        self._semantic_model.graph.add((connection_point_uri, core.SIM.inputPort, literal_receiver_property))
        
        if isinstance(sender_component, exception_classes):
            if outputPort not in sender_component.output:
                # If the property is not already an output, we assume it is a Scalar
                sender_component.output.update({outputPort: tps.Scalar()})
            else:
                pass
        else:
            message = f"The property \"{outputPort}\" is not a valid output for the component \"{sender_component.id}\" of type \"{type(sender_component)}\".\nThe valid output properties are: {','.join(list(sender_component.output.keys()))}"
            assert outputPort in (set(sender_component.input.keys()) | set(sender_component.output.keys())), message
        
        if isinstance(receiver_component, exception_classes):
            if inputPort not in receiver_component.input:
                # If the property is not already an input, we assume it is a Scalar
                receiver_component.input.update({inputPort: tps.Scalar()})
            else:
                assert isinstance(receiver_component.input[inputPort], tps.Vector), f"The input property \"{inputPort}\" for the component \"{receiver_component.id}\" of type \"{type(receiver_component)}\" is already set as a Scalar input."
        else:
            message = f"The property \"{inputPort}\" is not a valid input for the component \"{receiver_component.id}\" of type \"{type(receiver_component)}\".\nThe valid input properties are: {','.join(list(receiver_component.input.keys()))}"
            assert inputPort in receiver_component.input.keys(), message

        self._connection_counter += 1
        


    def remove_connection(self, sender_component: core.System, receiver_component: core.System, 
                          outputPort: str, inputPort: str) -> None:
        """
        Remove a connection between two components in the system.

        Args:
            sender_component (core.System): The component sending the connection.
            receiver_component (core.System): The component receiving the connection.
            sender_property_name (str): Name of the sender property.
            receiver_property_name (str): Name of the receiver property.

        Raises:
            ValueError: If the specified connection does not exist.
        """
        sender_component_connection = None
        for connection in sender_component.connectedThrough:
            if connection.outputPort == outputPort:
                sender_component_connection = connection
                break
        if sender_component_connection is None:
            raise ValueError(f"The sender component \"{sender_component.id}\" does not have a connection with the property \"{outputPort}\"")
        
        receiver_component_connection_point = None
        for connection_point in receiver_component.connectsAt:
            if connection_point.inputPort == inputPort:
                receiver_component_connection_point = connection_point
                break
        if receiver_component_connection_point is None:
            raise ValueError(f"The receiver component \"{receiver_component.id}\" does not have a connection point with the property \"{inputPort}\"")
        
        sender_component_connection.connectsSystemAt.remove(receiver_component_connection_point)
        receiver_component_connection_point.connectsSystemThrough.remove(sender_component_connection)

        if len(sender_component_connection.connectsSystemAt) == 0:
            sender_component.connectedThrough.remove(sender_component_connection)
            sender_component_connection.connectsSystem = None

        if len(receiver_component_connection_point.connectsSystemThrough) == 0:
            receiver_component.connectsAt.remove(receiver_component_connection_point)
            receiver_component_connection_point.connectionPointOf = None


        
        sender_component_uri = self._semantic_model.SIM.__getitem__(sender_component.id)
        receiver_component_uri = self._semantic_model.SIM.__getitem__(receiver_component.id)

        connection_uri = self._semantic_model.SIM.__getitem__(str(hash(sender_component_connection)))#self._semantic_model.SIM.__getitem__(sender_component.id + " " + sender_property_name)
        connection_point_uri = self._semantic_model.SIM.__getitem__(str(hash(receiver_component_connection_point)))#self._semantic_model.SIM.__getitem__(receiver_component.id + " " + receiver_property_name)

        literal_sender_property = list(self._semantic_model.graph.objects(connection_uri, core.SIM.outputPort))
        literal_receiver_property = list(self._semantic_model.graph.objects(connection_point_uri, core.SIM.inputPort))
        assert len(literal_sender_property)==1, "The connection has more than one output port."
        assert len(literal_receiver_property)==1, "The connection has more than one input port."
        literal_sender_property = literal_sender_property[0]
        literal_receiver_property = literal_receiver_property[0]

        # Remove the connections from the semantic model
        self._semantic_model.graph.remove((connection_uri, core.S4SYST.connectsSystemAt, connection_point_uri))
        self._semantic_model.graph.remove((connection_point_uri, core.S4SYST.connectsSystemThrough, connection_uri))

        if len(sender_component_connection.connectsSystemAt) == 0:
            self._semantic_model.graph.remove((sender_component_uri, core.S4SYST.connectedThrough, connection_uri))
            self._semantic_model.graph.remove((connection_uri, core.S4SYST.connectsSystem, sender_component_uri))
            self._semantic_model.graph.remove((connection_uri, core.SIM.outputPort, literal_sender_property))

        if len(receiver_component_connection_point.connectsSystemThrough) == 0:
            self._semantic_model.graph.remove((receiver_component_uri, core.S4SYST.connectsAt, connection_point_uri))
            self._semantic_model.graph.remove((connection_point_uri, core.S4SYST.connectionPointOf, receiver_component_uri))
            self._semantic_model.graph.remove((connection_point_uri, core.SIM.inputPort, literal_receiver_property))
        
        #Exception classes 
        exception_classes = ()
        if isinstance(sender_component, exception_classes):
            del sender_component.output[outputPort]

        if isinstance(receiver_component, exception_classes):
            del receiver_component.input[inputPort]

        self._connection_counter -= 1

    def count_components(self) -> int:
        return len(self._components)

    def count_connections(self) -> int:
        return self._semantic_model.count_triples(s=None, p=core.S4SYST.connectsSystemAt, o=None)


    def get_object_properties(self, object_: Any) -> Dict:
        """
        Get all properties of an object.

        Args:
            object_ (Any): The object to get properties from.

        Returns:
            Dict: A dictionary of object properties.
        """
        return {key: value for (key, value) in vars(object_).items()}
        
    def get_component_by_class(self, dict_: Dict, class_: Type, filter: Optional[Callable] = None) -> List:
        """
        Get components of a specific class from a dictionary.

        Args:
            dict_ (Dict): The dictionary to search.
            class_ (Type): The class to filter by.
            filter (Optional[Callable]): Additional filter function.

        Returns:
            List: List of components matching the class and filter.
        """
        if filter is None:
            filter = lambda v, class_: True
        return [v for v in dict_.values() if (isinstance(v, class_) and filter(v, class_))]

    
    def set_custom_initial_dict(self, _custom_initial_dict: Dict[str, Dict[str, Any]]) -> None:
        """
        Set custom initial values for components.

        Args:
            _custom_initial_dict (Dict[str, Dict[str, Any]]): Dictionary of custom initial values.

        Raises:
            AssertionError: If unknown component IDs are provided.
        """
        np_custom_initial_dict_ids = np.array(list(_custom_initial_dict.keys()))
        legal_ids = np.array([dict_id in self._components for dict_id in _custom_initial_dict])
        assert np.all(legal_ids), f"Unknown component id(s) provided in \"_custom_initial_dict\": {np_custom_initial_dict_ids[legal_ids==False]}"
        self._custom_initial_dict = _custom_initial_dict

    def set_initial_values(self) -> None:
        """
        Set initial values for all components in the model.
        """
        default_initial_dict = {
            systems.OutdoorEnvironmentSystem.__name__: {},
            # systems.OccupancySystem.__name__: {"scheduleValue": tps.Scalar(0)},
            systems.ScheduleSystem.__name__: {},
            systems.BuildingSpace0AdjBoundaryFMUSystem.__name__: {"indoorTemperature": tps.Scalar(21),
                                                        "indoorCo2Concentration": tps.Scalar(500)},
            systems.BuildingSpace0AdjBoundaryOutdoorFMUSystem.__name__: {"indoorTemperature": tps.Scalar(21),
                                                        "indoorCo2Concentration": tps.Scalar(500)},
            systems.BuildingSpace1AdjBoundaryFMUSystem.__name__: {"indoorTemperature": tps.Scalar(21),
                                                        "indoorCo2Concentration": tps.Scalar(500)},
            systems.BuildingSpace2SH1AdjBoundaryOutdoorFMUSystem.__name__: {"indoorTemperature": tps.Scalar(21),
                                                        "indoorCo2Concentration": tps.Scalar(500)},
            systems.BuildingSpace2AdjBoundaryFMUSystem.__name__: {"indoorTemperature": tps.Scalar(21),
                                                        "indoorCo2Concentration": tps.Scalar(500)},
            systems.BuildingSpace2AdjBoundaryOutdoorFMUSystem.__name__: {"indoorTemperature": tps.Scalar(21),
                                                        "indoorCo2Concentration": tps.Scalar(500)},
            systems.BuildingSpaceNoSH1AdjBoundaryOutdoorFMUSystem.__name__: {"indoorTemperature": tps.Scalar(21),
                                                        "indoorCo2Concentration": tps.Scalar(500)},
            systems.BuildingSpace1AdjBoundaryOutdoorFMUSystem.__name__: {"indoorTemperature": tps.Scalar(21),
                                                        "indoorCo2Concentration": tps.Scalar(500)},      
            systems.BuildingSpace11AdjBoundaryOutdoorFMUSystem.__name__: {"indoorTemperature": tps.Scalar(21),
                                                        "indoorCo2Concentration": tps.Scalar(500)},                                                                                   
            systems.PIControllerFMUSystem.__name__: {"inputSignal": tps.Scalar(0)},
            systems.RulebasedSetpointInputControllerSystem.__name__: {"inputSignal": tps.Scalar(0)},
            systems.ClassificationAnnControllerSystem.__name__: {"inputSignal": tps.Scalar(0)},
            systems.PIControllerFMUSystem.__name__: {"inputSignal": tps.Scalar(0)},
            systems.SequenceControllerSystem.__name__: {"inputSignal": tps.Scalar(0)},  
            systems.OnOffControllerSystem.__name__: {"inputSignal": tps.Scalar(0)},  
            systems.AirToAirHeatRecoverySystem.__name__: {"primaryTemperatureOut": tps.Scalar(21)},
            systems.CoilHeatingSystem.__name__: {"outletAirTemperature": tps.Scalar(21)},
            systems.CoilCoolingSystem.__name__: {},
            systems.CoilHeatingCoolingSystem.__name__: {"outletAirTemperature": tps.Scalar(21)},
            systems.DamperSystem.__name__: {"airFlowRate": tps.Scalar(0),
                                                "damperPosition": tps.Scalar(0)},
            systems.ValveSystem.__name__: {"waterFlowRate": tps.Scalar(0),
                                                "valvePosition": tps.Scalar(0)},
            systems.ValveFMUSystem.__name__: {"waterFlowRate": tps.Scalar(0),
                                                "valvePosition": tps.Scalar(0)},
            systems.FanSystem.__name__: {}, #Energy
            systems.FanFMUSystem.__name__: {"outletAirTemperature": tps.Scalar(21)}, #Energy
            systems.SpaceHeaterSystem.__name__: {#"outletWaterTemperature": tps.Vector(),
                                                    "Energy": tps.Scalar(0)},
            systems.SpaceHeaterStateSpace.__name__: {"Energy": tps.Scalar(0)},
            systems.SupplyFlowJunctionSystem.__name__: {"airFlowRateIn": tps.Scalar(0)},
            systems.ReturnFlowJunctionSystem.__name__: {"airFlowRateOut": tps.Scalar(0),
                                                           "airTemperatureOut": tps.Scalar(21)},
            systems.SensorSystem.__name__: {"measuredValue": tps.Scalar(0)},
            systems.ShadingDeviceSystem.__name__: {},
            systems.NeuralPolicyControllerSystem.__name__: {},
            systems.MeterSystem.__name__: {},
            systems.PiecewiseLinearSystem.__name__: {},
            # systems.PiecewiseLinearSupplyWaterTemperatureSystem.__name__: {},
            # systems.PiecewiseLinearScheduleSystem.__name__: {},
            systems.TimeSeriesInputSystem.__name__: {},
            systems.OnOffSystem.__name__: {},
            systems.BuildingSpaceStateSpace.__name__: {
                "indoorTemperature": tps.Scalar(21),
                "wallTemperature": tps.Scalar(21),
                "internalMassTemperature": tps.Scalar(21),
                "indoorCo2Concentration": tps.Scalar(500)
            }
        }
        initial_dict = {}
        for component in self._components.values():
            initial_dict[component.id] = {k: v.copy() for k, v in default_initial_dict[type(component).__name__].items()}
        if self._custom_initial_dict is not None:
            for key, value in self._custom_initial_dict.items():
                initial_dict[key].update(value)

        for component in self._components.values():
            component.output.update(initial_dict[component.id])

    def set_parameters_from_array(self, parameters: List[Any], component_list: List[core.System], attr_list: List[str]) -> None:
        """
        Set parameters for components from an array.

        Args:
            parameters (List[Any]): List of parameter values.
            component_list (List[core.System]): List of components to set parameters for.
            attr_list (List[str]): List of attribute names corresponding to the parameters.

        Raises:
            AssertionError: If a component doesn't have the specified attribute.
        """
        for i, (p, obj, attr) in enumerate(zip(parameters, component_list, attr_list)):
            assert rhasattr(obj, attr), f"The component with class \"{obj.__class__.__name__}\" and id \"{obj.id}\" has no attribute \"{attr}\"."
            rsetattr(obj, attr, p)

    def set_parameters_from_dict(self, parameters: Dict[str, Any], component_list: List[core.System], attr_list: List[str]) -> None:
        """
        Set parameters for components from a dictionary.

        Args:
            parameters (Dict[str, Any]): Dictionary of parameter values.
            component_list (List[core.System]): List of components to set parameters for.
            attr_list (List[str]): List of attribute names corresponding to the parameters.

        Raises:
            AssertionError: If a component doesn't have the specified attribute.
        """
        for (obj, attr) in zip(component_list, attr_list):
            assert rhasattr(obj, attr), f"The component with class \"{obj.__class__.__name__}\" and id \"{obj.id}\" has no attribute \"{attr}\"."
            rsetattr(obj, attr, parameters[attr])

    def cache(self, startTime: Optional[datetime.datetime] = None, endTime: Optional[datetime.datetime] = None, stepSize: Optional[int] = None) -> None:
        """
        Cache data and create folder structure for time series data.

        Args:
            startTime (Optional[datetime.datetime]): Start time for caching.
            endTime (Optional[datetime.datetime]): End time for caching.
            stepSize (Optional[int]): Time step size for caching.
        """
        c = self.get_component_by_class(self._components, (systems.SensorSystem, systems.MeterSystem, systems.OutdoorEnvironmentSystem, systems.TimeSeriesInputSystem))
        for component in c:
            component.initialize(startTime=startTime,
                                endTime=endTime,
                                stepSize=stepSize)
            
    def initialize(self,
                   startTime: Optional[datetime.datetime] = None,
                   endTime: Optional[datetime.datetime] = None,
                   stepSize: Optional[int] = None,
                   simulator: Optional[core.Simulator] = None) -> None:
        """
        Initialize the model for simulation.

        Args:
            startTime (Optional[datetime.datetime]): Start time for the simulation.
            endTime (Optional[datetime.datetime]): End time for the simulation.
            stepSize (Optional[int]): Time step size for the simulation.
        """
        # self.set_initial_values()
        self.check_for_for_missing_initial_values()
        for component in self._flat_execution_order:
            # component.clear_results()
            # component.initialize(startTime=startTime,
            #                     endTime=endTime,
            #                     stepSize=stepSize,
            #                     simulator=simulator)

            for v in component.input.values():
                v.reset()
                
            for v in component.output.values():
                v.reset()

            # Make the inputs and outputs aware of the execution order.
            # This is important to ensure that input tps.Vectors have the same order, allowing for instance element-wise operations.
            for i, connection_point in enumerate(component.connectsAt):
                for j, connection in enumerate(connection_point.connectsSystemThrough):
                    connected_component = connection.connectsSystem
                    if isinstance(component.input[connection_point.inputPort], tps.Vector):
                        if (component, connected_component, connection.outputPort, connection_point.inputPort) in self._translator.E_conn_to_sp_group:
                            sp, groups = self._translator.E_conn_to_sp_group[(component, connected_component, connection.outputPort, connection_point.inputPort)]
                            # Find the group of the connected component
                            modeled_match_nodes_ = self._translator.sim2sem_map[connected_component]
                            groups_matched = [g for g in groups if len(modeled_match_nodes_.intersection(set(g.values())))>0]
                            assert len(groups_matched)==1, "Only one group is allowed for each component."
                            group = groups_matched[0]
                            group_id = id(group)
                            component.input[connection_point.inputPort].update(group_id=group_id)
                        else:
                            component.input[connection_point.inputPort].update()

            component.initialize(startTime=startTime,
                                endTime=endTime,
                                stepSize=stepSize,
                                simulator=simulator)


    def _validate_model_definitions(self) -> None:
        """
        Validate the model definitions.

        This validation is not related to user inputs.
        It is related to programmatic errors.
        Therefore, we dont warn, but raise an error.
        """
        for component in self._components.values():
            for input in component.input.values():
                assert isinstance(input, (tps.Scalar, tps.Vector)), "Only vectors and scalars can be used as input to components"
            for output in component.output.values():
                assert isinstance(output, (tps.Scalar, tps.Vector)), "Only vectors and scalars can be used as output from components"

            if len(component.input.keys())==0:
                for key in component.output.keys():
                    output = component.output[key]
                    if isinstance(output, tps.Scalar): # TODO: Add support for vectors
                        assert output.is_leaf, f"|CLASS: {component.__class__.__name__}|ID: {component.id}|: The output \"{key}\" is not a leaf scalar. Only leaf scalars can be used as output from components with no inputs."

            else:
                for key in component.output.keys():
                    output = component.output[key]
                    if isinstance(output, tps.Scalar): # TODO: Add support for vectors
                        assert output.is_leaf==False, f"|CLASS: {component.__class__.__name__}|ID: {component.id}|: The output \"{key}\" is not a leaf scalar. Only non-leaf scalars can be used as output from components with inputs."

    def validate(self) -> None:
        """
        Validate the model by checking IDs and connections.
        """
        self._p.add_level()
        (validated_for_simulator1, validated_for_estimator1, validated_for_evaluator1, validated_for_monitor1) = self.validate_parameters()
        (validated_for_simulator2, validated_for_estimator2, validated_for_evaluator2, validated_for_monitor2) = self.validate_ids()
        (validated_for_simulator3, validated_for_estimator3, validated_for_evaluator3, validated_for_monitor3) = self.validate_connections()

        self._validated_for_simulator = validated_for_simulator1 and validated_for_simulator2 and validated_for_simulator3
        self._validated_for_estimator = validated_for_estimator1 and validated_for_estimator2 and validated_for_estimator3
        self._validated_for_evaluator = validated_for_evaluator1 and validated_for_evaluator2 and validated_for_evaluator3
        self._validated_for_monitor = validated_for_monitor1 and validated_for_monitor2 and validated_for_monitor3
        self._is_validated = self._validated_for_simulator and self._validated_for_estimator and self._validated_for_evaluator and self._validated_for_monitor
        self._p.remove_level()


        self._p("Validated for Simulator")
        if self._validated_for_simulator:
            status = "OK"
        else:
            status = "FAILED"
        
        self._p("Validated for Estimator", status=status)
        if self._validated_for_estimator:
            status = "OK"
        else:
            status = "FAILED"

        self._p("Validated for Evaluator", status=status)
        if self._validated_for_evaluator:
            status = "OK"
        else:
            status = "FAILED"

        self._p("Validated for Monitor", status=status)
        if self._validated_for_monitor:
            status = "OK"
        else:
            status = "FAILED"

        self._p("", plain=True, status=status)


        # assert validated, "The model is not valid. See the warnings above."

    def validate_parameters(self) -> None:
        """
        Validate the parameters of all components in the model.

        Raises:
            AssertionError: If any component has invalid parameters.
        """
        component_instances = list(self._components.values())
        _validated_for_simulator = True
        _validated_for_estimator = True
        _validated_for_evaluator = True
        _validated_for_monitor = True
        for component in component_instances:
            if hasattr(component, "validate"): #Check if component has validate method
                (validated_for_simulator_, validated_for_estimator_, validated_for_evaluator_, validated_for_monitor_) = component.validate(self._p)
                _validated_for_simulator = _validated_for_simulator and validated_for_simulator_
                _validated_for_estimator = _validated_for_estimator and validated_for_estimator_
                _validated_for_evaluator = _validated_for_evaluator and validated_for_evaluator_
                _validated_for_monitor = _validated_for_monitor and validated_for_monitor_
            else:
                config = component.config.copy()
                parameters = {attr: rgetattr(component, attr) for attr in config["parameters"]}
                is_none = [k for k,v in parameters.items() if v is None]
                if any(is_none):
                    message = f"|CLASS: {component.__class__.__name__}|ID: {component.id}|: Missing values for the following parameter(s) to enable use of Simulator, Evaluator, and Monitor:"
                    self._p(message, plain=True, status="[WARNING]")
                    self._p.add_level()
                    for par in is_none:
                        self._p(par, plain=True, status="")
                    self._p.remove_level()
                    # 
                    _validated_for_simulator = False
                    _validated_for_evaluator = False
                    _validated_for_monitor = False
        return (_validated_for_simulator, _validated_for_estimator, _validated_for_evaluator, _validated_for_monitor)
                
    def validate_ids(self) -> None:
        """
        Validate the IDs of all components in the model.

        Raises:
            AssertionError: If any component has an invalid ID.
        """
        validated = True
        component_instances = list(self._components.values())
        for component in component_instances:
            isvalid = np.array([x.isalnum() or x in self._valid_chars for x in component.id])
            np_id = np.array(list(component.id))
            violated_characters = list(np_id[isvalid==False])
            if not all(isvalid):
                message = f"|CLASS: {component.__class__.__name__}|ID: {component.id}|: Invalid id. The characters \"{', '.join(violated_characters)}\" are not allowed."
                self._p(message)
                validated = False
        return (validated, validated, validated, validated)


    def validate_connections(self) -> None:
        """
        Validate the connections between components in the model.

        Raises:
            AssertionError: If any required connections are missing.
        """
        component_instances = list(self._components.values())
        validated = True
        for component in component_instances:
            if len(component.connectedThrough)==0 and len(component.connectsAt)==0:
                warnings.warn(f"|CLASS: {component.__class__.__name__}|ID: {component.id}|: No connections. The component has been removed from the model.")
                self.remove_component(component)

            if hasattr(component, "optional_inputs"):
                optional_inputs = component.optional_inputs
            else:
                optional_inputs = []
            input_labels = [cp.inputPort for cp in component.connectsAt]
            first_input = True
            for req_input_label in component.input.keys():
                if req_input_label not in input_labels and req_input_label not in optional_inputs:
                    if first_input:
                        message = f"|CLASS: {component.__class__.__name__}|ID: {component.id}|: Missing connections for the following input(s) to enable use of Simulator, Estimator, Evaluator, and Monitor:"
                        self._p(message, plain=True, status="[WARNING]")
                        first_input = False
                        self._p.add_level()
                    self._p(req_input_label, plain=True)
                    validated = False
            if first_input==False:
                self._p.remove_level()
        return (validated, validated, validated, validated)

    def _load_parameters(self, force_config_update: bool = False) -> None:
        """
        Load parameters for all components from configuration files.

        Args:
            force_config_update (bool): If True, all parameters are read from the config file. If False, only the parameters that are None are read from the config file. If you want to use the fcn function
            to set the parameters, you should set force_config_update to False to avoid it being overwritten.
        """
        def extract_value(value):
            if hasattr(value, 'detach') and hasattr(value, 'numpy'):
                return float(value.detach().numpy())
            else:# isinstance(value, (int, float, type(None))):
                return value

        for component in self._components.values():
            assert hasattr(component, "config"), f"The class \"{component.__class__.__name__}\" has no \"config\" attribute."
            config = component.config.copy()
            assert "parameters" in config, f"The \"config\" attribute of class \"{component.__class__.__name__}\" has no \"parameters\" key."
            filename, isfile = self.get_dir(folder_list=["model_parameters", component.__class__.__name__], filename=f"{component.id}.json")
            config["parameters"] = {attr: extract_value(rgetattr(component, attr)) for attr in config["parameters"]}
            
            if isfile==False:
                with open(filename, 'w') as f:
                    json.dump(config, f, indent=4)
            else:
                with open(filename) as f:
                    config = json.load(f)

                if force_config_update:
                    attr_list = [k for k in config["parameters"].keys()]
                    component_list = [component for k in config["parameters"].keys()]
                else:
                    attr_list = [k for k in config["parameters"].keys() if rgetattr(component, k) is None]
                    component_list = [component for k in config["parameters"].keys() if rgetattr(component, k) is None]
                
                parameters = {k: float(v) if isnumeric(v) else v for k, v in config["parameters"].items()}
                self.set_parameters_from_dict(parameters, component_list, attr_list)

                if "readings" in config:
                    filename_ = config["readings"]["filename"]
                    datecolumn = config["readings"]["datecolumn"]
                    valuecolumn = config["readings"]["valuecolumn"]
                    if filename_ is not None:
                        component.filename = filename_

                        if datecolumn is not None:
                            component.datecolumn = datecolumn
                        elif isinstance(component, systems.OutdoorEnvironmentSystem)==False:
                            raise(ValueError(f"\"datecolumn\" is not defined in the \"readings\" key of the config file: {filename}"))

                        if valuecolumn is not None:
                            component.valuecolumn = valuecolumn
                        elif isinstance(component, systems.OutdoorEnvironmentSystem)==False:
                            raise(ValueError(f"\"valuecolumn\" is not defined in the \"readings\" key of the config file: {filename}"))

    def load(self, 
             fcn: Optional[Callable] = None, 
             verbose: bool = False, 
             validate_model: bool = True, 
             force_config_update: bool = False) -> None:
        """
        Load and set up the model for simulation.

        Args:
            semantic_model_filename (Optional[str]): Path to the semantic model configuration file.
            input_config (Optional[Dict]): Input configuration dictionary.
            fcn (Optional[Callable]): Custom function to be applied during model loading.
            create_object_graph (bool): Whether to create and save the object graph.
            create_signature_graphs (bool): Whether to create and save signature graphs.
            create_system_graph (bool): Whether to create and save the system graph.
            verbose (bool): Whether to print verbose output during loading.
            validate_model (bool): Whether to perform model validation.
        """
        if verbose:
            self._load(fcn=fcn,
                       validate_model=validate_model, 
                       force_config_update=force_config_update)
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._load(fcn=fcn,
                           validate_model=validate_model,
                           force_config_update=force_config_update)

    def _load(self, 
             fcn: Optional[Callable] = None, 
             validate_model: bool = True, 
             force_config_update: bool = False) -> None:
        """
        Internal method to load and set up the model for simulation.

        This method is called by load and performs the actual loading process.

        Args:
            fcn (Optional[Callable]): Custom function to be applied during model loading.
            validate_model (bool): Whether to perform model validation.
        """

        if self._is_loaded:
            warnings.warn("The model is already loaded. Resetting model.")
            self.reset()

        self._is_loaded = True

        self._p = PrintProgress()
        self._p("Loading model")
        self._p.add_level()

        if fcn is not None:
            assert callable(fcn), "The function to be applied during model loading is not callable."
            self._p(f"Applying user defined function")
            fcn(self)
        
        self._validate_model_definitions()

        self._p("Removing cycles")
        self._get_components_no_cycles()

        self._p("Determining execution order")
        self._get_execution_order()

        self._p("Loading parameters")
        self._load_parameters(force_config_update=force_config_update)

        if validate_model:
            self._p("Validating model")
            self.validate()
        self._p()
        print(self)

    def set_save_simulation_result(self, flag: bool=True, c: list=None):
        assert isinstance(flag, bool), "The flag must be a boolean."
        if c is not None:
            assert isinstance(c, list), "The c must be a list."
            for component in c:
                for input_key in component.input.keys():
                    if isinstance(component.input[input_key], tps.Scalar):
                        component.input[input_key].save_history = flag
                for output_key in component.output.keys():
                    if isinstance(component.output[output_key], tps.Scalar):
                        component.output[output_key].save_history = flag
        else:
            for component in self._components.values():
                for input_key in component.input.keys():
                    if isinstance(component.input[input_key], tps.Scalar):
                        component.input[input_key].save_history = flag
                for output_key in component.output.keys():
                    if isinstance(component.output[output_key], tps.Scalar):
                        component.output[output_key].save_history = flag


    def reset(self) -> None:
        """
        Reset the model to its initial state.
        """
        # Reset all the dictionaries and lists
        # self._components = {} ###
        self._custom_initial_dict = None ###
        self._execution_order = [] ###
        self._flat_execution_order = [] ###
        self._required_initialization_connections = [] ###
        self._components_no_cycles = {} ###

        # Reset the loaded state
        self._is_loaded = False ###
        self._is_validated = False ###

        # Reset any estimation results
        self._result = None ###

    def get_simple_graph(self, components) -> Dict:
        """
        Get a simple graph representation of the system graph.

        Returns:
            Dict: The simple graph representation.
        """
        simple_graph = {c: [] for c in components.values()}
        for component in components.values():
            for connection in component.connectedThrough:
                for connection_point in connection.connectsSystemAt:
                    receiver_component = connection_point.connectionPointOf
                    simple_graph[component].append(receiver_component)
        return simple_graph

    def get_simple_cycles(self, components: Dict) -> List[List[core.System]]:
        """
        Get the simple cycles in the system graph.

        Args:
            components (Dict): Dictionary of components.

        Returns:
            List[List[core.System]]: List of simple cycles.
        """
        G = self.get_simple_graph(components)
        cycles = simple_cycles(G)
        return cycles

    def _copy_components(self) -> core.System:
        """
        Copy the components of the model.
        """
        new_to_old_mapping = {}
        old_to_new_mapping = {}
        for component in self._components.values():
            if component not in old_to_new_mapping:
                new_component = copy.copy(component)
                new_component.connectedThrough = []
                new_component.connectsAt = []
                new_to_old_mapping[new_component] = component
                old_to_new_mapping[component] = new_component
            else:
                new_component = old_to_new_mapping[component]

            for connection in component.connectedThrough:
                for connection_point in connection.connectsSystemAt:
                    connected_component = connection_point.connectionPointOf
                    if connected_component not in old_to_new_mapping:
                        new_connected_component = copy.copy(connected_component)
                        new_connected_component.connectedThrough = []
                        new_connected_component.connectsAt = []
                        new_to_old_mapping[new_connected_component] = connected_component
                        old_to_new_mapping[connected_component] = new_connected_component
                    else:
                        new_connected_component = old_to_new_mapping[connected_component]
                    self.add_connection(new_component, new_connected_component, connection.outputPort, connection_point.inputPort)

        _new_components = {k: old_to_new_mapping[v] for k, v in self._components.items()}
        return _new_components


    def _get_components_no_cycles(self) -> None:
        """
        Create a dictionary of components without cycles.
        """
        self._components_no_cycles = self._copy_components()
        cycles = self.get_simple_cycles(self._components_no_cycles)
        self._required_initialization_connections = []
        for cycle in cycles:
            # c_from = [(i, c) for i, c in enumerate(cycle) if isinstance(c, core.Controller)] # TODO: Should there be a Controller superclass
            # For now, dont treat controller any different from other components.
            c_from = []
            if len(c_from)==1:
                idx = c_from[0][0]
                c_from = c_from[0][1]
            else:
                idx = 0
                c_from = cycle[0]

            if idx==len(cycle)-1:
                c_to = cycle[0]
            else:
                c_to = cycle[idx+1]


            for connection in c_from.connectedThrough.copy():
                for connection_point in connection.connectsSystemAt.copy():
                    if c_to==connection_point.connectionPointOf:
                        connection.connectsSystemAt.remove(connection_point)
                        connection_point.connectsSystemThrough.remove(connection)
                        self._required_initialization_connections.append(connection)

                        if len(connection_point.connectsSystemThrough)==0:
                            c_to.connectsAt.remove(connection_point)
                
                if len(connection.connectsSystemAt)==0:
                    c_from.connectedThrough.remove(connection)

    def load_estimation_result(self, filename: Optional[str] = None, result: Optional[Dict] = None) -> None:
        """
        Load a chain log from a file or dictionary.

        Args:
            filename (Optional[str]): The filename to load the chain log from.
            result (Optional[Dict]): The chain log dictionary to load.

        Raises:
            AssertionError: If invalid arguments are provided.
        """
        if result is not None:
            assert isinstance(result, dict), "Argument d must be a dictionary"
            cls_ = result.__class__
            self._result = cls_()
            for key, value in result.items():
                if "chain." not in key:
                    self._result[key] = copy.deepcopy(value)
                else:
                    self._result[key] = value
        else:
            assert isinstance(filename, str), "Argument filename must be a string"
            _, ext = os.path.splitext(filename)
            if ext==".pickle":
                with open(filename, 'rb') as handle:
                    self._result = pickle.load(handle)
                    
            elif ext==".npz":
                if "_ls.npz" in filename:
                    d = dict(np.load(filename, allow_pickle=True))
                    d = {k.replace(".", "_"): v for k,v in d.items()} # For backwards compatibility
                    self._result = estimator.LSEstimationResult(**d)
                elif "_mcmc.npz" in filename:
                    d = dict(np.load(filename, allow_pickle=True))
                    d = {k.replace(".", "_"): v for k,v in d.items()} # For backwards compatibility
                    self._result = estimator.MCMCEstimationResult(**d)
                else:
                    raise Exception(f"The estimation result file is not of a supported type. The file must be a .pickle, .npz file with the name containing \"_ls\" or \"_mcmc\".")
                

                for key, value in self._result.items():
                    self._result[key] = 1/self._result["chain_betas"] if key=="chain_T" else value
                    if self._result[key].size==1 and (len(self._result[key].shape)==0 or len(self._result[key].shape)==1):
                        self._result[key] = value.tolist()

                    elif key=="startTime_train" or key=="endTime_train" or key=="stepSize_train":
                        self._result[key] = value.tolist()
            else:
                raise Exception(f"The estimation result is of type {type(self._result)}. This type is not supported by the model class.")

        if isinstance(self._result, estimator.LSEstimationResult):
            theta = self._result["result_x"]
        elif isinstance(self._result, estimator.MCMCEstimationResult):
            parameter_chain = self._result["chain_x"][:,0,:,:]
            parameter_chain = parameter_chain.reshape((parameter_chain.shape[0]*parameter_chain.shape[1], parameter_chain.shape[2]))
            best_index = np.argmax(self._result["chain_logl"], axis=0)[0][0]
            theta = parameter_chain[best_index]
        else:
            raise Exception(f"The estimation result is of type {type(self._result)}. This type is not supported by the model class.")

        flat_component_list = [self._components[com_id] for com_id in self._result["component_id"]]
        flat_attr_list = self._result["component_attr"]
        theta_mask = self._result["theta_mask"]
        theta = theta[theta_mask]
        self.set_parameters_from_array(theta, flat_component_list, flat_attr_list)

    def check_for_for_missing_initial_values(self) -> None:
        """
        Check for missing initial values in components.

        Raises:
            Exception: If any component is missing an initial value.
        """
        for connection in self._required_initialization_connections:
            component = connection.connectsSystem
            if connection.outputPort not in component.output:
                raise Exception(f"The component with id: \"{component.id}\" and class: \"{component.__class__.__name__}\" is missing an initial value for the output: {connection.outputPort}")
            elif component.output[connection.outputPort].get() is None:
                raise Exception(f"The component with id: \"{component.id}\" and class: \"{component.__class__.__name__}\" is missing an initial value for the output: {connection.outputPort}")
                
    def _get_execution_order(self) -> None:
        """
        Determine the execution order of components.

        Raises:
            AssertionError: If cycles are detected in the model.
        """
        def _flatten(_list: List) -> List:
            """
            Flatten a nested list.

            Args:
                _list (List): The nested list to flatten.

            Returns:
                List: The flattened list.
            """
            return [item for sublist in _list for item in sublist]

        def _traverse(self, activeComponents) -> None:
            """
            Traverse the component graph to determine execution order.
            """
            activeComponentsNew = []
            component_group = []
            for component in activeComponents:
                component_group.append(component)
                for connection in component.connectedThrough:
                    for connection_point in connection.connectsSystemAt:
                        # connection_point = connection.connectsSystemAt
                        receiver_component = connection_point.connectionPointOf
                        connection_point.connectsSystemThrough.remove(connection)
                        if len(connection_point.connectsSystemThrough)==0:
                            receiver_component.connectsAt.remove(connection_point)

                        if len(receiver_component.connectsAt)==0:
                            activeComponentsNew.append(receiver_component)
            activeComponents = activeComponentsNew
            self._execution_order.append(component_group)
            return activeComponents

        initComponents = [v for v in self._components_no_cycles.values() if len(v.connectsAt)==0]
        activeComponents = initComponents
        self._execution_order = []
        while len(activeComponents)>0:
            activeComponents = _traverse(self, activeComponents)

        # Map the execution order from the no cycles component dictionary to the full component dictionary.
        self._execution_order = [[self._components[component.id] for component in component_group] for component_group in self._execution_order]

        # Map required initialization connections from the no cycles component dictionary to the full component dictionary.
        self._required_initialization_connections = [connection for no_cycle_connection in self._required_initialization_connections for connection in self._components[no_cycle_connection.connectsSystem.id].connectedThrough if connection.outputPort==no_cycle_connection.outputPort]

        self._flat_execution_order = _flatten(self._execution_order)
        assert len(self._flat_execution_order)==len(self._components_no_cycles), f"Cycles detected in the model. Inspect the generated file \"system_graph.png\" to see where."

    
    def visualize(self, query: str = None) -> None:
        """
        Visualize the simulation model.
        """
        self._semantic_model.visualize(query)
