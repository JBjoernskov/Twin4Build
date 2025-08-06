from __future__ import annotations

# Standard library imports
import copy
import datetime
import json
import os
import pickle
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

# Third party imports
import numpy as np
import torch
import torch.nn.parameter
from prettytable import PrettyTable
from rdflib import RDF, RDFS, Literal, Namespace

# Local application imports
import twin4build.core as core
import twin4build.estimator.estimator as estimator
import twin4build.systems as systems
import twin4build.utils.types as tps
from twin4build.model.semantic_model.semantic_model import get_short_name
from twin4build.utils.dict_utils import (
    compare_dict_structure,
    flatten_dict,
    merge_dicts,
)
from twin4build.utils.get_object_attributes import get_object_attributes
from twin4build.utils.isnumeric import isnumeric
from twin4build.utils.istype import istype
from twin4build.utils.mkdir_in_root import mkdir_in_root
from twin4build.utils.print_progress import PRINTPROGRESS, PrintProgress
from twin4build.utils.rdelattr import rdelattr
from twin4build.utils.rgetattr import rgetattr
from twin4build.utils.rhasattr import rhasattr
from twin4build.utils.rsetattr import rsetattr
from twin4build.utils.simple_cycle import simple_cycles

INVALID_ID_CHARS = ["_", "-", " ", "(", ")", "[", "]"]


class SimulationModel:
    r"""
    A simulation model for building digital twins.

    This class manages component collections, connections between components, cycle removal
    for feedback control loops, and topological sorting to determine optimal execution
    order for simulation.

    Mathematical Formulation:
    =========================

    The simulation model preparation process involves two main steps: cycle removal and
    topological sorting to create an executable simulation sequence.

    Component Dependency Graph:
    ---------------------------

    The simulation model can be represented as a directed multigraph :math:`G = (V, E, \iota)` comprising:

    .. math::

        V = \{c_1, c_2, ..., c_n\}

    .. math::

        E = \{e_1, e_2, e_3, ...\}

    .. math::

        \iota: E \rightarrow V \times V

    where:
        - :math:`V` is the set of vertices (components)
        - :math:`E` is the set of edge identifiers (connections between components)
        - :math:`\iota` is the incidence function mapping edges to vertex pairs
        - Each edge :math:`e_i \in E` with :math:`\iota(e_i) = (c_j, c_k)` indicates that component :math:`c_j` provides input to component :math:`c_k`
        - Multiple edges can map to the same vertex pair (multigraph): :math:`\iota(e_i) = \iota(e_j) = (c_p, c_q)`

    Optimized Cycle Removal Process:
    --------------------------------

    To find the execution order of the components (i.e. the topologically sorted order), we need to remove cycles from the dependency graph first.
    Such cycles can arise in the simulation model due to different reasons, e.g. modeling of feedback control loops or mutual dependencies between components (model a requires an input from model b and model b requires an input from model a).

    The optimized cycle removal process uses a greedy algorithm that minimizes the total number of edges removed:

    1. **Cycle Detection:** Identify the set of simple cycles :math:`\mathcal{C} = \{C_1, C_2, ..., C_m\}` in the graph where we can write one simple cycle as a sequence of edges :math:`C = ((c_1, c_2), (c_2, c_3), ..., (c_k, c_1))`,
    i.e. the cycle starts and ends at the same component and can't visit any other component more than once.

    2. **Edge Participation Analysis:** For each edge :math:`e \in E`, count its participation in cycles:

       .. math::

           p(e) = |\{C \in \mathcal{C} \; | \; e \in C\}|

       This gives the number of cycles that edge :math:`e` participates in.

    3. **Greedy Edge Selection:** Select the edge that participates in the maximum number of cycles:

       .. math::

           e^* = \underset{e \in E}{\operatorname{argmax}} \; p(e)

    4. **Iterative Removal:** Remove the selected edge and repeat until no cycles remain:

       .. math::

           E_{k+1} = E_k \setminus \{e^*_k\}

       where :math:`e^*_k` is the optimal edge selected at iteration :math:`k`.

    The process terminates when :math:`G_{final} = (V, E_{final})` is acyclic:

       .. math::

           E_{acyclic} = E_{final}, \quad \mathcal{C}(G_{final}) = \emptyset



    All removed edges become required initialization connections:

       .. math::

           E_{init} = E \setminus E_{acyclic}

       This means, for all :math:`(c_i, c_j) \in E_{init}`, :math:`c_j` must have initial values provided.


    Topological Sorting Process:
    -----------------------------

    After cycle removal, we need to find a topological ordering of the acyclic graph :math:`G_{acyclic} = (V, E_{acyclic})`.
    A topological ordering is a linear arrangement of vertices :math:`L` such that for every directed edge :math:`(c_i, c_j) \in E_{acyclic}`,
    component :math:`c_i` appears before component :math:`c_j` in the ordering. In practical terms, this means when executing component :math:`c_j`,
    all components :math:`c_i` that provides inputs to :math:`c_j` must have already been executed.

    The goal is to determine an execution sequence:

    .. math::

        L = (c_1, c_2, ..., c_n)

    And a priority level for each component:

    .. math::

        P = (p_1, p_2, ..., p_n)

    where:

        - Each :math:`L_p` contains components that can execute at priority level :math:`p`
        - Components with the same priority level can execute in parallel (no dependencies between them)

    All of the above prepares the model for simulation and is done when the :meth:`load` method is called.


    Attributes:
        id (str): Unique identifier for the model.
        components (dict): Dictionary of all components in the model.
        _execution_order (list): Ordered list of component groups for execution.
        _flat_execution_order (list): Flattened list of components in execution order.
        _components_no_cycles (dict): Copy of components with cycles removed.
        _required_initialization_connections (list): Connections that require initial values.

    See Also:
        :class:`twin4build.simulator.simulator.Simulator`: Handles simulation execution using the prepared execution order

    References:
        The methodology is based on: "An Ontology-based Innovative Energy Modeling
        Framework for Scalable and Adaptable Building Digital Twins" by Bjørnskov & Jradi.
        This class implements the optimized cycle removal and topological sorting procedures.

    Examples:
        Basic model setup and preparation:

        >>> model = SimulationModel(id="building_model")
        >>> # Create components
        >>> schedule = tb.ScheduleSystem(id="schedule")
        >>> damper = tb.DamperTorchSystem(id="damper")
        >>> # Add components to model
        >>> model.add_component(schedule)
        >>> model.add_component(damper)
        >>> # Connect schedule output to damper input
        >>> model.add_connection(schedule, damper, "scheduleValue", "damperPosition")
        >>> # Apply optimized cycle removal and topological sorting during model loading
        >>> model.load()
        >>> # Model is now ready for simulation with Simulator class
        >>> # Execution order and cycle-free structure are prepared with minimal edge removal
    """

    __slots__ = (
        "_id",
        "_components",
        "_saved_parameters",
        "_custom_initial_dict",
        "_execution_order",
        "_flat_execution_order",
        "_required_initialization_connections",
        "_components_no_cycles",
        "_is_loaded",
        "_is_validated",
        "_result",
        "_validated_for_simulator",
        "_validated_for_estimator",
        "_validated_for_optimizer",
        "_validated_for_monitor",
        "_dir_conf",
        "_semantic_model",
    )

    def __str__(self):
        t = PrettyTable(
            ["Number of components in simulation model: ", self.count_components()]
        )
        t.add_row(
            ["Number of connections in simulation model: ", self.count_connections()],
            divider=True,
        )
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
            cs = self.get_component_by_class(
                self._components, cls, filter=lambda v, class_: v.__class__ is class_
            )
            n = len(cs)
            for i, c in enumerate(cs):
                t.add_row([c.id, cls.__name__], divider=True if i == n - 1 else False)

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

        assert isinstance(id, str), f'Argument "id" must be of type {str(type(str))}'
        isvalid = np.array([x.isalnum() or x in INVALID_ID_CHARS for x in id])
        np_id = np.array(list(id))
        violated_characters = list(np_id[isvalid == False])
        assert all(
            isvalid
        ), f"The model with id \"{id}\" has an invalid id. The characters \"{', '.join(violated_characters)}\" are not allowed."
        self._id = id
        self._components = {}
        self._saved_parameters = {}
        self._custom_initial_dict = None
        self._is_loaded = False
        self._is_validated = False

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

    @property
    def components(self) -> dict:
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
        assert isinstance(dir_conf, list) and all(
            isinstance(x, str) for x in dir_conf
        ), f"The set value must be of type {list} and contain strings"
        self._dir_conf = dir_conf

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

    def add_component(
        self, component: core.System, components: Dict[str, core.System] = None
    ) -> None:
        """
        Add a component to the model.

        Args:
            component (core.System): The component to add.

        Raises:
            AssertionError: If the component is not an instance of core.System.
        """
        assert isinstance(
            component, core.System
        ), f'The argument "component" must be of type {core.System.__name__}'
        if components is None:
            components = self._components

        if component.id not in components:
            components[component.id] = component
        else:
            assert (
                components[component.id] == component
            ), f'The component with id "{component.id}" already exists in the model.'

        if components == self._components:
            self._update_literals(component)

    def make_pickable(self) -> None:
        """
        Make the model instance pickable by removing unpickable references.

        This method prepares the Model instance for use with multiprocessing, e.g. in the Estimator class.
        """
        # for c in self._components.values():
        #     print(f"Making {c.id} pickable")
        #     for k, input_ in c.input.items():
        #         print(f"Making {k} of pickable")
        #         input_.make_pickable()
        #     for k, output_ in c.output.items():
        #         print(f"Making {k} of pickable")
        #         output_.make_pickable()

        self.reset_torch_tensors()

        fmus = self.get_component_by_class(self._components, systems.fmuSystem)
        for fmu in fmus:
            if "fmu" in get_object_attributes(fmu):
                del fmu.fmu
                del fmu.fmu_initial_state
                fmu.INITIALIZED = False

    def reset_torch_tensors(self) -> None:
        """
        Reset all torch.Tensor objects in the model to remove TensorWrapper references.

        This method iterates through all components and their attributes to find torch.Tensor
        objects that might contain TensorWrapper (which causes pickling issues). It creates
        new tensors with the same values but without gradient tracking.

        This is particularly useful when switching from AD (automatic differentiation) to
        FD (finite difference) methods in the Estimator, as AD methods create gradient-tracking
        tensors that cannot be pickled for multiprocessing.
        """

        def reset_tensor(tensor):
            """
            Reset a torch tensor if it contains TensorWrapper or has gradient tracking.

            Args:
                tensor: The tensor to check and potentially reset
                path: Path for debugging purposes

            Returns:
                The original tensor or a new tensor without gradient tracking
            """
            assert isinstance(
                tensor, torch.Tensor
            ), f"The tensor must be of type {torch.Tensor.__name__}"

            # First handle special cases
            if isinstance(tensor, tps.Parameter):
                tensor = tps.Parameter(
                    tensor.get(),
                    min_value=tensor._min_value,
                    max_value=tensor._max_value,
                    requires_grad=False,
                )
            elif isinstance(tensor, tps.TensorParameter):
                tensor = tps.TensorParameter(
                    tensor.get(),
                    min_value=tensor._min_value,
                    max_value=tensor._max_value,
                    normalized=False,
                )
            elif isinstance(tensor, torch.Tensor):
                tensor = torch.tensor(tensor, dtype=torch.float64, requires_grad=False)

            return tensor

        def reset_object_tensors(obj, obj_path="", visited=None):
            """
            Recursively reset tensors in an object and its attributes.

            Args:
                obj: The object to process
                obj_path: Path for debugging purposes
                visited: Set of already visited object IDs to prevent infinite recursion
            """
            if obj is None:
                return

            # Initialize visited set if not provided
            if visited is None:
                visited = set()

            # Create a unique identifier for this object to prevent infinite recursion
            obj_id = id(obj)
            if obj_id in visited:
                return
            visited.add(obj_id)

            # print(f"Current object: {obj_path}")

            # Handle different types of objects
            if isinstance(obj, torch.Tensor):
                # Direct tensor - reset if needed
                return reset_tensor(obj)

            elif isinstance(obj, (list, tuple)):
                # Container - process each element
                for i, item in enumerate(obj):
                    item_path = f"{obj_path}[{i}]"
                    if isinstance(item, torch.Tensor):
                        new_item = reset_tensor(item)
                        if new_item is not item:
                            obj[i] = new_item
                    else:
                        # Recursively process non-tensor items
                        reset_object_tensors(item, item_path, visited)

            elif isinstance(obj, dict):
                # Dictionary - process each value
                for key, value in obj.items():
                    value_path = f"{obj_path}.{key}"
                    if isinstance(value, torch.Tensor):
                        new_value = reset_tensor(value)
                        if new_value is not value:
                            obj[key] = new_value
                    else:
                        # Recursively process non-tensor values
                        reset_object_tensors(value, value_path, visited)

            elif hasattr(obj, "__dict__"):
                # Object with attributes - process each attribute
                for attr_name, attr_value in obj.__dict__.items():
                    attr_path = f"{obj_path}.{attr_name}"
                    if isinstance(attr_value, torch.Tensor):
                        new_value = reset_tensor(attr_value)
                        if new_value is not attr_value:
                            setattr(obj, attr_name, new_value)
                    else:
                        # Recursively process non-tensor attributes
                        reset_object_tensors(attr_value, attr_path, visited)

        # print("Resetting torch tensors in model components...")

        # Process each component
        for comp_id, component in self._components.items():
            # print(f"Processing component: {comp_id}")

            # Reset tensors in the component itself
            reset_object_tensors(component, f"component.{comp_id}")

            # Reset tensors in component properties (input, output, parameters)
            # for prop_name in ['input', 'output', 'parameters']:
            #     if hasattr(component, prop_name):
            #         prop_value = getattr(component, prop_name)
            #         reset_object_tensors(prop_value, f"component.{comp_id}.{prop_name}")

        # print("Torch tensor reset complete.")

    def remove_component(
        self, component: core.System, components: Dict[str, core.System] = None
    ) -> None:
        """
        Remove a component from the model.

        Args:
            component (core.System): The component to remove.
        """
        # Connection to component
        for connection_point in component.connects_at.copy():
            for connection in connection_point.connects_system_through.copy():
                self.remove_connection(
                    connection.connects_system,
                    component,
                    connection.outputPort,
                    connection_point.inputPort,
                )

        # Connection from component
        for connection in component.connected_through.copy():
            for connection_point in connection.connects_system_at.copy():
                self.remove_connection(
                    component,
                    connection_point.connection_point_of,
                    connection.outputPort,
                    connection_point.inputPort,
                )

        if components is None:
            components = self._components

        del components[component.id]

    def add_connection(
        self,
        sender_component: core.System,
        receiver_component: core.System,
        outputPort: str,
        inputPort: str,
        components: Dict[str, core.System] = None,
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
        if components is None:
            components = self._components

        self.add_component(sender_component, components=components)
        self.add_component(receiver_component, components=components)

        found_connection_point = False
        # Check if there already is a connectionPoint with the same receiver_property_name
        for receiver_component_connection_point in receiver_component.connects_at:
            if receiver_component_connection_point.inputPort == inputPort:
                found_connection_point = True
                break

        found_connection = False
        # Check if there already is a connection with the same sender_property_name
        for sender_obj_connection in sender_component.connected_through:
            if sender_obj_connection.outputPort == outputPort:
                found_connection = True
                break

        if found_connection_point and found_connection:
            message = f'core.Connection between "{sender_component.id}" and "{receiver_component.id}" with the properties "{outputPort}" and "{inputPort}" already exists.'
            assert (
                receiver_component_connection_point
                not in sender_obj_connection.connects_system_at
            ), message

        if found_connection == False:
            sender_obj_connection = core.Connection(
                connects_system=sender_component, outputPort=outputPort
            )
            sender_component.connected_through.append(sender_obj_connection)

        if found_connection_point == False:
            receiver_component_connection_point = core.ConnectionPoint(
                connection_point_of=receiver_component, inputPort=inputPort
            )
            receiver_component.connects_at.append(receiver_component_connection_point)

        sender_obj_connection.connects_system_at.append(
            receiver_component_connection_point
        )
        receiver_component_connection_point.connects_system_through.append(
            sender_obj_connection
        )  # if sender_obj_connection not in receiver_component_connection_point.connects_system_through else None

        if components == self._components:
            sender_component_uri = self._semantic_model.SIM.__getitem__(
                sender_component.id
            )
            receiver_component_uri = self._semantic_model.SIM.__getitem__(
                receiver_component.id
            )

            sender_component_class_name = sender_component.__class__.__name__
            receiver_component_class_name = receiver_component.__class__.__name__

            connection_uri = self._semantic_model.SIM.__getitem__(
                str(hash(sender_obj_connection))
            )
            connection_point_uri = self._semantic_model.SIM.__getitem__(
                str(hash(receiver_component_connection_point))
            )

            literal_sender_property = Literal(
                outputPort
            )  # , datatype=core.namespace.XSD.string)
            literal_receiver_property = Literal(
                inputPort
            )  # , datatype=core.namespace.XSD.string)

            # Add the class of the components to the semantic model
            self._semantic_model.graph.add(
                (
                    sender_component_uri,
                    RDF.type,
                    core.namespace.SIM.__getitem__(sender_component_class_name),
                )
            )
            self._semantic_model.graph.add(
                (
                    receiver_component_uri,
                    RDF.type,
                    core.namespace.SIM.__getitem__(receiver_component_class_name),
                )
            )

            self._semantic_model.graph.add(
                (
                    core.namespace.SIM.__getitem__(sender_component_class_name),
                    RDFS.subClassOf,
                    core.namespace.S4SYST.System,
                )
            )
            self._semantic_model.graph.add(
                (
                    core.namespace.SIM.__getitem__(receiver_component_class_name),
                    RDFS.subClassOf,
                    core.namespace.S4SYST.System,
                )
            )

            # Add the class of the connections and connection points to the semantic model
            self._semantic_model.graph.add(
                (connection_uri, RDF.type, core.namespace.S4SYST.Connection)
            )
            self._semantic_model.graph.add(
                (connection_point_uri, RDF.type, core.namespace.S4SYST.ConnectionPoint)
            )

            # Add the forward connection to the semantic model
            self._semantic_model.graph.add(
                (
                    sender_component_uri,
                    core.namespace.S4SYST.connectedThrough,
                    connection_uri,
                )
            )
            self._semantic_model.graph.add(
                (
                    connection_uri,
                    core.namespace.S4SYST.connectsSystemAt,
                    connection_point_uri,
                )
            )
            self._semantic_model.graph.add(
                (
                    connection_point_uri,
                    core.namespace.S4SYST.connectionPointOf,
                    receiver_component_uri,
                )
            )

            # Add the reverse connection to the semantic model
            self._semantic_model.graph.add(
                (
                    connection_uri,
                    core.namespace.S4SYST.connectsSystem,
                    sender_component_uri,
                )
            )
            self._semantic_model.graph.add(
                (
                    connection_point_uri,
                    core.namespace.S4SYST.connectsSystemThrough,
                    connection_uri,
                )
            )
            self._semantic_model.graph.add(
                (
                    receiver_component_uri,
                    core.namespace.S4SYST.connectsAt,
                    connection_point_uri,
                )
            )

            self._semantic_model.graph.add(
                (connection_uri, core.namespace.SIM.outputPort, literal_sender_property)
            )
            self._semantic_model.graph.add(
                (
                    connection_point_uri,
                    core.namespace.SIM.inputPort,
                    literal_receiver_property,
                )
            )

    def remove_connection(
        self,
        sender_component: core.System,
        receiver_component: core.System,
        outputPort: str,
        inputPort: str,
        components: Dict[str, core.System] = None,
    ) -> None:
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
        if components is None:
            components = self._components

        sender_component_connection = None
        for connection in sender_component.connected_through:
            if connection.outputPort == outputPort:
                sender_component_connection = connection
                break
        if sender_component_connection is None:
            raise ValueError(
                f'The sender component "{sender_component.id}" does not have a connection with the property "{outputPort}"'
            )

        receiver_component_connection_point = None
        for connection_point in receiver_component.connects_at:
            if connection_point.inputPort == inputPort:
                receiver_component_connection_point = connection_point
                break
        if receiver_component_connection_point is None:
            raise ValueError(
                f'The receiver component "{receiver_component.id}" does not have a connection point with the property "{inputPort}"'
            )

        sender_component_connection.connects_system_at.remove(
            receiver_component_connection_point
        )
        receiver_component_connection_point.connects_system_through.remove(
            sender_component_connection
        )

        if len(sender_component_connection.connects_system_at) == 0:
            sender_component.connected_through.remove(sender_component_connection)
            sender_component_connection.connects_system = None

        if len(receiver_component_connection_point.connects_system_through) == 0:
            receiver_component.connects_at.remove(receiver_component_connection_point)
            receiver_component_connection_point.connection_point_of = None

        if components == self._components:
            sender_component_uri = self._semantic_model.SIM.__getitem__(
                sender_component.id
            )
            receiver_component_uri = self._semantic_model.SIM.__getitem__(
                receiver_component.id
            )

            connection_uri = self._semantic_model.SIM.__getitem__(
                str(hash(sender_component_connection))
            )  # self._semantic_model.SIM.__getitem__(sender_component.id + " " + sender_property_name)
            connection_point_uri = self._semantic_model.SIM.__getitem__(
                str(hash(receiver_component_connection_point))
            )  # self._semantic_model.SIM.__getitem__(receiver_component.id + " " + receiver_property_name)

            literal_sender_property = list(
                self._semantic_model.graph.objects(
                    connection_uri, core.namespace.SIM.outputPort
                )
            )
            literal_receiver_property = list(
                self._semantic_model.graph.objects(
                    connection_point_uri, core.namespace.SIM.inputPort
                )
            )
            assert (
                len(literal_sender_property) == 1
            ), "The connection has more than one output port."
            assert (
                len(literal_receiver_property) == 1
            ), "The connection has more than one input port."
            literal_sender_property = literal_sender_property[0]
            literal_receiver_property = literal_receiver_property[0]

            # Remove the connections from the semantic model
            self._semantic_model.graph.remove(
                (
                    connection_uri,
                    core.namespace.S4SYST.connectsSystemAt,
                    connection_point_uri,
                )
            )
            self._semantic_model.graph.remove(
                (
                    connection_point_uri,
                    core.namespace.S4SYST.connectsSystemThrough,
                    connection_uri,
                )
            )

            if len(sender_component_connection.connects_system_at) == 0:
                self._semantic_model.graph.remove(
                    (
                        sender_component_uri,
                        core.namespace.S4SYST.connectedThrough,
                        connection_uri,
                    )
                )
                self._semantic_model.graph.remove(
                    (
                        connection_uri,
                        core.namespace.S4SYST.connectsSystem,
                        sender_component_uri,
                    )
                )
                self._semantic_model.graph.remove(
                    (
                        connection_uri,
                        core.namespace.SIM.outputPort,
                        literal_sender_property,
                    )
                )

            if len(receiver_component_connection_point.connects_system_through) == 0:
                self._semantic_model.graph.remove(
                    (
                        receiver_component_uri,
                        core.namespace.S4SYST.connectsAt,
                        connection_point_uri,
                    )
                )
                self._semantic_model.graph.remove(
                    (
                        connection_point_uri,
                        core.namespace.S4SYST.connectionPointOf,
                        receiver_component_uri,
                    )
                )
                self._semantic_model.graph.remove(
                    (
                        connection_point_uri,
                        core.namespace.SIM.inputPort,
                        literal_receiver_property,
                    )
                )

    def count_components(self) -> int:
        return len(self._components)

    def count_connections(self) -> int:
        return self._semantic_model.count_triples(
            s=None, p=core.namespace.S4SYST.connectsSystemAt, o=None
        )

    def get_object_properties(self, object_: Any) -> Dict:
        """
        Get all properties of an object.

        Args:
            object_ (Any): The object to get properties from.

        Returns:
            Dict: A dictionary of object properties.
        """
        return {key: value for (key, value) in vars(object_).items()}

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
        if filter is None:
            filter = lambda v, class_: True
        return [
            v for v in dict_.values() if (isinstance(v, class_) and filter(v, class_))
        ]

    def set_custom_initial_dict(
        self, _custom_initial_dict: Dict[str, Dict[str, Any]]
    ) -> None:
        """
        Set custom initial values for components.

        Args:
            _custom_initial_dict (Dict[str, Dict[str, Any]]): Dictionary of custom initial values.

        Raises:
            AssertionError: If unknown component IDs are provided.
        """
        np_custom_initial_dict_ids = np.array(list(_custom_initial_dict.keys()))
        legal_ids = np.array(
            [dict_id in self._components for dict_id in _custom_initial_dict]
        )
        assert np.all(
            legal_ids
        ), f'Unknown component id(s) provided in "_custom_initial_dict": {np_custom_initial_dict_ids[legal_ids==False]}'
        self._custom_initial_dict = _custom_initial_dict

    def set_initial_values(self, dict_: Dict[str, Any]) -> None:
        """
        Set initial values for all components in the model.
        """
        for component in self._components.values():
            # Check that all keys in the dictionary are valid output properties
            for key in dict_[component.id].keys():
                assert (
                    key in component.output
                ), f'Invalid output property "{key}" for component "{component.id}"'
                assert isinstance(
                    dict_[component.id][key], component.output[key].__class__
                ), f'Invalid type for output property "{key}" for component "{component.id}"'
            component.output.update(dict_[component.id])

    def set_parameters_from_array(
        self,
        values: List[Any],
        components: List[core.System],
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
        if normalized is None:
            normalized = [False] * len(values)
        elif isinstance(normalized, bool):
            normalized = [normalized] * len(values)

        for i, (v, obj, attr, normalized_) in enumerate(
            zip(values, components, parameter_names, normalized)
        ):
            assert rhasattr(
                obj, attr
            ), f'The component with class "{obj.__class__.__name__}" and id "{obj.id}" has no attribute "{attr}".'

            if v is not None:
                obj_ = rgetattr(obj, attr)

                if isinstance(
                    obj_, tps.Parameter
                ):  # Only change underlying data in torch.Parameter
                    if overwrite:
                        if save_original:
                            if (
                                obj.id not in self._saved_parameters
                            ):  # Save the original parameter if we later need to restore it
                                self._saved_parameters[obj.id] = {}
                            self._saved_parameters[obj.id][attr] = obj_

                        new_param = tps.TensorParameter(
                            v,
                            min_value=obj_.min_value,
                            max_value=obj_.max_value,
                            normalized=normalized_,
                        )
                        rdelattr(obj, attr)
                        rsetattr(obj, attr, new_param)
                        # new_param.set(v, normalized=normalized_)
                    else:
                        obj_.set(v, normalized=normalized_)
                elif isinstance(obj_, tps.TensorParameter):
                    obj_.set(v, normalized=normalized_)
                else:
                    rsetattr(obj, attr, v)

    def restore_parameters(self, keep_values: bool = True) -> None:
        for obj in self._saved_parameters:
            for attr in self._saved_parameters[obj]:
                old_obj = rgetattr(self._components[obj], attr)
                v = old_obj.get()
                new_obj = self._saved_parameters[obj][attr]
                rdelattr(self._components[obj], attr)
                rsetattr(self._components[obj], attr, new_obj)
                if keep_values:
                    new_obj.set(v, normalized=False)

    def set_parameters_from_config(self, d: dict, component: core.System):
        """
        Recursively set parameters from a dictionary.
        """
        for key in d.keys():
            entry = d[key]
            cond = isinstance(entry, dict) and all(
                [rhasattr(component, k) for k in entry.keys()]
            )
            if cond:
                self.set_parameters_from_config(entry, component)
            else:
                self.set_parameters_from_array([entry], [component], [key])
        return

    def cache(
        self,
        startTime: Optional[datetime.datetime] = None,
        endTime: Optional[datetime.datetime] = None,
        stepSize: Optional[int] = None,
        simulator: Optional[core.Simulator] = None,
    ) -> None:
        """
        Cache data and create folder structure for time series data.

        Args:
            startTime (Optional[datetime.datetime]): Start time for caching.
            endTime (Optional[datetime.datetime]): End time for caching.
            stepSize (Optional[int]): Time step size for caching.
        """
        c = self.get_component_by_class(
            self._components,
            (
                systems.SensorSystem,
                systems.OutdoorEnvironmentSystem,
                systems.TimeSeriesInputSystem,
            ),
        )
        for component in c:
            component.initialize(
                startTime=startTime,
                endTime=endTime,
                stepSize=stepSize,
                simulator=simulator,
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
            for i, connection_point in enumerate(component.connects_at):
                for j, connection in enumerate(
                    connection_point.connects_system_through
                ):
                    connected_component = connection.connects_system
                    if isinstance(
                        component.input[connection_point.inputPort], tps.Vector
                    ):
                        if (
                            component,
                            connected_component,
                            connection.outputPort,
                            connection_point.inputPort,
                        ) in self._translator.E_conn_to_sp_group:
                            sp, groups = self._translator.E_conn_to_sp_group[
                                (
                                    component,
                                    connected_component,
                                    connection.outputPort,
                                    connection_point.inputPort,
                                )
                            ]
                            # Find the group of the connected component
                            modeled_match_nodes_ = self._translator.sim2sem_map[
                                connected_component
                            ]
                            groups_matched = [
                                g
                                for g in groups
                                if len(
                                    modeled_match_nodes_.intersection(set(g.values()))
                                )
                                > 0
                            ]
                            assert (
                                len(groups_matched) == 1
                            ), "Only one group is allowed for each component."
                            group = groups_matched[0]
                            group_id = id(group)
                            component.input[connection_point.inputPort].update(
                                group_id=group_id
                            )
                        else:
                            component.input[connection_point.inputPort].update()

            component.initialize(
                startTime=startTime,
                endTime=endTime,
                stepSize=stepSize,
                simulator=simulator,
            )

    def validate(self) -> None:
        """
        Validate the model by checking IDs and connections.
        """
        PRINTPROGRESS.add_level()
        (
            validated_for_simulator1,
            validated_for_estimator1,
            validated_for_optimizer1,
        ) = self.validate_components()
        (
            validated_for_simulator2,
            validated_for_estimator2,
            validated_for_optimizer2,
        ) = self.validate_ids()
        (
            validated_for_simulator3,
            validated_for_estimator3,
            validated_for_optimizer3,
        ) = self.validate_connections()

        self._validated_for_simulator = (
            validated_for_simulator1
            and validated_for_simulator2
            and validated_for_simulator3
        )
        self._validated_for_estimator = (
            validated_for_estimator1
            and validated_for_estimator2
            and validated_for_estimator3
        )
        self._validated_for_optimizer = (
            validated_for_optimizer1
            and validated_for_optimizer2
            and validated_for_optimizer3
        )
        self._is_validated = (
            self._validated_for_simulator
            and self._validated_for_estimator
            and self._validated_for_optimizer
        )
        PRINTPROGRESS.remove_level()

        PRINTPROGRESS("Validated for Simulator")
        if self._validated_for_simulator:
            status = "OK"
        else:
            status = "FAILED"
        PRINTPROGRESS("Validated for Estimator", status=status)

        if self._validated_for_estimator:
            status = "OK"
        else:
            status = "FAILED"
        PRINTPROGRESS("Validated for Optimizer", status=status)

        if self._validated_for_optimizer:
            status = "OK"
        else:
            status = "FAILED"
        PRINTPROGRESS("", plain=True, status=status)

        # assert validated, "The model is not valid. See the warnings above."

    def validate_components(self) -> None:
        """
        Validate the parameters of all components in the model.

        Raises:
            AssertionError: If any component has invalid parameters.
        """
        component_instances = list(self._components.values())
        _validated_for_simulator = True
        _validated_for_estimator = True
        _validated_for_optimizer = True

        for component in component_instances:
            if hasattr(component, "validate"):  # Check if component has validate method
                (
                    validated_for_simulator_,
                    validated_for_estimator_,
                    validated_for_optimizer_,
                ) = component.validate(PRINTPROGRESS)
                _validated_for_simulator = (
                    _validated_for_simulator and validated_for_simulator_
                )
                _validated_for_estimator = (
                    _validated_for_estimator and validated_for_estimator_
                )
                _validated_for_optimizer = (
                    _validated_for_optimizer and validated_for_optimizer_
                )
            else:
                # Validate parameters
                config = component.config.copy()
                parameters = {
                    attr: rgetattr(component, attr) for attr in config["parameters"]
                }
                is_none = [k for k, v in parameters.items() if v is None]
                if any(is_none):
                    message = f"|CLASS: {component.__class__.__name__}|ID: {component.id}|: Missing values for the following parameter(s) to enable use of Simulator, and Optimizer:"
                    PRINTPROGRESS(message, plain=True, status="[WARNING]")
                    PRINTPROGRESS.add_level()
                    for par in is_none:
                        PRINTPROGRESS(par, plain=True, status="")
                    PRINTPROGRESS.remove_level()

                    _validated_for_simulator = False
                    _validated_for_optimizer = False

                # Validate model definitions
                for input in component.input.values():
                    assert isinstance(
                        input, (tps.Scalar, tps.Vector)
                    ), "Only vectors and scalars can be used as input to components"

                for output in component.output.values():
                    assert isinstance(
                        output, (tps.Scalar, tps.Vector)
                    ), "Only vectors and scalars can be used as output from components"

                if len(component.connects_at) == 0:
                    for key in component.output.keys():
                        output = component.output[key]
                        if isinstance(
                            output, tps.Scalar
                        ):  # TODO: Add support for vectors
                            if output.is_leaf == False:
                                message = f'|CLASS: {component.__class__.__name__}|ID: {component.id}|: The output "{key}" is not a leaf scalar. Only leaf scalars can be used as output from components with no inputs.'
                                PRINTPROGRESS(message, plain=True, status="[WARNING]")
                                _validated_for_optimizer = False

                            # assert output.is_leaf, f"|CLASS: {component.__class__.__name__}|ID: {component.id}|: The output \"{key}\" is not a leaf scalar. Only leaf scalars can be used as output from components with no inputs."

                else:
                    for key in component.output.keys():
                        output = component.output[key]
                        if isinstance(
                            output, tps.Scalar
                        ):  # TODO: Add support for vectors
                            if output.is_leaf:
                                message = f'|CLASS: {component.__class__.__name__}|ID: {component.id}|: The output "{key}" is a leaf scalar. Only non-leaf scalars can be used as output from components with inputs.'
                                PRINTPROGRESS(message, plain=True, status="[WARNING]")
                                _validated_for_optimizer = False
                            # assert output.is_leaf==False, f"|CLASS: {component.__class__.__name__}|ID: {component.id}|: The output \"{key}\" is a leaf scalar. Only non-leaf scalars can be used as output from components with inputs."
        return (
            _validated_for_simulator,
            _validated_for_estimator,
            _validated_for_optimizer,
        )

    def validate_ids(self) -> None:
        """
        Validate the IDs of all components in the model.

        Raises:
            AssertionError: If any component has an invalid ID.
        """
        validated = True
        component_instances = list(self._components.values())
        for component in component_instances:
            isvalid = np.array(
                [x.isalnum() or x in INVALID_ID_CHARS for x in component.id]
            )
            np_id = np.array(list(component.id))
            violated_characters = list(np_id[isvalid == False])
            if not all(isvalid):
                message = f"|CLASS: {component.__class__.__name__}|ID: {component.id}|: Invalid id. The characters \"{', '.join(violated_characters)}\" are not allowed."
                PRINTPROGRESS(message)
                validated = False
        return (validated, validated, validated)

    def validate_connections(self) -> None:
        """
        Validate the connections between components in the model.

        Raises:
            AssertionError: If any required connections are missing.
        """
        component_instances = list(self._components.values())
        validated = True
        for component in component_instances:
            if (
                len(component.connected_through) == 0
                and len(component.connects_at) == 0
            ):
                message = f"|CLASS: {component.__class__.__name__}|ID: {component.id}|: The component is not connected to any other components."
                PRINTPROGRESS(message, plain=True, status="[WARNING]")
                # self.remove_component(component)

            if hasattr(component, "optional_inputs"):
                optional_inputs = component.optional_inputs
            else:
                optional_inputs = []
            input_labels = [cp.inputPort for cp in component.connects_at]
            first_input = True
            for req_input_label in component.input.keys():
                if (
                    req_input_label not in input_labels
                    and req_input_label not in optional_inputs
                ):
                    if first_input:
                        message = f"|CLASS: {component.__class__.__name__}|ID: {component.id}|: Missing connections for the following input(s) to enable use of Simulator, Estimator, and Optimizer:"
                        PRINTPROGRESS(message, plain=True, status="[WARNING]")
                        first_input = False
                        PRINTPROGRESS.add_level()
                    PRINTPROGRESS(req_input_label, plain=True)
                    validated = False
            if first_input == False:
                PRINTPROGRESS.remove_level()
        return (validated, validated, validated)

    def _load_parameters(self, force_config_overwrite: bool = False) -> None:
        """
        Load parameters for all components from configuration files.

        Args:
            force_config_overwrite (bool): If True, all parameters are read from the config file. If False, only the parameters that are None are read from the config file. If you want to use the fcn function
            to set the parameters, you should set force_config_overwrite to False to avoid it being overwritten.
        """
        PRINTPROGRESS.add_level()

        for component in self._components.values():
            assert hasattr(
                component, "config"
            ), f'The class "{component.__class__.__name__}" has no "config" attribute.'
            config_ = component.populate_config()

            # assert "parameters" in config_, f"The \"config\" attribute of class \"{component.__class__.__name__}\" has no \"parameters\" key."
            filename, isfile = self.get_dir(
                folder_list=["model_parameters", component.__class__.__name__],
                filename=f"{component.id}.json",
            )
            if isfile == False:
                with open(filename, "w") as f:
                    json.dump(config_, f, indent=4)
            else:
                with open(filename) as f:
                    config = json.load(f)

                comparison_result = compare_dict_structure(config_, config)
                if not comparison_result["structures_match"]:
                    message = f"|CLASS: {component.__class__.__name__}|ID: {component.id}|: Config structure mismatch."
                    PRINTPROGRESS(message, plain=True, status="[WARNING]")
                    PRINTPROGRESS.add_level()
                    if comparison_result["missing_in_1"]:
                        missing_msg = f"File config has unused parameters: {', '.join(sorted(comparison_result['missing_in_1']))}"
                        PRINTPROGRESS(missing_msg, plain=True, status="[WARNING]")
                    if comparison_result["missing_in_2"]:
                        missing_msg = f"File config is missing the following parameters: {', '.join(sorted(comparison_result['missing_in_2']))}"
                        PRINTPROGRESS(missing_msg, plain=True, status="[WARNING]")
                    PRINTPROGRESS.remove_level()

                if force_config_overwrite:
                    config_ = merge_dicts(config_, config, prioritize="dict2")
                else:
                    config_ = merge_dicts(
                        config_, config, prioritize="dict1"
                    )  # Prioritize config_ over config to allow user to change stuff in the fcn function (programatically)

                self.set_parameters_from_config(config_, component)

                with open(filename, "w") as f:
                    json.dump(config_, f, indent=4)

        PRINTPROGRESS.remove_level()

    def load(
        self,
        rdf_file: Optional[str] = None,
        fcn: Optional[Callable] = None,
        verbose: bool = False,
        validate_model: bool = True,
        force_config_overwrite: bool = False,
    ) -> None:
        """
        Load and set up the model for simulation.

        Args:
            rdf_file (Optional[str]): Path to a serialized model.
            fcn (Optional[Callable]): Custom function to be applied during model loading.
            verbose (bool): Whether to print verbose output during loading.
            validate_model (bool): Whether to perform model validation.
            force_config_overwrite (bool): If True, all parameters are read from the config file. If False, only the parameters that are None are read from the config file. If you want to use the fcn function
            to set the parameters, you should set force_config_overwrite to False to avoid it being overwritten.
        """
        if verbose:
            self._load(
                rdf_file=rdf_file,
                fcn=fcn,
                validate_model=validate_model,
                force_config_overwrite=force_config_overwrite,
                verbose=verbose,
            )
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._load(
                    rdf_file=rdf_file,
                    fcn=fcn,
                    validate_model=validate_model,
                    force_config_overwrite=force_config_overwrite,
                    verbose=verbose,
                )

    def _load(
        self,
        rdf_file: Optional[str] = None,
        fcn: Optional[Callable] = None,
        verbose: bool = False,
        validate_model: bool = True,
        force_config_overwrite: bool = False,
    ) -> None:
        """
        Internal method to load and set up the model for simulation.

        This method is called by load and performs the actual loading process.

        Args:
            fcn (Optional[Callable]): Custom function to be applied during model loading.
            validate_model (bool): Whether to perform model validation.
        """

        if self._is_loaded:
            warnings.warn(
                "The simulation model is already loaded. Resetting simulation model."
            )
            self.reset()

        self._is_loaded = True

        PRINTPROGRESS("Loading simulation model")
        PRINTPROGRESS.add_level()

        if rdf_file is not None:
            PRINTPROGRESS("Loading model from RDF file")
            self._load_model_from_rdf(rdf_file)

        if fcn is not None:
            assert callable(
                fcn
            ), "The function to be applied during model loading is not callable."
            PRINTPROGRESS("Applying user defined function")
            fcn(self)

        PRINTPROGRESS("Removing cycles")
        self._get_components_no_cycles()

        PRINTPROGRESS("Determining execution order")
        self._get_execution_order()

        PRINTPROGRESS("Loading parameters")
        self._load_parameters(force_config_overwrite=force_config_overwrite)

        if validate_model:
            PRINTPROGRESS("Validating model")
            self.validate()

        PRINTPROGRESS.remove_level()

        if verbose:
            print(self)

    def set_save_simulation_result(self, flag: bool = True, c: list = None):
        assert isinstance(flag, bool), "The flag must be a boolean."
        if c is not None:
            assert isinstance(c, list), "The c must be a list."
            for component in c:
                for input_key in component.input.keys():
                    if isinstance(component.input[input_key], tps.Scalar):
                        component.input[input_key].log_history = flag
                for output_key in component.output.keys():
                    if isinstance(component.output[output_key], tps.Scalar):
                        component.output[output_key].log_history = flag
        else:
            for component in self._components.values():
                for input_key in component.input.keys():
                    if isinstance(component.input[input_key], tps.Scalar):
                        component.input[input_key].log_history = flag
                for output_key in component.output.keys():
                    if isinstance(component.output[output_key], tps.Scalar):
                        component.output[output_key].log_history = flag

    def reset(self) -> None:
        """
        Reset the model to its initial state.
        """
        # Reset all the dictionaries and lists
        # self._components = {} ###
        self._custom_initial_dict = None  ###
        self._execution_order = []  ###
        self._flat_execution_order = []  ###
        self._required_initialization_connections = []  ###
        self._components_no_cycles = {}  ###
        self._saved_parameters = {}  ###

        # Reset the loaded state
        self._is_loaded = False  ###
        self._is_validated = False  ###

        # Reset any estimation results
        self._result = None  ###

    def get_simple_graph(self, components) -> Dict:
        """
        Get a simple graph representation of the system graph.
        This is a simplified version of the system graph that drops information about edge labels (Connection and ConnectionPoint pairs).

        Returns:
            Dict: The simple graph representation.
        """
        simple_graph = {c: set() for c in components.values()}
        for component in components.values():
            for connection in component.connected_through:
                for connection_point in connection.connects_system_at:
                    receiver_component = connection_point.connection_point_of

                    # If node component has multiple edges to node receiver_component, we will only add one edge to the simple graph (simple_graph[component] is a set).
                    # Later if this is part of a cycle, we will have to remove all edges between component and receiver_component.
                    simple_graph[component].add(receiver_component)
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
        _new_components = {}
        new_to_old_mapping = {}
        old_to_new_mapping = {}
        for component in self._components.values():
            if component not in old_to_new_mapping:
                new_component = copy.copy(component)
                new_component.connected_through = []
                new_component.connects_at = []
                new_to_old_mapping[new_component] = component
                old_to_new_mapping[component] = new_component
            else:
                new_component = old_to_new_mapping[component]

            for connection in component.connected_through:
                for connection_point in connection.connects_system_at:
                    connected_component = connection_point.connection_point_of
                    if connected_component not in old_to_new_mapping:
                        new_connected_component = copy.copy(connected_component)
                        new_connected_component.connected_through = []
                        new_connected_component.connects_at = []
                        new_to_old_mapping[new_connected_component] = (
                            connected_component
                        )
                        old_to_new_mapping[connected_component] = (
                            new_connected_component
                        )
                    else:
                        new_connected_component = old_to_new_mapping[
                            connected_component
                        ]
                    self.add_connection(
                        new_component,
                        new_connected_component,
                        connection.outputPort,
                        connection_point.inputPort,
                        components=_new_components,
                    )

        # _new_components = {k: old_to_new_mapping[v] for k, v in self._components.items()}
        return _new_components

    def _get_components_no_cycles(self) -> None:
        """
        Create a dictionary of components without cycles using an improved algorithm
        that minimizes the number of edges removed.
        """
        self._components_no_cycles = self._copy_components()
        self._required_initialization_connections = []

        # Use the improved cycle removal algorithm
        self._remove_cycles()

    def _remove_cycles(self) -> None:
        """
        Remove cycles using an improved algorithm that minimizes edge removal.

        This algorithm uses multiple strategies:
        1. Finds all simple cycles in the simplified graph (once)
        2. Counts how many cycles each component-to-component edge participates in
        3. Greedily removes edges that break the most cycles
        4. Updates cycle list incrementally instead of recalculating
        5. Repeats until no cycles remain

        Note: When an edge (c_from -> c_to) is selected for removal, ALL connections
        between those components are removed, as per the existing architecture.
        """
        iteration = 0
        max_iterations = 1000  # Safety limit to prevent infinite loops

        # Calculate all cycles once at the beginning
        cycles = list(self.get_simple_cycles(self._components_no_cycles))
        if not cycles:
            return  # No cycles to remove

        while iteration < max_iterations and cycles:
            iteration += 1

            # Count edge participation in remaining cycles
            edge_cycle_count = {}

            for cycle in cycles:
                for i in range(len(cycle)):
                    c_from = cycle[i]
                    c_to = cycle[
                        (i + 1) % len(cycle)
                    ]  # Next component in cycle (wraps around)

                    # Use simplified edge representation (just component pair)
                    edge_key = (c_from, c_to)

                    if edge_key not in edge_cycle_count:
                        edge_cycle_count[edge_key] = 0
                    edge_cycle_count[edge_key] += 1

            if not edge_cycle_count:
                break

            # Find the best edge to remove using multiple criteria
            best_edge = self._select_best_edge_to_remove(edge_cycle_count)

            # Remove ALL connections between the selected components
            c_from, c_to = best_edge
            self._remove_all_edges_between_components(c_from, c_to)

            # Update cycles list by removing cycles that contained the removed edge
            cycles = self._update_cycles_after_edge_removal(cycles, best_edge)

        if iteration >= max_iterations:
            print(
                f"Warning: Cycle removal reached maximum iterations ({max_iterations}). "
                "There might be remaining cycles."
            )

    def _update_cycles_after_edge_removal(self, cycles, removed_edge):
        """
        Update the cycles list after removing an edge, avoiding full recalculation.

        Args:
            cycles: Current list of cycles
            removed_edge: The edge (c_from, c_to) that was removed

        Returns:
            Updated list of cycles with broken cycles removed
        """
        c_from, c_to = removed_edge
        updated_cycles = []

        for cycle in cycles:
            # Check if this cycle contains the removed edge
            cycle_broken = False
            for i in range(len(cycle)):
                cycle_c_from = cycle[i]
                cycle_c_to = cycle[(i + 1) % len(cycle)]

                if cycle_c_from == c_from and cycle_c_to == c_to:
                    cycle_broken = True
                    break

            # Only keep cycles that don't contain the removed edge
            if not cycle_broken:
                updated_cycles.append(cycle)

        return updated_cycles

    def _select_best_edge_to_remove(self, edge_cycle_count):
        """
        Select the best edge to remove using multiple criteria.

        Priority order:
        1. Edges that participate in the most cycles
        2. Among ties, prefer edges from components with more outgoing connections

        Args:
            edge_cycle_count: Dictionary mapping (c_from, c_to) tuples to cycle participation count

        Returns:
            The best edge tuple (c_from, c_to) to remove
        """
        # Group edges by cycle participation count (descending)
        max_cycle_count = max(edge_cycle_count.values())
        best_edges = [
            edge for edge, count in edge_cycle_count.items() if count == max_cycle_count
        ]

        # If multiple edges have the same max count, apply additional criteria
        if len(best_edges) > 1:
            # Prefer edges from components with more outgoing connections
            def edge_priority(edge):
                c_from, c_to = edge
                # Higher number of outgoing connections = higher priority for removal
                outgoing_count = len(c_from.connected_through)
                return outgoing_count

            best_edges.sort(key=edge_priority, reverse=True)

        return best_edges[0]

    def _remove_all_edges_between_components(self, c_from, c_to):
        """
        Remove ALL connections between two components.

        This aligns with the existing architecture where the simplified graph
        collapses multiple edges into one, so removing an edge means removing
        all connections between those components.

        Args:
            c_from: Source component
            c_to: Target component
        """
        # Find and remove all connections from c_from to c_to
        connections_to_remove = []
        for connection in c_from.connected_through:
            for connection_point in connection.connects_system_at:
                if c_to == connection_point.connection_point_of:
                    connections_to_remove.append((connection, connection_point))

        # Remove the identified connections
        for connection, connection_point in connections_to_remove:
            connection.connects_system_at.remove(connection_point)
            connection_point.connects_system_through.remove(connection)
            self._required_initialization_connections.append(connection)

            # Clean up empty connection point
            if len(connection_point.connects_system_through) == 0:
                c_to.connects_at.remove(connection_point)

            # Clean up empty connection
            if len(connection.connects_system_at) == 0:
                c_from.connected_through.remove(connection)

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
            if ext == ".pickle":
                with open(filename, "rb") as handle:
                    self._result = pickle.load(handle)

            elif ext == ".npz":
                if "_ls.npz" in filename:
                    d = dict(np.load(filename, allow_pickle=True))
                    d = {
                        k.replace(".", "_"): v for k, v in d.items()
                    }  # For backwards compatibility
                    self._result = estimator.EstimationResult(**d)
                elif "_mcmc.npz" in filename:
                    d = dict(np.load(filename, allow_pickle=True))
                    d = {
                        k.replace(".", "_"): v for k, v in d.items()
                    }  # For backwards compatibility
                    self._result = estimator.EstimationResult(**d)
                else:
                    raise Exception(
                        'The estimation result file is not of a supported type. The file must be a .pickle, .npz file with the name containing "_ls" or "_mcmc".'
                    )

                for key, value in self._result.items():
                    self._result[key] = (
                        1 / self._result["chain_betas"] if key == "chain_T" else value
                    )
                    if self._result[key].size == 1 and (
                        len(self._result[key].shape) == 0
                        or len(self._result[key].shape) == 1
                    ):
                        self._result[key] = value.tolist()

                    elif (
                        key == "startTime_train"
                        or key == "endTime_train"
                        or key == "stepSize_train"
                    ):
                        self._result[key] = value.tolist()
            else:
                raise Exception(
                    f"The estimation result is of type {type(self._result)}. This type is not supported by the model class."
                )

        if isinstance(self._result, estimator.EstimationResult):
            theta = self._result["result_x"]
        else:
            raise Exception(
                f"The estimation result is of type {type(self._result)}. This type is not supported by the model class."
            )

        flat_components = [
            self._components[com_id] for com_id in self._result["component_id"]
        ]
        flat_attr_list = self._result["component_attr"]
        theta_mask = self._result["theta_mask"]
        theta = theta[theta_mask]
        self.set_parameters_from_array(theta, flat_components, flat_attr_list)

    def check_for_for_missing_initial_values(self) -> None:
        """
        Check for missing initial values in components.

        Raises:
            Exception: If any component is missing an initial value.
        """
        for connection in self._required_initialization_connections:
            component = connection.connects_system
            if connection.outputPort not in component.output:
                raise Exception(
                    f'The component with id: "{component.id}" and class: "{component.__class__.__name__}" is missing an initial value for the output: {connection.outputPort}'
                )
            elif component.output[connection.outputPort].get() is None:
                raise Exception(
                    f'The component with id: "{component.id}" and class: "{component.__class__.__name__}" is missing an initial value for the output: {connection.outputPort}'
                )

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
                for connection in component.connected_through:
                    for connection_point in connection.connects_system_at:
                        # connection_point = connection.connects_system_at
                        receiver_component = connection_point.connection_point_of
                        connection_point.connects_system_through.remove(connection)
                        if len(connection_point.connects_system_through) == 0:
                            receiver_component.connects_at.remove(connection_point)

                        if len(receiver_component.connects_at) == 0:
                            activeComponentsNew.append(receiver_component)
            activeComponents = activeComponentsNew
            self._execution_order.append(component_group)
            return activeComponents

        initComponents = [
            v for v in self._components_no_cycles.values() if len(v.connects_at) == 0
        ]
        activeComponents = initComponents
        self._execution_order = []
        while len(activeComponents) > 0:
            activeComponents = _traverse(self, activeComponents)

        # Map the execution order from the no cycles component dictionary to the full component dictionary.
        self._execution_order = [
            [self._components[component.id] for component in component_group]
            for component_group in self._execution_order
        ]

        # Map required initialization connections from the no cycles component dictionary to the full component dictionary.
        self._required_initialization_connections = [
            connection
            for no_cycle_connection in self._required_initialization_connections
            for connection in self._components[
                no_cycle_connection.connects_system.id
            ].connected_through
            if connection.outputPort == no_cycle_connection.outputPort
        ]

        self._flat_execution_order = _flatten(self._execution_order)
        assert len(self._flat_execution_order) == len(
            self._components_no_cycles
        ), 'Cycles detected in the model. Inspect the generated file "system_graph.png" to see where.'

    def _update_literals(self, component: core.System = None) -> None:
        """
        Update the literals in the semantic model.
        """

        def _update_literals_for_component(component: core.System) -> None:
            component_uri = self._semantic_model.SIM.__getitem__(component.id)
            for key, value in flatten_dict(component.populate_config(), component):
                if isinstance(value, dict):
                    value_ = json.dumps(value)
                    datatype = core.namespace.RDF.JSON
                else:
                    value_ = value
                    datatype = None

                # Check if the property is already in the semantic model
                literal_property = list(
                    self._semantic_model.graph.objects(
                        component_uri, core.namespace.SIM.__getitem__(key)
                    )
                )
                if len(literal_property) == 0:
                    # No literal in the semantic model.
                    # Add the literal to the semantic model.
                    literal_property = Literal(value_, datatype=datatype)
                    self._semantic_model.graph.add(
                        (
                            component_uri,
                            core.namespace.SIM.__getitem__(key),
                            literal_property,
                        )
                    )
                elif len(literal_property) == 1:
                    # There is one literal in the semantic model.
                    literal_property = literal_property[0]
                    # Remove the literal from the semantic model.
                    self._semantic_model.graph.remove(
                        (
                            component_uri,
                            core.namespace.SIM.__getitem__(key),
                            literal_property,
                        )
                    )
                    # Add the new literal to the semantic model.
                    literal_property = Literal(value_, datatype=datatype)
                    self._semantic_model.graph.add(
                        (
                            component_uri,
                            core.namespace.SIM.__getitem__(key),
                            literal_property,
                        )
                    )
                else:
                    # There are more than one literal in the semantic model.
                    raise Exception(
                        f'The component with id: "{component.id}" has more than one output port.'
                    )

        if component is None:
            for component in self._components.values():
                _update_literals_for_component(component)
        else:
            _update_literals_for_component(component)

    def serialize(self):
        """
        Serialize the simulation model.
        """
        self._update_literals()
        self._semantic_model.serialize()

    def visualize(self, query: str = None, literals: bool = True) -> None:
        """
        Visualize the simulation model.
        """
        self._update_literals()
        if query is None:
            if literals:
                query = None
            else:
                query = """
                CONSTRUCT {
                    ?s ?p ?o
                }
                WHERE {
                    ?s ?p ?o .
                    FILTER (?p = s4syst:connectsSystemAt || 
                            ?p = s4syst:connectedThrough || 
                            ?p = s4syst:connectionPointOf ||
                            ?p = sim:inputPort ||
                            ?p = sim:outputPort)
                }
                """
        self._semantic_model.visualize(query)

    def _load_model_from_rdf(self, rdf_file: str) -> None:
        """
        Load a complete model (components and connections) from an RDF file.
        This method reads the RDF file and reconstructs both components and their connections.

        Args:
            rdf_file (str): Path to the RDF file to load from
        """
        self._semantic_model = core.SemanticModel(
            id=self._id,
            rdf_file=rdf_file,
            namespaces={"SIM": core.namespace.SIM, "S4SYST": core.namespace.S4SYST},
            dir_conf=self._dir_conf + ["semantic_model"],
        )

        # Instantiate components with their attributes
        for sm_instance in self._semantic_model.get_instances_of_type(
            core.namespace.S4SYST.System
        ):
            t = [t for t in sm_instance.type if t.has_subclasses() == False][0]
            class_name = t.get_short_name()
            cls = getattr(systems, class_name)
            attributes = {}
            for pred, obj in sm_instance.get_predicate_object_pairs().items():
                for obj_ in obj:
                    if obj_.is_literal:
                        literal_value = obj_.uri.value
                        attributes[
                            get_short_name(pred, self._semantic_model.namespaces)
                        ] = literal_value
            component = cls(id=sm_instance.get_short_name(), **attributes)
            # Check if the component already exists
            self.add_component(component)

        # Go through all the connections (from - to) and add them to the simulation model
        for sm_instance in self._semantic_model.get_instances_of_type(
            core.namespace.S4SYST.System
        ):
            component = self._components[sm_instance.get_short_name()]
            predicate_object_pairs = sm_instance.get_predicate_object_pairs()
            if (
                core.namespace.S4SYST.connectedThrough in predicate_object_pairs
            ):  # You can have a System without connections so we need to check
                connections = predicate_object_pairs[
                    core.namespace.S4SYST.connectedThrough
                ]

                for connection in connections:
                    predicate_object_pairs_connection = (
                        connection.get_predicate_object_pairs()
                    )
                    outputPort = predicate_object_pairs_connection[
                        core.namespace.SIM.outputPort
                    ][
                        0
                    ].uri.value  # There can only be one output port per connection
                    connection_points = predicate_object_pairs_connection[
                        core.namespace.S4SYST.connectsSystemAt
                    ]

                    for connection_point in connection_points:
                        predicate_object_pairs_connection_point = (
                            connection_point.get_predicate_object_pairs()
                        )
                        receiver_component = predicate_object_pairs_connection_point[
                            core.namespace.S4SYST.connectionPointOf
                        ][
                            0
                        ]  # There can only be one connection point per connection
                        inputPort = predicate_object_pairs_connection_point[
                            core.namespace.SIM.inputPort
                        ][
                            0
                        ].uri.value  # There can only be one input port per connection point

                        receiver_component_id = receiver_component.get_short_name()
                        receiver_component = self._components[receiver_component_id]

                        self.add_connection(
                            sender_component=component,
                            receiver_component=receiver_component,
                            outputPort=outputPort,
                            inputPort=inputPort,
                        )
