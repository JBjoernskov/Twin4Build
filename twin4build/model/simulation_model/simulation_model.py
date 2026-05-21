from __future__ import annotations

# Standard library imports
import copy
import datetime
import json
import os
import pickle
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

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
from twin4build.utils.get_obj_attr import get_obj_attr
from twin4build.utils.mkdir_in_root import mkdir_in_root
from twin4build.utils.print_progress import LOGGER, autoreset_print
from twin4build.utils.rdelattr import rdelattr
from twin4build.utils.rgetattr import rgetattr
from twin4build.utils.rhasattr import rhasattr
from twin4build.utils.rsetattr import rsetattr
from twin4build.utils.simple_cycle import simple_cycles
from twin4build.utils.validate_period import validate_period

INVALID_ID_CHARS = ["_", "-", " ", "(", ")", "[", "]"]

# Hard cap on id length. Ids are used as filename components under
# ``model_parameters/<class>/<id>.json`` and similar; on Windows the full
# path must stay under MAX_PATH (260 chars) unless long-path support is
# enabled, so we bound the id at a value that leaves headroom for the
# project directory, class folder, and file extension.
MAX_ID_LEN = 100


def _check_id(id: str, kind: str) -> None:
    """Validate that ``id`` is a legal string of allowed characters and
    within :data:`MAX_ID_LEN`. Raises :class:`AssertionError` with a
    descriptive message on failure.

    ``kind`` is a short label (e.g. ``"model"`` or ``"component"``) used in
    the error message.
    """
    assert isinstance(id, str), f'Argument "id" must be of type {str(type(str))}'
    assert len(id) <= MAX_ID_LEN, (
        f'The {kind} with id "{id}" exceeds the maximum id length of '
        f"{MAX_ID_LEN} characters (got {len(id)}). Ids are used as "
        f"filename components; long ids trip Windows MAX_PATH and cause "
        f"OSError at load/save time."
    )
    isvalid = np.array([x.isalnum() or x in INVALID_ID_CHARS for x in id])
    np_id = np.array(list(id))
    violated_characters = list(np_id[isvalid == False])
    assert all(isvalid), (
        f'The {kind} with id "{id}" has an invalid id. The characters '
        f'"{", ".join(violated_characters)}" are not allowed.'
    )


def _convert_literal_value(value):
    """
    Convert an RDF literal value to its appropriate Python type.

    When RDF literals are loaded without explicit datatypes, they come back as strings.
    This function attempts to convert them to the appropriate Python type.
    Also unwraps single-element lists to scalar values for backward compatibility.

    Args:
        value: The literal value to convert (typically a string, but may already be parsed)

    Returns:
        The converted value with appropriate Python type (int, float, bool, dict, list, or str)
    """
    if value is None:
        return None

    # If it's already a list (e.g., rdflib parsed JSON), unwrap single-element lists
    if isinstance(value, list):
        if len(value) == 1:
            return value[0]
        return value

    # If it's another non-string type (dict, int, float, bool), return as-is
    if not isinstance(value, str):
        return value

    # Try to parse as JSON first (for dicts/lists)
    if value.startswith("{") or value.startswith("["):
        try:
            parsed = json.loads(value)
            # Unwrap single-element lists to scalar values for backward compatibility
            # This handles cases where tensor Parameters were serialized as [value]
            # but the component constructor expects a scalar
            if isinstance(parsed, list) and len(parsed) == 1:
                return parsed[0]
            return parsed
        except (json.JSONDecodeError, ValueError):
            pass

    # Handle boolean values
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False

    # Handle None
    if value == "None":
        return None

    # Try to convert to int first (more specific)
    try:
        # Check if it's a valid integer representation
        if "." not in value and "e" not in value.lower():
            return int(value)
    except ValueError:
        pass

    # Try to convert to float
    try:
        return float(value)
    except ValueError:
        pass

    # Return as string if no conversion succeeded
    return value


@autoreset_print
class SimulationModel:
    r"""
    A simulation model for building digital twins.

    This class manages component collections, connections between components, cycle removal
    for feedback control loops, and topological sorting to determine optimal execution
    order for simulation.

    Args:
        id: Unique identifier for the model.
        dir_conf: List of directories to store model-related files.

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
        "_translator",
        "_rewire_reports",
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
            id: Unique identifier for the model.
            dir_conf: List of directories to store model-related files.
        Raises:
            AssertionError: If the id is not a string or contains invalid characters.
        """
        self._id = id
        if dir_conf is None:
            self._dir_conf = ["generated_files", "models", self._id, "simulation_model"]
        else:
            self._dir_conf = dir_conf

        _check_id(id, kind="model")
        self._id = id
        self._components = {}
        self._execution_order = []
        self._flat_execution_order = []
        self._required_initialization_connections = []
        self._components_no_cycles = {}
        self._saved_parameters = {}
        self._custom_initial_dict = None
        self._is_loaded = False
        self._is_validated = False
        self._rewire_reports = {}

        self._semantic_model = core.SemanticModel(
            id=self._id,
            namespaces={
                "T4B": core.namespace.T4B,
                # "SAREF": core.namespace.SAREF,
                # "S4BLDG": core.namespace.S4BLDG,
                "S4SYST": core.namespace.S4SYST,
            },
            dir_conf=self._dir_conf + ["semantic_model"],
        )

        self._translator = None

    @property
    def components(self) -> dict:
        return self._components

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @property
    def is_validated(self) -> bool:
        return self._is_validated

    @property
    def rewire_reports(self) -> dict:
        """Per-CITS reports produced by the most recent :meth:`rewire` call."""
        return self._rewire_reports

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
        self._semantic_model.dir_conf = dir_conf + ["semantic_model"]

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
            self._update_literals([component])

        self._is_loaded = False

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
            if "fmu" in get_obj_attr(fmu):
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
                tensor = (
                    tensor.detach().clone().requires_grad_(False).type(torch.float64)
                )

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
                    connection.output_port,
                    connection_point.input_port,
                )

        # Connection from component
        for connection in component.connected_through.copy():
            for connection_point in connection.connects_system_at.copy():
                self.remove_connection(
                    component,
                    connection_point.connection_point_of,
                    connection.output_port,
                    connection_point.input_port,
                )

        if components is None:
            components = self._components

        if components is self._components:
            component_uri = self._semantic_model.T4B.__getitem__(component.id)
            self._semantic_model.instance_graph.remove((component_uri, None, None))
            self._semantic_model.instance_graph.remove((None, None, component_uri))

        del components[component.id]
        self._is_loaded = False

    @staticmethod
    def _resolve_port_index(
        port_index: Optional[Union[int, torch.Tensor]],
        this_port: Union[tps.Scalar, tps.Vector],
        other_port: Union[tps.Scalar, tps.Vector],
        this_port_name: str,
        other_port_name: str,
    ) -> Optional[Union[int, torch.Tensor]]:
        """
        Validate and resolve a port index for connections.

        Returns the appropriate index value: the provided index if valid,
        a generated range for vector-to-vector mappings, or None for scalars.
        """
        if port_index is not None:
            assert isinstance(
                this_port, tps.Vector
            ), f"If {this_port_name} port index is set, {this_port_name} port must be a vector"
            assert isinstance(
                port_index, (torch.Tensor, int)
            ), f"If {this_port_name} port index is set, it must either be an integer or a torch.Tensor"
            if isinstance(port_index, torch.Tensor):
                assert isinstance(other_port, tps.Vector), (
                    f"If {this_port_name} port index is set and is a torch.Tensor, "
                    f"{other_port_name} port must be a vector"
                )
            else:
                # ``int`` ``port_index`` selects one slot of this Vector port.
                # The opposite side may be either a Scalar (slot value flows
                # straight to/from a scalar peer) or a Vector with its own
                # explicit ``int`` slot (single-slot-to-single-slot bridge,
                # e.g. CITS.inputSignal[0] -> AHU.supplyDamperPosition[3]).
                # The translator's ``resolve_port_indices`` populates both
                # ints in that Vector->Vector case after Vector port slot
                # ordinals are resolved on both ends.
                assert isinstance(other_port, (tps.Scalar, tps.Vector)), (
                    f"If {this_port_name} port index is set and is an integer, "
                    f"{other_port_name} port must be a scalar or vector, got "
                    f"{other_port.__class__.__name__}"
                )
            return port_index
        else:
            if isinstance(other_port, tps.Vector) and isinstance(this_port, tps.Vector):
                return torch.arange(this_port.size)  # Map directly
            else:
                assert isinstance(this_port, tps.Scalar), (
                    f"If {this_port_name} port index is not set, both output and input ports "
                    f"must be scalars. Got {other_port_name} port type {other_port.__class__.__name__} "
                    f"and {this_port_name} port type {this_port.__class__.__name__}"
                )
                return None

    def add_connection(
        self,
        sender_component: core.System,
        receiver_component: core.System,
        output_port: str,
        input_port: str,
        output_port_index: [int, torch.Tensor] = None,
        input_port_index: [int, torch.Tensor] = None,
        components: Dict[str, core.System] = None,
    ) -> None:
        """
        Add a connection between two components in the system.

        Args:
            sender_component (core.System): The component sending the connection.
            receiver_component (core.System): The component receiving the connection.
            output_port (str): Name of the sender property.
            input_port (str): Name of the receiver property.
        Raises:
            AssertionError: If property names are invalid for the components.
            AssertionError: If a connection already exists.
        """
        if components is None:
            components = self._components

        self.add_component(sender_component, components=components)
        self.add_component(receiver_component, components=components)

        l = [f"'{k}'" for k in list(sender_component.output.keys())]
        message = f"The property '{output_port}' is not a valid output for the component '{sender_component.id}' of type '{type(sender_component)}'.\nThe valid output properties are:\n{' '.join(l)}"
        assert output_port in (
            set(sender_component.output.keys()) | set(sender_component.output.keys())
        ), message  # Before we joined input and output sets

        l = [f"'{k}'" for k in list(receiver_component.input.keys())]
        message = f"The property '{input_port}' is not a valid input for the component '{receiver_component.id}' of type '{type(receiver_component)}'.\nThe valid input properties are:\n{' '.join(l)}"
        assert input_port in receiver_component.input.keys(), message

        found_connection_point = False
        # Check if there already is a connectionPoint with the same receiver_property_name
        for receiver_component_connection_point in receiver_component.connects_at:
            if receiver_component_connection_point.input_port == input_port:
                found_connection_point = True
                break

        found_connection = False
        # Check if there already is a connection with the same sender_property_name
        for sender_obj_connection in sender_component.connected_through:
            if sender_obj_connection.output_port == output_port:
                found_connection = True
                break

        if found_connection_point and found_connection:
            message = f'core.Connection between "{sender_component.id}" and "{receiver_component.id}" with the properties "{output_port}" and "{input_port}" already exists.'
            assert (
                receiver_component_connection_point
                not in sender_obj_connection.connects_system_at
            ), message

        if found_connection == False:
            sender_obj_connection = core.Connection(
                connects_system=sender_component, output_port=output_port
            )
            sender_component.connected_through.append(sender_obj_connection)

        if found_connection_point == False:
            receiver_component_connection_point = core.ConnectionPoint(
                connection_point_of=receiver_component, input_port=input_port
            )
            receiver_component.connects_at.append(receiver_component_connection_point)

        sender_obj_connection.connects_system_at.append(
            receiver_component_connection_point
        )
        receiver_component_connection_point.connects_system_through.append(
            sender_obj_connection
        )  # if sender_obj_connection not in receiver_component_connection_point.connects_system_through else None

        input_idx = self._resolve_port_index(
            input_port_index,
            receiver_component.input[input_port],
            sender_component.output[output_port],
            "input",
            "output",
        )
        receiver_component_connection_point.set_input_port_index(
            sender_obj_connection, input_idx
        )

        output_idx = self._resolve_port_index(
            output_port_index,
            sender_component.output[output_port],
            receiver_component.input[input_port],
            "output",
            "input",
        )
        receiver_component_connection_point.set_output_port_index(
            sender_obj_connection, output_idx
        )

        if components == self._components:
            sender_component_uri = self._semantic_model.T4B.__getitem__(
                sender_component.id
            )
            receiver_component_uri = self._semantic_model.T4B.__getitem__(
                receiver_component.id
            )

            sender_component_class_name = sender_component.__class__.__name__
            receiver_component_class_name = receiver_component.__class__.__name__

            connection_uri = self._semantic_model.T4B.__getitem__(
                str(hash(sender_obj_connection))
            )
            connection_point_uri = self._semantic_model.T4B.__getitem__(
                str(hash(receiver_component_connection_point))
            )

            literal_sender_property = Literal(
                output_port
            )  # , datatype=core.namespace.XSD.string)
            literal_receiver_property = Literal(
                input_port
            )  # , datatype=core.namespace.XSD.string)

            # Add the class of the components to the semantic model
            self._semantic_model.instance_graph.add(
                (
                    sender_component_uri,
                    RDF.type,
                    core.namespace.T4B.__getitem__(sender_component_class_name),
                )
            )
            self._semantic_model.instance_graph.add(
                (
                    receiver_component_uri,
                    RDF.type,
                    core.namespace.T4B.__getitem__(receiver_component_class_name),
                )
            )

            self._semantic_model.instance_graph.add(
                (
                    core.namespace.T4B.__getitem__(sender_component_class_name),
                    RDFS.subClassOf,
                    core.namespace.S4SYST.System,
                )
            )
            self._semantic_model.instance_graph.add(
                (
                    core.namespace.T4B.__getitem__(receiver_component_class_name),
                    RDFS.subClassOf,
                    core.namespace.S4SYST.System,
                )
            )

            # Add the class of the connections and connection points to the semantic model
            self._semantic_model.instance_graph.add(
                (connection_uri, RDF.type, core.namespace.S4SYST.Connection)
            )
            self._semantic_model.instance_graph.add(
                (connection_point_uri, RDF.type, core.namespace.S4SYST.ConnectionPoint)
            )

            # Add the forward connection to the semantic model
            self._semantic_model.instance_graph.add(
                (
                    sender_component_uri,
                    core.namespace.S4SYST.connectedThrough,
                    connection_uri,
                )
            )
            self._semantic_model.instance_graph.add(
                (
                    connection_uri,
                    core.namespace.S4SYST.connectsSystemAt,
                    connection_point_uri,
                )
            )
            self._semantic_model.instance_graph.add(
                (
                    connection_point_uri,
                    core.namespace.S4SYST.connectionPointOf,
                    receiver_component_uri,
                )
            )

            # Add the reverse connection to the semantic model
            self._semantic_model.instance_graph.add(
                (
                    connection_uri,
                    core.namespace.S4SYST.connectsSystem,
                    sender_component_uri,
                )
            )
            self._semantic_model.instance_graph.add(
                (
                    connection_point_uri,
                    core.namespace.S4SYST.connectsSystemThrough,
                    connection_uri,
                )
            )
            self._semantic_model.instance_graph.add(
                (
                    receiver_component_uri,
                    core.namespace.S4SYST.connectsAt,
                    connection_point_uri,
                )
            )

            self._update_literals(
                components=[sender_component, receiver_component],
                connections=[sender_obj_connection],
                connection_points=[receiver_component_connection_point],
            )

            # self._semantic_model.instance_graph.add(
            #     (connection_uri, core.namespace.T4B.output_port, literal_sender_property)
            # )
            # self._semantic_model.instance_graph.add(
            #     (
            #         connection_point_uri,
            #         core.namespace.T4B.input_port,
            #         literal_receiver_property,
            #     )
            # )

            # # Add the input_port_index and output_port_index literals
            # # Serialize indices as JSON with connection hash as keys
            # input_port_index_dict = {
            #     str(hash(conn)): int(idx) if isinstance(idx, (int, torch.Tensor)) else idx
            #     for conn, idx in receiver_component_connection_point.input_port_index.items()
            # }
            # output_port_index_dict = {
            #     str(hash(conn)): int(idx) if isinstance(idx, (int, torch.Tensor)) else idx
            #     for conn, idx in receiver_component_connection_point.output_port_index.items()
            # }
            # literal_input_port_index = Literal(
            #     json.dumps(input_port_index_dict), datatype=core.namespace.RDF.JSON
            # )
            # literal_output_port_index = Literal(
            #     json.dumps(output_port_index_dict), datatype=core.namespace.RDF.JSON
            # )
            # self._semantic_model.instance_graph.add(
            #     (
            #         connection_point_uri,
            #         core.namespace.T4B.input_port_index,
            #         literal_input_port_index,
            #     )
            # )
            # self._semantic_model.instance_graph.add(
            #     (
            #         connection_point_uri,
            #         core.namespace.T4B.output_port_index,
            #         literal_output_port_index,
            #     )
            # )

        self._is_loaded = False

    def remove_connection(
        self,
        sender_component: core.System,
        receiver_component: core.System,
        output_port: str,
        input_port: str,
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
            if connection.output_port == output_port:
                sender_component_connection = connection
                break
        if sender_component_connection is None:
            raise ValueError(
                f'The sender component "{sender_component.id}" does not have a connection with the property "{output_port}"'
            )

        receiver_component_connection_point = None
        for connection_point in receiver_component.connects_at:
            if connection_point.input_port == input_port:
                receiver_component_connection_point = connection_point
                break
        if receiver_component_connection_point is None:
            raise ValueError(
                f'The receiver component "{receiver_component.id}" does not have a connection point with the property "{input_port}"'
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
            sender_component_uri = self._semantic_model.T4B.__getitem__(
                sender_component.id
            )
            receiver_component_uri = self._semantic_model.T4B.__getitem__(
                receiver_component.id
            )

            connection_uri = self._semantic_model.T4B.__getitem__(
                str(hash(sender_component_connection))
            )  # self._semantic_model.T4B.__getitem__(sender_component.id + " " + sender_property_name)
            connection_point_uri = self._semantic_model.T4B.__getitem__(
                str(hash(receiver_component_connection_point))
            )  # self._semantic_model.T4B.__getitem__(receiver_component.id + " " + receiver_property_name)

            literal_sender_property = list(
                self._semantic_model.instance_graph.objects(
                    connection_uri, core.namespace.T4B.output_port
                )
            )
            literal_receiver_property = list(
                self._semantic_model.instance_graph.objects(
                    connection_point_uri, core.namespace.T4B.input_port
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
            self._semantic_model.instance_graph.remove(
                (
                    connection_uri,
                    core.namespace.S4SYST.connectsSystemAt,
                    connection_point_uri,
                )
            )
            self._semantic_model.instance_graph.remove(
                (
                    connection_point_uri,
                    core.namespace.S4SYST.connectsSystemThrough,
                    connection_uri,
                )
            )

            if len(sender_component_connection.connects_system_at) == 0:
                self._semantic_model.instance_graph.remove(
                    (
                        sender_component_uri,
                        core.namespace.S4SYST.connectedThrough,
                        connection_uri,
                    )
                )
                self._semantic_model.instance_graph.remove(
                    (
                        connection_uri,
                        core.namespace.S4SYST.connectsSystem,
                        sender_component_uri,
                    )
                )
                self._semantic_model.instance_graph.remove(
                    (
                        connection_uri,
                        core.namespace.T4B.output_port,
                        literal_sender_property,
                    )
                )

            if len(receiver_component_connection_point.connects_system_through) == 0:
                self._semantic_model.instance_graph.remove(
                    (
                        receiver_component_uri,
                        core.namespace.S4SYST.connectsAt,
                        connection_point_uri,
                    )
                )
                self._semantic_model.instance_graph.remove(
                    (
                        connection_point_uri,
                        core.namespace.S4SYST.connectionPointOf,
                        receiver_component_uri,
                    )
                )
                self._semantic_model.instance_graph.remove(
                    (
                        connection_point_uri,
                        core.namespace.T4B.input_port,
                        literal_receiver_property,
                    )
                )
        self._is_loaded = False

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

    # def set_custom_initial_dict(
    #     self, _custom_initial_dict: Dict[str, Dict[str, Any]]
    # ) -> None:
    #     """
    #     Set custom initial values for components.

    #     Args:
    #         _custom_initial_dict (Dict[str, Dict[str, Any]]): Dictionary of custom initial values.

    #     Raises:
    #         AssertionError: If unknown component IDs are provided.
    #     """
    #     np_custom_initial_dict_ids = np.array(list(_custom_initial_dict.keys()))
    #     legal_ids = np.array(
    #         [dict_id in self._components for dict_id in _custom_initial_dict]
    #     )
    #     assert np.all(
    #         legal_ids
    #     ), f'Unknown component id(s) provided in "_custom_initial_dict": {np_custom_initial_dict_ids[legal_ids==False]}'
    #     self._custom_initial_dict = _custom_initial_dict

    def set_initial_values(
        self,
        values: List[Any] = None,
        components: List[core.System] = None,
        output_names: List[str] = None,
        **kwargs,
    ) -> None:
        """
        Set initial values for components in the model.

        Args:
            values (List[Any]): List of initial values to set.
            components (List[core.System]): List of components to set initial values for.
            output_names (List[str]): List of output property names corresponding to the values.

        Raises:
            AssertionError: If a component doesn't have the specified output property.
        """
        # Handle deprecated dict-based signature: set_initial_values(dict_)
        # Old format: dict_ = {component_id: {output_name: value, ...}, ...}
        old_dict = kwargs.get("dict_", None)
        if old_dict is None and isinstance(values, dict):
            old_dict = values
            values = None

        if old_dict is not None:
            warnings.warn(
                "The dict-based signature for set_initial_values(dict_) is deprecated. "
                "Use set_initial_values(values, components, output_names) instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            # Convert old dict format to new list format
            values = []
            components = []
            output_names = []
            for component_id, outputs in old_dict.items():
                component = self._components[component_id]
                for output_name, value in outputs.items():
                    values.append(value)
                    components.append(component)
                    output_names.append(output_name)

        for v, component, output_name in zip(values, components, output_names):
            assert (
                output_name in component.output
            ), f'Invalid output property "{output_name}" for component "{component.id}"'

            output_obj = component.output[output_name]
            if v is not None:
                # Set the tensor value for Scalar or Vector types
                if isinstance(output_obj, tps.Scalar):
                    output_obj.tensor = (
                        v
                        if isinstance(v, torch.Tensor)
                        else torch.tensor([v], dtype=torch.float64)
                    )
                elif isinstance(output_obj, tps.Vector):
                    output_obj.tensor = (
                        v
                        if isinstance(v, torch.Tensor)
                        else torch.tensor(v, dtype=torch.float64)
                    )
                else:
                    raise TypeError(
                        f'Output property "{output_name}" for component "{component.id}" '
                        f"is not a Scalar or Vector type"
                    )

    def set_parameters(
        self,
        values: List[Any],
        components: List[core.System],
        parameter_names: List[str],
        min_values: List[Any] = None,
        max_values: List[Any] = None,
        normalized: List[bool] = None,
        overwrite: bool = False,
        save_original: bool = False,
    ) -> None:
        """
        Set parameters for components from an array.

        Args:
            values (List[Any]): List of parameter values.
            components (List[core.System]): List of components to set parameters for.
            parameter_names (List[str]): List of attribute names corresponding to the parameters.
            normalized (List[bool]): List of booleans indicating if values are normalized.
            overwrite (bool): Whether to overwrite existing parameters.
            save_original (bool): Whether to save original parameters for later restoration.

        Raises:
            AssertionError: If a component doesn't have the specified attribute.
        """

        if normalized is None:
            normalized = [False] * len(values)
        elif isinstance(normalized, bool):
            normalized = [normalized] * len(values)

        # assert that min_values and max_values are either both None or both not None
        assert (min_values is None and max_values is None) or (
            min_values is not None and max_values is not None
        ), "min_values and max_values must both be None or both not None"

        if min_values is not None and max_values is not None:
            # Assert that the min_values and max_values are the same length as the values
            assert len(min_values) == len(
                values
            ), f"The length of min_values must be the same as the length of values. Got {len(min_values)} and {len(values)}"
            assert len(max_values) == len(
                values
            ), f"The length of max_values must be the same as the length of values. Got {len(max_values)} and {len(values)}"

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
                    if min_values is not None:
                        obj_.min_value = min_values[i]
                    if max_values is not None:
                        obj_.max_value = max_values[i]
                    if overwrite:
                        if save_original:
                            obj_key = id(obj)
                            if obj_key not in self._saved_parameters:
                                self._saved_parameters[obj_key] = {"__ref__": obj}
                            self._saved_parameters[obj_key][attr] = obj_

                        # Reconcile ``v`` with the *current* n_c of ``obj_``.
                        # The estimator captures n_c at parameter-list
                        # processing time, which happens BEFORE
                        # :meth:`SimulationModel.initialize`.  Sub-components
                        # like the AHU dampers get ``expand_to_n_c(n_branches)``
                        # inside their owner's ``initialize`` -- so by the time
                        # we land here ``obj_.min_value`` has shape ``(n_c,)``
                        # with ``n_c > 1`` even though the theta entry the
                        # solver hands back is scalar.  ``tps.TensorParameter``
                        # then infers ``n_c=1`` from a scalar ``v`` and refuses
                        # the ``(n_c,)`` bound (see
                        # :func:`_prepare_bound_value`).  Broadcasting ``v`` to
                        # the existing ``n_c`` matches the auto-discovery
                        # semantics: ``get_estimable_parameters`` emits one
                        # scalar bound, so every parallel branch shares the
                        # same denormalized value.  Callers wanting per-branch
                        # estimation must pass the vector form
                        # ``(comp, attr, [x0]*n_c, [lb]*n_c, [ub]*n_c)``.
                        target_n_c = getattr(obj_, "n_c", 1) or 1
                        if target_n_c > 1:
                            v_t = torch.as_tensor(v, dtype=torch.float64).reshape(-1)
                            if v_t.numel() == 1:
                                v_t = v_t.expand(target_n_c).clone()
                            elif v_t.numel() != target_n_c:
                                raise ValueError(
                                    f"Cannot reconcile value of shape "
                                    f"{tuple(v_t.shape)} with parameter "
                                    f"'{attr}' on '{obj.id}' (n_c="
                                    f"{target_n_c}).  Pass per-branch x0/lb/ub "
                                    f"as lists if you need distinct values."
                                )
                            v = v_t

                        new_param = tps.TensorParameter(
                            v,
                            min_value=obj_.min_value,
                            max_value=obj_.max_value,
                            normalized=normalized_,
                            scaling=getattr(obj_, "_scaling", "linear"),
                        )
                        rdelattr(obj, attr)
                        rsetattr(obj, attr, new_param)
                    else:
                        obj_.set(v, normalized=normalized_)
                elif isinstance(obj_, tps.TensorParameter):
                    obj_.set(v, normalized=normalized_)
                else:
                    rsetattr(obj, attr, v)

    def set_parameters_from_array(self, *args, **kwargs) -> None:
        """
        Deprecated: Use set_parameters instead.
        """
        warnings.warn(
            "Method 'set_parameters_from_array' is deprecated. Use 'set_parameters' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.set_parameters(*args, **kwargs)

    def restore_parameters(self, keep_values: bool = True) -> None:
        for obj_key in self._saved_parameters:
            saved = self._saved_parameters[obj_key]
            component = saved["__ref__"]
            for attr in saved:
                if attr == "__ref__":
                    continue
                old_obj = rgetattr(component, attr)
                v = old_obj.get()
                new_obj = saved[attr]
                rdelattr(component, attr)
                rsetattr(component, attr, new_obj)
                if keep_values:
                    new_obj.set(v, normalized=False)

    def set_dbconfigs(self, dbconfig: Dict[str, Any]) -> "SimulationModel":
        """Apply a database configuration to every component that supports it.

        Walks ``self.components.values()`` and calls
        ``component.set_dbconfig(dbconfig)`` on every component that exposes
        such a method.  Components without ``set_dbconfig`` are silently
        skipped, so this is safe to call on heterogeneous models -- only
        e.g. :class:`SensorSystem` instances are affected.

        Components that already carry a non-``None`` ``dbconfig`` are
        overwritten (the call is non-idempotent for non-None starting
        states); pass per-component ``dbconfig=None`` arguments via
        :meth:`set_parameters` if you need to clear or vary per-component.

        Args:
            dbconfig: The database configuration dict (typically
                ``{"table_name": ..., "db_host": ..., "db_port": ...,
                "db_name": ..., "db_user": ..., "db_password": ...}``).

        Returns:
            ``self`` for chaining.
        """
        for comp in self._components.values():
            setter = getattr(comp, "set_dbconfig", None)
            if callable(setter):
                setter(dbconfig)
        return self

    def fill_missing_inputs(
        self,
        defaults: Dict[Any, Any],
    ) -> "SimulationModel":
        """Attach providers for input ports the ontology did not wire.

        Each ``defaults`` entry says "this missing port should be driven
        by that source" -- with two axes of flexibility:

        * **Scope** (the key): apply to every component declaring a port
          name (port-only) OR to a specific component (component-scoped).
        * **Provider** (the value): a constant (wrapped in a flat
          :class:`ScheduleSystem`) OR a user-supplied :class:`core.System`
          (e.g. a :class:`SensorSystem` reading a BMS historian by
          ``uuid``) OR an explicit ``(component, output_port)`` pair when
          the provider has more than one output.

        Key shapes
        ----------
        * ``"port_name"`` -- port-only entry.  Connects every component
          that:

            1. declares an input port called ``port_name``
               (``port_name in component.input``), and
            2. has no incoming connection on that port yet, and
            3. is not covered by a component-scoped entry below.

          Idempotent across calls: the schedule id is
          ``_sched_{port_name}``, looked up before re-creating.
        * ``(component_or_id, port_name)`` -- component-scoped entry.
          ``component_or_id`` may be a :class:`core.System` instance or
          its id string.  Takes precedence over a port-only entry on
          the same port (so a building-wide constant can coexist with
          a per-AHU override).  The auto-generated schedule id is
          ``_sched_{component_id}_{port_name}`` so each scoped entry
          gets its own provider instance even if multiple components
          map the same port to the same value.

        Value shapes
        ------------
        * ``float | int | bool`` -- wrapped in a flat
          :class:`ScheduleSystem` whose ``scheduleValue`` output is
          ``value`` at every timestep.
        * :class:`core.System` -- used directly as the provider; the
          component is added to :attr:`self._components` if it isn't
          already.  The provider must have **exactly one** output
          port; that port is auto-selected.  Useful for plugging in a
          pre-built :class:`SensorSystem` (``uuid`` + ``dbconfig`` set)
          that reads a BMS historian: the connection is then a true
          exogenous time-series input, even though the BRICK ontology
          carries no equivalent URI.
        * ``(provider, output_port_name)`` -- when the provider has
          multiple outputs.  ``output_port_name`` must be a key of
          ``provider.output``.

        Conflict resolution
        -------------------
        Component-scoped entries always win over port-only entries on
        the same port; this lets a project-wide constant (e.g.
        ``"outdoorCO2": 400.0``) coexist with a per-component override
        (e.g. ``(ahu02, "supplyAirTemperatureSetpoint"): leaf_sensor``)
        without the user having to maintain two parallel dicts.  If
        the same ``(component_id, port_name)`` pair appears twice the
        later value wins -- standard ``dict`` semantics.

        Args:
            defaults: Mapping with the key/value shapes above.

        Returns:
            ``self`` for chaining.
        """

        def _has_incoming(component: core.System, input_port: str) -> bool:
            for cp in component.connects_at:
                if cp.input_port == input_port and cp.connects_system_through:
                    return True
            return False

        def _make_schedule(value: float, id_: str) -> core.System:
            # Empty rulesets + ``ruleset_default_value`` is the canonical
            # ``flat schedule'' shape in twin4build (see ScheduleSystem
            # docstring for the ruleset semantics).
            return systems.ScheduleSystem(
                weekDayRulesetDict={
                    "ruleset_default_value": value,
                    "ruleset_start_minute": [],
                    "ruleset_end_minute": [],
                    "ruleset_start_hour": [],
                    "ruleset_end_hour": [],
                    "ruleset_value": [],
                },
                add_noise=False,
                id=id_,
            )

        def _resolve_spec(
            spec: Any,
            schedule_id: str,
        ) -> tuple[core.System, str]:
            """Normalise a value-side spec to ``(provider, output_port)``.

            * Scalar -> create or reuse a flat ``ScheduleSystem`` with
              id ``schedule_id`` and emit ``scheduleValue``.
            * ``core.System`` -> use the unique output port (assert).
            * ``(System, output_port)`` -- use as-is after validating
              the output port exists.
            """
            # ``bool`` is a subclass of ``int`` in Python so the
            # ``(int, float)`` check below catches flat 0/1 enable/
            # disable schedules too -- intentional.
            if isinstance(spec, (int, float)):
                sched = self._components.get(schedule_id)
                if sched is None:
                    sched = _make_schedule(float(spec), schedule_id)
                return sched, "scheduleValue"
            if isinstance(spec, core.System):
                out_keys = list(spec.output.keys())
                assert len(out_keys) == 1, (
                    f"fill_missing_inputs: provider {spec.id!r} has "
                    f"{len(out_keys)} outputs ({out_keys!r}); pass a "
                    "(provider, output_port) tuple to disambiguate."
                )
                return spec, out_keys[0]
            if isinstance(spec, tuple) and len(spec) == 2:
                provider, out_port = spec
                assert isinstance(provider, core.System), (
                    "fill_missing_inputs: in a (provider, output_port) "
                    "tuple the first element must be a core.System "
                    f"instance, got {type(provider).__name__}."
                )
                assert out_port in provider.output, (
                    f"fill_missing_inputs: provider {provider.id!r} has "
                    f"no output port {out_port!r}; valid keys are "
                    f"{list(provider.output)}."
                )
                return provider, out_port
            raise TypeError(
                f"fill_missing_inputs: unsupported value spec "
                f"{type(spec).__name__}({spec!r}).  Use a scalar, a "
                "core.System instance, or a (component, output_port) "
                "tuple."
            )

        # Split entries by scope so port-only fan-out can skip any
        # (component, port) pair already overridden by a scoped entry.
        port_defaults: Dict[str, Any] = {}
        comp_defaults: Dict[tuple[str, str], Any] = {}
        for key, spec in defaults.items():
            if isinstance(key, str):
                port_defaults[key] = spec
                continue
            if isinstance(key, tuple) and len(key) == 2:
                comp_ref, port_name = key
                if isinstance(comp_ref, core.System):
                    comp_id = comp_ref.id
                elif isinstance(comp_ref, str):
                    comp_id = comp_ref
                else:
                    raise TypeError(
                        "fill_missing_inputs: component-scoped key "
                        "expects (System | id_str, port_name); got "
                        f"({type(comp_ref).__name__}, {type(port_name).__name__})."
                    )
                if not isinstance(port_name, str):
                    raise TypeError(
                        "fill_missing_inputs: component-scoped key's "
                        f"port_name must be a str, got {type(port_name).__name__}."
                    )
                comp_defaults[(comp_id, port_name)] = spec
                continue
            raise TypeError(
                f"fill_missing_inputs: unsupported key {key!r}.  Use a "
                "port-name str, or a (component | component_id, port_name) "
                "tuple."
            )

        # ---- Pass 1: port-only fan-out, skipping component-scoped overrides
        for port_name, spec in port_defaults.items():
            consumers = [
                comp
                for comp in list(self._components.values())
                if port_name in getattr(comp, "input", {})
                and not _has_incoming(comp, port_name)
                and (comp.id, port_name) not in comp_defaults
            ]
            if not consumers:
                continue
            provider, out_port = _resolve_spec(spec, f"_sched_{port_name}")
            for consumer in consumers:
                self.add_connection(provider, consumer, out_port, port_name)

        # ---- Pass 2: per-component overrides
        for (comp_id, port_name), spec in comp_defaults.items():
            consumer = self._components.get(comp_id)
            assert consumer is not None, (
                f"fill_missing_inputs: no component with id {comp_id!r} "
                "in this model; check the component reference or run "
                "this method after the component has been translated / "
                "added."
            )
            assert port_name in getattr(consumer, "input", {}), (
                f"fill_missing_inputs: component {comp_id!r} has no "
                f"input port {port_name!r}; valid ports are "
                f"{list(getattr(consumer, 'input', {}))}."
            )
            if _has_incoming(consumer, port_name):
                # Component-scoped override targeting an already-wired
                # port is almost certainly a config bug: the user asked
                # to drive a port that the ontology already populated.
                # Loudly skip rather than silently double-connect (the
                # downstream ``add_connection`` would assert anyway).
                LOGGER.warn(
                    "fill_missing_inputs: %s.%s already has an incoming "
                    "connection; component-scoped override skipped.",
                    comp_id, port_name,
                )
                continue
            provider, out_port = _resolve_spec(
                spec, f"_sched_{comp_id}_{port_name}"
            )
            self.add_connection(provider, consumer, out_port, port_name)

        return self

    def rewire(
        self,
        *,
        start_time: Any,
        end_time: Any,
        step_size: int,
        mode: str = "train",
        **rewire_kwargs: Any,
    ) -> "SimulationModel":
        """Run the data-driven CITS rewire on every PI-CITS in the graph.

        Should be called **before** :meth:`load`: the rewire modifies
        the topology (prunes losing sensor connections, repins CITS
        frozen state) which would otherwise invalidate the execution
        order computed by :meth:`load`.  It only needs the components
        and connections produced by :class:`~twin4build.translator.Translator`
        plus a configured dbconfig on the sensors -- no execution
        order, validated graph, or loaded parameters are required.

        Convenience entrypoint that dispatches to the internal
        :func:`_rewire_pi_loops` helper in
        :mod:`twin4build.systems.controller.controller_identification.pi_loop_rewire`.
        The helper:

          * loads timeseries for every connected
            :class:`SensorSystem` over the requested window,
          * scores every wired ``(sensor, setpoint)`` pair against
            the downstream actuator measurement,
          * prunes losing connections (so the surviving
            :class:`ControllerIdentificationPITorchSystem` has
            ``n_sensors = n_setpoints = 1`` and a single PI candidate),
          * writes data-driven seeds (``kp``, ``Ti``, ``output_min``,
            ``output_max``, ``default_output_0``, ``isReverse``,
            ``gate_0.threshold``, ``gate_0.band``, ``gamma_gate_0``)
            onto the surviving candidate,
          * pins ``alpha_0`` / ``beta_0`` / ``gamma_0`` / ``beta_b_0``
            to one-hot, ``gate_0.polarity`` to ``1.0`` and
            ``alpha_gate_{a}`` according to ``mode``.

        Per-CITS ``RewireReport`` objects are stored on
        ``self._rewire_reports`` for downstream inspection.

        Args:
            start_time, end_time, step_size: Window passed to
                :meth:`SensorSystem.initialize` so the rewire can
                read measurement values.
            mode: ``"train"`` -> ``alpha_gate_{a} = 1.0`` (gate active
                during Stage-1 estimation); ``"simulate"`` ->
                ``alpha_gate_{a} = 0.0`` (gate bypassed for Stage-2
                closed-loop simulation, PI passthrough).
            **rewire_kwargs: Forwarded verbatim to
                :func:`_rewire_pi_loops` (confidence thresholds,
                decade-pad widths, candidate filters, ...).  See the
                helper's docstring for the full list.

        Returns:
            ``self`` for chaining.
        """
        # Local import keeps the simulation-model module import-cycle-
        # free; the rewire helper depends on the loop-classifier,
        # actuator GMM, ... module graph which would otherwise pull
        # heavy dependencies into every ``SimulationModel`` import.
        from twin4build.systems.controller.controller_identification.pi_loop_rewire import (
            _rewire_pi_loops,
        )

        reports = _rewire_pi_loops(
            self,
            start_time=start_time,
            end_time=end_time,
            step_size=step_size,
            mode=mode,
            **rewire_kwargs,
        )
        self._rewire_reports = reports
        return self

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
        start_time: Optional[datetime.datetime] = None,
        end_time: Optional[datetime.datetime] = None,
        step_size: Optional[int] = None,
        simulator: Optional[core.Simulator] = None,
    ) -> None:
        """
        Cache data and create folder structure for time series data.

        Args:
            start_time (Optional[datetime.datetime]): Start time for caching.
            end_time (Optional[datetime.datetime]): End time for caching.
            step_size (Optional[int]): Time step size for caching.
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
                start_time=start_time,
                end_time=end_time,
                step_size=step_size,
                simulator=simulator,
            )

    def initialize(
        self,
        start_time: List[datetime.datetime],
        end_time: List[datetime.datetime],
        step_size: List[int],
        # simulator: Optional[core.Simulator] = None,
    ) -> None:
        """
        Initialize the model for simulation.

        Args:
            start_time (datetime.datetime): Start time for the simulation.
            end_time (datetime.datetime): End time for the simulation.
            step_size (int): Time step size for the simulation.
        """
        assert (
            self._is_loaded
        ), "The model is not loaded and cannot be simulated. Please call the load method first."

        # assert isinstance(
        #     simulator, core.Simulator
        # ), "simulator must be a core.Simulator object"

        # Validate and format as lists if needed
        # start_time, end_time, step_size = validate_period(start_time, end_time, step_size)

        # self.set_initial_values()
        for component in self._flat_execution_order:
            # for v in component.input.values():
            #     v.reset()

            # for v in component.output.values():
            #     v.reset()

            # Make the inputs and outputs aware of the execution order.
            # This is important to ensure that input tps.Vectors have the same order, allowing for instance element-wise operations.
            # for i, connection_point in enumerate(component.connects_at):

            #     update_input_port_index = False
            #     hash_array = torch.arange(
            #         len(connection_point.connects_system_through), dtype=torch.int64
            #     )
            #     for j, connection in enumerate(
            #         connection_point.connects_system_through
            #     ):
            #         connected_component = connection.connects_system
            #         if (
            #             isinstance(
            #                 component.input[connection_point.input_port], tps.Vector
            #             )
            #             and self._translator is not None
            #             and (
            #                 component,
            #                 connected_component,
            #                 connection.output_port,
            #                 connection_point.input_port,
            #             )
            #             in self._translator.E_conn_to_sp_group
            #         ):
            #             update_input_port_index = True
            #             sp, groups = self._translator.E_conn_to_sp_group[
            #                 (
            #                     component,
            #                     connected_component,
            #                     connection.output_port,
            #                     connection_point.input_port,
            #                 )
            #             ]
            #             # Find the group of the connected component
            #             modeled_match_nodes_ = self._translator.sim2sem_map[
            #                 connected_component
            #             ]
            #             groups_matched = [
            #                 g
            #                 for g in groups
            #                 if len(modeled_match_nodes_.intersection(set(g.values())))
            #                 > 0
            #             ]
            #             assert (
            #                 len(groups_matched) == 1
            #             ), "Only one group is allowed for each component."
            #             group = groups_matched[0]
            #             group_hash = hash(group)

            #             # component.input[connection_point.input_port].update(
            #             #     group_id=group_id
            #             # )

            #             ###########################
            #             hash_array[j] = group_hash
            #             # for idx, group_id in self.id_map.items():
            #             #     id_array[idx] = group_id
            #             # self.sorted_id_indices = torch.argsort(id_array)
            #             #########################################

            #     if update_input_port_index:
            #         for index, connection in zip(
            #             hash_array, connection_point.connects_system_through
            #         ):
            #             connection_point.set_input_port_index(connection, index)

            component.initialize(
                start_time=start_time,
                end_time=end_time,
                step_size=step_size,
            )

        # Check for missing initial values AFTER all components are initialized
        self.check_for_for_missing_initial_values()

    def validate(self) -> None:
        """
        Validate the model by checking IDs and connections.
        """
        LOGGER.add_level()

        LOGGER.task("Validating components")
        LOGGER.add_level()
        (
            validated_for_simulator_components,
            validated_for_estimator_components,
            validated_for_optimizer_components,
        ) = self.validate_components()
        if (
            validated_for_simulator_components
            and validated_for_estimator_components
            and validated_for_optimizer_components
        ) == False:
            LOGGER.error("Validating components", change_status=True)
        else:
            LOGGER.ok("Validating components", change_status=True)
        LOGGER.remove_level()

        LOGGER.task("Validating connections")
        LOGGER.add_level()
        (
            validated_for_simulator_connections,
            validated_for_estimator_connections,
            validated_for_optimizer_connections,
        ) = self.validate_connections()
        if (
            validated_for_simulator_connections
            and validated_for_estimator_connections
            and validated_for_optimizer_connections
        ) == False:
            LOGGER.error("Validating connections", change_status=True)
        else:
            LOGGER.ok("Validating connections", change_status=True)
        LOGGER.remove_level()

        self._validated_for_simulator = (
            validated_for_simulator_components and validated_for_simulator_connections
        )
        self._validated_for_estimator = (
            validated_for_estimator_components and validated_for_estimator_connections
        )
        self._validated_for_optimizer = (
            validated_for_optimizer_components and validated_for_optimizer_connections
        )
        self._is_validated = (
            self._validated_for_simulator
            and self._validated_for_estimator
            and self._validated_for_optimizer
        )

        if self._validated_for_simulator:
            LOGGER.ok("Validated for simulator")
        else:
            LOGGER.error("Validated for simulator.")
        if self._validated_for_estimator:
            LOGGER.ok("Validated for estimator")
        else:
            LOGGER.error("Validated for estimator.")
        if self._validated_for_optimizer:
            LOGGER.ok("Validated for optimizer")
        else:
            LOGGER.error("Validated for optimizer.")
        LOGGER.remove_level()

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
                ) = component.validate(LOGGER)
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
                    LOGGER.warning(
                        "Class: %s, id: %s: missing values for the following parameters to enable use of simulator and optimizer.",
                        component.__class__.__name__,
                        component.id,
                    )
                    LOGGER.add_level()
                    for par in is_none:
                        LOGGER.info("%s", par)
                    LOGGER.remove_level()

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
                                LOGGER.warning(
                                    'Class: %s, id: %s: the output "%s" is not a leaf scalar. Only leaf scalars can be used as output from components with no inputs.',
                                    component.__class__.__name__,
                                    component.id,
                                    key,
                                )
                                _validated_for_optimizer = False

                            # assert output.is_leaf, f"|CLASS: {component.__class__.__name__}|ID: {component.id}|: The output \"{key}\" is not a leaf scalar. Only leaf scalars can be used as output from components with no inputs."

                else:
                    for key in component.output.keys():
                        output = component.output[key]
                        if isinstance(
                            output, tps.Scalar
                        ):  # TODO: Add support for vectors
                            if output.is_leaf:
                                LOGGER.warning(
                                    'Class: %s, id: %s: the output "%s" is a leaf scalar. Only non-leaf scalars can be used as output from components with inputs.',
                                    component.__class__.__name__,
                                    component.id,
                                    key,
                                )
                                _validated_for_optimizer = False
                            # assert output.is_leaf==False, f"|CLASS: {component.__class__.__name__}|ID: {component.id}|: The output \"{key}\" is a leaf scalar. Only non-leaf scalars can be used as output from components with inputs."
        (
            __validated_for_simulator,
            __validated_for_estimator,
            __validated_for_optimizer,
        ) = self._validate_ids()

        _validated_for_simulator = (
            _validated_for_simulator and __validated_for_simulator
        )
        _validated_for_estimator = (
            _validated_for_estimator and __validated_for_estimator
        )
        _validated_for_optimizer = (
            _validated_for_optimizer and __validated_for_optimizer
        )

        return (
            _validated_for_simulator,
            _validated_for_estimator,
            _validated_for_optimizer,
        )

    def _validate_ids(self) -> None:
        """
        Validate the IDs of all components in the model.

        Raises:
            AssertionError: If any component has an invalid ID.
        """
        validated = True
        component_instances = list(self._components.values())
        for component in component_instances:
            # Length check — mirrors the one in ``SimulationModel.__init__``
            # / ``_check_id`` but downgraded to a logged validation
            # failure (matches the existing style here: invalid chars set
            # ``validated=False`` rather than raising, so callers can
            # collect all offenders in one pass).
            if len(component.id) > MAX_ID_LEN:
                LOGGER.error(
                    "Class: %s, id: %s: id length %d exceeds the maximum "
                    "of %d characters; this will trip Windows MAX_PATH "
                    "when the id is used as a filename component.",
                    component.__class__.__name__,
                    component.id,
                    len(component.id),
                    MAX_ID_LEN,
                )
                validated = False
            isvalid = np.array(
                [x.isalnum() or x in INVALID_ID_CHARS for x in component.id]
            )
            np_id = np.array(list(component.id))
            violated_characters = list(np_id[isvalid == False])
            if not all(isvalid):
                LOGGER.error(
                    "Class: %s, id: %s: invalid id, the characters \"%s\" are not allowed.",
                    component.__class__.__name__,
                    component.id,
                    ", ".join(violated_characters),
                )
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

            if hasattr(
                component, "validate_connections"
            ):  # Check if component has validate method
                validated = component.validate_connections(LOGGER)
            else:
                if (
                    len(component.connected_through) == 0
                    and len(component.connects_at) == 0
                ):
                    LOGGER.warning(
                        "Class: %s, id: %s: the component is not connected to any other components.",
                        component.__class__.__name__,
                        component.id,
                    )

                input_labels = [cp.input_port for cp in component.connects_at]
                first_input = True
                for req_input_label in component.input.keys():
                    if (
                        req_input_label not in input_labels
                        and component.input[req_input_label].optional == False
                    ):
                        if first_input:
                            LOGGER.warning(
                                "Class: %s, id: %s: missing connections for the following inputs to enable use of simulator, estimator, and optimizer.",
                                component.__class__.__name__,
                                component.id,
                            )
                            first_input = False
                            LOGGER.add_level()
                        LOGGER.info("%s", req_input_label)
                        validated = False
                if first_input == False:
                    LOGGER.remove_level()
        return (validated, validated, validated)

    def _load_parameters(self, force_config_overwrite: bool = False) -> None:
        """
        Load parameters for all components from configuration files.

        Args:
            force_config_overwrite (bool): If True, all parameters are read from the config file. If False, only the parameters that are None are read from the config file. If you want to use the fcn function
            to set the parameters, you should set force_config_overwrite to False to avoid it being overwritten.
        """

        LOGGER.add_level()

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
                    LOGGER.warning(
                        "Class: %s, id: %s: config structure mismatch.",
                        component.__class__.__name__,
                        component.id,
                    )
                    LOGGER.add_level()
                    if comparison_result["missing_in_1"]:
                        LOGGER.warning(
                            "Unused parameters in file config: %s.",
                            ", ".join(sorted(comparison_result["missing_in_1"])),
                        )
                    if comparison_result["missing_in_2"]:
                        LOGGER.warning(
                            "Missing parameters in file config: %s.",
                            ", ".join(sorted(comparison_result["missing_in_2"])),
                        )
                    LOGGER.remove_level()

                if force_config_overwrite:
                    config_ = merge_dicts(config_, config, prioritize="dict2")
                else:
                    config_ = merge_dicts(
                        config_, config, prioritize="dict1"
                    )  # Prioritize config_ over config to allow user to change stuff in the fcn function (programatically)

                self.set_parameters_from_config(config_, component)

                with open(filename, "w") as f:
                    json.dump(config_, f, indent=4)

        LOGGER.remove_level()

    def load(
        self,
        rdf_file: Optional[str] = None,
        fcn: Optional[Callable] = None,
        verbose: Union[int, None] = None,
        validate_model: bool = True,
        force_config_overwrite: bool = False,
        logfile: Optional[str] = None,
    ) -> None:
        """
        Load and set up the model for simulation.

        Args:
            rdf_file: Path to a serialized model.
            fcn: Custom function to be applied during model loading.
            verbose: Verbosity level controlling the amount of output. 0 to disable, 1-n to contol how many levels to print.
            validate_model: Whether to perform model validation.
            force_config_overwrite: If True, all parameters are read from the config file. If False, only the parameters that are None are read from the config file. If you want to use the fcn function
            to set the parameters, you should set force_config_overwrite to False to avoid it being overwritten.
            logfile: Path to the log file.
        """
        if verbose:
            self._load(
                rdf_file=rdf_file,
                fcn=fcn,
                validate_model=validate_model,
                force_config_overwrite=force_config_overwrite,
                verbose=verbose,
                logfile=logfile,
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
                    logfile=logfile,
                )

    # @reset_print
    def _load(
        self,
        rdf_file: Optional[str],
        fcn: Optional[Callable],
        verbose: int,
        validate_model: bool,
        force_config_overwrite: bool,
        logfile: Optional[str],
    ) -> None:
        """
        Internal method to load and set up the model for simulation.

        This method is called by load and performs the actual loading process.

        Args:
            rdf_file: Path to a serialized model.
            fcn: Custom function to be applied during model loading.
            verbose: Verbosity level controlling the amount of output. 0 to disable, 1-n to contol how many levels to print.
            validate_model: Whether to perform model validation.
            force_config_overwrite: If True, all parameters are read from the config file. If False, only the parameters that are None are read from the config file. If you want to use the fcn function
            to set the parameters, you should set force_config_overwrite to False to avoid it being overwritten.
            logfile: Path to the log file.
        """
        # if not LOGGER.is_active:
        #     reset_PRINTPROGRESS = True
        # else:
        #     reset_PRINTPROGRESS = False

        if verbose is not None:
            LOGGER.verbose = verbose
        LOGGER.logfile = logfile

        if self._is_loaded:
            self._reset()

        LOGGER.task("Loading simulation model")
        LOGGER.add_level()

        if rdf_file is not None:
            LOGGER.task("Loading model from RDF file")
            self._load_model_from_rdf(rdf_file)
            LOGGER.ok("Loading model from RDF file", change_status=True)

        if fcn is not None:
            assert callable(
                fcn
            ), "The function to be applied during model loading is not callable."
            LOGGER.task("Applying user-defined function")
            fcn(self)
            LOGGER.ok("Applying user-defined function", change_status=True)

        LOGGER.task("Preparing for topological sorting")
        self._get_components_no_cycles()
        LOGGER.ok("Preparing for topological sorting", change_status=True)

        LOGGER.task("Determining execution order")
        self._get_execution_order()
        LOGGER.ok("Determining execution order", change_status=True)

        LOGGER.task("Loading parameters")
        self._load_parameters(force_config_overwrite=force_config_overwrite)
        LOGGER.ok("Loading parameters", change_status=True)

        if validate_model:
            LOGGER.task("Validating model")
            self.validate()
            LOGGER.ok("Validating model", change_status=True)

        LOGGER.remove_level()
        LOGGER.ok("Loading simulation model", change_status=True)

        self._is_loaded = True

        # if reset_PRINTPROGRESS:
        #     LOGGER.reset()

        # if verbose:
        #     print(self)

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

    def _reset(self) -> None:
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

    def _get_simple_graph(self, components) -> Dict:
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

    def _get_simple_cycles(self, components: Dict) -> List[List[core.System]]:
        """
        Get the simple cycles in the system graph.

        Args:
            components (Dict): Dictionary of components.

        Returns:
            List[List[core.System]]: List of simple cycles.
        """
        G = self._get_simple_graph(components)
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
                self.add_component(new_component, _new_components)
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
                        connection.output_port,
                        connection_point.input_port,
                        output_port_index=connection_point.output_port_index[
                            connection
                        ],
                        input_port_index=connection_point.input_port_index[connection],
                        components=_new_components,
                    )

        # _new_components = {k: old_to_new_mapping[v] for k, v in self._components.items()}
        return _new_components

    def _get_components_no_cycles(self) -> None:
        """
        Create a dictionary of components without cycles using an improved algorithm
        that minimizes the number of edges removed.
        """
        LOGGER.add_level()
        LOGGER.task("Copying components")
        self._components_no_cycles = self._copy_components()
        LOGGER.ok("Copying components", change_status=True)
        self._required_initialization_connections = []

        # Use the improved cycle removal algorithm
        self._remove_cycles()
        LOGGER.remove_level()

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

        LOGGER.task("Detecting cycles")
        LOGGER.add_level()

        # Calculate all cycles once at the beginning
        cycles = list(self._get_simple_cycles(self._components_no_cycles))
        LOGGER.info("Found %d cycles", len(cycles))
        if not cycles:
            LOGGER.info("No cycles found")
            LOGGER.remove_level()
            LOGGER.ok("Detecting cycles", change_status=True)
            return  # No cycles to remove
        LOGGER.remove_level()
        LOGGER.ok("Detecting cycles", change_status=True)

        LOGGER.task("Removing cycles")
        LOGGER.add_level()
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

        reached_max_iterations = iteration >= max_iterations
        if reached_max_iterations:
            LOGGER.warning(
                "Cycle removal stopped after %d iterations.",
                max_iterations,
            )

        LOGGER.remove_level()
        if reached_max_iterations:
            LOGGER.warning("Removing cycles", change_status=True)
        else:
            LOGGER.ok("Removing cycles", change_status=True)

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
            LOGGER.info(
                "Multiple component pairs have the same cycle participation count (%d)",
                max_cycle_count,
            )
            LOGGER.add_level()
            for edge in best_edges:
                LOGGER.info("(%s, %s)", edge[0].id, edge[1].id)
            LOGGER.remove_level()

            # Prefer edges from components with more outgoing connections
            def edge_priority(edge):
                c_from, c_to = edge
                # Higher number of outgoing connections = higher priority for removal
                outgoing_count = len(c_from.connected_through)
                return outgoing_count

            best_edges.sort(key=edge_priority, reverse=True)

        LOGGER.info(
            "Selected component pair: (%s, %s)",
            best_edges[0][0].id,
            best_edges[0][1].id,
        )
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
        LOGGER.add_level()
        # Find and remove all connections from c_from to c_to
        connections_to_remove = []
        for connection in c_from.connected_through:
            for connection_point in connection.connects_system_at:
                if c_to == connection_point.connection_point_of:
                    connections_to_remove.append((connection, connection_point))
                    LOGGER.info(
                        "Removing connection: %s.%s --> %s.%s",
                        c_from.id,
                        connection.output_port,
                        c_to.id,
                        connection_point.input_port,
                    )

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
        LOGGER.remove_level()

    def load_estimation_result(
        self,
        filename: Optional[str] = None,
        result: Optional[Dict] = None,
        verbose: int = 0,
    ) -> None:
        """
        Load a chain log from a file or dictionary.

        Args:
            filename (Optional[str]): The filename to load the chain log from.
            result (Optional[Dict]): The chain log dictionary to load.
            verbose (int): If > 0, print applied parameter values for verification.

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
            else:
                raise Exception(f"The file {filename} is not a pickle file.")

        assert isinstance(
            self._result, estimator.EstimationResult
        ), f"The estimation result must be of type estimator.EstimationResult. The provided estimation result is of type {type(self._result)}."
        result_x = self._result["result_x"]

        # Build extended lookup including nested sub-objects (e.g.
        # OccupancySystem._DamperParams) that have their own id but are
        # not registered as top-level components.  nn.Module stores
        # child modules in _modules rather than __dict__, so we use
        # .modules() to walk the full hierarchy.
        component_lookup = dict(self._components)
        for comp in self._components.values():
            if isinstance(comp, torch.nn.Module):
                for child in comp.modules():
                    if (
                        child is not comp
                        and hasattr(child, "id")
                        and child.id not in component_lookup
                    ):
                        component_lookup[child.id] = child
            else:
                for attr_val in vars(comp).values():
                    if hasattr(attr_val, "id") and attr_val.id not in component_lookup:
                        component_lookup[attr_val.id] = attr_val

        flat_components = [
            component_lookup[com_id] for com_id in self._result["component_id"]
        ]
        flat_attr_list = self._result["component_attr"]
        theta_mask = self._result["theta_mask"]
        theta_slices = self._result["theta_slices"]
        lb = self._result["lb"]
        ub = self._result["ub"]

        # Use theta_slices to properly map from the flat unique-parameter
        # arrays (result_x, lb, ub) back to per-component values.
        # This correctly handles parameters with n_c > 1 and shared parameters.
        values = []
        min_values = []
        max_values = []
        for param_idx in theta_mask:
            start, end = theta_slices[param_idx]
            values.append(result_x[start:end])
            min_values.append(lb[start:end])
            max_values.append(ub[start:end])

        self.set_parameters(
            values,
            flat_components,
            flat_attr_list,
            min_values=min_values,
            max_values=max_values,
        )

        if verbose > 0:
            theta_mask = self._result["theta_mask"]
            theta_slices = self._result["theta_slices"]
            LOGGER.info("Load estimation result: applied parameters")
            for comp, attr, param_idx in zip(
                flat_components, flat_attr_list, theta_mask
            ):
                start, end = theta_slices[param_idx]
                raw = result_x[start:end]
                obj = rgetattr(comp, attr)
                actual = obj.get() if hasattr(obj, "get") else obj
                LOGGER.info(
                    "%s.%s: pickle: %s, actual: %s",
                    comp.id,
                    attr,
                    raw,
                    actual,
                )

    def check_for_for_missing_initial_values(self) -> None:
        """
        Check for missing initial values in components.

        Raises:
            Exception: If any component is missing an initial value.
        """
        for connection in self._required_initialization_connections:
            component = connection.connects_system
            if connection.output_port not in component.output:
                raise Exception(
                    f'The component with id: "{component.id}" and class: "{component.__class__.__name__}" is missing an initial value for the output: {connection.output_port}'
                )
            elif component.output[connection.output_port].get() is None:
                raise Exception(
                    f'The component with id: "{component.id}" and class: "{component.__class__.__name__}" is missing an initial value for the output: {connection.output_port}'
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

        LOGGER.add_level()
        LOGGER.task("Running Kahn's algorithm")
        LOGGER.add_level()

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
            if connection.output_port == no_cycle_connection.output_port
        ]

        self._flat_execution_order = _flatten(self._execution_order)
        assert len(self._flat_execution_order) == len(
            self._components_no_cycles
        ), "Cycles detected in the model. This should not happen. Please report this issue."

        for i, component_group in enumerate(self._execution_order):
            LOGGER.section("Priority: %d", i)
            LOGGER.add_level()
            for component in component_group:
                LOGGER.info("%s", component.id)
            LOGGER.remove_level()
        LOGGER.remove_level()
        LOGGER.ok("Running Kahn's algorithm", change_status=True)

        LOGGER.remove_level()

    def _update_literals(
        self,
        components: List[core.System] = None,
        connections: List[core.Connection] = None,
        connection_points: List[core.ConnectionPoint] = None,
    ) -> None:
        """
        Update the literals in the semantic model.
        """

        def _update_literals_for_component(
            component: core.System,
            connection: core.Connection = None,
            connection_point: core.ConnectionPoint = None,
        ) -> None:
            component_uri = self._semantic_model.T4B.__getitem__(component.id)
            for key, value in flatten_dict(component.populate_config(), component):
                if isinstance(value, (dict, list)):
                    # Serialize dicts and lists as JSON with datatype
                    value_ = json.dumps(value)
                    datatype = core.namespace.RDF.JSON
                else:
                    value_ = value
                    datatype = None

                # Check if the property is already in the semantic model
                literal_property = list(
                    self._semantic_model.instance_graph.objects(
                        component_uri, core.namespace.T4B.__getitem__(key)
                    )
                )
                if len(literal_property) == 0:
                    # No literal in the semantic model.
                    # Add the literal to the semantic model.
                    literal_property = Literal(value_, datatype=datatype)
                    self._semantic_model.instance_graph.add(
                        (
                            component_uri,
                            core.namespace.T4B.__getitem__(key),
                            literal_property,
                        )
                    )
                elif len(literal_property) == 1:
                    # There is one literal in the semantic model.
                    literal_property = literal_property[0]
                    # Remove the literal from the semantic model.
                    self._semantic_model.instance_graph.remove(
                        (
                            component_uri,
                            core.namespace.T4B.__getitem__(key),
                            literal_property,
                        )
                    )
                    # Add the new literal to the semantic model.
                    literal_property = Literal(value_, datatype=datatype)
                    self._semantic_model.instance_graph.add(
                        (
                            component_uri,
                            core.namespace.T4B.__getitem__(key),
                            literal_property,
                        )
                    )
                else:
                    # There are more than one literal in the semantic model.
                    raise Exception(
                        f'The component with id: "{component.id}" has more than one output port.'
                    )

        def _update_literals_for_connection(connection: core.Connection) -> None:
            """
            Update the literals for a connection in the semantic model.
            Updates output_port.
            """
            connection_uri = self._semantic_model.T4B.__getitem__(str(hash(connection)))

            # Define the literals to update
            literals_to_update = {
                "output_port": connection.output_port,
            }

            for key, value in literals_to_update.items():
                if isinstance(value, (dict, list)):
                    # Serialize dicts and lists as JSON with datatype
                    value_ = json.dumps(value)
                    datatype = core.namespace.RDF.JSON
                else:
                    value_ = value
                    datatype = None

                # Check if the property is already in the semantic model
                literal_property = list(
                    self._semantic_model.instance_graph.objects(
                        connection_uri, core.namespace.T4B.__getitem__(key)
                    )
                )
                if len(literal_property) == 0:
                    # No literal in the semantic model.
                    # Add the literal to the semantic model.
                    literal_property = Literal(value_, datatype=datatype)
                    self._semantic_model.instance_graph.add(
                        (
                            connection_uri,
                            core.namespace.T4B.__getitem__(key),
                            literal_property,
                        )
                    )
                elif len(literal_property) == 1:
                    # There is one literal in the semantic model.
                    literal_property = literal_property[0]
                    # Remove the literal from the semantic model.
                    self._semantic_model.instance_graph.remove(
                        (
                            connection_uri,
                            core.namespace.T4B.__getitem__(key),
                            literal_property,
                        )
                    )
                    # Add the new literal to the semantic model.
                    literal_property = Literal(value_, datatype=datatype)
                    self._semantic_model.instance_graph.add(
                        (
                            connection_uri,
                            core.namespace.T4B.__getitem__(key),
                            literal_property,
                        )
                    )
                else:
                    # There are more than one literal in the semantic model.
                    raise Exception(
                        f'The connection has more than one literal for "{key}".'
                    )

        def _update_literals_for_connection_point(
            connection_point: core.ConnectionPoint,
        ) -> None:
            """
            Update the literals for a connection point in the semantic model.
            Updates input_port, input_port_index, and output_port_index.
            """
            connection_point_uri = self._semantic_model.T4B.__getitem__(
                str(hash(connection_point))
            )

            # Define the literals to update
            literals_to_update = {
                "input_port": connection_point.input_port,
                "input_port_index": {
                    str(hash(conn)): (
                        int(idx) if isinstance(idx, (int, torch.Tensor)) else idx
                    )
                    for conn, idx in connection_point.input_port_index.items()
                },
                "output_port_index": {
                    str(hash(conn)): (
                        int(idx) if isinstance(idx, (int, torch.Tensor)) else idx
                    )
                    for conn, idx in connection_point.output_port_index.items()
                },
            }

            for key, value in literals_to_update.items():
                if isinstance(value, (dict, list)):
                    # Serialize dicts and lists as JSON with datatype
                    value_ = json.dumps(value)
                    datatype = core.namespace.RDF.JSON
                else:
                    value_ = value
                    datatype = None

                # Check if the property is already in the semantic model
                literal_property = list(
                    self._semantic_model.instance_graph.objects(
                        connection_point_uri, core.namespace.T4B.__getitem__(key)
                    )
                )
                if len(literal_property) == 0:
                    # No literal in the semantic model.
                    # Add the literal to the semantic model.
                    literal_property = Literal(value_, datatype=datatype)
                    self._semantic_model.instance_graph.add(
                        (
                            connection_point_uri,
                            core.namespace.T4B.__getitem__(key),
                            literal_property,
                        )
                    )
                elif len(literal_property) == 1:
                    # There is one literal in the semantic model.
                    literal_property = literal_property[0]
                    # Remove the literal from the semantic model.
                    self._semantic_model.instance_graph.remove(
                        (
                            connection_point_uri,
                            core.namespace.T4B.__getitem__(key),
                            literal_property,
                        )
                    )
                    # Add the new literal to the semantic model.
                    literal_property = Literal(value_, datatype=datatype)
                    self._semantic_model.instance_graph.add(
                        (
                            connection_point_uri,
                            core.namespace.T4B.__getitem__(key),
                            literal_property,
                        )
                    )
                else:
                    # There are more than one literal in the semantic model.
                    raise Exception(
                        f'The connection point has more than one literal for "{key}".'
                    )

        if components is None and connections is None and connection_points is None:
            for component in self._components.values():
                _update_literals_for_component(component)
                # Also update literals for all connections of this component
                for connection in component.connected_through:
                    _update_literals_for_connection(connection)
                # Also update literals for all connection points of this component
                for connection_point in component.connects_at:
                    _update_literals_for_connection_point(connection_point)

        if components is not None:
            for component in components:
                _update_literals_for_component(component)

        if connections is not None:
            for connection in connections:
                _update_literals_for_connection(connection)
        if connection_points is not None:
            for connection_point in connection_points:
                _update_literals_for_connection_point(connection_point)

    def serialize(self):
        """
        Serialize the simulation model.
        """
        # dummy_start_time = [datetime.datetime.now()] * len(self._components)
        # dummy_end_time = [datetime.datetime.now()] * len(self._components)
        # dummy_step_size = [1]
        # self.load(verbose=False)
        # self.initialize(dummy_start_time, dummy_end_time, dummy_step_size)
        self._update_literals()
        self._semantic_model.serialize()

    def visualize(
        self,
        query: str = None,
        literals: bool = True,
        forward_only: bool = False,
        compressed: bool = False,
        **kwargs,
    ) -> None:
        """
        Visualize the simulation model.

        Args:
            query: Custom SPARQL CONSTRUCT query. If None, a default query is used.
            literals: If True, include all literals. If False, only include connection-related properties.
            forward_only: If True, only include forward flow (System -> Connection -> ConnectionPoint -> System).
                         If False, include both forward and reverse relationships.
            compressed: If True, remove intermediate Connection and ConnectionPoint nodes
                       and show direct edges between system components with port labels
                       like ``"output: YYY\\ninput: XXX"``.
            **kwargs: Additional arguments passed to semantic_model.visualize().
        """
        self._update_literals()
        if compressed:
            forward_only = True
        if query is None:
            if forward_only and literals:
                # Forward flow + all literals
                # Forward: connectedThrough, connectsSystemAt, connectionPointOf
                # All literals except rdf:type and rdfs:subClassOf
                # Exclude reverse relationships: connectsSystem, connectsSystemThrough, connectsAt
                query = """
                CONSTRUCT {
                    ?s ?p ?o
                }
                WHERE {
                    ?s ?p ?o .
                    FILTER (?p != rdf:type && 
                            ?p != rdfs:subClassOf &&
                            ?p != s4syst:connectsSystem &&
                            ?p != s4syst:connectsSystemThrough &&
                            ?p != s4syst:connectsAt)
                }
                """
            elif forward_only and not literals:
                # Forward flow + only port literals
                query = """
                CONSTRUCT {
                    ?s ?p ?o
                }
                WHERE {
                    ?s ?p ?o .
                    FILTER (?p = s4syst:connectedThrough || 
                            ?p = s4syst:connectsSystemAt || 
                            ?p = s4syst:connectionPointOf ||
                            ?p = t4b:input_port ||
                            ?p = t4b:output_port ||
                            ?p = t4b:input_port_index ||
                            ?p = t4b:output_port_index)
                }
                """
            elif not forward_only and literals:
                # All relationships + all literals
                query = None
            else:
                # All relationships + only port literals
                query = """
                CONSTRUCT {
                    ?s ?p ?o
                }
                WHERE {
                    ?s ?p ?o .
                    FILTER (?p = s4syst:connectedThrough || 
                            ?p = s4syst:connectsSystemAt || 
                            ?p = s4syst:connectionPointOf ||
                            ?p = s4syst:connectsSystem ||
                            ?p = s4syst:connectsSystemThrough ||
                            ?p = s4syst:connectsAt ||
                            ?p = t4b:input_port ||
                            ?p = t4b:output_port ||
                            ?p = t4b:input_port_index ||
                            ?p = t4b:output_port_index)
                }
                """
        if compressed:
            kwargs["pydot_transform"] = self._build_compressed_transform()
        self._semantic_model.visualize(query, **kwargs)

    @staticmethod
    def _build_compressed_transform():
        """Build a pydot_transform callback that collapses Connection/ConnectionPoint
        nodes into direct edges labelled with port names."""

        def _compress(dg):
            # Third party imports
            import pydotplus as pdp
            from bs4 import BeautifulSoup

            def _unquote(name):
                """Strip surrounding double-quotes that pydotplus may add."""
                s = name.strip()
                if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
                    return s[1:-1]
                return s

            def _find_prop(props, prop_name):
                """Find a property value whose key ends with exactly *prop_name*."""
                for key, val in props.items():
                    k = key.strip()
                    if (
                        k == prop_name
                        or k.endswith(":" + prop_name)
                        or k.endswith("/" + prop_name)
                    ):
                        return val.strip('"')
                return "?"

            # --- 1. Parse node labels ----------------------------------------
            node_info = {}  # norm_name -> {type, props, orig_name}
            conn_nodes = set()
            cp_nodes = set()
            all_headers = set()

            for node in dg.get_nodes():
                orig_name = node.get_name()
                name = _unquote(orig_name)
                attrs = node.obj_dict.get("attributes", {})
                if "label" not in attrs:
                    continue
                soup = BeautifulSoup(attrs["label"], "html.parser")
                rows = soup.find_all("tr")
                if not rows:
                    continue

                header = rows[0].get_text().strip()
                all_headers.add(header)
                props = {}
                for row in rows:
                    cols = row.find_all("td")
                    if len(cols) == 2:
                        props[cols[0].get_text().strip()] = cols[1].get_text().strip()

                node_info[name] = {"type": header, "props": props, "orig": orig_name}
                if header == "Connection":
                    conn_nodes.add(name)
                elif header == "ConnectionPoint":
                    cp_nodes.add(name)

            intermediate = conn_nodes | cp_nodes
            if not intermediate:
                warnings.warn(
                    f"compressed=True: found 0 Connection/ConnectionPoint nodes. "
                    f"Node type headers present: {all_headers}",
                    stacklevel=4,
                )
                return

            # --- 2. Build adjacency using normalised names --------------------
            outgoing = {}  # norm_src -> [norm_dst, ...]
            incoming = {}  # norm_dst -> [norm_src, ...]
            norm_to_orig = {}

            for orig_src, orig_dst in dg.obj_dict["edges"]:
                ns, nd = _unquote(orig_src), _unquote(orig_dst)
                outgoing.setdefault(ns, []).append(nd)
                incoming.setdefault(nd, []).append(ns)
                norm_to_orig.setdefault(ns, orig_src)
                norm_to_orig.setdefault(nd, orig_dst)

            for nname, info in node_info.items():
                norm_to_orig.setdefault(nname, info["orig"])

            # --- 3. Trace chains and collect new direct edges -----------------
            new_edges = []
            for conn in conn_nodes:
                output_port = _find_prop(node_info[conn]["props"], "output_port")
                senders = [s for s in incoming.get(conn, []) if s not in intermediate]
                cps = [d for d in outgoing.get(conn, []) if d in cp_nodes]

                for cp in cps:
                    input_port = _find_prop(node_info[cp]["props"], "input_port")
                    receivers = [
                        d for d in outgoing.get(cp, []) if d not in intermediate
                    ]

                    for sender in senders:
                        for receiver in receivers:
                            label = f"output: {output_port}\\ninput: {input_port}"
                            new_edges.append((sender, receiver, label))

            # Deduplicate in case the RDF graph contained redundant paths
            seen = set()
            unique_edges = []
            for entry in new_edges:
                if entry not in seen:
                    seen.add(entry)
                    unique_edges.append(entry)
            new_edges = unique_edges

            if not new_edges:
                warnings.warn(
                    f"compressed=True: found {len(conn_nodes)} Connection and "
                    f"{len(cp_nodes)} ConnectionPoint nodes but could not trace "
                    f"any complete chains. Check edge directions.",
                    stacklevel=4,
                )

            # --- 4. Remove intermediate edges and nodes -----------------------
            dg.obj_dict["edges"] = {
                (src, dst): edge_list
                for (src, dst), edge_list in dg.obj_dict["edges"].items()
                if _unquote(src) not in intermediate
                and _unquote(dst) not in intermediate
            }

            for name in intermediate:
                orig = node_info[name]["orig"]
                if orig in dg.obj_dict["nodes"]:
                    del dg.obj_dict["nodes"][orig]

            # --- 5. Add new direct edges --------------------------------------
            for sender, receiver, label in new_edges:
                orig_src = norm_to_orig.get(sender, sender)
                orig_dst = norm_to_orig.get(receiver, receiver)
                edge = pdp.Edge(orig_src, orig_dst)
                edge.obj_dict["attributes"]["label"] = f'"{label}"'
                edge.obj_dict["attributes"]["fontsize"] = "7"
                edge.obj_dict["attributes"]["fontname"] = "Courier"
                dg.add_edge(edge)

        return _compress

    def _load_model_from_rdf(self, rdf_file: str) -> None:
        """
        Load a complete model (components and connections) from an RDF file.
        This method reads the RDF file and reconstructs both components and their connections.

        Args:
            rdf_file (str): Path to the RDF file to load from
        """
        LOGGER.add_level()
        self._semantic_model = core.SemanticModel(
            id=self._id,
            rdf_file=rdf_file,
            namespaces={"T4B": core.namespace.T4B, "S4SYST": core.namespace.S4SYST},
            dir_conf=self._dir_conf + ["semantic_model"],
        )

        LOGGER.task("Instantiating components")
        LOGGER.add_level()

        # print(f"sm instances: {self._semantic_model.get_instances_of_type(core.namespace.S4SYST.System)}")

        # print("all triples:")
        # for triple in self._semantic_model.instance_graph:
        # print(triple)

        # Instantiate components with their attributes
        for sm_instance in self._semantic_model.get_instances_of_type(
            core.namespace.S4SYST.System
        ):
            t = sm_instance.get_most_specific_type()
            class_name = t.get_short_name()
            cls = getattr(systems, class_name)
            attributes = {}
            for pred, obj in sm_instance.get_predicate_object_pairs().items():
                for obj_ in obj:
                    if isinstance(obj_, core.SemanticLiteral):
                        literal_value = obj_.uri.value
                        # Convert string literals to appropriate Python types
                        literal_value = _convert_literal_value(literal_value)
                        attributes[
                            get_short_name(pred, self._semantic_model.namespaces)
                        ] = literal_value

            LOGGER.info(
                "Component: %s, type: %s",
                sm_instance.get_short_name(),
                class_name,
            )
            component = cls(id=sm_instance.get_short_name(), **attributes)
            # Check if the component already exists
            self.add_component(component)
        LOGGER.remove_level()
        LOGGER.ok("Instantiating components", change_status=True)

        LOGGER.task("Making connections")
        LOGGER.add_level()

        # Step 1: Read all connection info from the RDF graph while the
        # original Connection / ConnectionPoint triples are still present.
        pending_connections = []
        for sm_instance in self._semantic_model.get_instances_of_type(
            core.namespace.S4SYST.System
        ):
            component = self._components[sm_instance.get_short_name()]
            predicate_object_pairs = sm_instance.get_predicate_object_pairs()
            if core.namespace.S4SYST.connectedThrough not in predicate_object_pairs:
                continue

            connections = predicate_object_pairs[core.namespace.S4SYST.connectedThrough]

            for connection in connections:
                predicate_object_pairs_connection = (
                    connection.get_predicate_object_pairs()
                )
                output_port = predicate_object_pairs_connection[
                    core.namespace.T4B.output_port
                ][0].uri.value
                connection_points = predicate_object_pairs_connection[
                    core.namespace.S4SYST.connectsSystemAt
                ]

                for connection_point in connection_points:
                    predicate_object_pairs_connection_point = (
                        connection_point.get_predicate_object_pairs()
                    )
                    receiver_component = predicate_object_pairs_connection_point[
                        core.namespace.S4SYST.connectionPointOf
                    ][0]
                    input_port = predicate_object_pairs_connection_point[
                        core.namespace.T4B.input_port
                    ][0].uri.value

                    conn_key = connection.get_short_name()
                    input_port_index = None
                    output_port_index = None
                    if (
                        core.namespace.T4B.input_port_index
                        in predicate_object_pairs_connection_point
                    ):
                        raw = predicate_object_pairs_connection_point[
                            core.namespace.T4B.input_port_index
                        ][0].uri.value
                        parsed = _convert_literal_value(raw)
                        if isinstance(parsed, dict) and parsed:
                            input_port_index = parsed.get(conn_key)
                    if (
                        core.namespace.T4B.output_port_index
                        in predicate_object_pairs_connection_point
                    ):
                        raw = predicate_object_pairs_connection_point[
                            core.namespace.T4B.output_port_index
                        ][0].uri.value
                        parsed = _convert_literal_value(raw)
                        if isinstance(parsed, dict) and parsed:
                            output_port_index = parsed.get(conn_key)

                    receiver_component_id = receiver_component.get_short_name()
                    receiver_component = self._components[receiver_component_id]

                    pending_connections.append(
                        {
                            "sender": component,
                            "receiver": receiver_component,
                            "output_port": output_port,
                            "input_port": input_port,
                            "input_port_index": input_port_index,
                            "output_port_index": output_port_index,
                        }
                    )

        # Step 2: Remove old Connection / ConnectionPoint instances and
        # their triples.  add_connection() will recreate them with fresh
        # Python-object hashes.  Without this cleanup the graph would
        # contain BOTH the original URIs from the file AND new URIs from
        # add_connection, duplicating every edge.
        ig = self._semantic_model.instance_graph
        for conn_type in (
            core.namespace.S4SYST.Connection,
            core.namespace.S4SYST.ConnectionPoint,
        ):
            for subj in list(ig.subjects(RDF.type, conn_type)):
                ig.remove((subj, None, None))
                ig.remove((None, None, subj))

        # Step 3: Rebuild connections (adds Python objects + fresh RDF triples)
        for data in pending_connections:
            LOGGER.info(
                "Adding connection: %s.%s --> %s.%s",
                data["sender"].id,
                data["output_port"],
                data["receiver"].id,
                data["input_port"],
            )
            self.add_connection(
                sender_component=data["sender"],
                receiver_component=data["receiver"],
                output_port=data["output_port"],
                input_port=data["input_port"],
                input_port_index=data["input_port_index"],
                output_port_index=data["output_port_index"],
            )

        LOGGER.remove_level()
        LOGGER.ok("Making connections", change_status=True)

        LOGGER.remove_level()
