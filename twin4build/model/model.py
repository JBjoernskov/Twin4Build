# Standard library imports
import datetime
import shutil
import warnings
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

# Third party imports
import numpy as np
import pandas as pd
import torch
from prettytable import PrettyTable

# Local application imports
import twin4build.core as core
import twin4build.utils.types as tps
from twin4build.utils.mkdir_in_root import mkdir_in_root
from twin4build.utils.logger import LOGGER, autoreset_print


@autoreset_print
class Model:
    r"""
    A unified interface for building digital twin models.

    Args:
        id: Unique identifier for the model.

    This class serves as a composed interface that integrates both simulation and semantic
    modeling capabilities for building digital twins. It combines the functionality of
    :class:`SimulationModel` and :class:`SemanticModel` into a single, user-friendly interface.

    Composition Architecture
    ------------------------

    The Model class acts as a facade that orchestrates two core components:

    1. **SimulationModel:** Handles the computational aspects including cycle removal
       (Algorithm 1), topological sorting (Algorithm 2), component management, and
       preparation for simulation execution.

    2. **SemanticModel:** Manages the ontological representation using SAREF4SYST,
       RDF graphs, semantic queries, and metadata for interoperability.

    Key Responsibilities
    --------------------

    - **Unified Interface:** Provides a single entry point for both simulation and semantic operations
    - **Component Management:** Delegates component operations to the appropriate underlying model
    - **Model Lifecycle:** Orchestrates loading, validation, and execution preparation
    - **Data Integration:** Coordinates between semantic metadata and simulation execution
    - **Interoperability:** Ensures consistent SAREF-compliant representation across both models

    Usage Pattern
    -------------

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
    >>> space = tb.BuildingSpaceSystem(id="office_space")
    >>> heater = tb.SpaceHeaterSystem(id="radiator")
    >>> model.add_component(space)
    >>> model.add_component(heater)
    >>>
    >>> # Add connections (updates both simulation and semantic models)
    >>> model.add_connection(space, heater, "indoorTemperature", "indoorTemperature")
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
    >>> if model.simulation_model.is_loaded:
    ...     simulator = tb.Simulator(model)
    ...     # Run simulation...

    Loading from RDF file:

    >>> # Load existing semantic model and convert to simulation model
    >>> model = tb.Model(id="restored_model")
    >>> model.load(semantic_model_filename="my_building.ttl")
    >>> # Model now contains both semantic and simulation representations
    """

    __slots__ = (
        "_id",
        "_simulation_model",
        "_semantic_model",
        "_translator",
        "_dir_conf",
        "_component_to_meta",
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
            id: Unique identifier for the model.

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
                "T4B": core.namespace.T4B,
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
        self._translator = None
        self._component_to_meta: Dict[str, Tuple[Any, int]] = {}

    @classmethod
    def from_translation(
        cls,
        *,
        id: str,
        semantic_model: "core.SemanticModel",
        simulation_model: "core.SimulationModel",
        translator: "core.Translator",
    ) -> "Model":
        """Wrap the three artefacts produced by :meth:`Translator.translate`
        into a :class:`Model` without re-creating empty halves.

        Used by the translator to return a fully-wired Model. End users do
        not normally call this; build a Model from scratch with
        ``Model(id=...)`` or via ``Translator().translate(semantic_model)``.

        Args:
            id: The Model id (drives the on-disk directory layout under
                ``generated_files/models/<id>/``).
            semantic_model: The semantic model the translator consumed.
            simulation_model: The simulation graph the translator produced.
            translator: The translator instance whose ``sim2sem_map`` /
                ``sem2sim_map`` link the two models.
        """
        m = cls.__new__(cls)
        m._id = id
        m._dir_conf = ["generated_files", "models", id]
        m._semantic_model = semantic_model
        m._simulation_model = simulation_model
        m._translator = translator
        # Re-anchor on-disk directories so semantic_model and simulation_model
        # share the same Model-level parent.  Each underlying model exposes a
        # ``dir_conf`` setter that keeps internal sub-paths in sync.
        try:
            m._semantic_model.dir_conf = m._dir_conf + ["semantic_model"]
        except Exception:
            pass
        try:
            m._simulation_model.dir_conf = m._dir_conf + ["simulation_model"]
        except Exception:
            pass
        return m


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

    def get_component(self, component_id: str):
        """Component by id -- regular components and the fused state-space
        blocks that execute in place of clusters (see
        :meth:`SimulationModel.get_component`)."""
        return self.simulation_model.get_component(component_id)

    @property
    def device(self):
        """Device on which simulation tensors live (default cpu)."""
        return self.simulation_model.device

    @property
    def dtype(self):
        """Floating-point dtype used for simulation tensors."""
        return self.simulation_model.dtype

    def to(self, device=None, dtype=None) -> "Model":
        """Move the model to a device and/or floating-point dtype.

        Torch-style API: ``model.to("cuda")`` moves all component tensors to
        the GPU and makes the simulator, estimator and optimizer compute
        there; ``model.to("cuda", torch.float32)`` additionally switches to
        single precision (a large speedup on consumer GPUs, where fp64
        throughput is typically 1/32 of fp32, at reduced accuracy).
        Scipy/IPOPT solvers and plotting remain on the CPU -- only small
        parameter vectors cross that boundary per iteration.

        Performance expectation: for a single simulation or estimation of a
        small model (a handful of zones), the GPU does NOT beat the CPU --
        each step is a microsecond-scale operation and kernel-launch latency
        dominates.  Device support is the enabler for *batched* workflows
        (multi-start estimation, scenario/ensemble studies, portfolios of
        buildings via the ``n_c`` component-batch dimension) and for large
        many-zone models, where per-candidate cost falls almost linearly
        with batch size.  See
        ``twin4build/examples/gpu_benchmark_estimation.ipynb`` and
        ``twin4build/examples/gpu_benchmark_optimizer.ipynb``.

        Example:
            >>> model.load()
            >>> model.to("cuda")                      # fp64 on the GPU
            >>> model.to("cuda", torch.float32)       # fp32 (consumer GPUs)
            >>> model.to("cpu", torch.float64)        # back to the default

        Args:
            device: Target device (``"cpu"``, ``"cuda"``, ``torch.device``).
            dtype: Optional floating-point dtype.  Note this is a
                process-wide setting (it configures the framework default
                used for new tensor allocations), so all models in the
                process should use the same dtype.

        Returns:
            The model itself, for chaining.
        """
        self.simulation_model.to(device=device, dtype=dtype)
        return self

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
        output_port: str,
        input_port: str,
        output_port_index: [int, torch.Tensor] = None,
        input_port_index: [int, torch.Tensor] = None,
    ) -> None:
        """
        Add a connection between two components in the system.

        Args:
            sender_component (core.System): The component sending the connection.
            receiver_component (core.System): The component receiving the connection.
            output_port (str): Name of the sender property.
            input_port (str): Name of the receiver property.
            output_port_index (Optional[Union[int, torch.Tensor]]): Index into the sender's
                output port when it is a vector port. Defaults to None (scalar ports).
            input_port_index (Optional[Union[int, torch.Tensor]]): Index into the receiver's
                input port when it is a vector port. Defaults to None (scalar ports).

        Raises:
            AssertionError: If property names are invalid for the components.
            AssertionError: If a connection already exists.
        """
        self.simulation_model.add_connection(
            sender_component=sender_component,
            receiver_component=receiver_component,
            output_port=output_port,
            input_port=input_port,
            output_port_index=output_port_index,
            input_port_index=input_port_index,
        )

    def remove_connection(
        self,
        sender_component: "core.System",
        receiver_component: "core.System",
        output_port: str,
        input_port: str,
    ) -> None:
        """
        Remove a connection between two components in the system.

        Args:
            sender_component (core.System): The component sending the connection.
            receiver_component (core.System): The component receiving the connection.
            output_port (str): Name of the sender property.
            input_port (str): Name of the receiver property.

        Raises:
            ValueError: If the specified connection does not exist.
        """
        self.simulation_model.remove_connection(
            sender_component=sender_component,
            receiver_component=receiver_component,
            output_port=output_port,
            input_port=input_port,
        )

    def get_components_by_class(
        self, class_: Type, filter: Optional[Callable] = None
    ) -> List:
        """Return components on this model that are instances of ``class_``."""
        return self.simulation_model.get_component_by_class(
            dict_=self.simulation_model.components, class_=class_, filter=filter
        )

    def get_component_by_class(
        self, dict_: Dict = None, class_: Type = None, filter: Optional[Callable] = None
    ) -> List:
        """Deprecated: use :meth:`get_components_by_class` (removed in 2.1)."""
        from twin4build.utils.deprecation import deprecate_name

        deprecate_name("get_component_by_class", "get_components_by_class")
        if class_ is None and dict_ is not None and not isinstance(dict_, dict):
            # Called as get_component_by_class(SomeSystem)
            class_ = dict_
            dict_ = self.simulation_model.components
        if dict_ is None:
            dict_ = self.simulation_model.components
        return self.simulation_model.get_component_by_class(
            dict_=dict_, class_=class_, filter=filter
        )

    # def set_custom_initial_dict(
    #     self, custom_initial_dict: Dict[str, Dict[str, Any]]
    # ) -> None:
    #     """
    #     Set custom initial values for components.

    #     Args:
    #         custom_initial_dict (Dict[str, Dict[str, Any]]): Dictionary of custom initial values.

    #     Raises:
    #         AssertionError: If unknown component IDs are provided.
    #     """
    #     self.simulation_model.set_custom_initial_dict(
    #         custom_initial_dict=custom_initial_dict
    #     )

    def set_initial_values(
        self,
        values: List[Any] = None,
        components: List["core.System"] = None,
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
            # Pass the old dict to simulation_model which will handle conversion
            self.simulation_model.set_initial_values(dict_=old_dict)
            return

        self.simulation_model.set_initial_values(
            values=values,
            components=components,
            output_names=output_names,
        )

    def set_parameters(
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
            components (List[core.System]): List of components to set parameters for.
            parameter_names (List[str]): List of attribute names corresponding to the parameters.
            normalized (List[bool]): List of booleans indicating if values are normalized.
            overwrite (bool): Whether to overwrite existing parameters.
            save_original (bool): Whether to save original parameters for later restoration.

        Raises:
            AssertionError: If a component doesn't have the specified attribute.
        """
        self.simulation_model.set_parameters(
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

    # ------------------------------------------------------------------
    # Dataset-configuration helpers (bridging semantic + simulation)
    # ------------------------------------------------------------------
    def set_dbconfigs(self, dbconfig: Dict[str, Any]) -> "Model":
        """Apply a database configuration to every applicable component.

        Facade for :meth:`SimulationModel.set_dbconfigs`. Returns ``self``
        for chaining.
        """
        self.simulation_model.set_dbconfigs(dbconfig)
        return self

    def fill_missing_inputs(self, defaults: Dict[Any, Any]) -> "Model":
        """Attach providers (constants or user-supplied systems) for
        input ports the ontology did not wire.

        Facade for :meth:`SimulationModel.fill_missing_inputs`; see that
        method's docstring for the full key/value shape grammar.  In
        short, ``defaults`` accepts both port-only entries (``str ->
        value``) and component-scoped entries (``(component | id, port)
        -> value``); values may be scalars (wrapped in a flat schedule),
        :class:`core.System` providers (single-output auto-detected),
        or ``(provider, output_port)`` tuples.  Returns ``self`` for
        chaining.
        """
        self.simulation_model.fill_missing_inputs(defaults)
        return self

    def rewire(
        self,
        *,
        start_time: Any,
        end_time: Any,
        step_size: int,
        mode: str = "train",
        **rewire_kwargs: Any,
    ) -> "Model":
        """Run the data-driven CITS rewire over the model's graph.

        Should be called **before** :meth:`load` -- see
        :meth:`SimulationModel.rewire` for the rationale.  Facade for
        :meth:`SimulationModel.rewire`; in particular the ``mode``
        kwarg selects gate-active (``"train"``) vs gate-bypassed
        (``"simulate"``) frozen-pin configuration.

        Returns:
            ``self`` for chaining.
        """
        self.simulation_model.rewire(
            start_time=start_time,
            end_time=end_time,
            step_size=step_size,
            mode=mode,
            **rewire_kwargs,
        )
        return self

    def set_transformations(
        self, mapping: Dict[Any, Callable[[Any], Any]]
    ) -> "Model":
        """Apply unit-conversion callables to components by semantic class.

        Keys are RDF class IRIs (``rdflib.URIRef``, strings, or
        :class:`SemanticType` instances; e.g. ``BRICK.Temperature_Sensor``).
        Each simulation component's semantic counterpart is looked up via
        ``self._translator.sim2sem_map``; its rdf:types (with
        ``rdfs:subClassOf`` transitive closure from the embedded
        :class:`SemanticModel`) are matched against the mapping.

        Most-specific class wins on ambiguity: when two rule keys both
        match a component's types, the rule whose key is a subclass of
        the other's is preferred.  Identical-specificity collisions log
        a warning and the first-declared rule is kept.

        Components implement ``set_transformation(self, fn)``; components
        without that method are silently skipped (so this is safe to call
        on heterogeneous models where only e.g. ``SensorSystem`` cares
        about transformations).

        This is a real method (not a facade) because it requires the
        semantic + sim2sem map that only ``Model`` carries; the
        underlying ``SimulationModel`` is intentionally kept ontology-
        free.

        Returns:
            ``self`` for chaining.

        Raises:
            RuntimeError: When the model was not produced by a Translator
                (no ``sim2sem_map`` available).  Build the model via
                ``Translator().translate(semantic_model)`` to use this
                helper.
        """
        if self._translator is None:
            raise RuntimeError(
                "Model.set_transformations requires a translator-built "
                "model (no sim2sem_map is available on this Model). "
                "Construct the Model via "
                "`Translator().translate(semantic_model)` to use this "
                "helper, or set per-component transformations directly "
                "via `component.set_transformation(fn)`."
            )
        sim2sem = self._translator.sim2sem_map
        semantic = self._semantic_model

        # Resolve every rule key to a SemanticType so we can compare in
        # the (rdfs:subClassOf-closed) class hierarchy.  Accept URIRef,
        # plain str, or pre-built SemanticType instances.
        rules: List[Tuple[Any, Callable[[Any], Any], Any]] = []
        for key, fn in mapping.items():
            if isinstance(key, core.SemanticType):
                stype = key
            else:
                stype = semantic.get_type(key)
            rules.append((stype, fn, key))

        def _is_strict_subclass(a, b) -> bool:
            """``True`` iff ``a`` is a strict subclass of ``b``."""
            if str(a.uri) == str(b.uri):
                return False
            return any(str(sc.uri) == str(b.uri) for sc in a.super_classes)

        for component in self._simulation_model.components.values():
            setter = getattr(component, "set_transformation", None)
            if not callable(setter):
                continue
            semantic_nodes = sim2sem.get(component)
            if not semantic_nodes:
                continue

            # Gather every rule that matches at least one of the
            # component's semantic counterparts.
            matched: List[Tuple[Any, Callable[[Any], Any], Any]] = []
            for stype, fn, original_key in rules:
                hit = False
                for node in semantic_nodes:
                    isinstance_check = getattr(node, "isinstance", None)
                    if callable(isinstance_check) and isinstance_check(
                        stype.uri
                    ):
                        hit = True
                        break
                if hit:
                    matched.append((stype, fn, original_key))

            if not matched:
                continue

            # Most-specific wins; first-declared on ties.
            winner = matched[0]
            for cand in matched[1:]:
                if _is_strict_subclass(cand[0], winner[0]):
                    winner = cand
                elif _is_strict_subclass(winner[0], cand[0]):
                    continue
                else:
                    warnings.warn(
                        f"set_transformations: rules for {cand[2]!r} and "
                        f"{winner[2]!r} both match component "
                        f"{component.id!r} with no subclass relationship; "
                        f"keeping the first-declared rule ({winner[2]!r}).",
                        stacklevel=2,
                    )
            setter(winner[1])

        return self

    def cache(
        self,
        start_time: Optional[datetime.datetime] = None,
        end_time: Optional[datetime.datetime] = None,
        step_size: Optional[int] = None,
        simulator: Optional["core.Simulator"] = None,
    ) -> None:
        """
        Cache data and create folder structure for time series data.

        Args:
            start_time (Optional[datetime.datetime]): Start time for caching.
            end_time (Optional[datetime.datetime]): End time for caching.
            step_size (Optional[int]): Time step size for caching.
            simulator (Optional[core.Simulator]): Simulator instance passed to the
                data-source components' initialize methods.
        """
        self.simulation_model.cache(
            start_time=start_time,
            end_time=end_time,
            step_size=step_size,
            simulator=simulator,
        )

    def initialize(
        self,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
        step_size: int,
    ) -> None:
        """
        Initialize the model for simulation.

        Args:
            start_time (datetime.datetime): Start time for the simulation.
            end_time (datetime.datetime): End time for the simulation.
            step_size (int): Time step size for the simulation.
        """
        self.simulation_model.initialize(start_time, end_time, step_size)

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
        filename: Optional[str] = None,
        fcn: Optional[Callable] = None,
        draw_semantic_model: bool = True,
        draw_simulation_model: bool = True,
        validate_model: bool = True,
        force_config_overwrite: bool = False,
        logfile: Optional[str] = None,
        enable_fusion: bool = True,
        semantic_model_filename: Optional[str] = None,
        simulation_model_filename: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Load and set up the simulation half of the model.

        Preferred usage::

            sm = tb.SemanticModel(rdf_file="building.ttl", id="building")
            model = tb.Translator().translate(sm)
            model.load()

        Or restore a serialized simulation model::

            model.load(filename="path/to/serialized_sim...")

        Args:
            filename: Path to a serialized simulation model (preferred).
            fcn: Custom function applied during model loading.
            draw_semantic_model: Whether to draw the semantic model graph.
            draw_simulation_model: Whether to draw the simulation model graph.
            validate_model: Whether to perform model validation.
            force_config_overwrite: If True, all parameters are read from config.
            logfile: Path to the plain LOGGER file.
            enable_fusion: Whether to fuse connected state-space clusters at load.
            semantic_model_filename: Deprecated (removed in 2.1). Use
                ``SemanticModel`` + ``Translator.translate`` instead.
            simulation_model_filename: Deprecated alias for ``filename`` (removed in 2.1).
        """
        from twin4build.utils.deprecation import deprecate_name

        if "verbose" in kwargs:
            deprecate_name("verbose=", "LOGGER.verbose")
            LOGGER.verbose = kwargs.pop("verbose")
        if kwargs:
            raise TypeError(
                f"Model.load() got unexpected keyword arguments: {sorted(kwargs)}"
            )

        if semantic_model_filename is not None:
            deprecate_name(
                "semantic_model_filename=",
                "SemanticModel(rdf_file=...) + Translator.translate(...)",
            )
        if simulation_model_filename is not None:
            deprecate_name("simulation_model_filename=", "filename=")
            if filename is None:
                filename = simulation_model_filename

        if LOGGER.verbose:
            self._load(
                semantic_model_filename=semantic_model_filename,
                simulation_model_filename=filename,
                fcn=fcn,
                draw_semantic_model=draw_semantic_model,
                draw_simulation_model=draw_simulation_model,
                validate_model=validate_model,
                force_config_overwrite=force_config_overwrite,
                logfile=logfile,
                enable_fusion=enable_fusion,
            )
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._load(
                    semantic_model_filename=semantic_model_filename,
                    simulation_model_filename=filename,
                    fcn=fcn,
                    draw_semantic_model=draw_semantic_model,
                    draw_simulation_model=draw_simulation_model,
                    validate_model=validate_model,
                    force_config_overwrite=force_config_overwrite,
                    logfile=logfile,
                    enable_fusion=enable_fusion,
                )

    def _load(
        self,
        semantic_model_filename: Optional[str],
        simulation_model_filename: Optional[str],
        fcn: Optional[Callable],
        draw_semantic_model: bool,
        draw_simulation_model: bool,
        validate_model: bool,
        force_config_overwrite: bool,
        logfile: Optional[str],
        enable_fusion: bool = True,
    ) -> None:
        """
        Internal method to load and set up the model for simulation.

        This method is called by load and performs the actual loading process.

        Args:
            semantic_model_filename: Path to the semantic model configuration file.
            simulation_model_filename: Path to the simulation model configuration file.
            fcn: Custom function to be applied during model loading.
            draw_semantic_model: Whether to create and save the object graph.
            draw_simulation_model: Whether to create and save the system graph.
            validate_model: Whether to perform model validation.
            force_config_overwrite: Whether to force the configuration file to be overwritten.
            logfile: Path to the log file.
        """
        assert (
            semantic_model_filename is None or simulation_model_filename is None
        ), "Providing both semantic_model_filename and simulation_model_filename is currently not supported."

        # if self._is_loaded:
        #     warnings.warn("The model is already loaded. Resetting model.")
        #     self.reset()

        # if verbose is not None:
        #     LOGGER.verbose = verbose
        LOGGER.logfile = logfile

        LOGGER.task("Loading model")
        LOGGER.add_level()
        # self.add_outdoor_environment()
        if semantic_model_filename is not None:
            apply_translator = True
            LOGGER.task("Parsing semantic model")
            self._semantic_model = core.SemanticModel(
                rdf_file=semantic_model_filename,
                namespaces={"T4B": core.namespace.T4B},
                dir_conf=self.dir_conf + ["semantic_model"],
                id=f"{self._id}_semantic_model",
            )
            # self._semantic_model.reason()
            LOGGER.ok("Parsing semantic model", change_status=True)
            if draw_semantic_model:
                app_path = shutil.which("dot")
                assert (
                    app_path is not None
                ), "dot not found. Is Graphviz installed? If you are purposefully using twin4build without Graphviz, you should set draw_semantic_model to False."
                LOGGER.task("Drawing semantic model")
                LOGGER.add_level()
                self._semantic_model.visualize()
                LOGGER.remove_level()
                LOGGER.ok("Drawing semantic model", change_status=True)

        else:
            apply_translator = False

        if apply_translator:
            self._translator = core.Translator()
            # ``translate`` now returns a fully-wired ``Model``; extract the
            # simulation half and discard the wrapper since ``self`` is the
            # Model we are populating here.
            translated_model = self._translator.translate(
                self._semantic_model
            )
            self._simulation_model = translated_model.simulation_model
            self._simulation_model.dir_conf = self.dir_conf + ["simulation_model"]

        self._simulation_model.enable_fusion = enable_fusion
        self._simulation_model.load(
            rdf_file=simulation_model_filename,
            fcn=fcn,
            validate_model=validate_model,
            force_config_overwrite=force_config_overwrite,
            logfile=logfile,
        )

        if draw_simulation_model:
            # Get all filenames generated in the folder dirname
            app_path = shutil.which("dot")
            assert (
                app_path is not None
            ), "dot not found. Is Graphviz installed? If you are purposefully using twin4build without Graphviz, you should set draw_simulation_model to False."

            LOGGER.task("Drawing simulation model")
            LOGGER.add_level()
            self._simulation_model.visualize()
            LOGGER.remove_level()
            LOGGER.ok("Drawing simulation model", change_status=True)

        LOGGER.remove_level()
        LOGGER.ok("Loading model", change_status=True)

        # LOGGER.reset()

    def fcn(self) -> None:
        """
        Placeholder for a custom function to be applied during model loading.
        """

    def set_save_simulation_result(self, flag: bool = True, c: list = None):
        """
        Enable or disable history logging of simulation results for components.

        Args:
            flag (bool): Whether to save (log) simulation results.
            c (Optional[list]): List of components to apply the flag to.
                If None, the flag is applied to all components in the model.
        """
        self.simulation_model.set_save_simulation_result(flag=flag, c=c)

    def load_estimation_result(
        self,
        filename: Optional[str] = None,
        result: Optional[Dict] = None,
        # verbose: int = 0,
    ) -> None:
        """
        Load an estimation result from a file or dictionary.

        Args:
            filename (Optional[str]): The filename to load the estimation result from.
            result (Optional[Dict]): The estimation result dictionary to load.

        Raises:
            AssertionError: If invalid arguments are provided.
        """
        self.simulation_model.load_estimation_result(
            filename=filename,
            result=result,
            # verbose=verbose,
        )

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
            core.SemanticObject: The semantic object mapped to the simulation component.

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
        Serialize both halves of the model.
        """
        self._semantic_model.serialize()
        self._simulation_model.serialize()

    def visualize(self, **kwargs) -> None:
        """
        Visualize the model.  Keyword arguments are forwarded to
        :meth:`SimulationModel.visualize` (e.g. ``forward_only=True``,
        ``compressed=True``).  The semantic model is only re-drawn when
        called without arguments to match the standalone behaviour.
        """
        if not kwargs:
            self._semantic_model.visualize()
        self._simulation_model.visualize(**kwargs)


    def build_compiled_model(self) -> "Model":
        """Build a compiled Model with batched meta components.

        Groups components within each execution group by their signature
        (class, port structure, parameter keys, state hints).  For each group
        of compatible components a single *meta component* is created -- an
        instance of the **same class** whose ``n_c`` (parallel-components)
        dimension equals the group size.

        ``tps.Parameter`` / ``tps.TensorParameter`` attributes listed in
        ``component.parameter`` are stacked along the ``n_c`` axis so that a
        single ``do_step`` call computes all batched components in parallel
        via PyTorch broadcasting.

        **After calling this method:**

        1. Call ``compiled.load(draw_semantic_model=False,
           draw_simulation_model=False)`` to compute execution order.
        2. When initialising the compiled model for simulation, pass
           ``n_c = component._n_c_compiled`` to every I/O port's
           ``initialize()`` so that tensors are allocated with the correct
           parallel-component dimension.

        Returns
        -------
        Model
            A new ``Model`` whose components are the batched meta components
            and whose connections mirror the original model's wiring.

        Notes
        -----
        * The mapping from original component ids to ``(meta, i_c)`` is
          stored in ``self._component_to_meta`` for later look-up.
        * Components with ``n_c == 1`` (singletons) are still wrapped in a
          fresh meta instance for uniformity.
        * Constructor defaults are used when instantiating meta components;
          all component classes must therefore accept ``id`` as their only
          required keyword argument.
        """
        compiled = Model(id=f"{self.id}_compiled")
        self._component_to_meta = {}

        # -- Phase 1: create one meta component per signature group --------
        # Classes that must never be lumped together even when they share a
        # signature.  Each instance is kept as-is (n_c = 1).
        from twin4build.systems.outdoor_environment.outdoor_environment_system import OutdoorEnvironmentSystem
        from twin4build.systems.schedule.schedule_system import ScheduleSystem
        from twin4build.systems.sensor.sensor_system import SensorSystem
        _NO_BATCH_CLASSES = (OutdoorEnvironmentSystem, ScheduleSystem, SensorSystem)

        for group_idx, group in enumerate(
            self.simulation_model.execution_order
        ):
            by_sig: "OrderedDict[Tuple[Any, ...], List[Any]]" = OrderedDict()
            for comp in group:
                sig = self._component_signature(comp)
                if isinstance(comp, _NO_BATCH_CLASSES):
                    # Force a unique key so this component is never grouped
                    sig = (*sig, id(comp))
                by_sig.setdefault(sig, []).append(comp)

            for blk_idx, (sig, comps) in enumerate(by_sig.items()):
                n_c = len(comps)
                cls = comps[0].__class__
                if n_c == 1:
                    meta = cls(id=comps[0].id)
                    meta._n_c_compiled = 1
                    meta._source_component_ids = (meta.id,)
                    self._copy_data_source_attrs(comps[0], meta)
                else:
                    meta_id = f"g{group_idx}_b{blk_idx}_{cls.__name__}"
                    meta = cls(id=meta_id)
                    meta._n_c_compiled = n_c
                    meta._source_component_ids = tuple(
                        c.id for c in comps
                    )

                self._batch_parameters(meta, comps, n_c)
                self._copy_init_attrs(meta, comps[0])

                for i_c, c in enumerate(comps):
                    self._component_to_meta[c.id] = (meta, i_c)

                compiled.add_component(meta)

        # -- Phase 2: wire connections between meta components -------------
        # Collect all (sender_ic, receiver_ic) pairs per unique connection
        # key so we can build the i_c mapping tensors.
        connection_map: Dict[tuple, dict] = {}
        for group in self.simulation_model.execution_order:
            for comp in group:
                for conn in comp.connected_through:
                    for cp in conn.connects_system_at:
                        receiver = cp.connection_point_of
                        s_meta, s_ic = self._component_to_meta[comp.id]
                        r_meta, r_ic = self._component_to_meta[receiver.id]

                        key = (
                            id(s_meta),
                            conn.output_port,
                            id(r_meta),
                            cp.input_port,
                        )
                        if key not in connection_map:
                            connection_map[key] = {
                                "s_meta": s_meta,
                                "r_meta": r_meta,
                                "output_port": conn.output_port,
                                "input_port": cp.input_port,
                                "out_v_idx": cp.output_port_index.get(conn),
                                "in_v_idx": cp.input_port_index.get(conn),
                                "ic_pairs": [],
                            }
                        connection_map[key]["ic_pairs"].append((s_ic, r_ic))

        for info in connection_map.values():
            compiled.add_connection(
                info["s_meta"],
                info["r_meta"],
                info["output_port"],
                info["input_port"],
                output_port_index=info["out_v_idx"],
                input_port_index=info["in_v_idx"],
            )

            pairs = info["ic_pairs"]
            s_n_c = getattr(info["s_meta"], "_n_c_compiled", 1)
            r_n_c = getattr(info["r_meta"], "_n_c_compiled", 1)

            if s_n_c == 1 and r_n_c == 1:
                continue

            sorted_pairs = sorted(pairs)
            is_aligned = (
                len(sorted_pairs) == s_n_c == r_n_c
                and all(s == r for s, r in sorted_pairs)
                and [p[0] for p in sorted_pairs] == list(range(s_n_c))
            )
            if is_aligned:
                continue

            s_ics = torch.tensor([p[0] for p in pairs], dtype=torch.long)
            r_ics = torch.tensor([p[1] for p in pairs], dtype=torch.long)

            # Walk the compiled model's connection graph to find the
            # ConnectionPoint + Connection objects we just created.
            for cp in info["r_meta"].connects_at:
                if cp.input_port != info["input_port"]:
                    continue
                for conn_obj in cp.connects_system_through:
                    if (
                        conn_obj.connects_system is info["s_meta"]
                        and conn_obj.output_port == info["output_port"]
                    ):
                        cp.set_output_component_index(conn_obj, s_ics)
                        cp.set_input_component_index(conn_obj, r_ics)
                        break
                break

        return compiled

    # -- compiled-model look-ups ------------------------------------------

    def get_block_id_for_component(
        self, component_id: str
    ) -> Optional[str]:
        """Return the meta-component id that batches *component_id*."""
        entry = self._component_to_meta.get(component_id)
        if entry is None:
            return None
        meta, _ = entry
        return meta.id

    def get_compiled_component_info(
        self, component_id: str
    ) -> Optional[Tuple[Any, int]]:
        """Return ``(meta_component, i_c_index)`` for an original component.

        Returns ``None`` if *component_id* has not been compiled.
        """
        return self._component_to_meta.get(component_id)

    # -- data-source attribute copying ------------------------------------

    _DATA_SOURCE_ATTRS = (
        # boolean flags
        "use_spreadsheet", "use_database", "use_df", "use_dict",
        # ScheduleSystem / SensorSystem
        "filename", "df", "date_column", "value_column",
        "uuid", "name", "dbconfig",
        # ScheduleSystem rulesets
        "weekday_ruleset", "weekend_ruleset",
        "monday_ruleset", "tuesday_ruleset",
        "wednesday_ruleset", "thursday_ruleset",
        "friday_ruleset", "saturday_ruleset",
        "sunday_ruleset",
        # OutdoorEnvironmentSystem
        "filename_outdoorTemperature", "filename_globalIrradiation",
        "filename_outdoorCo2Concentration",
        "datecolumn_outdoorTemperature", "valuecolumn_outdoorTemperature",
        "datecolumn_globalIrradiation", "valuecolumn_globalIrradiation",
        "datecolumn_outdoorCo2Concentration",
        "valuecolumn_outdoorCo2Concentration",
        "uuid_outdoorTemperature", "dbconfig_outdoorTemperature",
        "uuid_globalIrradiation", "dbconfig_globalIrradiation",
        "uuid_outdoorCo2Concentration", "dbconfig_outdoorCo2Concentration",
    )

    def _copy_data_source_attrs(self, src: Any, dst: Any) -> None:
        """Copy data-source configuration from *src* to *dst*.

        Copies boolean flags (``use_spreadsheet``, ``use_database``,
        ``use_df``, ``use_dict``) and their associated file / database /
        ruleset attributes so that a freshly instantiated component can
        initialise its data source the same way the original could.
        """
        for attr in self._DATA_SOURCE_ATTRS:
            val = getattr(src, attr, None)
            if val is None:
                continue
            try:
                setattr(dst, attr, val)
            except AttributeError:
                pass

    # -- parameter batching -----------------------------------------------

    @staticmethod
    def _resolve_dotted_attr(obj: Any, dotted_name: str) -> Any:
        """Traverse a dotted attribute path (e.g. ``"thermal.C_air"``)."""
        for part in dotted_name.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                return None
        return obj

    @staticmethod
    def _set_dotted_attr(obj: Any, dotted_name: str, value: Any) -> None:
        """Set an attribute via a dotted path (e.g. ``"thermal.C_air"``)."""
        parts = dotted_name.split(".")
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], value)

    def _batch_parameters(
        self, meta: Any, components: List, n_c: int
    ) -> None:
        """Stack calibration parameters along the ``n_c`` dimension.

        Only ``tps.Parameter`` and ``tps.TensorParameter`` attributes whose
        names appear in ``components[0].parameter`` are batched.  Each
        original component contributes one slice along ``n_c``.

        Supports dotted parameter names (e.g. ``"thermal.C_air"``) for
        composite components that wrap sub-models.
        """
        param_dict = getattr(components[0], "parameter", None)
        if not param_dict:
            return

        for param_name in param_dict:
            originals = []
            for c in components:
                p = self._resolve_dotted_attr(c, param_name)
                if p is None:
                    return
                originals.append(p)

            first = originals[0]

            if isinstance(first, tps.Parameter):
                vals = torch.stack([p.get().squeeze() for p in originals])
                mins = torch.stack(
                    [p.min_value.squeeze() for p in originals]
                )
                maxs = torch.stack(
                    [p.max_value.squeeze() for p in originals]
                )
                self._set_dotted_attr(
                    meta,
                    param_name,
                    tps.Parameter(
                        vals,
                        min_value=mins,
                        max_value=maxs,
                        requires_grad=first.requires_grad,
                        n_c=n_c,
                    ),
                )

            elif isinstance(first, tps.TensorParameter):
                vals = torch.stack([p.get().squeeze() for p in originals])
                mins = (
                    torch.stack(
                        [p.min_value.squeeze() for p in originals]
                    )
                    if first.min_value is not None
                    else None
                )
                maxs = (
                    torch.stack(
                        [p.max_value.squeeze() for p in originals]
                    )
                    if first.max_value is not None
                    else None
                )
                self._set_dotted_attr(
                    meta,
                    param_name,
                    tps.TensorParameter(
                        vals,
                        min_value=mins,
                        max_value=maxs,
                        normalized=False,
                        n_c=n_c,
                    ),
                )

    # -- non-parameter attribute copying -----------------------------------

    # Attributes that must be copied verbatim from a source component to
    # the meta component because they affect model structure (matrix sizes,
    # topology) but are plain values rather than tps.Parameter instances.
    # Keyed by fully-qualified class name.  Supports dotted paths for
    # composite components (e.g. ``"thermal.some_attr"``).
    _INIT_ATTRS_TO_COPY: Dict[str, Tuple[str, ...]] = {
        "twin4build.systems.space_heater.space_heater_system.SpaceHeaterSystem": (
            "nelements",
            "Q_flow_nominal_sh",
            "T_a_nominal_sh",
            "T_b_nominal_sh",
            "TAir_nominal_sh",
        ),
        "twin4build.systems.building_space.building_space_system.BuildingSpaceSystem": (),
        "twin4build.systems.controller.setpoint_controller.pid_controller"
        ".pid_controller_system.PIDControllerSystem": (
            "is_reverse",
        ),
    }

    def _copy_init_attrs(
        self, meta: Any, source: Any
    ) -> None:
        """Copy non-Parameter constructor attributes from *source* to *meta*.

        Only the attributes listed in ``_INIT_ATTRS_TO_COPY`` for the
        component's class are copied.  This ensures that the meta component
        has the correct structural values (e.g. ``nelements``) even though
        ``_batch_parameters`` only handles ``tps.Parameter`` objects.
        Supports dotted paths for composite components.

        Additionally, for composite building-space components, topology
        values (``n_walls``, ``n_boundary_temperature``) are
        computed from the source component's connection graph -- which is
        available even before ``initialize()`` has been called.
        """
        fqn = f"{source.__class__.__module__}.{source.__class__.__name__}"
        attrs = self._INIT_ATTRS_TO_COPY.get(fqn, ())
        for attr in attrs:
            val = self._resolve_dotted_attr(source, attr)
            if val is not None:
                self._set_dotted_attr(meta, attr, val)

        # For BuildingSpaceSystem: derive topology from the source
        # component's connects_at (populated by model.load()) and push
        # it through the thermal sub-model's setter so the manual flag
        # is set and initialize() skips its own connects_at discovery.
        from twin4build.systems.building_space.building_space_system import (
            BuildingSpaceSystem,
        )
        if isinstance(source, BuildingSpaceSystem):
            cp_boundary = [
                cp for cp in source.connects_at
                if cp.input_port == "boundaryTemperature"
            ]
            n_boundary = (
                len(cp_boundary[0].connects_system_through) if cp_boundary else 0
            )
            cp_wall = [
                cp for cp in source.connects_at
                if cp.input_port == "wallHeatGain"
            ]
            n_walls = (
                len(cp_wall[0].connects_system_through) if cp_wall else 0
            )
            meta.thermal.n_walls = n_walls
            meta.thermal.n_boundary_temperature = n_boundary

    # -- signature helpers ------------------------------------------------

    def _component_signature(self, component: Any) -> Tuple[Any, ...]:
        input_signature = self._port_dict_signature(
            getattr(component, "input", {})
        )
        output_signature = self._port_dict_signature(
            getattr(component, "output", {})
        )

        parameter_keys: Tuple[str, ...] = tuple(
            sorted(getattr(component, "parameter", {}).keys())
        )

        state_hints = (
            getattr(component, "n_states", None),
            getattr(component, "n_inputs", None),
            getattr(component, "n_outputs", None),
        )

        return (
            component.__class__.__module__,
            component.__class__.__name__,
            input_signature,
            output_signature,
            parameter_keys,
            state_hints,
        )

    def _port_dict_signature(
        self, port_dict: Dict[str, Any]
    ) -> Tuple[Any, ...]:
        items = []
        for key in sorted(port_dict.keys()):
            items.append((key, self._port_signature(port_dict[key])))
        return tuple(items)

    def _port_signature(self, port: Any) -> Tuple[Any, ...]:
        if isinstance(port, tps.Scalar):
            return (
                "Scalar",
                bool(getattr(port, "_optional", False)),
                bool(getattr(port, "_is_leaf", False)),
            )
        if isinstance(port, tps.Vector):
            return (
                "Vector",
                getattr(port, "_n_v", None),
                bool(getattr(port, "_optional", False)),
                bool(getattr(port, "_is_leaf", False)),
            )
        return (port.__class__.__name__,)