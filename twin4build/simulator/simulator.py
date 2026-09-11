from __future__ import annotations

# Standard library imports
import datetime
import time
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

# Third party imports
import numpy as np
import pandas as pd
import torch
from fmpy.fmi1 import FMICallException
from tqdm import tqdm

# Local application imports
import twin4build.core as core
import twin4build.systems as systems
def _has_triton() -> bool:
    """Inductor's CUDA backend needs Triton; absent on Windows torch wheels."""
    try:
        from torch.utils._triton import has_triton
    except Exception:  # noqa: BLE001
        return False
    try:
        return bool(has_triton())
    except Exception:  # noqa: BLE001
        return False


def _functorch_active() -> bool:
    try:
        return bool(torch._C._are_functorch_transforms_active())
    except AttributeError:  # pragma: no cover - very old torch
        return False


from twin4build.simulator._functional import (
    FunctionalModel,
    StateLayout,
    collect_stateful,
    functional_rollout,
    functional_rollout_batched,
    record_exogenous_inputs,
)
from twin4build.simulator._functional_simulation import run_functional_simulation
from twin4build.utils.deprecation import reject_unexpected_kwargs
from twin4build.utils.logger import LOGGER
from twin4build.utils.simulation_time import get_simulation_timesteps
from twin4build.utils.validate_period import validate_period

# import george
# from george import kernels


class Simulator:
    r"""
    A simulator for building digital twins.

    This class simulates :class:`~twin4build.model.Model` or :class:`~twin4build.model.simulation_model.simulation_model.SimulationModel` in a time-stepping manner.
    It takes a prepared model with a predetermined execution order and runs the
    simulation by calling each component in sequence for each timestep.

    The simulator handles the coordination between components, ensuring that
    outputs from one component are properly passed as inputs to connected
    components during each simulation timestep.

    Args:
        model: The model to be simulated.

    Mathematical Formulation
    ------------------------

    The simulator operates on a directed multigraph :math:`G = (V, E, \iota, \alpha, \beta)` comprising:

    .. math::

        V = \{c_1, c_2, ..., c_n\}

    .. math::

        E = \{e_1, e_2, e_3, ...\}

    .. math::

        \iota: E \rightarrow V \times V

    .. math::

        \alpha: E \rightarrow \text{Ports}

    .. math::

        \beta: E \rightarrow \text{Ports}

    where:
        - :math:`V` is the set of vertices (components)
        - :math:`E` is the set of edge identifiers (connections between components)
        - :math:`\iota` is the incidence function mapping edges to vertex pairs
        - :math:`\alpha` maps each edge to an input port
        - :math:`\beta` maps each edge to an output port
        - Each edge :math:`e_a \in E` with :math:`\iota(e_a) = (c_i, c_j)` indicates that component :math:`c_i` provides input to component :math:`c_j`
        - Multiple edges can map to the same vertex pair (multigraph): :math:`\iota(e_a) = \iota(e_b) = (c_i, c_j)`

    Execution Sequence
    ~~~~~~~~~~~~~~~~~~

    The execution sequence is determined by the model preparation phase
    (see :class:`~twin4build.model.simulation_model.simulation_model.SimulationModel`):

    .. math::

        L = (c_1, c_2, ..., c_n)

    Time-Stepping Simulation
    ~~~~~~~~~~~~~~~~~~~~~~~~

    For each timestep :math:`t \in (t_{start}, t_{start} + \Delta t, ..., t_{end})`,
    the simulator executes each component :math:`c_j` in the specified order :math:`L`.

    First, for component :math:`c_j`, collect inputs from all connected components:

    Component :math:`c_j` has input vector :math:`\mathbf{x}_j \in \mathbb{R}^{n_j^{in}}` and output vector :math:`\mathbf{y}_j \in \mathbb{R}^{n_j^{out}}`
    where :math:`n_j^{in}` and :math:`n_j^{out}` are the numbers of input and output ports respectively.

    For each input edge of component :math:`c_j`: :math:`e_i \in E` with :math:`\iota(e_i) = (c_i, c_j)`:

    .. math::

        x_{j,\alpha(e_i)} = y_{i,\beta(e_i)}

    where:

        - :math:`\alpha(e_i)` and :math:`\beta(e_i)` are the input and output ports for edge :math:`e_i`

    After collecting the inputs, execute the step function of the component:

    .. math::

        \mathbf{y}_{j,t} = f_j(\mathbf{x}_{j,t}, \mathbf{s}_{j,t}, t, \Delta t)

    where:

        - :math:`\mathbf{x}_{j,t}` is the input sequence for component :math:`j` at time :math:`t`
        - :math:`\mathbf{y}_{j,t}` is the output sequence from component :math:`j` at time :math:`t`
        - :math:`\mathbf{s}_{j,t}` is the internal state of component :math:`j` at time :math:`t`
        - :math:`f_j` is the component's dynamics function
        - :math:`\alpha(e)` and :math:`\beta(e)` define the specific input/output ports for edge :math:`e`

    Shorthand Notation
    ~~~~~~~~~~~~~~~~~~

    The complete simulation process described above can be represented using the compact notation:

    .. math::

        \boldsymbol{\hat{Y}} = \mathcal{M}(\boldsymbol{X}, \boldsymbol{t}, \boldsymbol{\theta})

    where:
        - :math:`\mathcal{M}` represents the complete simulation model (this Simulator class)
        - :math:`\boldsymbol{X} \in \mathbb{R}^{n_x \times n_t}` are the input variables (disturbances, setpoints, etc.)
        - :math:`\boldsymbol{t} \in \mathbb{R}^{n_t}` are the simulation timesteps
        - :math:`\boldsymbol{\theta} \in \mathbb{R}^{n_p}` are the model parameters
        - :math:`\boldsymbol{\hat{Y}} \in \mathbb{R}^{n_y \times n_t}` are the system outputs (predictions, performance metrics)

    This notation encapsulates the entire time-stepping simulation process including component
    execution order, input gathering, and temporal evolution as described in the sections above.
    This is what happens when we call :class:`~twin4build.simulator.Simulator.simulate`.
    We will use this notation in other parts of the documentation.

    Examples
    --------
    Basic simulation execution:

    >>> import twin4build as tb
    >>> import datetime
    >>>
    >>> # Create and prepare model
    >>> model = tb.SimulationModel(id="building_model")
    >>> # ... add components and connections ...
    >>> model.load()  # Prepares execution order
    >>>
    >>> # Create simulator and run simulation
    >>> simulator = tb.Simulator(model)
    >>> start_time = datetime.datetime(2024, 1, 1, 0, 0, 0)
    >>> end_time = datetime.datetime(2024, 1, 2, 0, 0, 0)
    >>> step_size = 3600  # 1 hour
    >>>
    >>> simulator.simulate(
    ...     start_time=start_time,
    ...     end_time=end_time,
    ...     step_size=step_size
    ... )
    >>>
    >>> # Access simulation results from the component output ports
    >>> space = model.components["space"]
    >>> temperature_history = space.output["indoorTemperature"].history()
    """

    _EXECUTION_MODES = ("object", "functional")
    _EXECUTION_BACKENDS = ("eager", "cuda_graph")
    _COMPILE_STEP_OPTIONS = (True, False, "auto")

    def __init__(
        self,
        model: core.Model,
        execution_mode: str = "object",
        execution_backend: str = "eager",
        compile_step: Union[bool, str] = "auto",
    ):
        """
        Initialize the Simulator instance.

        Creates a new simulator object that can be used to run simulations
        and perform parameter estimation or optimization.

        Args:
            model: The model to be simulated.
            execution_mode: ``"object"`` for normal component stepping or
                ``"functional"`` for the sequential ``F_aug`` rollout.
            compile_step: ``True``, ``False`` or ``"auto"`` (default).  Compile
                the functional transform-mode step with ``torch.compile``
                (Inductor) before it is captured or run eagerly.  ``"auto"``
                enables it on CUDA when the torch build has Triton; ``True``
                requires Triton and raises otherwise.  Compiled steps keep the
                same results (relative differences at 1e-15) with a several-
                fold smaller CUDA graph and faster replay; the first call pays
                a one-time compile of tens of seconds.
            execution_backend: ``"eager"`` or ``"cuda_graph"``. CUDA Graphs
                require functional mode and a CUDA model.

        Notes:
            The simulator maintains internal state about the current simulation,
            including time steps and component states.
        """
        if execution_mode not in self._EXECUTION_MODES:
            raise ValueError(
                f"execution_mode must be one of {self._EXECUTION_MODES}; "
                f"got {execution_mode!r}"
            )
        if execution_backend not in self._EXECUTION_BACKENDS:
            raise ValueError(
                f"execution_backend must be one of {self._EXECUTION_BACKENDS}; "
                f"got {execution_backend!r}"
            )
        if execution_backend == "cuda_graph" and execution_mode != "functional":
            raise ValueError(
                "execution_backend='cuda_graph' requires " "execution_mode='functional'"
            )
        if compile_step not in self._COMPILE_STEP_OPTIONS:
            raise ValueError(
                f"compile_step must be one of {self._COMPILE_STEP_OPTIONS}; "
                f"got {compile_step!r}"
            )
        if compile_step is True and not _has_triton():
            raise ValueError(
                "compile_step=True requires a Triton-capable torch build "
                "(Inductor's CUDA backend); this torch has none. Use "
                "compile_step='auto' to compile only where available."
            )
        self.model = model
        self.execution_mode = execution_mode
        self.execution_backend = execution_backend
        self.compile_step = compile_step
        self._functional_session = None
        self._functional_setup_count = 0
        self._functional_setup_seconds = 0.0
        self._exogenous_recording_cache = None
        self._exogenous_recording_cache_hit = False
        self._exogenous_recording_cache_hits = 0
        self._exogenous_recording_cache_misses = 0
        self._cuda_graph_capture_count = 0
        self._cuda_graph_replay_count = 0
        self._cuda_graph_capture_seconds = 0.0
        self._cuda_graph_replay_seconds = 0.0

    def clear_execution_cache(self) -> None:
        """Discard cached functional sessions and exogenous recordings.

        Cache keys detect normal changes to periods, initialized source data,
        model parameters and topology. Call this method after mutating a custom
        data source whose state is not represented by tensor-like component
        attributes or port histories.
        """
        session, self._functional_session = self._functional_session, None
        if session is not None:
            session.close()
        self._exogenous_recording_cache = None
        self._exogenous_recording_cache_hit = False

    def close(self) -> None:
        """Deterministically release cached execution resources."""
        self.clear_execution_cache()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
        return False

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    @staticmethod
    def _assign_component_inputs(
        component: core.System,
        step_index: int,
    ) -> None:
        """
        Assign inputs to a component from connected components.

        Args:
            component (core.System): The component to assign inputs to.
            step_index (int): The current timestep index.

        """
        # Gather all needed inputs for the component through all ingoing connections
        for connection_point in component.connects_at:
            for connection in connection_point.connects_system_through:
                connected_component = connection.connects_system
                input_port_index = connection_point.input_port_index[connection]
                output_port_index = connection_point.output_port_index[connection]
                output_component_index = connection_point.output_component_index.get(
                    connection, slice(None)
                )
                input_component_index = connection_point.input_component_index.get(
                    connection, slice(None)
                )
                try:
                    component.input[connection_point.input_port]._set(
                        connected_component.output[connection.output_port].get(
                            i_v=output_port_index,
                            i_c=output_component_index,
                        ),
                        i_t=step_index,
                        i_v=input_port_index,
                        i_c=input_component_index,
                    )
                except IndexError as exc:
                    raise IndexError(
                        f"Batched component-index mapping failed for "
                        f"{connected_component.id}.{connection.output_port} -> "
                        f"{component.id}.{connection_point.input_port}; "
                        f"output_i_c={output_component_index}, "
                        f"input_i_c={input_component_index}"
                    ) from exc

    @staticmethod
    def _validate_component_inputs(model):
        """Validate connected input histories with one normal-path host sync."""
        connected_ports = []
        seen = set()
        for component in model.components.values():
            for connection_point in component.connects_at:
                port_name = connection_point.input_port
                port = component.input[port_name]
                if id(port) not in seen:
                    connected_ports.append((component, port_name, port))
                    seen.add(id(port))

        checks = [
            torch.isfinite(
                port._history if port._history is not None else port._tensor
            ).all()
            for _, _, port in connected_ports
        ]
        if not checks:
            return 0, 0
        all_finite = bool(torch.stack(checks).all().item())
        if all_finite:
            return len(checks), 1

        for component, port_name, port in connected_ports:
            value = port._history if port._history is not None else port._tensor
            invalid = ~torch.isfinite(value)
            if not bool(invalid.any().item()):
                continue
            index = invalid.nonzero()[0].detach().cpu().tolist()
            location = []
            labels = (
                ("timestep", "period", "component_index", "vector_index")
                if port._history is not None
                else ("period", "component_index", "vector_index")
            )
            for label, position in zip(labels, index):
                location.append(f"{label}={position}")
            offending = value[tuple(index)].detach().cpu().item()
            LOGGER.debug("Component input history: %s", value)
            raise ValueError(
                f"Input {port_name} of component {component.id} is non-finite "
                f"at {', '.join(location)}; value={offending}"
            )
        raise ValueError("Connected component input validation failed")

    @staticmethod
    def _do_system_time_step(
        model: core.Model,
        second_time: List[float],
        date_time: List[datetime.datetime],
        step_size: List[int],
        step_index: int,
        iteration_method: str,
    ) -> None:
        """
        Execute a time step for all components in the model.

        This method executes components in the order specified by the model's execution
        order, ensuring proper propagation of information through the system. It:
        1. Executes components in groups based on dependencies
        2. Updates component states after all executions
        3. Handles both FMU and non-FMU components

        The iteration method (gauss-seidel or jacobi) determines how inputs are assigned:
        - gauss-seidel: Inputs are assigned immediately before each component executes,
          allowing later components to use updated outputs from earlier components
        - jacobi: All components execute first using previous inputs, then all inputs
          are assigned for the next timestep

        Args:
            model (core.Model): The model containing components to simulate.
            second_time (List[float]): Per-period simulation time in seconds at this step.
            date_time (List[datetime.datetime]): Per-period datetime at this step.
            step_size (List[int]): Per-period step size in seconds.
            step_index (int): The current timestep index.
            iteration_method (str): "gauss-seidel" or "jacobi" (see above).

        Notes:
            - Components are executed sequentially based on their dependencies
            - Component execution order is determined by the model's execution_order attribute
        """
        if iteration_method == "gauss-seidel":
            for component_group in model.execution_order:
                for component in component_group:
                    Simulator._assign_component_inputs(component, step_index)
                    component.do_step(
                        second_time,
                        date_time,
                        step_size,
                        step_index,
                    )

        elif iteration_method == "jacobi":
            # Iterate the EXECUTING components (flat execution order), not
            # model.components: fused state-space clusters execute through
            # their FusedStateSpaceSystem, and the member components must not
            # step themselves.
            executing = getattr(model, "_flat_execution_order", None)
            if executing is None:
                executing = model.flat_execution_order

            # Execute all components first
            for component in executing:
                component.do_step(
                    second_time,
                    date_time,
                    step_size,
                    step_index,
                )

            # Then assign inputs for next timestep
            for component in executing:
                Simulator._assign_component_inputs(component, step_index)

    @staticmethod
    def get_simulation_timesteps(
        start_time: Union[List[datetime.datetime], datetime.datetime],
        end_time: Union[List[datetime.datetime], datetime.datetime],
        step_size: Union[List[int], int],
    ) -> Tuple[np.ndarray, np.ndarray, int, List[int]]:
        """
        Generate simulation timesteps between start and end times.

        Creates arrays of both second-based and datetime-based timesteps for each
        simulation period using the specified step sizes. Shorter periods are
        padded with NaN up to the longest period's length.

        Args:
            start_time: Start time(s) of the simulation. A single datetime or a
                list of datetimes (one per period).
            end_time: End time(s) of the simulation. Same form as ``start_time``.
            step_size: Step size(s) in seconds. A single int or a list of ints.

        Returns:
            Tuple of four elements:
                - second_time_steps (np.ndarray): Shape ``(n_periods, max_timesteps)``,
                  time in seconds since each period's start (NaN-padded).
                - date_time_steps (np.ndarray): Shape ``(n_periods, max_timesteps)``,
                  datetimes (NaN-padded).
                - max_timesteps (int): Length of the longest period.
                - n_timesteps (List[int]): Actual number of steps per period.
        """
        return get_simulation_timesteps(start_time, end_time, step_size)

    def set_simulation_timesteps(
        self, start_time: datetime.datetime, end_time: datetime.datetime, step_size: int
    ) -> None:
        """
        Compute and store simulation timesteps on the simulator instance.

        Sets the ``second_time_steps`` and ``date_time_steps`` attributes from
        :meth:`get_simulation_timesteps`.

        Args:
            start_time: Start time(s) of the simulation.
            end_time: End time(s) of the simulation.
            step_size: Step size(s) in seconds.
        """
        self.second_time_steps, self.date_time_steps, _, _ = (
            Simulator.get_simulation_timesteps(start_time, end_time, step_size)
        )

    def simulate(
        self,
        start_time: Union[List[datetime.datetime], datetime.datetime] = None,
        end_time: Union[List[datetime.datetime], datetime.datetime] = None,
        step_size: Union[List[int], int] = None,
        show_progress_bar: bool = True,
        iteration_method: str = "gauss-seidel",
        after_initialize=None,
        execution_mode: str = None,
        execution_backend: str = None,
        **kwargs,
    ) -> None:
        """
        Simulate the model between the specified dates with the given timestep.

        This method:
            1. Initializes the model and simulation parameters
            2. Generates simulation timesteps
            3. Executes the simulation loop with optional progress bar
            4. Updates component states at each timestep

        Args:
            start_time: Start time(s) of the simulation (timezone-aware). A single
                datetime or a list of datetimes for batched multi-period simulation.
            end_time: End time(s) of the simulation (timezone-aware). Same form as
                ``start_time``.
            step_size: Step size(s) in seconds. A single int or a list of ints.
            show_progress_bar: Whether to show a progress bar during simulation.
            iteration_method: The iteration method to use for component execution.
                - "gauss-seidel": Components are executed sequentially with immediate input updates (default)
                - "jacobi": All components execute first, then inputs are assigned
            after_initialize: Optional zero-argument callable fired after model
                (re)initialization and before the time loop. Used by the
                collocation transcription to inject per-segment initial states;
                ``None`` (default) is a no-op.
            execution_mode: Optional per-call override of ``"object"`` or
                ``"functional"``.
            execution_backend: Optional per-call override of ``"eager"`` or
                ``"cuda_graph"``.

        Raises:
            AssertionError: If input parameters are invalid or missing timezone info.
            FMICallException: If the FMU simulation fails.
        """
        mode = self.execution_mode if execution_mode is None else execution_mode
        backend = (
            self.execution_backend if execution_backend is None else execution_backend
        )
        if mode not in self._EXECUTION_MODES:
            raise ValueError(
                f"execution_mode must be one of {self._EXECUTION_MODES}; "
                f"got {mode!r}"
            )
        if backend not in self._EXECUTION_BACKENDS:
            raise ValueError(
                f"execution_backend must be one of {self._EXECUTION_BACKENDS}; "
                f"got {backend!r}"
            )
        if backend == "cuda_graph" and mode != "functional":
            raise ValueError(
                "execution_backend='cuda_graph' requires " "execution_mode='functional'"
            )
        self._last_execution_mode = mode
        self._last_execution_backend = backend

        for legacy_key, new_key in (
            ("startTime", "start_time"),
            ("endTime", "end_time"),
            ("stepSize", "step_size"),
        ):
            if legacy_key in kwargs:
                raise TypeError(
                    f"`{legacy_key}` has been removed. Use `{new_key}` instead."
                )
        reject_unexpected_kwargs("Simulator.simulate", kwargs)

        start_time, end_time, step_size = validate_period(
            start_time, end_time, step_size
        )

        self.debug_str = []  # TODO: remove this
        assert all(
            start_time_.tzinfo is not None for start_time_ in start_time
        ), "All start_times must have a timezone"
        assert all(
            end_time_.tzinfo is not None for end_time_ in end_time
        ), "All end_times must have a timezone"
        assert all(
            isinstance(step_size_, int) for step_size_ in step_size
        ), "All step_sizes must be integers"
        self.start_time = start_time
        self.end_time = end_time
        self.step_size = step_size
        self.iteration_method = iteration_method
        self.get_simulation_timesteps(start_time, end_time, step_size)
        second_time_steps, date_time_steps, max_timesteps, _ = (
            Simulator.get_simulation_timesteps(start_time, end_time, step_size)
        )
        self.second_time_steps = second_time_steps
        self.date_time_steps = date_time_steps
        self.n_timesteps = max_timesteps
        initialization_started = None
        functional_device = None
        if mode == "functional":
            simulation_model = (
                getattr(self.model, "_simulation_model", None) or self.model
            )
            functional_device = torch.device(
                getattr(simulation_model, "device", torch.device("cpu"))
            )
            if functional_device.type == "cuda":
                torch.cuda.synchronize(functional_device)
            initialization_started = time.perf_counter()
        self.model.initialize(start_time, end_time, step_size)
        if initialization_started is not None:
            if functional_device.type == "cuda":
                torch.cuda.synchronize(functional_device)
            self._functional_initialization_seconds = (
                time.perf_counter() - initialization_started
            )
        # Optional hook fired after (re)initialization, before the time loop.
        # Used by multiple-shooting / collocation estimation to overwrite each
        # segment's initial state with the optimizer's boundary decision
        # variables -- ``model.initialize`` above has just reset every stateful
        # component to its default/output-derived state, so this is the point
        # to inject the per-segment states.  Default ``None`` => no-op, so
        # ordinary simulation is unaffected.
        if after_initialize is not None:
            after_initialize()
        if mode == "functional":
            if iteration_method != "gauss-seidel":
                raise RuntimeError(
                    f"execution_mode={mode!r} supports only gauss-seidel " "semantics"
                )
            simulation_model = (
                getattr(self.model, "_simulation_model", None) or self.model
            )
            device = getattr(simulation_model, "device", torch.device("cpu"))
            if backend == "cuda_graph" and torch.device(device).type != "cuda":
                raise RuntimeError(
                    "execution_backend='cuda_graph' requires a functional "
                    "CUDA model; call model.to('cuda') first"
                )
            run_functional_simulation(
                self,
                backend,
                step_size,
                max_timesteps,
                len(start_time),
            )
            return
        self._last_execution_metadata = {
            "execution_mode": mode,
            "execution_backend": backend,
        }
        if show_progress_bar:
            for step_index in tqdm(
                range(max_timesteps),
                total=max_timesteps,
            ):
                second_time = second_time_steps[:, step_index]
                date_time = date_time_steps[:, step_index]

                self._do_system_time_step(
                    self.model,
                    second_time,
                    date_time,
                    step_size,
                    step_index,
                    iteration_method,
                )
        else:
            for step_index in range(max_timesteps):
                second_time = second_time_steps[:, step_index]
                date_time = date_time_steps[:, step_index]

                self._do_system_time_step(
                    self.model,
                    second_time,
                    date_time,
                    step_size,
                    step_index,
                    iteration_method,
                )
        validation_started = time.perf_counter()
        validation_check_count, validation_host_sync_count = (
            self._validate_component_inputs(self.model)
        )
        self._last_execution_metadata.update(
            {
                "validation_seconds": time.perf_counter() - validation_started,
                "validation_check_count": validation_check_count,
                "validation_host_sync_count": validation_host_sync_count,
            }
        )

    # -- functional-map simulation --------------------------------------------
    # The model can also be simulated as a pure sequential rollout of ONE
    # functional one-step map (every supported component's ``do_step``
    # delegates to a pure ``forward``).

    def build_functional_model(
        self,
        theta_spec=None,
        measurements=None,
        outputs=None,
        step_size=None,
    ):
        """Build the pure one-step map for the current model.

        Runs the shared structural checks and returns
        ``(layout, functional_model)``. Raises ``RuntimeError`` if the model
        cannot be expressed as a functional
        map; callers treat that as "fall back to the object-graph engine".

        Args:
            theta_spec: List of ``(component, attr)`` estimated parameters in
                decision-vector order, or ``(component, attr, theta_index)``
                with an explicit index into theta -- several entries may share
                one index (shared parameters).  ``None`` -> no estimated
                parameters.
            measurements: Measuring devices whose modelled ``measuredValue``
                the map must return (Estimator data-fit signals).
            outputs: List of ``(component, out_port)`` arbitrary outputs the
                map must return (Optimizer objective/constraint signals).
            step_size: Step size in seconds -- a scalar or the per-period
                list; all periods must share one step size.

        Returns:
            ``(layout, functional_model)``: the
            :class:`~twin4build.simulator._functional.StateLayout` of the
            stateful components and the
            :class:`~twin4build.simulator._functional.FunctionalModel`.
        """
        stateful = collect_stateful(self.model)
        if not stateful:
            raise RuntimeError("no stateful components")
        layout = StateLayout(stateful)
        steps = step_size if isinstance(step_size, (list, tuple)) else [step_size]
        steps = [int(s) for s in steps]
        if len(set(steps)) != 1:
            raise RuntimeError("mixed step sizes across periods")
        functional_model = FunctionalModel(
            self.model,
            layout.components,
            list(theta_spec or []),
            steps[0],
            measurements=measurements,
            outputs=outputs,
        )
        if functional_model.D != layout.width:
            raise RuntimeError("functional model state width mismatch")
        return layout, functional_model

    def record_exogenous_inputs(
        self,
        functional_model,
        start_time,
        end_time,
        step_size,
        layout=None,
        meas_ids=(),
    ):
        """Record the functional model's exogenous input tape.

        Args:
            functional_model: The model returned by
                :meth:`build_functional_model`.
            start_time: Per-period start times (list).
            end_time: Per-period end times (list).
            step_size: Per-period step sizes (list).
            layout: Optional ``StateLayout``; when given, per-period initial
                (augmented) states ``state0`` / ``Y0`` are also returned.
            meas_ids: Measuring-device ids whose ``measuredValue`` to sample.

        Returns:
            ``SimpleNamespace`` of per-period lists: ``state0``, ``Y0``,
            ``exogenous_tape``, ``feedback_tape``, ``measurement_tape``,
            ``n_timesteps``.
        """
        return record_exogenous_inputs(
            self,
            functional_model,
            start_time,
            end_time,
            step_size,
            layout=layout,
            meas_ids=meas_ids,
        )

    def rollout_functional(
        self,
        functional_model,
        y0,
        theta,
        exogenous_tape,
        *,
        transform_mode: bool = False,
    ) -> torch.Tensor:
        """Sequentially roll the functional model over one period.

        Args:
            functional_model: The model returned by
                :meth:`build_functional_model`.
            y0: ``(D_aug,)`` augmented initial state ``[state0 | FB[0]]``.
            theta: ``(n_theta,)`` physical parameters in theta_spec order.
            exogenous_tape: ``(n_t, n_exogenous)`` recorded inputs.

        Returns:
            ``(n_t, n_meas)`` modelled outputs; differentiable w.r.t.
            ``theta``, ``y0`` and ``exogenous_tape``.

        With ``compile_step`` active for ``theta``'s device the transform-mode
        step runs through :attr:`FunctionalModel.compiled_step`; the
        cache-using (``transform_mode=False``) rollout is never compiled.
        """
        step = None
        if (
            transform_mode
            and not _functorch_active()  # vmap/jacfwd over a compiled fn is unsupported
            and self.step_compilation_active(theta.device)
        ):
            step = functional_model.compiled_step
        return functional_rollout(
            functional_model,
            y0,
            theta,
            exogenous_tape,
            transform_mode=transform_mode,
            step=step,
        )

    def rollout_functional_batched(
        self, functional_model, Y0, Theta, exogenous_tape
    ) -> torch.Tensor:
        """Roll a batch of parameter vectors over one period (transform mode).

        ``Y0 (B, D_aug)``, ``Theta (B, n_theta)`` -> ``(B, n_t, n_meas)``.
        With ``compile_step`` active for ``Theta``'s device the compiled
        batched step (``compile(vmap(F_aug))``) is used; otherwise the scalar
        rollout is ``vmap``-ed.
        """
        step = None
        if not _functorch_active() and self.step_compilation_active(Theta.device):
            step = functional_model.compiled_batched_step
        return functional_rollout_batched(
            functional_model, Y0, Theta, exogenous_tape, step=step
        )

    def step_compilation_active(self, device) -> bool:
        """Whether functional rollouts on ``device`` use the compiled step.

        ``compile_step=True`` always (validated at construction), ``False``
        never, ``"auto"`` on CUDA devices when this torch has Triton (Linux
        wheels ship it; Windows wheels do not, and then the eager step is
        used exactly as before).
        """
        if self.compile_step is True:
            return True
        if self.compile_step is False:
            return False
        device = torch.device(device)
        return device.type == "cuda" and _has_triton()
