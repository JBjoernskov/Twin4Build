from __future__ import annotations

# Standard library imports
import datetime
import math
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
from twin4build.utils.logger import LOGGER
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

    _EXECUTION_MODES = ("object_graph", "composed")

    def __init__(self, model: core.Model, execution_mode: str = "object_graph"):
        """
        Initialize the Simulator instance.

        Creates a new simulator object that can be used to run simulations
        and perform parameter estimation or optimization.

        Args:
            model: The model to be simulated.
            execution_mode: Default execution engine. ``"object_graph"`` is
                the general port/history engine. ``"composed"`` enables the
                pure composed-map engine for compatible differentiable
                workflows such as estimation; :meth:`simulate` may override
                it per call.

        Notes:
            The simulator maintains internal state about the current simulation,
            including time steps and component states.
        """
        if execution_mode not in self._EXECUTION_MODES:
            raise ValueError(
                f"execution_mode must be one of {self._EXECUTION_MODES}; "
                f"got {execution_mode!r}"
            )
        self.model = model
        self.execution_mode = execution_mode

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

        Raises:
            ValueError: If any input value is NaN.
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
                component.input[connection_point.input_port]._set(
                    connected_component.output[connection.output_port].get(
                        i_v=output_port_index,
                        i_c=output_component_index,
                    ),
                    i_t=step_index,
                    i_v=input_port_index,
                    i_c=input_component_index,
                )

                # Actually, we HAVE to check for nans because it breaks jacobian calculation in optimizer will include nans which breaks scipy solver.
                if torch.any(
                    torch.isnan(component.input[connection_point.input_port].get())
                ):
                    LOGGER.debug(
                        "Component input: %s",
                        component.input[connection_point.input_port].get(),
                    )
                    raise ValueError(
                        f"Input {connection_point.input_port} of component {component.id} is NaN"
                    )

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
        if isinstance(start_time, datetime.datetime):
            start_time = [start_time]
        if isinstance(end_time, datetime.datetime):
            end_time = [end_time]
        if isinstance(step_size, int):
            step_size = [step_size]
        second_time_steps = []
        date_time_steps = []
        n_timesteps = []
        for start_time_, end_time_, step_size_ in zip(start_time, end_time, step_size):
            n_steps = math.floor((end_time_ - start_time_).total_seconds() / step_size_)
            second_time_steps.append([i * step_size_ for i in range(n_steps)])
            date_time_steps.append(
                [
                    start_time_ + datetime.timedelta(seconds=i * step_size_)
                    for i in range(n_steps)
                ]
            )
            n_timesteps.append(n_steps)
        max_timesteps = max(
            [len(second_time_steps_) for second_time_steps_ in second_time_steps]
        )
        second_time_steps = [
            second_time_steps_ + [np.nan] * (max_timesteps - len(second_time_steps_))
            for second_time_steps_ in second_time_steps
        ]
        date_time_steps = [
            date_time_steps_ + [np.nan] * (max_timesteps - len(date_time_steps_))
            for date_time_steps_ in date_time_steps
        ]

        second_time_steps = np.array(second_time_steps)
        date_time_steps = np.array(date_time_steps)
        return second_time_steps, date_time_steps, max_timesteps, n_timesteps

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
            execution_mode: Optional per-call override of the mode selected at
                construction. The composed mode preserves the normal history
                contract; its reusable tensor rollout is exposed through the
                composed-map methods below.

        Raises:
            AssertionError: If input parameters are invalid or missing timezone info.
            FMICallException: If the FMU simulation fails.
        """
        mode = self.execution_mode if execution_mode is None else execution_mode
        if mode not in self._EXECUTION_MODES:
            raise ValueError(
                f"execution_mode must be one of {self._EXECUTION_MODES}; "
                f"got {mode!r}"
            )
        # Public simulation must populate every component port/history,
        # including non-composable data/FMUs. The object-graph traversal is the
        # materialization pass for that contract. In composed mode its result
        # also provides the reference capture reused by pure tensor rollouts;
        # Estimator/Optimizer therefore avoid this traversal on subsequent
        # objective evaluations.
        self._last_execution_mode = mode

        for legacy_key, new_key in (
            ("startTime", "start_time"),
            ("endTime", "end_time"),
            ("stepSize", "step_size"),
        ):
            if legacy_key in kwargs:
                raise TypeError(
                    f"`{legacy_key}` has been removed. Use `{new_key}` instead."
                )

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
        self.model.initialize(start_time, end_time, step_size)
        second_time_steps, date_time_steps, max_timesteps, _ = (
            Simulator.get_simulation_timesteps(start_time, end_time, step_size)
        )
        self.second_time_steps = second_time_steps
        self.date_time_steps = date_time_steps
        self.n_timesteps = max_timesteps
        self.model.initialize(start_time, end_time, step_size)
        # Optional hook fired after (re)initialization, before the time loop.
        # Used by multiple-shooting / collocation estimation to overwrite each
        # segment's initial state with the optimizer's boundary decision
        # variables -- ``model.initialize`` above has just reset every stateful
        # component to its default/output-derived state, so this is the point
        # to inject the per-segment states.  Default ``None`` => no-op, so
        # ordinary simulation is unaffected.
        if after_initialize is not None:
            after_initialize()
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

    # -- composed-map simulation (fast paths) ---------------------------------
    # The model can also be simulated as a pure sequential rollout of ONE
    # composed one-step map (every composable component's ``do_step`` delegates
    # to a pure ``forward``; see twin4build/simulator/_composed.py).  The
    # Estimator's fast single-shooting and collocation transcriptions and the
    # Optimizer's fast control objective are all built on these three methods.

    def compose(
        self,
        theta_spec=None,
        measurements=None,
        outputs=None,
        step_size=None,
    ):
        """Build the pure one-step map for the current model.

        Runs the shared structural checks and returns ``(layout, composer)``.
        Raises ``RuntimeError`` if the model cannot be expressed as a composed
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
            ``(layout, composer)``: the
            :class:`~twin4build.simulator._composed.StateLayout` of the
            stateful components and the
            :class:`~twin4build.simulator._composed.OneStepComposer`.
        """
        from twin4build.simulator._composed import (
            OneStepComposer,
            StateLayout,
            collect_stateful,
        )

        stateful = collect_stateful(self.model)
        if not stateful:
            raise RuntimeError("no stateful components")
        layout = StateLayout(stateful)
        if any(getattr(c, "n_c", 1) != 1 for c in layout.components):
            raise RuntimeError("n_c > 1 states")
        steps = step_size if isinstance(step_size, (list, tuple)) else [step_size]
        steps = [int(s) for s in steps]
        if len(set(steps)) != 1:
            raise RuntimeError("mixed step sizes across periods")
        composer = OneStepComposer(
            self.model,
            layout.components,
            list(theta_spec or []),
            steps[0],
            measurements=measurements,
            outputs=outputs,
        )
        if composer.D != layout.width:
            raise RuntimeError("composer state width mismatch")
        return layout, composer

    def capture_rollout(
        self, composer, start_time, end_time, step_size, layout=None, meas_ids=()
    ):
        """One batched reference ``do_step`` rollout capturing the composed
        map's frozen inputs (see
        :func:`~twin4build.simulator._composed.capture_reference_rollout`).

        Args:
            composer: The composer returned by :meth:`compose`.
            start_time: Per-period start times (list).
            end_time: Per-period end times (list).
            step_size: Per-period step sizes (list).
            layout: Optional ``StateLayout``; when given, per-period initial
                (augmented) states ``state0`` / ``Y0`` are also returned.
            meas_ids: Measuring-device ids whose ``measuredValue`` to sample.

        Returns:
            ``SimpleNamespace`` of per-period lists: ``state0``, ``Y0``,
            ``CAP``, ``FB``, ``MEAS``, ``n_t``.
        """
        from twin4build.simulator._composed import capture_reference_rollout

        return capture_reference_rollout(
            self, composer, start_time, end_time, step_size,
            layout=layout, meas_ids=meas_ids,
        )

    def rollout_composed(
        self, composer, y0, theta, cap, *, transform_mode: bool = False
    ) -> torch.Tensor:
        """Sequentially roll the composed map over one period (see
        :func:`~twin4build.simulator._composed.sequential_rollout`).

        Args:
            composer: The composer returned by :meth:`compose`.
            y0: ``(D_aug,)`` augmented initial state ``[state0 | FB[0]]``.
            theta: ``(n_theta,)`` physical parameters in theta_spec order.
            cap: ``(n_t, n_captured)`` captured inputs for the period.

        Returns:
            ``(n_t, n_meas)`` modelled outputs; differentiable w.r.t.
            ``theta``, ``y0`` and ``cap``.
        """
        from twin4build.simulator._composed import sequential_rollout

        return sequential_rollout(
            composer, y0, theta, cap, transform_mode=transform_mode
        )
