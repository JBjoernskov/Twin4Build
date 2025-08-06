from __future__ import annotations

# Standard library imports
import datetime
import math
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

# Third party imports
import numpy as np
import pandas as pd
import torch

# import george
# from george import kernels
from fmpy.fmi2 import FMICallException
from tqdm import tqdm

# Local application imports
import twin4build.core as core
import twin4build.systems as systems


class Simulator:
    r"""
    A simulator for building digital twins.

    This class simulates :class:`~twin4build.model.Model` or :class:`~twin4build.model.simulation_model.simulation_model.SimulationModel` in a time-stepping manner.
    It takes a prepared model with a predetermined execution order and runs the
    simulation by calling each component in sequence for each timestep.

    The simulator handles the coordination between components, ensuring that
    outputs from one component are properly passed as inputs to connected
    components during each simulation timestep.

    Mathematical Formulation:
    =========================

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

    Execution Sequence:
    -------------------

    The execution sequence is determined by the model preparation phase
    (see :class:`~twin4build.model.simulation_model.simulation_model.SimulationModel`):

    .. math::

        L = (c_1, c_2, ..., c_n)

    Time-Stepping Simulation:
    --------------------------

    For each timestep :math:`t \in (t_{start}, t_{start} + \Delta t, ..., t_{end})`,
    the simulator executes each component :math:`c_j` in the specified order :math:`L`.

    First, for component :math:`c_j`, collect inputs from all connected components:

    Component :math:`c_j` has input sequence :math:`\mathbf{x}_j = (x_{j,1}, x_{j,2}, ..., x_{j,n_j^{in}})`
    and output sequence :math:`\mathbf{y}_j = (y_{j,1}, y_{j,2}, ..., y_{j,n_j^{out}})` where :math:`n_j^{in}` and :math:`n_j^{out}`
    are the numbers of input and output ports respectively.

    For each edge :math:`e \in E` with :math:`\iota(e) = (c_i, c_j)`:

    .. math::

        x_{j,\alpha(e)} = y_{i,\beta(e)}

    where:

        - :math:`\alpha(e)` and :math:`\beta(e)` are the input and output ports for edge :math:`e`

    After collecting the inputs, execute the step function of the component:

    .. math::

        \mathbf{y}_{j,t} = f_j(\mathbf{x}_{j,t}, \mathbf{s}_{j,t}, t, \Delta t)

    where:

        - :math:`\mathbf{x}_{j,t}` is the input sequence for component :math:`j` at time :math:`t`
        - :math:`\mathbf{y}_{j,t}` is the output sequence from component :math:`j` at time :math:`t`
        - :math:`\mathbf{s}_{j,t}` is the internal state of component :math:`j` at time :math:`t`
        - :math:`f_j` is the component's dynamics function
        - :math:`\alpha(e)` and :math:`\beta(e)` define the specific input/output ports for edge :math:`e`

    Shorthand Notation:
    -------------------

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
    ...     startTime=start_time,
    ...     endTime=end_time,
    ...     stepSize=step_size
    ... )
    >>>
    >>> # Access simulation results
    >>> results = simulator.get_simulation_readings()
    """

    def __init__(self, model: core.Model):
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

    def _do_component_timestep(self, component: core.System) -> None:
        """
        Perform a single timestep for a component.

        Args:
            component (core.System): The component to simulate.

        Raises:
            AssertionError: If any input value is NaN.
        """
        # Gather all needed inputs for the component through all ingoing connections
        for connection_point in component.connects_at:
            for connection in connection_point.connects_system_through:
                connected_component = connection.connects_system

                component.input[connection_point.inputPort].set(
                    connected_component.output[connection.outputPort].get(),
                    stepIndex=self.stepIndex,
                )

                if torch.isnan(component.input[connection_point.inputPort].get()):
                    for s in self.debug_str:
                        print(s)
                    raise ValueError(
                        f"Input {connection_point.inputPort} of component {component.id} is NaN"
                    )

        component.do_step(
            self.secondTime,
            self.dateTime,
            self.stepSize,
            self.stepIndex,
        )

    def _do_system_time_step(self, model: core.Model) -> None:
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
            - Component execution order is determined by the model's execution_order attribute
            - Updates are propagated through the flat_execution_order after main execution
        """
        for component_group in model.execution_order:
            for component in component_group:
                self._do_component_timestep(component)

    def get_simulation_timesteps(
        self, startTime: datetime, endTime: datetime, stepSize: int
    ) -> None:
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
        n_timesteps = math.floor((endTime - startTime).total_seconds() / stepSize)
        self.secondTimeSteps = [i * stepSize for i in range(n_timesteps)]
        self.dateTimeSteps = [
            startTime + datetime.timedelta(seconds=i * stepSize)
            for i in range(n_timesteps)
        ]

    def simulate(
        self,
        startTime: datetime,
        endTime: datetime,
        stepSize: int,
        show_progress_bar: bool = True,
        debug: bool = False,
    ) -> None:
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
        self.debug_str = []  # TODO: remove this
        assert (
            startTime.tzinfo is not None
        ), "The argument startTime must have a timezone"
        assert endTime.tzinfo is not None, "The argument endTime must have a timezone"
        assert isinstance(stepSize, int), "The argument stepSize must be an integer"
        self.startTime = startTime
        self.endTime = endTime
        self.stepSize = stepSize
        self.debug = debug
        self.get_simulation_timesteps(startTime, endTime, stepSize)
        self.model.initialize(startTime, endTime, stepSize, self)
        if show_progress_bar:
            for self.stepIndex, (self.secondTime, self.dateTime) in tqdm(
                enumerate(zip(self.secondTimeSteps, self.dateTimeSteps)),
                total=len(self.dateTimeSteps),
            ):
                self._do_system_time_step(self.model)
        else:
            for self.stepIndex, (self.secondTime, self.dateTime) in enumerate(
                zip(self.secondTimeSteps, self.dateTimeSteps)
            ):
                self._do_system_time_step(self.model)
        if self.debug:
            for s in self.debug_str:
                print(s)

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
        sensor_instances = self.model.get_component_by_class(
            self.model.components, systems.SensorSystem
        )

        for sensor in sensor_instances:
            key = list(sensor.output.keys())[0]
            simulation_readings = sensor.output[key].history.detach()
            df_simulation_readings.insert(0, sensor.id, simulation_readings)

        return df_simulation_readings

    def get_actual_readings(
        self,
        startTime: datetime,
        endTime: datetime,
        stepSize: int,
        reading_type: str = "all",
    ) -> pd.DataFrame:
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
        assert (
            reading_type in allowed_reading_types
        ), f"The \"reading_type\" argument must be one of the following: {', '.join(allowed_reading_types)} - \"{reading_type}\" was provided."
        # print("Collecting actual readings...")
        """
        This is a temporary method for retrieving actual sensor readings.
        Currently it simply reads from csv files containing historic data.
        """
        self.get_simulation_timesteps(startTime, endTime, stepSize)
        df_actual_readings = pd.DataFrame()
        time = self.dateTimeSteps
        df_actual_readings.insert(0, "time", time)
        df_actual_readings = df_actual_readings.set_index("time")
        sensor_instances = self.model.get_component_by_class(
            self.model.components, systems.SensorSystem
        )

        for sensor in sensor_instances:
            sensor.initialize(startTime, endTime, stepSize, self)
            # sensor.set_is_physical_system()
            if sensor.physicalSystem is not None:
                if reading_type == "all":
                    actual_readings = sensor.get_physical_readings(
                        startTime, endTime, stepSize, self
                    )
                    df_actual_readings.insert(0, sensor.id, actual_readings)
                elif reading_type == "input" and sensor.is_leaf:
                    actual_readings = sensor.get_physical_readings(
                        startTime, endTime, stepSize, self
                    )
                    df_actual_readings.insert(0, sensor.id, actual_readings)

        return df_actual_readings
