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
    A class for simulating models in the twin4build framework.

    This class provides methods to simulate models, perform Bayesian inference,
    and retrieve simulation results. It handles both standard simulations and
    Gaussian process-based inference.

    Mathematical Formulation:

    1. Time Step Calculation:

       .. math::

          n_{timesteps} = \left\lfloor \frac{t_{end} - t_{start}}{\Delta t} \right\rfloor

       where:
          :math:`n_{timesteps}` is the number of simulation timesteps
          :math:`t_{start}` is the simulation start time
          :math:`t_{end}` is the simulation end time
          :math:`\Delta t` is the timestep size

    2. Component State Update:

       For each component :math:`c` at timestep :math:`k`:

       .. math::

          \mathbf{x}_c(k+1) = f_c(\mathbf{x}_c(k), \mathbf{u}_c(k), \Delta t)

       .. math::

          \mathbf{y}_c(k) = g_c(\mathbf{x}_c(k), \mathbf{u}_c(k))

       where:
          :math:`\mathbf{x}_c` is the component state vector
          :math:`\mathbf{u}_c` is the component input vector
          :math:`\mathbf{y}_c` is the component output vector
          :math:`f_c` is the state update function
          :math:`g_c` is the output function
          :math:`\Delta t` is the timestep size

    3. Simulation Progression:

       For each timestep :math:`k` from 0 to :math:`n_{timesteps}-1`:

       .. math::

          t_k = t_{start} + k \cdot \Delta t

       where:
          :math:`t_k` is the simulation time at step :math:`k`

    4. Component Execution Order:

       Components are executed in topological order based on their dependencies:

       .. math::

          \mathcal{O} = \{C_1, C_2, ..., C_n\}

       where:
          :math:`\mathcal{O}` is the ordered set of components
          :math:`C_i` is a component or group of independent components
          Components in the same group can be executed in parallel

    Attributes:
       model (Model): The model to be simulated.
       secondTime (float): Current simulation time in seconds.
       dateTime (datetime): Current simulation datetime.
       stepSize (int): Simulation step size in seconds.
       startTime (datetime): Simulation start time.
       endTime (datetime): Simulation end time.
       secondTimeSteps (List[float]): List of simulation timesteps in seconds.
       dateTimeSteps (List[datetime]): List of simulation timesteps as datetime objects.
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
        for connection_point in component.connectsAt:
            for connection in connection_point.connectsSystemThrough:
                connected_component = connection.connectsSystem
                # self.debug_str.append(f"Component {component.id} input {connection_point.inputPort} set to {connected_component.output[connection.outputPort].get()}")
                # print(f"Component {component.id} input {connection_point.inputPort} set to {connected_component.output[connection.outputPort].get()}")

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
            secondTime=self.secondTime,
            dateTime=self.dateTime,
            stepSize=self.stepSize,
            stepIndex=self.stepIndex,
        )

        # print("--------------------------------")
        # print(component.id)
        # for k, v in component.output.items():
        #     print(f"{k}: {v.get()}, requires_grad: {v.get().requires_grad}")
        #     if v._do_normalization and v.is_leaf:
        #         print(f"{k}: {v._normalized_history.requires_grad}")

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
        self.model.initialize(
            startTime=startTime, endTime=endTime, stepSize=stepSize, simulator=self
        )
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
            sensor.initialize(startTime, endTime, stepSize, simulator=self)
            # sensor.set_is_physical_system()
            if sensor.physicalSystem is not None:
                if reading_type == "all":
                    actual_readings = sensor.get_physical_readings(
                        startTime, endTime, stepSize
                    )
                    df_actual_readings.insert(0, sensor.id, actual_readings)
                elif reading_type == "input" and sensor.is_leaf:
                    actual_readings = sensor.get_physical_readings(
                        startTime, endTime, stepSize
                    )
                    df_actual_readings.insert(0, sensor.id, actual_readings)

        return df_actual_readings
