from __future__ import annotations

# Standard library imports
import datetime
from typing import List, Union

# Third party imports
# from twin4build.utils.plot.simulation_result import SimulationResult
from prettytable import PrettyTable

# Local application imports
import twin4build.core as core
from twin4build.utils.rgetattr import rgetattr
from twin4build.utils.rhasattr import rhasattr

import torch
class System:
    """
    A base-class representing a component model used as part of a simulation model.
    The methods :func:`~twin4build.systems.saref4syst.system.System.initialize`, :func:`~twin4build.systems.saref4syst.system.System.do_step` must be implemented by the subclass.

    Args:
        connects_at: A list of connection points that the system connects to.
        connected_through: A list of systems that the system connects through.
        input: A dictionary of inputs to the system.
        output: A dictionary of outputs from the system.
        outputGradient: A dictionary of output gradients to the system.
        parameterGradient: A dictionary of parameter gradients to the system.
        id: The id of the system.
    """

    sp = None

    def __str__(self):
        t = PrettyTable(field_names=["input", "output"], divider=True)
        title = f"Component overview    id: {self.id}"
        t.title = title
        input_list = list(self.input.keys())
        output_list = list(self.output.keys())
        n = max(len(input_list), len(output_list))
        for j in range(n):
            i = input_list[j] if j < len(input_list) else ""
            o = output_list[j] if j < len(output_list) else ""
            t.add_row([i, o], divider=True if j == len(input_list) - 1 else False)

        return t.get_string()

    def __init__(
        self,
        connects_at: Union[list, None] = None,
        connected_through: Union[list, None] = None,
        input: Union[dict, None] = None,
        output: Union[dict, None] = None,
        id: Union[str, None] = None,
        **kwargs,
    ):
        """
        Initialize a System object.

        Args:
            connects_at: A list of connection points that the system connects to.
            connected_through: A list of systems that the system connects through.
            input: A dictionary of inputs to the system.
            output: A dictionary of outputs from the system.
            outputGradient: A dictionary of output gradients to the system.
            parameterGradient: A dictionary of parameter gradients to the system.
            id: The id of the system.
        """
        assert isinstance(connects_at, list) or connects_at is None, (
            'Attribute "connects_at" is of type "'
            + str(type(connects_at))
            + '" but must be of type "'
            + str(list)
            + '"'
        )
        assert isinstance(connected_through, list) or connected_through is None, (
            'Attribute "connected_through" is of type "'
            + str(type(connected_through))
            + '" but must be of type "'
            + str(list)
            + '"'
        )
        assert isinstance(input, dict) or input is None, (
            'Attribute "input" is of type "'
            + str(type(input))
            + '" but must be of type "'
            + str(dict)
            + '"'
        )
        assert isinstance(output, dict) or output is None, (
            'Attribute "output" is of type "'
            + str(type(output))
            + '" but must be of type "'
            + str(dict)
            + '"'
        )
        assert isinstance(id, str), (
            'Attribute "id" is of type "'
            + str(type(id))
            + '" but must be of type "'
            + str(str)
            + '"'
        )
        if connects_at is None:
            connects_at = []
        if connected_through is None:
            connected_through = []
        if input is None:
            input = {}
        if output is None:
            output = {}
        self._connects_at = connects_at
        self._connected_through = connected_through
        self._input = input
        self._output = output
        self._id = id
        self._n_c = 1  # Number of parallel components (for vectorization)

    @classmethod
    def add_signature_pattern(cls, signature_pattern: core.SignaturePattern) -> None:
        """
        Add a signature pattern to the system.
        """
        if cls.sp is None:
            cls.sp = []
        cls.sp.append(signature_pattern)

    @property
    def connects_at(self) -> list:
        """
        Get the connection points.
        """
        return self._connects_at

    @property
    def connected_through(self) -> list:
        """
        Get the connected systems through.
        """
        return self._connected_through

    @property
    def input(self) -> dict:
        """
        Get the input of the system.
        """
        return self._input

    @property
    def output(self) -> dict:
        """
        Get the output of the system.
        """
        return self._output

    @property
    def id(self) -> str:
        """
        Get the id of the system.
        """
        return self._id

    @input.setter
    def input(self, value: dict) -> None:
        """
        Set the input of the system.
        """
        self._input = value

    @output.setter
    def output(self, value: dict) -> None:
        """
        Set the output of the system.
        """
        self._output = value

    @connects_at.setter
    def connects_at(self, value: list) -> None:
        """
        Set the connection points.
        """
        self._connects_at = value

    @connected_through.setter
    def connected_through(self, value: list) -> None:
        """
        Set the connected systems through.
        """
        self._connected_through = value

    @id.setter
    def id(self, value: str) -> None:
        """
        Set the id of the system.
        """
        self._id = value

    @property
    def n_c(self) -> int:
        """
        Get the number of parallel components (for n_c vectorization).
        """
        return self._n_c

    @n_c.setter
    def n_c(self, value: int) -> None:
        """
        Set the number of parallel components (for n_c vectorization).
        """
        self._n_c = value

    def get_n_v_from_connections(self, input_port_name: str) -> int:
        """
        Determine the required n_v (vector size) for a given input port by examining
        connection points and finding the largest input_port_index.

        Args:
            input_port_name: Name of the input port to check.

        Returns:
            int: Required n_v (max index + 1), or 1 if no connections found.
        """
        max_index = 0
        found_connection = False

        for cp in self.connects_at:
            if cp.input_port == input_port_name:
                # Get all indices from the input_port_index dict
                for index in cp.input_port_index.values():
                    found_connection = True
                    if isinstance(index, int):
                        idx_val = index
                    elif hasattr(index, 'max'):
                        idx_val = int(index.max().item())
                    else:
                        idx_val = int(index)
                    max_index = max(max_index, idx_val)

        n_v =  max_index + 1
        if found_connection==False:
            n_v = None
        return n_v  # Default to None if no connections

    def initialize(
        self,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
        step_size: int,
    ) -> None:
        """
        Initialize the system.

        Args:
            start_time (datetime.datetime): The start time of the simulation.
            end_time (datetime.datetime): The end time of the simulation.
            step_size (int): The step size of the simulation in seconds.
            simulator (core.Simulator): The simulator.
        """
        _, _, max_timesteps, _ = core.Simulator.get_simulation_timesteps(
            start_time, end_time, step_size
        )
        batch_size = len(start_time)
        for input in self.input.values():
            input.initialize(n_t=max_timesteps, n_s=batch_size)
        for output in self.output.values():
            output.initialize(n_t=max_timesteps, n_s=batch_size)

    def do_step(
        self,
        second_time: List[float],
        date_time: List[datetime.datetime],
        step_size: List[int],
        step_index: int,
    ) -> None:
        """
        Do a single step of the system.

        Args:
            second_time (List[float]): The current time in seconds.
            date_time (List[datetime.datetime]): The current date and time.
            step_size (List[int]): The step size of the simulation in seconds.
            step_index (int): The current step index.
        """
        pass

    def populate_config(self) -> dict:
        """
        Populate the config of the system.

        Returns:
            dict: The config of the system.
        """

        def extract_value(value):
            if hasattr(value, "detach") and hasattr(value, "numpy"):
                # Use .tolist() to convert numpy types to Python native types
                # This ensures values like np.float64(1.0) become 1.0
                return value.get().detach().numpy().flatten().tolist()
            else:  # isinstance(value, (int, float, type(None))):
                return value

        d = {}
        for key, value in self.config.items():
            # Check if all keys in the value dict are valid attributes of the object
            cond = isinstance(value, dict) and all(
                [rhasattr(self, k) for k in value.keys()]
            )
            if cond:
                d[key] = self.populate_config(value)
            else:
                assert isinstance(
                    value, list
                ), f"Invalid config value type: {type(value)}. Must be a list of attributes."
                d[key] = {}
                for attr in value:
                    v = rgetattr(self, attr)
                    v = extract_value(v)
                    d[key][attr] = v
        return d
