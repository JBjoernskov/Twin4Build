from __future__ import annotations

# Standard library imports
import datetime
from typing import Union

# Third party imports
# from twin4build.utils.plot.simulation_result import SimulationResult
from prettytable import PrettyTable

# Local application imports
import twin4build.core as core
from twin4build.utils.rgetattr import rgetattr
from twin4build.utils.rhasattr import rhasattr


class System:
    """
    A base-class representing a component model used as part of a simulation model.
    The methods :func:`~twin4build.systems.saref4syst.system.System.initialize`, :func:`~twin4build.systems.saref4syst.system.System.do_step` must be implemented by the subclass.
    """

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
            connects_at (list, optional): A list of connection points that the system connects to. Defaults to None.
            connected_through (list, optional): A list of systems that the system connects through. Defaults to None.
            input (dict, optional): A dictionary of inputs to the system. Defaults to None.
            output (dict, optional): A dictionary of outputs from the system. Defaults to None.
            outputGradient (dict, optional): A dictionary of output gradients to the system. Defaults to None.
            parameterGradient (dict, optional): A dictionary of parameter gradients to the system. Defaults to None.
            id (str, optional): The id of the system. Defaults to None.
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

    def initialize(
        self,
        startTime: datetime.datetime,
        endTime: datetime.datetime,
        stepSize: int,
        simulator: core.Simulator,
    ) -> None:
        """
        Initialize the system.

        Args:
            startTime (datetime.datetime): The start time of the simulation.
            endTime (datetime.datetime): The end time of the simulation.
            stepSize (int): The step size of the simulation in seconds.
            simulator (core.Simulator): The simulator.
        """
        pass

    def do_step(
        self,
        secondTime: float,
        dateTime: datetime.datetime,
        stepSize: int,
        stepIndex: int,
    ) -> None:
        """
        Do a single step of the system.

        Args:
            secondTime (float): The current time in seconds.
            dateTime (datetime.datetime): The current date and time.
            stepSize (int): The step size of the simulation in seconds.
            stepIndex (int): The current step index.
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
                return float(value.get().detach().numpy())
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
