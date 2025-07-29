from __future__ import annotations

# Standard library imports
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
    A class representing a system.
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
        connectedTo: Union[list, None] = None,
        hasSubSystem: Union[list, None] = None,
        subSystemOf: Union[list, None] = None,
        connectsAt: Union[list, None] = None,
        connectedThrough: Union[list, None] = None,
        input: Union[dict, None] = None,
        output: Union[dict, None] = None,
        id: Union[str, None] = None,
        **kwargs,
    ):
        """
        Initialize a System object.

        Args:
            connectedTo (list, optional): A list of systems that the system is connected to. Defaults to None.
            hasSubSystem (list, optional): A list of systems that the system has as a sub system. Defaults to None.
            subSystemOf (list, optional): A list of systems that the system is a sub system of. Defaults to None.
            connectsAt (list, optional): A list of connection points that the system connects to. Defaults to None.
            connectedThrough (list, optional): A list of systems that the system connects through. Defaults to None.
            input (dict, optional): A dictionary of inputs to the system. Defaults to None.
            output (dict, optional): A dictionary of outputs from the system. Defaults to None.
            outputGradient (dict, optional): A dictionary of output gradients to the system. Defaults to None.
            parameterGradient (dict, optional): A dictionary of parameter gradients to the system. Defaults to None.
            id (str, optional): The id of the system. Defaults to None.
        """
        # super().__init__(**kwargs)
        assert isinstance(connectedTo, list) or connectedTo is None, (
            'Attribute "connectedTo" is of type "'
            + str(type(connectedTo))
            + '" but must be of type "'
            + str(list)
            + '"'
        )
        assert isinstance(hasSubSystem, list) or hasSubSystem is None, (
            'Attribute "hasSubSystem" is of type "'
            + str(type(hasSubSystem))
            + '" but must be of type "'
            + str(list)
            + '"'
        )
        assert isinstance(subSystemOf, list) or subSystemOf is None, (
            'Attribute "subSystemOf" is of type "'
            + str(type(subSystemOf))
            + '" but must be of type "'
            + str(list)
            + '"'
        )
        assert isinstance(connectsAt, list) or connectsAt is None, (
            'Attribute "connectsAt" is of type "'
            + str(type(connectsAt))
            + '" but must be of type "'
            + str(list)
            + '"'
        )
        assert isinstance(connectedThrough, list) or connectedThrough is None, (
            'Attribute "connectedThrough" is of type "'
            + str(type(connectedThrough))
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
        if connectedTo is None:
            connectedTo = []
        if hasSubSystem is None:
            hasSubSystem = []
        if subSystemOf is None:
            subSystemOf = []
        if connectsAt is None:
            connectsAt = []
        if connectedThrough is None:
            connectedThrough = []
        if input is None:
            input = {}
        if output is None:
            output = {}
        self.connectedTo = connectedTo
        self.hasSubSystem = hasSubSystem
        self.subSystemOf = subSystemOf
        self.connectsAt = connectsAt
        self.connectedThrough = connectedThrough
        self.input = input
        self.output = output
        self.id = id

    def populate_config(self) -> dict:
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
