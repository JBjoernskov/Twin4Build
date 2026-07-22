from __future__ import annotations

# Standard library imports
from typing import Optional, Union

# Third party imports
import torch

# Local application imports
import twin4build.core as core


class ConnectionPoint:
    """
    A class representing a connection point of a system, i.e. an input to a system.

    Args:
        connection_point_of: The system that the connection point is part of. Defaults to None.
        connects_system_through: A list of Connection objects that the connection point
            receives from. Defaults to None (empty list).
        input_port: The name of the property that the connection point receives. Defaults to None.
    """

    def __init__(
        self,
        connection_point_of: Union[core.System, None] = None,
        connects_system_through: Union[list, None] = None,
        input_port: Optional[str] = None,
    ):
        """
        Initialize a ConnectionPoint object.

        Args:
            connection_point_of: The system that the connection point is part of. Defaults to None.
            connects_system_through: A list of Connection objects that the connection point
                receives from. Defaults to None (empty list).
            input_port: The name of the property that the connection point receives. Defaults to None.
        """
        assert (
            isinstance(connection_point_of, core.System) or connection_point_of is None
        ), (
            'Attribute "connection_point_of" is of type "'
            + str(type(connection_point_of))
            + '" but must be of type "'
            + str(core.System)
            + '"'
        )
        assert (
            isinstance(connects_system_through, list) or connects_system_through is None
        ), (
            'Attribute "connects_system_through" is of type "'
            + str(type(connects_system_through))
            + '" but must be of type "'
            + str(list)
            + '"'
        )
        assert isinstance(input_port, str) or input_port is None, (
            'Attribute "input_port" is of type "'
            + str(type(input_port))
            + '" but must be of type "'
            + str(str)
            + '"'
        )
        # Store attributes as private variables
        self._connectionPointOf = connection_point_of
        self._connectsSystemThrough = connects_system_through
        self._input_port = input_port

        if self._connectsSystemThrough is None:
            self._connectsSystemThrough = []

        self._input_port_index = {}
        self._output_port_index = {}
        self._input_component_index = {}
        self._output_component_index = {}

    @property
    def connection_point_of(self) -> Union[core.System, None]:
        """
        Get the system that the connection point is part of.
        """
        return self._connectionPointOf

    @connection_point_of.setter
    def connection_point_of(self, value: Union[core.System, None]) -> None:
        """
        Set the system that the connection point is part of.
        """
        self._connectionPointOf = value

    @property
    def connects_system_through(self) -> list:
        """
        Get the list of Connection objects that the connection point receives from.
        """
        return self._connectsSystemThrough

    @connects_system_through.setter
    def connects_system_through(self, value: list) -> None:
        """
        Set the list of Connection objects that the connection point receives from.
        """
        self._connectsSystemThrough = value

    @property
    def input_port(self) -> Optional[str]:
        """
        Get the name of the property that the connection point receives.
        """
        return self._input_port

    @input_port.setter
    def input_port(self, value: Optional[str]) -> None:
        """
        Set the name of the property that the connection point receives.
        """
        self._input_port = value

    @property
    def input_port_index(self) -> Union[int, torch.Tensor]:
        """
        Get the dict mapping each Connection to its input port index
        (an int or torch.Tensor).
        """
        return self._input_port_index

    @property
    def output_port_index(self) -> Union[int, torch.Tensor]:
        """
        Get the dict mapping each Connection to its output port index
        (an int or torch.Tensor).
        """
        return self._output_port_index

    def set_input_port_index(
        self, connection: core.Connection, index: [int, torch.Tensor]
    ) -> None:
        """
        Set the index of the input port.
        """
        self._input_port_index[connection] = index

    def set_output_port_index(
        self, connection: core.Connection, index: [int, torch.Tensor]
    ) -> None:
        """
        Set the index of the output port.
        """
        self._output_port_index[connection] = index

    @property
    def input_component_index(self) -> dict:
        """
        Get the component index dict for the input side (i_c dimension).
        """
        return self._input_component_index

    @property
    def output_component_index(self) -> dict:
        """
        Get the component index dict for the output side (i_c dimension).
        """
        return self._output_component_index

    def set_input_component_index(
        self, connection: core.Connection, index: [int, torch.Tensor]
    ) -> None:
        """
        Set the component index on the input side (i_c dimension).
        """
        self._input_component_index[connection] = index

    def set_output_component_index(
        self, connection: core.Connection, index: [int, torch.Tensor]
    ) -> None:
        """
        Set the component index on the output side (i_c dimension).
        """
        self._output_component_index[connection] = index
