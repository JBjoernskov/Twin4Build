from __future__ import annotations

# Standard library imports
from typing import Optional, Union

# Local application imports
import twin4build.core as core


class ConnectionPoint:
    """
    A class representing a connection point of a system, i.e. an input to a system.
    """

    def __init__(
        self,
        connection_point_of: Union[core.System, None] = None,
        connects_system_through: Union[list, None] = None,
        inputPort: Optional[str] = None,
    ):
        """
        Initialize a ConnectionPoint object.

        Args:
            connection_point_of (System, optional): The system that the connection point is part of. Defaults to None.
            connects_system_through (list, optional): A list of systems that the connection point connects to. Defaults to None.
            inputPort (str, optional): The name of the property that the connection point receives. Defaults to None.
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
        assert isinstance(inputPort, str) or inputPort is None, (
            'Attribute "inputPort" is of type "'
            + str(type(inputPort))
            + '" but must be of type "'
            + str(str)
            + '"'
        )
        # Store attributes as private variables
        self._connectionPointOf = connection_point_of
        self._connectsSystemThrough = connects_system_through
        self._inputPort = inputPort

        if self._connectsSystemThrough is None:
            self._connectsSystemThrough = []

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
        Get the list of systems that the connection point connects to.
        """
        return self._connectsSystemThrough

    @connects_system_through.setter
    def connects_system_through(self, value: list) -> None:
        """
        Set the list of systems that the connection point connects to.
        """
        self._connectsSystemThrough = value

    @property
    def inputPort(self) -> Optional[str]:
        """
        Get the name of the property that the connection point receives.
        """
        return self._inputPort

    @inputPort.setter
    def inputPort(self, value: Optional[str]) -> None:
        """
        Set the name of the property that the connection point receives.
        """
        self._inputPort = value
