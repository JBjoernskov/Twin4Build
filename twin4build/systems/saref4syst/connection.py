from __future__ import annotations

# Standard library imports
from typing import Optional, Union

# Local application imports
import twin4build.core as core


class Connection:
    """
    A class representing a connection of a system, i.e. an output of a system.
    """

    def __init__(
        self,
        connects_system: Union[core.System, None] = None,
        connects_system_at: Union[list, None] = None,
        outputPort: Optional[str] = None,
    ):
        """
        Initialize a Connection object.

        Args:
            connects_system (System, optional): The system that the connection is part of. Defaults to None.
            connects_system_at (ConnectionPoint, optional): The connection point that the connection is part of. Defaults to None.
            outputPort (str, optional): The name of the property that the connection sends. Defaults to None.
        """

        assert isinstance(connects_system, core.System) or connects_system is None, (
            'Attribute "connects_system" is of type "'
            + str(type(connects_system))
            + '" but must be of type "'
            + str(core.System)
            + '"'
        )
        assert isinstance(connects_system_at, list) or connects_system_at is None, (
            'Attribute "connects_system_at" is of type "'
            + str(type(connects_system_at))
            + '" but must be of type "'
            + str(list)
            + '"'
        )
        assert isinstance(outputPort, str) or outputPort is None, (
            'Attribute "outputPort" is of type "'
            + str(type(outputPort))
            + '" but must be of type "'
            + str(str)
            + '"'
        )
        # if connects_system_at is None:
        #     connects_system_at = []
        self.connects_system = connects_system
        self.connects_system_at = connects_system_at
        self.outputPort = outputPort

        if self.connects_system_at is None:
            self.connects_system_at = []
