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
        connectsSystem: Union[core.System, None] = None,
        connectsSystemAt: Union[list, None] = None,
        outputPort: Optional[str] = None,
    ):
        """
        Initialize a Connection object.

        Args:
            connectsSystem (System, optional): The system that the connection is part of. Defaults to None.
            connectsSystemAt (ConnectionPoint, optional): The connection point that the connection is part of. Defaults to None.
            outputPort (str, optional): The name of the property that the connection sends. Defaults to None.
        """

        assert isinstance(connectsSystem, core.System) or connectsSystem is None, (
            'Attribute "connectsSystem" is of type "'
            + str(type(connectsSystem))
            + '" but must be of type "'
            + str(core.System)
            + '"'
        )
        assert isinstance(connectsSystemAt, list) or connectsSystemAt is None, (
            'Attribute "connectsSystemAt" is of type "'
            + str(type(connectsSystemAt))
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
        # if connectsSystemAt is None:
        #     connectsSystemAt = []
        self.connectsSystem = connectsSystem
        self.connectsSystemAt = connectsSystemAt
        self.outputPort = outputPort

        if self.connectsSystemAt is None:
            self.connectsSystemAt = []
