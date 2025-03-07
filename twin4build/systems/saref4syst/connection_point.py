from __future__ import annotations
from typing import Union, Optional
import twin4build.core as core
class ConnectionPoint:
    """
    A class representing a connection point of a system, i.e. an input to a system.

    Attributes:
        connectionPointOf (System): The system that the connection point is part of.
        connectsSystemThrough (list): A list of systems that the connection point connects to.
        receiverPropertyName (str): The name of the property that the connection point receives.
    """
    def __init__(self,
                connectionPointOf: Union[core.System, None]=None,
                connectsSystemThrough: Union[list, None]=None, 
                receiverPropertyName: Optional[str] = None):
        """
        Initialize a ConnectionPoint object.

        Args:
            connectionPointOf (System, optional): The system that the connection point is part of. Defaults to None.
            connectsSystemThrough (list, optional): A list of systems that the connection point connects to. Defaults to None.
            receiverPropertyName (str, optional): The name of the property that the connection point receives. Defaults to None.
        """
        assert isinstance(connectionPointOf, core.System) or connectionPointOf is None, "Attribute \"connectionPointOf\" is of type \"" + str(type(connectionPointOf)) + "\" but must be of type \"" + str(core.System) + "\""
        assert isinstance(connectsSystemThrough, list) or connectsSystemThrough is None, "Attribute \"connectsSystemThrough\" is of type \"" + str(type(connectsSystemThrough)) + "\" but must be of type \"" + str(list) + "\""
        assert isinstance(receiverPropertyName, str) or receiverPropertyName is None, "Attribute \"receiverPropertyName\" is of type \"" + str(type(receiverPropertyName)) + "\" but must be of type \"" + str(str) + "\""
        self.connectionPointOf = connectionPointOf
        self.connectsSystemThrough = connectsSystemThrough
        self.receiverPropertyName = receiverPropertyName

        if self.connectsSystemThrough is None:
            self.connectsSystemThrough = []
        
        
