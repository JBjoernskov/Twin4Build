from __future__ import annotations
from typing import Union, Optional
import twin4build.core as core
class Connection:
    """
    A class representing a connection of a system, i.e. an output of a system.
    """
    def __init__(self,
                connectsSystem: Union[core.System ,None]=None,
                connectsSystemAt: Union[list ,None]=None,
                senderPropertyName: Optional[str] = None):
        """
        Initialize a Connection object.

        Args:
            connectsSystem (System, optional): The system that the connection is part of. Defaults to None.
            connectsSystemAt (ConnectionPoint, optional): The connection point that the connection is part of. Defaults to None.
            senderPropertyName (str, optional): The name of the property that the connection sends. Defaults to None.
        """
        
        assert isinstance(connectsSystem, core.System) or connectsSystem is None, "Attribute \"connectsSystem\" is of type \"" + str(type(connectsSystem)) + "\" but must be of type \"" + str(core.System) + "\""
        assert isinstance(connectsSystemAt, list) or connectsSystemAt is None, "Attribute \"connectsSystemAt\" is of type \"" + str(type(connectsSystemAt)) + "\" but must be of type \"" + str(list) + "\""
        assert isinstance(senderPropertyName, str) or senderPropertyName is None, "Attribute \"senderPropertyName\" is of type \"" + str(type(senderPropertyName)) + "\" but must be of type \"" + str(str) + "\""
        # if connectsSystemAt is None:
        #     connectsSystemAt = []
        self.connectsSystem = connectsSystem
        self.connectsSystemAt = connectsSystemAt
        self.senderPropertyName = senderPropertyName

        if self.connectsSystemAt is None:
            self.connectsSystemAt = []
        
        
