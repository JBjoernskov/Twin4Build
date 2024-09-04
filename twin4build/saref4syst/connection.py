from __future__ import annotations
from typing import Union
import twin4build.saref4syst.system as system
import twin4build.saref4syst.connection_point as connection_point

class Connection:
    def __init__(self,
                connectsSystem: Union[system.System ,None]=None,
                connectsSystemAt: Union[connection_point.ConnectionPoint ,None]=None,
                senderPropertyName=None):
        assert isinstance(connectsSystem, system.System) or connectsSystem is None, "Attribute \"connectsSystem\" is of type \"" + str(type(connectsSystem)) + "\" but must be of type \"" + str(system.System) + "\""
        assert isinstance(connectsSystemAt, connection_point.ConnectionPoint) or connectsSystemAt is None, "Attribute \"connectsSystemAt\" is of type \"" + str(type(connectsSystemAt)) + "\" but must be of type \"" + str(connection_point.ConnectionPoint) + "\""
        # if connectsSystemAt is None:
        #     connectsSystemAt = []
        self.connectsSystem = connectsSystem
        self.connectsSystemAt = connectsSystemAt
        self.senderPropertyName = senderPropertyName
        
        
