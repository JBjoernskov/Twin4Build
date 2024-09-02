from __future__ import annotations
from typing import Union
import twin4build.saref4syst.system as system
import twin4build.saref4syst.connection as connection
from twin4build.logger.Logging import Logging
logger = Logging.get_logger("ai_logfile")

class ConnectionPoint:
    def __init__(self,
                connectionPointOf: Union[system.System, None]=None,
                connectsSystemThrough: Union[connection.Connection, None]=None, 
                receiverPropertyName=None):
        logger.info("[ConnectionPoint Class] : Entered in __init__ Function")
        assert isinstance(connectionPointOf, system.System) or connectionPointOf is None, "Attribute \"connectionPointOf\" is of type \"" + str(type(connectionPointOf)) + "\" but must be of type \"" + str(system.System) + "\""
        assert isinstance(connectsSystemThrough, connection.Connection) or connectsSystemThrough is None, "Attribute \"connectsSystemAt\" is of type \"" + str(type(connectsSystemThrough)) + "\" but must be of type \"" + str(connection.Connection) + "\""
        self.connectionPointOf = connectionPointOf
        self.connectsSystemThrough = connectsSystemThrough
        self.receiverPropertyName = receiverPropertyName
        logger.info("[ConnectionPoint Class] : Exited from __init__ Function")
        
        
