from __future__ import annotations
from typing import Union
from .system import System
from .connection import Connection

from twin4build.logger.Logging import Logging

logger = Logging.get_logger("ai_logile")

class ConnectionPoint:
    def __init__(self,
                connectionPointOf: Union[System, None]=None,
                connectsSystemThrough: Union[Connection, None]=None, 
                recieverPropertyName=None):
        
        logger.info("[Connection Point Class] : Entered in Initialise Function")

        assert isinstance(connectionPointOf, System) or connectionPointOf is None, "Attribute \"connectionPointOf\" is of type \"" + str(type(connectionPointOf)) + "\" but must be of type \"" + str(System) + "\""
        assert isinstance(connectsSystemThrough, Connection) or connectsSystemThrough is None, "Attribute \"connectsSystemAt\" is of type \"" + str(type(connectsSystemThrough)) + "\" but must be of type \"" + str(Connection) + "\""
        self.connectionPointOf = connectionPointOf
        self.connectsSystemThrough = connectsSystemThrough
        self.recieverPropertyName = recieverPropertyName

        
        logger.info("[Connection Point Class] : Exited from Initialise Function")
