from __future__ import annotations
from typing import Union
from .system import System
from .connection import Connection
class ConnectionPoint:
    def __init__(self,
                connectionPointOf: Union[System, None] = None,
                connectsSystemThrough: Union[Connection, None] = None, 
                recieverPropertyName = None):
        assert isinstance(connectionPointOf, System) or connectionPointOf is None, "Attribute \"connectionPointOf\" is of type \"" + str(type(connectionPointOf)) + "\" but must be of type \"" + str(System) + "\""
        assert isinstance(connectsSystemThrough, Connection) or connectsSystemThrough is None, "Attribute \"connectsSystemAt\" is of type \"" + str(type(connectsSystemThrough)) + "\" but must be of type \"" + str(Connection) + "\""
        self.connectionPointOf = connectionPointOf
        self.connectsSystemThrough = connectsSystemThrough
        self.recieverPropertyName = recieverPropertyName ###