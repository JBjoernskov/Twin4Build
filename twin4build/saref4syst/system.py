from __future__ import annotations
from typing import Union
from twin4build.utils.plot.report import SimulationResult
import itertools

from twin4build.logger.Logging import Logging

logger = Logging.get_logger("ai_logfile")

class System(SimulationResult):
    # id_iter = itertools.count()
    def __init__(self,
                connectedTo: Union[list, None]=None,
                hasSubSystem: Union[list, None]=None,
                subSystemOf: Union[list, None]=None,
                connectsAt: Union[list, None]=None,
                connectedThrough: Union[list, None]=None, 
                input: Union[dict, None]=None,
                output: Union[dict, None]=None,
                id: Union[str, None]=None,
                **kwargs):
        
        logger.info("[System Class] : Entered in Intialise Function")

        super().__init__(**kwargs)
        assert isinstance(connectedTo, list) or connectedTo is None, "Attribute \"connectedTo\" is of type \"" + str(type(connectedTo)) + "\" but must be of type \"" + str(list) + "\""
        assert isinstance(hasSubSystem, list) or hasSubSystem is None, "Attribute \"hasSubSystem\" is of type \"" + str(type(hasSubSystem)) + "\" but must be of type \"" + str(list) + "\""
        assert isinstance(subSystemOf, list) or subSystemOf is None, "Attribute \"subSystemOf\" is of type \"" + str(type(subSystemOf)) + "\" but must be of type \"" + str(list) + "\""
        assert isinstance(connectsAt, list) or connectsAt is None, "Attribute \"connectsAt\" is of type \"" + str(type(connectsAt)) + "\" but must be of type \"" + str(list) + "\""
        assert isinstance(connectedThrough, list) or connectedThrough is None, "Attribute \"connectedThrough\" is of type \"" + str(type(connectedThrough)) + "\" but must be of type \"" + str(list) + "\""
        assert isinstance(input, dict) or input is None, "Attribute \"input\" is of type \"" + str(type(input)) + "\" but must be of type \"" + str(dict) + "\""
        assert isinstance(output, dict) or output is None, "Attribute \"output\" is of type \"" + str(type(output)) + "\" but must be of type \"" + str(dict) + "\""
        assert isinstance(id, str), "Attribute \"id\" is of type \"" + str(type(id)) + "\" but must be of type \"" + str(str) + "\""
        if connectedTo is None:
            connectedTo = []
        if hasSubSystem is None:
            hasSubSystem = []
        if subSystemOf is None:
            subSystemOf = []
        if connectsAt is None:
            connectsAt = []
        if connectedThrough is None:
            connectedThrough = []
        if input is None:
            input = {}
        if output is None:
            output = {}
        self.connectedTo = connectedTo
        self.hasSubSystem = hasSubSystem
        self.subSystemOf = subSystemOf
        self.connectsAt = connectsAt
        self.connectedThrough = connectedThrough
        self.input = input 
        self.output = output
        self.id = id

        
        logger.info("[System Class] : Exited from Intialise Function")
