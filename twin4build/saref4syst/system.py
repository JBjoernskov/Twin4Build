from __future__ import annotations
from typing import Union
from twin4build.utils.plot.simulation_result import SimulationResult
import itertools
from twin4build.logger.Logging import Logging
# from enum import 
logger = Logging.get_logger("ai_logfile")

class System(SimulationResult):
    # id_iter = itertools.count()
    def __init__(self,
                connectedTo: Union[list, None]=None,
                connectedBefore: Union[list, None]=None, #Assymetric subproperty of connectedTo
                connectedAfter: Union[list, None]=None, #Assymetric subproperty of connectedTo
                hasSubSystem: Union[list, None]=None,
                subSystemOf: Union[list, None]=None,
                connectsAt: Union[list, None]=None,
                connectedThrough: Union[list, None]=None, 
                input: Union[dict, None]=None,
                output: Union[dict, None]=None,
                outputGradient: Union[dict, None]=None,
                parameterGradient: Union[dict, None]=None,
                id: Union[str, None]=None,
                **kwargs):
        logger.info("[System Class] : Entered in __init__ Function")
        super().__init__(**kwargs)
        assert isinstance(connectedTo, list) or connectedTo is None, "Attribute \"connectedTo\" is of type \"" + str(type(connectedTo)) + "\" but must be of type \"" + str(list) + "\""
        assert isinstance(connectedBefore, list) or connectedBefore is None, "Attribute \"connectedBefore\" is of type \"" + str(type(connectedBefore)) + "\" but must be of type \"" + str(list) + "\""
        assert isinstance(connectedAfter, list) or connectedAfter is None, "Attribute \"connectedAfter\" is of type \"" + str(type(connectedAfter)) + "\" but must be of type \"" + str(list) + "\""
        assert isinstance(hasSubSystem, list) or hasSubSystem is None, "Attribute \"hasSubSystem\" is of type \"" + str(type(hasSubSystem)) + "\" but must be of type \"" + str(list) + "\""
        assert isinstance(subSystemOf, list) or subSystemOf is None, "Attribute \"subSystemOf\" is of type \"" + str(type(subSystemOf)) + "\" but must be of type \"" + str(list) + "\""
        assert isinstance(connectsAt, list) or connectsAt is None, "Attribute \"connectsAt\" is of type \"" + str(type(connectsAt)) + "\" but must be of type \"" + str(list) + "\""
        assert isinstance(connectedThrough, list) or connectedThrough is None, "Attribute \"connectedThrough\" is of type \"" + str(type(connectedThrough)) + "\" but must be of type \"" + str(list) + "\""
        assert isinstance(input, dict) or input is None, "Attribute \"input\" is of type \"" + str(type(input)) + "\" but must be of type \"" + str(dict) + "\""
        assert isinstance(output, dict) or output is None, "Attribute \"output\" is of type \"" + str(type(output)) + "\" but must be of type \"" + str(dict) + "\""
        assert isinstance(id, str), "Attribute \"id\" is of type \"" + str(type(id)) + "\" but must be of type \"" + str(str) + "\""
        if connectedTo is None:
            connectedTo = []
        if connectedBefore is None:
            connectedBefore = []
        if connectedAfter is None:
            connectedAfter = []
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
        if outputGradient is None:
            outputGradient = {}
        if parameterGradient is None:
            parameterGradient = {}
        self.connectedTo = connectedTo
        self.connectedBefore = connectedBefore
        self.connectedAfter = connectedAfter
        self.hasSubSystem = hasSubSystem
        self.subSystemOf = subSystemOf
        self.connectsAt = connectsAt
        self.connectedThrough = connectedThrough
        self.input = input 
        self.output = output
        self.outputGradient = outputGradient
        self.parameterGradient = parameterGradient
        self.id = id
        logger.info("[System Class] : Exited from __init__ Function")
