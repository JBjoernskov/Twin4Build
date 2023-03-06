from __future__ import annotations
from typing import Union
import twin4build.saref.device.device as device
import twin4build.saref.function.sensing_function.sensing_function as sensing_function
class Sensor(device.Device):
    def __init__(self,
                hasFunction: Union[sensing_function.SensingFunction, None]=None, 
                **kwargs):
        super().__init__(**kwargs)
        assert isinstance(hasFunction, sensing_function.SensingFunction) or hasFunction is None, "Attribute \"hasFunction\" is of type \"" + str(type(hasFunction)) + "\" but must be of type \"" + str(sensing_function.SensingFunction) + "\""
        self.hasFunction = hasFunction