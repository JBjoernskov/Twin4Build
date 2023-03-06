from __future__ import annotations
from typing import Union
import twin4build.saref.device.device as device
import twin4build.saref.function.metering_function.metering_function as metering_function
class Meter(device.Device):
    def __init__(self,
                hasFunction: Union[metering_function.MeteringFunction, None]=None, 
                **kwargs):
        super().__init__(**kwargs)
        assert isinstance(hasFunction, metering_function.MeteringFunction) or hasFunction is None, "Attribute \"hasFunction\" is of type \"" + str(type(hasFunction)) + "\" but must be of type \"" + str(metering_function.MeteringFunction) + "\""
        self.hasFunction = hasFunction