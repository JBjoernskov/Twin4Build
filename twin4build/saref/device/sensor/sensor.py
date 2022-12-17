from __future__ import annotations
from typing import Union
import twin4build.saref.device.device as device
class Sensor(device.Device):
    def __init__(self,
                hasFunction: Union[SensingFunction, None] = None, 
                **kwargs):
        super().__init__(**kwargs)