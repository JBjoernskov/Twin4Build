from __future__ import annotations
from typing import Union
import twin4build.saref.function.function as function
import twin4build.saref.property_.property_ as property_
class SensingFunction(function.Function):
    def __init__(self,
                hasSensingRange: Union[list, None] = None,
                hasSensorType: Union[property_.Property, None] = None,
                **kwargs):
        super().__init__(**kwargs)
        assert isinstance(hasSensingRange, list) or hasSensingRange is None, "Attribute \"hasSensingRange\" is of type \"" + str(type(hasSensingRange)) + "\" but must be of type \"" + str(list) + "\""
        assert isinstance(hasSensorType, property_.Property) or hasSensorType is None, "Attribute \"hasCommand\" is of type \"" + str(type(hasSensorType)) + "\" but must be of type \"" + str(property_.Property) + "\""
