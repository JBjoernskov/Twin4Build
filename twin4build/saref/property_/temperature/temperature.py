from __future__ import annotations
from typing import Union
import twin4build.saref.property_.property_ as property_
class Temperature(property_.Property):
    MEASURING_UNCERTAINTY = 0.36*5/9 #Convert from fahrenheit to celcius
    MEASURING_TYPE = "A" #Absolute
    def __init__(self,
                **kwargs):
        super().__init__(**kwargs)
