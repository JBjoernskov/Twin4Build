from __future__ import annotations
from typing import Union
import twin4build.saref.property_.property_ as property_
class Power(property_.Property):
    MEASURING_UNCERTAINTY = 5
    MEASURING_TYPE = "P" #Percentage
    def __init__(self,
                **kwargs):
        super().__init__(**kwargs)