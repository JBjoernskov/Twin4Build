from __future__ import annotations
from typing import Union
import twin4build.saref.property_.property_ as property_
class Energy(property_.Property):
    MEASURING_UNCERTAINTY = 0
    MEASURING_TYPE = "A" #Absolute
    def __init__(self,
                **kwargs):
        super().__init__(**kwargs)