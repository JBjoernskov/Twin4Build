from __future__ import annotations
from typing import Union
import twin4build.saref.property_.property_ as property_
class Flow(property_.Property):
    # Assumming "Pitot Tube" from "Error Analysis of Measurement and Control Techniques 
    # of Outside Air Intake Rates in VAV Systems", C. C. Schroeder, M. Krarti, M. J. Brandemuehl
    MEASURING_UNCERTAINTY = 5
    MEASURING_TYPE = "P" #Percent
    def __init__(self,
                **kwargs):
        super().__init__(**kwargs)
