from __future__ import annotations
from typing import Union
import twin4build.saref.unit_of_measure.unit_of_measure as unit_of_measure
import twin4build.saref.property_.property_ as property_
class PropertyValue:
    def __init__(self,
                hasValue: Union[float, int, None]=None,
                isMeasuredIn: Union[unit_of_measure.UnitOfMeasure, None]=None,
                isValueOfProperty: Union[property_.Property, None]=None):
        assert isinstance(hasValue, float) or isinstance(hasValue, int) or isinstance(hasValue, str) or hasValue is None, "Attribute \"hasValue\" is of type \"" + str(type(hasValue)) + "\" but must be of type \"" + str(int) + "\""
        assert isMeasuredIn is None or issubclass(isMeasuredIn, unit_of_measure.UnitOfMeasure), "Attribute \"isMeasuredIn\" is of type \"" + str(isMeasuredIn.__name__) + "\" but must be of type \"" + "<class 'type'>" + "\""
        assert isinstance(isValueOfProperty, property_.Property) or isValueOfProperty is None, "Attribute \"isValueOfProperty\" is of type \"" + str(type(isValueOfProperty)) + "\" but must be of type \"" + str(property_.Property) + "\""
        self.hasValue = hasValue
        self.isMeasuredIn = isMeasuredIn
        self.isValueOfProperty = isValueOfProperty