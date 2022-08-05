from __future__ import annotations
from typing import Union
import twin4build.saref.date_time.date_time as date_time
import twin4build.saref.unit_of_measure.unit_of_measure as unit_of_measure
import twin4build.saref.feature_of_interest.feature_of_interest as feature_of_interest
import twin4build.saref.property_.property_ as property_
class Measurement:
    def __init__(self,
                hasTimeStamp: Union[date_time.DateTime, None] = None,
                hasValue: Union[float, int, None] = None,
                isMeasuredIn: Union[unit_of_measure.UnitOfMeasure, None] = None,
                isMeasurementOf: Union[feature_of_interest.FeatureOfInterest, None] = None,
                relatesToProperty: Union[property_.Property, None] = None):
        assert isinstance(hasTimeStamp, date_time.DateTime) or hasTimeStamp is None, "Attribute \"hasTimeStamp\" is of type \"" + str(type(hasTimeStamp)) + "\" but must be of type \"" + str(date_time.DateTime) + "\""
        assert isinstance(hasValue, float) or isinstance(hasValue, int) or hasValue is None, "Attribute \"hasValue\" is of type \"" + str(type(hasValue)) + "\" but must be of type \"" + str(int) + "\""
        assert isinstance(isMeasuredIn, unit_of_measure.UnitOfMeasure) or isMeasuredIn is None, "Attribute \"isMeasuredIn\" is of type \"" + str(type(isMeasuredIn)) + "\" but must be of type \"" + str(unit_of_measure.UnitOfMeasure) + "\""
        assert isinstance(isMeasurementOf, feature_of_interest.FeatureOfInterest) or isMeasurementOf is None, "Attribute \"isMeasurementOf\" is of type \"" + str(type(isMeasurementOf)) + "\" but must be of type \"" + str(feature_of_interest.FeatureOfInterest) + "\""
        assert isinstance(relatesToProperty, property_.Property) or relatesToProperty is None, "Attribute \"relatesToProperty\" is of type \"" + str(type(relatesToProperty)) + "\" but must be of type \"" + str(property_.Property) + "\""
        self.hasTimeStamp = hasTimeStamp
        self.hasValue = hasValue
        self.isMeasuredIn = isMeasuredIn
        self.isMeasurementOf = isMeasurementOf
        self.relatesToProperty = relatesToProperty