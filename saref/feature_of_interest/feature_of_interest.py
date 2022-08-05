from __future__ import annotations
from typing import Union
import saref.measurement.measurement as measurement
import saref.property_.property_ as property_
class FeatureOfInterest:
    def __init__(self,
                hasMeasurement: Union[None, measurement.Measurement] = None,
                hasProperty: Union[None, property_.Property] = None):
        assert isinstance(hasMeasurement, measurement.Measurement) or hasMeasurement is None, "Attribute \"hasMeasurement\" is of type \"" + str(type(hasMeasurement)) + "\" but must be of type \"" + str(measurement.Measurement) + "\""
        assert isinstance(hasProperty, property_.Property) or hasProperty is None, "Attribute \"hasProperty\" is of type \"" + str(type(hasProperty)) + "\" but must be of type \"" + str(property_.Property) + "\""
        self.hasMeasurement = hasMeasurement
        self.hasProperty = hasProperty
        