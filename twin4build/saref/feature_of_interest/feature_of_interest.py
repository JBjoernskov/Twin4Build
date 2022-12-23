from __future__ import annotations
from typing import Union
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import twin4build.saref.measurement.measurement as measurement
    import twin4build.saref.property_.property_ as property_
class FeatureOfInterest:
    def __init__(self,
                hasMeasurement: Union[None, measurement.Measurement] = None,
                hasProperty: Union[None, list] = None,
                **kwargs):
        super().__init__(**kwargs)
        print(kwargs)
        import twin4build.saref.measurement.measurement as measurement
        import twin4build.saref.property_.property_ as property_
        assert isinstance(hasMeasurement, measurement.Measurement) or hasMeasurement is None, "Attribute \"hasMeasurement\" is of type \"" + str(type(hasMeasurement)) + "\" but must be of type \"" + str(measurement.Measurement) + "\""
        assert isinstance(hasProperty, list) or hasProperty is None, "Attribute \"hasProperty\" is of type \"" + str(type(hasProperty)) + "\" but must be of type \"" + str(list) + "\""
        self.hasMeasurement = hasMeasurement
        self.hasProperty = hasProperty
        