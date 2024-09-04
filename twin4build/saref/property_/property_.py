from __future__ import annotations
from typing import Union
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import twin4build.saref.device.device as device
    import twin4build.saref.feature_of_interest.feature_of_interest as feature_of_interest
    import twin4build.saref.measurement.measurement as measurement
import os 
import sys
uppath = lambda _path,n: os.sep.join(_path.split(os.sep)[:-n])
file_path = uppath(os.path.abspath(__file__), 4)
sys.path.append(file_path)

class Property:
    def __init__(self,
                isControlledBy: Union[list, None]=None,
                # isactuatedByDevice: Union[device.Device, None]=None,
                isObservedBy: Union[list, None]=None,
                isPropertyOf: Union[feature_of_interest.FeatureOfInterest, None]=None,
                relatesToMeasurement: Union[measurement.Measurement, None]=None):
        import twin4build.saref.device.device as device
        import twin4build.saref.feature_of_interest.feature_of_interest as feature_of_interest
        import twin4build.saref.measurement.measurement as measurement
        assert isinstance(isControlledBy, list) or isControlledBy is None, "Attribute \"isControlledBy\" is of type \"" + str(type(isControlledBy)) + "\" but must be of type \"" + str(list) + "\""
        # assert isinstance(isactuatedByDevice, device.Device) or isactuatedByDevice is None, "Attribute \"isactuatedByDevice\" is of type \"" + str(type(isactuatedByDevice)) + "\" but must be of type \"" + str(device.Device) + "\""
        assert isinstance(isObservedBy, list) or isObservedBy is None, "Attribute \"isObservedBy\" is of type \"" + str(type(isObservedBy)) + "\" but must be of type \"" + str(list) + "\""
        assert isinstance(isPropertyOf, feature_of_interest.FeatureOfInterest) or isPropertyOf is None, "Attribute \"isPropertyOf\" is of type \"" + str(type(isPropertyOf)) + "\" but must be of type \"" + str(feature_of_interest.FeatureOfInterest) + "\""
        assert isinstance(relatesToMeasurement, measurement.Measurement) or relatesToMeasurement is None, "Attribute \"relatesToMeasurement\" is of type \"" + str(type(relatesToMeasurement)) + "\" but must be of type \"" + str(measurement.Measurement) + "\""
        if isControlledBy is None:
            isControlledBy = []
        if isObservedBy is None:
            isObservedBy = []
        self.isControlledBy = isControlledBy
        self.isObservedBy = isObservedBy
        self.isPropertyOf = isPropertyOf
        self.relatesToMeasurement = relatesToMeasurement