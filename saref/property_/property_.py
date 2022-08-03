from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ctypes import Union
    from ..device.device import Device
    from ..feature_of_interest.feature_of_interest import FeatureOfInterest
    from ..measurement.measurement import Measurement

class Property:
    def __init__(self,
                isControlledByDevice = None,
                isMeasuredByDevice = None,
                isPropertyOf = None,
                relatesToMeasurement = None):
        isControlledByDevice = Union(None, Device)
        isMeasuredByDevice = Union(None, Device)
        isPropertyOf = Union(None, FeatureOfInterest)
        relatesToMeasurement = Union(None, Measurement)
        self.isControlledByDevice = isControlledByDevice
        self.isMeasuredByDevice = isMeasuredByDevice
        self.isPropertyOf = isPropertyOf
        self.relatesToMeasurement = relatesToMeasurement