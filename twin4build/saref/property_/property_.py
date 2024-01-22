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

from twin4build.logger.Logging import Logging

logger = Logging.get_logger("ai_logfile")

class Property:
    def __init__(self,
                isControlledByDevice: Union(device.Device, None)=None,
                isactuatedByDevice: Union(device.Device, None)=None,
                isMeasuredByDevice: Union(device.Device, None)=None,
                isPropertyOf: Union(feature_of_interest.FeatureOfInterest, None)=None,
                relatesToMeasurement: Union(measurement.Measurement, None)=None):
        
        logger.info("[Saref.Property] : Entered in Initialise Class")

        import twin4build.saref.device.device as device
        import twin4build.saref.feature_of_interest.feature_of_interest as feature_of_interest
        import twin4build.saref.measurement.measurement as measurement
        assert isinstance(isControlledByDevice, device.Device) or isControlledByDevice is None, "Attribute \"isControlledByDevice\" is of type \"" + str(type(isControlledByDevice)) + "\" but must be of type \"" + str(device.Device) + "\""
        assert isinstance(isactuatedByDevice, device.Device) or isactuatedByDevice is None, "Attribute \"isactuatedByDevice\" is of type \"" + str(type(isactuatedByDevice)) + "\" but must be of type \"" + str(device.Device) + "\""
        assert isinstance(isMeasuredByDevice, device.Device) or isMeasuredByDevice is None, "Attribute \"isMeasuredByDevice\" is of type \"" + str(type(isMeasuredByDevice)) + "\" but must be of type \"" + str(device.Device) + "\""
        assert isinstance(isPropertyOf, feature_of_interest.FeatureOfInterest) or isPropertyOf is None, "Attribute \"isPropertyOf\" is of type \"" + str(type(isPropertyOf)) + "\" but must be of type \"" + str(feature_of_interest.FeatureOfInterest) + "\""
        assert isinstance(relatesToMeasurement, measurement.Measurement) or relatesToMeasurement is None, "Attribute \"relatesToMeasurement\" is of type \"" + str(type(relatesToMeasurement)) + "\" but must be of type \"" + str(measurement.Measurement) + "\""
        self.isControlledByDevice = isControlledByDevice
        self.isactuatedByDevice = isactuatedByDevice ###
        self.isMeasuredByDevice = isMeasuredByDevice
        self.isPropertyOf = isPropertyOf
        self.relatesToMeasurement = relatesToMeasurement

        
        logger.info("[Saref.Property] : Exited from Initialise Class")
