from __future__ import annotations
from typing import Union
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import twin4build.saref.measurement.measurement as measurement
    import twin4build.saref.property_.property_ as property_

import os 
import sys 

uppath = lambda _path,n: os.sep.join(_path.split(os.sep)[:-n])
file_path = uppath(os.path.abspath(__file__), 4)
sys.path.append(file_path)

from twin4build.logger.Logging import Logging


logger = Logging.get_logger("ai_logfile")

class FeatureOfInterest:
    def __init__(self,
                hasMeasurement: Union[None, measurement.Measurement]=None,
                hasProperty: Union[None, list]=None,
                **kwargs):
        
        logger.info("[Feature Of Interest] : Entered in Initialise Function")

        super().__init__(**kwargs)
        import twin4build.saref.measurement.measurement as measurement
        import twin4build.saref.property_.property_ as property_
        assert isinstance(hasMeasurement, measurement.Measurement) or hasMeasurement is None, "Attribute \"hasMeasurement\" is of type \"" + str(type(hasMeasurement)) + "\" but must be of type \"" + str(measurement.Measurement) + "\""
        assert isinstance(hasProperty, list) or hasProperty is None, "Attribute \"hasProperty\" is of type \"" + str(type(hasProperty)) + "\" but must be of type \"" + str(list) + "\""
        if hasProperty is None:
            hasProperty = []
        self.hasMeasurement = hasMeasurement
        self.hasProperty = hasProperty

        logger.info("[Feature Of Interest] : Exited from Initialise Function")
