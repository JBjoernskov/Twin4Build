from __future__ import annotations
from typing import Union
import twin4build.saref.function.function as function
import twin4build.saref.commodity.commodity as commodity
import twin4build.saref.property_.property_ as property_

import os 
import sys 

uppath = lambda _path,n: os.sep.join(_path.split(os.sep)[:-n])
file_path = uppath(os.path.abspath(__file__), 5)
sys.path.append(file_path)

from twin4build.logger.Logging import Logging

logger = Logging.get_logger("ai_logfile")

class MeteringFunction(function.Function):
    def __init__(self,
                hasMeterReading: Union[list, None]=None,
                hasMeterReadingType: Union[commodity.Commodity, property_.Property, None]=None,
                **kwargs):
        
        logger.info("[Metering Function] : Entered in Initialise Function")
        
        super().__init__(**kwargs)
        assert isinstance(hasMeterReading, list) or hasMeterReading is None, "Attribute \"hasMeterReading\" is of type \"" + str(type(hasMeterReading)) + "\" but must be of type \"" + str(list) + "\""
        assert isinstance(hasMeterReadingType, commodity.Commodity) or isinstance(hasMeterReadingType, property_.Property) or hasMeterReadingType is None, "Attribute \"hasMeterReadingType\" is of type \"" + str(type(hasMeterReadingType)) + "\" but must be of type \"" + str(property_.Property) + "\" or \"" + str(property_.Property) + "\""
        self.hasMeterReading = hasMeterReading
        self.hasMeterReadingType = hasMeterReadingType

        
        logger.info("[Metering Function] : Exited from Initialise Function")
        
