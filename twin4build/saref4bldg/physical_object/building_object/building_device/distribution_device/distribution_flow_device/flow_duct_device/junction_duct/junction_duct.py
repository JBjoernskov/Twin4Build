import twin4build.saref4bldg.physical_object.building_object.building_device.distribution_device.distribution_flow_device.flow_duct_device.flow_duct_device as flow_duct_device
from typing import Union
import twin4build.saref.property_value.property_value as property_value
import twin4build.saref.property_.s4bldg_property.s4bldg_property as s4bldg_property
from twin4build.logger.Logging import Logging
import twin4build.saref4syst.system as system

logger = Logging.get_logger("ai_logfile")

class JunctionDuct(system.System):
    def __init__(self,
                **kwargs):
        logger.info("[junction duct class] : Entered in Initialise Function")
        super().__init__(**kwargs)

        logger.info("[junction duct class] : Exited from Initialise Function")


