import twin4build.saref4bldg.physical_object.building_object.building_device.distribution_device.distribution_flow_device.flow_duct_device.flow_duct_device as flow_duct_device
from typing import Union
import twin4build.saref.property_value.property_value as property_value
import twin4build.saref.property_.s4bldg_property.s4bldg_property as s4bldg_property
from twin4build.logger.Logging import Logging

logger = Logging.get_logger("ai_logfile")

class JunctionDuct(flow_duct_device.FlowDuctDevice):
    def __init__(self,
                airFlowRateBias: Union[property_value.PropertyValue, None] = None,
                **kwargs):
        logger.info("[junction duct class] : Entered in Initialise Function")
        super().__init__(**kwargs)

        airFlowRateBias_ = s4bldg_property.NominalAirFlowRate()

        if airFlowRateBias is not None:
            airFlowRateBias = property_value.PropertyValue(hasValue=airFlowRateBias.hasValue,
                                                            isMeasuredIn=airFlowRateBias.isMeasuredIn,
                                                            isValueOfProperty=airFlowRateBias_)
        else:
            airFlowRateBias = property_value.PropertyValue(isValueOfProperty=airFlowRateBias_)
        self.hasProperty.append(airFlowRateBias_)
        self.hasPropertyValue.append(airFlowRateBias)

        logger.info("[junction duct class] : Exited from Initialise Function")

    @property
    def airFlowRateBias(self):
        el = [el for el in self.hasPropertyValue if isinstance(el.isValueOfProperty, s4bldg_property.NominalAirFlowRate)]
        return el[0] if len(el) > 0 else None

