import twin4build.saref4bldg.physical_object.building_object.building_device.distribution_device.distribution_flow_device.flow_moving_device.flow_moving_device as flow_moving_device
from typing import Union
import twin4build.saref.property_value.property_value as property_value
import twin4build.saref.property_.s4bldg_property.s4bldg_property as s4bldg_property
import twin4build.saref.unit_of_measure.unit_of_measure as unit_of_measure
from twin4build.logger.Logging import Logging
logger = Logging.get_logger("ai_logfile")

class Pump(flow_moving_device.FlowMovingDevice):
    def __init__(self,
                connectionSize: Union[property_value.PropertyValue, None] = None,
                flowResistanceMax: Union[property_value.PropertyValue, None] = None,
                flowResistanceMin: Union[property_value.PropertyValue, None] = None,
                netPositiveSuctionHead: Union[property_value.PropertyValue, None] = None,
                nomminalRotationSpeed: Union[property_value.PropertyValue, None] = None,
                operationTemperatureMax: Union[property_value.PropertyValue, None] = None,
                operationTemperatureMin: Union[property_value.PropertyValue, None] = None,
                pumpFlowRateMax: Union[property_value.PropertyValue, None] = None,
                pumpFlowRateMin: Union[property_value.PropertyValue, None] = None,
                **kwargs):
        logger.info("[fan class] : Entered in Initialise Function")
        super().__init__(**kwargs)

        connectionSize_ = s4bldg_property.ConnectionSize()
        if connectionSize is not None:
            connectionSize = property_value.PropertyValue(hasValue=connectionSize.hasValue,
                                                            isMeasuredIn=connectionSize.isMeasuredIn,
                                                            isValueOfProperty=connectionSize_)
        else:
            connectionSize = property_value.PropertyValue(isValueOfProperty=connectionSize_)
        self.hasProperty.append(connectionSize_)
        self.hasPropertyValue.append(connectionSize)

        flowResistanceMax_ = s4bldg_property.FlowResistanceMax()
        if flowResistanceMax is not None:
            flowResistanceMax = property_value.PropertyValue(hasValue=flowResistanceMax.hasValue,
                                    isMeasuredIn=flowResistanceMax.isMeasuredIn,
                                    isValueOfProperty=flowResistanceMax_)
        else:
            flowResistanceMax = property_value.PropertyValue(isValueOfProperty=flowResistanceMax_)
        self.hasProperty.append(flowResistanceMax_)
        self.hasPropertyValue.append(flowResistanceMax)

        flowResistanceMin_ = s4bldg_property.FlowResistanceMin()
        if flowResistanceMin is not None:
            flowResistanceMin = property_value.PropertyValue(hasValue=flowResistanceMin.hasValue,
                                    isMeasuredIn=flowResistanceMin.isMeasuredIn,
                                    isValueOfProperty=flowResistanceMin_)
        else:
            flowResistanceMin = property_value.PropertyValue(isValueOfProperty=flowResistanceMin_)
        self.hasProperty.append(flowResistanceMin_)
        self.hasPropertyValue.append(flowResistanceMin)

        netPositiveSuctionHead_ = s4bldg_property.NetPositiveSuctionHead()
        if netPositiveSuctionHead is not None:
            netPositiveSuctionHead = property_value.PropertyValue(hasValue=netPositiveSuctionHead.hasValue,
                                      isMeasuredIn=netPositiveSuctionHead.isMeasuredIn,
                                      isValueOfProperty=netPositiveSuctionHead_)
        else:
            netPositiveSuctionHead = property_value.PropertyValue(isValueOfProperty=netPositiveSuctionHead_)
        self.hasProperty.append(netPositiveSuctionHead_)
        self.hasPropertyValue.append(netPositiveSuctionHead)

        nomminalRotationSpeed_ = s4bldg_property.NominalRotationSpeed()
        if nomminalRotationSpeed is not None:
            nomminalRotationSpeed = property_value.PropertyValue(hasValue=nomminalRotationSpeed.hasValue,
                                     isMeasuredIn=nomminalRotationSpeed.isMeasuredIn,
                                     isValueOfProperty=nomminalRotationSpeed_)
        else:
            nomminalRotationSpeed = property_value.PropertyValue(isValueOfProperty=nomminalRotationSpeed_)
        self.hasProperty.append(nomminalRotationSpeed_)
        self.hasPropertyValue.append(nomminalRotationSpeed)

        operationTemperatureMax_ = s4bldg_property.OperationTemperatureMax()
        if operationTemperatureMax is not None:
            operationTemperatureMax = property_value.PropertyValue(hasValue=operationTemperatureMax.hasValue,
                                       isMeasuredIn=operationTemperatureMax.isMeasuredIn,
                                       isValueOfProperty=operationTemperatureMax_)
        else:
            operationTemperatureMax = property_value.PropertyValue(isValueOfProperty=operationTemperatureMax_)
        self.hasProperty.append(operationTemperatureMax_)
        self.hasPropertyValue.append(operationTemperatureMax)

        operationTemperatureMin_ = s4bldg_property.OperationTemperatureMin()
        if operationTemperatureMin is not None:
            operationTemperatureMin = property_value.PropertyValue(hasValue=operationTemperatureMin.hasValue,
                                       isMeasuredIn=operationTemperatureMin.isMeasuredIn,
                                       isValueOfProperty=operationTemperatureMin_)
        else:
            operationTemperatureMin = property_value.PropertyValue(isValueOfProperty=operationTemperatureMin_)
        self.hasProperty.append(operationTemperatureMin_)
        self.hasPropertyValue.append(operationTemperatureMin)

        pumpFlowRateMax_ = s4bldg_property.PumpFlowRateMax()
        if pumpFlowRateMax is not None:
            pumpFlowRateMax = property_value.PropertyValue(hasValue=pumpFlowRateMax.hasValue,
                                   isMeasuredIn=pumpFlowRateMax.isMeasuredIn,
                                   isValueOfProperty=pumpFlowRateMax_)
        else:
            pumpFlowRateMax = property_value.PropertyValue(isValueOfProperty=pumpFlowRateMax_)
        self.hasProperty.append(pumpFlowRateMax_)
        self.hasPropertyValue.append(pumpFlowRateMax)

        pumpFlowRateMin_ = s4bldg_property.PumpFlowRateMin()
        if pumpFlowRateMin is not None:
            pumpFlowRateMin = property_value.PropertyValue(hasValue=pumpFlowRateMin.hasValue,
                                   isMeasuredIn=pumpFlowRateMin.isMeasuredIn,
                                   isValueOfProperty=pumpFlowRateMin_)
        else:
            pumpFlowRateMin = property_value.PropertyValue(isValueOfProperty=pumpFlowRateMin_)
        self.hasProperty.append(pumpFlowRateMin_)
        self.hasPropertyValue.append(pumpFlowRateMin)
        

        logger.info("[fan class] : Exited from Initialise Function")

    @property
    def pumpFlowRateMin(self):
        el = [el for el in self.hasPropertyValue if isinstance(el.isValueOfProperty, s4bldg_property.PumpFlowRateMin)]
        return el[0] if len(el) > 0 else None
    
    @property
    def connectionSize(self):
        el = [el for el in self.hasPropertyValue if isinstance(el.isValueOfProperty, s4bldg_property.ConnectionSize)]
        return el[0] if len(el) > 0 else None

    @property
    def flowResistanceMax(self):
        el = [el for el in self.hasPropertyValue if isinstance(el.isValueOfProperty, s4bldg_property.FlowResistanceMax)]
        return el[0] if len(el) > 0 else None

    @property
    def flowResistanceMin(self):
        el = [el for el in self.hasPropertyValue if isinstance(el.isValueOfProperty, s4bldg_property.FlowResistanceMin)]
        return el[0] if len(el) > 0 else None

    @property
    def netPositiveSuctionHead(self):
        el = [el for el in self.hasPropertyValue if isinstance(el.isValueOfProperty, s4bldg_property.NetPositiveSuctionHead)]
        return el[0] if len(el) > 0 else None

    @property
    def nomminalRotationSpeed(self):
        el = [el for el in self.hasPropertyValue if isinstance(el.isValueOfProperty, s4bldg_property.NominalRotationSpeed)]
        return el[0] if len(el) > 0 else None

    @property
    def operationTemperatureMax(self):
        el = [el for el in self.hasPropertyValue if isinstance(el.isValueOfProperty, s4bldg_property.OperationTemperatureMax)]
        return el[0] if len(el) > 0 else None

    @property
    def operationTemperatureMin(self):
        el = [el for el in self.hasPropertyValue if isinstance(el.isValueOfProperty, s4bldg_property.OperationTemperatureMin)]
        return el[0] if len(el) > 0 else None

    @property
    def pumpFlowRateMax(self):
        el = [el for el in self.hasPropertyValue if isinstance(el.isValueOfProperty, s4bldg_property.PumpFlowRateMax)]
        return el[0] if len(el) > 0 else None