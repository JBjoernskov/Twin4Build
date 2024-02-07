import twin4build.saref4bldg.physical_object.building_object.building_device.distribution_device.distribution_flow_device.flow_moving_device.flow_moving_device as flow_moving_device
from typing import Union
import twin4build.saref.measurement.measurement as measurement
from twin4build.logger.Logging import Logging
logger = Logging.get_logger("ai_logfile")

class Pump(flow_moving_device.FlowMovingDevice):
    def __init__(self,
                connectionSize: Union[measurement.Measurement, None] = None,
                flowResistanceMax: Union[measurement.Measurement, None] = None,
                flowResistanceMin: Union[measurement.Measurement, None] = None,
                netPositiveSuctionHead: Union[measurement.Measurement, None] = None,
                nomminalRotationSpeed: Union[measurement.Measurement, None] = None,
                operationTemperatureMax: Union[measurement.Measurement, None] = None,
                operationTemperatureMin: Union[measurement.Measurement, None] = None,
                pumpFlowRateMax: Union[measurement.Measurement, None] = None,
                pumpFlowRateMin: Union[measurement.Measurement, None] = None,
                **kwargs):

        
        logger.info("[fan class] : Entered in Initialise Function")

        super().__init__(**kwargs)
        assert isinstance(connectionSize, measurement.Measurement) or connectionSize is None, "Attribute \"connectionSize\" is of type \"" + str(type(connectionSize)) + "\" but must be of type \"" + str(measurement.Measurement) + "\""
        assert isinstance(flowResistanceMax, measurement.Measurement) or flowResistanceMax is None, "Attribute \"flowResistanceMax\" is of type \"" + str(type(flowResistanceMax)) + "\" but must be of type \"" + str(measurement.Measurement) + "\""
        assert isinstance(flowResistanceMin, measurement.Measurement) or flowResistanceMin is None, "Attribute \"flowResistanceMin\" is of type \"" + str(type(flowResistanceMin)) + "\" but must be of type \"" + str(measurement.Measurement) + "\""
        assert isinstance(netPositiveSuctionHead, measurement.Measurement) or netPositiveSuctionHead is None, "Attribute \"netPositiveSuctionHead\" is of type \"" + str(type(netPositiveSuctionHead)) + "\" but must be of type \"" + str(measurement.Measurement) + "\""
        assert isinstance(nomminalRotationSpeed, measurement.Measurement) or nomminalRotationSpeed is None, "Attribute \"nomminalRotationSpeed\" is of type \"" + str(type(nomminalRotationSpeed)) + "\" but must be of type \"" + str(measurement.Measurement) + "\""
        assert isinstance(operationTemperatureMax, measurement.Measurement) or operationTemperatureMax is None, "Attribute \"operationTemperatureMax\" is of type \"" + str(type(operationTemperatureMax)) + "\" but must be of type \"" + str(measurement.Measurement) + "\""
        assert isinstance(operationTemperatureMin, measurement.Measurement) or operationTemperatureMin is None, "Attribute \"operationTemperatureMin\" is of type \"" + str(type(operationTemperatureMin)) + "\" but must be of type \"" + str(measurement.Measurement) + "\""
        assert isinstance(pumpFlowRateMax, measurement.Measurement) or pumpFlowRateMax is None, "Attribute \"pumpFlowRateMax\" is of type \"" + str(type(pumpFlowRateMax)) + "\" but must be of type \"" + str(measurement.Measurement) + "\""
        assert isinstance(pumpFlowRateMin, measurement.Measurement) or pumpFlowRateMin is None, "Attribute \"pumpFlowRateMin\" is of type \"" + str(type(pumpFlowRateMin)) + "\" but must be of type \"" + str(measurement.Measurement) + "\""

        self.connectionSize = connectionSize
        self.flowResistanceMax = flowResistanceMax
        self.flowResistanceMin = flowResistanceMin
        self.netPositiveSuctionHead = netPositiveSuctionHead
        self.nomminalRotationSpeed = nomminalRotationSpeed
        self.operationTemperatureMax = operationTemperatureMax
        self.operationTemperatureMin = operationTemperatureMin
        self.pumpFlowRateMax = pumpFlowRateMax
        self.pumpFlowRateMin = pumpFlowRateMin

        logger.info("[fan class] : Exited from Initialise Function")
