import twin4build.saref4bldg.physical_object.building_object.building_device.distribution_device.distribution_flow_device.energy_conversion_device.energy_conversion_device as energy_conversion_device
import saref.measurement.measurement as measurement
from typing import Union
class AirToAirHeatRecovery(energy_conversion_device.EnergyConversionDevice):
    def __init__(self,
                hasDefrost: Union[bool, None] = None,
                heatTransferTypeEnum: Union[str, None] = None,
                operationTemperatureMax: Union[measurement.Measurement, None] = None,
                operationTemperatureMin: Union[measurement.Measurement, None]  = None,
                primaryAirFlowRateMax: Union[measurement.Measurement, None]  = None,
                primaryAirFlowRateMin: Union[measurement.Measurement, None]  = None,
                secondaryAirFlowRateMax: Union[measurement.Measurement, None]  = None,
                secondaryAirFlowRateMin: Union[measurement.Measurement, None]  = None,
                **kwargs):
        super().__init__(**kwargs)
        assert isinstance(hasDefrost, bool) or hasDefrost is None, "Attribute \"hasDefrost\" is of type \"" + str(type(hasDefrost)) + "\" but must be of type \"" + str(bool) + "\""
        assert isinstance(heatTransferTypeEnum, str) or heatTransferTypeEnum is None, "Attribute \"heatTransferTypeEnum\" is of type \"" + str(type(heatTransferTypeEnum)) + "\" but must be of type \"" + str(str) + "\""
        assert isinstance(operationTemperatureMax, measurement.Measurement) or operationTemperatureMax is None, "Attribute \"operationTemperatureMax\" is of type \"" + str(type(operationTemperatureMax)) + "\" but must be of type \"" + str(measurement.Measurement) + "\""
        assert isinstance(operationTemperatureMin, measurement.Measurement) or operationTemperatureMin is None, "Attribute \"operationTemperatureMin\" is of type \"" + str(type(operationTemperatureMin)) + "\" but must be of type \"" + str(measurement.Measurement) + "\""
        assert isinstance(primaryAirFlowRateMax, measurement.Measurement) or primaryAirFlowRateMax is None, "Attribute \"primaryAirFlowRateMax\" is of type \"" + str(type(primaryAirFlowRateMax)) + "\" but must be of type \"" + str(measurement.Measurement) + "\""
        assert isinstance(primaryAirFlowRateMin, measurement.Measurement) or primaryAirFlowRateMin is None, "Attribute \"primaryAirFlowRateMin\" is of type \"" + str(type(primaryAirFlowRateMin)) + "\" but must be of type \"" + str(measurement.Measurement) + "\""
        assert isinstance(secondaryAirFlowRateMax, measurement.Measurement) or secondaryAirFlowRateMax is None, "Attribute \"secondaryAirFlowRateMax\" is of type \"" + str(type(secondaryAirFlowRateMax)) + "\" but must be of type \"" + str(measurement.Measurement) + "\""
        assert isinstance(secondaryAirFlowRateMin, measurement.Measurement) or secondaryAirFlowRateMin is None, "Attribute \"secondaryAirFlowRateMin\" is of type \"" + str(type(secondaryAirFlowRateMin)) + "\" but must be of type \"" + str(measurement.Measurement) + "\""
        self.hasDefrost = hasDefrost
        self.heatTransferTypeEnum = heatTransferTypeEnum
        self.operationTemperatureMax = operationTemperatureMax
        self.operationTemperatureMin = operationTemperatureMin
        self.primaryAirFlowRateMax = primaryAirFlowRateMax
        self.primaryAirFlowRateMin = primaryAirFlowRateMin
        self.secondaryAirFlowRateMax = secondaryAirFlowRateMax
        self.secondaryAirFlowRateMin = secondaryAirFlowRateMin