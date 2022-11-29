import twin4build.saref4bldg.physical_object.building_object.building_device.distribution_device.distribution_flow_device.energy_conversion_device.energy_conversion_device as energy_conversion_device
from typing import Union
import twin4build.saref.measurement.measurement as measurement
class Coil(energy_conversion_device.EnergyConversionDevice):
    def __init__(self,
                airFlowRateMax: Union[measurement.Measurement, None] = None,
                airFlowRateMin: Union[measurement.Measurement, None] = None,
                nominalLatentCapacity: Union[measurement.Measurement, None] = None,
                nominalSensibleCapacity: Union[measurement.Measurement, None] = None,
                nominalUa: Union[measurement.Measurement, None] = None,
                operationTemperatureMax: Union[measurement.Measurement, None] = None,
                operationTemperatureMin: Union[measurement.Measurement, None] = None,
                placementType: Union[str, None] = None,
                operationMode: Union[str, None] = None,
                **kwargs):
        super().__init__(**kwargs)
        assert isinstance(airFlowRateMax, measurement.Measurement) or airFlowRateMax is None, "Attribute \"airFlowRateMax\" is of type \"" + str(type(airFlowRateMax)) + "\" but must be of type \"" + str(measurement.Measurement) + "\""
        assert isinstance(airFlowRateMin, measurement.Measurement) or airFlowRateMin is None, "Attribute \"airFlowRateMin\" is of type \"" + str(type(airFlowRateMin)) + "\" but must be of type \"" + str(measurement.Measurement) + "\""
        assert isinstance(nominalLatentCapacity, measurement.Measurement) or nominalLatentCapacity is None, "Attribute \"nominalLatentCapacity\" is of type \"" + str(type(nominalLatentCapacity)) + "\" but must be of type \"" + str(measurement.Measurement) + "\""
        assert isinstance(nominalSensibleCapacity, measurement.Measurement) or nominalSensibleCapacity is None, "Attribute \"nominalSensibleCapacity\" is of type \"" + str(type(nominalSensibleCapacity)) + "\" but must be of type \"" + str(measurement.Measurement) + "\""
        assert isinstance(nominalUa, measurement.Measurement) or nominalUa is None, "Attribute \"nominalUa\" is of type \"" + str(type(nominalUa)) + "\" but must be of type \"" + str(measurement.Measurement) + "\""
        assert isinstance(operationTemperatureMax, measurement.Measurement) or operationTemperatureMax is None, "Attribute \"operationTemperatureMax\" is of type \"" + str(type(operationTemperatureMax)) + "\" but must be of type \"" + str(measurement.Measurement) + "\""
        assert isinstance(operationTemperatureMin, measurement.Measurement) or operationTemperatureMin is None, "Attribute \"operationTemperatureMin\" is of type \"" + str(type(operationTemperatureMin)) + "\" but must be of type \"" + str(measurement.Measurement) + "\""
        assert isinstance(placementType, str) or placementType is None, "Attribute \"placementType\" is of type \"" + str(type(placementType)) + "\" but must be of type \"" + str(str) + "\""
        assert isinstance(operationMode, str) or operationMode is None, "Attribute \"operationMode\" is of type \"" + str(type(operationMode)) + "\" but must be of type \"" + str(str) + "\""
        self.airFlowRateMax = airFlowRateMax
        self.airFlowRateMin = airFlowRateMin
        self.nominalLatentCapacity = nominalLatentCapacity
        self.nominalSensibleCapacity = nominalSensibleCapacity
        self.nominalUa = nominalUa
        self.operationTemperatureMax = operationTemperatureMax
        self.operationTemperatureMin = operationTemperatureMin
        self.placementType = placementType
        self.operationMode = operationMode