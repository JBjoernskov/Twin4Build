import twin4build.saref4bldg.physical_object.building_object.building_device.distribution_device.distribution_flow_device.flow_moving_device.flow_moving_device as flow_moving_device
from typing import Union
import twin4build.saref.measurement.measurement as measurement
class Fan(flow_moving_device.FlowMovingDevice):
    def __init__(self,
                capacityControlType: Union[str, None] = None,
                motorDriveType: Union[str, None] = None,
                nominalAirFlowRate: Union[measurement.Measurement, None] = None,
                nominalPowerRate: Union[measurement.Measurement, None] = None,
                nominalRotationSpeed: Union[measurement.Measurement, None] = None,
                nominalStaticPressure: Union[measurement.Measurement, None] = None,
                nominalTotalPressure: Union[measurement.Measurement, None] = None,
                operationTemperatureMax: Union[measurement.Measurement, None] = None,
                operationTemperatureMin: Union[measurement.Measurement, None] = None,
                operationalRiterial: Union[measurement.Measurement, None] = None,
                **kwargs):
        super().__init__(**kwargs)
        assert isinstance(capacityControlType, str) or capacityControlType is None, "Attribute \"capacityControlType\" is of type \"" + str(type(capacityControlType)) + "\" but must be of type \"" + str(str) + "\""
        assert isinstance(motorDriveType, str) or motorDriveType is None, "Attribute \"motorDriveType\" is of type \"" + str(type(motorDriveType)) + "\" but must be of type \"" + str(str) + "\""
        assert isinstance(nominalAirFlowRate, measurement.Measurement) or nominalAirFlowRate is None, "Attribute \"nominalAirFlowRate\" is of type \"" + str(type(nominalAirFlowRate)) + "\" but must be of type \"" + str(measurement.Measurement) + "\""
        assert isinstance(nominalPowerRate, measurement.Measurement) or nominalPowerRate is None, "Attribute \"nominalPowerRate\" is of type \"" + str(type(nominalPowerRate)) + "\" but must be of type \"" + str(measurement.Measurement) + "\""
        assert isinstance(nominalRotationSpeed, measurement.Measurement) or nominalRotationSpeed is None, "Attribute \"nominalRotationSpeed\" is of type \"" + str(type(nominalRotationSpeed)) + "\" but must be of type \"" + str(measurement.Measurement) + "\""
        assert isinstance(nominalStaticPressure, measurement.Measurement) or nominalStaticPressure is None, "Attribute \"nominalStaticPressure\" is of type \"" + str(type(nominalStaticPressure)) + "\" but must be of type \"" + str(measurement.Measurement) + "\""
        assert isinstance(nominalTotalPressure, measurement.Measurement) or nominalTotalPressure is None, "Attribute \"nominalTotalPressure\" is of type \"" + str(type(nominalTotalPressure)) + "\" but must be of type \"" + str(measurement.Measurement) + "\""
        assert isinstance(operationTemperatureMax, measurement.Measurement) or operationTemperatureMax is None, "Attribute \"operationTemperatureMax\" is of type \"" + str(type(operationTemperatureMax)) + "\" but must be of type \"" + str(measurement.Measurement) + "\""
        assert isinstance(operationTemperatureMin, measurement.Measurement) or operationTemperatureMin is None, "Attribute \"operationTemperatureMin\" is of type \"" + str(type(operationTemperatureMin)) + "\" but must be of type \"" + str(measurement.Measurement) + "\""
        assert isinstance(operationalRiterial, measurement.Measurement) or operationalRiterial is None, "Attribute \"operationalRiterial\" is of type \"" + str(type(operationalRiterial)) + "\" but must be of type \"" + str(measurement.Measurement) + "\""
        self.capacityControlType = capacityControlType
        self.motorDriveType = motorDriveType
        self.nominalAirFlowRate = nominalAirFlowRate
        self.nominalPowerRate = nominalPowerRate
        self.nominalRotationSpeed = nominalRotationSpeed
        self.nominalStaticPressure = nominalStaticPressure
        self.nominalTotalPressure = nominalTotalPressure
        self.operationTemperatureMax = operationTemperatureMax
        self.operationTemperatureMin = operationTemperatureMin
        self.operationalRiterial = operationalRiterial