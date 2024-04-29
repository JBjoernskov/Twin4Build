import twin4build.saref4bldg.physical_object.building_object.building_device.distribution_device.distribution_flow_device.flow_moving_device.flow_moving_device as flow_moving_device
from typing import Union
import twin4build.saref.measurement.measurement as measurement
import twin4build.saref.property_value.property_value as property_value
import twin4build.saref.property_.s4bldg_property.s4bldg_property as s4bldg_property
import twin4build.saref.unit_of_measure.unit_of_measure as unit_of_measure
from twin4build.logger.Logging import Logging

logger = Logging.get_logger("ai_logfile")

class Fan(flow_moving_device.FlowMovingDevice):
    def __init__(self,
                capacityControlType: Union[str, None] = None,
                motorDriveType: Union[str, None] = None,
                nominalAirFlowRate: Union[property_value.PropertyValue, None] = None,
                nominalPowerRate: Union[property_value.PropertyValue, None] = None,
                nominalRotationSpeed: Union[property_value.PropertyValue, None] = None,
                nominalStaticPressure: Union[property_value.PropertyValue, None] = None,
                nominalTotalPressure: Union[property_value.PropertyValue, None] = None,
                operationTemperatureMax: Union[property_value.PropertyValue, None] = None,
                operationTemperatureMin: Union[property_value.PropertyValue, None] = None,
                operationalRiterial: Union[property_value.PropertyValue, None] = None,
                **kwargs):
        logger.info("[fan class] : Entered in Initialise Function")
        super().__init__(**kwargs)

        capacityControlType_ = s4bldg_property.CapacityControlType()
        if capacityControlType is not None:
            capacityControlType = property_value.PropertyValue(hasValue=capacityControlType.hasValue,
                                                                isMeasuredIn=capacityControlType.isMeasuredIn,
                                                                isValueOfProperty=capacityControlType_)
        else:
            capacityControlType = property_value.PropertyValue(isValueOfProperty=capacityControlType_)
        self.hasProperty.append(capacityControlType_)
        self.hasPropertyValue.append(capacityControlType)

        motorDriveType_ = s4bldg_property.MotorDriveType()
        if motorDriveType is not None:
            motorDriveType = property_value.PropertyValue(hasValue=motorDriveType.hasValue,
                                                        isMeasuredIn=motorDriveType.isMeasuredIn,
                                                        isValueOfProperty=motorDriveType_)
        else:
            motorDriveType = property_value.PropertyValue(isValueOfProperty=motorDriveType_)
        self.hasProperty.append(motorDriveType_)
        self.hasPropertyValue.append(motorDriveType)

        nominalAirFlowRate_ = s4bldg_property.NominalAirFlowRate()
        if nominalAirFlowRate is not None:
            nominalAirFlowRate = property_value.PropertyValue(hasValue=nominalAirFlowRate.hasValue,
                                                            isMeasuredIn=nominalAirFlowRate.isMeasuredIn,
                                                            isValueOfProperty=nominalAirFlowRate_)
        else:
            nominalAirFlowRate = property_value.PropertyValue(isValueOfProperty=nominalAirFlowRate_)
        self.hasProperty.append(nominalAirFlowRate_)
        self.hasPropertyValue.append(nominalAirFlowRate)

        nominalPowerRate_ = s4bldg_property.NominalPowerRate()
        if nominalPowerRate is not None:
            nominalPowerRate = property_value.PropertyValue(hasValue=nominalPowerRate.hasValue,
                                                            isMeasuredIn=nominalPowerRate.isMeasuredIn,
                                                            isValueOfProperty=nominalPowerRate_)
        else:
            nominalPowerRate = property_value.PropertyValue(isValueOfProperty=nominalPowerRate_)
        self.hasProperty.append(nominalPowerRate_)
        self.hasPropertyValue.append(nominalPowerRate)

        nominalRotationSpeed_ = s4bldg_property.NominalRotationSpeed()
        if nominalRotationSpeed is not None:
            nominalRotationSpeed = property_value.PropertyValue(hasValue=nominalRotationSpeed.hasValue,
                                                                isMeasuredIn=nominalRotationSpeed.isMeasuredIn,
                                                                isValueOfProperty=nominalRotationSpeed_)
        else:
            nominalRotationSpeed = property_value.PropertyValue(isValueOfProperty=nominalRotationSpeed_)
        self.hasProperty.append(nominalRotationSpeed_)
        self.hasPropertyValue.append(nominalRotationSpeed)

        nominalStaticPressure_ = s4bldg_property.NominalStaticPressure()
        if nominalStaticPressure is not None:
            nominalStaticPressure = property_value.PropertyValue(hasValue=nominalStaticPressure.hasValue,
                                                                 isMeasuredIn=nominalStaticPressure.isMeasuredIn,
                                                                 isValueOfProperty=nominalStaticPressure_)
        else:
            nominalStaticPressure = property_value.PropertyValue(isValueOfProperty=nominalStaticPressure_)
        self.hasProperty.append(nominalStaticPressure_)
        self.hasPropertyValue.append(nominalStaticPressure)

        nominalTotalPressure_ = s4bldg_property.NominalTotalPressure()
        if nominalTotalPressure is not None:
            nominalTotalPressure = property_value.PropertyValue(hasValue=nominalTotalPressure.hasValue,
                                                                isMeasuredIn=nominalTotalPressure.isMeasuredIn,
                                                                isValueOfProperty=nominalTotalPressure_)
        else:
            nominalTotalPressure = property_value.PropertyValue(isValueOfProperty=nominalTotalPressure_)
        self.hasProperty.append(nominalTotalPressure_)
        self.hasPropertyValue.append(nominalTotalPressure)

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

        operationalRiterial_ = s4bldg_property.OperationalRiterial()
        if operationalRiterial is not None:
            operationalRiterial = property_value.PropertyValue(hasValue=operationalRiterial.hasValue,
                                                               isMeasuredIn=operationalRiterial.isMeasuredIn,
                                                               isValueOfProperty=operationalRiterial_)
        else:
            operationalRiterial = property_value.PropertyValue(isValueOfProperty=operationalRiterial_)
        self.hasProperty.append(operationalRiterial_)
        self.hasPropertyValue.append(operationalRiterial)


        logger.info("[fan class] : Exited from Initialise Function")

    @property
    def capacityControlType(self):
        el = [el for el in self.hasPropertyValue if isinstance(el.isValueOfProperty, s4bldg_property.CapacityControlType)]
        return el[0] if len(el) > 0 else None
    
    @property
    def motorDriveType(self):
        el = [el for el in self.hasPropertyValue if isinstance(el.isValueOfProperty, s4bldg_property.MotorDriveType)]
        return el[0] if len(el) > 0 else None

    @property
    def nominalAirFlowRate(self):
        el = [el for el in self.hasPropertyValue if isinstance(el.isValueOfProperty, s4bldg_property.NominalAirFlowRate)]
        return el[0] if len(el) > 0 else None

    @property
    def nominalPowerRate(self):
        el = [el for el in self.hasPropertyValue if isinstance(el.isValueOfProperty, s4bldg_property.NominalPowerRate)]
        return el[0] if len(el) > 0 else None

    @property
    def nominalRotationSpeed(self):
        el = [el for el in self.hasPropertyValue if isinstance(el.isValueOfProperty, s4bldg_property.NominalRotationSpeed)]
        return el[0] if len(el) > 0 else None

    @property
    def nominalStaticPressure(self):
        el = [el for el in self.hasPropertyValue if isinstance(el.isValueOfProperty, s4bldg_property.NominalStaticPressure)]
        return el[0] if len(el) > 0 else None

    @property
    def nominalTotalPressure(self):
        el = [el for el in self.hasPropertyValue if isinstance(el.isValueOfProperty, s4bldg_property.NominalTotalPressure)]
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
    def operationalRiterial(self):
        el = [el for el in self.hasPropertyValue if isinstance(el.isValueOfProperty, s4bldg_property.OperationalRiterial)]
        return el[0] if len(el) > 0 else None