import twin4build.saref4bldg.physical_object.building_object.building_device.distribution_device.distribution_flow_device.energy_conversion_device.energy_conversion_device as energy_conversion_device
from typing import Union
import twin4build.saref.property_value.property_value as property_value
import twin4build.saref.property_.s4bldg_property.s4bldg_property as s4bldg_property
import twin4build.saref.unit_of_measure.unit_of_measure as unit_of_measure
import twin4build.saref.measurement.measurement as measurement
from twin4build.logger.Logging import Logging
logger = Logging.get_logger("ai_logfile")


class Coil(energy_conversion_device.EnergyConversionDevice):
    def __init__(self,
                airFlowRateMax: Union[property_value.PropertyValue, None] = None,
                airFlowRateMin: Union[property_value.PropertyValue, None] = None,
                nominalLatentCapacity: Union[property_value.PropertyValue, None] = None,
                nominalSensibleCapacity: Union[property_value.PropertyValue, None] = None,
                nominalUa: Union[property_value.PropertyValue, None] = None,
                operationTemperatureMax: Union[property_value.PropertyValue, None] = None,
                operationTemperatureMin: Union[property_value.PropertyValue, None] = None,
                placementType: Union[str, None] = None,
                **kwargs):
        
        logger.info("[Coil class] : Entered in Initialise Function")
        super().__init__(**kwargs)
        airFlowRateMax_ = s4bldg_property.AirFlowRateMax()
        if airFlowRateMax is not None:
            airFlowRateMax = property_value.PropertyValue(hasValue=airFlowRateMax.hasValue,
                                                            isMeasuredIn=airFlowRateMax.isMeasuredIn,
                                                            isValueOfProperty=airFlowRateMax_)
        else:
            airFlowRateMax = property_value.PropertyValue(isValueOfProperty=airFlowRateMax_)
        self.hasProperty.append(airFlowRateMax_)
        self.hasPropertyValue.append(airFlowRateMax)

        airFlowRateMin_ = s4bldg_property.AirFlowRateMin()
        if airFlowRateMin is not None:
            airFlowRateMin = property_value.PropertyValue(hasValue=airFlowRateMin.hasValue,
                                                          isMeasuredIn=airFlowRateMin.isMeasuredIn,
                                                          isValueOfProperty=airFlowRateMin_)
        else:
            airFlowRateMin = property_value.PropertyValue(isValueOfProperty=airFlowRateMin_)
        self.hasProperty.append(airFlowRateMin_)
        self.hasPropertyValue.append(airFlowRateMin)

        nominalLatentCapacity_ = s4bldg_property.NominalLatentCapacity()
        if nominalLatentCapacity is not None:
            nominalLatentCapacity = property_value.PropertyValue(hasValue=nominalLatentCapacity.hasValue,
                                                                 isMeasuredIn=nominalLatentCapacity.isMeasuredIn,
                                                                 isValueOfProperty=nominalLatentCapacity_)
        else:
            nominalLatentCapacity = property_value.PropertyValue(isValueOfProperty=nominalLatentCapacity_)
        self.hasProperty.append(nominalLatentCapacity_)
        self.hasPropertyValue.append(nominalLatentCapacity)

        nominalSensibleCapacity_ = s4bldg_property.NominalSensibleCapacity()
        if nominalSensibleCapacity is not None:
            nominalSensibleCapacity = property_value.PropertyValue(hasValue=nominalSensibleCapacity.hasValue,
                                                                   isMeasuredIn=nominalSensibleCapacity.isMeasuredIn,
                                                                   isValueOfProperty=nominalSensibleCapacity_)
        else:
            nominalSensibleCapacity = property_value.PropertyValue(isValueOfProperty=nominalSensibleCapacity_)
        self.hasProperty.append(nominalSensibleCapacity_)
        self.hasPropertyValue.append(nominalSensibleCapacity)

        nominalUa_ = s4bldg_property.NominalUa()
        if nominalUa is not None:
            nominalUa = property_value.PropertyValue(hasValue=nominalUa.hasValue,
                                                     isMeasuredIn=nominalUa.isMeasuredIn,
                                                     isValueOfProperty=nominalUa_)
        else:
            nominalUa = property_value.PropertyValue(isValueOfProperty=nominalUa_)
        self.hasProperty.append(nominalUa_)
        self.hasPropertyValue.append(nominalUa)

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

        placementType_ = s4bldg_property.PlacementType()
        if placementType is not None:
            placementType = property_value.PropertyValue(hasValue=placementType.hasValue,
                                                         isMeasuredIn=placementType.isMeasuredIn,
                                                         isValueOfProperty=placementType_)
        else:
            placementType = property_value.PropertyValue(isValueOfProperty=placementType_)
        self.hasProperty.append(placementType_)
        self.hasPropertyValue.append(placementType)


        logger.info("[Coil class] : Exited from Initialise Function")

    @property
    def airFlowRateMax(self):
        el = [el for el in self.hasPropertyValue if isinstance(el.isValueOfProperty, s4bldg_property.AirFlowRateMax)]
        return el[0] if len(el) > 0 else None

    @property
    def airFlowRateMin(self):
        el = [el for el in self.hasPropertyValue if isinstance(el.isValueOfProperty, s4bldg_property.AirFlowRateMin)]
        return el[0] if len(el) > 0 else None

    @property
    def nominalLatentCapacity(self):
        el = [el for el in self.hasPropertyValue if isinstance(el.isValueOfProperty, s4bldg_property.NominalLatentCapacity)]
        return el[0] if len(el) > 0 else None

    @property
    def nominalSensibleCapacity(self):
        el = [el for el in self.hasPropertyValue if isinstance(el.isValueOfProperty, s4bldg_property.NominalSensibleCapacity)]
        return el[0] if len(el) > 0 else None

    @property
    def nominalUa(self):
        el = [el for el in self.hasPropertyValue if isinstance(el.isValueOfProperty, s4bldg_property.NominalUa)]
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
    def placementType(self):
        el = [el for el in self.hasPropertyValue if isinstance(el.isValueOfProperty, s4bldg_property.PlacementType)]
        return el[0] if len(el) > 0 else None

