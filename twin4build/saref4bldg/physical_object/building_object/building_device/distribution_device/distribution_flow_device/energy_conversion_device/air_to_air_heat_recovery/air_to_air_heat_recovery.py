import twin4build.saref4bldg.physical_object.building_object.building_device.distribution_device.distribution_flow_device.energy_conversion_device.energy_conversion_device as energy_conversion_device
import twin4build.saref.property_value.property_value as property_value
import twin4build.saref.property_.s4bldg_property.s4bldg_property as s4bldg_property
import twin4build.saref.unit_of_measure.unit_of_measure as unit_of_measure
from typing import Union


class AirToAirHeatRecovery(energy_conversion_device.EnergyConversionDevice):
    def __init__(self,
                hasDefrost: Union[bool, None] = None,
                heatTransferTypeEnum: Union[str, None] = None,
                operationTemperatureMax: Union[property_value.PropertyValue, None] = None,
                operationTemperatureMin: Union[property_value.PropertyValue, None]  = None,
                primaryAirFlowRateMax: Union[property_value.PropertyValue, None]  = None,
                primaryAirFlowRateMin: Union[property_value.PropertyValue, None]  = None,
                secondaryAirFlowRateMax: Union[property_value.PropertyValue, None]  = None,
                secondaryAirFlowRateMin: Union[property_value.PropertyValue, None]  = None,
                **kwargs):
        super().__init__(**kwargs)

        hasDefrost_ = s4bldg_property.HasDefrost()
        if hasDefrost is not None:
            hasDefrost = property_value.PropertyValue(hasValue=hasDefrost.hasValue,
                                                            isMeasuredIn=hasDefrost.isMeasuredIn,
                                                            isValueOfProperty=hasDefrost_)
        else:
            hasDefrost = property_value.PropertyValue(isValueOfProperty=hasDefrost_)
        self.hasProperty.append(hasDefrost_)
        self.hasPropertyValue.append(hasDefrost)

        heatTransferTypeEnum_ = s4bldg_property.HeatTransferTypeEnum()
        if heatTransferTypeEnum is not None:
            heatTransferTypeEnum = property_value.PropertyValue(hasValue=heatTransferTypeEnum.hasValue,
                                                                isMeasuredIn=heatTransferTypeEnum.isMeasuredIn,
                                                                isValueOfProperty=heatTransferTypeEnum_)
        else:
            heatTransferTypeEnum = property_value.PropertyValue(isValueOfProperty=heatTransferTypeEnum_)
        self.hasProperty.append(heatTransferTypeEnum_)
        self.hasPropertyValue.append(heatTransferTypeEnum)

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

        primaryAirFlowRateMax_ = s4bldg_property.PrimaryAirFlowRateMax()
        if primaryAirFlowRateMax is not None:
            primaryAirFlowRateMax = property_value.PropertyValue(hasValue=primaryAirFlowRateMax.hasValue,
                                                                 isMeasuredIn=primaryAirFlowRateMax.isMeasuredIn,
                                                                 isValueOfProperty=primaryAirFlowRateMax_)
        else:
            primaryAirFlowRateMax = property_value.PropertyValue(isValueOfProperty=primaryAirFlowRateMax_)
        self.hasProperty.append(primaryAirFlowRateMax_)
        self.hasPropertyValue.append(primaryAirFlowRateMax)

        primaryAirFlowRateMin_ = s4bldg_property.PrimaryAirFlowRateMin()
        if primaryAirFlowRateMin is not None:
            primaryAirFlowRateMin = property_value.PropertyValue(hasValue=primaryAirFlowRateMin.hasValue,
                                                                 isMeasuredIn=primaryAirFlowRateMin.isMeasuredIn,
                                                                 isValueOfProperty=primaryAirFlowRateMin_)
        else:
            primaryAirFlowRateMin = property_value.PropertyValue(isValueOfProperty=primaryAirFlowRateMin_)
        self.hasProperty.append(primaryAirFlowRateMin_)
        self.hasPropertyValue.append(primaryAirFlowRateMin)

        secondaryAirFlowRateMax_ = s4bldg_property.SecondaryAirFlowRateMax()
        if secondaryAirFlowRateMax is not None:
            secondaryAirFlowRateMax = property_value.PropertyValue(hasValue=secondaryAirFlowRateMax.hasValue,
                                                                   isMeasuredIn=secondaryAirFlowRateMax.isMeasuredIn,
                                                                   isValueOfProperty=secondaryAirFlowRateMax_)
        else:
            secondaryAirFlowRateMax = property_value.PropertyValue(isValueOfProperty=secondaryAirFlowRateMax_)
        self.hasProperty.append(secondaryAirFlowRateMax_)
        self.hasPropertyValue.append(secondaryAirFlowRateMax)

        secondaryAirFlowRateMin_ = s4bldg_property.SecondaryAirFlowRateMin()
        if secondaryAirFlowRateMin is not None:
            secondaryAirFlowRateMin = property_value.PropertyValue(hasValue=secondaryAirFlowRateMin.hasValue,
                                                                   isMeasuredIn=secondaryAirFlowRateMin.isMeasuredIn,
                                                                   isValueOfProperty=secondaryAirFlowRateMin_)
        else:
            secondaryAirFlowRateMin = property_value.PropertyValue(isValueOfProperty=secondaryAirFlowRateMin_)
        self.hasProperty.append(secondaryAirFlowRateMin_)
        self.hasPropertyValue.append(secondaryAirFlowRateMin)


    @property
    def hasDefrost(self):
        el = [el for el in self.hasPropertyValue if isinstance(el.isValueOfProperty, s4bldg_property.HasDefrost)]
        return el[0] if len(el) > 0 else None

    @property
    def heatTransferTypeEnum(self):
        el = [el for el in self.hasPropertyValue if isinstance(el.isValueOfProperty, s4bldg_property.HeatTransferTypeEnum)]
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
    def primaryAirFlowRateMax(self):
        el = [el for el in self.hasPropertyValue if isinstance(el.isValueOfProperty, s4bldg_property.PrimaryAirFlowRateMax)]
        return el[0] if len(el) > 0 else None

    @property
    def primaryAirFlowRateMin(self):
        el = [el for el in self.hasPropertyValue if isinstance(el.isValueOfProperty, s4bldg_property.PrimaryAirFlowRateMin)]
        return el[0] if len(el) > 0 else None

    @property
    def secondaryAirFlowRateMax(self):
        el = [el for el in self.hasPropertyValue if isinstance(el.isValueOfProperty, s4bldg_property.SecondaryAirFlowRateMax)]
        return el[0] if len(el) > 0 else None

    @property
    def secondaryAirFlowRateMin(self):
        el = [el for el in self.hasPropertyValue if isinstance(el.isValueOfProperty, s4bldg_property.SecondaryAirFlowRateMin)]
        return el[0] if len(el) > 0 else None
    



    # @property
    # def hasDefrost(self):
    #     return self.hasPropertyValue[0]
    
    # @property
    # def heatTransferTypeEnum(self):
    #     return self.hasPropertyValue[1]
    
    # @property
    # def operationTemperatureMax(self):
    #     return self.hasPropertyValue[2]

    # @property
    # def operationTemperatureMin(self):
    #     return self.hasPropertyValue[3]

    # @property
    # def primaryAirFlowRateMax(self):
    #     return self.hasPropertyValue[4]

    # @property
    # def primaryAirFlowRateMin(self):
    #     return self.hasPropertyValue[5]

    # @property
    # def secondaryAirFlowRateMax(self):
    #     return self.hasPropertyValue[6]

    # @property
    # def secondaryAirFlowRateMin(self):
    #     return self.hasPropertyValue[7]