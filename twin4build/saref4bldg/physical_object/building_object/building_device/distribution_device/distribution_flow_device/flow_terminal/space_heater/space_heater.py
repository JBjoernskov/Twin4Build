import twin4build.saref4bldg.physical_object.building_object.building_device.distribution_device.distribution_flow_device.flow_terminal.flow_terminal as flow_terminal
from typing import Union
import twin4build.saref.property_value.property_value as property_value
import twin4build.saref.property_.s4bldg_property.s4bldg_property as s4bldg_property

class SpaceHeater(flow_terminal.FlowTerminal):
    def __init__(self,
                bodyMass: Union[property_value.PropertyValue, None] = None, 
                energySource: Union[str, None] = None, 
                heatTransferDimension: Union[str, None] = None, 
                heatTransferMedium: Union[str, None] = None, 
                numberOfPanels: Union[property_value.PropertyValue, None] = None, 
                numberOfSections: Union[property_value.PropertyValue, None] = None, 
                outputCapacity: Union[property_value.PropertyValue, None] = None, 
                placementType: Union[str, None] = None, 
                temperatureClassification: Union[str, None] = None, 
                thermalEfficiency: Union[property_value.PropertyValue, None] = None, 
                thermalMassHeatCapacity: Union[property_value.PropertyValue, None] = None, 
                **kwargs):
        super().__init__(**kwargs)

        bodyMass_ = s4bldg_property.AirFlowRateMax()
        if bodyMass is not None:
            bodyMass = property_value.PropertyValue(hasValue=bodyMass.hasValue,
                                                            isMeasuredIn=bodyMass.isMeasuredIn,
                                                            isValueOfProperty=bodyMass_)
        else:
            bodyMass = property_value.PropertyValue(isValueOfProperty=bodyMass_)
        self.hasProperty.append(bodyMass_)
        self.hasPropertyValue.append(bodyMass)

        energySource_ = s4bldg_property.EnergySource()
        if energySource is not None:
            energySource = property_value.PropertyValue(hasValue=energySource.hasValue,
                                isMeasuredIn=energySource.isMeasuredIn,
                                isValueOfProperty=energySource_)
        else:
            energySource = property_value.PropertyValue(isValueOfProperty=energySource_)
        self.hasProperty.append(energySource_)
        self.hasPropertyValue.append(energySource)

        heatTransferDimension_ = s4bldg_property.HeatTransferDimension()
        if heatTransferDimension is not None:
            heatTransferDimension = property_value.PropertyValue(hasValue=heatTransferDimension.hasValue,
                                     isMeasuredIn=heatTransferDimension.isMeasuredIn,
                                     isValueOfProperty=heatTransferDimension_)
        else:
            heatTransferDimension = property_value.PropertyValue(isValueOfProperty=heatTransferDimension_)
        self.hasProperty.append(heatTransferDimension_)
        self.hasPropertyValue.append(heatTransferDimension)

        heatTransferMedium_ = s4bldg_property.HeatTransferMedium()
        if heatTransferMedium is not None:
            heatTransferMedium = property_value.PropertyValue(hasValue=heatTransferMedium.hasValue,
                                      isMeasuredIn=heatTransferMedium.isMeasuredIn,
                                      isValueOfProperty=heatTransferMedium_)
        else:
            heatTransferMedium = property_value.PropertyValue(isValueOfProperty=heatTransferMedium_)
        self.hasProperty.append(heatTransferMedium_)
        self.hasPropertyValue.append(heatTransferMedium)

        numberOfPanels_ = s4bldg_property.NumberOfPanels()
        if numberOfPanels is not None:
            numberOfPanels = property_value.PropertyValue(hasValue=numberOfPanels.hasValue,
                                  isMeasuredIn=numberOfPanels.isMeasuredIn,
                                  isValueOfProperty=numberOfPanels_)
        else:
            numberOfPanels = property_value.PropertyValue(isValueOfProperty=numberOfPanels_)
        self.hasProperty.append(numberOfPanels_)
        self.hasPropertyValue.append(numberOfPanels)

        numberOfSections_ = s4bldg_property.NumberOfSections()
        if numberOfSections is not None:
            numberOfSections = property_value.PropertyValue(hasValue=numberOfSections.hasValue,
                                    isMeasuredIn=numberOfSections.isMeasuredIn,
                                    isValueOfProperty=numberOfSections_)
        else:
            numberOfSections = property_value.PropertyValue(isValueOfProperty=numberOfSections_)
        self.hasProperty.append(numberOfSections_)
        self.hasPropertyValue.append(numberOfSections)

        outputCapacity_ = s4bldg_property.OutputCapacity()
        if outputCapacity is not None:
            outputCapacity = property_value.PropertyValue(hasValue=outputCapacity.hasValue,
                                  isMeasuredIn=outputCapacity.isMeasuredIn,
                                  isValueOfProperty=outputCapacity_)
        else:
            outputCapacity = property_value.PropertyValue(isValueOfProperty=outputCapacity_)
        self.hasProperty.append(outputCapacity_)
        self.hasPropertyValue.append(outputCapacity)

        placementType_ = s4bldg_property.PlacementType()
        if placementType is not None:
            placementType = property_value.PropertyValue(hasValue=placementType.hasValue,
                                 isMeasuredIn=placementType.isMeasuredIn,
                                 isValueOfProperty=placementType_)
        else:
            placementType = property_value.PropertyValue(isValueOfProperty=placementType_)
        self.hasProperty.append(placementType_)
        self.hasPropertyValue.append(placementType)

        temperatureClassification_ = s4bldg_property.TemperatureClassification()
        if temperatureClassification is not None:
            temperatureClassification = property_value.PropertyValue(hasValue=temperatureClassification.hasValue,
                                         isMeasuredIn=temperatureClassification.isMeasuredIn,
                                         isValueOfProperty=temperatureClassification_)
        else:
            temperatureClassification = property_value.PropertyValue(isValueOfProperty=temperatureClassification_)
        self.hasProperty.append(temperatureClassification_)
        self.hasPropertyValue.append(temperatureClassification)

        thermalEfficiency_ = s4bldg_property.ThermalEfficiency()
        if thermalEfficiency is not None:
            thermalEfficiency = property_value.PropertyValue(hasValue=thermalEfficiency.hasValue,
                                     isMeasuredIn=thermalEfficiency.isMeasuredIn,
                                     isValueOfProperty=thermalEfficiency_)
        else:
            thermalEfficiency = property_value.PropertyValue(isValueOfProperty=thermalEfficiency_)
        self.hasProperty.append(thermalEfficiency_)
        self.hasPropertyValue.append(thermalEfficiency)

        thermalMassHeatCapacity_ = s4bldg_property.ThermalMassHeatCapacity()
        if thermalMassHeatCapacity is not None:
            thermalMassHeatCapacity = property_value.PropertyValue(hasValue=thermalMassHeatCapacity.hasValue,
                                       isMeasuredIn=thermalMassHeatCapacity.isMeasuredIn,
                                       isValueOfProperty=thermalMassHeatCapacity_)
        else:
            thermalMassHeatCapacity = property_value.PropertyValue(isValueOfProperty=thermalMassHeatCapacity_)
        self.hasProperty.append(thermalMassHeatCapacity_)
        self.hasPropertyValue.append(thermalMassHeatCapacity)

    @property
    def bodyMass(self):
        el = [el for el in self.hasPropertyValue if isinstance(el.isValueOfProperty, s4bldg_property.BodyMass)]
        return el[0] if len(el) > 0 else None
    
    @property
    def energySource(self):
        el = [el for el in self.hasPropertyValue if isinstance(el.isValueOfProperty, s4bldg_property.EnergySource)]
        return el[0] if len(el) > 0 else None

    @property
    def heatTransferDimension(self):
        el = [el for el in self.hasPropertyValue if isinstance(el.isValueOfProperty, s4bldg_property.HeatTransferDimension)]
        return el[0] if len(el) > 0 else None

    @property
    def heatTransferMedium(self):
        el = [el for el in self.hasPropertyValue if isinstance(el.isValueOfProperty, s4bldg_property.HeatTransferMedium)]
        return el[0] if len(el) > 0 else None

    @property
    def numberOfPanels(self):
        el = [el for el in self.hasPropertyValue if isinstance(el.isValueOfProperty, s4bldg_property.NumberOfPanels)]
        return el[0] if len(el) > 0 else None

    @property
    def numberOfSections(self):
        el = [el for el in self.hasPropertyValue if isinstance(el.isValueOfProperty, s4bldg_property.NumberOfSections)]
        return el[0] if len(el) > 0 else None

    @property
    def outputCapacity(self):
        el = [el for el in self.hasPropertyValue if isinstance(el.isValueOfProperty, s4bldg_property.OutputCapacity)]
        return el[0] if len(el) > 0 else None

    @property
    def placementType(self):
        el = [el for el in self.hasPropertyValue if isinstance(el.isValueOfProperty, s4bldg_property.PlacementType)]
        return el[0] if len(el) > 0 else None

    @property
    def temperatureClassification(self):
        el = [el for el in self.hasPropertyValue if isinstance(el.isValueOfProperty, s4bldg_property.TemperatureClassification)]
        return el[0] if len(el) > 0 else None

    @property
    def thermalEfficiency(self):
        el = [el for el in self.hasPropertyValue if isinstance(el.isValueOfProperty, s4bldg_property.ThermalEfficiency)]
        return el[0] if len(el) > 0 else None

    @property
    def thermalMassHeatCapacity(self):
        el = [el for el in self.hasPropertyValue if isinstance(el.isValueOfProperty, s4bldg_property.ThermalMassHeatCapacity)]
        return el[0] if len(el) > 0 else None