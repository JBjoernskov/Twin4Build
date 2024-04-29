import twin4build.saref4bldg.physical_object.building_object.building_device.distribution_device.distribution_flow_device.flow_terminal.flow_terminal as flow_terminal
from typing import Union
import twin4build.saref.measurement.measurement as measurement
import twin4build.saref.property_value.property_value as property_value
import twin4build.saref.property_.s4bldg_property.s4bldg_property as s4bldg_property
import twin4build.saref.unit_of_measure.unit_of_measure as unit_of_measure
from twin4build.logger.Logging import Logging

logger = Logging.get_logger("ai_logfile")

logger.info("Space Heater FILE")

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
        logger.info("[space heater] : Entered in Initialise Function")
        super().__init__(**kwargs)

        if bodyMass is None:
            bodyMass = property_value.PropertyValue()
        if numberOfPanels is None:
            numberOfPanels = property_value.PropertyValue()
        if numberOfSections is None:
            numberOfSections = property_value.PropertyValue()
        if outputCapacity is None:
            outputCapacity = property_value.PropertyValue()
        if thermalEfficiency is None:
            thermalEfficiency = property_value.PropertyValue()
        if thermalMassHeatCapacity is None:
            thermalMassHeatCapacity = property_value.PropertyValue()

        bodyMass_ = s4bldg_property.BodyMass()
        energySource_ = s4bldg_property.EnergySource()
        heatTransferDimension_ = s4bldg_property.HeatTransferDimension()
        heatTransferMedium_ = s4bldg_property.HeatTransferMedium()
        numberOfPanels_ = s4bldg_property.NumberOfPanels()
        numberOfSections_ = s4bldg_property.NumberOfSections()
        outputCapacity_ = s4bldg_property.OutputCapacity()
        placementType_ = s4bldg_property.PlacementType()
        temperatureClassification_ = s4bldg_property.TemperatureClassification()
        thermalEfficiency_ = s4bldg_property.ThermalEfficiency()
        thermalMassHeatCapacity_ = s4bldg_property.ThermalMassHeatCapacity()

        self.hasProperty.extend([bodyMass_,
                            energySource_,
                            heatTransferDimension_,
                            heatTransferMedium_,
                            numberOfPanels_,
                            numberOfSections_,
                            outputCapacity_,
                            placementType_,
                            temperatureClassification_,
                            thermalEfficiency_,
                            thermalMassHeatCapacity_])
        
        self.hasPropertyValue.extend([property_value.PropertyValue(hasValue=bodyMass.hasValue,
                                                            isMeasuredIn=bodyMass.isMeasuredIn,
                                                            isValueOfProperty=bodyMass_),
                                property_value.PropertyValue(hasValue=energySource,
                                                            isMeasuredIn=unit_of_measure.UnitOfMeasure,
                                                            isValueOfProperty=energySource_),
                                property_value.PropertyValue(hasValue=heatTransferDimension,
                                                            isMeasuredIn=unit_of_measure.UnitOfMeasure,
                                                            isValueOfProperty=heatTransferDimension_),
                                property_value.PropertyValue(hasValue=heatTransferMedium,
                                                            isMeasuredIn=unit_of_measure.UnitOfMeasure,
                                                            isValueOfProperty=heatTransferMedium_),
                                property_value.PropertyValue(hasValue=numberOfPanels.hasValue,
                                                            isMeasuredIn=numberOfPanels.isMeasuredIn,
                                                            isValueOfProperty=numberOfPanels_),
                                property_value.PropertyValue(hasValue=numberOfSections.hasValue,
                                                            isMeasuredIn=numberOfSections.isMeasuredIn,
                                                            isValueOfProperty=numberOfSections_),
                                property_value.PropertyValue(hasValue=outputCapacity.hasValue,
                                                            isMeasuredIn=outputCapacity.isMeasuredIn,
                                                            isValueOfProperty=outputCapacity_),
                                property_value.PropertyValue(hasValue=placementType,
                                                            isMeasuredIn=unit_of_measure.UnitOfMeasure,
                                                            isValueOfProperty=placementType_),
                                property_value.PropertyValue(hasValue=temperatureClassification,
                                                            isMeasuredIn=unit_of_measure.UnitOfMeasure,
                                                            isValueOfProperty=temperatureClassification_),
                                property_value.PropertyValue(hasValue=thermalEfficiency.hasValue,
                                                            isMeasuredIn=thermalEfficiency.isMeasuredIn,
                                                            isValueOfProperty=thermalEfficiency_),
                                property_value.PropertyValue(hasValue=thermalMassHeatCapacity.hasValue,
                                                            isMeasuredIn=thermalMassHeatCapacity.isMeasuredIn,
                                                            isValueOfProperty=thermalMassHeatCapacity_)])

        

        
        # assert isinstance(bodyMass, property_value.PropertyValue) or bodyMass is None, "Attribute \"bodyMass\" is of type \"" + str(type(bodyMass)) + "\" but must be of type \"" + str(property_value.PropertyValue) + "\""
        # assert isinstance(energySource, str) or energySource is None, "Attribute \"energySource\" is of type \"" + str(type(energySource)) + "\" but must be of type \"" + str(str) + "\""
        # assert isinstance(heatTransferDimension, str) or heatTransferDimension is None, "Attribute \"heatTransferDimension\" is of type \"" + str(type(heatTransferDimension)) + "\" but must be of type \"" + str(str) + "\""
        # assert isinstance(heatTransferMedium, str) or heatTransferMedium is None, "Attribute \"heatTransferMedium\" is of type \"" + str(type(heatTransferMedium)) + "\" but must be of type \"" + str(str) + "\""
        # assert isinstance(numberOfPanels, int) or numberOfPanels is None, "Attribute \"numberOfPanels\" is of type \"" + str(type(numberOfPanels)) + "\" but must be of type \"" + str(int) + "\""
        # assert isinstance(numberOfSections, int) or numberOfSections is None, "Attribute \"numberOfSections\" is of type \"" + str(type(numberOfSections)) + "\" but must be of type \"" + str(int) + "\""
        # assert isinstance(outputCapacity, property_value.PropertyValue) or outputCapacity is None, "Attribute \"outputCapacity\" is of type \"" + str(type(outputCapacity)) + "\" but must be of type \"" + str(property_value.PropertyValue) + "\""
        # assert isinstance(placementType, str) or placementType is None, "Attribute \"placementType\" is of type \"" + str(type(placementType)) + "\" but must be of type \"" + str(str) + "\""
        # assert isinstance(temperatureClassification, str) or temperatureClassification is None, "Attribute \"temperatureClassification\" is of type \"" + str(type(temperatureClassification)) + "\" but must be of type \"" + str(str) + "\""
        # assert isinstance(thermalEfficiency, property_value.PropertyValue) or thermalEfficiency is None, "Attribute \"thermalEfficiency\" is of type \"" + str(type(thermalEfficiency)) + "\" but must be of type \"" + str(property_value.PropertyValue) + "\""
        # assert isinstance(thermalMassHeatCapacity, property_value.PropertyValue) or thermalMassHeatCapacity is None, "Attribute \"thermalMassHeatCapacity\" is of type \"" + str(type(thermalMassHeatCapacity)) + "\" but must be of type \"" + str(property_value.PropertyValue) + "\""
        # self.bodyMass = bodyMass
        # self.energySource = energySource
        # self.heatTransferDimension = heatTransferDimension
        # self.heatTransferMedium = heatTransferMedium
        # self.numberOfPanels = numberOfPanels
        # self.numberOfSections = numberOfSections
        # self.outputCapacity = outputCapacity
        # self.placementType = placementType
        # self.temperatureClassification = temperatureClassification
        # self.thermalEfficiency = thermalEfficiency
        # self.thermalMassHeatCapacity = thermalMassHeatCapacity

        logger.info("[space heater] : Exited from Initialise Function")
       

    @property
    def bodyMass(self):
        return self.hasPropertyValue[0]
    
    @property
    def energySource(self):
        return self.hasPropertyValue[1]
    
    @property
    def heatTransferDimension(self):
        return self.hasPropertyValue[2]
    
    @property
    def heatTransferMedium(self):
        return self.hasPropertyValue[3]
    
    @property
    def numberOfPanels(self):
        return self.hasPropertyValue[4]
    
    @property
    def numberOfSections(self):
        return self.hasPropertyValue[5]
    
    @property
    def outputCapacity(self):
        return self.hasPropertyValue[6]
    
    @property
    def placementType(self):
        return self.hasPropertyValue[7]
    
    @property
    def temperatureClassification(self):
        return self.hasPropertyValue[8]
    
    @property
    def thermalEfficiency(self):
        return self.hasPropertyValue[9]
    
    @property
    def thermalMassHeatCapacity(self):
        return self.hasPropertyValue[10]