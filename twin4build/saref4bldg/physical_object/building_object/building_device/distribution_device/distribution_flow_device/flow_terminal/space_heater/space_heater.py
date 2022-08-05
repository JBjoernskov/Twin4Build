import twin4build.saref4bldg.physical_object.building_object.building_device.distribution_device.distribution_flow_device.flow_terminal.flow_terminal as flow_terminal
from typing import Union
import twin4build.saref.measurement.measurement as measurement
class SpaceHeater(flow_terminal.FlowTerminal):
    def __init__(self,
                bodyMass: Union[measurement.Measurement, None] = None, 
                energySource: Union[str, None] = None, 
                heatTransferDimension: Union[str, None] = None, 
                heatTransferMedium: Union[str, None] = None, 
                numberOfPanels: Union[measurement.Measurement, None] = None, 
                numberOfSections: Union[measurement.Measurement, None] = None, 
                outputCapacity: Union[measurement.Measurement, None] = None, 
                placementType: Union[str, None] = None, 
                temperatureClassification: Union[str, None] = None, 
                thermalEfficiency: Union[measurement.Measurement, None] = None, 
                thermalMassHeatCapacity: Union[measurement.Measurement, None] = None, 
                **kwargs):
        super().__init__(**kwargs)
        assert isinstance(bodyMass, measurement.Measurement) or bodyMass is None, "Attribute \"bodyMass\" is of type \"" + str(type(bodyMass)) + "\" but must be of type \"" + str(measurement.Measurement) + "\""
        assert isinstance(energySource, str) or energySource is None, "Attribute \"energySource\" is of type \"" + str(type(energySource)) + "\" but must be of type \"" + str(str) + "\""
        assert isinstance(heatTransferDimension, str) or heatTransferDimension is None, "Attribute \"heatTransferDimension\" is of type \"" + str(type(heatTransferDimension)) + "\" but must be of type \"" + str(str) + "\""
        assert isinstance(heatTransferMedium, str) or heatTransferMedium is None, "Attribute \"heatTransferMedium\" is of type \"" + str(type(heatTransferMedium)) + "\" but must be of type \"" + str(str) + "\""
        assert isinstance(numberOfPanels, int) or numberOfPanels is None, "Attribute \"numberOfPanels\" is of type \"" + str(type(numberOfPanels)) + "\" but must be of type \"" + str(int) + "\""
        assert isinstance(numberOfSections, int) or numberOfSections is None, "Attribute \"numberOfSections\" is of type \"" + str(type(numberOfSections)) + "\" but must be of type \"" + str(int) + "\""
        assert isinstance(outputCapacity, measurement.Measurement) or outputCapacity is None, "Attribute \"outputCapacity\" is of type \"" + str(type(outputCapacity)) + "\" but must be of type \"" + str(measurement.Measurement) + "\""
        assert isinstance(placementType, str) or placementType is None, "Attribute \"placementType\" is of type \"" + str(type(placementType)) + "\" but must be of type \"" + str(str) + "\""
        assert isinstance(temperatureClassification, str) or temperatureClassification is None, "Attribute \"temperatureClassification\" is of type \"" + str(type(temperatureClassification)) + "\" but must be of type \"" + str(str) + "\""
        assert isinstance(thermalEfficiency, measurement.Measurement) or thermalEfficiency is None, "Attribute \"thermalEfficiency\" is of type \"" + str(type(thermalEfficiency)) + "\" but must be of type \"" + str(measurement.Measurement) + "\""
        assert isinstance(thermalMassHeatCapacity, measurement.Measurement) or thermalMassHeatCapacity is None, "Attribute \"thermalMassHeatCapacity\" is of type \"" + str(type(thermalMassHeatCapacity)) + "\" but must be of type \"" + str(measurement.Measurement) + "\""
        self.bodyMass = bodyMass
        self.energySource = energySource
        self.heatTransferDimension = heatTransferDimension
        self.heatTransferMedium = heatTransferMedium
        self.numberOfPanels = numberOfPanels
        self.numberOfSections = numberOfSections
        self.outputCapacity = outputCapacity
        self.placementType = placementType
        self.temperatureClassification = temperatureClassification
        self.thermalEfficiency = thermalEfficiency
        self.thermalMassHeatCapacity = thermalMassHeatCapacity
        