from ..flow_terminal import FlowTerminal
class SpaceHeater(FlowTerminal):
    def __init__(self,
                bodyMass = None, 
                energySource = None, 
                heatTransferDimension = None, 
                heatTransferMedium = None, 
                numberOfPanels = None, 
                numberOfSections = None, 
                outputCapacity = None, 
                placementType = None, 
                temperatureClassification = None, 
                thermalEfficiency = None, 
                thermalMassHeatCapacity = None, 
                **kwargs):
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
        super().__init__(**kwargs)