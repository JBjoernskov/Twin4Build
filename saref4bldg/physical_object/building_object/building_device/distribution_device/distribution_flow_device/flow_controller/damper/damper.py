from ..flow_controller import FlowController
class Damper(FlowController):
    def __init__(self,
                airFlowRateMax = None,
                bladeAction = None,
                bladeEdge = None,
                bladeShape = None,
                bladeThickness = None,
                closeOffRating = None,
                faceArea = None,
                frameDepth = None,
                frameThickness = None,
                frameType = None,
                leakageFullyClosed = None,
                nominalAirFlowRate = None,
                numberOfBlades = None,
                openPressureDrop = None,
                operation = None,
                operationTemperatureMax = None,
                operationTemperatureMin = None,
                orientation = None,
                temperatureRating = None,
                workingPressureMax = None,
                **kwargs):
        super().__init__(**kwargs)
        self.airFlowRateMax = airFlowRateMax
        self.bladeAction = bladeAction
        self.bladeEdge = bladeEdge
        self.bladeShape = bladeShape
        self.bladeThickness = bladeThickness
        self.closeOffRating = closeOffRating
        self.faceArea = faceArea
        self.frameDepth = frameDepth
        self.frameThickness = frameThickness
        self.frameType = frameType
        self.leakageFullyClosed = leakageFullyClosed
        self.nominalAirFlowRate = nominalAirFlowRate
        self.numberOfBlades = numberOfBlades
        self.openPressureDrop = openPressureDrop
        self.operation = operation
        self.operationTemperatureMax = operationTemperatureMax
        self.operationTemperatureMin = operationTemperatureMin
        self.orientation = orientation
        self.temperatureRating = temperatureRating
        self.workingPressureMax = workingPressureMax        
        