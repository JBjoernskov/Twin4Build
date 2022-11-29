import twin4build.saref4bldg.physical_object.building_object.building_device.distribution_device.distribution_flow_device.flow_controller.flow_controller as flow_controller
from typing import Union
import twin4build.saref.measurement.measurement as measurement
class Damper(flow_controller.FlowController):
    def __init__(self,
                airFlowRateMax: Union[measurement.Measurement, None] = None,
                bladeAction: Union[str, None] = None,
                bladeEdge: Union[str, None] = None,
                bladeShape: Union[str, None] = None,
                bladeThickness: Union[measurement.Measurement, None] = None,
                closeOffRating: Union[measurement.Measurement, None] = None,
                faceArea: Union[measurement.Measurement, None] = None,
                frameDepth: Union[measurement.Measurement, None] = None,
                frameThickness: Union[measurement.Measurement, None] = None,
                frameType: Union[str, None] = None,
                leakageFullyClosed: Union[measurement.Measurement, None] = None,
                nominalAirFlowRate: Union[measurement.Measurement, None] = None,
                numberOfBlades: Union[measurement.Measurement, None] = None,
                openPressureDrop: Union[measurement.Measurement, None] = None,
                operation: Union[str, None] = None,
                operationTemperatureMax: Union[measurement.Measurement, None] = None,
                operationTemperatureMin: Union[measurement.Measurement, None] = None,
                orientation: Union[str, None] = None,
                temperatureRating: Union[measurement.Measurement, None] = None,
                workingPressureMax: Union[measurement.Measurement, None] = None,
                operationMode: Union[str, None] = None,
                **kwargs):
        super().__init__(**kwargs)
        assert isinstance(airFlowRateMax, measurement.Measurement) or airFlowRateMax is None, "Attribute \"airFlowRateMax\" is of type \"" + str(type(airFlowRateMax)) + "\" but must be of type \"" + str(measurement.Measurement) + "\""
        assert isinstance(bladeAction, str) or bladeAction is None, "Attribute \"bladeAction\" is of type \"" + str(type(bladeAction)) + "\" but must be of type \"" + str(str) + "\""
        assert isinstance(bladeEdge, str) or bladeEdge is None, "Attribute \"bladeEdge\" is of type \"" + str(type(bladeEdge)) + "\" but must be of type \"" + str(str) + "\""
        assert isinstance(bladeShape, str) or bladeShape is None, "Attribute \"bladeShape\" is of type \"" + str(type(bladeShape)) + "\" but must be of type \"" + str(str) + "\""
        assert isinstance(airFlowRateMax, measurement.Measurement) or airFlowRateMax is None, "Attribute \"airFlowRateMax\" is of type \"" + str(type(airFlowRateMax)) + "\" but must be of type \"" + str(measurement.Measurement) + "\""
        assert isinstance(bladeAction, measurement.Measurement) or bladeAction is None, "Attribute \"bladeAction\" is of type \"" + str(type(bladeAction)) + "\" but must be of type \"" + str(measurement.Measurement) + "\""
        assert isinstance(bladeEdge, measurement.Measurement) or bladeEdge is None, "Attribute \"bladeEdge\" is of type \"" + str(type(bladeEdge)) + "\" but must be of type \"" + str(measurement.Measurement) + "\""
        assert isinstance(bladeShape, measurement.Measurement) or bladeShape is None, "Attribute \"bladeShape\" is of type \"" + str(type(bladeShape)) + "\" but must be of type \"" + str(measurement.Measurement) + "\""
        assert isinstance(bladeThickness, measurement.Measurement) or bladeThickness is None, "Attribute \"bladeThickness\" is of type \"" + str(type(bladeThickness)) + "\" but must be of type \"" + str(measurement.Measurement) + "\""
        assert isinstance(closeOffRating, measurement.Measurement) or closeOffRating is None, "Attribute \"closeOffRating\" is of type \"" + str(type(closeOffRating)) + "\" but must be of type \"" + str(measurement.Measurement) + "\""
        assert isinstance(faceArea, measurement.Measurement) or faceArea is None, "Attribute \"faceArea\" is of type \"" + str(type(faceArea)) + "\" but must be of type \"" + str(measurement.Measurement) + "\""
        assert isinstance(frameDepth, measurement.Measurement) or frameDepth is None, "Attribute \"frameDepth\" is of type \"" + str(type(frameDepth)) + "\" but must be of type \"" + str(measurement.Measurement) + "\""
        assert isinstance(frameThickness, measurement.Measurement) or frameThickness is None, "Attribute \"frameThickness\" is of type \"" + str(frameThickness) + "\" but must be of type \"" + str(measurement.Measurement) + "\""
        assert isinstance(frameType, str) or frameType is None, "Attribute \"frameType\" is of type \"" + str(type(frameType)) + "\" but must be of type \"" + str(str) + "\""
        assert isinstance(leakageFullyClosed, measurement.Measurement) or leakageFullyClosed is None, "Attribute \"leakageFullyClosed\" is of type \"" + str(type(leakageFullyClosed)) + "\" but must be of type \"" + str(measurement.Measurement) + "\""
        assert isinstance(nominalAirFlowRate, measurement.Measurement) or nominalAirFlowRate is None, "Attribute \"nominalAirFlowRate\" is of type \"" + str(type(nominalAirFlowRate)) + "\" but must be of type \"" + str(measurement.Measurement) + "\""
        assert isinstance(numberOfBlades, measurement.Measurement) or numberOfBlades is None, "Attribute \"numberOfBlades\" is of type \"" + str(type(numberOfBlades)) + "\" but must be of type \"" + str(measurement.Measurement) + "\""
        assert isinstance(openPressureDrop, measurement.Measurement) or openPressureDrop is None, "Attribute \"openPressureDrop\" is of type \"" + str(type(openPressureDrop)) + "\" but must be of type \"" + str(measurement.Measurement) + "\""
        assert isinstance(operation, str) or operation is None, "Attribute \"operation\" is of type \"" + str(operation) + "\" but must be of type \"" + str(str) + "\""
        assert isinstance(operationTemperatureMax, measurement.Measurement) or operationTemperatureMax is None, "Attribute \"operationTemperatureMax\" is of type \"" + str(type(operationTemperatureMax)) + "\" but must be of type \"" + str(measurement.Measurement) + "\""
        assert isinstance(operationTemperatureMin, measurement.Measurement) or operationTemperatureMin is None, "Attribute \"operationTemperatureMin\" is of type \"" + str(type(operationTemperatureMin)) + "\" but must be of type \"" + str(measurement.Measurement) + "\""
        assert isinstance(orientation, str) or orientation is None, "Attribute \"orientation\" is of type \"" + str(type(orientation)) + "\" but must be of type \"" + str(str) + "\""
        assert isinstance(temperatureRating, measurement.Measurement) or temperatureRating is None, "Attribute \"temperatureRating\" is of type \"" + str(type(temperatureRating)) + "\" but must be of type \"" + str(measurement.Measurement) + "\""
        assert isinstance(workingPressureMax, measurement.Measurement) or workingPressureMax is None, "Attribute \"workingPressureMax\" is of type \"" + str(type(workingPressureMax)) + "\" but must be of type \"" + str(measurement.Measurement) + "\""
        assert isinstance(operationMode, str) or operationMode is None, "Attribute \"operationMode\" is of type \"" + str(type(operationMode)) + "\" but must be of type \"" + str(str) + "\""
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
        self.operationMode = operationMode 
        