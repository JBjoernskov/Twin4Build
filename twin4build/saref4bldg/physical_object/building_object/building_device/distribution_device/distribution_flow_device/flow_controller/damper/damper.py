import twin4build.saref4bldg.physical_object.building_object.building_device.distribution_device.distribution_flow_device.flow_controller.flow_controller as flow_controller
from typing import Union
import twin4build.saref.property_value.property_value as property_value
import twin4build.saref.property_.s4bldg_property.s4bldg_property as s4bldg_property

class Damper(flow_controller.FlowController):
    def __init__(self,
                airFlowRateMax: Union[property_value.PropertyValue, None] = None,
                bladeAction: Union[str, None] = None,
                bladeEdge: Union[str, None] = None,
                bladeShape: Union[str, None] = None,
                bladeThickness: Union[property_value.PropertyValue, None] = None,
                closeOffRating: Union[property_value.PropertyValue, None] = None,
                faceArea: Union[property_value.PropertyValue, None] = None,
                frameDepth: Union[property_value.PropertyValue, None] = None,
                frameThickness: Union[property_value.PropertyValue, None] = None,
                frameType: Union[str, None] = None,
                leakageFullyClosed: Union[property_value.PropertyValue, None] = None,
                nominalAirFlowRate: Union[property_value.PropertyValue, None] = None,
                numberOfBlades: Union[property_value.PropertyValue, None] = None,
                openPressureDrop: Union[property_value.PropertyValue, None] = None,
                operation: Union[str, None] = None,
                operationTemperatureMax: Union[property_value.PropertyValue, None] = None,
                operationTemperatureMin: Union[property_value.PropertyValue, None] = None,
                orientation: Union[str, None] = None,
                temperatureRating: Union[property_value.PropertyValue, None] = None,
                workingPressureMax: Union[property_value.PropertyValue, None] = None,
                **kwargs):
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

        bladeAction_ = s4bldg_property.BladeAction()
        if bladeAction is not None:
            bladeAction = property_value.PropertyValue(hasValue=bladeAction.hasValue,
                                                       isMeasuredIn=bladeAction.isMeasuredIn,
                                                       isValueOfProperty=bladeAction_)
        else:
            bladeAction = property_value.PropertyValue(isValueOfProperty=bladeAction_)
        self.hasProperty.append(bladeAction_)
        self.hasPropertyValue.append(bladeAction)

        bladeEdge_ = s4bldg_property.BladeEdge()
        if bladeEdge is not None:
            bladeEdge = property_value.PropertyValue(hasValue=bladeEdge.hasValue,
                                                     isMeasuredIn=bladeEdge.isMeasuredIn,
                                                     isValueOfProperty=bladeEdge_)
        else:
            bladeEdge = property_value.PropertyValue(isValueOfProperty=bladeEdge_)
        self.hasProperty.append(bladeEdge_)
        self.hasPropertyValue.append(bladeEdge)

        bladeShape_ = s4bldg_property.BladeShape()
        if bladeShape is not None:
            bladeShape = property_value.PropertyValue(hasValue=bladeShape.hasValue,
                                                      isMeasuredIn=bladeShape.isMeasuredIn,
                                                      isValueOfProperty=bladeShape_)
        else:
            bladeShape = property_value.PropertyValue(isValueOfProperty=bladeShape_)
        self.hasProperty.append(bladeShape_)
        self.hasPropertyValue.append(bladeShape)

        bladeThickness_ = s4bldg_property.BladeThickness()
        if bladeThickness is not None:
            bladeThickness = property_value.PropertyValue(hasValue=bladeThickness.hasValue,
                                                          isMeasuredIn=bladeThickness.isMeasuredIn,
                                                          isValueOfProperty=bladeThickness_)
        else:
            bladeThickness = property_value.PropertyValue(isValueOfProperty=bladeThickness_)
        self.hasProperty.append(bladeThickness_)
        self.hasPropertyValue.append(bladeThickness)

        closeOffRating_ = s4bldg_property.CloseOffRating()
        if closeOffRating is not None:
            closeOffRating = property_value.PropertyValue(hasValue=closeOffRating.hasValue,
                                                          isMeasuredIn=closeOffRating.isMeasuredIn,
                                                          isValueOfProperty=closeOffRating_)
        else:
            closeOffRating = property_value.PropertyValue(isValueOfProperty=closeOffRating_)
        self.hasProperty.append(closeOffRating_)
        self.hasPropertyValue.append(closeOffRating)

        faceArea_ = s4bldg_property.FaceArea()
        if faceArea is not None:
            faceArea = property_value.PropertyValue(hasValue=faceArea.hasValue,
                                                    isMeasuredIn=faceArea.isMeasuredIn,
                                                    isValueOfProperty=faceArea_)
        else:
            faceArea = property_value.PropertyValue(isValueOfProperty=faceArea_)
        self.hasProperty.append(faceArea_)
        self.hasPropertyValue.append(faceArea)

        frameDepth_ = s4bldg_property.FrameDepth()
        if frameDepth is not None:
            frameDepth = property_value.PropertyValue(hasValue=frameDepth.hasValue,
                                                      isMeasuredIn=frameDepth.isMeasuredIn,
                                                      isValueOfProperty=frameDepth_)
        else:
            frameDepth = property_value.PropertyValue(isValueOfProperty=frameDepth_)
        self.hasProperty.append(frameDepth_)
        self.hasPropertyValue.append(frameDepth)

        frameThickness_ = s4bldg_property.FrameThickness()
        if frameThickness is not None:
            frameThickness = property_value.PropertyValue(hasValue=frameThickness.hasValue,
                                                          isMeasuredIn=frameThickness.isMeasuredIn,
                                                          isValueOfProperty=frameThickness_)
        else:
            frameThickness = property_value.PropertyValue(isValueOfProperty=frameThickness_)
        self.hasProperty.append(frameThickness_)
        self.hasPropertyValue.append(frameThickness)

        frameType_ = s4bldg_property.FrameType()
        if frameType is not None:
            frameType = property_value.PropertyValue(hasValue=frameType.hasValue,
                                                     isMeasuredIn=frameType.isMeasuredIn,
                                                     isValueOfProperty=frameType_)
        else:
            frameType = property_value.PropertyValue(isValueOfProperty=frameType_)
        self.hasProperty.append(frameType_)
        self.hasPropertyValue.append(frameType)

        leakageFullyClosed_ = s4bldg_property.LeakageFullyClosed()
        if leakageFullyClosed is not None:
            leakageFullyClosed = property_value.PropertyValue(hasValue=leakageFullyClosed.hasValue,
                                                              isMeasuredIn=leakageFullyClosed.isMeasuredIn,
                                                              isValueOfProperty=leakageFullyClosed_)
        else:
            leakageFullyClosed = property_value.PropertyValue(isValueOfProperty=leakageFullyClosed_)
        self.hasProperty.append(leakageFullyClosed_)
        self.hasPropertyValue.append(leakageFullyClosed)

        nominalAirFlowRate_ = s4bldg_property.NominalAirFlowRate()
        if nominalAirFlowRate is not None:
            nominalAirFlowRate = property_value.PropertyValue(hasValue=nominalAirFlowRate.hasValue,
                                                              isMeasuredIn=nominalAirFlowRate.isMeasuredIn,
                                                              isValueOfProperty=nominalAirFlowRate_)
        else:
            nominalAirFlowRate = property_value.PropertyValue(isValueOfProperty=nominalAirFlowRate_)
        self.hasProperty.append(nominalAirFlowRate_)
        self.hasPropertyValue.append(nominalAirFlowRate)

        numberOfBlades_ = s4bldg_property.NumberOfBlades()
        if numberOfBlades is not None:
            numberOfBlades = property_value.PropertyValue(hasValue=numberOfBlades.hasValue,
                                                          isMeasuredIn=numberOfBlades.isMeasuredIn,
                                                          isValueOfProperty=numberOfBlades_)
        else:
            numberOfBlades = property_value.PropertyValue(isValueOfProperty=numberOfBlades_)
        self.hasProperty.append(numberOfBlades_)
        self.hasPropertyValue.append(numberOfBlades)

        openPressureDrop_ = s4bldg_property.OpenPressureDrop()
        if openPressureDrop is not None:
            openPressureDrop = property_value.PropertyValue(hasValue=openPressureDrop.hasValue,
                                                            isMeasuredIn=openPressureDrop.isMeasuredIn,
                                                            isValueOfProperty=openPressureDrop_)
        else:
            openPressureDrop = property_value.PropertyValue(isValueOfProperty=openPressureDrop_)
        self.hasProperty.append(openPressureDrop_)
        self.hasPropertyValue.append(openPressureDrop)

        operation_ = s4bldg_property.Operation()
        if operation is not None:
            operation = property_value.PropertyValue(hasValue=operation.hasValue,
                                                     isMeasuredIn=operation.isMeasuredIn,
                                                     isValueOfProperty=operation_)
        else:
            operation = property_value.PropertyValue(isValueOfProperty=operation_)
        self.hasProperty.append(operation_)
        self.hasPropertyValue.append(operation)

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

        orientation_ = s4bldg_property.Orientation()
        if orientation is not None:
            orientation = property_value.PropertyValue(hasValue=orientation.hasValue,
                                                       isMeasuredIn=orientation.isMeasuredIn,
                                                       isValueOfProperty=orientation_)
        else:
            orientation = property_value.PropertyValue(isValueOfProperty=orientation_)
        self.hasProperty.append(orientation_)
        self.hasPropertyValue.append(orientation)

        temperatureRating_ = s4bldg_property.TemperatureRating()
        if temperatureRating is not None:
            temperatureRating = property_value.PropertyValue(hasValue=temperatureRating.hasValue,
                                                             isMeasuredIn=temperatureRating.isMeasuredIn,
                                                             isValueOfProperty=temperatureRating_)
        else:
            temperatureRating = property_value.PropertyValue(isValueOfProperty=temperatureRating_)
        self.hasProperty.append(temperatureRating_)
        self.hasPropertyValue.append(temperatureRating)

        workingPressureMax_ = s4bldg_property.WorkingPressureMax()
        if workingPressureMax is not None:
            workingPressureMax = property_value.PropertyValue(hasValue=workingPressureMax.hasValue,
                                                              isMeasuredIn=workingPressureMax.isMeasuredIn,
                                                              isValueOfProperty=workingPressureMax_)
        else:
            workingPressureMax = property_value.PropertyValue(isValueOfProperty=workingPressureMax_)
        self.hasProperty.append(workingPressureMax_)
        self.hasPropertyValue.append(workingPressureMax)
    
    @property
    def airFlowRateMax(self):
        el = [el for el in self.hasPropertyValue if isinstance(el.isValueOfProperty, s4bldg_property.AirFlowRateMax)]
        return el[0] if len(el) > 0 else None

    @property
    def bladeAction(self):
        el = [el for el in self.hasPropertyValue if isinstance(el.isValueOfProperty, s4bldg_property.BladeAction)]
        return el[0] if len(el) > 0 else None

    @property
    def bladeEdge(self):
        el = [el for el in self.hasPropertyValue if isinstance(el.isValueOfProperty, s4bldg_property.BladeEdge)]
        return el[0] if len(el) > 0 else None
    
    @property
    def bladeShape(self):
        el = [el for el in self.hasPropertyValue if isinstance(el.isValueOfProperty, s4bldg_property.BladeShape)]
        return el[0] if len(el) > 0 else None
    
    @property
    def bladeThickness(self):
        el = [el for el in self.hasPropertyValue if isinstance(el.isValueOfProperty, s4bldg_property.BladeThickness)]
        return el[0] if len(el) > 0 else None
    
    @property
    def closeOffRating(self):
        el = [el for el in self.hasPropertyValue if isinstance(el.isValueOfProperty, s4bldg_property.CloseOffRating)]
        return el[0] if len(el) > 0 else None
    
    @property
    def faceArea(self):
        el = [el for el in self.hasPropertyValue if isinstance(el.isValueOfProperty, s4bldg_property.FaceArea)]
        return el[0] if len(el) > 0 else None
    
    @property
    def frameDepth(self):
        el = [el for el in self.hasPropertyValue if isinstance(el.isValueOfProperty, s4bldg_property.FrameDepth)]
        return el[0] if len(el) > 0 else None
    
    @property
    def frameThickness(self):
        el = [el for el in self.hasPropertyValue if isinstance(el.isValueOfProperty, s4bldg_property.FrameThickness)]
        return el[0] if len(el) > 0 else None
    
    @property
    def frameType(self):
        el = [el for el in self.hasPropertyValue if isinstance(el.isValueOfProperty, s4bldg_property.FrameType)]
        return el[0] if len(el) > 0 else None
    
    @property
    def leakageFullyClosed(self):
        el = [el for el in self.hasPropertyValue if isinstance(el.isValueOfProperty, s4bldg_property.LeakageFullyClosed)]
        return el[0] if len(el) > 0 else None
    
    @property
    def nominalAirFlowRate(self):
        el = [el for el in self.hasPropertyValue if isinstance(el.isValueOfProperty, s4bldg_property.NominalAirFlowRate)]
        return el[0] if len(el) > 0 else None
    
    @property
    def numberOfBlades(self):
        el = [el for el in self.hasPropertyValue if isinstance(el.isValueOfProperty, s4bldg_property.NumberOfBlades)]
        return el[0] if len(el) > 0 else None
    
    @property
    def openPressureDrop(self):
        el = [el for el in self.hasPropertyValue if isinstance(el.isValueOfProperty, s4bldg_property.OpenPressureDrop)]
        return el[0] if len(el) > 0 else None
    
    @property
    def operation(self):
        el = [el for el in self.hasPropertyValue if isinstance(el.isValueOfProperty, s4bldg_property.Operation)]
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
    def orientation(self):
        el = [el for el in self.hasPropertyValue if isinstance(el.isValueOfProperty, s4bldg_property.Orientation)]
        return el[0] if len(el) > 0 else None
    
    @property
    def temperatureRating(self):
        el = [el for el in self.hasPropertyValue if isinstance(el.isValueOfProperty, s4bldg_property.TemperatureRating)]
        return el[0] if len(el) > 0 else None
    
    @property
    def workingPressureMax(self):
        el = [el for el in self.hasPropertyValue if isinstance(el.isValueOfProperty, s4bldg_property.WorkingPressureMax)]
        return el[0] if len(el) > 0 else None
    
