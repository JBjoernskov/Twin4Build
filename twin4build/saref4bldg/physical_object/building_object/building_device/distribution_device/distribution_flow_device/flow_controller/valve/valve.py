import twin4build.saref4bldg.physical_object.building_object.building_device.distribution_device.distribution_flow_device.flow_controller.flow_controller as flow_controller
from typing import Union
import twin4build.saref.property_value.property_value as property_value
from twin4build.logger.Logging import Logging
import twin4build.saref.property_.s4bldg_property.s4bldg_property as s4bldg_property
import twin4build.saref.unit_of_measure.unit_of_measure as unit_of_measure
logger = Logging.get_logger("ai_logfile")

class Valve(flow_controller.FlowController):
    def __init__(self,
                closeOffRating: Union[property_value.PropertyValue, None] = None, 
                flowCoefficient: Union[property_value.PropertyValue, None] = None, 
                size: Union[property_value.PropertyValue, None] = None, 
                testPressure: Union[property_value.PropertyValue, None] = None, 
                valveMechanism: Union[str, None] = None, 
                valveOperation: Union[str, None] = None, 
                valvePattern: Union[str, None] = None, 
                workingPressure: Union[property_value.PropertyValue, None] = None,
                **kwargs):
        
        logger.info("[Valve] : Entered in Initialise Function")
        super().__init__(**kwargs)

        closeOffRating_ = s4bldg_property.CloseOffRating()
        if closeOffRating is not None:
            closeOffRating = property_value.PropertyValue(hasValue=closeOffRating.hasValue,
                                                            isMeasuredIn=closeOffRating.isMeasuredIn,
                                                            isValueOfProperty=closeOffRating_)
        else:
            closeOffRating = property_value.PropertyValue(isValueOfProperty=closeOffRating_)
        self.hasProperty.append(closeOffRating_)
        self.hasPropertyValue.append(closeOffRating)

        flowCoefficient_ = s4bldg_property.FlowCoefficient()
        if flowCoefficient is not None:
            flowCoefficient = property_value.PropertyValue(hasValue=flowCoefficient.hasValue,
                                                            isMeasuredIn=flowCoefficient.isMeasuredIn,
                                                            isValueOfProperty=flowCoefficient_)
        else:
            flowCoefficient = property_value.PropertyValue(isValueOfProperty=flowCoefficient_)
        self.hasProperty.append(flowCoefficient_)
        self.hasPropertyValue.append(flowCoefficient)

        size_ = s4bldg_property.Size()
        if size is not None:
            size = property_value.PropertyValue(hasValue=size.hasValue,
                                                isMeasuredIn=size.isMeasuredIn,
                                                isValueOfProperty=size_)
        else:
            size = property_value.PropertyValue(isValueOfProperty=size_)
        self.hasProperty.append(size_)
        self.hasPropertyValue.append(size)

        testPressure_ = s4bldg_property.TestPressure()
        if testPressure is not None:
            testPressure = property_value.PropertyValue(hasValue=testPressure.hasValue,
                                                        isMeasuredIn=testPressure.isMeasuredIn,
                                                        isValueOfProperty=testPressure_)
        else:
            testPressure = property_value.PropertyValue(isValueOfProperty=testPressure_)
        self.hasProperty.append(testPressure_)
        self.hasPropertyValue.append(testPressure)

        valveMechanism_ = s4bldg_property.ValveMechanism()
        if valveMechanism is not None:
            valveMechanism = property_value.PropertyValue(hasValue=valveMechanism.hasValue,
                                                            isMeasuredIn=valveMechanism.isMeasuredIn,
                                                            isValueOfProperty=valveMechanism_)
        else:
            valveMechanism = property_value.PropertyValue(isValueOfProperty=valveMechanism_)
        self.hasProperty.append(valveMechanism_)
        self.hasPropertyValue.append(valveMechanism)

        valveOperation_ = s4bldg_property.ValveOperation()
        if valveOperation is not None:
            valveOperation = property_value.PropertyValue(hasValue=valveOperation.hasValue,
                                                            isMeasuredIn=valveOperation.isMeasuredIn,
                                                            isValueOfProperty=valveOperation_)
        else:
            valveOperation = property_value.PropertyValue(isValueOfProperty=valveOperation_)
        self.hasProperty.append(valveOperation_)
        self.hasPropertyValue.append(valveOperation)

        valvePattern_ = s4bldg_property.ValvePattern()
        if valvePattern is not None:
            valvePattern = property_value.PropertyValue(hasValue=valvePattern.hasValue,
                                                        isMeasuredIn=valvePattern.isMeasuredIn,
                                                        isValueOfProperty=valvePattern_)
        else:
            valvePattern = property_value.PropertyValue(isValueOfProperty=valvePattern_)
        self.hasProperty.append(valvePattern_)
        self.hasPropertyValue.append(valvePattern)


        logger.info("[Valve] : Exited from Initialise Function")

    @property
    def closeOffRating(self):
        el = [el for el in self.hasPropertyValue if isinstance(el.isValueOfProperty, s4bldg_property.CloseOffRating)]
        return el[0] if len(el) > 0 else None
    
    @property
    def flowCoefficient(self):
        el = [el for el in self.hasPropertyValue if isinstance(el.isValueOfProperty, s4bldg_property.FlowCoefficient)]
        return el[0] if len(el) > 0 else None

    @property
    def size(self):
        el = [el for el in self.hasPropertyValue if isinstance(el.isValueOfProperty, s4bldg_property.Size)]
        return el[0] if len(el) > 0 else None

    @property
    def testPressure(self):
        el = [el for el in self.hasPropertyValue if isinstance(el.isValueOfProperty, s4bldg_property.TestPressure)]
        return el[0] if len(el) > 0 else None

    @property
    def valveMechanism(self):
        el = [el for el in self.hasPropertyValue if isinstance(el.isValueOfProperty, s4bldg_property.ValveMechanism)]
        return el[0] if len(el) > 0 else None

    @property
    def valveOperation(self):
        el = [el for el in self.hasPropertyValue if isinstance(el.isValueOfProperty, s4bldg_property.ValveOperation)]
        return el[0] if len(el) > 0 else None

    @property
    def valvePattern(self):
        el = [el for el in self.hasPropertyValue if isinstance(el.isValueOfProperty, s4bldg_property.ValvePattern)]
        return el[0] if len(el) > 0 else None
