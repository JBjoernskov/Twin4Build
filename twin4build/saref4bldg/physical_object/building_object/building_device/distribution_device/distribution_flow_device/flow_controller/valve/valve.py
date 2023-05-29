import twin4build.saref4bldg.physical_object.building_object.building_device.distribution_device.distribution_flow_device.flow_controller.flow_controller as flow_controller
from typing import Union
import twin4build.saref.measurement.measurement as measurement

from twin4build.logger.Logging import Logging

logger = Logging.get_logger("ai_logfile")

class Valve(flow_controller.FlowController):
    def __init__(self,
                closeOffRating: Union[measurement.Measurement, None] = None, 
                flowCoefficient: Union[measurement.Measurement, None] = None, 
                size: Union[measurement.Measurement, None] = None, 
                testPressure: Union[measurement.Measurement, None] = None, 
                valveMechanism: Union[str, None] = None, 
                valveOperation: Union[str, None] = None, 
                valvePattern: Union[str, None] = None, 
                workingPressure: Union[measurement.Measurement, None] = None,
                **kwargs):
        
        logger.info("[Valve] : Entered in Initialise Function")

        super().__init__(**kwargs)
        assert isinstance(closeOffRating, measurement.Measurement) or closeOffRating is None, "Attribute \"closeOffRating\" is of type \"" + str(type(closeOffRating)) + "\" but must be of type \"" + str(measurement.Measurement) + "\""
        assert isinstance(flowCoefficient, measurement.Measurement) or flowCoefficient is None, "Attribute \"flowCoefficient\" is of type \"" + str(type(flowCoefficient)) + "\" but must be of type \"" + str(measurement.Measurement) + "\""
        assert isinstance(size, measurement.Measurement) or size is None, "Attribute \"size\" is of type \"" + str(type(size)) + "\" but must be of type \"" + str(measurement.Measurement) + "\""
        assert isinstance(testPressure, measurement.Measurement) or testPressure is None, "Attribute \"testPressure\" is of type \"" + str(type(testPressure)) + "\" but must be of type \"" + str(measurement.Measurement) + "\""
        assert isinstance(valveMechanism, str) or valveMechanism is None, "Attribute \"valveMechanism\" is of type \"" + str(type(valveMechanism)) + "\" but must be of type \"" + str(str) + "\""
        assert isinstance(valveOperation, str) or valveOperation is None, "Attribute \"valveOperation\" is of type \"" + str(type(valveOperation)) + "\" but must be of type \"" + str(str) + "\""
        assert isinstance(valvePattern, str) or valvePattern is None, "Attribute \"valvePattern\" is of type \"" + str(type(valvePattern)) + "\" but must be of type \"" + str(str) + "\""
        assert isinstance(workingPressure, measurement.Measurement) or workingPressure is None, "Attribute \"workingPressure\" is of type \"" + str(type(workingPressure)) + "\" but must be of type \"" + str(measurement.Measurement) + "\""
        self.closeOffRating = closeOffRating
        self.flowCoefficient = flowCoefficient
        self.size = size
        self.testPressure = testPressure
        self.valveMechanism = valveMechanism
        self.valveOperation = valveOperation
        self.valvePattern = valvePattern
        self.workingPressure = workingPressure

        logger.info("[Valve] : Exited from Initialise Function")
