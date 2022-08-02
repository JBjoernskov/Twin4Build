from ..flow_controller import FlowController
class Valve(FlowController):
    def __init__(self,
                closeOffRating = None, 
                flowCoefficient = None, 
                size = None, 
                testPressure = None, 
                valveMechanism = None, 
                valveOperation = None, 
                valvePattern = None, 
                workingPressure = None,
                **kwargs):
        super().__init__(**kwargs)

        self.closeOffRating = closeOffRating
        self.flowCoefficient = flowCoefficient
        self.size = size
        self.testPressure = testPressure
        self.valveMechanism = valveMechanism
        self.valveOperation = valveOperation
        self.valvePattern = valvePattern
        self.workingPressure = workingPressure