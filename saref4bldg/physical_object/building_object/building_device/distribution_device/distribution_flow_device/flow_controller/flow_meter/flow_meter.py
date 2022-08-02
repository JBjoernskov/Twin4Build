from ..flow_controller import FlowController
class FlowMeter(FlowController):
    def __init__(self,
                readOutType = None,
                remoteReading = None,
                **kwargs):
        super().__init__(**kwargs)
        self.readOutType = readOutType
        self.remoteReading = remoteReading