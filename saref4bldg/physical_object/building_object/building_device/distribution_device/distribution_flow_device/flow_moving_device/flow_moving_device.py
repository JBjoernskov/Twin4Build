from ..distribution_flow_device import DistributionFlowDevice
class FlowMovingDevice(DistributionFlowDevice):
    def __init__(self,
                **kwargs):
        super().__init__(**kwargs)