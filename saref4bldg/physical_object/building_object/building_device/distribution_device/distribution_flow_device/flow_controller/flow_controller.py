from ..distribution_flow_device import DistributionFlowDevice
class FlowController(DistributionFlowDevice):
    def __init__(self,
                **kwargs):
        super().__init__(**kwargs)