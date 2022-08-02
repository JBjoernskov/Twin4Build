from ..distribution_flow_device import DistributionFlowDevice
class FlowTerminal(DistributionFlowDevice):
    def __init__(self,
                **kwargs):
        super().__init__(**kwargs)