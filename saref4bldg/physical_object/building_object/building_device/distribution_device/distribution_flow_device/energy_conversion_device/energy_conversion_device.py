from ..distribution_flow_device import DistributionFlowDevice
class EnergyConversionDevice(DistributionFlowDevice):
    def __init__(self,
                **kwargs):
        super().__init__(**kwargs)