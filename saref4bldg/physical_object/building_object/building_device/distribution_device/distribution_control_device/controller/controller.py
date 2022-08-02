from ..distribution_control_device import DistributionControlDevice
class Controller(DistributionControlDevice):
    def __init__(self,
                **kwargs):
        super().__init__(**kwargs)
        