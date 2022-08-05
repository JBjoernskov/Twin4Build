import saref4bldg.physical_object.building_object.building_device.distribution_device.distribution_device as distribution_device
class DistributionControlDevice(distribution_device.DistributionDevice):
    def __init__(self,
                **kwargs):
        super().__init__(**kwargs)