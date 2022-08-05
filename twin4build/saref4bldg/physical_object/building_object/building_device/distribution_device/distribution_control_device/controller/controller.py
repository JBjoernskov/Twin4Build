import twin4build.saref4bldg.physical_object.building_object.building_device.distribution_device.distribution_control_device.distribution_control_device as distribution_control_device
class Controller(distribution_control_device.DistributionControlDevice):
    def __init__(self,
                **kwargs):
        super().__init__(**kwargs)