import twin4build.saref4bldg.physical_object.building_object.building_device.building_device as building_device
class DistributionDevice(building_device.BuildingDevice):
    def __init__(self,
                **kwargs):
        super().__init__(**kwargs)