from saref.device.device import Device
from saref4bldg.physical_object.building_object.building_object import BuildingObject

class BuildingDevice(BuildingObject, Device):
    def __init__(self,
                **kwargs):
        super().__init__(**kwargs)