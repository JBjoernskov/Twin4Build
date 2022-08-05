from __future__ import annotations
import saref.device.device as device
import saref4bldg.physical_object.building_object.building_object as building_object
class BuildingDevice(building_object.BuildingObject, device.Device):
    def __init__(self,
                **kwargs):
        super().__init__(**kwargs)