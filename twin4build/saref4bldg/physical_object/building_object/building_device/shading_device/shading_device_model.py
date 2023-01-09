from __future__ import annotations
from typing import Union
import twin4build.saref4bldg.physical_object.building_object.building_device.shading_device.shading_device as shading_device
class ShadingDeviceModel(shading_device.ShadingDevice):
    def __init__(self,
                **kwargs):
        super().__init__(**kwargs)

    def do_step(self):
        self.output["shadePosition"] = self.input["shadePosition"]