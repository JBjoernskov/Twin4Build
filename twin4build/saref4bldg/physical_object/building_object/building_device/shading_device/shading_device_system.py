from __future__ import annotations
from typing import Union
import twin4build.saref4bldg.physical_object.building_object.building_device.shading_device.shading_device as shading_device
import twin4build.utils.input_output_types as tps

class ShadingDeviceSystem(shading_device.ShadingDevice):
    def __init__(self,
                **kwargs):
        super().__init__(**kwargs)
        self.input = {"shadePosition": tps.Scalar()}
        self.output = {"shadePosition": tps.Scalar()}

    def cache(self,
            startTime=None,
            endTime=None,
            stepSize=None):
        pass

    def initialize(self,
                    startTime=None,
                    endTime=None,
                    stepSize=None,
                    model=None):
        pass

    def do_step(self, secondTime=None, dateTime=None, stepSize=None):
        self.output["shadePosition"].set(self.input["shadePosition"])