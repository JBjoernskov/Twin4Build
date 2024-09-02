from __future__ import annotations
from typing import Union
import twin4build.saref4bldg.physical_object.building_object.building_device.shading_device.shading_device as shading_device

from twin4build.logger.Logging import Logging

logger = Logging.get_logger("ai_logfile")

logger.info("Shading Device Model")

class ShadingDeviceSystem(shading_device.ShadingDevice):
    def __init__(self,
                **kwargs):
        super().__init__(**kwargs)

        logger.info("[shading device model] : Entered Intialise Method")

        self.input = {"shadePosition": None}
        self.output = {"shadePosition": None}

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
        logger.info("[shading device model] : Entered do_step Method")

        self.output["shadePosition"] = self.input["shadePosition"]