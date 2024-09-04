import twin4build.saref4bldg.physical_object.building_object.building_device.distribution_device.distribution_control_device.controller.controller as controller
from typing import Union
class ClassificationAnnController(controller.Controller):
    def __init__(self,
                **kwargs):
        super().__init__(**kwargs)
        