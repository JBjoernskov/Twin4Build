import twin4build.saref4bldg.physical_object.building_object.building_device.distribution_device.distribution_control_device.distribution_control_device as distribution_control_device
from typing import Union
class Controller(distribution_control_device.DistributionControlDevice):
    def __init__(self,
                controllingProperty: Union[str, None] = None,
                **kwargs):
        super().__init__(**kwargs)
        assert isinstance(controllingProperty, str) or controllingProperty is None, "Attribute \"controllingProperty\" is of type \"" + str(type(controllingProperty)) + "\" but must be of type \"" + str + "\""
        self.controllingProperty = controllingProperty