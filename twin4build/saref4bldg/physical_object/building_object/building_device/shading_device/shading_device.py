from __future__ import annotations
from typing import Union
import twin4build.saref4bldg.physical_object.building_object.building_device.building_device as building_device
import twin4build.base as base

class ShadingDevice(building_device.BuildingDevice):
    def __init__(self,
                isExternal: Union[bool, None]=None, 
                mechanicalOperated: Union[bool, None]=None, 
                roughness: Union[str, None]=None, 
                shadingDeviceType: Union[str, None]=None, 
                solarReflectance: Union[base.PropertyValue, None]=None, 
                solarTransmittance: Union[base.PropertyValue, None]=None, 
                thermalTransmittance: Union[base.PropertyValue, None]=None, 
                visibleLightReflectance: Union[base.PropertyValue, None]=None, 
                visibleLightTransmittance: Union[base.PropertyValue, None]=None, 
                **kwargs):
        super().__init__(**kwargs)
        assert isinstance(isExternal, bool) or isExternal is None, "Attribute \"isExternaldp\" is of type \"" + str(type(isExternal)) + "\" but must be of type \"" + bool + "\""
        assert isinstance(mechanicalOperated, bool) or mechanicalOperated is None, "Attribute \"mechanicalOperateddp\" is of type \"" + str(type(mechanicalOperated)) + "\" but must be of type \"" + bool + "\""
        assert isinstance(roughness, str) or roughness is None, "Attribute \"roughnessdp\" is of type \"" + str(type(roughness)) + "\" but must be of type \"" + str + "\""
        assert isinstance(shadingDeviceType, str) or shadingDeviceType is None, "Attribute \"shadingDeviceTypedp\" is of type \"" + str(type(shadingDeviceType)) + "\" but must be of type \"" + str + "\""
        assert isinstance(solarReflectance, base.PropertyValue) or solarReflectance is None, "Attribute \"solarReflectanceop\" is of type \"" + str(type(solarReflectance)) + "\" but must be of type \"" + base.PropertyValue + "\""
        assert isinstance(solarTransmittance, base.PropertyValue) or solarTransmittance is None, "Attribute \"solarTransmittanceop\" is of type \"" + str(type(solarTransmittance)) + "\" but must be of type \"" + base.PropertyValue + "\""
        assert isinstance(thermalTransmittance, base.PropertyValue) or thermalTransmittance is None, "Attribute \"thermalTransmittanceop\" is of type \"" + str(type(thermalTransmittance)) + "\" but must be of type \"" + base.PropertyValue + "\""
        assert isinstance(visibleLightReflectance, base.PropertyValue) or visibleLightReflectance is None, "Attribute \"visibleLightReflectanceop\" is of type \"" + str(type(visibleLightReflectance)) + "\" but must be of type \"" + base.PropertyValue + "\""
        assert isinstance(visibleLightTransmittance, base.PropertyValue) or visibleLightTransmittance is None, "Attribute \"visibleLightTransmittanceop\" is of type \"" + str(type(visibleLightTransmittance)) + "\" but must be of type \"" + base.PropertyValue + "\""

        self.isExternaldp = isExternal
        self.mechanicalOperateddp = mechanicalOperated
        self.roughnessdp = roughness
        self.shadingDeviceTypedp = shadingDeviceType
        self.solarReflectanceop = solarReflectance
        self.solarTransmittanceop = solarTransmittance
        self.thermalTransmittanceop = thermalTransmittance
        self.visibleLightReflectanceop = visibleLightReflectance
        self.visibleLightTransmittanceop = visibleLightTransmittance

        