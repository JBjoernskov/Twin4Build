from __future__ import annotations
import twin4build.saref4bldg.physical_object.physical_object as physical_object
class Device(physical_object.PhysicalObject):
    def __init__(self,
                **kwargs):
        super().__init__(**kwargs)