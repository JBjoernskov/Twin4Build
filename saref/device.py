from saref4bldg.physical_object.physical_object import PhysicalObject
class Device(PhysicalObject):
    def __init__(self,
                **kwargs):
        super().__init__(**kwargs)