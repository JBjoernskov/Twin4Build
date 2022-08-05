import twin4build.saref4bldg.physical_object.physical_object as physical_object
class BuildingObject(physical_object.PhysicalObject):
    def __init__(self,
                **kwargs):
        super().__init__(**kwargs)