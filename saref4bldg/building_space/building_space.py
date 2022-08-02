from  saref4syst.system import System
class BuildingSpace(System):
    def __init__(self,
                hasSpace = None,
                isSpaceOf = None,
                contains = None,
                **kwargs):
        super().__init__(**kwargs)
        self.hasSpace = hasSpace
        self.isSpaceOf = isSpaceOf
        self.contains = contains