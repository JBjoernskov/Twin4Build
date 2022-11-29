import twin4build.saref4syst.system as system
class PhysicalObject(system.System):
    def __init__(self,
                isContainedIn = None,
                **kwargs):
        super().__init__(**kwargs)
        self.isContainedIn = isContainedIn