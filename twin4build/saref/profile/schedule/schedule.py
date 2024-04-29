from twin4build.saref4syst.system import System
from twin4build.saref.profile.profile import Profile
class Schedule(System, Profile):
    def __init__(self,
                 **kwargs):
        super().__init__(**kwargs)
