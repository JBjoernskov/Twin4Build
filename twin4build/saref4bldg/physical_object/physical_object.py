import twin4build.saref4syst.system as system
import twin4build.saref.feature_of_interest.feature_of_interest as feature_of_interest
class PhysicalObject(feature_of_interest.FeatureOfInterest, system.System):
    def __init__(self,
                isContainedIn=None,
                **kwargs):
        super().__init__(**kwargs)
        self.isContainedIn = isContainedIn