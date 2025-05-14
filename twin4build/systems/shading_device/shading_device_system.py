import twin4build.utils.input_output_types as tps
import twin4build.core as core
from typing import Optional
class ShadingDeviceSystem(core.System):
    def __init__(self,
                **kwargs):
        super().__init__(**kwargs)
        self.input = {"shadePosition": tps.Scalar()}
        self.output = {"shadePosition": tps.Scalar()}

    def cache(self,
            startTime=None,
            endTime=None,
            stepSize=None):
        pass

    def initialize(self,
                    startTime=None,
                    endTime=None,
                    stepSize=None,
                    model=None):
        pass

    def do_step(self, secondTime=None, dateTime=None, stepSize=None, stepIndex: Optional[int] = None):
        self.output["shadePosition"].set(self.input["shadePosition"], stepIndex)