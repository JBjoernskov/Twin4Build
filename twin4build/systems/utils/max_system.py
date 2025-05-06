import twin4build.core as core
import twin4build.utils.input_output_types as tps
import torch
class MaxSystem(core.System):
    """
    If value>=threshold set to on_value else set to off_value
    """
    def __init__(self,
                **kwargs):
        super().__init__(**kwargs)
        
        self.input = {"inputs": tps.Vector()}
        self.output = {"value": tps.Scalar()}
    
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
    def do_step(self, secondTime=None, dateTime=None, stepSize=None):
        self.output["value"].set(torch.max(self.input["inputs"].get()))