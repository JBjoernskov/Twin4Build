import twin4build.core as core
import twin4build.utils.types as tps
import torch
from typing import Optional
class MaxSystem(core.System):
    """
    If value>=threshold set to on_value else set to off_value
    """
    def __init__(self,
                **kwargs):
        super().__init__(**kwargs)
        
        self.input = {"inputs": tps.Vector()}
        self.output = {"value": tps.Scalar()}

    def initialize(self,
                    startTime=None,
                    endTime=None,
                    stepSize=None,
                    simulator=None):
        pass
    def do_step(self, 
                secondTime=None, 
                dateTime=None, 
                stepSize=None,
                stepIndex: Optional[int] = None,
                simulator: Optional[core.Simulator] = None):
        self.output["value"].set(torch.max(self.input["inputs"].get()), stepIndex)