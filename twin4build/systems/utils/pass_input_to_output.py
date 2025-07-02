import twin4build.core as core
import twin4build.utils.types as tps
from typing import Optional
class PassInputToOutput(core.System):
    """
    This component simply passes inputs to outputs during simulation.
    """
    def __init__(self,
                **kwargs):
        super().__init__(**kwargs)
        self.input = {"value": tps.Scalar()}
        self.output = {"value": tps.Scalar()}
        self._config = {"parameters": []}

    @property
    def config(self):
        return self._config

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
        self.output["value"].set(self.input["value"], stepIndex)