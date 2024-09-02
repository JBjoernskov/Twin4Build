import twin4build.saref4syst.system as system
from numpy import NaN
class FlowJunctionSystem(system.System):
    """
             Supply node:
ROOM<---OUT   |
        \     |
          \   |
            \ |
             |<------- In
            /         
          /
        /
ROOM<---OUT


             Exhaust node:
ROOM--->IN    |
        \     |
          \   |
            \ |
             |--------> OUT 
            /         
          /
         /
ROOM--->IN
    """
    def __init__(self, 
                operationMode=None,
                **kwargs):
        super().__init__(**kwargs)
        self.operationMode = operationMode
        self._config = {"parameters": []}

    @property
    def config(self):
        return self._config

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
        self.output["flowRate"] = sum(v for k, v in self.input.items() if "flowRate" in k)
        if self.output["flowRate"]!=0 and self.operationMode=="return":
            self.output["flowTemperatureOut"] = sum(v*self.input[k.replace("flowRate", "flowTemperatureIn")] for k,v in self.input.items() if "flowRate" in k)/self.output["flowRate"]
        else:
            self.output["flowTemperatureOut"] = NaN


