from twin4build.saref4syst.system import System
class ToDegCFromDegK(System):
    def __init__(self):
        super().__init__()
        self.input = {"K": None}
        self.output = {"C": None}

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
        self.output["C"] = self.input["K"]-273.15
