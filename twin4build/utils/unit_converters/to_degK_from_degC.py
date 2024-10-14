from twin4build.saref4syst.system import System
class ToDegKFromDegC(System):
    def __init__(self):
        super().__init__()
        self.input = {"C": None}
        self.output = {"K": None}

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
        self.output["K"].set(self.input["C"]+273.15)