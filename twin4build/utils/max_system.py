from twin4build.saref4syst.system import System

class MaxSystem(System):
    """
    If value>=threshold set to on_value else set to off_value
    """
    def __init__(self,
                **kwargs):
        super().__init__(**kwargs)
        
        self.input = {}
        self.output = {"value": None}
    
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
        self.output["value"] = max(self.input.values())