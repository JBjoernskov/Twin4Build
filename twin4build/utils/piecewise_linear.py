from twin4build.saref4syst.system import System
from pwlf import PiecewiseLinFit

class PiecewiseLinear(System):
    def __init__(self,
                **kwargs):
        super().__init__(**kwargs)

    def initialize(self,
                    startPeriod=None,
                    endPeriod=None,
                    stepSize=None):
        pass

    def do_step(self, secondTime=None, dateTime=None, stepSize=None):
        X = list(self.input.values())[0]
        key = list(self.output.keys())[0]
        self.output[key] = self.model.predict(X)[0] ###

    def calibrate(self, input=None, output=None, n_line_segments=None):
        X = input.iloc[:,0]
        self.model = PiecewiseLinFit(X, output)
        res = self.model.fit(n_line_segments)
