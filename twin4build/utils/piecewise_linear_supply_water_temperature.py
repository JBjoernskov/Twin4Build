from twin4build.saref4syst.system import System
from pwlf import PiecewiseLinFit

class PiecewiseLinearSupplyWaterTemperature(System):
    def __init__(self,
                **kwargs):
        super().__init__(**kwargs)

    def initialize(self,
                    startPeriod=None,
                    endPeriod=None,
                    stepSize=None):
        pass

    def do_step(self, secondTime=None, dateTime=None, stepSize=None):
        '''
             takes in the current time and uses the calibrated model to make a prediction for the output value
        '''

        X = list(self.input.values())[0]
        key = list(self.output.keys())[0]
        if dateTime.hour>=5 and dateTime.hour<=7:
            self.output[key] = self.model["boost"].predict(X)[0] ###
        else:
            self.output[key] = self.model["normal"].predict(X)[0] ###


    def calibrate(self, input=None, output=None, n_line_segments=None):
        '''
             uses input and output data to train a piecewise linear regression model with a specified number of line segments.
        '''

        self.model = {}
        for key in input.keys():
            X = input[key].iloc[:,0]
            self.model[key] = PiecewiseLinFit(X, output[key])
            res = self.model[key].fit(n_line_segments[key])
