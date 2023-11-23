from twin4build.saref4syst.system import System
from pwlf import PiecewiseLinFit
import copy
import numpy as np
class PiecewiseLinear(System):
    def __init__(self,
                X=None,
                Y=None,
                **kwargs):
        super().__init__(**kwargs)

        self.X = X
        self.Y = Y

        if X and Y: #Not None
            self.XY = np.array([X, Y]).transpose()
            self.get_a_b_vectors()

    def cache(self,
            startPeriod=None,
            endPeriod=None,
            stepSize=None):
        pass

    def initialize(self,
                    startPeriod=None,
                    endPeriod=None,
                    stepSize=None):
        self.outputUncertainty = copy.deepcopy(self.output)
        key = list(self.outputUncertainty.keys())[0]
        self.outputUncertainty[key] = 0

    def get_a_b_vectors(self):
        self.a_vec = (self.XY[1:,1] - self.XY[0:-1,1])/((self.XY[1:,0] - self.XY[0:-1,0]))
        self.b_vec = self.XY[0:-1,1] - self.a_vec*self.XY[0:-1,0]

    def get_Y(self, X):
        if X <= self.XY[0,0]:
            Y = self.XY[0,1]

        elif X >= self.XY[-1,0]:
            Y = self.XY[-1,1]

        else:
            cond = X < self.XY[:,0]
            idx = np.where(cond)[0][0]-1
            a = self.a_vec[idx]
            b = self.b_vec[idx]
            Y = a*X + b
        return Y
    
    def do_step_new(self, secondTime=None, dateTime=None, stepSize=None):
        X = list(self.input.values())[0]
        key = list(self.output.keys())[0]
        self.output[key] = self.get_Y(X)

    def do_step(self, secondTime=None, dateTime=None, stepSize=None):
        '''
             takes an input value X, feeds it through a pre-trained model, and stores the output in self.output. 
        '''

        X = list(self.input.values())[0]
        key = list(self.output.keys())[0]
        self.output[key] = self.model.predict(X)[0] ###

    def calibrate(self, input=None, output=None, n_line_segments=None):
        '''
            Fits a piecewise linear model to the input-output data and stores it in self.model.
        '''
        
        X = input.iloc[:,0]
        self.model = PiecewiseLinFit(X, output)
        res = self.model.fit(n_line_segments)








