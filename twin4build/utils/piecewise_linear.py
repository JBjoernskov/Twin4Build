from twin4build.saref4syst.system import System
import datetime
from random import randrange
import random
from pwlf import PiecewiseLinFit

class PiecewiseLinear(System):
    def __init__(self,
                **kwargs):
        super().__init__(**kwargs)
        self.X = X
        self.Y = Y
        self.model = PiecewiseLinFit(self.X, self.Y)
        res = model.fit(2)

    def initialize(self):
        pass

    def do_step(self, time=None, stepSize=None):
        self.model.predict(input) ###

    def calibrate(self, input=None, output=None):
        res = self.model.fit(2)
