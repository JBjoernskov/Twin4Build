from twin4build.saref4syst.system import System
from twin4build.utils.schedule import ScheduleSystem
from twin4build.utils.piecewise_linear import PiecewiseLinearSystem
import numpy as np
class PiecewiseLinearScheduleSystem(PiecewiseLinearSystem, ScheduleSystem):
    def __init__(self,
                **kwargs):
        super().__init__(**kwargs)
        self.input = {}
        self.output = {}

    def cache(self,
            startTime=None,
            endTime=None,
            stepSize=None):
        pass

    def initialize(self,
                    startTime=None,
                    endTime=None,
                    stepSize=None):
        pass

    def do_step(self, secondTime=None, dateTime=None, stepSize=None):
        '''
             takes in the current time and uses the calibrated model to make a prediction for the output value
        '''
        schedule_value = self.get_schedule_value(dateTime)
        self.XY = np.array([schedule_value["X"], schedule_value["Y"]]).transpose()
        self.get_a_b_vectors()

        X = list(self.input.values())[0]
        key = list(self.output.keys())[0]
        self.output[key] = self.get_Y(X)

