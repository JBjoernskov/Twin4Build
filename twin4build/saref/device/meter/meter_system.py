from twin4build.saref.device.meter.meter import Meter
import numpy as np
import copy
class MeterSystem(Meter):
    def __init__(self,
                **kwargs):
        super().__init__(**kwargs)

    def initialize(self,
                    startPeriod=None,
                    endPeriod=None,
                    stepSize=None):
        self.inputUncertainty = copy.deepcopy(self.input)
        property_ = self.measuresProperty
        if property_.MEASURING_TYPE=="P":
            key = list(self.inputUncertainty.keys())[0]
            self.inputUncertainty[key] = property_.MEASURING_UNCERTAINTY/100
        else:
            key = list(self.inputUncertainty.keys())[0]
            self.inputUncertainty[key] = property_.MEASURING_UNCERTAINTY

    def do_step(self, secondTime=None, dateTime=None, stepSize=None):
        self.output = self.input
        self.outputUncertainty = self.inputUncertainty

    def get_subset_gradient(self, x_key, y_keys=None, as_dict=False):
        if as_dict==False:
            return np.array([1])
        else:
            return {key: 1 for key in y_keys}