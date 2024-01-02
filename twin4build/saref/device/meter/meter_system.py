from twin4build.saref.device.meter.meter import Meter
from twin4build.utils.time_series_input import TimeSeriesInputSystem
from twin4build.utils.pass_input_to_output import PassInputToOutput
import numpy as np
import copy
class MeterSystem(Meter):
    def __init__(self,
                 physicalSystemFilename=None,
                 addUncertainty=False,
                **kwargs):
        super().__init__(**kwargs)
        self.physicalSystemFilename = physicalSystemFilename
        self.addUncertainty = addUncertainty
        if self.physicalSystemFilename is not None:
            self.physicalSystem = TimeSeriesInputSystem(id=f"time series input - {self.id}", filename=self.physicalSystemFilename)

    def cache(self,
            startTime=None,
            endTime=None,
            stepSize=None):
        pass
        
    def initialize(self,
                    startTime=None,
                    endTime=None,
                    stepSize=None):
        assert (len(self.connectsAt)==0 and self.physicalSystemFilename is None)==False, f"Meter object \"{self.id}\" has no inputs and the argument \"physicalSystemFilename\" in the constructor was not provided."
        if len(self.connectsAt)==0:
            self.do_step_instance = self.physicalSystem
            self.isPhysicalSystem = True
        else:
            self.do_step_instance = PassInputToOutput(id="pass input to output")
            self.isPhysicalSystem = False
        self.do_step_instance.input = self.input
        self.do_step_instance.output = self.output
        self.do_step_instance.initialize(startTime,
                                        endTime,
                                        stepSize)

        self.inputUncertainty = copy.deepcopy(self.input)
        percentile = 2
        self.standardDeviation = self.measuresProperty.MEASURING_UNCERTAINTY/percentile
        # property_ = self.measuresProperty
        # if property_.MEASURING_TYPE=="P":
        #     key = list(self.inputUncertainty.keys())[0]
        #     self.inputUncertainty[key] = property_.MEASURING_UNCERTAINTY/100
        # else:
        #     key = list(self.inputUncertainty.keys())[0]
        #     self.inputUncertainty[key] = property_.MEASURING_UNCERTAINTY

    def do_step(self, secondTime=None, dateTime=None, stepSize=None):
        self.do_step_instance.input = self.input
        self.do_step_instance.do_step(secondTime=secondTime, dateTime=dateTime, stepSize=stepSize)
        if self.addUncertainty:
            for key in self.do_step_instance.output:
                self.output[key] = self.do_step_instance.output[key] + np.random.normal(0, self.standardDeviation)
        else:
            self.output = self.do_step_instance.output

    def get_subset_gradient(self, x_key, y_keys=None, as_dict=False):
        if as_dict==False:
            return np.array([1])
        else:
            return {key: 1 for key in y_keys}
        
    def get_physical_readings(self,
                            startTime=None,
                            endTime=None,
                            stepSize=None):
        self.physicalSystem.initialize(startTime,
                                        endTime,
                                        stepSize)
        return self.physicalSystem.physicalSystemReadings