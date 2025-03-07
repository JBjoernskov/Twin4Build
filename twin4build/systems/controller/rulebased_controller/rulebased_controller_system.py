import twin4build.utils.input_output_types as tps
import twin4build.core as core

class RulebasedControllerSystem(core.System):
    def __init__(self, 
                **kwargs):
        super().__init__(**kwargs)
        self.input = {"actualValue": tps.Scalar()}
        self.output = {"inputSignal": tps.Scalar()}
        self.interval = 99
        self._config = {"parameters": ["interval"]}

    @property
    def config(self):
        return self._config

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
        self.hold_900_signal = False
        self.hold_750_signal = False
        self.hold_600_signal = False

    def do_step(self, secondTime=None, dateTime=None, stepSize=None):
        if self.input["actualValue"]>900 or self.hold_900_signal:
            self.output["inputSignal"].set(1)
            if self.input["actualValue"]>900-self.interval:
                self.hold_900_signal = True
            else:
                self.hold_900_signal = False
        
        elif self.input["actualValue"]>750 or self.hold_750_signal:
            self.output["inputSignal"].set(0.7)
            if self.input["actualValue"]>750-self.interval:
                self.hold_750_signal = True
            else:
                self.hold_750_signal = False

        elif self.input["actualValue"]>600 or self.hold_600_signal:
            self.output["inputSignal"].set(0.45)
            if self.input["actualValue"]>600-self.interval:
                self.hold_600_signal = True
            else:
                self.hold_600_signal = False

        else:
            self.output["inputSignal"].set(0)



