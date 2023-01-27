import twin4build.saref.device.meter.meter as meter
class MeterModel(meter.Meter):
    def __init__(self,
                **kwargs):
        super().__init__(**kwargs)

    def initialize(self):
        pass

    def do_step(self, time=None, stepSize=None):
        self.output = self.input