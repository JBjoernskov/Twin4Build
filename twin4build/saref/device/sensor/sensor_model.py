import twin4build.saref.device.sensor.sensor as sensor
class SensorModel(sensor.Sensor):
    def __init__(self,
                **kwargs):
        super().__init__(**kwargs)

    def initialize(self,
                    startPeriod=None,
                    endPeriod=None,
                    stepSize=None):
        pass

    def do_step(self, secondTime=None, dateTime=None, stepSize=None):
        self.output = self.input