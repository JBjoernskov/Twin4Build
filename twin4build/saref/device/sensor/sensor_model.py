import twin4build.saref.device.sensor.sensor as sensor
class SensorModel(sensor.Sensor):
    def __init__(self,
                **kwargs):
        super().__init__(**kwargs)

    def do_step(self):
        self.output = self.input