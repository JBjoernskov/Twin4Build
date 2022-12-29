import twin4build.saref.device.sensor.sensor as sensor
class SensorModel(sensor.Sensor):
    def __init__(self,
                **kwargs):
        super().__init__(**kwargs)

    def update_output(self):
        self.output = self.input