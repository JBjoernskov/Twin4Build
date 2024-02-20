import twin4build.saref.property_.temperature.temperature as temperature
class InletTemperature(temperature.Temperature):
    def __init__(self,
                **kwargs):
        super().__init__(**kwargs)