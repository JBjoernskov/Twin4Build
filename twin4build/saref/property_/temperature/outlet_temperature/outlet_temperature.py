import twin4build.saref.property_.temperature.temperature as temperature
class OutletTemperature(temperature.Temperature):
    def __init__(self,
                **kwargs):
        super().__init__(**kwargs)
