import twin4build.saref.property_.temperature.inlet_temperature.inlet_temperature as inlet_temperature
class PrimaryInletTemperature(inlet_temperature.InletTemperature):
    def __init__(self,
                **kwargs):
        super().__init__(**kwargs)