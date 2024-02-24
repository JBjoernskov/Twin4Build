import twin4build.saref.property_.temperature.outlet_temperature.outlet_temperature as outlet_temperature
class SecondaryOutletTemperature(outlet_temperature.OutletTemperature):
    def __init__(self,
                **kwargs):
        super().__init__(**kwargs)
