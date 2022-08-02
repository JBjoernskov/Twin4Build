from ..energy_conversion_device import EnergyConversionDevice
class Coil(EnergyConversionDevice):
    def __init__(self,
                airFlowRateMax = None,
                airFlowRateMin = None,
                nominalLatentCapacity = None,
                nominalSensibleCapacity = None,
                nominalUa = None,
                operationTemperatureMax = None,
                operationTemperatureMin = None,
                placementType = None,
                **kwargs):
        super().__init__(**kwargs)
        self.airFlowRateMax = airFlowRateMax
        self.airFlowRateMin = airFlowRateMin
        self.nominalLatentCapacity = nominalLatentCapacity
        self.nominalSensibleCapacity = nominalSensibleCapacity
        self.nominalUa = nominalUa
        self.operationTemperatureMax = operationTemperatureMax
        self.operationTemperatureMin = operationTemperatureMin
        self.placementType = placementType