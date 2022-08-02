from ..energy_conversion_device import EnergyConversionDevice
class AirToAirHeatRecovery(EnergyConversionDevice):
    def __init__(self,
                hasDefrost = None,
                heatTransferTypeEnum = None,
                operationTemperatureMax = None,
                operationTemperatureMin = None,
                primaryAirFlowRateMax = None,
                primaryAirFlowRateMin = None,
                secondaryAirFlowRateMax = None,
                secondaryAirFlowRateMin = None,
                **kwargs):
        super().__init__(**kwargs)
        self.hasDefrost = hasDefrost
        self.heatTransferTypeEnum = heatTransferTypeEnum
        self.operationTemperatureMax = operationTemperatureMax
        self.operationTemperatureMin = operationTemperatureMin
        self.primaryAirFlowRateMax = primaryAirFlowRateMax
        self.primaryAirFlowRateMin = primaryAirFlowRateMin
        self.secondaryAirFlowRateMax = secondaryAirFlowRateMax
        self.secondaryAirFlowRateMin = secondaryAirFlowRateMin