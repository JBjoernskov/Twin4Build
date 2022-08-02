from .coil import Coil
class CoilModel(Coil):
    def __init__(self,
                specificHeatCapacityAir = None,
                isHeatingCoil = None,
                isCoolingCoil = None,
                **kwargs):
        super().__init__(**kwargs)
        self.specificHeatCapacityAir = specificHeatCapacityAir ###
        self.isHeatingCoil = isHeatingCoil ###
        self.isCoolingCoil = isCoolingCoil ###

    def update_output(self):
        if self.isHeatingCoil:
            if self.input["supplyAirTemperature"] < self.input["supplyAirTemperatureSetpoint"]:
                Q = self.input["supplyAirFlowRate"]*self.specificHeatCapacityAir*(self.input["supplyAirTemperatureSetpoint"] - self.input["supplyAirTemperature"])
            else:
                Q = 0
        elif self.isCoolingCoil:
            if self.input["supplyAirTemperature"] > self.input["supplyAirTemperatureSetpoint"]:
                Q = self.input["supplyAirFlowRate"]*self.specificHeatCapacityAir*(self.input["supplyAirTemperature"] - self.input["supplyAirTemperatureSetpoint"])
            else:
                Q = 0
        else:
            raise Exception("Coil is neither defined as heating or cooling. Set either \"isHeatingCoil\" or \"isCoolingCoil\" to True")

        self.output["Power"] = Q


        