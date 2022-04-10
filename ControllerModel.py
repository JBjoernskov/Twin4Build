
class ControllerModel():
    def __init__(self):
        pass

    def update_output(self):
        if self.isTemperatureController:
            self.output["valveSignal"] = 1
        elif self.isCo2Controller:
            self.output["supplyDamperSignal"] = 1
            self.output["returnDamperSignal"] = 1
        else:
            raise Exception("Controller is neither defined as temperature or CO2 controller. Set either \"isTemperatureController\" or \"isCo2Controller\" to True")
