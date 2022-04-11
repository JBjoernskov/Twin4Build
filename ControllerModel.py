
class ControllerModel():
    def __init__(self, 
                **kwargs):
        super().__init__(**kwargs)
        self.acc_err = 0
        self.prev_err = 0

    def update_output(self):
        if self.isTemperatureController:
            self.output["valveSignal"] = 1

        elif self.isCo2Controller:

            err = self.input["indoorCo2Concentration"]-self.input["indoorCo2ConcentrationSetpoint"]

            p = err*self.k_p
            i = self.acc_err*self.k_i
            d = (err-self.prev_err)*self.k_d

            signal_value = p + i + d
            
            if signal_value>1:
                signal_value = 1
            elif signal_value<0:
                signal_value = 0

            self.acc_err += err
            self.prev_err = err


            self.output["supplyDamperSignal"] = signal_value
            self.output["returnDamperSignal"] = signal_value
        else:
            raise Exception("Controller is neither defined as temperature or CO2 controller. Set either \"isTemperatureController\" or \"isCo2Controller\" to True")
