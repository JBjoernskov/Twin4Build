from .controller import Controller
class ControllerModel(Controller):
    def __init__(self, 
                isTemperatureController = None,
                isCo2Controller = None,
                k_p = None,
                k_i = None,
                k_d = None,
                **kwargs):
        super().__init__(**kwargs)
        self.acc_err = 0
        self.prev_err = 0

        self.isTemperatureController = isTemperatureController ###
        self.isCo2Controller = isCo2Controller ###
        self.k_p = k_p
        self.k_i = k_i
        self.k_d = k_d

    def update_output(self):
        if self.isTemperatureController:
            err = self.input["indoorTemperatureSetpoint"]-self.input["indoorTemperature"]
            p = err*self.k_p
            i = self.acc_err*self.k_i
            d = (err-self.prev_err)*self.k_d
            signal_value = p + i + d
            if signal_value>1:
                signal_value = 1
                self.acc_err = 0
                self.prev_err = 0
            elif signal_value<0:
                signal_value = 0
                self.acc_err = 0
                self.prev_err = 0
            else:
                self.acc_err += err
                self.prev_err = err

            self.output["valveSignal"] = signal_value

        elif self.isCo2Controller:
            err = self.input["indoorCo2Concentration"]-self.input["indoorCo2ConcentrationSetpoint"]
            p = err*self.k_p
            i = self.acc_err*self.k_i
            d = (err-self.prev_err)*self.k_d
            signal_value = p + i + d #round( ,1)
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
