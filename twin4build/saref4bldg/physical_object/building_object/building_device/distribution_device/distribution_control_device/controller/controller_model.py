from .controller import Controller
class ControllerModel(Controller):
    def __init__(self, 
                # isTemperatureController = None,
                # isCo2Controller = None,
                K_p = None,
                K_i = None,
                K_d = None,
                **kwargs):
        super().__init__(**kwargs)
        self.acc_err = 0
        self.prev_err = 0

        # self.isTemperatureController = isTemperatureController ###
        # self.isCo2Controller = isCo2Controller ###
        self.K_p = K_p
        self.K_i = K_i
        self.K_d = K_d

    def update_output(self):
        # if self.isTemperatureController:
        err = self.input["setpointValue"]-self.input["actualValue"]
        p = err*self.K_p
        i = self.acc_err*self.K_i
        d = (err-self.prev_err)*self.K_d
        signal_value = p + i + d
        if signal_value>1:
            signal_value = 1
            # self.acc_err = 0
            self.prev_err = 0
        elif signal_value<1e-2:
            signal_value = 0
            # self.acc_err = 0
            self.prev_err = 0
        else:
            self.acc_err += err
            self.prev_err = err

        self.output["inputSignal"] = signal_value

        # elif self.isCo2Controller:
        #     err = self.input["indoorCo2Concentration"]-self.input["indoorCo2ConcentrationSetpoint"]
        #     p = err*self.k_p
        #     i = self.acc_err*self.k_i
        #     d = (err-self.prev_err)*self.k_d
        #     signal_value = p + i + d #round( ,1)
        #     if signal_value>1:
        #         signal_value = 1
        #     elif signal_value<0:
        #         signal_value = 0
        #     self.acc_err += err
        #     self.prev_err = err
        #     self.output["supplyDamperSignal"] = signal_value
        #     self.output["returnDamperSignal"] = signal_value
        # else:
        #     raise Exception("Controller is neither defined as temperature or CO2 controller. Set either \"isTemperatureController\" or \"isCo2Controller\" to True")
