from .valve import Valve
import twin4build.saref.measurement.measurement as measurement
class ValveModel(Valve):
    def __init__(self, 
                waterFlowRateMax = None, 
                valveAuthority = None,
                **kwargs):
        super().__init__(**kwargs)
        # assert isinstance(waterFlowRateMax, measurement.Measurement) or waterFlowRateMax is None, "Attribute \"closeOffRating\" is of type \"" + str(type(waterFlowRateMax)) + "\" but must be of type \"" + str(measurement.Measurement) + "\""
        assert isinstance(valveAuthority, measurement.Measurement) or valveAuthority is None, "Attribute \"valveAuthority\" is of type \"" + str(type(valveAuthority)) + "\" but must be of type \"" + str(measurement.Measurement) + "\""
        self.waterFlowRateMax = self.flowCoefficient.hasValue/(1/self.testPressure.hasValue)**0.5/3600*1000
        self.valveAuthority = valveAuthority ###

    def update_output(self):
        u_norm = self.input["valvePosition"]/(self.input["valvePosition"]**2*(1-self.valveAuthority.hasValue)+self.valveAuthority.hasValue)**(0.5)
        m_w = u_norm*self.waterFlowRateMax
        self.output["waterFlowRate"] = m_w
    