from .valve import Valve
import twin4build.saref.measurement.measurement as measurement
class ValveModel(Valve):
    def __init__(self, 
                waterFlowRateMax = None, 
                valveAuthority = None,
                **kwargs):
        super().__init__(**kwargs)
        assert isinstance(waterFlowRateMax, measurement.Measurement) or waterFlowRateMax is None, "Attribute \"closeOffRating\" is of type \"" + str(type(waterFlowRateMax)) + "\" but must be of type \"" + str(measurement.Measurement) + "\""
        assert isinstance(valveAuthority, measurement.Measurement) or valveAuthority is None, "Attribute \"flowCoefficient\" is of type \"" + str(type(valveAuthority)) + "\" but must be of type \"" + str(measurement.Measurement) + "\""
        self.waterFlowRateMax = waterFlowRateMax ###
        self.valveAuthority = valveAuthority ###

    def update_output(self):
        u_norm = self.input["valveSignal"]/(self.input["valveSignal"]**2*(1-self.valveAuthority.hasValue)+self.valveAuthority.hasValue)**(0.5)
        m_w = u_norm*self.waterFlowRateMax.hasValue
        self.output["waterFlowRate"] = m_w
    