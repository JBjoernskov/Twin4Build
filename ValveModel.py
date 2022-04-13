import Saref4Syst

class ValveModel(Saref4Syst.System):
    def __init__(self, 
                waterFlowRateMax = None, 
                valveAuthority = None,
                **kwargs):
        super().__init__(**kwargs)

        self.waterFlowRateMax = waterFlowRateMax ###
        self.valveAuthority = valveAuthority ###

    def update_output(self):
        u_norm = self.input["valveSignal"]/(self.input["valveSignal"]**2*(1-self.valveAuthority)+self.valveAuthority)**(0.5)
        m_w = u_norm*self.waterFlowRateMax
        self.output["waterFlowRate"] = m_w
    