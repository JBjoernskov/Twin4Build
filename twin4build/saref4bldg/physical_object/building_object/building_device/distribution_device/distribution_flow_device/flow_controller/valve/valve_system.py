from .valve import Valve
import numpy as np

class ValveSystem(Valve):
    def __init__(self, 
                waterFlowRateMax=None, 
                valveAuthority=None,
                **kwargs):
        super().__init__(**kwargs)
        assert isinstance(waterFlowRateMax, float) or waterFlowRateMax is None, "Attribute \"closeOffRating\" is of type \"" + str(type(waterFlowRateMax)) + "\" but must be of type \"" + str(float) + "\""
        assert isinstance(valveAuthority, float) or isinstance(valveAuthority, int) or valveAuthority is None, "Attribute \"valveAuthority\" is of type \"" + str(type(valveAuthority)) + "\" but must be of type \"" + str(float) + "\""
        self.waterFlowRateMax = waterFlowRateMax
        self.valveAuthority = valveAuthority

        self.input = {"valvePosition": None}
        self.output = {"waterFlowRate": None}
        self.outputUncertainty = {"waterFlowRate": 0}
        self._config = {"parameters": ["waterFlowRateMax",
                                       "valveAuthority"]}

    @property
    def config(self):
        return self._config

    def cache(self,
            startTime=None,
            endTime=None,
            stepSize=None):
        pass

    def initialize(self,
                    startTime=None,
                    endTime=None,
                    stepSize=None,
                    model=None):
        pass

    def do_step(self, secondTime=None, dateTime=None, stepSize=None):
        u_norm = self.input["valvePosition"]/(self.input["valvePosition"]**2*(1-self.valveAuthority)+self.valveAuthority)**(0.5)
        m_w = u_norm*self.waterFlowRateMax
        self.output["valvePosition"] = self.input["valvePosition"]
        self.output["waterFlowRate"] = m_w

    def get_subset_gradient(self, x_key, y_keys=None, as_dict=False):
        if x_key=="valvePosition":
            grad = [(self.valveAuthority*self.waterFlowRateMax)/((-self.valveAuthority*self.input["valvePosition"]+self.valveAuthority+self.input["valvePosition"]**2)**(3/2))]
        
        if x_key=="valveAuthority":
            grad = [(0.5*self.waterFlowRateMax*self.input["valvePosition"]*(self.input["valvePosition"]**2-1))/((-self.valveAuthority*self.input["valvePosition"]**2+self.valveAuthority+self.input["valvePosition"]**2)**1.5)]

        if x_key=="waterFlowRateMax":
            grad = [self.output["waterFlowRate"]/self.waterFlowRateMax]
        if as_dict==False:
            grad = np.array([grad])
        else:
            grad = {key: value for key, value in zip(y_keys, grad)}
        return grad