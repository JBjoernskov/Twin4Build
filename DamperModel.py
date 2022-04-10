
import math

class DamperModel():
    def __init__(self):
        pass

    def update_output(self):
        m_a = self.a*math.exp(self.b*self.input[self.DamperSignalName]) + self.c
        self.output[self.AirFlowRateName] = m_a


    