from scipy.optimize import least_squares
import numpy as np
import twin4build.utils.input_output_types as tps
import twin4build.core as core
import datetime
from typing import Optional
from twin4build.translator.translator import SignaturePattern, Node, Exact, SinglePath, MultiPath, Optional_

def get_signature_pattern():
    node0 = Node(cls=core.S4BLDG.SetpointController)
    node1 = Node(cls=core.SAREF.Sensor)
    node2 = Node(cls=core.SAREF.Property)
    node3 = Node(cls=core.S4BLDG.Schedule)
    node4 = Node(cls=core.XSD.boolean)
    sp = SignaturePattern(semantic_model_=core.ontologies, ownedBy="PIControllerFMUSystem")
    sp.add_triple(Exact(subject=node0, object=node2, predicate=core.SAREF.observes))
    sp.add_triple(Exact(subject=node1, object=node2, predicate=core.SAREF.observes))
    sp.add_triple(Exact(subject=node0, object=node3, predicate=core.SAREF.hasProfile))
    sp.add_triple(Exact(subject=node0, object=node4, predicate=core.S4BLDG.isReverse))

    sp.add_input("actualValue", node1, "measuredValue")
    sp.add_input("setpointValue", node3, "scheduleValue")
    sp.add_parameter("isReverse", node4)
    sp.add_modeled_node(node0)
    return sp

class PIDControllerSystem(core.System):
    sp = [get_signature_pattern()]
    def __init__(self, 
                # isTemperatureController=None,
                # isCo2Controller=None,
                kp=None,
                ki=None,
                kd=None,
                **kwargs):
        super().__init__(**kwargs)
        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.input = {"actualValue": tps.Scalar(),
                    "setpointValue": tps.Scalar()}
        self.output = {"inputSignal": tps.Scalar()}
        self._config = {"parameters": ["kp",
                                       "ki",
                                       "kd"]}

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
                    simulator=None):
        self.acc_err = 0
        self.prev_err = 0

    def do_step(self, 
                secondTime: Optional[float] = None, 
                dateTime: Optional[datetime.datetime] = None, 
                stepSize: Optional[float] = None, 
                stepIndex: Optional[int] = None) -> None:
        err = self.input["setpointValue"]-self.input["actualValue"]
        p = err*self.kp
        i = self.acc_err*self.ki
        d = (err-self.prev_err)*self.kd
        signal_value = p + i + d
        if signal_value>1:
            signal_value = 1
            self.acc_err = 1/self.ki
            self.prev_err = 0
        elif signal_value<0:
            signal_value = 0
            self.acc_err = 0
            self.prev_err = 0
        else:
            self.acc_err += err
            self.prev_err = err

        self.output["inputSignal"].set(signal_value, stepIndex)
