from twin4build.saref4syst.system import System
from twin4build.saref.profile.schedule.schedule_system import ScheduleSystem
from twin4build.utils.piecewise_linear import PiecewiseLinearSystem
import numpy as np
from twin4build.utils.signature_pattern.signature_pattern import SignaturePattern, Node
import twin4build.base as base

def get_signature_pattern():
    node0 = Node(cls=(base.Schedule,))
    sp = SignaturePattern(ownedBy="PiecewiseLinearScheduleSystem", priority=0)
    sp.add_modeled_node(node0)
    return sp


class PiecewiseLinearScheduleSystem(PiecewiseLinearSystem, ScheduleSystem):
    sp = [get_signature_pattern()]
    def __init__(self,
                **kwargs):
        super().__init__(**kwargs)
        self.input = {}
        self.output = {}
        self._config = {"parameters": ["weekDayRulesetDict",
                                        "weekendRulesetDict",
                                        "mondayRulesetDict",
                                        "tuesdayRulesetDict",
                                        "wednesdayRulesetDict",
                                        "thursdayRulesetDict",
                                        "fridayRulesetDict",
                                        "saturdayRulesetDict",
                                        "sundayRulesetDict",
                                        "add_noise"]}

    @property
    def config(self):
        return self._config

    def do_step(self, secondTime=None, dateTime=None, stepSize=None):
        '''
             takes in the current time and uses the calibrated model to make a prediction for the output value
        '''
        schedule_value = self.get_schedule_value(dateTime)
        self.XY = np.array([schedule_value["X"], schedule_value["Y"]]).transpose()
        self.get_a_b_vectors()

        X = list(self.input.values())[0]
        key = list(self.output.keys())[0]
        self.output[key].set(self.get_Y(X))

