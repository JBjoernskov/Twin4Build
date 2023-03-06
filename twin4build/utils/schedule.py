from twin4build.saref4syst.system import System
import datetime
from random import randrange
import random

class Schedule(System):
    def __init__(self,
                rulesetDict=None,
                add_noise = False,
                **kwargs):
        super().__init__(**kwargs)

        self.rulesetDict = rulesetDict
        self.add_noise = add_noise
        random.seed(0)

    def initialize(self,
                    startPeriod=None,
                    endPeriod=None,
                    stepSize=None):
        pass

    def do_step(self, secondTime=None, dateTime=None, stepSize=None):
        if dateTime.minute==0: #Compute a new noise value if a new hour is entered in the simulation
            self.noise = randrange(-4,4)
        if dateTime.hour==0 and dateTime.minute==0: #Compute a new bias value if a new day is entered in the simulation
            self.bias = randrange(-10,10)

        n = len(self.rulesetDict["ruleset_start_hour"])
        found_match = False
        for i_rule in range(n):
            if self.rulesetDict["ruleset_start_hour"][i_rule] == dateTime.hour and dateTime.minute >= self.rulesetDict["ruleset_start_minute"][i_rule]:
                self.output["scheduleValue"] = self.rulesetDict["ruleset_value"][i_rule]
                found_match = True
                break
            elif self.rulesetDict["ruleset_start_hour"][i_rule] < dateTime.hour and dateTime.hour < self.rulesetDict["ruleset_end_hour"][i_rule]:
                self.output["scheduleValue"] = self.rulesetDict["ruleset_value"][i_rule]
                found_match = True
                break
            elif self.rulesetDict["ruleset_end_hour"][i_rule] == dateTime.hour and dateTime.minute <= self.rulesetDict["ruleset_end_minute"][i_rule]:
                self.output["scheduleValue"] = self.rulesetDict["ruleset_value"][i_rule]
                found_match = True
                break

        if found_match == False:
            self.output["scheduleValue"] = self.rulesetDict["ruleset_default_value"]
        elif self.add_noise and self.output["scheduleValue"]>0: 
            self.output["scheduleValue"] += self.noise + self.bias
            if self.output["scheduleValue"]<0:
                self.output["scheduleValue"] = 0