from twin4build.saref4syst.system import System
import datetime
from random import randrange


class Schedule(System):
    def __init__(self,
                startPeriod = None,
                timeStep = None,
                rulesetDict = None,
                add_noise = False,
                **kwargs):
        super().__init__(**kwargs)

        self.time = startPeriod
        self.timeStep = timeStep
        self.rulesetDict = rulesetDict
        self.add_noise = add_noise

    def update_output(self):
        n = len(self.rulesetDict["ruleset_start_hour"])
        found_match = False
        for i_rule in range(n):
            if self.rulesetDict["ruleset_start_hour"][i_rule] == self.time.hour and self.time.minute >= self.rulesetDict["ruleset_start_minute"][i_rule]:
                self.output["scheduleValue"] = self.rulesetDict["ruleset_value"][i_rule]
                found_match = True
                break
            elif self.rulesetDict["ruleset_start_hour"][i_rule] < self.time.hour and self.time.hour < self.rulesetDict["ruleset_end_hour"][i_rule]:
                self.output["scheduleValue"] = self.rulesetDict["ruleset_value"][i_rule]
                found_match = True
                break
            elif self.rulesetDict["ruleset_end_hour"][i_rule] == self.time.hour and self.time.minute <= self.rulesetDict["ruleset_end_minute"][i_rule]:
                self.output["scheduleValue"] = self.rulesetDict["ruleset_value"][i_rule]
                found_match = True
                break

           
        
        if found_match == False:
            self.output["scheduleValue"] = self.rulesetDict["ruleset_default_value"]
        elif self.add_noise:
            
            if self.time.minute==0:
                noise =  randrange(-5,5)
                self.rulesetDict["ruleset_value"][i_rule] += noise
                self.output["scheduleValue"] += noise

            if self.output["scheduleValue"]<0:
                self.rulesetDict["ruleset_value"][i_rule] = 0
                self.output["scheduleValue"] = 0

        self.time += datetime.timedelta(seconds = self.timeStep)
