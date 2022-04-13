import Saref4Syst
import datetime

class Schedule(Saref4Syst.System):
    def __init__(self,
                startPeriod = None,
                timeStep = None,
                rulesetDict = None,
                **kwargs):
        super().__init__(**kwargs)

        self.time = startPeriod
        self.timeStep = timeStep
        self.rulesetDict = rulesetDict

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

        self.time += datetime.timedelta(seconds = self.timeStep)
