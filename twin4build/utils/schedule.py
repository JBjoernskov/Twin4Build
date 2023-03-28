from twin4build.saref4syst.system import System
import datetime
from random import randrange
import random

class Schedule(System):
    def __init__(self,
                weekDayRulesetDict=None,
                weekendRulesetDict=None,
                mondayRulesetDict=None,
                tuesdayRulesetDict=None,
                wednesdayRulesetDict=None,
                thursdayRulesetDict=None,
                fridayRulesetDict=None,
                saturdayRulesetDict=None,
                sundayRulesetDict=None,
                add_noise = False,
                **kwargs):
        super().__init__(**kwargs)

        self.weekDayRulesetDict = weekDayRulesetDict
        self.weekendRulesetDict = weekendRulesetDict
        self.mondayRulesetDict = mondayRulesetDict
        self.tuesdayRulesetDict = tuesdayRulesetDict
        self.wednesdayRulesetDict = wednesdayRulesetDict
        self.thursdayRulesetDict = thursdayRulesetDict
        self.fridayRulesetDict = fridayRulesetDict
        self.saturdayRulesetDict = saturdayRulesetDict
        self.sundayRulesetDict = sundayRulesetDict


        if mondayRulesetDict is None:
            self.mondayRulesetDict = self.weekDayRulesetDict
        if tuesdayRulesetDict is None:
            self.tuesdayRulesetDict = self.weekDayRulesetDict
        if wednesdayRulesetDict is None:
            self.wednesdayRulesetDict = self.weekDayRulesetDict
        if thursdayRulesetDict is None:
            self.thursdayRulesetDict = self.weekDayRulesetDict
        if fridayRulesetDict is None:
            self.fridayRulesetDict = self.weekDayRulesetDict
        if saturdayRulesetDict is None:
            if weekendRulesetDict is None:
                self.saturdayRulesetDict = self.weekDayRulesetDict
            else:
                self.saturdayRulesetDict = self.weekendRulesetDict
        if sundayRulesetDict is None:
            if weekendRulesetDict is None:
                self.sundayRulesetDict = self.weekDayRulesetDict
            else:
                self.sundayRulesetDict = self.weekendRulesetDict



        self.add_noise = add_noise
        random.seed(0)

        self.input = {}
        self.output = {"scheduleValue": None}

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


        if dateTime.weekday()==0: 
            rulesetDict = self.mondayRulesetDict
        elif dateTime.weekday()==1:
            rulesetDict = self.tuesdayRulesetDict
        elif dateTime.weekday()==2:
            rulesetDict = self.wednesdayRulesetDict
        elif dateTime.weekday()==3:
            rulesetDict = self.thursdayRulesetDict
        elif dateTime.weekday()==4:
            rulesetDict = self.fridayRulesetDict
        elif dateTime.weekday()==5:
            rulesetDict = self.saturdayRulesetDict
        elif dateTime.weekday()==6:
            rulesetDict = self.sundayRulesetDict

        n = len(rulesetDict["ruleset_start_hour"])
        found_match = False
        for i_rule in range(n):
            if rulesetDict["ruleset_start_hour"][i_rule] == dateTime.hour and dateTime.minute >= rulesetDict["ruleset_start_minute"][i_rule]:
                self.output["scheduleValue"] = rulesetDict["ruleset_value"][i_rule]
                found_match = True
                break
            elif rulesetDict["ruleset_start_hour"][i_rule] < dateTime.hour and dateTime.hour < rulesetDict["ruleset_end_hour"][i_rule]:
                self.output["scheduleValue"] = rulesetDict["ruleset_value"][i_rule]
                found_match = True
                break
            elif rulesetDict["ruleset_end_hour"][i_rule] == dateTime.hour and dateTime.minute <= rulesetDict["ruleset_end_minute"][i_rule]:
                self.output["scheduleValue"] = rulesetDict["ruleset_value"][i_rule]
                found_match = True
                break

        if found_match == False:
            self.output["scheduleValue"] = rulesetDict["ruleset_default_value"]
        elif self.add_noise and self.output["scheduleValue"]>0: 
            self.output["scheduleValue"] += self.noise + self.bias
            if self.output["scheduleValue"]<0:
                self.output["scheduleValue"] = 0