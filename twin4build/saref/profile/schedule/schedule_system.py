
from random import randrange
import random
from twin4build.saref4syst.system import System
from twin4build.utils.signature_pattern.signature_pattern import SignaturePattern, Node, Exact, IgnoreIntermediateNodes
import twin4build.base as base
from twin4build.utils.time_series_input import TimeSeriesInputSystem


def get_signature_pattern():
    node0 = Node(cls=(base.Schedule,), id="<Schedule<SUB>1</SUB>>")
    sp = SignaturePattern(ownedBy="ScheduleSystem", priority=10)
    sp.add_modeled_node(node0)
    return sp

class ScheduleSystem(base.Schedule, System):
    sp = [get_signature_pattern()]
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
                parameterize_weekDayRulesetDict=False,
                useFile=False,
                filename=None,
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
        self.add_noise = add_noise
        self.parameterize_weekDayRulesetDict = parameterize_weekDayRulesetDict
        self.useFile = useFile
        self.filename = filename
        self.datecolumn = 0
        self.valuecolumn = 1
        random.seed(0)
        self.input = {}
        self.output = {"scheduleValue": None}
        self._config = {"parameters": ["weekDayRulesetDict",
                                        "weekendRulesetDict",
                                        "mondayRulesetDict",
                                        "tuesdayRulesetDict",
                                        "wednesdayRulesetDict",
                                        "thursdayRulesetDict",
                                        "fridayRulesetDict",
                                        "saturdayRulesetDict",
                                        "sundayRulesetDict",
                                        "add_noise",
                                        "parameterize_weekDayRulesetDict",
                                        "useFile"],
                        "readings": {"filename": self.filename,
                                     "datecolumn": self.datecolumn,
                                     "valuecolumn": self.valuecolumn}}

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
        self.noise = 0
        self.bias = 0
        assert (self.useFile and self.filename is None)==False, "filename must be provided if useFile is True."
        assert (self.useFile==False and self.weekDayRulesetDict is None)==False, "weekDayRulesetDict must be provided if useFile is False."

        
        if self.mondayRulesetDict is None:
            self.mondayRulesetDict = self.weekDayRulesetDict
        if self.tuesdayRulesetDict is None:
            self.tuesdayRulesetDict = self.weekDayRulesetDict
        if self.wednesdayRulesetDict is None:
            self.wednesdayRulesetDict = self.weekDayRulesetDict
        if self.thursdayRulesetDict is None:
            self.thursdayRulesetDict = self.weekDayRulesetDict
        if self.fridayRulesetDict is None:
            self.fridayRulesetDict = self.weekDayRulesetDict
        if self.saturdayRulesetDict is None:
            if self.weekendRulesetDict is None:
                self.saturdayRulesetDict = self.weekDayRulesetDict
            else:
                self.saturdayRulesetDict = self.weekendRulesetDict
        if self.sundayRulesetDict is None:
            if self.weekendRulesetDict is None:
                self.sundayRulesetDict = self.weekDayRulesetDict
            else:
                self.sundayRulesetDict = self.weekendRulesetDict
        assert self.useFile or self.weekDayRulesetDict is not None, "weekDayRulesetDict must be provided as argument."
        assert self.useFile or self.mondayRulesetDict is not None, "mondayRulesetDict must be provided as argument."
        assert self.useFile or self.tuesdayRulesetDict is not None, "tuesdayRulesetDict must be provided as argument."
        assert self.useFile or self.wednesdayRulesetDict is not None, "wednesdayRulesetDict must be provided as argument."
        assert self.useFile or self.thursdayRulesetDict is not None, "thursdayRulesetDict must be provided as argument."
        assert self.useFile or self.fridayRulesetDict is not None, "fridayRulesetDict must be provided as argument."
        assert self.useFile or self.saturdayRulesetDict is not None, "saturdayRulesetDict must be provided as argument."
        assert self.useFile or self.sundayRulesetDict is not None, "sundayRulesetDict must be provided as argument."

        

        if self.useFile:
            self.do_step_instance = TimeSeriesInputSystem(id=f"time series input - {self.id}", filename=self.filename, datecolumn=self.datecolumn, valuecolumn=self.valuecolumn)
            self.do_step_instance.input = self.input
            self.do_step_instance.output = self.output
            self.do_step_instance.initialize(startTime,
                                            endTime,
                                            stepSize)
        else:
            required_dicts = [self.mondayRulesetDict, self.tuesdayRulesetDict, self.wednesdayRulesetDict, self.thursdayRulesetDict, self.fridayRulesetDict, self.saturdayRulesetDict, self.sundayRulesetDict]
            required_keys = ["ruleset_start_minute", "ruleset_end_minute", "ruleset_start_hour", "ruleset_end_hour", "ruleset_value"]
            for rulesetDict in required_dicts:
                has_key = False
                len_key = None
                for key in required_keys:
                    if key in rulesetDict:
                        if len_key is not None:
                            assert len(rulesetDict[key])==len_key, "All keys in rulesetDict must have the same length."
                        len_key = len(rulesetDict[key])
                        has_key = True
                if has_key==False:
                    for key in required_keys:
                        rulesetDict[key] = []
                else:
                    for key in required_keys:
                        if key not in rulesetDict:
                            rulesetDict[key] = [0]*len_key



        

    def get_schedule_value(self, dateTime):
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
                schedule_value = rulesetDict["ruleset_value"][i_rule]
                found_match = True
                break
            elif rulesetDict["ruleset_start_hour"][i_rule] < dateTime.hour and dateTime.hour < rulesetDict["ruleset_end_hour"][i_rule]:
                schedule_value = rulesetDict["ruleset_value"][i_rule]
                found_match = True
                break
            elif rulesetDict["ruleset_end_hour"][i_rule] == dateTime.hour and dateTime.minute <= rulesetDict["ruleset_end_minute"][i_rule]:
                schedule_value = rulesetDict["ruleset_value"][i_rule]
                found_match = True
                break

        if found_match == False:
            schedule_value = rulesetDict["ruleset_default_value"]
        elif self.add_noise and schedule_value>0: 
            schedule_value += self.noise + self.bias
            if schedule_value<0:
                schedule_value = 0
        return schedule_value

    def do_step(self, secondTime=None, dateTime=None, stepSize=None):
        '''
            simulates a schedule and calculates the schedule value based on rulesets defined for different weekdays and times. 
            It also adds noise and bias to the calculated value.
        '''
        if self.useFile:
            self.do_step_instance.do_step(secondTime, dateTime, stepSize)
            self.output = self.do_step_instance.output
        else:
            self.output["scheduleValue"] = self.get_schedule_value(dateTime)

        