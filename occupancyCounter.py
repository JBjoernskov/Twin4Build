import Saref4Syst
import datetime

class occupancyCounter(Saref4Syst.System):
    def __init__(self,
                timeStep = None,
                startPeriod = None,
                **kwargs):
        super().__init__(**kwargs)
        self.timeStep = timeStep
        self.time = startPeriod

        self.ruleset_default_value = 0
        self.ruleset_start_minute = [0,0,0,0,0]
        self.ruleset_end_minute = [60,60,60,60,60]
        self.ruleset_start_hour = [0,6,8,12,18]
        self.ruleset_end_hour = [6,8,12,18,22]
        self.ruleset_value = [0,5,15,30,15]


        self.ruleset_dict = {
            "ruleset_default_value": self.ruleset_default_value,
            "ruleset_start_minute": self.ruleset_start_minute,
            "ruleset_end_minute": self.ruleset_end_minute,
            "ruleset_start_hour": self.ruleset_start_hour,
            "ruleset_end_hour": self.ruleset_end_hour,
            "ruleset_value": self.ruleset_value
        }

        

    def get_occupancy(self):
        n = len(self.ruleset_dict["ruleset_start_hour"])
        found_match = False
        for i_rule in range(n):
            cond_hour = self.time.hour >= self.ruleset_dict["ruleset_start_hour"][i_rule] and self.time.hour <= self.ruleset_dict["ruleset_end_hour"][i_rule]
            cond_minute = self.time.minute >= self.ruleset_dict["ruleset_start_minute"][i_rule] and self.time.minute <= self.ruleset_dict["ruleset_end_minute"][i_rule]
            if cond_hour and cond_minute:
                setpoint = self.ruleset_dict["ruleset_value"][i_rule]
                found_match = True
        
        if found_match == False:
            setpoint = self.ruleset_dict["ruleset_default_value"]
                    
        return setpoint


    def update_output(self):
        self.output["numberOfPeople"] = self.get_occupancy()
        self.time += datetime.timedelta(minutes = self.timeStep)