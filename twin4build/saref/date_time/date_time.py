from __future__ import annotations
from typing import Union
import datetime
from dateutil.tz import tzutc
class DateTime(datetime.datetime):
    def __new__(self, 
                year: Union[int, None]=None,
                month: Union[int, None]=None,
                day: Union[int, None]=None,
                hour: Union[int, None]=None,
                minute: Union[int, None]=None,
                second: Union[int, None]=None):
        # assert isinstance(year, int) or year is None, "Attribute \"year\" is of type \"" + str(type(year)) + "\" but must be of type \"" + str(int) + "\""
        # assert isinstance(month, int) or month is None, "Attribute \"month\" is of type \"" + str(type(month)) + "\" but must be of type \"" + str(int) + "\""
        # assert isinstance(day, int) or day is None, "Attribute \"day\" is of type \"" + str(type(day)) + "\" but must be of type \"" + str(int) + "\""
        # assert isinstance(hour, int) or hour is None, "Attribute \"hour\" is of type \"" + str(type(hour)) + "\" but must be of type \"" + str(int) + "\""
        # assert isinstance(minute, int) or minute is None, "Attribute \"minute\" is of type \"" + str(type(minute)) + "\" but must be of type \"" + str(int) + "\""
        # assert isinstance(second, int) or second is None, "Attribute \"second\" is of type \"" + str(type(second)) + "\" but must be of type \"" + str(int) + "\""
        return datetime.datetime.__new__(self,
                                        year=year, 
                                        month=month, 
                                        day=day, 
                                        hour=hour, 
                                        minute=minute, 
                                        second=second, 
                                        tzinfo=tzutc())
        