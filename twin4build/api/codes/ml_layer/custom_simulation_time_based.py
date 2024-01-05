
import pytz , os , sys
from datetime import datetime , timedelta

from request_to_api import request_class

class CustomRequestSimulation:
    def __init__(self, start_time, end_time,warm_up) -> None:
        # Configuration function called of the request class
        self.request_obj = request_class()
        self.denmark_timezone = pytz.timezone('Europe/Copenhagen')
        self.time_format = '%Y-%m-%d %H:%M:%S%z'

        self.start_time = self.get_formatted_time(start_time)
        self.end_time = self.get_formatted_time(end_time)
        self.warm_up = self.start_time - timedelta(hours=warm_up)

        self.request_simulation()

    def get_formatted_time(self, raw_time):
        # Create datetime object from raw time string
        time = datetime.strptime(raw_time, '%Y,%m,%d,%H,%M,%S')
        
        # Add timezone information
        time = self.denmark_timezone.localize(time)

        # Format the time
        formatted_time = time.strftime(self.time_format)
        return datetime.strptime(formatted_time,self.time_format)
    
    def request_simulation(self):
        self.request_obj.request_to_simulator_api(self.start_time,self.end_time,self.warm_up,False)


if __name__ == "__main__":

    # format for time input : year , month , day, hours , minute , second
    start_time_str = "2024,01,01,12,14,22"
    end_time_str = "2024,01,05,12,14,22"
    warm_up = 12

    custom_request_simulation = CustomRequestSimulation(start_time_str, end_time_str,warm_up)