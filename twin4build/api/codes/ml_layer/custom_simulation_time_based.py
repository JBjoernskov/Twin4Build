
import pytz , os , sys
from datetime import datetime , timedelta , timezone

###Only for testing before distributing package
if __name__ == '__main__':
    # Define a function to move up in the directory hierarchy
    uppath = lambda _path, n: os.sep.join(_path.split(os.sep)[:-n])
    # Calculate the file path using the uppath function
    file_path = uppath(os.path.abspath(__file__), 5)
    # Append the calculated file path to the system path
    sys.path.append(file_path)

from twin4build.api.codes.ml_layer.request_to_api import request_class

class CustomRequestSimulation:
    def __init__(self, start_time, end_time,warm_up) -> None:
        # Configuration function called of the request class
        self.request_obj = request_class()
        self.denmark_timezone = pytz.timezone('Europe/Copenhagen')
        self.time_format = '%Y-%m-%d %H:%M:%S%z'

        self.start_time = start_time
        self.end_time = end_time
        self.warm_up = warm_up

        self.request_simulation()

    def get_formatted_time(self, start_time, end_time,warm_up):
        # Create datetime object from raw time string
        start_time = datetime.strptime(start_time, '%Y,%m,%d,%H,%M,%S')
        start_time = start_time.replace(tzinfo=timezone(timedelta(hours=1)))

        warm_uptime = start_time - timedelta(hours=warm_up)
        formatted_warmup_time = warm_uptime.strftime(self.time_format)

        formatted_start_time = start_time.strftime(self.time_format)
 
        end_time = datetime.strptime(end_time, '%Y,%m,%d,%H,%M,%S')
        end_time = end_time.replace(tzinfo=timezone(timedelta(hours=1)))
        formatted_end_time = end_time.strftime(self.time_format)

        return formatted_start_time,formatted_end_time,formatted_warmup_time
    
    def request_simulation(self):
        formatted_start_time,formatted_end_time,formatted_warmup_time = self.get_formatted_time(
            self.start_time,self.end_time,self.warm_up
        )
        self.request_obj.request_to_simulator_api(formatted_start_time,formatted_end_time,formatted_warmup_time,True)

if __name__ == "__main__":

    # format for time input : year , month , day, hours , minute , second
    start_time_str = "2024,01,27,00,00,00"
    end_time_str = "2024,01,29,00,00,00"
    warm_up = 12

    custom_request_simulation = CustomRequestSimulation(start_time_str, end_time_str,warm_up)