
import pytz , os , sys , json
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

class WhatIfSimulation:
    def __init__(self) -> None:
        self.request_obj = request_class()

        self.denmark_timezone = pytz.timezone('Europe/Copenhagen')
        self.time_format = '%Y-%m-%d %H:%M:%S%z'

        self.read_json()

    def read_json(self):

        # Define the path for the config.json file
        config_json_path = os.path.join(os.path.abspath(
            uppath(os.path.abspath(__file__), 4)), "config", "whatif_config.json")

        # Read JSON data from the config file
        with open(config_json_path, 'r') as json_file:
            json_data = json.loads(json_file.read())

        self.isForecastSimulation = json_data['isForecastSimulation']

        if self.isForecastSimulation:
            self.warm_up = json_data['forecast']['warm_up']
            self.start_time = json_data['forecast']['start_time']
            self.end_time = json_data['forecast']['end_time']

        else:
            self.warm_up = json_data['history']['warm_up']
            self.start_time = json_data['history']['start_time']
            self.end_time = json_data['history']['end_time']

    def get_formatted_time(self):
        # Create datetime object from raw time string
        start_time = datetime.strptime(self.start_time, '%Y,%m,%d,%H,%M,%S')
        start_time = start_time.replace(tzinfo=timezone(timedelta(hours=1)))

        warm_uptime = start_time - timedelta(hours=self.warm_up)
        formatted_warmup_time = warm_uptime.strftime(self.time_format)

        formatted_start_time = start_time.strftime(self.time_format)
 
        end_time = datetime.strptime(self.end_time, '%Y,%m,%d,%H,%M,%S')
        end_time = end_time.replace(tzinfo=timezone(timedelta(hours=1)))
        formatted_end_time = end_time.strftime(self.time_format)

        return formatted_start_time,formatted_end_time,formatted_warmup_time
    
    def request_simulation(self):
        formatted_start_time,formatted_end_time,formatted_warmup_time = self.get_formatted_time()
        self.request_obj.request_to_simulator_api(formatted_start_time,formatted_end_time,formatted_warmup_time,self.isForecastSimulation)

if __name__ == "__main__":
    whatif_simulation = WhatIfSimulation()

    whatif_simulation.request_simulation()




