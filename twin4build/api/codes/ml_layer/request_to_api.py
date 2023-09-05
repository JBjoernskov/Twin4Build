import os 
import sys
import time
import schedule
import json
import requests
from datetime import datetime , timedelta

###Only for testing before distributing package
if __name__ == '__main__':
    # Define a function to move up in the directory hierarchy
    uppath = lambda _path, n: os.sep.join(_path.split(os.sep)[:-n])
    # Calculate the file path using the uppath function
    file_path = uppath(os.path.abspath(__file__), 5)
    # Append the calculated file path to the system path
    sys.path.append(file_path)

from twin4build.api.codes.ml_layer.input_data import input_data
from twin4build.api.codes.database.db_data_handler import db_connector
from twin4build.config.Config import ConfigReader
from twin4build.logger.Logging import Logging

#from twin4build.api.codes.ml_layer.simulator_api import SimulatorAPI

# Initialize the logger
logger = Logging.get_logger('ai_logfile')

"""
Right now we are connecting 2 times with DB that needs to be corrected.
"""
class request_class:
     
    def __init__(self):
        # Initialize the configuration, database connection, process input data, and disconnect
        self.get_configuration()
        self.db_handler = db_connector()
        self.db_handler.connect()

        #creating object of input data class
        self.data_obj = input_data()

    def get_configuration(self):
            # Read configuration using ConfigReader
            try:
                self.conf = ConfigReader()
                config_path = os.path.join(os.path.abspath(
                uppath(os.path.abspath(__file__), 4)), "config", "conf.ini")
                self.config = self.conf.read_config_section(config_path)
                logger.info("[request_class]: Configuration has been read from file")
            except Exception as e:
                logger.error("Error reading configuration: %s", str(e))

    def create_json_file(self,object,filepath):
        try:
            json_data = json.dumps(object)

            # storing the json object in json file at specified path
            with open(filepath,"w") as file:
                file.write(json_data)

        except Exception as file_error:
            logger.error("An error has occured : ",file_error)

    def convert_response_to_list(self,response_dict):
    # Extract the keys from the response dictionary
        keys = response_dict.keys()

        # Initialize an empty list to store the result
        result = []

        try:

            # Iterate over the data and create dictionaries
            for i in range(len(response_dict["time"])):
                data_dict = {}
                for key in keys:
                    data_dict[key] = response_dict[key][i]
                result.append(data_dict)

            #temp file finally we will comment it out
            self.create_json_file(result,"response_converted_test_data.json")

            return result
        
        except Exception as converion_error:
            logger.error('An error has occured',converion_error)
    

    def request_to_simulator_api(self,start_time,end_time):
        
        try :
            #url of web service will be placed here
            url = self.config["simulation_api_cred"]["url"]

            # get data from multiple sources code wiil be called here
            logger.info("[request_class]:Getting input data from input_data class")
            i_data = self.data_obj.input_data_for_simulation(start_time,end_time)

            # creating test input json file
            self.create_json_file(i_data,"inputs_test_data.json")

            #simulator_obj = SimulatorAPI()
            #results=simulator_obj.run_simulation(_data)

            #we will send a request to API and srote its response here
            response = requests.post(url,json=i_data)

            model_output_data = response.json()

            #this is a temp file we will comment this out 
            self.create_json_file(model_output_data,"response_test_data.json")

            formatted_response_list_data = self.convert_response_to_list(response_dict=model_output_data)

            res = []

            for i in formatted_response_list_data:
                res.append(self.data_obj.transform_dict(i))
            self.db_handler.add_data("ml_simulation_results",inputs=res)

            '''

            # Check if the request was successful (HTTP status code 200)
            if response.status_code == 200:
                for data in formatted_response_list_data:
                    data = self.data_obj.transform_dict(data)

                    data['input_start_datetime'] = start_time
                    data['input_end_datetime'] = end_time
                    data['spacename'] = i_data['metadata']['roomname']

                    self.db_handler.add_data(table_name="ml_simulation_results",inputs=data)

                    #finally we are going to commnet this code
                    

                    logger.info("[request_class]: data from the reponse is added to the database in table")
            else:
                print("get a reponse from api other than 200 response is: %s"%str(response.status_code))
                logger.info("[request_class]:get a reponse from api other than 200 response is: %s"%str(response.status_code))
            '''
        except Exception as e :
            print("Error: %s" %e)
            logger.error("An Exception occured while requesting to simulation API:",e)
            try:
                self.db_handler.disconnect()
                self.data_obj.db_disconnect()
            except Exception as disconnect_error:
                logger.info("[request_to_simulator_api]:disconnect error Error is : %s"%(disconnect_error))


def getDateTime():
    
    current_time = datetime.now()
    end_time = current_time -  timedelta(hours=2)
    start_time = end_time -  timedelta(hours=2)
    
    formatted_endtime= end_time.strftime('%Y-%m-%d %H:%M:%S')
    formatted_startime= start_time.strftime('%Y-%m-%d %H:%M:%S')

    return formatted_startime,formatted_endtime
            

if __name__ == '__main__':
    
    request_obj = request_class()

    def request_simulator():
        start_time, end_time = getDateTime()
        request_obj.request_to_simulator_api(start_time, end_time)

    # Schedule subsequent function calls at 2-hour intervals
    interval = 2 * 60 * 60  # 2 hours in seconds

    # Create a schedule job that runs the request_simulator function every 2 hours
    schedule.every(interval).seconds.do(request_simulator)

    print("Function called at:", time.strftime("%Y-%m-%d %H:%M:%S"))

    count = 0
    while True:
        try :
            schedule.run_pending()
            time.sleep(interval)
        except Exception as schedule_error:
            schedule.cancel_job()
            request_obj.db_handler.disconnect()
            request_obj.data_obj.db_disconnect()
            logger.error("An Error has occured:",schedule_error)
            break
        count  = count+1

    # This loop will keep the scheduler running indefinitely, calling the function every 2 hours.

    """Keep in mind that if you want to stop the scheduler at some point, 
    you'll need to add a condition to exit the loop. This is a basic example, 
    and in a real-world application, you might want to handle exceptions, manage the scheduler's state, 
    and possibly use a more robust scheduling library depending on your requirements."""
    



