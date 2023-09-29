import os 
import sys
import time
import pytz
import schedule
import json
import requests
import pandas as pd
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

from twin4build.api.codes.ml_layer.validator import Validator

#from twin4build.api.codes.ml_layer.simulator_api import SimulatorAPI

# Initialize the logger
logger = Logging.get_logger('API_logfile')

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

        self.validator = Validator()


    def get_configuration(self):
            # Read configuration using ConfigReader
            try:
                self.conf = ConfigReader()
                config_path = os.path.join(os.path.abspath(
                uppath(os.path.abspath(__file__), 4)), "config", "conf.ini")
                self.config = self.conf.read_config_section(config_path)
                logger.info("[request_class]: Configuration has been read from file")

                self.table_to_add_data = self.config['simulation_variables']['table_to_add_data']
                return self.config
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
            return None
        

    def extract_actual_simulation(self,model_output_data,start_time,end_time):
        "We are discarding warmuptime here and only considering actual simulation time "

        model_output_data_df = pd.DataFrame(model_output_data)

        model_output_data_df['time'] = model_output_data_df['time'].apply(lambda x: datetime.strptime(x, '%Y-%m-%dT%H:%M:%S').strftime('%Y-%m-%d %H:%M:%S'))

        model_output_data_df_filtered = model_output_data_df[(model_output_data_df['time'] >= start_time) & (model_output_data_df['time'] < end_time)]

        filtered_simulation_dict = model_output_data_df_filtered.to_dict(orient="list")

        return filtered_simulation_dict
    

    def request_to_simulator_api(self,start_time,end_time,time_with_warmup):
        try :
            #url of web service will be placed here
            url = self.config["simulation_api_cred"]["url"]

            # get data from multiple sources code wiil be called here
            logger.info("[request_class]:Getting input data from input_data class")

            i_data = self.data_obj.input_data_for_simulation(time_with_warmup,end_time)

            self.create_json_file(i_data,"inputs_test_data.json")
            # validating the inputs coming ..
            input_validater = self.validator.validate_input_data(i_data)

            # creating test input json file it's temporary

            if input_validater:
                #we will send a request to API and store its response here
                response = requests.post(url,json=i_data)

                # Check if the request was successful (HTTP status code 200)
                if response.status_code == 200:
                    model_output_data = response.json()

                    response_validater = self.validator.validate_response_data(model_output_data)

                    #validating the response
                    if response_validater:
                        #filtering out the data between the start and end time ...
                        model_output_data = self.extract_actual_simulation(model_output_data,start_time,end_time)

                        formatted_response_list_data = self.convert_response_to_list(response_dict=model_output_data)

                        # storing the list of all the rows needed to be saved in database
                        input_list_data = self.data_obj.transform_list(formatted_response_list_data)

                        with open('input_list_data.json','w') as f:
                            f.write(json.dumps(input_list_data))

                        self.db_handler.add_data(self.table_to_add_data,inputs=input_list_data)

                        logger.info("[request_class]: data from the reponse is added to the database in table")
                    else:
                        print("Response data is not correct please look into that")
                        logger.info("[request_class]:Response data is not correct please look into that ")         
                else:
                    print("get a reponse from api other than 200 response is: %s"%str(response.status_code))
                    logger.info("[request_class]:get a reponse from api other than 200 response is: %s"%str(response.status_code))
            else:
                print("Input data is not correct please look into that")
                logger.info("[request_class]:Input data is not correct please look into that ")

        except Exception as e :
            print("Error: %s" %e)
            logger.error("An Exception occured while requesting to simulation API:",e)

            try:
                self.db_handler.disconnect()
                self.data_obj.db_disconnect()
            except Exception as disconnect_error:
                logger.info("[request_to_simulator_api]:disconnect error Error is : %s"%(disconnect_error))
            

if __name__ == '__main__':
        
    request_obj = request_class()
    temp_config = request_obj.get_configuration()

    try:
        simulation_duration = int(temp_config["simulation_variables"]["simulation_duration"])
        warmup_time = int(temp_config["simulation_variables"]["warmup_time"])
        #warmup_flag = int(temp_config["simulation_variables"]["warmup_flag"])

        def getDateTime(simulation_duration):
            # Define the Denmark time zone
            denmark_timezone = pytz.timezone('Europe/Copenhagen')

            # Get the current time in the Denmark time zone
            current_time_denmark = datetime.now(denmark_timezone)
            
            end_time = current_time_denmark -  timedelta(hours=3)
            start_time = end_time -  timedelta(hours=simulation_duration)

            total_start_time = start_time-  timedelta(hours=warmup_time)
            formatted_total_start_time= total_start_time.strftime('%Y-%m-%d %H:%M:%S')
            
            formatted_endtime= end_time.strftime('%Y-%m-%d %H:%M:%S')
            formatted_startime= start_time.strftime('%Y-%m-%d %H:%M:%S')

            return formatted_startime,formatted_endtime,formatted_total_start_time
            
            #return formatted_startime,formatted_endtime,formatted_total_start_time

        def request_simulator():
            start_time, end_time,warmup_time = getDateTime(simulation_duration)
            logger.info("[request_to_api:main]:start and end time is")
            request_obj.request_to_simulator_api(start_time, end_time,warmup_time)
            
        # Schedule subsequent function calls at 2-hour intervals
        sleep_interval = simulation_duration * 60 * 60  # 2 hours in seconds

        request_simulator()
        # Create a schedule job that runs the request_simulator function every 2 hours
        schedule.every(sleep_interval).seconds.do(request_simulator)

        while True:
            try :
                schedule.run_pending()
                print("Function called at:", time.strftime("%Y-%m-%d %H:%M:%S"))
                logger.info("[main]:Function called at:: %s"%time.strftime("%Y-%m-%d %H:%M:%S"))
                time.sleep(sleep_interval)      
           
            except Exception as schedule_error:
                schedule.cancel_job()
                request_obj.db_handler.disconnect()
                request_obj.data_obj.db_disconnect()
                logger.error("An Error has occured:",schedule_error)
                break

    except Exception as e:
        logger.error("Exception occured while reading config data",e)
        print("Exception occured while reading config data",e)
