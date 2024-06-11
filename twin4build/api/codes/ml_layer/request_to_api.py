import os 
import sys
import time
import pytz
import json
import requests
import pandas as pd
from datetime import datetime,timedelta

###Only for testing before distributing package
if __name__ == '__main__':
    uppath = lambda _path, n: os.sep.join(_path.split(os.sep)[:-n])
    file_path = uppath(os.path.abspath(__file__), 5)
    sys.path.append(file_path)

else: from twin4build.utils.uppath import uppath

# import custom modules
from twin4build.api.codes.ml_layer.input_data import input_data
from twin4build.api.codes.database.db_data_handler import db_connector
from twin4build.config.Config import ConfigReader
from twin4build.logger.Logging import Logging
from twin4build.api.codes.ml_layer.request_timer import RequestTimer
from twin4build.api.codes.ml_layer.validator import Validator
from twin4build.api.codes.ml_layer.output_data import *

#from twin4build.api.codes.ml_layer.simulator_api import SimulatorAPI
# Initialize the logger
logger = Logging.get_logger('API_logfile')

"""
Right now we are connecting 2 times with DB that needs to be corrected.
"""
class request_class:
     
    def __init__(self):
        # Initialize the configuration, database connection, process input data, and disconnect
        logger.info("[request_to_api]: Entered initialise function")
        self.config = self.get_configuration()
        print(self.config)
        
        self.url = self.config['simulation_api_cred']['url']
        self.history_table_to_add_data = self.config['simulation_variables']['table_to_add_data']
        self.forecast_table_to_add_data =  self.config['forecast_simulation_variables']['table_to_add_data']
        self.ventilation_table_to_add_data =  'ml_ventilation_simulation_results'
        

        self.db_handler = db_connector()
        self.db_handler.connect()
        self.time_format = '%Y-%m-%d %H:%M:%S%z'

        #creating object of input data class
        self.data_obj = input_data()

        self.validator = Validator()
        logger.info("[request_to_api]: Exited initialise function")

    def get_configuration(self):       
        '''
            Function to connect to the config file
        '''
        try:
            self.conf = ConfigReader()
            config_path = os.path.join(os.path.abspath(
                uppath(os.path.abspath(__file__), 4)), "config", "conf.ini")
            self.config = self.conf.read_config_section(config_path)
            logger.info("[request_class]: Configuration has been read from file")

            #url of web service will be placed here
            #url = self.config["simulation_api_cred"]["url"]
            return self.config
        except Exception as e:
            logger.error("Error reading configuration: %s", str(e))
            print(e)

    # this function creates json file of the object passed - used in testing
    def create_json_file(self,object,filepath):
        try:
            json_data = json.dumps(object)

            # storing the json object in json file at specified path
            with open(filepath,"w") as file:
                file.write(json_data)

        except Exception as file_error:
            logger.error("An error has occured : %s",str(file_error))


    def extract_actual_simulation(self,model_output_data,start_time,end_time):
        "We are discarding warmuptime here and only considering actual simulation time "

        #self.create_json_file(model_output_data,"response_before_transformation.json")

        # print("start time:",start_time,'\n') # 2023-12-12 02:48:38+0100
        model_output_data_df = pd.DataFrame(model_output_data)
        model_output_data_df['time'] = model_output_data_df['time'].apply(lambda x: datetime.strptime(x, '%Y-%m-%dT%H:%M:%S%z').strftime(self.time_format))
        
        #print(model_output_data_df['time'][0]) # 2023-12-11 15:01:27+0100
        model_output_data_df_filtered = model_output_data_df[(model_output_data_df['time'] >= start_time) & (model_output_data_df['time'] < end_time)]
        filtered_simulation_dict = model_output_data_df_filtered.to_dict(orient="list")
        logger.info("[request_to_api]: Extracted Actual Simulation from the response")
        
        return filtered_simulation_dict
    
    def create_dmi_forecast_key(self,i_data):
        try:
            i_data['inputs_sensor']['ml_inputs_dmi'] = i_data['inputs_sensor']['ml_forecast_inputs_dmi']
            i_data['inputs_sensor'].pop('ml_forecast_inputs_dmi',None)
            return i_data
        except Exception as error_creating:
            logger.error("[request_to_api] : create_dmi_forecast_key error %s",str(error_creating))
    
    def request_to_simulator_api(self,start_time,end_time,time_with_warmup,forecast):
        #try :
            # get data from multiple sources code wiil be called here
            logger.info("[request_class]:Getting input data from input_data class")

            i_data = self.data_obj.input_data_for_simulation(time_with_warmup,end_time,forecast)

            # validating the inputs coming ..
            input_validater = self.validator.validate_input_data(i_data,forecast)

            if forecast:
                i_data = self.create_dmi_forecast_key(i_data)
            
            self.create_json_file(i_data,"input_data_space.json")
      
            if input_validater:
                #we will send a request to API and store its response here
                response = requests.post(self.url,json=i_data)
                # Check if the request was successful (HTTP status code 200)
                if response.status_code == 200:
                    model_output_data = response.json()
                    
                    self.create_json_file(model_output_data,"raw_model_output.json")

                    response_validater = self.validator.validate_response_data(model_output_data)
                    #validating the response
                    if response_validater:
                        #filtering out the data between the start and end time ...
                        model_output_data = self.extract_actual_simulation(model_output_data,start_time,end_time)

                        formatted_response_list_data = convert_response_to_list(model_output_data)

                        # storing the list of all the rows needed to be saved in database
                        input_list_data = space_output_formating(formatted_response_list_data,start_time,end_time)
                                    
                        self.create_json_file(input_list_data,"response_after_transformation.json")

                        if not forecast: # forecast & history table names will get switched 
                            table_to_add_data = self.history_table_to_add_data # ml_simulation_results
                        else:
                            table_to_add_data = self.forecast_table_to_add_data

                        # self.db_handler.add_data(table_to_add_data,inputs=input_list_data)

                        logger.info("[request_class]: data from the reponse is added to the database in table")  
                    else:
                        print("Response data is not correct please look into that")
                        logger.info("[request_class]:Response data is not correct please look into that ")
                else:
                    print("get a reponse from api other than 200 response is: %s"%str(response.status_code))
                    logger.info("[request_class]:get a reponse from api other than 200")
            else:
                print("Input data is not correct please look into that")
                logger.info("[request_class]:Input data is not correct please look into that ")

        # except Exception as e :
        #     print("Error: %s" %e)
        #     logger.error("An Exception occured while requesting to simulation API: %s",str(e))

            try:
                self.db_handler.disconnect()
                self.data_obj.db_disconnect()
            except Exception as disconnect_error:
                logger.info("[request_to_simulator_api]:disconnect error %s",str(disconnect_error))
                
    def request_to_ventilation_api(self,start_time,end_time):
        try :
            # get data from multiple sources code wiil be called here
            logger.info("[ventilation request_class]:Getting ventilation input data from input_data class")

            #fetch input data
            input_data = self.data_obj.input_data_for_ventilation(start_time,end_time)

            # validating the inputs data
            input_validater = self.validator.validate_ventilation_input(input_data)

            self.create_json_file(input_data,"ventilation_input_data.json")

            url = "http://127.0.0.1:8070/simulate_ventilation"

            if input_validater:
                #we will send a request to API and store its response here
                response = requests.post(url,json=input_data)

                #print(response.status_code)
                # Check if the request was successful (HTTP status code 200)
                if response.status_code == 200:
                    model_output_data = response.json()

                    #self.create_json_file(model_output_data,"raw_ventilation_model_output.json")

                    #validating the response
                    response_validater = self.validator.validate_ventilation_response(model_output_data)
                    
                    if response_validater:

                    #     #filtering out the data between the start and end time ...
                    #     model_output_data = self.extract_actual_simulation(model_output_data,start_time,end_time)

                        # storing the list of all the rows needed to be saved in database
                        db_table_data = ventilation_output_formating(model_output_data)

                        #self.create_json_file(input_list_data,"response_after_transformation.json")

                        #self.db_handler.add_large_data(self.ventilation_table_to_add_data,db_table_data)

                        logger.info("[request_class]: Ventilation data from the reponse is added to the database in table")  
                    else:
                        print("Response data is not correct please look into that")
                        logger.info("[request_class]:Response data is not correct please look into that ")
                else:
                    print("get a reponse from api other than 200 response is: %s"%str(response.status_code))
                    logger.info("[request_class]:get a reponse from api other than 200")
            else:
                print("Input data is not correct please look into that")
                logger.info("[request_class]:Input data is not correct please look into that ")

        except Exception as e :
            print("Error: %s" %e)
            logger.error("An Exception occured while requesting to simulation API: %s",str(e))

            try:
                self.db_handler.disconnect()
                self.data_obj.db_disconnect()
            except Exception as disconnect_error:
                logger.info("[request_to_simulator_api]:disconnect error %s",str(disconnect_error))

if __name__ == '__main__':

    request_class_obj= request_class()
    config = request_class_obj.get_configuration()
    
    request_timer_obj = RequestTimer(request_class_obj)

    simulation_duration = int(config["simulation_variables"]["simulation_duration"])
        
    #sleep_interval = simulation_duration * 60 * 60  # 1 hours in seconds

    sleep_interval = 120

    simulation_count  = 0


    while True:
        try :
            request_timer_obj.request_for_history_simulations()
            if simulation_count%3==0:
                #we are running forecasting every 3 hours
                request_timer_obj.request_for_forcasting_simulations()
            
            time.sleep(2)
            request_timer_obj.request_for_ventilation_simulation()
                
            #counter that adds up with 1 every hour
            simulation_count += 1
            print("Function called at:", time.strftime("%Y-%m-%d %H:%M:%S"))
            logger.info("[main]:Function called at:: %s"%time.strftime("%Y-%m-%d %H:%M:%S"))
            time.sleep(sleep_interval)
        except Exception as schedule_error:
            print("Error occured in scheduling and error is:",schedule_error)
            request_class_obj.db_handler.disconnect()
            request_class_obj.data_obj.db_disconnect()
            logger.error("An Error has occured: %s",str(schedule_error))
            time.sleep(sleep_interval)

     
