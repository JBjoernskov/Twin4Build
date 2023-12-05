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

# Initialize the logger
logger = Logging.get_logger('API_logfile')


class RequestTimer:
    def __init__(self,request_class_obj) -> None:
        # Get the current time in the Denmark time zone
        self.denmark_timezone = pytz.timezone('Europe/Copenhagen')
        self.current_time_denmark = datetime.now(self.denmark_timezone)
       
        self.simulation_count = 1

        self.request_obj  = request_class_obj
        self.config = self.request_obj.get_configuration()

        self.simulation_duration = int(self.config["simulation_variables"]["simulation_duration"])
        self.forecast_simulation_duration = int(self.config["forecast_simulation_variables"]["forecast_simulation_duration"])
        self.warmup_time = int(self.config["simulation_variables"]["warmup_time"])

        self.forecast_simulation_duration = 12

        
    def get_history_date(self):
        end_time = self.current_time_denmark - timedelta(hours=3)
        start_time = end_time - timedelta(hours=self.simulation_duration)

        total_start_time = start_time -  timedelta(hours=self.warmup_time)
        formatted_total_start_time= total_start_time.strftime('%Y-%m-%d %H:%M:%S')
        
        formatted_endtime= end_time.strftime('%Y-%m-%d %H:%M:%S')
        formatted_startime= start_time.strftime('%Y-%m-%d %H:%M:%S')

        return formatted_startime,formatted_endtime,formatted_total_start_time
        
    def get_forecast_date(self):
        # end time - 3 = start without warmup 
        start_time = self.current_time_denmark - timedelta(hours=3)
        # start time now + 24 hours = 3 , 30
        end_time = start_time +  timedelta(hours=self.forecast_simulation_duration) # simulation_duration = 24 
        # start time - 12 hours = warm up  3 28
        total_start_time = start_time - timedelta(hours=self.warmup_time)

        #  return start , formatt start , end 
        forecast_total_start_time= total_start_time.strftime('%Y-%m-%d %H:%M:%S')
        forecast_endtime= end_time.strftime('%Y-%m-%d %H:%M:%S')
        forecast_startime= start_time.strftime('%Y-%m-%d %H:%M:%S')
        #return formatted_startime = actual start time , 3 peeche ,formatted_endtime- next day ,formatted_total_start_time - watnpm - 3 , 28
        return forecast_startime,forecast_endtime,forecast_total_start_time
    
    def request_for_forcasting_simulations(self):
        # make changes as per forcasting times 
        start_time, end_time,warmup_time = self.get_forecast_date()

        logger.info("[request_to_api:main]:start and end time is")
        self.request_obj.request_to_simulator_api(start_time, end_time,warmup_time,forecast=True)

    def request_for_history_simulations(self):
        start_time, end_time,warmup_time = self.get_history_date()

        logger.info("[request_to_api:main]:start and end time is")
        self.request_obj.request_to_simulator_api(start_time, end_time,warmup_time,forecast=False)

    def request_simulator(self):
        if self.simulation_count == 1:
            self.request_for_history_simulations()
            self.request_for_forcasting_simulations()

            self.simulation_count += 1
            self.simulation_last_time =  datetime.now(self.denmark_timezone)

        else:
            self.forecast_simulation_run_time = 1

            time_now = self.current_time_denmark.now()
            self.time_difference  = time_now - self.simulation_last_time

            self.simulation_count += 1
            self.simulation_last_time =  datetime.now(self.denmark_timezone)
           
            if self.time_difference >= 3 or simualtion_count % 3 == 0:
                self.request_for_forcasting_simulations(self.forecast_simulation_run_time)
                simualtion_count = 0
