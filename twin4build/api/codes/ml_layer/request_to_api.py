import os 
import sys
import time
import sched
import requests



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

import pandas as pd

# Initialize the logger
logger = Logging.get_logger('ai_logfile')


def temp_function(metadata):
     #step 1:"Read csv file from filename as a dataframe"
     #spep 2: convert to dict
     #treat this dict as response from API

    csv_path = os.path.join(os.path.abspath(
                  uppath(os.path.abspath(__file__), 1)), "output.csv")
    
    dataframe = pd.DataFrame(pd.read_csv(csv_path))

    columns_to_replace = {
        'time' : 'simulation_time',
        'Outdoor environment ||| outdoorTemperature' : 'outdoorenvironment_outdoortemperature',
        'Outdoor environment ||| globalIrradiation'  : 'outdoorenvironment_globalirradiation',
        'OE20-601b-2 ||| indoorTemperature' : 'indoortemperature',
        'OE20-601b-2 ||| indoorCo2Concentration' : 'indoorco2concentration',
        'Supply damper ||| airFlowRate' : 'supplydamper_airflowrate',
        'Supply damper ||| damperPosition' : 'supplydamper_damperposition',
        'Exhaust damper ||| airFlowRate' : 'exhaustdamper_airflowrate',
        'Exhaust damper ||| damperPosition' : 'exhaustdamper_damperposition',
        'Space heater ||| outletWaterTemperature' : 'spaceheater_outletwatertemperature',
        'Space heater ||| Power' : 'spaceheater_power',
        'Space heater ||| Energy' : 'spaceheater_energy',
        'Valve ||| waterFlowRate' : 'valve_waterflowrate',
        'Valve ||| valvePosition' : 'valve_valveposition',
        'Temperature controller ||| inputSignal' : 'temperaturecontroller_inputsignal',
        'CO2 controller ||| inputSignal' : 'co2controller_inputsignal',
        'OE20-601b-2| temperature sensor ||| indoorTemperature' : 'temperaturesensor_indoortemperature',
        'OE20-601b-2| Valve position sensor ||| valvePosition': 'valvepositionsensor_valveposition',
        'OE20-601b-2| Damper position sensor ||| damperPosition': 'damperpositionsensor_damperposition',
        'OE20-601b-2| CO2 sensor ||| indoorCo2Concentration': 'co2sensor_indoorco2concentration',
        'OE20-601b-2| Heating meter ||| Energy': 'heatingmeter_energy',
        'OE20-601b-2| Occupancy schedule ||| scheduleValue': 'occupancyschedule_schedulevalue',
        'OE20-601b-2| Temperature setpoint schedule ||| scheduleValue': 'temperaturesetpointschedule_schedulevalue',
        'Heating system| Supply water temperature schedule ||| supplyWaterTemperatureSetpoint': 'supplywatertemperatureschedule_supplywatertemperaturesetpoint',
        'Ventilation system| Supply air temperature schedule ||| scheduleValue': 'ventilationsystem_supplyairtemperatureschedule_schedulevaluet'
    }

    dataframe.rename(columns=columns_to_replace,inplace=True)
    
    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
    file_path = os.path.join(desktop_path, "output.csv")

    dataframe.to_csv(file_path,index=False)

    new_df = pd.DataFrame(pd.read_csv(file_path))

    new_df['input_start_datetime'] = metadata['start_time']
    new_df['input_end_datetime'] = metadata['end_time']
    new_df['spacename'] = metadata['roomname']

    list_of_dicts = new_df.to_dict(orient='records')

    return list_of_dicts
    


class request_class:
     
    def __init__(self):
        # Initialize the configuration, database connection, process input data, and disconnect
        try:
                self.get_configuration()
                #self.db_connect()
        except Exception as e:
                logger.error("An error occurred during data conversion: %s", str(e))

    def get_configuration(self):
            # Read configuration using ConfigReader
            try:
                self.conf = ConfigReader()
                config_path = os.path.join(os.path.abspath(
                uppath(os.path.abspath(__file__), 4)), "config", "conf.ini")
                self.config = self.conf.read_config_section(config_path)
                logger.info("[DBConnector: Configuration has been read from file]")
            except Exception as e:
                logger.error("Error reading configuration: %s", str(e))


    def request_to_simulator_api(self):

        #url of web service will be placed here
        url = self.config["simulation_api_cred"]["url"]

        # get data from multiple sources code wiil be called here
        logger.info("Getting input data from input_data class")
        data_obj = input_data()
        _data = data_obj.input_data_for_simulation()

        #we will send a request to API and srote its response here

        try : 

            '''
            response = requests.post(url,_data)

            # assuming response.data is of ml_simulation_results table format 
            
            inputs = {
                "creation_start_date": self.config['data_fetching_config']['start_time'],
                "creation_end_date": self.config['data_fetching_config']['end_time']
            }
            
            response_data = data_obj.output_data(response.data,inputs)

            # Check if the request was successful (HTTP status code 200)
            if response.status_code == 200:
                db = db_connector()
                
                db.connect()
                db.add_data(table_name="ml_simulation_results",inputs=response_data)
                db.disconnect()

                logger.info("data from the reponse is added to the database in ml_simulation_results table")
            else:
                logger.error("")
            '''

            data  = temp_function(_data['metadata'])

            db = db_connector()
            db.connect()
            for input in data:
                db.add_data(table_name="ml_simulation_results",inputs=input)
            db.disconnect()


        except Exception as e :
            logger.error("An Exception occured while requesting to simulation API:",e)


if __name__ == '__main__':
    # Initialize the scheduler
    scheduler = sched.scheduler(time.time, time.sleep)

    # function to be called
    reuest_obj = request_class()
    
    # Schedule the first function call
    scheduler.enter(0, 1, reuest_obj.request_to_simulator_api(), ())
    #logger.info("Function called at:", time.strftime("%Y-%m-%d %H:%M:%S"))
    print("Function called at:", time.strftime("%Y-%m-%d %H:%M:%S"))

    # Schedule subsequent function calls at 2-hour intervals
    interval = 2 * 60 * 60  # 2 hours in seconds

    count = 0
    while True:
        scheduler.run()
        time.sleep(interval)
        if count >= 3:
            break
        count  = count+1

    # This loop will keep the scheduler running indefinitely, calling the function every 2 hours.

    """Keep in mind that if you want to stop the scheduler at some point, 
    you'll need to add a condition to exit the loop. This is a basic example, 
    and in a real-world application, you might want to handle exceptions, manage the scheduler's state, 
    and possibly use a more robust scheduling library depending on your requirements."""
    





