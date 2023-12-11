
from twin4build.logger.Logging import Logging
import datetime
from dateutil.parser import parse
#from twin4build.api.codes.ml_layer.simulator_api import SimulatorAPI

# Initialize the logger
logger = Logging.get_logger('API_logfile')

class Validator:
    def __init__(self):
        self.time_format = '%Y-%m-%d %H:%M:%S%z'

    def is_outside_bounds(self, input_data, data_time_stamps):
        startTime = datetime.datetime.strptime(input_data["metadata"]["start_time"], self.time_format)
        endTime = datetime.datetime.strptime(input_data["metadata"]["end_time"], self.time_format)
        observed = sorted([parse(date_str) for date_str in data_time_stamps])

        if endTime<startTime:
            return True
        elif endTime<observed[0]:
            return True
        elif startTime>observed[-1]:
            return True
        
        return False
    
    def validate_input_data(self,input_data,forecast):
            '''
                This function validates the input data and return true or false as response
            '''
            try:
                if len(input_data['inputs_sensor'])  < 1 :
                    return False
                
                # validation for the forecast input data

                if 'ml_inputs' not in input_data['inputs_sensor'].keys():
                    return False

                if not forecast:        
                    if 'ml_inputs_dmi' not in input_data['inputs_sensor'].keys() :
                        return False
                    
                    dmi = input_data['inputs_sensor']['ml_inputs_dmi']

                    if ('observed' not in dmi.keys()):
                        return False
                    else:
                        if self.is_outside_bounds(input_data, dmi["observed"]):
                            logger.error(f"The provided start_time \"{input_data['metadata']['start_time']}\" and end_time \"{input_data['metadata']['end_time']}\" are valid for the provided weather data.")
                            return False
                        

                else:
                    if 'ml_forecast_inputs_dmi' not in input_data['inputs_sensor'].keys() :
                        return False
                    
                    f_i = input_data['inputs_sensor']['ml_forecast_inputs_dmi']
                    
                    if ('observed' not in f_i.keys()):
                        return False
                    else:
                        if self.is_outside_bounds(input_data, f_i["observed"]):
                            logger.error(f"The provided start_time \"{input_data['metadata']['start_time']}\" and end_time \"{input_data['metadata']['end_time']}\" are valid for the provided weather data.")
                            return False

                # getting the dmi inputs from the ml_inputs dict
                    
                ml_i = input_data['inputs_sensor']['ml_inputs']
                # checking for the start time in metadata and observed values in the dmi inputs 
                if(input_data["metadata"]['start_time'] == '') or ('damper' not in ml_i.keys()) :
                    logger.error("Invalid input data got")
                    return False
                else:
                    return True
                
            except Exception as input_data_valid_error:
                logger.error('An error has occured while validating input data ',input_data_valid_error)
                return False
    
    def validate_response_data(self,reponse_data):
        try :
            #check if response data is None 
            if(reponse_data is None or reponse_data == {}):
                logger.error("No data came from the database , no table got ")
                return False
            
            # searching for the time key in the reponse data else returning false
            if("time" not in reponse_data.keys()):
                logger.error("Invalid response data ")
                return False
            
            #checking the time list is null then returning false
            elif (len(reponse_data['time']) < 1):
                logger.error("Invalid response data ")
                return False
            else:
                return True
            
        except Exception as response_data_valid_error:
            logger.error('An error has occured while validating response data ',response_data_valid_error)
            return False
