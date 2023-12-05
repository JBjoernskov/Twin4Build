
from twin4build.logger.Logging import Logging

#from twin4build.api.codes.ml_layer.simulator_api import SimulatorAPI

# Initialize the logger
logger = Logging.get_logger('API_logfile')

class Validator:
    def __init__(self):
        pass

    
    def validate_input_data(self,input_data,forecast):
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
                    if 'ml_forecast_inputs_dmi' not in input_data['inputs_sensor'].keys() :
                        return False
                    
                    f_i = input_data['inputs_sensor']['ml_forecast_inputs_dmi']
                    
                    if ('observed' not in f_i.keys()):
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
                logger.error("No data came from the database , no table got maybe ")
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
