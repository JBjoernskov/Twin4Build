
from twin4build.logger.Logging import Logging
import datetime
from dateutil.parser import parse
import jsonschema
from jsonschema import validate
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

        #startTime =  '2023-12-11 16:23:45+01:00'
        #endTime = '2023-12-12 16:23:45+01:00'
        #Observed[0] = '2023-11-23 06:00:00+00:00' 
        #Observed[-1] = '2023-12-07 03:00:00+00:00'

        if endTime<startTime:
            print("this should be true endTime < startTime" )
            return True
        elif endTime<observed[0]:
            print("this should be true endTime<observed[0]:" )
            return True
        elif startTime>observed[-1]:
            print("this should be true startTime>observed[-1]:" )
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
                            logger.error(f"The provided start_time \"{input_data['metadata']['start_time']}\" and end_time \"{input_data['metadata']['end_time']}\" are not valid for the provided weather data.")
                            return False
                        
                else:
                     # ml_forecast_input_dmi is named as ml_inputs_dmi for the forecast 
                    if 'ml_forecast_inputs_dmi' not in input_data['inputs_sensor'].keys() :
                        return False
                    
                    f_i = input_data['inputs_sensor']['ml_forecast_inputs_dmi']
                    
                    if ('observed' not in f_i.keys()):
                        return False
                    else:
                        if self.is_outside_bounds(input_data, f_i["observed"]):
                            logger.error(f"The provided start_time \"{input_data['metadata']['start_time']}\" and end_time \"{input_data['metadata']['end_time']}\" are not valid for the provided weather data.")
                            return False

                # getting the dmi inputs from the ml_inputs dict
                    
                ml_i = input_data['inputs_sensor']['ml_inputs']
                # checking for the start time in metadata and observed values in the dmi inputs 
                if(input_data["metadata"]['start_time'] == '') or ('damper' not in ml_i.keys()):
                    logger.error("Invalid input data got")
                    return False
                else:
                    return True
                
            except Exception as input_data_valid_error:
                logger.error("An error has occured while validating input data ")
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
            print(response_data_valid_error)
            logger.error("An error has occured while validating response data")
            return False
        
    def validate_ventilation_input(self,json_data):
        schema = {
            "type": "object",
            "properties": {
                "metadata": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"},
                        "start_time": {"type": "string", "format": "date-time"},
                        "end_time": {"type": "string", "format": "date-time"},
                        "stepSize": {"type": "integer"}
                    },
                    "required": ["location", "start_time", "end_time", "stepSize"]
                },
                "rooms_sensor_data": {
                    "type": "object",
                    "patternProperties": {
                        "^.*$": {
                            "type": "object",
                            "properties": {
                                "time": {"type": "array", "items": {"type": "string", "format": "date-time"}},
                                "co2": {"type": "array", "items": {"type": "number"}},
                                "damper_position": {"type": "array", "items": {"type": "number"}}
                            },
                            "required": ["time", "co2", "damper_position"]
                        }
                    }
                }
            },
            "required": ["metadata", "rooms_sensor_data"]
        }

        try:
            validate(instance=json_data, schema=schema)
            print("Validation successful.")
            return True
        except jsonschema.exceptions.ValidationError as e:
            print(f"Validation failed: {e.message}")
            return False
        

    def validate_ventilation_response(self,response):
        #check if response data is None 
        if(response is None or response == {}):
            return False, "received None "
            
        # Check if the required keys are present
        required_keys = ['common_data', 'rooms']
        if not all(key in response for key in required_keys):
            return False, 'Missing required keys in the response'

        # Validate common_data
        common_data = response['common_data']
        if not all(key in common_data for key in ['Sum_of_damper_air_flow_rates', 'Simulation_time']):
            return False, 'Missing required keys in common_data'

        # Validate rooms
        rooms = response['rooms']
        for room_name, room_data in rooms.items():
            if not all(key in room_data for key in ['Supply_damper_{}_airFlowRate'.format(room_name),
                                                    'Supply_damper_{}_damperPosition'.format(room_name)]):
                return False, f'Missing required keys in room {room_name} data'

            # Validate data types and lengths
            if not (isinstance(room_data['Supply_damper_{}_airFlowRate'.format(room_name)], list) and
                    isinstance(room_data['Supply_damper_{}_damperPosition'.format(room_name)], list)):
                return False, f'Invalid data types in room {room_name} data'

            if len(room_data['Supply_damper_{}_airFlowRate'.format(room_name)]) != len(common_data['Simulation_time']) or \
            len(room_data['Supply_damper_{}_damperPosition'.format(room_name)]) != len(common_data['Simulation_time']):
                return False, f'Length mismatch in room {room_name} data'

        print("Ventilation Output Validation sucessfull")
        return True, 'Response is valid'


