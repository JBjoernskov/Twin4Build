
"""Scripts/Functions to convert Data into Input format  """

# Import necessary modules
import os
import sys
import json
from datetime import datetime,timedelta 

# Only for testing before distributing package
if __name__ == '__main__':
    # Define a function to move up in the directory hierarchy
    def uppath(_path, n): return os.sep.join(_path.split(os.sep)[:-n])
    # Calculate the file path using the uppath function
    file_path = uppath(os.path.abspath(__file__), 5)
    # Append the calculated file path to the system path
    sys.path.append(file_path)

# Import custom modules
from twin4build.api.codes.database.db_data_handler import db_connector
from twin4build.config.Config import ConfigReader
from twin4build.logger.Logging import Logging
from twin4build.utils.uppath import uppath

# Initialize a logger
logger = Logging.get_logger("API_logfile")

# Create a class to handle input data conversion
class input_data:
      def __init__(self):
            # Initialize the configuration, database connection, process input data, and disconnect
            self.get_configuration()
            self.db_connect()
            self.time_format = '%Y-%m-%d %H:%M:%S%z'
            #self.input_data_for_simulation()

      def get_configuration(self):
            '''
            Function to connect to the config file
            '''
            try:
                  self.conf = ConfigReader()
                  config_path = os.path.join(os.path.abspath(
                  uppath(os.path.abspath(__file__), 4)), "config", "conf.ini")
                  self.config = self.conf.read_config_section(config_path)
                  logger.info("[DBConnector: Configuration has been read from file]")
            except Exception as e:
                  logger.error("Error reading configuration: %s", str(e))

      def db_connect(self):
            # Connect to the database using db_connector
            try:
                  self.connector = db_connector()
                  self.connector.connect()
            except Exception as e:
                  logger.error("Error connecting to the database: %s", str(e))

      def db_disconnect(self):
            # Disconnect to the database using db_connector
            try:
                  self.connector.disconnect()
            except Exception as e:
                  logger.error("Error connecting to the database: %s", str(e))

      def data_from_db(self, roomname, table_names, data_fething_method):
            """Retrieve data from the database using specified methods"""
            self.db_data = {}
            _data = []

            try:
                  for table_name in table_names:
                        
                        if data_fething_method == "get_all_inputs":
                              _data = self.connector.get_all_inputs(table_name)
                              
                              #self.db_data[table_name] = _data

                        if data_fething_method == "get_data_using_datetime":

                              _data = self.connector.get_data_using_datetime(
                                    tablename=table_name, roomname=roomname, starttime=self.start_time, endtime=self.end_time)
                              
                              #self.db_data[table_name] = _data
                              logger.info("Retrieved data for table: %s", table_name)
                        
                        if data_fething_method == "get_latest_values":
                              
                              _data = [self.connector.get_latest_values(
                                    table_name, roomname)]
                              #self.db_data[table_name] = _data       
                              logger.info("Retrieved data for table: %s", table_name)
                        
                        if table_name == 'ml_forecast_inputs_dmi':
                              _data = self.connector.get_filtered_forecast_inputs("ml_forecast_inputs_dmi",start_time=self.start_time,end_time=self.end_time)


                        self.db_data[table_name] = _data
                   
            except Exception as e:
                  logger.error("Error fetching data from the database: %s", str(e))
                  self.db_data = {}  # Initialize an empty dictionary in case of error

            return self.db_data

      def get_filter_columns(self, table_name):
            """Get filter columns based on the table name"""

            columns_string = ""

            try:
                  if table_name == "ml_inputs":
                        columns_string = self.config['ml_inputs_column_filters']['columns']
                  elif table_name == "ml_inputs_dmi":
                        columns_string = self.config['ml_inputs_dmi_column_filters']['columns']

                  elif table_name == "ml_forecast_inputs_dmi":
                        columns_string = self.config["ml_forecast_inputs_dmi"]["columns"]

                  else :
                        columns_string = self.config['ml_inputs_dmi_column_filters']['columns']
            
                  # converting config.ini string data to the list of string separted by ','
                  columns = [column.strip() for column in columns_string.split(',')]

                  return columns

            except Exception as e:
                  print('No columns got for data filtering using customed inputs',e)
                  logger.error('No columns got for data filtering using cusomted inputs %s',str(e))

                  if table_name == "ml_inputs":
                        columns = ['opcuats','co2concentration','damper','shadingposition','temperature']
                  elif table_name == "ml_inputs_dmi":
                        columns = ['observed','radia_glob','temp_dry']

                  return columns
            
      def read_config_json(self,config_filename, levels_up=4, config_folder="config"):
            """
            Reads a JSON configuration file located `levels_up` directories above the current file's directory
            in the specified `config_folder`.
            
            Args:
                  config_filename (str): The name of the configuration file.
                  levels_up (int): The number of directories to traverse upwards.
                  config_folder (str): The folder name where the configuration file is located.
                  
            Returns:
                  dict: The JSON data loaded from the configuration file.
            """
            # Construct the path to the configuration file
            config_json_path = os.path.join(
                  os.path.abspath(uppath(os.path.abspath(__file__), levels_up)),
                  config_folder,
                  config_filename
            )
            
            # Read JSON data from the config file
            with open(config_json_path, 'r') as json_file:
                  json_data = json_file.read()
            
            # Parse and return the JSON data
            return json.loads(json_data)

      def input_data_for_simulation(self,start_time,end_time,forecast):

            '''
                  function wich transforms the information from the database 
                  and config files and format as been required
            '''

            # read the configuration from config.json
            input_schedules = self.read_config_json(config_filename ="config.json" )

            self.start_time  = start_time
            self.end_time  = end_time

            # Create a dictionary to store input data
            self.input_data = {}

            metadata = {}
            metadata["location"] = self.config["input_data_metadata"]["location"]
            metadata["building_id"] = self.config["input_data_metadata"]["building_id"]
            metadata["floor_number"] = self.config["input_data_metadata"]["floor_number"]
            metadata["room_id"] = self.config["input_data_metadata"]["room_id"]
            metadata["start_time"] = self.start_time 
            metadata["end_time"] = self.end_time
            metadata['roomname'] = self.config['data_fetching_config']['roomname']
            metadata['stepSize'] = int(self.config['model']['stepSize'])

            # Get sensor data from the database
            room_name = self.config["data_fetching_config"]["roomname"]
            table_names = self.config["data_fetching_config"]["table_names"]
            data_fetching_method = self.config["data_fetching_config"]["function_names"]

            table_names_string = self.config["data_fetching_config"]["table_names"]

            # Read table_names from config.ini file and convert to a list of table_name strings
            table_names = [name.strip() for name in table_names_string.split(',')]

            sensor_data_dict = self.data_from_db(
                  roomname=room_name, table_names=table_names, data_fething_method=data_fetching_method)
            
            #print("__________________",sensor_data_dict.items())
            
            #print("++++++++++++++",room_name,table_names,data_fetching_method)

            input_sensor_data = {}

            # Iterate through the sensor data and filter columns
            column_filter = []
            for table_name, sensor_data_list in sensor_data_dict.items():
                  column_filter = self.get_filter_columns(table_name=table_name)

                  data = {table_name: {}}

                  for data_point in sensor_data_list:
                        for field, value in data_point.__dict__.items():
                              if field in column_filter:
                                    if field not in data[table_name]:
                                          data[table_name][field] = []
                                          if field == 'forecast_time':
                                                data[table_name]['observed'] = []
                                    data[table_name][field].append(str(value))
                                    if field == 'forecast_time':
                                          data[table_name]['observed'].append(str(value))

                  input_sensor_data.update(data)

            input_sensor_data['ml_forecast_inputs_dmi'].pop('forecast_time',None)

            # Preprocess and organize the input data
            self.input_data["metadata"] = metadata
            self.input_data["inputs_sensor"] = input_sensor_data
            self.input_data["input_schedules"] = input_schedules

            if forecast:
                  self.input_data['inputs_sensor'].pop('ml_inputs_dmi',None)
            else:
                  self.input_data['inputs_sensor'].pop('ml_forecast_inputs_dmi',None)

            logger.info("Input data has been successfully processed and saved.")
            
            dummy_ml_inputs = {
                  "damper" : ["49.80392"],
                  "opcuats" : ["2024-01-29 00:00:07+00"],
                  "shadingposition" : ["None"],
                  "co2concentration" :["496.0"],
                  "temperature" : ["21.0"] 
            }

            try:
                  if len(self.input_data['inputs_sensor']['ml_inputs']) < 1:
                        self.input_data['inputs_sensor']['ml_inputs'] = dummy_ml_inputs
            except Exception as input_key_error:
                  print("Error occured during data fetching and error is",input_key_error)
                  pass
                  
            return self.input_data
            
            # except Exception as e:
            #       print('An Exception occured in input_data_for_simulation',e)
            #       logger.error('An Exception occured in input_data_for_simulation %s',str(e))

            #       return None
            
      def input_data_for_ventilation(self,start_time,end_time):
            """
            Retrieve data of multiple rooms from the database based and format that for ventilation models.

            Args:
                  starttime (datetime): Start time of the desired time range.
                  endtime (datetime): End time of the desired time range.

            Returns:
                  list: A list JSON/Dictonary file .

            """
            # Create a dictionary to store input data
            input_data = {}

            #format time if required 
            # self.start_time  = start_time
            # self.end_time  = end_time

            metadata = {}
            metadata["location"] = self.config["input_data_metadata"]["location"]
            metadata["start_time"] = start_time 
            metadata["end_time"] = end_time
            metadata['stepSize'] = int(self.config['model']['stepSize'])

            # Get sensor data from the database
            room_names = self.config["data_fetching_ventilation_system_ve01"]["room_names"]
            table_name = self.config["data_fetching_ventilation_system_ve01"]["table_name"]
            run_dummy_simulation = int(self.config["data_fetching_ventilation_system_ve01"]["run_dummy_simulation"])

            # Read room_names from config.ini file and convert to a list of table_name strings
            room_names = [name.strip() for name in room_names.split(',')]

            column_filter = []
            column_filter = self.get_filter_columns(table_name)
            # we only want time,co2,damper_postion therefor removing last 2 columns
            column_filter = column_filter[:3]

            converted_data = {}

            if run_dummy_simulation:
                  print("Fetching Ventilation dummy Data")
                  table_name = "ml_ventilation_dummy_inputs"
                  # these line of code is for dummy data fetching 
                  queried_data = self.connector.get_multiple_rooms_data_filterby_time(table_name,room_names,start_time,end_time)

                  # Iterate over the queried data
                  for row in queried_data:
                        room_name = row[0] 
                        if room_name == "OE22-601B-00":
                              room_name = "OE22-601B-0"
                        simulation_time= datetime.strftime(row[1],self.time_format)
                        #simulation_time= datetime.strftime(simulation_time,self.time_format) 
                        co2_concentration = row[2]  
                        air_damper_position = row[3]  
                        
                        # Check if the room_name is already present in the dictionary
                        if room_name in converted_data:
                              # If room_name exists, append the data to its lists
                              converted_data[room_name]['time'].append(simulation_time)
                              converted_data[room_name]['co2'].append(co2_concentration)
                              converted_data[room_name]['damper_position'].append(air_damper_position)
                        else:
                              # If room_name does not exist, create a new dictionary for the room
                              converted_data[room_name] = {
                                    'time': [simulation_time],
                                    'co2': [co2_concentration],
                                    'damper_position': [air_damper_position]
                              }
            else:
                  print("Fetching Ventilation Real Sensor Data")
                  for room_name in room_names:
                        sensor_data_list = self.connector.get_data_using_datetime(
                              tablename=table_name, roomname=room_name, starttime=start_time, endtime=end_time)
                        
                        # Replace "O" with "OE" in room name
                        converted_room_name = room_name.replace("O", "OE")

                        # Initialize the converted data dictionary for this room if it doesn't exist
                        converted_data[converted_room_name] = {'time': [], 'co2': [], 'damper_position': []}

                        # Process sensor data
                        for data_point in sensor_data_list:
                              for field, value in data_point.__dict__.items():
                                    if field in column_filter:
                                          if field == 'opcuats':
                                                converted_data[converted_room_name]['time'].append(str(value))
                                          if field == 'co2concentration':
                                                converted_data[converted_room_name]['co2'].append(value)
                                          if field == 'damper':
                                                converted_data[converted_room_name]['damper_position'].append(value)
                              
                        # Check if data is missing for the specific room
                        if not converted_data[converted_room_name]['time'] or not converted_data[converted_room_name]['co2'] or not converted_data[converted_room_name]['damper_position']:
                              converted_data[converted_room_name]['sensor_data_flag'] = True

                              # Example of printing the converted data with the flag
                              print(f"Room: {converted_room_name}, Missing sensor data ")
                              
                              json_filename  = converted_room_name+"_schedules.json"
                              converted_data[converted_room_name]["schedules"] = self.read_config_json(config_filename = json_filename)
                  
            # Preprocess and organize the input data
            input_data["metadata"] = metadata
            input_data["rooms_sensor_data"] = converted_data

            return input_data


# # Example usage when the script is run directly
if __name__ == "__main__":
    # Create an instance of the input_data class
    inputdata = input_data()
#     start_time = '2024-02-12 02:13:46+00'
#     end_time = '2024-02-13 10:13:46+00'
#     input_temp_data=inputdata.input_data_for_ventilation(start_time,end_time)

#     from twin4build.api.codes.ml_layer.validator import Validator 

#     ventilation_validator = Validator()

#     ventilation_results = ventilation_validator.validate_ventilation_input(input_temp_data)

#     print(ventilation_results)


    current_time = datetime.now()
    end_time = current_time -  timedelta(hours=4)
    start_time = end_time -  timedelta(hours=5)
    forecast = False
    
    formatted_endtime= end_time.strftime('%Y-%m-%d %H:%M:%S%z')
    formatted_startime= start_time.strftime('%Y-%m-%d %H:%M:%S%z')

    input_temp_data=inputdata.input_data_for_ventilation(formatted_startime,formatted_endtime)

    #File path to save JSON data
    file_path = 'ventilation_input_data.json'

    # Save dictionary to JSON file
    with open(file_path, 'w') as json_file:
      json.dump(input_temp_data, json_file, indent=4)
      #print(input_temp_data)
    
    #inputdata.input_data_for_simulation(formatted_startime,formatted_endtime,forecast)