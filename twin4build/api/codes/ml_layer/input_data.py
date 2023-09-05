
"""Scripts/Functions to convert Data into Input format  """

# Import necessary modules
import os
import sys
import json
from datetime import datetime

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
logger = Logging.get_logger("ai_logfile")

# Create a class to handle input data conversion
class input_data:
      def __init__(self):
            # Initialize the configuration, database connection, process input data, and disconnect
            self.get_configuration()
            self.db_connect()
            #self.input_data_for_simulation()

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
            sensor_data = []

            try:
                  for table_name in table_names:
                        if data_fething_method == "get_all_inputs":
                              sensor_data = self.connector.get_all_inputs(table_name)

                        if data_fething_method == "get_data_using_datetime":
                              start_datetime = self.config["data_fetching_config"]["start_time"]
                              end_datetime = self.config["data_fetching_config"]["end_time"]

                              # Parse date and time strings into datetime objects
                              start_datetime = datetime.strptime(
                                    start_datetime, '%Y-%m-%d %H:%M:%S')
                              end_datetime = datetime.strptime(
                                    end_datetime, '%Y-%m-%d %H:%M:%S')

                              sensor_data = self.connector.get_data_using_datetime(
                                    tablename=table_name, roomname=roomname, starttime=start_datetime, endtime=end_datetime)
                              logger.info("Retrieved data for table: %s", table_name)
                        elif data_fething_method == "get_latest_values":
                              sensor_data = [self.connector.get_latest_values(
                                    table_name, roomname)]
                              logger.info("Retrieved data for table: %s", table_name)

                  
                        # storing data in the form of dict as table_name : data list
                        self.db_data[table_name] = sensor_data

            except Exception as e:
                  logger.error("Error fetching data from the database: %s", str(e))
                  self.db_data = {}  # Initialize an empty dictionary in case of error

            return self.db_data

      def get_filter_columns(self, table_name):
            """Get filter columns based on the table name"""
            columns_string = ""

            if table_name == "ml_inputs":
                  columns_string = self.config['ml_inputs_column_filters']['columns']
            elif table_name == "ml_inputs_dmi":
                  columns_string = self.config['ml_inputs_dmi_column_filters']['columns']

            # converting config.ini string data to the list of string separted by ','
            columns = [column.strip() for column in columns_string.split(',')]

            return columns

      def input_data_for_simulation(self):
            # Define the path for the config.json file
            config_json_path = os.path.join(os.path.abspath(
                  uppath(os.path.abspath(__file__), 4)), "config", "config.json")

            # Read JSON data from the config file
            with open(config_json_path, 'r') as json_file:
                  json_data = json_file.read()

            # Parse the JSON data
            data = json.loads(json_data)

            # Create a dictionary to store input data
            self.input_data = {}

            metadata = {}
            metadata["location"] = self.config["input_data_metadata"]["location"]
            metadata["building_id"] = self.config["input_data_metadata"]["building_id"]
            metadata["floor_number"] = self.config["input_data_metadata"]["floor_number"]
            metadata["room_id"] = self.config["input_data_metadata"]["room_id"]
            metadata["start_time"] = self.config["data_fetching_config"]["start_time"]
            metadata["end_time"] = self.config["data_fetching_config"]["end_time"]
            metadata['roomname'] = self.config['data_fetching_config']['roomname']
            metadata['stepSize'] = int(self.config['model']['stepSize'])
             
            # please add start and end period in metadat

            input_schedules = {}
            input_schedules["temperature_setpoint_schedule"] = data["temperature_setpoint_schedule"]
            input_schedules["shade_schedule"] = data["shade_schedule"]
            input_schedules["occupancy_schedule"] = data["occupancy_schedule"]
            input_schedules["supply_water_temperature_schedule_pwlf"] = data["supply_water_temperature_schedule_pwlf"]

            # Get sensor data from the database
            room_name = self.config["data_fetching_config"]["roomname"]
            table_names = self.config["data_fetching_config"]["table_names"]
            data_fetching_method = self.config["data_fetching_config"]["function_names"]

            table_names_string = self.config["data_fetching_config"]["table_names"]

            # Read table_names from config.ini file and convert to a list of table_name strings
            table_names = [name.strip() for name in table_names_string.split(',')]

            sensor_data_dict = self.data_from_db(
                  roomname=room_name, table_names=table_names, data_fething_method=data_fetching_method)

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
                                    data[table_name][field].append(str(value))
                  input_sensor_data.update(data)

            # Preprocess and organize the input data
            self.input_data["metadata"] = metadata
            self.input_data["inputs_sensor"] = input_sensor_data
            self.input_data["input_schedules"] = input_schedules

            logger.info("Input data has been successfully processed and saved.")
            
            return self.input_data

      def output_data(self,response):

            # changing information as required in the response
            response = {
                        "spacename": "Living Room",
                        "indoor_temperature": 23.5,
                        "co2concentration": 600.0,
                        "heat_consumption": 1200.0,
                        "creation_start_date" : self.input_data['metadata']['start_time'],
                        "creation_end_date" : self.input_data['metadata']['end_time'] ,   
                        "simulation_time": "2023-08-01",
                        "input_outdoor_temperature": 30.0,
                        "input_solar_irradiation": 800.0
                  }
            
            logger.info("Changes made in the reponse and is returned")
            
            return response

# Example usage when the script is run directly
if __name__ == "__main__":
    # Create an instance of the input_data class
    inputdata = input_data()

    inputdata.input_data_for_simulation()