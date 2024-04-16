
"""
Class For ml_inputs table in postgres
used SQLAlchemy for ORM to connect with database
ConfigReader is used to get configuration data for the database url
"""

# import libraries
import os
import sys
import json
from datetime import datetime, timedelta
from dateutil.parser import parse
from sqlalchemy.dialects.postgresql import insert

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import declarative_base
from sqlalchemy import desc , TIMESTAMP

# Create a base class for SQLAlchemy declarative models
Base = declarative_base()

# Only for testing before distributing package
if __name__ == '__main__':
    def uppath(_path, n): return os.sep.join(_path.split(os.sep)[:-n])
    file_path = uppath(os.path.abspath(__file__), 5)
    sys.path.append(file_path)

    
# Import necessary modules and packages
from twin4build.utils.uppath import uppath
from twin4build.logger.Logging import Logging
from twin4build.config.Config import ConfigReader
from twin4build.api.codes.database.db_tables import *


# Import required modules from custom packages

# Initialize the logger
logger = Logging.get_logger('API_logfile')

# Define a class to handle database connections and operations
class db_connector:
    def __init__(self):
        # Initialize the logger and read configuration
        logger.info("[DBConnector : Initialise Function]")
        self.get_configuration()
        self.connection_string = self.get_connection_string()

        # Define table classes for reference
        self.tables = {
            "ml_inputs": ml_inputs,
            "ml_inputs_dmi": ml_inputs_dmi,
            "ml_simulation_results": ml_simulation_results,
            "ml_forecast_simulation_results" : MLForecastSimulationResult,
            "ml_forecast_inputs_dmi" : MLForecastInputsDMI,
            "ml_what_if_results": ml_what_if_results,
            'ml_ventilation_dummy_inputs':ventilation_dummy_inputs,
            'ml_ventilation_simulation_results':ventilation_simulation_results
        }

    # Configuration function get read data from config.ini file
    def get_configuration(self):
        '''
            Function to connect to the config file
        '''
        logger.info("[DBConnector : Configuration  Function]")
        try:
            conf = ConfigReader()
            config_path = os.path.join(os.path.abspath(
                uppath(os.path.abspath(__file__), 4)), "config", "conf.ini")
            self.config = conf.read_config_section(config_path)
            logger.info("[DBConnector : configuration hasd been read from file ]")
        except Exception as e:
            logger.error("[db_connector] : Error reading config file Exception Occured:")
            print("[db_connector] : Error reading config file Exception Occured:", e)

    # this funtion returns the connection string for the databse
    def get_connection_string(self):
        '''
            Reading configration data from config.ini file using ConfigReader
        '''
        logger.info("[DBConnector : Connection String Function Entered]")
        self.username = self.config['db_cred']['username']
        self.password = self.config['db_cred']['password']
        self.host = self.config['db_cred']['host']
        self.port = self.config['db_cred']['port']
        self.database_name = self.config['db_cred']['database_name']

        # Create the database connection string
        connection_string = f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database_name}"

        logger.info("[DBConnector : Connection String Function Exit]")
        return connection_string

    def connect(self):
        """
        using SQLAlchemy to connect to the database
        """
        logger.info("[DBConnector : Database  Connect Function Entered]")
        schema = self.config['db_cred']['schema']

        # pay special aatention on schema
        self.engine = create_engine(self.connection_string, connect_args={
                                    'options': '-csearch_path={}'.format(schema)})
        Session = sessionmaker(bind=self.engine)
        self.session = Session()

        try:
            self.engine.connect()
            logger.info("Connection to PostgreSQL established successfully.")
            print("Connection to PostgreSQL established successfully.")
        except Exception as e:
            logger.error("Error connecting to PostgresSQL : ")
            print(f"Error connecting to PostgreSQL: {e}")

    def disconnect(self):
        """
            Dis-Connecting from the Database 
        """
        try:
            self.engine.dispose()
            logger.info("Connection to PostgresSQL closed Successfully")
            print("Connection to PostgreSQL closed successfully.")

        except Exception as e:
            logger.error("Error disconnecting from PostgreSQL")
            print(f"Error disconnecting from PostgreSQL: {e}")

    def create_table(self):
        # schema = self.config['db_cred']['schema']
        # self.engine.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
        # Base.metadata.create_all(self.engine, schema=schema)
        logger.info("DBConnector Class : Creating Table")
        Base.metadata.create_all(self.engine)

    def add_large_data(self, table_name, inputs):
        if table_name == 'ml_ventilation_dummy_inputs':
                self.session.bulk_insert_mappings(self.tables[table_name],inputs)      
                self.session.commit()
                self.session.close()


    def add_data(self, table_name, inputs):
        """
            Adding data to table in the database
        """

        try:
            '''
            inputs_data = self.tables[table_name](**inputs)
            self.session.add(inputs_data)
            self.session.commit()
            #self.session.close()

            '''

            if len(inputs) < 1:
                logger.error("Empty data got for entering to database")
                print("Empty data got for entering to database")
                return None
            
            if table_name == 'ml_forecast_inputs_dmi':
                self.session.bulk_insert_mappings(self.tables[table_name],inputs)      
                self.session.commit()

                                    
            # Track whether any updates or additions have occurred
            updated = False
            added = False
                            
            for input_data in inputs:
                # Check if a record with the same simulation_time already exists

                existing_record = (
                    self.session.query(self.tables[table_name])
                    .filter_by(simulation_time=input_data['simulation_time'])
                    .first()
                )

                if existing_record:
                        # Update existing record
                        self.session.query(self.tables[table_name]).filter_by(id=existing_record.id).update(input_data)
                        self.session.commit()
                        updated = True

                else:
                    # Insert new record
                    self.session.bulk_insert_mappings(self.tables[table_name], [input_data])      
                    self.session.commit()
                    added = True

            if updated:
                logger.info(" updated values to the database %s table",table_name)
                print(" updated values to database",table_name)                   
            
            if added:
                logger.info(" added to the database %s",table_name)
                print(" added to database",table_name)

            self.session.close()

        except Exception as e:
            logger.error("Failed to add  to the database and error is: ")
            print("Failed to add  to database and error is: ", e)

    def get_all_inputs(self, table_name):
        """
        Query all data from the specified table.

        Args:
            table_name (str): Name of the table to query.

        Returns:
            list: A list of queried data from the specified table.
        """
        queried_data = []

        try:

            if table_name == 'ml_simulation_results':
                queried_data = self.session.query(ml_simulation_results).all()
                return queried_data

            if table_name == 'ml_forecast_inputs_dmi':
                queried_data = self.session.query(self.tables[table_name]).all()
                self.session.close()
                logger.info("retrieved from the database")
                print(f"{table_name} retrieved from database")
                return queried_data
            
            if table_name == 'ml_inputs_dmi':
                queried_data = self.session.query(self.tables[table_name]).order_by(
                    desc(self.tables[table_name].time_index)).all()
            
            self.session.close()
            logger.info("retrieved from the database")
            print(f"{table_name} retrieved from database")
            return queried_data
        
        except Exception as e:
            logger.error("Failed to retrieve from database and error is: ")
            print(
                f"Failed to retrieve {table_name} from database and error is: ", e)



    def get_filtered_forecast_inputs(self, table_name,start_time,end_time):
        """
        Query data from the forcast inputs dmi based on start time and end time

        Args:
            table_name (str): Name of the table to query.

        Returns:
            list: A list of queried data from the specified table.
        """
        queried_data = []
    
        ### ADDED BY JAKOB FROM SDU ###
        start_time_filter = parse(start_time)
        end_time_filter = parse(end_time)
        start_time_filter = start_time_filter.replace(second=0, microsecond=0, minute=0, hour=start_time_filter.hour)-timedelta(hours=1) # Floor the start time. We subtract 1 hours to make sure machine precision doesn't influence the filtering
        end_time_filter = end_time_filter.replace(second=0, microsecond=0, minute=0, hour=end_time_filter.hour)+timedelta(hours=2) # Ceil the end time. We add 2 hours to make sure machine precision doesn't influence the filtering
        start_time_filter = start_time_filter.strftime('%Y-%m-%d %H:%M:%S%z')
        end_time_filter = end_time_filter.strftime('%Y-%m-%d %H:%M:%S%z')
        ###############################
        
        try:
            queried_data = self.session.query(self.tables[table_name]).filter(
                self.tables[table_name].forecast_time >= start_time_filter,
                self.tables[table_name].forecast_time <= end_time_filter
            ).all()
            
            self.session.close()
            logger.info("retrieved from the database")
            print(f"{table_name} retrieved from database")
            return queried_data
        except Exception as e:
            logger.error("Failed to retrieve from database and error is: ")
            print(
                f"Failed to retrieve {table_name} from database and error is: ", e)

    def get_latest_values(self, table_name, roomname):
        """
        Query latest data from the specified table.

        Args:
            table_name (str): Name of the table to query.
            roomname (str) : Name of the room for which query needs to be performed

        Returns:
            list: A list of queried data from the specified table.
        """
        # change this query based on tabled
        queried_data = []
        try:
            # Query the table and order the results by time_index in descending order
            if table_name == "ml_inputs":
                queried_data = self.session.query(self.tables[table_name]).filter_by(
                    name=roomname).order_by(desc(self.tables[table_name].time_index)).first()
            elif table_name == "ml_inputs_dmi":
                queried_data = self.session.query(self.tables[table_name]).order_by(
                    desc(self.tables[table_name].time_index)).first()

            self.session.close()

            if queried_data:
                logger.info("Latest retrieved from the database")
                print(f"Latest {table_name} retrieved from the database")
                return queried_data
            else:
                logger.info("No found in the database")
                print(f"No {table_name} found in the database")
                return None

        except Exception as e:
            logger.error("Failed to retrieve latest from the database, and error is: ")
            print(
                f"Failed to retrieve latest {table_name} from the database, and error is: ", e)
            return None

    def get_data_using_datetime(self, tablename, roomname, starttime, endtime):
        """
        Retrieve data from the ml_inputs table based on the specified time range.

        Args:
            starttime (datetime): Start time of the desired time range.
            endtime (datetime): End time of the desired time range.

        Returns:
            list: A list of queried data for single room within the specified time range.
        """

        queried_data = []

        try:
            if tablename == "ml_inputs":
                queried_data = self.session.query(self.tables[tablename]).filter_by(name=roomname).filter(
                    self.tables[tablename].opcuats >= starttime,
                    self.tables[tablename].opcuats <= endtime
                ).order_by(self.tables[tablename].opcuats).all()

            if tablename == "ml_inputs_dmi":
                queried_data = self.session.query(self.tables[tablename]).filter(
                    self.tables[tablename].observed >= starttime,
                    self.tables[tablename].observed <= endtime
                ).order_by(self.tables[tablename].observed).all()

                self.session.close()

            logger.info(" retrieved from the database based on time range")
            print(f"{tablename} retrieved from database based on time range")
            return queried_data
        
        except Exception as e:
            logger.error("Failed to retrieve from database based on time range ")
            print(
                f"Failed to retrieve {tablename} from database based on time range, and error is: ", e)
        return None
    
    def get_data_using_forecast(self,forecast_time):
        queried_data = []

        try:
            queried_data = self.session.query(MLForecastInputsDMI).filter(
                MLForecastInputsDMI.forecast_time == forecast_time
            ).all()

            return queried_data
        
        except Exception as e:
            return queried_data
        
    def update_forecast_data(self, forecast_time, updated_values):
        """
        Update forecast data in the ml_forecast_inputs_dmi table based on the specified forecast time.

        Args:
            forecast_time (datetime): Forecast time to identify the record to update.
            updated_values (dict): Dictionary containing the fields and their updated values.

        Returns:
            bool: True if the update is successful, False otherwise.
        """
        try:
            # Query the record to update
            records_to_update = self.session.query(MLForecastInputsDMI).filter(
                MLForecastInputsDMI.forecast_time == forecast_time
            ).all()

            if records_to_update:
                # Update the record with the new values
                for record in records_to_update:
                    for key, value in updated_values.items():
                        setattr(record, key, value)

                # Commit the changes to the database
                self.session.commit()
                self.session.close()

                logger.info("Forecast data updated successfully for forecast_time")
                print(f"Forecast data updated successfully for forecast_time: {forecast_time}")
                return True
            else:
                logger.info("No forecast data found for the specified forecast_time")
                print(f"No forecast data found for the specified forecast_time: {forecast_time}")
                return False

        except Exception as e:
            self.session.rollback()
            logger.error("Failed to update forecast data for forecast_time")
            print(f"Failed to update forecast data for forecast_time {forecast_time}. Error: {e}")
            return False
        
    def get_multiple_rooms_data_filterby_time(self, table_name, room_names, start_time, end_time):
        """
        Retrieve data of multiple rooms from the database based on the specified time range.

        Args:
            starttime (datetime): Start time of the desired time range.
            endtime (datetime): End time of the desired time range.

        Returns:
            list: A list of queried data within the specified time range.
        """
        queried_data = []
        
        start_time_filter = parse(start_time)
        end_time_filter = parse(end_time)
        start_time_filter = start_time_filter.replace(second=0, microsecond=0, minute=0, hour=start_time_filter.hour)-timedelta(hours=1) # Floor the start time. We subtract 1 hours to make sure machine precision doesn't influence the filtering
        end_time_filter = end_time_filter.replace(second=0, microsecond=0, minute=0, hour=end_time_filter.hour)+timedelta(hours=2) # Ceil the end time. We add 2 hours to make sure machine precision doesn't influence the filtering
        start_time_filter = start_time_filter.strftime('%Y-%m-%d %H:%M:%S%z')
        end_time_filter = end_time_filter.strftime('%Y-%m-%d %H:%M:%S%z')

        try:
            queried_data = self.session.query(
                self.tables[table_name].room_name,
                self.tables[table_name].simulation_time,
                self.tables[table_name].co2concentration,
                self.tables[table_name].air_damper_position).filter(
                self.tables[table_name].room_name.in_(room_names),
                self.tables[table_name].simulation_time >= start_time_filter,
                self.tables[table_name].simulation_time <= end_time_filter
            )
            
            self.session.close()
            logger.info("retrieved from the database")
            print(f"{table_name} retrieved from database")
            return queried_data
        except Exception as e:
            logger.error("Failed to retrieve from database and error is: ")
            print(
                f"Failed to retrieve {table_name} from database and error is: ", e)

# Example usage:
if __name__ == "__main__":
    connector = db_connector()
    connector.connect()

    
    #connector.create_table()
    #roomname = "O20-601b-2"
    room_names = ['0E22-604-00', '0E22-601B-00', 'OE22-604D-2']
    table_name = "ml_ventilation_dummy_inputs"
    start_time = '2024-02-12 02:13:46+00'
    end_time = '2024-02-14 10:13:46+00'
    queried_data = connector.get_multiple_rooms_data_filterby_time(table_name, room_names, start_time, end_time)

     # Print the queried data
    for row in queried_data:
        print("Simulation Time:", row.simulation_time)
        print("CO2:", row.co2concentration)
        print("Damper:", row.air_damper_position)
        print()

    #all_inputs = connector.get_all_inputs(tablename)

   #print(len(multiple_rooms_data))


    connector.disconnect()

'''
# Create a sample data dictionary
sample_data = [{
    'spacename': 'SampleSpace',
    'simulation_time': datetime.now(),
    'outdoorenvironment_outdoortemperature': 25.0,
    'outdoorenvironment_globalirradiation': 500.0,
    'indoortemperature': 22.0,
    'indoorco2concentration': 400.0,
    'supplydamper_airflowrate': 1000.0,
    'supplydamper_damperposition': 0.7,
    'exhaustdamper_airflowrate': 800.0,
    'exhaustdamper_damperposition': 0.5,
    'spaceheater_outletwatertemperature': 'High',
    'spaceheater_power': 2000.0,
    'spaceheater_energy': 500.0,
    'valve_waterflowrate': 50.0,
    'valve_valveposition': 0.8,
    'temperaturecontroller_inputsignal': 23.5,
    'co2controller_inputsignal': 450.0,
    'temperaturesensor_indoortemperature': 22.5,
    'valvepositionsensor_valveposition': 0.75,
    'damperpositionsensor_damperposition': 0.6,
    'co2sensor_indoorco2concentration': 420.0,
    'heatingmeter_energy': 700.0,
    'occupancyschedule_schedulevalue': 1.0,
    'temperaturesetpointschedule_schedulevalue': 24.0,
    'supplywatertemperatureschedule_supplywatertemperaturesetpoint': 60.0,
    'ventilationsystem_supplyairtemperatureschedule_schedulevaluet': 26.0,
    'input_start_datetime': datetime.now(),
    'input_end_datetime': datetime.now(),
}]

sample_dict = [{
    'forecast_time': '2023-11-22 12:00:00',
    'latitude': 40.7128,
    'longitude': -74.0060,
    'radia_glob': 300.5,
    'temp_dry': 25.5,
    'stationid': 1001
}]

#connector.add_data(tablename,sample_dict)

#updated_values_dict = {
#   'stationid': 1002
#} 

#data = connector.get_all_inputs(tablename)
#connector.update_forecast_data('2023-11-23 07:00:00+00',updated_values_dict)



# data = connector.get_data_using_datetime(roomname=roomname,tablename="ml_inputs",endtime=end_datetime,starttime=start_datetime)
'''