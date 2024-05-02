
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


from Config import ConfigReader
from dmi_tables import *


# Define a class to handle database connections and operations
class db_connector:
    def __init__(self):
        # Initialize the logger and read configuration
        self.get_configuration()
        self.connection_string = self.get_connection_string()

        # Define table classes for reference
        self.tables = {
            "ml_forecast_inputs_dmi" : MLForecastInputsDMI,
        }

    # Configuration function get read data from config.ini file
    def get_configuration(self):
        '''
            Function to connect to the config file
        '''
        try:
            conf = ConfigReader()
            config_path = "dmi_api_config.ini"
            self.config = conf.read_config_section(config_path)
        except Exception as e:
            print("[db_connector] : Error reading config file Exception Occured:", e)

    # this funtion returns the connection string for the databse
    def get_connection_string(self):
        '''
            Reading configration data from config.ini file using ConfigReader
        '''
        self.username = self.config['db_cred']['username']
        self.password = self.config['db_cred']['password']
        self.host = self.config['db_cred']['host']
        self.port = self.config['db_cred']['port']
        self.database_name = self.config['db_cred']['database_name']

        # Create the database connection string
        connection_string = f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database_name}"

        return connection_string

    def connect(self):
        """
        using SQLAlchemy to connect to the database
        """
        schema = self.config['db_cred']['schema']

        # pay special aatention on schema
        self.engine = create_engine(self.connection_string, connect_args={
                                    'options': '-csearch_path={}'.format(schema)})
        Session = sessionmaker(bind=self.engine)
        self.session = Session()

        try:
            self.engine.connect()
            print("Connection to PostgreSQL established successfully.")
        except Exception as e:
            print(f"Error connecting to PostgreSQL: {e}")

    def disconnect(self):
        """
            Dis-Connecting from the Database 
        """
        try:
            self.engine.dispose()
            print("Connection to PostgreSQL closed successfully.")

        except Exception as e:
            print(f"Error disconnecting from PostgreSQL: {e}")

    def create_table(self):
        # schema = self.config['db_cred']['schema']
        # self.engine.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
        # Base.metadata.create_all(self.engine, schema=schema)
        Base.metadata.create_all(self.engine)


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
                print(" updated values to database",table_name)                   
            
            if added:
                print(" added to database",table_name)

            self.session.close()

        except Exception as e:
            print("Failed to add  to database and error is: ", e)
            
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

                print(f"Forecast data updated successfully for forecast_time: {forecast_time}")
                return True
            else:
                print(f"No forecast data found for the specified forecast_time: {forecast_time}")
                return False

        except Exception as e:
            self.session.rollback()
            print(f"Failed to update forecast data for forecast_time {forecast_time}. Error: {e}")
            return False
        
    

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

