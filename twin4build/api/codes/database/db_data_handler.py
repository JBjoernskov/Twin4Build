
"""
Class For ml_inputs table in postgres
used SQLAlchemy for ORM to connect with database
ConfigReader is used to get configuration data for the database url
"""

# Import necessary modules and packages
from twin4build.utils.uppath import uppath
from twin4build.logger.Logging import Logging
from twin4build.config.Config import ConfigReader
import os
import sys
from datetime import datetime

from sqlalchemy import create_engine, Column, String, TEXT, DateTime, Integer, Float, JSON, BIGINT, BigInteger
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import declarative_base
from sqlalchemy import desc

# Create a base class for SQLAlchemy declarative models
Base = declarative_base()

# Only for testing before distributing package
if __name__ == '__main__':
    def uppath(_path, n): return os.sep.join(_path.split(os.sep)[:-n])
    file_path = uppath(os.path.abspath(__file__), 5)
    sys.path.append(file_path)

# Import required modules from custom packages

# Initialize the logger
logger = Logging.get_logger('ai_logfile')

# Define the base class for the ml_inputs table
Base = declarative_base()

# Define a class representing the 'ml_inputs' table in the database


class ml_inputs(Base):
    # Specify the table name
    __tablename__ = 'ml_inputs'

    # Define columns for the table
    entity_id = Column(String)
    entity_type = Column(String)
    time_index = Column(DateTime)
    __original_ngsi_entity__ = Column(JSON)
    instanceid = Column(String)
    datecreated = Column(DateTime)
    datemodified = Column(DateTime)
    iscontainedinbuildingspace = Column(String)
    co2concentration = Column(Float)
    damper = Column(Float)
    name = Column(String)
    opcuats = Column(DateTime)
    radiator = Column(Float)
    shadingposition = Column(Float)
    temperature = Column(Float)
    id = Column(BIGINT, primary_key=True,  nullable=False)


class ml_simulation_results(Base):
    # Specify the table name
    __tablename__ = 'ml_simulation_results'

    # Define columns for the table
    id = Column(Integer, primary_key=True, autoincrement=True)
    spacename = Column(String, nullable=False)
    simulation_time = Column(DateTime)
    outdoorenvironment_outdoortemperature = Column(Float)
    outdoorenvironment_globalirradiation = Column(Float)
    indoortemperature = Column(Float)
    indoorco2concentration = Column(Float)
    supplydamper_airflowrate = Column(Float)
    supplydamper_damperposition = Column(Float)
    exhaustdamper_airflowrate = Column(Float)
    exhaustdamper_damperposition = Column(Float)
    spaceheater_outletwatertemperature = Column(Float)
    spaceheater_power = Column(Float)
    spaceheater_energy = Column(Float)
    valve_waterflowrate = Column(Float)
    valve_valveposition = Column(Float)
    temperaturecontroller_inputsignal = Column(Float)
    co2controller_inputsignal = Column(Float)
    temperaturesensor_indoortemperature = Column(Float)
    valvepositionsensor_valveposition = Column(Float)
    damperpositionsensor_damperposition = Column(Float)
    co2sensor_indoorco2concentration = Column(Float)
    heatingmeter_energy = Column(Float)
    occupancyschedule_schedulevalue = Column(Float)
    temperaturesetpointschedule_schedulevalue = Column(Float)
    supplywatertemperatureschedule_supplywatertemperaturesetpoint = Column(Float)
    ventilationsystem_supplyairtemperatureschedule_schedulevaluet = Column(TEXT)
    input_start_datetime = Column(DateTime)
    input_end_datetime = Column(DateTime)


class ml_inputs_dmi(Base):
    # Specify the table name
    __tablename__ = 'ml_inputs_dmi'

    # Define columns for the table
    entity_id = Column(TEXT(), primary_key=True)
    entity_type = Column(TEXT())
    time_index = Column(DateTime(timezone=True), nullable=False)
    fiware_servicepath = Column(TEXT())
    __original_ngsi_entity__ = Column(JSON)
    instanceid = Column(TEXT())
    latitude = Column(Float)
    longitude = Column(Float)
    observed = Column(DateTime(timezone=True))
    radia_glob = Column(Float)
    stationid = Column(BigInteger)
    temp_dry = Column(Float)
    location = Column(String)
    location_centroid = Column(String)
    id = Column(BIGINT, primary_key=True,  nullable=False)


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
            "ml_simulation_results": ml_simulation_results
        }

    # Configuration function get read data from config.ini file
    def get_configuration(self):
        logger.info("[DBConnector : Configuration  Function]")
        try:
            conf = ConfigReader()
            config_path = os.path.join(os.path.abspath(
                uppath(os.path.abspath(__file__), 4)), "config", "conf.ini")
            self.config = conf.read_config_section(config_path)
            logger.info(
                "[DBConnector : configuration hasd been read from file ]")
        except Exception as e:
            logger.error(
                "[db_connector] : Error reading config file Exception Occured:", e)
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
            logger.error(f"Error connecting to PostgresSQL : {e}")
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
            logger.error(f"Error disconnecting from PostgreSQL: {e}")
            print(f"Error disconnecting from PostgreSQL: {e}")

    def create_table(self):
        # schema = self.config['db_cred']['schema']
        # self.engine.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
        # Base.metadata.create_all(self.engine, schema=schema)
        logger.info("DBConnector Class : Creating Table")
        Base.metadata.create_all(self.engine)

    def add_data(self, table_name, inputs):
        """
            Adding data to table in the database
        """

        try:
            inputs_data = self.tables[table_name](**inputs)
            self.session.add(inputs_data)
            self.session.commit()
            self.session.close()
            logger.info(" added to the database")
            print(" added to database")

        except Exception as e:
            logger.error("Failed to add  to the database and error is: ", e)
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
            queried_data = self.session.query(self.tables[table_name]).order_by(
                desc(self.tables[table_name].time_index)).all()
            self.session.close()
            logger.info(f"{table_name} retrieved from the database")
            print(f"{table_name} retrieved from database")
            return queried_data
        except Exception as e:
            logger.error(
                f"Failed to retrieve {table_name} from database and error is: ", e)
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
                logger.info(f"Latest {table_name} retrieved from the database")
                print(f"Latest {table_name} retrieved from the database")
                return queried_data
            else:
                logger.info(f"No {table_name} found in the database")
                print(f"No {table_name} found in the database")
                return None

        except Exception as e:
            logger.error(
                f"Failed to retrieve latest {table_name} from the database, and error is: ", e)
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
            list: A list of queried data within the specified time range.
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

                logger.info(
                    f"{tablename} retrieved from the database based on time range")
                print(f"{tablename} retrieved from database based on time range")
            return queried_data
        except Exception as e:
            logger.error(
                f"Failed to retrieve {tablename} from database based on time range, and error is: ", e)
            print(
                f"Failed to retrieve {tablename} from database based on time range, and error is: ", e)
        return None


# Example usage:
if __name__ == "__main__":
    connector = db_connector()
    connector.connect()
    # connector.create_table()
    roomname = "O20-601b-2"

    start_datetime = "2023-08-17 08:50:00"
    end_datetime = "2023-08-22 10:40:00"

    start_datetime = datetime.strptime(start_datetime, '%Y-%m-%d %H:%M:%S')
    end_datetime = datetime.strptime(end_datetime, '%Y-%m-%d %H:%M:%S')

    tablename = "ml_simulation_results"

    inputs = {
        'simulation_time': '2022-01-03 00:00:00', 
        'outdoorenvironment_outdoortemperature': 8.426348689, 
        'outdoorenvironment_globalirradiation': 0.0, 
        'indoortemperature': 21.10011443454423, 
        'indoorco2concentration': 493.15416255441016, 
        'supplydamper_airflowrate': 0.0, 
        'supplydamper_damperposition': 0.0, 
        'exhaustdamper_airflowrate': 0.0, 
        'exhaustdamper_damperposition': 0.0, 
        'spaceheater_outletwatertemperature': '[20.848304165052905, 20.84829637244803, 20.848290848601764, 20.848286353047286, 20.84828264071496, 20.848279569306204, 20.848277026456365, 20.84827492014335, 20.848273174664612, 20.848271727655494]', 
        'spaceheater_power': -65.37979713403159, 
        'spaceheater_energy': -0.0108966328556719,
        'valve_waterflowrate': 0.0, 
        'valve_valveposition': 0.0, 
        'temperaturecontroller_inputsignal': 0.0, 
        'co2controller_inputsignal': 0.0, 
        'temperaturesensor_indoortemperature': 21.10011443454423, 
        'valvepositionsensor_valveposition': 0.0,
        'damperpositionsensor_damperposition': 0.0, 
        'co2sensor_indoorco2concentration': 493.15416255441016, 
        'heatingmeter_energy': -0.0108966328556719, 
        'occupancyschedule_schedulevalue': 0, 
        'temperaturesetpointschedule_schedulevalue': 20, 
        'supplywatertemperatureschedule_supplywatertemperaturesetpoint': 60.5, 
        'ventilationsystem_supplyairtemperatureschedule_schedulevaluet': 21, 
        'input_start_date': '2023-08-22 02:50:00', 
        'input_end_date': '2023-08-23 10:40:00', 
        'spacename': 'O20-601b-2'
        }

    connector.add_data(table_name=tablename, inputs=inputs)

    connector.disconnect()