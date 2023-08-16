
"""
Class For ml_inputs table in postgres
used SQLAlchemy for ORM to connect with database
ConfigReader is used to get configuration data for the database url
"""

import os 
import sys

from sqlalchemy import create_engine, Column, String , DateTime , Float , JSON,BIGINT
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import declarative_base
from sqlalchemy import desc
from datetime import datetime
from datetime import timezone

Base = declarative_base()

###Only for testing before distributing package
if __name__ == '__main__':
    uppath = lambda _path,n: os.sep.join(_path.split(os.sep)[:-n])
    file_path = uppath(os.path.abspath(__file__), 5)
    sys.path.append(file_path)

from twin4build.config.Config import ConfigReader
from twin4build.logger.Logging import Logging

logger = Logging.get_logger('ai_logfile')

Base = declarative_base()


class ml_inputs_table(Base):
    __tablename__ = 'ml_inputs'
    
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
    Id = Column(BIGINT,primary_key=True,  nullable=False)


class db_connector:
    def __init__(self):
        logger.info("[DBConnector : Initialise Function]")
        self.get_configuration()
        self.connection_string = self.get_connection_string()

    def get_configuration(self):
       logger.info("[DBConnector : Configuration  Function]")
       try:
        conf=ConfigReader()
        config_path = os.path.join(os.path.abspath(uppath(os.path.abspath(__file__), 4)), "config", "conf.ini")
        self.config=conf.read_config_section(config_path)
        logger.info("[DBConnector : configuration hasd been read from file ]")
       except Exception as e :
           logger.error("[db_connector] : Error reading config file Exception Occured:",e)
           print("[db_connector] : Error reading config file Exception Occured:",e)

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
        self.engine = create_engine(self.connection_string, connect_args={'options': '-csearch_path={}'.format(schema)})
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
        #schema = self.config['db_cred']['schema']
        #self.engine.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
        #Base.metadata.create_all(self.engine, schema=schema)
        logger.info("DBConnector Class : Creating Table")
        Base.metadata.create_all(self.engine)

    def add_ml_inputs(self, ml_inputs):
        """
            Adding data to ml_inputs table in the database
        """

        try:
            ml_inputs_data = ml_inputs_table(**ml_inputs)
            self.session.add(ml_inputs_data)
            self.session.commit()
            self.session.close()
            logger.info("ML Inputs added to the database")
            print("ML inputs added to database")

        except Exception as e:
            logger.error("Failed to add ML Inputs to the database and error is: ",e)
            print("Failed to add ML inputs to database and error is: ",e)

    def get_all_ml_inputs(self,roomname):
        """
            Querying all the data from the ml_input table 
        """
        try:
            queried_data = self.session.query(ml_inputs_table).filter_by(name=roomname).all()
            self.session.close()
            logger.info("ML Inputs retrieved from the database")
            print("ML inputs retrieved from database")
            return queried_data
        except Exception as e:
            logger.error("Failed to retrieve ML inputs from database and error is: ",e)
            print("Failed to retrieve ML inputs from database and error is: ",e)

    def get_latest_values(self,roomname):
        """
        Fetch the latest data from the ml_input table
        """
        try:
            # Query the table and order the results by time_index in descending order
            queried_data = self.session.query(ml_inputs_table).filter_by(name=roomname).order_by(desc(ml_inputs_table.time_index)).first()
            self.session.close()

            if queried_data:
                logger.info("Latest ML Inputs retrieved from the database")
                print("Latest ML inputs retrieved from the database")
                return queried_data
            else:
                logger.info("No ML Inputs found in the database")
                print("No ML inputs found in the database")
                return None

        except Exception as e:
            logger.error("Failed to retrieve latest ML inputs from the database, and error is: ", e)
            print("Failed to retrieve latest ML inputs from the database, and error is: ", e)
            return None 
        
                

    def get_data_using_datetime(self, roomname,starttime, endtime):
            """
            Retrieve data from the ml_inputs table based on the specified time range.

            Args:
                starttime (datetime): Start time of the desired time range.
                endtime (datetime): End time of the desired time range.

            Returns:
                list: A list of queried data within the specified time range.
            """
            try:
                queried_data = self.session.query(ml_inputs_table).filter_by(name=roomname).filter(
                    ml_inputs_table.time_index >= starttime,
                    ml_inputs_table.time_index <= endtime
                ).all()
                self.session.close()

                logger.info("ML Inputs retrieved from the database based on time range")
                print("ML inputs retrieved from database based on time range")
                return queried_data
            except Exception as e:
                logger.error("Failed to retrieve ML inputs from database based on time range, and error is: ", e)
                print("Failed to retrieve ML inputs from database based on time range, and error is: ", e)
                return None
            
    def get_ml_inputs_by_co2_range(self, roomname,min_co2, max_co2):
        try:
            queried_data = self.session.query(ml_inputs_table).filter_by(name=roomname).filter(
                ml_inputs_table.co2concentration >= min_co2,
                ml_inputs_table.co2concentration <= max_co2
            ).all()
            self.session.close()

            logger.info("ML Inputs retrieved from the database based on CO2 concentration range")
            print("ML inputs retrieved from the database based on CO2 concentration range")

        except Exception as e:
            logger.error("Failed to retrieve ML inputs from database based on co2 range, and error is: ", e)
            print("Failed to retrieve ML inputs from database based on co2 range, and error is: ", e)
            return None



# Example usage:
if __name__ == "__main__":
    connector = db_connector()
    connector.connect()
    connector.create_table()
    roomname = "O20-601b-2"

    # get_all_ml_inputs is retuning all the data from the datatable 
    data = connector.get_latest_values(roomname=roomname)
    print(data.co2concentration, data.damper)

    

    connector.disconnect()

