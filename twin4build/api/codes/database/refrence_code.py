from sqlalchemy import create_engine, Column, Integer, String, MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

class PostgreSQLConnector:
    def __init__(self, username, password, host, port, database_name):
        # Create the database connection string
        self.connection_string = f"postgresql://{username}:{password}@{host}:{port}/{database_name}"
        
        # Create the engine
        self.engine = create_engine(self.connection_string)
        
        # Create the base class for declarative models
        self.Base = declarative_base()
        
        # Create a session
        self.Session = sessionmaker(bind=self.engine)
        
    def connect(self):
        try:
            self.engine.connect()
            print("Connection to PostgreSQL established successfully.")
        except Exception as e:
            print(f"Error connecting to PostgreSQL: {e}")
        
    def disconnect(self):
        try:
            self.engine.dispose()
            print("Connection to PostgreSQL closed successfully.")
        except Exception as e:
            print(f"Error disconnecting from PostgreSQL: {e}")
    
    def create_table(self, table_name):
        class TableModel(self.Base):
            __tablename__ = table_name
            id = Column(Integer, primary_key=True)
            name = Column(String)
            # Add more columns here as needed
        
        # Create the table
        self.Base.metadata.create_all(self.engine)
        print(f"Table '{table_name}' created successfully.")
        
    def query_table(self, table_name):
        Session = self.Session()
        try:
            table_data = Session.query(getattr(self.Base.classes, table_name)).all()
            return table_data
        except Exception as e:
            print(f"Error querying the table '{table_name}': {e}")
        finally:
            Session.close()

# Example usage:
if __name__ == "__main__":
    # Replace these values with your PostgreSQL credentials
    username = "your_username"
    password = "your_password"
    host = "localhost"
    port = 5432
    database_name = "your_database_name"
    
    connector = PostgreSQLConnector(username, password, host, port, database_name)
    connector.connect()
    
    # Replace "your_table_name" with the actual name of your table
    table_name = "your_table_name"
    connector.create_table(table_name)
    
    # Query the table
    table_data = connector.query_table(table_name)
    for row in table_data:
        print(row.id, row.name)
    
    connector.disconnect()
