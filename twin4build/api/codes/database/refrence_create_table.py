from sqlalchemy import create_engine, Column, Integer, String, MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Replace 'your_postgresql_username', 'your_postgresql_password', and 'your_postgresql_dbname' with your actual database credentials.
DATABASE_URI = 'postgresql://your_postgresql_username:your_postgresql_password@localhost:5432/your_postgresql_dbname'

# Create the SQLAlchemy engine
engine = create_engine(DATABASE_URI)
Base = declarative_base()
metadata = MetaData()

# Define your table model
class SampleTable(Base):
    __tablename__ = 'sample_table'
    id = Column(Integer, primary_key=True)
    name = Column(String(50))
    age = Column(Integer)

# Create the table in the database
def create_table():
    Base.metadata.create_all(engine)

# Function to add data to the table
def add_data(name, age):
    Session = sessionmaker(bind=engine)
    session = Session()
    data = SampleTable(name=name, age=age)
    session.add(data)
    session.commit()
    session.close()

# Function to retrieve all data from the table
def get_all_data():
    Session = sessionmaker(bind=engine)
    session = Session()
    data = session.query(SampleTable).all()
    session.close()
    return data

# Function to retrieve data by name
def get_data_by_name(name):
    Session = sessionmaker(bind=engine)
    session = Session()
    data = session.query(SampleTable).filter_by(name=name).all()
    session.close()
    return data

# Uncomment the line below if you want to create the table at this point
# create_table()


# Assuming you have already imported the code from above
# create_table()  # Uncomment this line if you want to create the table at this point

# Add data to the table
add_data("John", 30)
add_data("Alice", 25)

# Retrieve all data from the table
all_data = get_all_data()
print("All data:")
for data in all_data:
    print(f"ID: {data.id}, Name: {data.name}, Age: {data.age}")

# Retrieve data by name
name = "John"
data_by_name = get_data_by_name(name)
print(f"Data for {name}:")
for data in data_by_name:
    print(f"ID: {data.id}, Name: {data.name}, Age: {data.age}")
