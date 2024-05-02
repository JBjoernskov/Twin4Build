# import libraries

from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, String, TEXT, DateTime, Integer, Float, JSON, BIGINT, BigInteger,TIMESTAMP

# Create a base class for SQLAlchemy declarative models
Base = declarative_base()


# Define a class representing the 'ml_forecast_inputs_dmi' table in the database
class MLForecastInputsDMI(Base):
    __tablename__ = 'ml_forecast_inputs_dmi'
    
    id = Column(BigInteger, primary_key=True, server_default="nextval('ml_schema.ml_forecast_inputs_dmi_id_seq'::regclass)", nullable=False)
    forecast_time = Column(TIMESTAMP(timezone=True))
    latitude = Column(Float)
    longitude = Column(Float)
    radia_glob = Column(Float)
    temp_dry = Column(Float)
    stationid = Column(BigInteger)

