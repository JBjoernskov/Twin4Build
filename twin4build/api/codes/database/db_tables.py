# import libraries

from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, String, TEXT, DateTime, Integer, Float, JSON, BIGINT, BigInteger,TIMESTAMP

# Create a base class for SQLAlchemy declarative models
Base = declarative_base()


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

# Define a class representing the 'ml_simulation_results' table in the database
class ml_simulation_results(Base):
    __tablename__ = 'ml_simulation_results'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    spacename = Column(String, nullable=False)
    simulation_time = Column(DateTime(timezone=True))
    outdoorenvironment_outdoortemperature = Column(Float)
    outdoorenvironment_globalirradiation = Column(Float)
    indoortemperature = Column(Float)
    indoorco2concentration = Column(Float)
    supplydamper_airflowrate = Column(Float)
    supplydamper_damperposition = Column(Float)
    exhaustdamper_airflowrate = Column(Float)
    exhaustdamper_damperposition = Column(Float)
    spaceheater_outletwatertemperature = Column(String)
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
    ventilationsystem_supplyairtemperatureschedule_schedulevaluet = Column(Float)
    input_start_datetime = Column(DateTime(timezone=True))
    input_end_datetime = Column(DateTime(timezone=True))

# Define a class representing the 'ml_inputs_dmi' table in the database
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


# Define a class representing the 'ml_forecast_simulation_results' table in the database
class MLForecastSimulationResult(Base):
    __tablename__ = 'ml_forecast_simulation_results'

    id = Column(Integer, primary_key=True, autoincrement=True)
    spacename = Column(String, nullable=False)
    simulation_time = Column(DateTime(timezone=True))
    outdoorenvironment_outdoortemperature = Column(Float)
    outdoorenvironment_globalirradiation = Column(Float)
    indoortemperature = Column(Float)
    indoorco2concentration = Column(Float)
    supplydamper_airflowrate = Column(Float)
    supplydamper_damperposition = Column(Float)
    exhaustdamper_airflowrate = Column(Float)
    exhaustdamper_damperposition = Column(Float)
    spaceheater_outletwatertemperature = Column(String)
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
    ventilationsystem_supplyairtemperatureschedule_schedulevaluet = Column(Float)
    input_start_datetime = Column(DateTime(timezone=True))
    input_end_datetime = Column(DateTime(timezone=True))


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

class ml_what_if_results(Base):
    __tablename__ = 'ml_what_if_results'

    id = Column(Integer, primary_key=True, autoincrement=True)
    spacename = Column(String, nullable=False)
    simulation_time = Column(DateTime(timezone=True))
    outdoorenvironment_outdoortemperature = Column(Float)
    outdoorenvironment_globalirradiation = Column(Float)
    indoortemperature = Column(Float)
    indoorco2concentration = Column(Float)
    supplydamper_airflowrate = Column(Float)
    supplydamper_damperposition = Column(Float)
    exhaustdamper_airflowrate = Column(Float)
    exhaustdamper_damperposition = Column(Float)
    spaceheater_outletwatertemperature = Column(String)
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
    ventilationsystem_supplyairtemperatureschedule_schedulevaluet = Column(Float)
    input_start_datetime = Column(DateTime(timezone=True))
    input_end_datetime = Column(DateTime(timezone=True))
    user_name= Column(String)
    user_id = Column(Float)
    scenario_name = Column(String)
    scenario_id = Column(Float)


class ventilation_simulation_results(Base):
    __tablename__ = 'ml_ventilation_simulation_results'

    id = Column(Integer, primary_key=True, autoincrement=True)
    room_name = Column(String, nullable=False)
    ventilation_system_name = Column(String)
    simulation_time = Column(DateTime(timezone=True))
    total_air_flow_rate = Column(Float)
    damper_position = Column(Float)
    air_flow_rate = Column(Float)

class ventilation_dummy_inputs(Base):
    __tablename__ = 'ml_ventilation_dummy_inputs'

    id = Column(Integer, primary_key=True, autoincrement=True)
    room_name = Column(String, nullable=False)
    ventilation_system_name = Column(String)
    simulation_time = Column(DateTime(timezone=True))
    co2concentration = Column(Float)
    temperature = Column(Float)
    air_damper_position = Column(Float)
    radiator_valve_position = Column(Float)
    
    

