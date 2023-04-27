import os
import sys
import json
import pickle
import datetime
import numpy as np
import pandas as pd
from dateutil.tz import tzutc
import matplotlib.pyplot as plt

#This is a temp Fix
file_path = "D:\Projects\Twin4Build"
sys.path.append(file_path)

#from twin4build.utils.uppath import uppath
from twin4build.ml_pipelines.ml_pipe_data_collection import DataCollection
from twin4build.ml_pipelines.ml_pipe_data_preparation import sample_data
from twin4build.utils.data_loaders.load_from_file import load_from_file

def insert_data():
    df_input = pd.DataFrame()

    stepSize = 600
    startPeriod = datetime.datetime(year=2021, month=1, day=1, hour=0, minute=0, second=0, tzinfo=tzutc())
    endPeriod = datetime.datetime(year=2023, month=2, day=15, hour=0, minute=0, second=0, tzinfo=tzutc())
    format = "%m/%d/%Y %I:%M:%S %p"

    file_path = "D:\\Projects\\Twin4Build\\twin4build"
    filename = os.path.join(file_path, "test", "data", "time_series_data", "weather_DMI.csv")
    df_weather_DMI = load_from_file(filename=filename, stepSize=stepSize, start_time=startPeriod, end_time=endPeriod, format=format, dt_limit=1200)

    filename = os.path.join(file_path, "test", "data", "time_series_data", "weather_BMS.csv")
    df_weather_BMS = load_from_file(filename=filename, stepSize=stepSize, start_time=startPeriod, end_time=endPeriod, format=format, dt_limit=1200)
    #df_weather_BMS["outdoorTemperature"] = (df_weather_BMS["outdoorTemperature"]-32)*5/9 #convert from fahrenheit to celcius

    filename = os.path.join(file_path, "test", "data", "time_series_data", "VA01.csv")
    VA01 = load_from_file(filename=filename, stepSize=stepSize, start_time=startPeriod, end_time=endPeriod, format=format, dt_limit=999999)
    VA01["FTF1"] = (VA01["FTF1"]-32)*5/9 #convert from fahrenheit to celcius
    VA01["FTF1_SV"] = (VA01["FTF1_SV"]-32)*5/9 #convert from fahrenheit to celcius

    filename = os.path.join(file_path, "test", "data", "time_series_data", "VE02_FTI1.csv") ####
    VE02_FTI1 = load_from_file(filename=filename, stepSize=stepSize, start_time=startPeriod, end_time=endPeriod, format=format, dt_limit=999999)

    filename = os.path.join(file_path, "test", "data", "time_series_data", "OE20-601b-2.csv")
    space_data = load_from_file(filename=filename, stepSize=stepSize, start_time=startPeriod, end_time=endPeriod, format=format, dt_limit=1200)

    filename = os.path.join(file_path, "test", "data", "time_series_data", "OE20-601b-1_Indoor air temperature (Celcius).csv")
    space_data1 = load_from_file(filename=filename, stepSize=stepSize, start_time=startPeriod, end_time=endPeriod, format=format, dt_limit=1200)

    filename = os.path.join(file_path, "test", "data", "time_series_data", "OE20-603-1_Indoor air temperature (Celcius).csv")
    space_data2 = load_from_file(filename=filename, stepSize=stepSize, start_time=startPeriod, end_time=endPeriod, format=format, dt_limit=1200)

    filename = os.path.join(file_path, "test", "data", "time_series_data", "OE20-603c-2_Indoor air temperature (Celcius).csv")
    space_data3 = load_from_file(filename=filename, stepSize=stepSize, start_time=startPeriod, end_time=endPeriod, format=format, dt_limit=1200)

    response_filename = os.path.join(file_path, "test", "data", "time_series_data", "OE20-601b-2_kafka_temperature.txt")
    data = [json.loads(line) for line in open(response_filename, 'rb')]
    data = data[1:] #remove header information
    data = np.array([row[0][0] for row in data])
    data = data[data[:, 0].argsort()]
    constructed_time_list,constructed_value_list,got_data = sample_data(data=data, stepSize=stepSize, start_time=startPeriod, end_time=endPeriod, dt_limit=1200)

    df_input.insert(0, "Time", space_data["Time stamp"])
    # df_input.insert(1, "indoorTemperature", constructed_value_list)
    df_input.insert(1, "indoorTemperature", space_data["Indoor air temperature (Celcius)"])
    df_input.insert(2, "spaceHeaterAddedEnergy", space_data["Space heater valve position (0-100%)"]*VA01["FTF1"])
    df_input.insert(3, "ventilationAddedEnergy", space_data["Damper valve position (0-100%)"]*VE02_FTI1["FTI1"])
    df_input.insert(4, "ventilationRemovedEnergy", space_data["Damper valve position (0-100%)"]*space_data["Indoor air temperature (Celcius)"])
    df_input.insert(5, "globalIrradiation", df_weather_DMI["globalIrradiation"])
    df_input.insert(6, "outdoorTemperature", df_weather_BMS["outdoorTemperature"])
    df_input.insert(7, "adjacentIndoorTemperature_OE20-601b-1", space_data1["Indoor air temperature (Celcius)"])
    df_input.insert(8, "adjacentIndoorTemperature_OE20-603-1", space_data2["Indoor air temperature (Celcius)"])
    df_input.insert(9, "adjacentIndoorTemperature_OE20-603c-2", space_data3["Indoor air temperature (Celcius)"])

    time_of_day = (df_input["Time"].dt.hour*60+df_input["Time"].dt.minute)/(23*60+50)
    time_of_year = ((df_input["Time"].dt.dayofyear-1)*24*60+df_input["Time"].dt.hour*60+df_input["Time"].dt.minute)/(364*24*60 + 23*60 + 50)

    df_input["time_of_day_cos"] = np.cos(2*np.pi*time_of_day)
    df_input["time_of_day_sin"] = np.sin(2*np.pi*time_of_day)
    df_input["time_of_year_cos"] = np.cos(2*np.pi*time_of_year)
    df_input["time_of_year_sin"] = np.sin(2*np.pi*time_of_year)

    #df_input.plot(subplots=True)

    remove_start_date_list = ["2022-02-02 18:00:00+00:00", "2022-04-03 00:00:00+00:00"]
    remove_end_date_list = ["2022-02-04 07:00:00+00:00", "2022-04-03 13:00:00+00:00"]

    #Filter to only consider winter data
    df_input[(df_input["Time"].dt.month < 10) & (df_input["Time"].dt.month > 4)] = np.nan

    for remove_start_date,remove_end_date in zip(remove_start_date_list, remove_end_date_list):
        df_input[(df_input["Time"]>=remove_start_date) & (df_input["Time"]<=remove_end_date)] = np.nan
    
    
    df_input["Time"] = space_data["Time stamp"]
    return (df_input)


