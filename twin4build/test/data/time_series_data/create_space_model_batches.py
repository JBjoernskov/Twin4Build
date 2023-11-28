import pandas as pd
import os
import sys
import pickle
if __name__ == '__main__':
    uppath = lambda _path,n: os.sep.join(_path.split(os.sep)[:-n])
    file_path = uppath(os.path.abspath(__file__), 5)
    sys.path.append(file_path)
from twin4build.utils.preprocessing.data_collection import DataCollection
from twin4build.utils.data_loaders.load_spreadsheet import load_spreadsheet
import pandas as pd
from twin4build.utils.uppath import uppath
import datetime
import numpy as np
from dateutil.tz import tzutc
import matplotlib.pyplot as plt
from twin4build.utils.data_loaders.load_spreadsheet import load_spreadsheet
from twin4build.utils.preprocessing.data_sampler import data_sampler
import json

def main():
    df_input = pd.DataFrame()

    
    stepSize = 600
    startPeriod = datetime.datetime(year=2021, month=1, day=1, hour=0, minute=0, second=0, tzinfo=tzutc())
    endPeriod = datetime.datetime(year=2023, month=2, day=15, hour=0, minute=0, second=0, tzinfo=tzutc())
    format = "%m/%d/%Y %I:%M:%S %p"

    filename = os.path.join(os.path.abspath(uppath(os.path.abspath(__file__), 4)), "test", "data", "time_series_data", "weather_DMI_shifted.csv")
    df_weather_DMI = load_spreadsheet(filename=filename, stepSize=stepSize, start_time=startPeriod, end_time=endPeriod, format=format, dt_limit=1200)

    filename = os.path.join(os.path.abspath(uppath(os.path.abspath(__file__), 4)), "test", "data", "time_series_data", "weather_BMS.csv")
    df_weather_BMS = load_spreadsheet(filename=filename, stepSize=stepSize, start_time=startPeriod, end_time=endPeriod, format=format, dt_limit=1200)
    # df_weather_BMS["outdoorTemperature"] = (df_weather_BMS["outdoorTemperature"]-32)*5/9 #convert from fahrenheit to celcius

    filename = os.path.join(os.path.abspath(uppath(os.path.abspath(__file__), 4)), "test", "data", "time_series_data", "VA01.csv")
    VA01 = load_spreadsheet(filename=filename, stepSize=stepSize, start_time=startPeriod, end_time=endPeriod, format=format, dt_limit=999999)
    VA01["FTF1"] = (VA01["FTF1"]-32)*5/9 #convert from fahrenheit to celcius
    VA01["FTF1_SV"] = (VA01["FTF1_SV"]-32)*5/9 #convert from fahrenheit to celcius

    filename = os.path.join(os.path.abspath(uppath(os.path.abspath(__file__), 4)), "test", "data", "time_series_data", "VE02_FTI1.csv") ####
    VE02_FTI1 = load_spreadsheet(filename=filename, stepSize=stepSize, start_time=startPeriod, end_time=endPeriod, format=format, dt_limit=999999)

    filename = os.path.join(os.path.abspath(uppath(os.path.abspath(__file__), 4)), "test", "data", "time_series_data", "OE20-601b-2.csv")
    space_data = load_spreadsheet(filename=filename, stepSize=stepSize, start_time=startPeriod, end_time=endPeriod, format=format, dt_limit=1200)

    filename = os.path.join(os.path.abspath(uppath(os.path.abspath(__file__), 4)), "test", "data", "time_series_data", "OE20-601b-1_Indoor air temperature (Celcius).csv")
    space_data1 = load_spreadsheet(filename=filename, stepSize=stepSize, start_time=startPeriod, end_time=endPeriod, format=format, dt_limit=1200)

    filename = os.path.join(os.path.abspath(uppath(os.path.abspath(__file__), 4)), "test", "data", "time_series_data", "OE20-603-1_Indoor air temperature (Celcius).csv")
    space_data2 = load_spreadsheet(filename=filename, stepSize=stepSize, start_time=startPeriod, end_time=endPeriod, format=format, dt_limit=1200)

    filename = os.path.join(os.path.abspath(uppath(os.path.abspath(__file__), 4)), "test", "data", "time_series_data", "OE20-603c-2_Indoor air temperature (Celcius).csv")
    space_data3 = load_spreadsheet(filename=filename, stepSize=stepSize, start_time=startPeriod, end_time=endPeriod, format=format, dt_limit=1200)


    response_filename = os.path.join(uppath(os.path.abspath(__file__), 4), "test", "data", "time_series_data", "OE20-601b-2_kafka_temperature.txt")
    data = [json.loads(line) for line in open(response_filename, 'rb')]
    data = data[1:] #remove header information
    data = np.array([row[0][0] for row in data])
    data = data[data[:, 0].argsort()]
    constructed_time_list,constructed_value_list,got_data = data_sampler(data=data, stepSize=stepSize, start_time=startPeriod, end_time=endPeriod, dt_limit=1200)


    
    # bool_isnan = np.isnan(constructed_value_list)
    # constructed_value_list[bool_isnan] = space_data["Indoor air temperature (Celcius)"].iloc[bool_isnan]
    df_input.insert(0, "Time", space_data["Time stamp"])
    # # df_input.insert(1, "indoorTemperature", constructed_value_list)
    # df_input.insert(1, "indoorTemperature", space_data["Indoor air temperature (Celcius)"])
    # df_input.insert(2, "spaceHeaterAddedEnergy", space_data["Space heater valve position (0-100%)"]*VA01["FTF1"])
    # df_input.insert(3, "ventilationAddedEnergy", space_data["Damper valve position (0-100%)"]*VE02_FTI1["FTI1"])
    # df_input.insert(4, "ventilationRemovedEnergy", space_data["Damper valve position (0-100%)"]*space_data["Indoor air temperature (Celcius)"])
    # df_input.insert(5, "globalIrradiation", df_weather_DMI["globalIrradiation"])
    # df_input.insert(6, "outdoorTemperature", df_weather_BMS["outdoorTemperature"])
    # df_input.insert(7, "adjacentIndoorTemperature_OE20-601b-1", space_data1["Indoor air temperature (Celcius)"])
    # df_input.insert(8, "adjacentIndoorTemperature_OE20-603-1", space_data2["Indoor air temperature (Celcius)"])
    # df_input.insert(9, "adjacentIndoorTemperature_OE20-603c-2", space_data3["Indoor air temperature (Celcius)"])

    df_input.insert(1, "indoorTemperature", space_data["Indoor air temperature (Celcius)"])
    df_input.insert(2, "radiatorValvePosition", space_data["Space heater valve position (0-100%)"]/100)
    df_input.insert(3, "supplyWaterTemperature", VA01["FTF1"])
    df_input.insert(4, "damperPosition", space_data["Damper valve position (0-100%)"]/100)
    df_input.insert(5, "supplyAirTemperature", VE02_FTI1["FTI1"])
    df_input.insert(6, "outdoorTemperature", df_weather_BMS["outdoorTemperature"])
    df_input.insert(7, "globalIrradiation", df_weather_DMI["globalIrradiation"])
    df_input.insert(8, "adjacentIndoorTemperature_OE20-601b-1", space_data1["Indoor air temperature (Celcius)"])
    df_input.insert(9, "adjacentIndoorTemperature_OE20-603-1", space_data2["Indoor air temperature (Celcius)"])
    df_input.insert(10, "adjacentIndoorTemperature_OE20-603c-2", space_data3["Indoor air temperature (Celcius)"])


    time_of_day = (df_input["Time"].dt.hour*60+df_input["Time"].dt.minute)/(23*60+50)
    time_of_year = ((df_input["Time"].dt.dayofyear-1)*24*60+df_input["Time"].dt.hour*60+df_input["Time"].dt.minute)/(364*24*60 + 23*60 + 50)

    df_input["time_of_day_cos"] = np.cos(2*np.pi*time_of_day)
    df_input["time_of_day_sin"] = np.sin(2*np.pi*time_of_day)
    df_input["time_of_year_cos"] = np.cos(2*np.pi*time_of_year)
    df_input["time_of_year_sin"] = np.sin(2*np.pi*time_of_year)

    print(df_input)

    df_input.set_index("Time").plot(subplots=True)

    ## Smoothing
    fig, ax = plt.subplots()
    figdt, axdt = plt.subplots()
    axdt.plot(df_input["Time"], df_input["indoorTemperature"].diff(), color="red")
    ax.plot(df_input["Time"], df_input["indoorTemperature"], color="red")
    window = 5
    df_input["indoorTemperature"] = df_input["indoorTemperature"].rolling(window, min_periods=1).mean().shift(-(window-1))
    axdt.plot(df_input["Time"], df_input["indoorTemperature"].diff(), color="black")
    ax.plot(df_input["Time"], df_input["indoorTemperature"], color="black")
    plt.show()


    # df_input.plot(subplots=True)


    remove_start_date_list = ["2022-02-02 18:00:00+00:00", "2022-04-03 00:00:00+00:00", "2022-06-17 00:00:00+00:00"]
    remove_end_date_list = ["2022-02-04 07:00:00+00:00", "2022-04-03 13:00:00+00:00", "2022-08-01 00:00:00+00:00"]

    #Filter to only consider winter data
    df_input[(df_input["Time"].dt.month < 11) & (df_input["Time"].dt.month > 3)] = np.nan

    for remove_start_date,remove_end_date in zip(remove_start_date_list, remove_end_date_list):
        df_input[(df_input["Time"]>=remove_start_date) & (df_input["Time"]<=remove_end_date)] = np.nan
    
    
    df_input["Time"] = space_data["Time stamp"]


    name = "Ø20-601b-2"
    data_collection = DataCollection(name, df_input, nan_interpolation_gap_limit=36, n_sequence=144)
    data_collection.prepare_for_data_batches()

    print(data_collection.sequence_distribution_list)
    print(data_collection.sequence_distribution_by_season_vec)


    df_input.set_index("Time", inplace=True)
    # df_input.plot(subplots=True)

    # n=0
    # for col in data_collection.data_matrix.transpose():
    #     plt.figure()
    #     plt.plot(data_collection.time,col)
    #     plt.title("matrix" + str(n))
    #     n += 1
    # plt.show()

    save_folder = os.path.join(os.path.abspath(uppath(os.path.abspath(__file__), 1)), "space_model_dataset")
    data_collection.create_data_batches(save_folder=save_folder)

    save_folder = os.path.join(os.path.abspath(uppath(os.path.abspath(__file__), 2)), "space_models", "BMS_data")
    data_collection.save_building_data_collection_dict(save_folder=save_folder)


if __name__ == '__main__':
    main()
