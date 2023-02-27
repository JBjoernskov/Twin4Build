import pandas as pd
import os
import sys
import pickle
if __name__ == '__main__':
    uppath = lambda _path,n: os.sep.join(_path.split(os.sep)[:-n])
    file_path = uppath(os.path.abspath(__file__), 5)
    sys.path.append(file_path)
from twin4build.utils.preprocessing.data_collection import DataCollection
from twin4build.utils.data_loaders.load_from_file import load_from_file
import pandas as pd
from twin4build.utils.uppath import uppath
import datetime
import numpy as np
from dateutil.tz import tzutc
import matplotlib.pyplot as plt
from twin4build.utils.data_loaders.load_from_file import load_from_file
from twin4build.utils.preprocessing.data_preparation import sample_data
import json

def main():
    df_input = pd.DataFrame()

    
    stepSize = 600
    startPeriod = datetime.datetime(year=2021, month=1, day=1, hour=0, minute=0, second=0, tzinfo=tzutc())
    endPeriod = datetime.datetime(year=2023, month=2, day=15, hour=0, minute=0, second=0, tzinfo=tzutc())
    format = "%m/%d/%Y %I:%M:%S %p"

    filename = os.path.join(os.path.abspath(uppath(os.path.abspath(__file__), 4)), "test", "data", "time_series_data", "weather_DMI.csv")
    df_weather_DMI = load_from_file(filename=filename, stepSize=stepSize, start_time=startPeriod, end_time=endPeriod, format=format, dt_limit=1200)

    filename = os.path.join(os.path.abspath(uppath(os.path.abspath(__file__), 4)), "test", "data", "time_series_data", "weather_BMS.csv")
    df_weather_BMS = load_from_file(filename=filename, stepSize=stepSize, start_time=startPeriod, end_time=endPeriod, format=format, dt_limit=1200)
    df_weather_BMS["outdoorTemperature"] = (df_weather_BMS["outdoorTemperature"]-32)*5/9 #convert from fahrenheit to celcius

    filename = os.path.join(os.path.abspath(uppath(os.path.abspath(__file__), 4)), "test", "data", "time_series_data", "VA01.csv")
    VA01 = load_from_file(filename=filename, stepSize=stepSize, start_time=startPeriod, end_time=endPeriod, format=format, dt_limit=999999)
    VA01["FTF1"] = (VA01["FTF1"]-32)*5/9 #convert from fahrenheit to celcius

    filename = os.path.join(os.path.abspath(uppath(os.path.abspath(__file__), 4)), "test", "data", "time_series_data", "VE02_FTI1.csv")
    VE02_FTI1 = load_from_file(filename=filename, stepSize=stepSize, start_time=startPeriod, end_time=endPeriod, format=format, dt_limit=999999)
    VE02_FTI1["FTI1"] = (VE02_FTI1["FTI1"]-32)*5/9 #convert from fahrenheit to celcius

    filename = os.path.join(os.path.abspath(uppath(os.path.abspath(__file__), 4)), "test", "data", "time_series_data", "OE20-601b-2.csv")
    space_data = load_from_file(filename=filename, stepSize=stepSize, start_time=startPeriod, end_time=endPeriod, format=format, dt_limit=1200)


    response_filename = os.path.join(uppath(os.path.abspath(__file__), 4), "test", "data", "time_series_data", "OE20-601b-2_kafka_temperature.txt")
    data = [json.loads(line) for line in open(response_filename, 'rb')]
    data = data[1:] #remove header information
    data = np.array([row[0][0] for row in data])
    data = data[data[:, 0].argsort()]
    constructed_time_list,constructed_value_list,got_data = sample_data(data=data, stepSize=stepSize, start_time=startPeriod, end_time=endPeriod, dt_limit=1200)


    

    df_input.insert(0, "Time", space_data["Time stamp"])
    df_input.insert(1, "indoorTemperature", space_data["Indoor air temperature (Celcius)"])
    # df_input.insert(1, "indoorTemperature", constructed_value_list)
    df_input.insert(2, "CO2", space_data["CO2 (ppm)"])
    df_input.insert(3, "radiatorValvePosition", space_data["Space heater valve position (0-100%)"])
    df_input.insert(4, "supplyWaterTemperature", VA01["FTF1"])
    df_input.insert(5, "damperPosition", space_data["Damper valve position (0-100%)"])
    df_input.insert(6, "supplyAirTemperature", VE02_FTI1["FTI1"])
    df_input.insert(7, "outdoorTemperature", df_weather_BMS["outdoorTemperature"])
    df_input.insert(8, "globalIrradiation", df_weather_DMI["globalIrradiation"])



    


    #Filter to only consider winter data
    df_input[(df_input["Time"].dt.month < 10) & (df_input["Time"].dt.month > 3)] = np.nan
    df_input["Time"] = space_data["Time stamp"]


    name = "Ø20-601b-2"
    data_collection = DataCollection(name, df_input)
    data_collection.prepare_for_data_batches()

    df_input.set_index("Time", inplace=True)
    df_input.plot(subplots=True)


    df_test = pd.DataFrame(data_collection.clean_data_dict)
    df_test.insert(0, "time", data_collection.time)
    df_test.set_index("time", inplace=True)
    df_test.plot(subplots=True)


    # n=0
    # for col in data_collection.data_matrix.transpose():
    #     plt.figure()
    #     plt.plot(data_collection.time,col)
    #     plt.title("matrix" + str(n))
    #     n += 1
        


    plt.show()

    save_folder = os.path.join(os.path.abspath(uppath(os.path.abspath(__file__), 1)), "space_model_batches")
    data_collection.create_data_batches(save_folder=save_folder)

    save_folder = os.path.join(os.path.abspath(uppath(os.path.abspath(__file__), 2)), "space_models", "BMS_data")
    data_collection.save_building_data_collection_dict(save_folder=save_folder)


if __name__ == '__main__':
    main()
