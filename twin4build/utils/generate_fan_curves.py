import os
import sys
import datetime
from dateutil.tz import tzutc
import pandas as pd
import matplotlib.pyplot as plt
import json
import numpy as np
import seaborn as sns
import numpy as np
###Only for testing before distributing package
if __name__ == '__main__':
    uppath = lambda _path,n: os.sep.join(_path.split(os.sep)[:-n])
    file_path = uppath(os.path.abspath(__file__), 3)
    sys.path.append(file_path)
print(file_path)
from twin4build.utils.data_loaders.load_spreadsheet import load_spreadsheet

def test():

    # waterFlowRateMax = abs(space_heater.outputCapacity.hasValue/Constants.specificHeatCapacity["water"]/(space_heater.nominalSupplyTemperature-space_heater.nominalReturnTemperature))
    df = pd.DataFrame()

    startPeriod = datetime.datetime(year=2022, month=1, day=1, hour=0, minute=0, second=0) 
    endPeriod = datetime.datetime(year=2023, month=1, day=1, hour=0, minute=0, second=0)

    # startPeriod = datetime.datetime(year=2022, month=2, day=1, hour=10, minute=0, second=0) 
    # endPeriod = datetime.datetime(year=2022, month=2, day=1, hour=16, minute=0, second=0)
    format = "%m/%d/%Y %I:%M:%S %p"
    stepSize = 60

    filename = os.path.join(os.path.abspath(uppath(os.path.abspath(__file__), 3)), "twin4build", "test", "data", "time_series_data", "VE02_airflowrate_supply_kg_s.csv")
    VE02_supply_air = load_spreadsheet(filename=filename, stepSize=stepSize, start_time=startPeriod, end_time=endPeriod, format=format, dt_limit=9999)

    filename = os.path.join(os.path.abspath(uppath(os.path.abspath(__file__), 3)), "twin4build", "test", "data", "time_series_data", "VE02_VIS.csv")
    VE02_VIS = load_spreadsheet(filename=filename, stepSize=stepSize, start_time=startPeriod, end_time=endPeriod, format=format, dt_limit=9999)

    filename = os.path.join(os.path.abspath(uppath(os.path.abspath(__file__), 3)), "twin4build", "test", "data", "time_series_data", "VE02_power_VI.csv")
    VE02_power_VI = load_spreadsheet(filename=filename, stepSize=stepSize, start_time=startPeriod, end_time=endPeriod, format=format, dt_limit=9999)

    df.insert(0, "time", VE02_supply_air["Time stamp"])
    df.insert(1, "primaryAirFlowRate", VE02_supply_air["primaryAirFlowRate"])
    df.insert(2, "VE02_VIS", VE02_VIS["VE02_VIS"])
    df.insert(3, "VE02_power_VI", VE02_power_VI["VE02_power_VI"])

    tol = 10
    df = df[df["VE02_power_VI"]>tol]

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(df["primaryAirFlowRate"].values, df["VE02_VIS"].values, df["VE02_power_VI"].values)


    fig, ax = plt.subplots()
    n_bins = 20
    max_speed = 100 #Percent
    bins = np.linspace(start=0, stop=max_speed, num=n_bins+1)
    best = 0
    x0_lower_limit = 50
    for x0, x1 in zip(bins[:-1], bins[1:]):
        df_temp = df[(df["VE02_VIS"]>=x0)&(df["VE02_VIS"]<=x1)]
        if df_temp.shape[0]>best and x0>x0_lower_limit:
            df_best = df_temp
            x0_best = x0
            x1_best = x1
            best = df_temp.shape[0]
        print(df_temp.shape[0])
        text = f"{int(x0)}:{int(x1)}"
        ax.text(df_temp["primaryAirFlowRate"].mean(),df_temp["VE02_power_VI"].mean(),text)
        ax.scatter(df_temp["primaryAirFlowRate"], df_temp["VE02_power_VI"], s=1, label=text)
    print(x0_best)
    fig, ax = plt.subplots()
    ax.scatter(df_best["primaryAirFlowRate"], df_best["VE02_power_VI"], label=f"{int(x0)}:{int(x1)}")
    
    
    
    
    fig, ax = plt.subplots()    
    cm = sns.color_palette("icefire", as_cmap=True)
    sc = ax.scatter(df["primaryAirFlowRate"], df["VE02_power_VI"], c=df["VE02_VIS"], cmap=cm)
    ax.set_xlabel("Massflow [kg/s]", fontsize=12)
    ax.set_ylabel("Power [W]", fontsize=12)
    cbar = plt.colorbar(sc)
    cbar.set_label('Speed [%]', rotation=270, fontsize=12, labelpad=15)
    plt.show()
    

if __name__ == '__main__':
    test()



