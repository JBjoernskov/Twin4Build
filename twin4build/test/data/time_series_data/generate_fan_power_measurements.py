import os
import sys
import datetime
from dateutil.tz import tzutc
import pandas as pd
import matplotlib.pyplot as plt
import json
import numpy as np
import seaborn as sns
###Only for testing before distributing package
if __name__ == '__main__':
    uppath = lambda _path,n: os.sep.join(_path.split(os.sep)[:-n])
    file_path = uppath(os.path.abspath(__file__), 5)
    sys.path.append(file_path)

from twin4build.utils.data_loaders.load_from_file import load_from_file

def test():

    # waterFlowRateMax = abs(space_heater.outputCapacity.hasValue/Constants.specificHeatCapacity["water"]/(space_heater.nominalSupplyTemperature-space_heater.nominalReturnTemperature))
    VE02_power_VI = pd.DataFrame()
    VE02_power_VU = pd.DataFrame()

    startPeriod = datetime.datetime(year=2022, month=1, day=1, hour=0, minute=0, second=0) 
    endPeriod = datetime.datetime(year=2023, month=1, day=1, hour=0, minute=0, second=0)

    # startPeriod = datetime.datetime(year=2022, month=2, day=1, hour=10, minute=0, second=0) 
    # endPeriod = datetime.datetime(year=2022, month=2, day=1, hour=16, minute=0, second=0)
    format = "%m/%d/%Y %I:%M:%S %p"


    

    stepSize = 60

    filename = os.path.join(os.path.abspath(uppath(os.path.abspath(__file__), 4)), "test", "data", "time_series_data", "VE02_airflowrate_supply_kg_s.csv")
    VE02_supply_air = load_from_file(filename=filename, stepSize=stepSize, start_time=startPeriod, end_time=endPeriod, format=format, dt_limit=9999)

    filename = os.path.join(os.path.abspath(uppath(os.path.abspath(__file__), 4)), "test", "data", "time_series_data", "VE02_airflowrate_return_kg_s.csv")
    VE02_return_air = load_from_file(filename=filename, stepSize=stepSize, start_time=startPeriod, end_time=endPeriod, format=format, dt_limit=9999)

    filename = os.path.join(os.path.abspath(uppath(os.path.abspath(__file__), 4)), "test", "data", "time_series_data", "VE02_SEL_VI.csv")
    VE02_SEL_VI = load_from_file(filename=filename, stepSize=stepSize, start_time=startPeriod, end_time=endPeriod, format=format, dt_limit=9999)

    filename = os.path.join(os.path.abspath(uppath(os.path.abspath(__file__), 4)), "test", "data", "time_series_data", "VE02_SEL_VU.csv")
    VE02_SEL_VU = load_from_file(filename=filename, stepSize=stepSize, start_time=startPeriod, end_time=endPeriod, format=format, dt_limit=9999)

    VE02_power_VI.insert(0, "time", VE02_supply_air["Time stamp"])
    VE02_power_VI.insert(0, "VE02_power_VI", VE02_supply_air["primaryAirFlowRate"]*VE02_SEL_VI["VE02_SEL_VI"]/1.225)
    filename = os.path.join(os.path.abspath(uppath(os.path.abspath(__file__), 4)), "test", "data", "time_series_data", "VE02_power_VI.csv")
    VE02_power_VI.set_index("time").to_csv(filename, index=True)

    VE02_power_VU.insert(0, "time", VE02_return_air["Time stamp"])
    VE02_power_VU.insert(0, "VE02_power_VU", VE02_return_air["secondaryAirFlowRate"]*VE02_SEL_VU["VE02_SEL_VU"]/1.225)
    filename = os.path.join(os.path.abspath(uppath(os.path.abspath(__file__), 4)), "test", "data", "time_series_data", "VE02_power_VU.csv")
    VE02_power_VU.set_index("time").to_csv(filename, index=True)


    plt.scatter(VE02_supply_air["primaryAirFlowRate"], VE02_power_VI["VE02_power_VI"])
    plt.show()

if __name__ == '__main__':
    test()



