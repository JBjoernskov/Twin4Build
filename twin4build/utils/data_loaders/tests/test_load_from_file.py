import os
import unittest
import sys
import pandas as pd
if __name__ == '__main__':
    uppath = lambda _path,n: os.sep.join(_path.split(os.sep)[:-n])
    file_path = uppath(os.path.abspath(__file__), 5)
    sys.path.append(file_path)
    print(file_path)
from twin4build.utils.data_loaders.load_spreadsheet import load_spreadsheet
from twin4build.utils.uppath import uppath
import datetime
from dateutil.tz import tzutc

@unittest.skipIf(False, 'Currently not used')
def test_load_spreadsheet():
    stepSize = 60
    startPeriod = datetime.datetime(year=2021, month=1, day=1, hour=0, minute=0, second=0, tzinfo=tzutc())
    endPeriod = datetime.datetime(year=2023, month=3, day=31, hour=0, minute=0, second=0, tzinfo=tzutc())
    # filename = os.path.join(os.path.abspath(uppath(os.path.abspath(__file__), 1)), "test_weather_data.csv")
    # filename = r"C:\Users\jabj\OneDrive - Syddansk Universitet\PhD_Project_Jakob\Twin4build\python\BuildingEnergyModel\Twin4build-Case-Studies\DP37\data\Rooms\OD095_01_007A\OD095_01_007A_L95_LC02_BQA008_S1.plc_SENSOR_VALUE.csv"
    
    filename = os.path.join(os.path.abspath(uppath(os.path.abspath(__file__), 4)), "test", "data", "time_series_data", "VE02_SEL_VI.csv")
    df_sel = load_spreadsheet(filename=filename, start_time=startPeriod, end_time=endPeriod, stepSize=stepSize, cache=False, resample=True, clip=True)
    print(df_sel)

    filename = os.path.join(os.path.abspath(uppath(os.path.abspath(__file__), 4)), "test", "data", "time_series_data", "VE02_airflowrate_supply_kg_s.csv")
    df_mf = load_spreadsheet(filename=filename, start_time=startPeriod, end_time=endPeriod, stepSize=stepSize, cache=False, resample=True, clip=True)
    print(df_mf)

    df_power = pd.DataFrame()
    df_power.insert(0, "datetime", df_sel.index)
    df_power = df_power.set_index("datetime")
    df_power.insert(0, "supply_fan_power", df_sel.iloc[:,0]*df_mf.iloc[:,0]/1.225)
    print(df_power)

    filename = os.path.join(os.path.abspath(uppath(os.path.abspath(__file__), 4)), "test", "data", "time_series_data", "supply_fan_power.csv")
    df_power.to_csv(filename, index=True)

    import matplotlib.pyplot as plt
    df_power.plot()
    plt.show()

    print(df_power)


if __name__=="__main__":
    test_load_spreadsheet()