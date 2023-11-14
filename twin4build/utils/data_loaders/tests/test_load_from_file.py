import os
import sys
if __name__ == '__main__':
    uppath = lambda _path,n: os.sep.join(_path.split(os.sep)[:-n])
    file_path = uppath(os.path.abspath(__file__), 5)
    sys.path.append(file_path)
    print(file_path)
from twin4build.utils.data_loaders.load_from_file import load_from_file
from twin4build.utils.uppath import uppath
import datetime
from dateutil.tz import tzutc
import matplotlib.pyplot as plt
import pandas as pd

filename = os.path.join(os.path.abspath(uppath(os.path.abspath(__file__), 4)), "test", "data", "time_series_data", "weather_BMS.csv")
stepSize = 600
format = "%m/%d/%Y %I:%M:%S %p"
startPeriod = datetime.datetime(year=2021, month=1, day=1, hour=0, minute=0, second=0, tzinfo=tzutc())
endPeriod = datetime.datetime(year=2023, month=3, day=31, hour=0, minute=0, second=0, tzinfo=tzutc())
df_BMS = load_from_file(filename=filename,stepSize=stepSize, start_time=startPeriod, end_time=endPeriod, format=format, dt_limit=999999)
# df_BMS.plot(subplots=True)

filename = os.path.join(os.path.abspath(uppath(os.path.abspath(__file__), 4)), "test", "data", "time_series_data", "weather_DMI.csv")
df_DMI = load_from_file(filename=filename,stepSize=stepSize, start_time=startPeriod, end_time=endPeriod, format=format, dt_limit=999999)
# df_DMI.set_index("Time stamp", inplace=True)
# df_DMI = df_DMI.shift(periods=2, freq="H")


# filename = os.path.join(os.path.abspath(uppath(os.path.abspath(__file__), 4)), "test", "data", "time_series_data", "weather_DMI_shifted.csv")
# df_DMI.to_csv(filename, date_format=format, index=True)



df = pd.DataFrame()
df.insert(0, "time", df_BMS["Time stamp"])
df.insert(0, "BMS", df_BMS["outdoorTemperature"])
df.insert(0, "DMI", df_DMI["outdoorTemperature"])
df.set_index("time").plot()
plt.show()