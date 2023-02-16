import os
import sys
if __name__ == '__main__':
    uppath = lambda _path,n: os.sep.join(_path.split(os.sep)[:-n])
    file_path = uppath(os.path.abspath(__file__), 4)
    sys.path.append(file_path)
    print(file_path)
from twin4build.utils.data_loaders.load_from_file import load_from_file
from twin4build.utils.uppath import uppath
import datetime
from dateutil.tz import tzutc
import matplotlib.pyplot as plt

filename = os.path.join(os.path.abspath(uppath(os.path.abspath(__file__), 4)), "test", "data", "time_series_data", "OE20-601b-2.csv")
stepSize = 600
startPeriod = datetime.datetime(year=2021, month=1, day=1, hour=0, minute=0, second=0, tzinfo=tzutc())
endPeriod = datetime.datetime(year=2023, month=3, day=31, hour=0, minute=0, second=0, tzinfo=tzutc())
df_sample = load_from_file(filename=filename,stepSize=stepSize, start_time=startPeriod, end_time=endPeriod)
df_sample.plot(subplots=True)
plt.show()