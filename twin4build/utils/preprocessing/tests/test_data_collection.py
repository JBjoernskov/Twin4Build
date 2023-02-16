import pandas as pd
import os
import sys
import pickle
if __name__ == '__main__':
    uppath = lambda _path,n: os.sep.join(_path.split(os.sep)[:-n])
    file_path = uppath(os.path.abspath(__file__), 4)
    sys.path.append(file_path)
from twin4build.utils.preprocessing.data_collection import DataCollection
from twin4build.utils.data_loaders.load_from_file import load_from_file
import pandas as pd
from twin4build.utils.uppath import uppath
import datetime
import numpy as np
from dateutil.tz import tzutc
import matplotlib.pyplot as plt





filename = os.path.join(os.path.abspath(uppath(os.path.abspath(__file__), 3)), "test", "data", "time_series_data", "VE02.csv")
stepSize = 60
startPeriod = datetime.datetime(year=2021, month=1, day=1, hour=0, minute=0, second=0, tzinfo=tzutc())
endPeriod = datetime.datetime(year=2023, month=3, day=31, hour=0, minute=0, second=0, tzinfo=tzutc())
df_sample = load_from_file(filename=filename, stepSize=stepSize, start_time=startPeriod, end_time=endPeriod)

data_collection = DataCollection(df_sample)

print(data_collection.raw_data_dict)




df_clean = pd.DataFrame(data_collection.clean_data_dict)
df_clean.iloc[:,0:4] = (df_clean.iloc[:,0:4]-32)*5/9
df_clean.insert(0,"time",data_collection.time)
df_clean = df_clean.set_index("time")
print(df_clean)
df_clean.plot(subplots=True)
plt.show()