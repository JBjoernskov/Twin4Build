import pandas as pd
import os
import sys
import unittest
if __name__ == '__main__':
    uppath = lambda _path,n: os.sep.join(_path.split(os.sep)[:-n])
    file_path = uppath(os.path.abspath(__file__), 5)
    sys.path.append(file_path)
from twin4build.utils.preprocessing.data_collection import DataCollection
from twin4build.utils.data_loaders.load_spreadsheet import load_spreadsheet
import pandas as pd
from twin4build.utils.uppath import uppath
import datetime
from dateutil.tz import tzutc
import matplotlib.pyplot as plt



@unittest.skipIf(False, 'Currently not used')
def test_data_collection():
    filename = os.path.join(os.path.abspath(uppath(os.path.abspath(__file__), 1)), "test", "test_data.csv")
    stepSize = 60
    startTime = datetime.datetime(year=2021, month=1, day=1, hour=0, minute=0, second=0, tzinfo=tzutc())
    endTime = datetime.datetime(year=2023, month=3, day=31, hour=0, minute=0, second=0, tzinfo=tzutc())
    df_sample = load_spreadsheet(filename=filename, stepSize=stepSize, start_time=startTime, end_time=endTime)
    data_collection = DataCollection(df_sample)
    # df_clean = pd.DataFrame(data_collection.clean_data_dict)
    # df_clean.iloc[:,0:4] = (df_clean.iloc[:,0:4]-32)*5/9
    # df_clean.insert(0,"time",data_collection.time)
    # df_clean = df_clean.set_index("time")
    # df_clean.plot(subplots=True)
    # plt.show()