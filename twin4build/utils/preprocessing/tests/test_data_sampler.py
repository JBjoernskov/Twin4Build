import pandas as pd
import os
import sys
import unittest
import datetime
import numpy as np
from dateutil.tz import tzutc
if __name__ == '__main__':
    uppath = lambda _path,n: os.sep.join(_path.split(os.sep)[:-n])
    file_path = uppath(os.path.abspath(__file__), 5)
    sys.path.append(file_path)
from twin4build.utils.preprocessing.data_sampler import data_sampler
from twin4build.utils.uppath import uppath


# filepath = os.path.join(os.path.abspath(uppath(os.path.abspath(__file__), 2)), "test", "data", "time_series_data", "VE02.xlsx")
# df_VE02 = pd.read_excel(filepath)
# filepath = os.path.join(os.path.abspath(uppath(os.path.abspath(__file__), 2)), "test", "data", "time_series_data", "OE20-601b-2.xlsx")
# df_OE20_601b_2 = pd.read_excel(filepath)
# filename = "VE02.pickle"
# filehandler = open(filename, 'wb')
# pickle.dump((df_VE02), filehandler)
# filename = "OE20_601b_2.pickle"
# filehandler = open(filename, 'wb')
# pickle.dump((df_OE20_601b_2), filehandler)

@unittest.skipIf(False, 'Currently not used')
def test_data_sampler():
    filename = os.path.join(os.path.abspath(uppath(os.path.abspath(__file__), 1)), "test", "test_data.csv")
    filehandler = open(filename, 'rb')
    df = pd.read_csv(filehandler, low_memory=False)
    format = "%m/%d/%Y %I:%M:%S %p"
    n = df.shape[0]
    data = np.zeros((n,2))
    time = np.vectorize(lambda data:datetime.datetime.strptime(data, format)) (df["Time stamp"])
    epoch_timestamp = np.vectorize(lambda data:datetime.datetime.timestamp(data)) (time)
    data[:,0] = epoch_timestamp
    data[:,1] = df["FTI1"].to_numpy()
    stepSize = 600
    startPeriod = datetime.datetime(year=2021, month=1, day=1, hour=0, minute=0, second=0, tzinfo=tzutc())
    endPeriod = datetime.datetime(year=2023, month=12, day=31, hour=0, minute=0, second=0, tzinfo=tzutc())
    constructed_time_list,constructed_value_list,got_data = data_sampler(data=data, stepSize=stepSize, start_time=startPeriod, end_time=endPeriod)
