import pandas as pd
import os
import sys
import pickle
from twin4build.utils.preprocessing.data_preparation import sample_data
import pandas as pd
from twin4build.utils.uppath import uppath
import datetime
import numpy as np
from dateutil.tz import tzutc
import os
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
from dateutil import parser

from twin4build.logger.Logging import Logging

logger = Logging.get_logger("ai_logfile")


def load_from_file(filename, stepSize=None, start_time=None, end_time=None, format=None, dt_limit=None):
    filehandler = open(filename, 'rb')
    name, file_extension = os.path.splitext(filename)

    #Check if file is cached
    startPeriod_str = start_time.strftime('%d-%m-%Y')
    endPeriod_str = end_time.strftime('%d-%m-%Y')
    cached_filename = f"name({os.path.basename(name)})_stepSize({str(stepSize)})_startPeriod({startPeriod_str})_endPeriod({endPeriod_str})_cached.pickle"
    cached_filename = os.path.join(os.path.abspath(uppath(os.path.abspath(__file__), 3)), "test", "data", "time_series_data", "cached_data", cached_filename)
    if os.path.isfile(cached_filename):
        df_sample = pd.read_pickle(cached_filename)
    else:

        if file_extension==".csv":
            df = pd.read_csv(filehandler, low_memory=False)

        elif file_extension==".xlsx":
            df = pd.read_excel(filehandler)

        else:
            logger.error((f"Invalid file extension: {file_extension}"))
            raise Exception(f"Invalid file extension: {file_extension}")

        for column in df.columns.to_list()[1:]:
            df[column] = pd.to_numeric(df[column], errors='coerce') #Remove string entries

        n = df.shape[0]
        data = np.zeros((n,2))
        time = np.vectorize(lambda data:datetime.datetime.strptime(data, format)) (df.iloc[:, 0])
        epoch_timestamp = np.vectorize(lambda data:datetime.datetime.timestamp(data)) (time)
        data[:,0] = epoch_timestamp
        df_sample = pd.DataFrame()
        for j, column in enumerate(df.columns):
            if j>0:
                data[:,1] = df[column].to_numpy()
                if np.isnan(data[:,1]).all():
                    print(f"Dropping column: {column}")
                else:
                    constructed_time_list,constructed_value_list,got_data = sample_data(data=data, stepSize=stepSize, start_time=start_time, end_time=end_time, dt_limit=dt_limit)
                    if got_data==True:
                        df_sample[column] = constructed_value_list
                    else:
                        print(f"Dropping column: {column}")
        df_sample.insert(0, df.columns.values[0], constructed_time_list)
        df_sample.to_pickle(cached_filename)
    return df_sample
