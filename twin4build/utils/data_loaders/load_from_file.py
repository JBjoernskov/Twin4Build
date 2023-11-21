import pandas as pd
import os
from twin4build.utils.preprocessing.data_sampler import data_sampler
import pandas as pd
from twin4build.utils.uppath import uppath
import datetime
import numpy as np
import os
from dateutil import parser
from twin4build.logger.Logging import Logging
from dateutil.parser import parse
from twin4build.utils.create_dir_in_main import create_dir_in_main
logger = Logging.get_logger("ai_logfile")


def load_from_file(filename, stepSize=None, start_time=None, end_time=None, date_format=None, dt_limit=None):
    name, file_extension = os.path.splitext(filename)

    #Check if file is cached
    startPeriod_str = start_time.strftime('%d-%m-%Y %H-%M-%S')
    endPeriod_str = end_time.strftime('%d-%m-%Y %H-%M-%S')
    cached_filename = f"name({os.path.basename(name)})_stepSize({str(stepSize)})_startPeriod({startPeriod_str})_endPeriod({endPeriod_str})_cached.pickle"
    cached_filename = create_dir_in_main(folder_list=["generated_files", "cached_data"], filename=cached_filename)
    if os.path.isfile(cached_filename):
        df_sample = pd.read_pickle(cached_filename)
    else:
        with open(filename, 'rb') as filehandler:
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
        if date_format is None:
            time = np.vectorize(lambda data:parse(data)) (df.iloc[:, 0])
        else:
            time = np.vectorize(lambda data:datetime.datetime.strptime(data, date_format)) (df.iloc[:, 0])
        epoch_timestamp = np.vectorize(lambda data:datetime.datetime.timestamp(data)) (time)

        data[:,0] = epoch_timestamp
        df_sample = pd.DataFrame()
        for column in df.columns.to_list()[1:]:
            data[:,1] = df[column].to_numpy()
            if np.isnan(data[:,1]).all():
                print(f"Bad data quality. All of data contains NaN values in file: \n\"{filename}\"")
                print(f"Dropping column: {column}")
            else:
                constructed_time_list,constructed_value_list,got_data = data_sampler(data=data, stepSize=stepSize, start_time=start_time, end_time=end_time, dt_limit=dt_limit)
                if got_data==True:
                    df_sample[column] = constructed_value_list[:,0]
                else:
                    print(f"Bad data quality. Most of data contains NaN values in file: \n\"{filename}\"")
                    print(f"Dropping column: {column}")
        df_sample.insert(0, df.columns.values[0], constructed_time_list)
        df_sample.to_pickle(cached_filename)
    return df_sample
