import pandas as pd
from pandas.core.series import Series
import os
from twin4build.utils.preprocessing.data_sampler import data_sampler
import pandas as pd
from twin4build.utils.uppath import uppath
import numpy as np
import os
import pytz
from twin4build.logger.Logging import Logging
from dateutil.tz import gettz
from twin4build.utils.mkdir_in_root import mkdir_in_root
logger = Logging.get_logger("ai_logfile")
import time as t

def load_spreadsheet(filename, 
                     stepSize=None, 
                     start_time=None, 
                     end_time=None, 
                     date_format=None, 
                     dt_limit=None, 
                     resample=True, 
                     clip=True, 
                     cache=True, 
                     cache_root=None, 
                     tz="Europe/Copenhagen", 
                     preserve_order=True):
    """
    This function loads a spead either in .csv or .xlsx format.
    The datetime should in the first column - timezone-naive inputs are localized as "tz", while timezone-aware inputs are converted to "tz".
    All data except for datetime column is converted to numeric data.

    tz: can be "UTC+2", "GMT-8" (no trailing zeros) or timezone name "Europe/Copenhagen"

    preserve_order: If True, the order of rows in the spreadsheet are important in order to resolve DST when timezone information is not available

    PRINT THE FOLLOWING TO SEE AVAILABLE NAMES:
    from dateutil.zoneinfo import getzoneinfofile_stream, ZoneInfoFile
    print(ZoneInfoFile(getzoneinfofile_stream()).zones.keys())
    """
    name, file_extension = os.path.splitext(filename)

    if cache:
        #Check if file is cached
        startPeriod_str = start_time.strftime('%d-%m-%Y %H-%M-%S')
        endPeriod_str = end_time.strftime('%d-%m-%Y %H-%M-%S')
        cached_filename = f"name({os.path.basename(name)})_stepSize({str(stepSize)})_startPeriod({startPeriod_str})_endPeriod({endPeriod_str})_cached.pickle"
        cached_filename = mkdir_in_root(folder_list=["generated_files", "cached_data"], filename=cached_filename, root=cache_root)
    if cache and os.path.isfile(cached_filename):
        df = pd.read_pickle(cached_filename)
    else:
        with open(filename, 'rb') as filehandler:
            
            if file_extension==".csv":
                df = pd.read_csv(filehandler, low_memory=False)#, parse_dates=[0])
            elif file_extension==".xlsx":
                df = pd.read_excel(filehandler)
            else:
                logger.error((f"Invalid file extension: {file_extension}"))
                raise Exception(f"Invalid file extension: {file_extension}")
        
        for column in df.columns.to_list()[1:]:
            df[column] = pd.to_numeric(df[column], errors='coerce') #Remove string entries
        df = df.rename(columns={df.columns[0]: 'datetime'})
        df["datetime"] = pd.to_datetime(df["datetime"])
        if df["datetime"].apply(lambda x:x.tzinfo is not None).any():
            df["datetime"] = df["datetime"].apply(lambda x:x.tz_convert("UTC"))

        # df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
        # tz_list = ["GMT", "UTC"]
        # is_member = any([s in tz for s in tz_list])
        # for tz_i in tz_list:
        #     if tz_i in tz:
        #         tz = tz.replace(tz_i, "GMT")
        #         tz = f"Etc/{tz}"
        #         break

        df = df.set_index(pd.DatetimeIndex(df['datetime']))
        df = df.drop(columns=["datetime"])
        

        if preserve_order:
            # Detect if dates are reverse
            diff_seconds = df.index.to_series().diff().dt.total_seconds()
            frac_neg = np.sum(diff_seconds<0)/diff_seconds.size
            if frac_neg>=0.95:
                df = df.iloc[::-1]
            elif frac_neg>0.05 and frac_neg<0.95:
                raise Exception("\"preserve_order\" is true, but the datetime order cannot be determined.")
            
        df = df.dropna()
        # df = df.sort_index()
        
        #Check if the first index is timezone aware
        if df.index[0].tzinfo is None:
            df = df.tz_localize(gettz(tz), ambiguous='infer', nonexistent="NaT")
        else:
            df = df.tz_convert(gettz(tz))

        
        # It has been observed that duplicate dates can occur either due to measuring/logging malfunctions
        # or due to change of daylight saving time where an hour occurs twice in fall.
        df = df.groupby(level=0).mean()

        if clip:
            df = df[start_time:end_time]

        if resample:
            df = df.resample(f"{stepSize}S", origin=start_time).ffill()
        
        

        if cache:
            df.to_pickle(cached_filename)

    return df
