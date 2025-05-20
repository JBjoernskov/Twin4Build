r"""
Data loading and processing functions for spreadsheet data.

Mathematical Formulation:

1. Time Series Resampling:
   a) Constant Resampling:
      For each time step :math:`t`:

      .. math::

         y(t) = y(t_{last})

      where :math:`t_{last}` is the last available data point before :math:`t`

   b) Linear Resampling:
      For each time step :math:`t`:

      .. math::

         y(t) = y(t_1) + \frac{t - t_1}{t_2 - t_1} \cdot (y(t_2) - y(t_1))

      where:
      - :math:`t_1` is the last available data point before :math:`t`
      - :math:`t_2` is the first available data point after :math:`t`

2. Time Zone Conversion:
   For a time :math:`t` in timezone :math:`TZ_1`:

   .. math::

      t_{TZ_2} = t_{TZ_1} + \Delta TZ

   where:
   - :math:`t_{TZ_2}` is the time in target timezone
   - :math:`\Delta TZ` is the time difference between timezones

3. Data Clipping:
   For a time series :math:`y(t)`:

   .. math::

      y_{clipped}(t) = \begin{cases}
      y(t) & \text{if } t_{start} \leq t < t_{end} \\
      \text{undefined} & \text{otherwise}
      \end{cases}
"""

import pandas as pd
import os
import pandas as pd
import numpy as np
import os
from dateutil.tz import gettz
from twin4build.utils.mkdir_in_root import mkdir_in_root
from dateutil.parser import parse
def parseDateStr(s):
    if s != '':
        try:
            return np.datetime64(parse(s))
        except ValueError:
            return np.datetime64('NaT')
    else: return np.datetime64('NaT')     

def sample_from_df(df,
                   datecolumn=0,
                   stepSize=None,
                     start_time=None,
                     end_time=None,
                     resample=True,
                     resample_method="linear",
                     clip=True,
                     tz="Europe/Copenhagen",
                     preserve_order=True):
    
    df = df.rename(columns={df.columns.to_list()[datecolumn]: 'datetime'})
    for column in df.columns.to_list():
        if column!="datetime":
            df[column] = pd.to_numeric(df[column], errors='coerce') #Remove string entries

    df["datetime"] = pd.to_datetime(df["datetime"])#), format=format)
    if df["datetime"].apply(lambda x:x.tzinfo is not None).any():
        has_tz = True
        df["datetime"] = df["datetime"].apply(lambda x:x.tz_convert("UTC"))
    else:
        has_tz = False

    df = df.set_index(pd.DatetimeIndex(df['datetime']))
    df = df.drop(columns=["datetime"])

    if preserve_order and has_tz==False:
        # Detect if dates are reverse
        diff_seconds = df.index.to_series().diff().dt.total_seconds()
        frac_neg = np.sum(diff_seconds<0)/diff_seconds.size
        if frac_neg>=0.95:
            df = df.iloc[::-1]
        elif frac_neg>0.05 and frac_neg<0.95:
            raise Exception("\"preserve_order\" is true, but the datetime order cannot be determined.")
    else:
        df = df.sort_index()
        
    df = df.dropna(how="all")
    
    
    #Check if the first index is timezone aware
    if df.index[0].tzinfo is None:
        df = df.tz_localize(gettz(tz), ambiguous='infer', nonexistent="NaT")
    else:
        df = df.tz_convert(gettz(tz))

    # Duplicate dates can occur either due to measuring/logging malfunctions
    # or due to change of daylight saving time where an hour occurs twice in fall.
    df = df.groupby(level=0).mean()

    if start_time.tzinfo is None:
            start_time = start_time.astimezone(tz=gettz(tz))
    if end_time.tzinfo is None:
        end_time = end_time.astimezone(tz=gettz(tz))

    if resample:
        allowable_resample_methods = ["constant", "linear"]
        assert resample_method in allowable_resample_methods, f"resample_method \"{resample_method}\" is not valid. The options are: {', '.join(allowable_resample_methods)}"
        if resample_method=="constant":
            df = df.resample(f"{stepSize}s", origin=start_time).ffill().bfill()
        elif resample_method=="linear":
            oidx = df.index
            nidx = pd.date_range(start_time, end_time, freq=f"{stepSize}s")
            df = df.reindex(oidx.union(nidx)).interpolate('index').reindex(nidx)

    if clip:
        df = df[(df.index >= start_time) & (df.index < end_time)] # Exclude end time for similar behavior as normal python slicing
    return df

def load_spreadsheet(filename,
                     datecolumn=0,
                     valuecolumn=None,
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
        startTime_str = start_time.strftime('%d-%m-%Y %H-%M-%S')
        endTime_str = end_time.strftime('%d-%m-%Y %H-%M-%S')
        cached_filename = f"name({os.path.basename(name)})_stepSize({str(stepSize)})_startTime({startTime_str})_endTime({endTime_str})_cached.pickle"
        cached_filename, isfile = mkdir_in_root(folder_list=["generated_files", "cached_data"], filename=cached_filename, root=cache_root)
    if cache and os.path.isfile(cached_filename):
        df = pd.read_pickle(cached_filename)
    else:
        with open(filename, 'rb') as filehandler:
            if file_extension==".csv":
                df = pd.read_csv(filehandler, low_memory=False)#, parse_dates=[0])
            elif file_extension==".xlsx":
                df = pd.read_excel(filehandler)
            else:
                raise Exception(f"Invalid file extension: {file_extension}")

        if valuecolumn is not None:
            valuename = df.columns[valuecolumn]
        df = sample_from_df(df,
                            datecolumn,
                            stepSize=stepSize,
                            start_time=start_time,
                            end_time=end_time,
                            resample=resample,
                            clip=clip,
                            tz=tz,
                            preserve_order=preserve_order)
        
        if valuecolumn is not None:
            df = df[valuename]

        if cache:
            df.to_pickle(cached_filename)

    return df
