import os
import unittest
from twin4build.utils.data_loaders.load_from_file import load_from_file
from twin4build.utils.uppath import uppath
import datetime
from dateutil.tz import tzutc

@unittest.skipIf(False, 'Currently not used')
def test_load_from_file():
    stepSize = 600
    format = "%m/%d/%Y %I:%M:%S %p"
    startPeriod = datetime.datetime(year=2021, month=1, day=1, hour=0, minute=0, second=0, tzinfo=tzutc())
    endPeriod = datetime.datetime(year=2023, month=3, day=31, hour=0, minute=0, second=0, tzinfo=tzutc())
    filename = os.path.join(os.path.abspath(uppath(os.path.abspath(__file__), 1)), "test_weather_data.csv")
    df_DMI = load_from_file(filename=filename,stepSize=stepSize, start_time=startPeriod, end_time=endPeriod, format=format, dt_limit=999999)
