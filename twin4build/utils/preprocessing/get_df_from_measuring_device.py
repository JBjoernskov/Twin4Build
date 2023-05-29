import pandas as pd
from twin4build.saref.device.sensor.sensor import Sensor
from twin4build.saref.device.meter.meter import Meter
def get_df_from_measuring_device(measuring_device):
    """
    Takes a Sensor or Meter instance as input and outputs a dataframe 
    """
    assert isinstance(measuring_device, Sensor) or isinstance(measuring_device, Meter), f"The input argument \"measuring_device\" must be either a Sensor or Meter instance."
    df = pd.DataFrame()
    if isinstance(measuring_device, Sensor):
        measurements_list = measuring_device.hasFunction.hasSensingRange
        time_list = [measurement.hasTimeStamp for measurement in measurements_list]
        value_list = [measurement.hasValue for measurement in measurements_list]
        unit_list = [measurement.isMeasuredIn.__name__ for measurement in measurements_list]

        
    elif isinstance(measuring_device, Meter):
        measurements_list = measuring_device.hasFunction.hasMeterReading
        time_list = [measurement.hasTimeStamp for measurement in measurements_list]
        value_list = [measurement.hasValue for measurement in measurements_list]
        unit_list = [measurement.isMeasuredIn.__name__ for measurement in measurements_list]

    df.insert(0, "time", time_list)
    df.insert(1, "value", value_list)
    df.insert(2, "unit", unit_list)
    df.set_index("time", inplace=True)
    df.sort_index(inplace=True)
    return df
    
