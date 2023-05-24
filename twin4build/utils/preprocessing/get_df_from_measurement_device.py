import pandas as pd
from twin4build.saref.device.sensor.sensor import Sensor
from twin4build.saref.device.meter.meter import Meter
def get_df_from_measurement_device(measurement_device):
    assert isinstance(measurement_device, Sensor) or isinstance(measurement_device, Meter), f"The input argument \"measurement_device\" must be either a Sensor or Meter instance."
    df = pd.DataFrame()
    if isinstance(measurement_device, Sensor):
        measurements_list = measurement_device.hasFunction.hasSensingRange
        time_list = [measurement.hasTimeStamp for measurement in measurements_list]
        value_list = [measurement.hasValue for measurement in measurements_list]
        unit_list = [measurement.isMeasuredIn.__name_ for measurement in measurements_list]

        df.insert(0, "time", time_list)
        df.insert(1, "value", value_list)
        df.insert(2, "unit", unit_list)
        return df
    
