import pandas as pd
from twin4build.saref.device.sensor.sensor import Sensor
from twin4build.saref.device.meter.meter import Meter
from twin4build.saref.measurement.measurement import Measurement
from twin4build.saref.date_time.date_time import DateTime
from twin4build.saref.property_.temperature.temperature import Temperature
from twin4build.saref.property_.flow.flow import Flow
from twin4build.saref.function.sensing_function.sensing_function import SensingFunction
from twin4build.saref.function.metering_function.metering_function import MeteringFunction
from twin4build.saref.unit_of_measure.temperature_unit.temperature_unit import TemperatureUnit
from twin4build.saref.unit_of_measure.flow_unit.flow_unit import FlowUnit
import sys
def get_measuring_device_from_df(df, measuring_device_class, property_class, unit_of_measure_class, id):
    """
    Takes as input DataFrame instance with timestamps as index and "value" as column name.
    Outputs a Sensor or Meter instance containing the data.
    """
    measuring_device_class = getattr(sys.modules[__name__], measuring_device_class)
    property_class = getattr(sys.modules[__name__], property_class)
    unit_of_measure_class = getattr(sys.modules[__name__], unit_of_measure_class)

    legal_property_class_list = [Temperature, Flow]
    assert measuring_device_class is Sensor or measuring_device_class is Meter, f"The input argument \"measuring_device_class\" must be either Sensor or Meter class."
    assert property_class in legal_property_class_list, f"The input argument \"unit_of_measure\" must one of the following classes: {','.join([el.__name__ for el in legal_property_class_list])}."
    if measuring_device_class is Sensor:
        measuring_device = Sensor(hasFunction=SensingFunction(hasSensingRange=[]),
                                  measuresProperty=property_class(),
                                  id=id)
        
        for index, row in df.iterrows():
            timestamp = DateTime(year=index.year,
                                 month=index.month,
                                 day=index.day,
                                 hour=index.hour,
                                 minute=index.minute,
                                 second=index.second)
            
            measurement = Measurement(hasTimeStamp=timestamp, 
                                      hasValue=row["value"], 
                                      isMeasuredIn=unit_of_measure_class)
            measuring_device.hasFunction.hasSensingRange.append(measurement)

    elif measuring_device_class is Meter:
        measuring_device = Meter(hasFunction=MeteringFunction(hasMeterReading=[]),
                                measuresProperty=property_class(),
                                id=id)
        for index, row in df.iterrows():
            timestamp = DateTime(year=index.year,
                                 month=index.month,
                                 day=index.day,
                                 hour=index.hour,
                                 minute=index.minute,
                                 second=index.second)
            measurement = Measurement(hasTimeStamp=timestamp, 
                                      hasValue=row["value"],
                                      isMeasuredIn=unit_of_measure_class)
            measuring_device.hasFunction.hasMeterReading.append(measurement)

    return measuring_device
    
