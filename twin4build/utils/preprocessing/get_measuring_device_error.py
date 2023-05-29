from twin4build.saref.property_.temperature.temperature import Temperature
from twin4build.saref.property_.Co2.Co2 import Co2
from twin4build.saref.property_.flow.flow import Flow
from twin4build.saref.property_.opening_position.opening_position import OpeningPosition #This is in use
from twin4build.saref.property_.energy.energy import Energy #This is in use
from twin4build.saref.device.sensor.sensor import Sensor
from twin4build.saref.device.meter.meter import Meter
from twin4build.utils.preprocessing.get_df_from_measuring_device import get_df_from_measuring_device

def _get_error_from_absolute_range(df, range):
    df["lower_value"] = df["value"]-range
    df["upper_value"] = df["value"]+range
    return df

def _get_error_from_percentage_range(df, range):
    df["lower_value"] = df["value"]*(1-range/100)
    df["upper_value"] = df["value"]*(1+range/100)
    return df

def get_measuring_device_error(measuring_device):
    assert isinstance(measuring_device, Sensor) or isinstance(measuring_device, Meter), f"The input argument \"measuring_device\" must be either a Sensor or Meter instance."
    _property = measuring_device.measuresProperty
    df = get_df_from_measuring_device(measuring_device)

    if isinstance(_property, Temperature):
        # Assumming "Thermistor" from "Sensor Accuracy and Calibration Theory and Practical Application", K. Stum
        range = 0.36*5/9 # Convert range from fahrenheit to Celcius or Kelvin
        df = _get_error_from_absolute_range(df, range)

    elif isinstance(_property, Co2):
        raise Exception("Not implemented yet")

    elif isinstance(_property, Flow):
        # Assumming "Pitot Tube" from "Error Analysis of Measurement and Control Techniques 
        # of Outside Air Intake Rates in VAV Systems", C. C. Schroeder, M. Krarti, M. J. Brandemuehl
        range = 5 #Percent
        df = _get_error_from_percentage_range(df, range)

    else:
        raise Exception("Not implemented yet")
    return df