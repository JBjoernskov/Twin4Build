from twin4build.saref.property_.temperature.temperature import Temperature
from twin4build.saref.property_.Co2.Co2 import Co2
from twin4build.saref.property_.opening_position.opening_position import OpeningPosition #This is in use
from twin4build.saref.property_.energy.energy import Energy #This is in use
from twin4build.saref.device.sensor.sensor import Sensor
from twin4build.saref.device.meter.meter import Meter
def get_measurement_device_error(measurement_device):
    assert isinstance(measurement_device, Sensor) or isinstance(measurement_device, Meter), f"The input argument \"measurement_device\" must be either a Sensor or Meter instance."
    _property = measurement_device.measuresProperty
    if isinstance(_property, Temperature):
        get_df_from_measurement_device()

    elif isinstance(_property, Co2):