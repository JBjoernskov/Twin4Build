import datetime
from twin4build.saref.date_time.date_time import DateTime
from twin4build.saref.device.meter.meter import Meter
from twin4build.saref.device.sensor.sensor import Sensor
from twin4build.saref.function.metering_function.metering_function import MeteringFunction
from twin4build.saref.function.sensing_function.sensing_function import SensingFunction
from twin4build.saref.measurement.measurement import Measurement
from twin4build.utils.configReader import configReader
import requests
import dateutil.parser


class quantumLeapReader:

    def __init__(self):
        self.config = configReader().read_config()

    def get_fiware_access_token(self):
        data = dict(
            client_id=self.config.tokenClientId,
            client_secret=self.config.tokenSecret,
            grant_type='client_credentials',
            scope=self.config.scope
        )

        resp = requests.post(url=self.config.tokenUrl, data=data)
        data = resp.json()
        return data['access_token']

    def load_object_from_quantum_leap(self, accessToken: str, id: str, params: dict):

        headers = {'Authorization': 'Bearer ' + accessToken,
                   'Link': self.config.fiwareContextLink + ';rel="http://www.w3.org/ns/json-ld#context"; type="application/ld+json"',
                   'Content-Type': 'application/json',
                   'NGSILD-Tenant': self.config.tenant,
                   'Accept': '*/*'}
        resp = requests.get(url=self.config.quantumLeapBaseUrl +
                            '/v2/entities/'+id, params=params, headers=headers)
        data = resp.json()
        return data

    def get_attribute(self, arr: list, name: str):
        for attr in arr:
            if attr["attrName"] == name:
                return attr
        return None

    def get_sensor(self, sensor_id: str, dateFrom: datetime.datetime, dateTo: datetime.datetime):
        """
        Reads sensor from Quantum Leap
        """

        access_token = self.get_fiware_access_token()
        df_Sensor = self.load_object_from_quantum_leap(
            access_token, 'urn:ngsi-ld:Sensor:'+sensor_id, dict(fromDate=dateFrom.isoformat(), toDate=dateTo.isoformat()))

        sensor = Sensor(
            hasFunction=SensingFunction(hasSensingRange=list())
        )

        for idx, measuredAt in enumerate(df_Sensor["index"]):

            parsedDateTime = dateutil.parser.isoparse(measuredAt)
            sarefDateTime = DateTime(parsedDateTime.year, parsedDateTime.month, parsedDateTime.day,
                                     parsedDateTime.hour, parsedDateTime.minute, parsedDateTime.second)
            sensor.hasFunction.hasSensingRange.append(Measurement(
                hasTimeStamp=sarefDateTime,
                hasValue=self.get_attribute(df_Sensor["attributes"], "https://saref.etsi.org/core/hasSensingRange")["values"][idx]))

            sensor.hasSensorType = self.get_attribute(
                df_Sensor["attributes"], "https://saref.etsi.org/core/hasSensorType")["values"][idx]
            sensor.id = self.get_attribute(df_Sensor["attributes"], "name")[
                "values"][idx]

        return sensor

    def get_meter(self, meter_id: str, dateFrom: datetime.datetime, dateTo: datetime.datetime):
        """
        Reads meter from Quantum Leap
        """

        access_token = self.get_fiware_access_token()
        df_meter = self.load_object_from_quantum_leap(
            access_token, 'urn:ngsi-ld:Meter:'+meter_id, dict(fromDate=dateFrom.isoformat(), toDate=dateTo.isoformat()))

        meter = Meter(
            hasFunction=MeteringFunction(hasMeterReading=list())
        )

        for idx, measuredAt in enumerate(df_meter["index"]):

            parsedDateTime = dateutil.parser.isoparse(measuredAt)
            sarefDateTime = DateTime(parsedDateTime.year, parsedDateTime.month, parsedDateTime.day,
                                     parsedDateTime.hour, parsedDateTime.minute, parsedDateTime.second)
            meter.hasFunction.hasMeterReading.append(Measurement(
                hasTimeStamp=sarefDateTime,
                hasValue=self.get_attribute(df_meter["attributes"], "https://saref.etsi.org/core/hasMeterReading")["values"][idx]))

            meter.hasMeterReadingType = self.get_attribute(
                df_meter["attributes"], "https://saref.etsi.org/core/hasMeterReadingType")["values"][idx]
            meter.id = self.get_attribute(df_meter["attributes"], "name")[
                "values"][idx]

        return meter
